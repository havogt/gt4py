# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import functools
import hashlib
import sys
import textwrap
from collections.abc import Iterable
from typing import Any, Optional

from gt4py.eve import codegen
from gt4py.eve.codegen import FormatTemplate as as_fmt, MakoTemplate as as_mako
from gt4py.next import (
    backend as next_backend,
    common,
    config,
    custom_layout_allocators as next_allocators,
)
from gt4py.next.ffront import foast_to_gtir, foast_to_past, past_to_itir
from gt4py.next.iterator import ir as itir, transforms as itir_transforms
from gt4py.next.otf import definitions, stages, workflow
from gt4py.next.otf.compilation import cache as gtx_cache
from gt4py.next.type_system import type_info, type_specifications as ts


_PROGRAM_FILENAME = "program.py"


def _create_tmp(axes: str, origin: str, shape: str, dtype: ts.TypeSpec) -> str:
    if isinstance(dtype, ts.TupleType):
        return f"({','.join(_create_tmp(axes, origin, shape, dt) for dt in dtype.types)},)"
    else:
        assert isinstance(dtype, ts.ScalarType)
        return (
            f"gtx.as_field([{axes}], np.empty({shape}, dtype=np.dtype('{dtype}')), origin={origin})"
        )


class EmbeddedDSL(codegen.TemplatedGenerator):
    Sym = as_fmt("{id}")
    SymRef = as_fmt("{id}")

    def visit_Literal(self, node: itir.Literal, **kwargs: Any) -> str:
        if (
            isinstance(node.type, ts.ScalarType)
            and type_info.is_floating_point(node.type)
            and node.value in ["inf", "-inf", "nan"]
        ):
            dtype = node.type.kind.name.lower()
            if node.value == "inf":
                return f"np.{dtype}(np.inf)"
            elif node.value == "-inf":
                return f"-np.{dtype}(np.inf)"
            elif node.value == "nan":
                return f"np.{dtype}(np.nan)"
        return node.value

    NoneLiteral = as_fmt("None")
    OffsetLiteral = as_fmt("{value}")
    AxisLiteral = as_fmt("{value}")

    def visit_CartesianOffset(self, node: itir.CartesianOffset, **kwargs: Any) -> str:
        return f"gtx.CartesianConnectivity({node.domain.value}, codomain={node.codomain.value})"

    FunCall = as_fmt("{fun}({','.join(args)})")
    Lambda = as_mako("(lambda ${','.join(params)}: ${expr})")
    FunctionDefinition = as_mako(
        """
@fundef
def ${id}(${','.join(params)}):
    return ${expr}
    """
    )
    Program = as_mako(
        """
${''.join(function_definitions)}
@fendef
def ${id}(${','.join(params)}):
    ${'\\n    '.join(declarations)}
    ${'\\n    '.join(body)}
    """
    )
    SetAt = as_mako("set_at(${expr}, ${domain}, ${target})")
    IfStmt = as_mako("""if_stmt(${cond}, 
        lambda: [${','.join(true_branch)}],
        lambda: [${','.join(false_branch)}]
    )""")

    def visit_Temporary(self, node: itir.Temporary, **kwargs: Any) -> str:
        assert (
            isinstance(node.domain, itir.FunCall)
            and isinstance(node.domain.fun, itir.SymRef)
            and node.domain.fun.id in ("cartesian_domain", "unstructured_domain")
        )
        assert all(
            isinstance(r, itir.FunCall) and r.fun == itir.SymRef(id="named_range")
            for r in node.domain.args
        )
        domain_ranges = [self.visit(r.args) for r in node.domain.args]  # type: ignore[attr-defined] # `node.domain` is `FunCall` checked in previous assert
        axes = ", ".join(label for label, _, _ in domain_ranges)
        origin = "{" + ", ".join(f"{label}: -{start}" for label, start, _ in domain_ranges) + "}"
        shape = "(" + ", ".join(f"{stop}-{start}" for _, start, stop in domain_ranges) + ")"
        assert node.dtype
        return f"{node.id} = {_create_tmp(axes, origin, shape, node.dtype)}"


def _generate_source(
    ir: itir.Program,
    debug: bool,
    use_embedded: bool,
    offset_provider: common.OffsetProvider,
    transforms: itir_transforms.GTIRTransform,
) -> tuple[str, str]:
    """Generate the Python source for an ITIR program. Returns ``(source_code, entry_point_name)``."""
    ir = transforms(ir, offset_provider=offset_provider)
    program = EmbeddedDSL.apply(ir)
    if debug:
        program = codegen.format_python_source(program)

    offset_literals: Iterable[str] = (
        ir.pre_walk_values()
        .if_isinstance(itir.OffsetLiteral)
        .getattr("value")
        .if_isinstance(str)
        .to_set()
    )
    axis_literals_set: Iterable[itir.AxisLiteral] = (
        ir.pre_walk_values().if_isinstance(itir.AxisLiteral).to_set()
    )

    if use_embedded:
        builtins_import = "from gt4py.next.iterator.embedded import *"
    else:
        builtins_import = "from gt4py.next.iterator.builtins import *"

    header = textwrap.dedent(
        f"""
        import numpy as np
        import gt4py.next as gtx
        {builtins_import}
        from gt4py.next.iterator.runtime import *
        """
    )

    offset_literals_src = "\n".join(f'{o} = offset("{o}")' for o in offset_literals)
    axis_literals_src = "\n".join(
        f'{o.value} = gtx.Dimension("{o.value}", kind=gtx.DimensionKind("{o.kind}"))'
        for o in axis_literals_set
    )
    source_code = f"{header}{offset_literals_src}\n{axis_literals_src}\n{program}"

    assert isinstance(ir, itir.Program)
    return source_code, str(ir.id)


def _find_module_qualname(obj: Any) -> tuple[str, str] | None:
    """Find ``(module, attr)`` such that ``getattr(sys.modules[module], attr) is obj``."""
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not hasattr(mod, "__dict__"):
            continue
        for attr_name, value in vars(mod).items():
            if value is obj:
                return mod_name, attr_name
    return None


def _dispatch_backend_import_stmt(backend: next_backend.Backend | None) -> str:
    # The loader file has to reconstruct ``dispatch_backend`` at load time without
    # the in-memory object being available. We do that by writing an ``import``
    # statement into the generated source, which only works if the Backend
    # instance is reachable via ``sys.modules`` (i.e. is a module-global). A
    # Backend created ad-hoc via ``GTFNBackendFactory(...)`` or
    # ``make_dace_backend(...)`` and held only in a local/fixture scope is not
    # findable; this is an accepted limitation because Roundtrip is a
    # development/debugging backend, not a production target. See the
    # ``Roundtrip.dispatch_backend`` field for the user-facing contract.
    if backend is None:
        return "_dispatch_backend = None"
    qualname = _find_module_qualname(backend)
    if qualname is None:
        raise NotImplementedError(
            "Roundtrip's ``dispatch_backend`` must be reachable as a module-level "
            "attribute (so it can be re-imported by the generated loader). "
            "Factory-constructed backends that are only bound to a local variable "
            "or fixture cannot be serialized into the compilation artifact. Either "
            "assign the backend to a module global, or use a Roundtrip variant "
            "with ``dispatch_backend=None``/``embedded.run_roundtrip_executor``."
        )
    module, attr = qualname
    return f"from {module} import {attr} as _dispatch_backend"


def _column_axis_repr(column_axis: common.Dimension | None) -> str:
    if column_axis is None:
        return "None"
    return (
        f"gtx_common.Dimension({column_axis.value!r}, "
        f"kind=gtx_common.DimensionKind({column_axis.kind.value!r}))"
    )


def _render_loader(
    entry_point_name: str, column_axis_repr: str, dispatch_backend_import: str
) -> str:
    return textwrap.dedent(f"""\
        import importlib.util
        from gt4py.next import common as gtx_common

        {dispatch_backend_import}


        def load(src_dir):
            spec = importlib.util.spec_from_file_location(
                f"gt4py.__compiled_programs__.{{src_dir.name}}._roundtrip",
                src_dir / {_PROGRAM_FILENAME!r},
            )
            assert spec is not None and spec.loader is not None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fencil = getattr(mod, {entry_point_name!r})
            captured_column_axis = {column_axis_repr}

            def decorated_fencil(*args, offset_provider, out=None, column_axis=None, **kwargs):
                if out is not None:
                    args = (*args, out)
                fencil(
                    *args,
                    offset_provider=offset_provider,
                    backend=_dispatch_backend,
                    column_axis=captured_column_axis,
                    **kwargs,
                )

            return decorated_fencil
        """)


@dataclasses.dataclass(frozen=True)
class Roundtrip(workflow.Workflow[definitions.CompilableProgramDef, stages.CompilationArtifact]):
    """Generate-and-exec Python ``Workflow``.

    .. note::
       ``dispatch_backend`` must be reachable as a module-level attribute: the
       generated loader file re-imports it by ``module.attribute`` qualname. A
       ``Backend`` constructed ad-hoc (e.g. via ``GTFNBackendFactory(...)`` or
       ``make_dace_backend(...)``) and held only in a local variable or pytest
       fixture is *not* supported and will raise ``NotImplementedError`` at
       compile time. Either assign it to a module global, or use one of the
       module-level Roundtrip backends in this file. Roundtrip is a development
       backend, so we accept this constraint rather than carry a more elaborate
       Backend-serialization mechanism.
    """

    debug: Optional[bool] = None
    use_embedded: bool = True
    dispatch_backend: Optional[next_backend.Backend] = None
    transforms: itir_transforms.GTIRTransform = itir_transforms.apply_common_transforms  # type: ignore[assignment] # TODO(havogt): cleanup interface of `apply_common_transforms`

    def __call__(self, inp: definitions.CompilableProgramDef) -> stages.CompilationArtifact:
        debug = config.DEBUG if self.debug is None else self.debug

        source_code, entry_point_name = _generate_source(
            inp.data,
            offset_provider=inp.args.offset_provider,
            debug=debug,
            use_embedded=self.use_embedded,
            transforms=self.transforms,
        )

        column_axis_repr = _column_axis_repr(inp.args.column_axis)
        dispatch_backend_import = _dispatch_backend_import_stmt(self.dispatch_backend)
        digest = hashlib.sha256(
            "\0".join(
                (source_code, entry_point_name, column_axis_repr, dispatch_backend_import)
            ).encode()
        ).hexdigest()
        src_dir = (
            gtx_cache.get_cache_base_path(config.BUILD_CACHE_LIFETIME)
            / f"roundtrip_{entry_point_name}_{digest}"
        )
        src_dir.mkdir(parents=True, exist_ok=True)
        (src_dir / _PROGRAM_FILENAME).write_text(source_code)
        (src_dir / stages._LOADER_MODULE_FILENAME).write_text(
            _render_loader(
                entry_point_name=entry_point_name,
                column_axis_repr=column_axis_repr,
                dispatch_backend_import=dispatch_backend_import,
            )
        )

        return stages.CompilationArtifact(src_dir=src_dir)


# TODO(tehrengruber): introduce factory
default = next_backend.Backend(
    name="roundtrip",
    executor=Roundtrip(
        transforms=functools.partial(
            itir_transforms.apply_common_transforms,
            extract_temporaries=False,
        )
    ),
    allocator=next_allocators.StandardCPUFieldBufferAllocator(),
    transforms=next_backend.DEFAULT_TRANSFORMS,
)
with_temporaries = next_backend.Backend(
    name="roundtrip_with_temporaries",
    executor=Roundtrip(
        transforms=functools.partial(
            itir_transforms.apply_common_transforms,
            extract_temporaries=True,
        )
    ),
    allocator=next_allocators.StandardCPUFieldBufferAllocator(),
    transforms=next_backend.DEFAULT_TRANSFORMS,
)
no_transforms = next_backend.Backend(
    name="roundtrip",
    executor=Roundtrip(transforms=lambda o, *, offset_provider: o),
    allocator=next_allocators.StandardCPUFieldBufferAllocator(),
    transforms=next_backend.DEFAULT_TRANSFORMS,
)


gtir = next_backend.Backend(
    name="roundtrip_gtir",
    executor=Roundtrip(transforms=itir_transforms.apply_fieldview_transforms),  # type: ignore[arg-type] # don't understand why mypy complains
    allocator=next_allocators.StandardCPUFieldBufferAllocator(),
    transforms=next_backend.Transforms(
        past_to_itir=past_to_itir.past_to_gtir_factory(),
        foast_to_itir=foast_to_gtir.adapted_foast_to_gtir_factory(cached=True),
        field_view_op_to_prog=foast_to_past.operator_to_program_factory(
            foast_to_itir_step=foast_to_gtir.adapted_foast_to_gtir_factory()
        ),
    ),
)
foast_to_gtir_step = foast_to_gtir.adapted_foast_to_gtir_factory(cached=True)
