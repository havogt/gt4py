# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import copy

from gt4py._core import filecache
import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next.iterator import builtins, ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next import fingerprinting
from gt4py.next.otf import arguments, artifacts, stages
from gt4py.next.program_processors.codegens.gtfn import gtfn_module
from gt4py.next.program_processors.runners import gtfn
from gt4py.next.type_system import type_specifications as ts, type_translation
from gt4py.next import custom_layout_allocators as next_allocators

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import cartesian_case, cartesian_case_no_backend
from next_tests.integration_tests.cases_utils import (
    KDim,
    exec_alloc_descriptor,
)


@pytest.fixture
def program_example():
    IDim = gtx.Dimension("I")
    params = [gtx.as_field([IDim], np.empty((1,), dtype=np.float32)), np.float32(3.14)]
    param_types = [type_translation.from_value(param) for param in params]

    domain = itir.FunCall(
        fun=itir.SymRef(id="cartesian_domain"),
        args=[
            itir.FunCall(
                fun=itir.SymRef(id="named_range"),
                args=[
                    itir.AxisLiteral(value="I"),
                    im.literal("0", builtins.INTEGER_INDEX_BUILTIN),
                    im.literal("10", builtins.INTEGER_INDEX_BUILTIN),
                ],
            )
        ],
    )
    program = itir.Program(
        id="example",
        params=[im.sym(name, type_) for name, type_ in zip(("buf", "sc"), param_types)],
        function_definitions=[
            itir.FunctionDefinition(
                id="stencil",
                params=[itir.Sym(id="buf"), itir.Sym(id="sc")],
                expr=im.literal("1", "float32"),
            )
        ],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(itir.SymRef(id="stencil"), domain)(
                    itir.SymRef(id="buf"), itir.SymRef(id="sc")
                ),
                domain=domain,
                target=itir.SymRef(id="buf"),
            )
        ],
    )
    return program, params


def test_codegen(program_example):
    fencil, parameters = program_example
    module = gtfn_module.translate_program_cpu(
        stages.CompilableProgramDef(
            data=fencil,
            args=arguments.CompileTimeArgs.from_concrete(*parameters, **{"offset_provider": {}}),
        )
    )
    assert module.entry_point.name == fencil.id
    assert any(d.name == "gridtools_cpu" for d in module.library_deps)
    assert isinstance(module.code_spec, artifacts.CPPCodeSpec)


def test_hash_and_diskcache(program_example, tmp_path):
    fencil, parameters = program_example
    compilable_program = stages.CompilableProgramDef(
        data=fencil,
        args=arguments.CompileTimeArgs.from_concrete(*parameters, **{"offset_provider": {}}),
    )
    hash = fingerprinting.strict_fingerprinter(compilable_program)

    cache = filecache.FileCache(tmp_path)
    cache[hash] = compilable_program

    # check content of cash file
    reopened_cache = filecache.FileCache(tmp_path)
    assert hash in reopened_cache
    compilable_program_from_cache = reopened_cache[hash]
    assert compilable_program == compilable_program_from_cache
    del reopened_cache[hash]  # delete data

    # hash creation is deterministic
    assert hash == fingerprinting.strict_fingerprinter(compilable_program)
    assert hash == fingerprinting.strict_fingerprinter(compilable_program_from_cache)

    # hash is different if program changes
    altered_program_id = copy.deepcopy(compilable_program)
    altered_program_id.data.id = "example2"
    assert fingerprinting.strict_fingerprinter(
        compilable_program
    ) != fingerprinting.strict_fingerprinter(altered_program_id)

    altered_program_offset_provider = copy.deepcopy(compilable_program)
    object.__setattr__(altered_program_offset_provider.args, "offset_provider", {"Koff": KDim})
    assert fingerprinting.strict_fingerprinter(
        compilable_program
    ) != fingerprinting.strict_fingerprinter(altered_program_offset_provider)

    altered_program_column_axis = copy.deepcopy(compilable_program)
    object.__setattr__(altered_program_column_axis.args, "column_axis", KDim)
    assert fingerprinting.strict_fingerprinter(
        compilable_program
    ) != fingerprinting.strict_fingerprinter(altered_program_column_axis)


def test_gtfn_file_cache(program_example):
    fencil, parameters = program_example
    compilable_program = stages.CompilableProgramDef(
        data=fencil,
        args=arguments.CompileTimeArgs.from_concrete(*parameters, **{"offset_provider": {}}),
    )
    cached_gtfn_translation_step = gtfn.GTFNCompileWorkflowFactory(
        cached_translation=True
    ).translation

    bare_gtfn_translation_step = gtfn.GTFNCompileWorkflowFactory(
        cached_translation=False
    ).translation

    cache_key = cached_gtfn_translation_step.cache_key(compilable_program)

    # ensure the actual cached step in the backend generates the cache item for the test
    if cache_key in (translation_cache := cached_gtfn_translation_step.cache):
        del translation_cache[cache_key]
    cached_gtfn_translation_step(compilable_program)
    assert bare_gtfn_translation_step(compilable_program) == cached_gtfn_translation_step(
        compilable_program
    )

    assert cache_key in cached_gtfn_translation_step.cache
    assert (
        bare_gtfn_translation_step(compilable_program)
        == cached_gtfn_translation_step.cache[cache_key]
    )


def _copy_program_source(sizes: dict[gtx.Dimension, int | str]) -> str:
    """CUDA source, without whitespace, of `out ← as_fieldop(deref)(inp)` on a cartesian domain of `sizes`."""
    dims = list(sizes)
    params = [gtx.as_field(dims, np.empty([2] * len(dims))) for _ in range(2)]
    domain = im.domain(
        gtx.GridType.CARTESIAN,
        {d: (0, im.ref(n) if isinstance(n, str) else n) for d, n in sizes.items()},
    )
    symbolic = [
        im.sym(n, ts.ScalarType(kind=ts.ScalarKind.INT32))
        for n in sizes.values()
        if isinstance(n, str)
    ]
    program = itir.Program(
        id="copy",
        params=[
            *(
                im.sym(name, type_translation.from_value(p))
                for name, p in zip(("out", "inp"), params)
            ),
            *symbolic,
        ],
        function_definitions=[],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(im.lambda_("a")(im.deref("a")), domain)("inp"),
                domain=domain,
                target=im.ref("out"),
            )
        ],
    )
    module = gtfn_module.GTFNTranslationStep(device_type=gtx.DeviceType.CUDA)(
        stages.CompilableProgramDef(
            data=program,
            args=arguments.CompileTimeArgs.from_concrete(
                *params, *(np.int32(2) for _ in symbolic), offset_provider={}
            ),
        )
    )
    return "".join(module.source_code.split())


IDim, JDim, XDim = gtx.Dimension("I"), gtx.Dimension("J"), gtx.Dimension("X")
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)


def test_vertical_dim_on_threads_beyond_three_dims():
    source = _copy_program_source({IDim: 100, JDim: 116, XDim: 1, K: 80})

    # 100 * 116 * 80 points, about 185k threads per launch: each thread loops 5 levels
    assert "keys<I_t,J_t,K_t,X_t>" in source
    assert (
        "usingloop_block_sizes_t=gridtools::meta::list<gridtools::meta::list<K_t,gridtools::integral_constant<int,5>>>;"
        in source
    )
    assert "gpu<generated::block_sizes_t,generated::loop_block_sizes_t>" in source


def test_vertical_loop_block_needs_static_sizes():
    source = _copy_program_source({IDim: "n", JDim: 116, XDim: 1, K: 80})

    assert "keys<I_t,J_t,K_t,X_t>" in source
    assert "usingloop_block_sizes_t=gridtools::meta::list<>;" in source


def test_three_dims_keep_their_mapping():
    source = _copy_program_source({IDim: 1000, JDim: 1000, K: 80})

    assert "keys<I_t,J_t,K_t>" in source
    assert "usingloop_block_sizes_t=gridtools::meta::list<>;" in source
