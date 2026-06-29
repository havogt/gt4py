# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import textwrap
import warnings
from collections.abc import Callable, MutableSequence, Sequence
from typing import Any

import dace
import dace.codegen.compiler as dace_compiler
import factory

from gt4py._core import definitions as core_defs, locking
from gt4py.next import common, config
from gt4py.next.otf import code_specs, definitions, stages, workflow
from gt4py.next.otf.compilation import cache as gtx_cache
from gt4py.next.program_processors.runners.dace.workflow import common as gtx_wfdcommon


_SDFG_FILENAME = "sdfg.json"
_BINDING_FILENAME = "binding.py"


def _add_tx_markers(sdfg: dace.SDFG) -> None:
    has_gpu_schedule = any(
        getattr(node, "schedule", dace.dtypes.ScheduleType.Default) in dace.dtypes.GPU_SCHEDULES
        for node, _ in sdfg.all_nodes_recursive()
    )

    if has_gpu_schedule:
        sdfg.instrument = dace.dtypes.InstrumentationType.GPU_TX_MARKERS
        for node, _ in sdfg.all_nodes_recursive():
            # Also adds markers to map scopes that are NOT scheduled on GPU
            if isinstance(node, (dace.nodes.MapEntry, dace.sdfg.SDFGState)):
                node.instrument = dace.dtypes.InstrumentationType.GPU_TX_MARKERS


class CompiledDaceProgram:
    sdfg_program: dace.CompiledSDFG

    # Sorted list of SDFG arguments as they appear in program ABI and corresponding data type;
    # scalar arguments that are not used in the SDFG will not be present.
    sdfg_argtypes: list[dace.dtypes.Data]

    # The compiled program contains a callable object to update the SDFG arguments list.
    update_sdfg_ctype_arglist: Callable[
        [
            core_defs.DeviceType,
            Sequence[dace.dtypes.Data],
            Sequence[Any],
            MutableSequence[Any],
            common.OffsetProvider,
        ],
        None,
    ]

    # Processed argument vectors that are passed to `CompiledSDFG.fast_call()`. `None`
    #  means that it has not been initialized, i.e. no call was ever performed.
    #  - csdfg_argv: Arguments used for calling the actual compiled SDFG, will be updated.
    #  - csdfg_init_argv: Arguments used for initialization; used only the first time and
    #       never updated.
    csdfg_argv: MutableSequence[Any] | None
    csdfg_init_argv: Sequence[Any] | None

    def __init__(
        self,
        program: dace.CompiledSDFG,
        bind_func_name: str,
        binding_source_code: str,
    ):
        self.sdfg_program = program

        # `dace.CompiledSDFG.arglist()` returns an ordered dictionary that maps the argument
        # name to its data type, in the same order as arguments appear in the program ABI.
        # This is also the same order of arguments in `dace.CompiledSDFG._lastargs[0]`.
        self.sdfg_argtypes = list(program.sdfg.arglist().values())

        # The binding source code is Python tailored to this specific SDFG.
        # We dynamically compile that function and add it to the compiled program.
        global_namespace: dict[str, Any] = {}
        exec(binding_source_code, global_namespace)
        self.update_sdfg_ctype_arglist = global_namespace[bind_func_name]
        # For debug purpose, we set a unique module name on the compiled function.
        self.update_sdfg_ctype_arglist.__module__ = os.path.basename(program.sdfg.build_folder)

        # Since the SDFG hasn't been called yet.
        self.csdfg_argv = None
        self.csdfg_init_argv = None

    def construct_arguments(self, **kwargs: Any) -> None:
        """
        This function will process the arguments and store the processed argument
        vectors in `self.csdfg_args`, to call them use `self.fast_call()`.
        """
        with dace.config.set_temporary("compiler", "allow_view_arguments", value=True):
            csdfg_argv, csdfg_init_argv = self.sdfg_program.construct_arguments(**kwargs)
        # Note we only care about `csdfg_argv` (normal call), since we have to update it,
        #  we ensure that it is a `list`.
        self.csdfg_argv = [*csdfg_argv]
        self.csdfg_init_argv = csdfg_init_argv

    def fast_call(self) -> None:
        """
        Perform a call to the compiled SDFG using the previously generated argument
        vectors, see `self.construct_arguments()`.
        """
        assert self.csdfg_argv is not None and self.csdfg_init_argv is not None, (
            "Argument vector was not set properly."
        )
        self.sdfg_program.fast_call(
            self.csdfg_argv, self.csdfg_init_argv, do_gpu_check=config.DEBUG
        )

    def __call__(self, **kwargs: Any) -> None:
        """Call the compiled SDFG with the given arguments.

        Note that this function will not update the argument vectors stored inside
        `self`. Furthermore, it is not recommended to use this function as it is
        very slow.
        """
        warnings.warn(
            "Called an SDFG through the standard DaCe interface is not recommended, use `fast_call()` instead.",
            stacklevel=1,
        )
        result = self.sdfg_program(**kwargs)
        assert result is None


def _render_dace_loader(
    library_relpath: pathlib.Path, bind_func_name: str, device_type: core_defs.DeviceType
) -> str:
    return textwrap.dedent(f"""\
        import json
        import dace
        import dace.codegen.compiler as dace_compiler
        from gt4py._core import definitions as core_defs
        from gt4py.next.program_processors.runners.dace.workflow import (
            compilation as gtx_wfdcompilation,
            decoration as gtx_wfddecoration,
        )


        def load(src_dir):
            sdfg = dace.SDFG.from_json(
                json.loads((src_dir / {_SDFG_FILENAME!r}).read_text())
            )
            sdfg_program = dace_compiler.get_program_handle(
                src_dir / {str(library_relpath)!r}, sdfg
            )
            binding_source = (src_dir / {_BINDING_FILENAME!r}).read_text()
            program = gtx_wfdcompilation.CompiledDaceProgram(
                sdfg_program, {bind_func_name!r}, binding_source
            )
            return gtx_wfddecoration.convert_args(
                program, device=core_defs.DeviceType.{device_type.name}
            )
        """)


@dataclasses.dataclass(frozen=True)
class DaCeCompiler(
    workflow.ChainableWorkflowMixin[
        stages.CompilableProject[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec],
        stages.CompilationArtifact,
    ],
    workflow.ReplaceEnabledWorkflowMixin[
        stages.CompilableProject[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec],
        stages.CompilationArtifact,
    ],
    definitions.CompilationStep[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec],
):
    """Run the DaCe build system and write a self-contained loader file next to the .so."""

    bind_func_name: str
    cache_lifetime: config.BuildCacheLifetime
    device_type: core_defs.DeviceType
    add_gpu_trace_markers: bool = dataclasses.field(
        default_factory=lambda: config.ADD_GPU_TRACE_MARKERS
    )
    cmake_build_type: config.CMakeBuildType = dataclasses.field(
        default_factory=lambda: config.CMAKE_BUILD_TYPE
    )

    def __call__(
        self,
        inp: stages.CompilableProject[code_specs.SDFGCodeSpec, code_specs.PythonCodeSpec],
    ) -> stages.CompilationArtifact:
        with gtx_wfdcommon.dace_context(
            device_type=self.device_type,
            cmake_build_type=self.cmake_build_type,
        ):
            sdfg_build_folder = pathlib.Path(gtx_cache.get_cache_folder(inp, self.cache_lifetime))
            sdfg_build_folder.mkdir(parents=True, exist_ok=True)

            sdfg = dace.SDFG.from_json(inp.program_source.source_code)

            # Add TX markers to the generated GPU code for trace visualization tools.
            if self.add_gpu_trace_markers and self.device_type == core_defs.CUPY_DEVICE_TYPE:
                _add_tx_markers(sdfg)

            sdfg.build_folder = str(sdfg_build_folder)
            with locking.lock(sdfg_build_folder):
                sdfg.compile(validate=False, return_program_handle=False)
                # ``build_folder_mode`` is set by ``dace_context``; resolve the library
                # path inside so ``get_binary_name`` sees the same mode dace built under.
                library_path = dace_compiler.get_binary_name(
                    object_folder=sdfg_build_folder, sdfg_name=sdfg.name
                )
                assert inp.binding_source is not None
                sdfg_text = (
                    inp.program_source.source_code
                    if isinstance(inp.program_source.source_code, str)
                    else json.dumps(inp.program_source.source_code)
                )
                (sdfg_build_folder / _SDFG_FILENAME).write_text(sdfg_text)
                (sdfg_build_folder / _BINDING_FILENAME).write_text(inp.binding_source.source_code)
                (sdfg_build_folder / stages._LOADER_MODULE_FILENAME).write_text(
                    _render_dace_loader(
                        library_relpath=library_path.relative_to(sdfg_build_folder),
                        bind_func_name=self.bind_func_name,
                        device_type=self.device_type,
                    )
                )

        return stages.CompilationArtifact(src_dir=sdfg_build_folder)


class DaCeCompilationStepFactory(factory.Factory):
    class Meta:
        model = DaCeCompiler
