# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import pathlib
import textwrap
from typing import Protocol, TypeVar

from gt4py._core import definitions as core_defs, locking
from gt4py.next import config
from gt4py.next.otf import code_specs, definitions, stages, workflow
from gt4py.next.otf.compilation import build_data, cache


def is_compiled(data: build_data.BuildData) -> bool:
    return data.status >= build_data.BuildStatus.COMPILED


def module_exists(data: build_data.BuildData, src_dir: pathlib.Path) -> bool:
    return (src_dir / data.module).exists()


CodeSpecT = TypeVar("CodeSpecT", bound=code_specs.SourceCodeSpec)
TargetCodeSpecT = TypeVar("TargetCodeSpecT", bound=code_specs.SourceCodeSpec)
CPPLikeCodeSpecT = TypeVar("CPPLikeCodeSpecT", bound=code_specs.CPPLikeCodeSpec)


class BuildSystemProjectGenerator(Protocol[CodeSpecT, TargetCodeSpecT]):
    def __call__(
        self,
        source: stages.CompilableProject[CodeSpecT, TargetCodeSpecT],
        cache_lifetime: config.BuildCacheLifetime,
    ) -> stages.BuildSystemProject[CodeSpecT, TargetCodeSpecT]: ...


@dataclasses.dataclass(frozen=True)
class CPPCompiler(
    workflow.ChainableWorkflowMixin[
        stages.CompilableProject[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
        stages.CompilationArtifact,
    ],
    workflow.ReplaceEnabledWorkflowMixin[
        stages.CompilableProject[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
        stages.CompilationArtifact,
    ],
    definitions.CompilationStep[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
):
    """Drive a CPP-style build system and write a ``_gt4py_load.py`` next to the .so.

    Backends override ``_render_loader`` to emit their own loader-file body.
    """

    cache_lifetime: config.BuildCacheLifetime
    builder_factory: BuildSystemProjectGenerator[CPPLikeCodeSpecT, code_specs.PythonCodeSpec]
    device_type: core_defs.DeviceType
    force_recompile: bool = False

    def __call__(
        self,
        inp: stages.CompilableProject[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
    ) -> stages.CompilationArtifact:
        src_dir = cache.get_cache_folder(inp, self.cache_lifetime)

        # If we are compiling the same program at the same time (e.g. multiple MPI ranks),
        # we need to make sure that only one of them accesses the same build directory for compilation.
        with locking.lock(src_dir):
            data = build_data.read_data(src_dir)

            if not data or not is_compiled(data) or self.force_recompile:
                self.builder_factory(inp, self.cache_lifetime).build()

            new_data = build_data.read_data(src_dir)

            if not new_data or not is_compiled(new_data) or not module_exists(new_data, src_dir):
                raise CompilationError(
                    f"On-the-fly compilation unsuccessful for '{inp.program_source.entry_point.name}'."
                )

            (src_dir / stages._LOADER_MODULE_FILENAME).write_text(
                self._render_loader(new_data.module, new_data.entry_point_name)
            )

        return stages.CompilationArtifact(src_dir=src_dir)

    def _render_loader(self, module: pathlib.Path, entry_point_name: str) -> str:
        return textwrap.dedent(f"""\
            from gt4py.next.otf.compilation import importer


            def load(src_dir):
                m = importer.import_from_path(
                    src_dir / {str(module)!r},
                    sys_modules_prefix="gt4py.__compiled_programs__.",
                )
                return getattr(m, {entry_point_name!r})
            """)


class CompilationError(RuntimeError): ...
