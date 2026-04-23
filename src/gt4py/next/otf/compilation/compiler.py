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
from typing import Protocol, TypeVar

import factory

from gt4py._core import locking
from gt4py.next import config
from gt4py.next.otf import code_specs, definitions, stages, workflow
from gt4py.next.otf.compilation import build_data, cache, importer


T = TypeVar("T")


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
class Compiler(
    workflow.ChainableWorkflowMixin[
        stages.CompilableProject[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
        stages.ExecutableProgram,
    ],
    workflow.ReplaceEnabledWorkflowMixin[
        stages.CompilableProject[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
        stages.ExecutableProgram,
    ],
    definitions.CompilationStep[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
):
    """Use any build system (via configured factory) to compile a GT4Py program to a ``gt4py.next.otf.stages.CompiledProgram``."""

    cache_lifetime: config.BuildCacheLifetime
    builder_factory: BuildSystemProjectGenerator[CPPLikeCodeSpecT, code_specs.PythonCodeSpec]
    force_recompile: bool = False

    def build(
        self,
        inp: stages.CompilableProject[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
    ) -> stages.BuildArtifact:
        """Run codegen + native build only. Returns a picklable on-disk artifact descriptor.

        Split out from ``__call__`` so that the heavy part (safe to run in a worker process)
        is separable from importing the freshly built module (which must happen in the process
        that will eventually call it).
        """
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

        return stages.BuildArtifact(
            src_dir=src_dir, module=new_data.module, entry_point_name=new_data.entry_point_name
        )

    def __call__(
        self,
        inp: stages.CompilableProject[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
    ) -> stages.ExecutableProgram:
        return self.load(self.build(inp))

    def load(self, artifact: stages.BuildArtifact) -> stages.ExecutableProgram:
        """Dynamically import a previously-built module and return its entry point.

        Counterpart to :meth:`build`: runs whatever process will ultimately call the program.
        Kept as an instance method (not a free function) so that every compilation step in
        the OTF workflow exposes a uniform ``build`` / ``load`` contract and recipes can
        dispatch through it without knowing backend specifics.
        """
        return load_artifact(artifact)


def load_artifact(artifact: stages.BuildArtifact) -> stages.ExecutableProgram:
    """Dynamically import a previously-built module and return its entry point."""
    m = importer.import_from_path(
        artifact.src_dir / artifact.module, sys_modules_prefix="gt4py.__compiled_programs__."
    )
    return getattr(m, artifact.entry_point_name)


class CompilerFactory(factory.Factory):
    class Meta:
        model = Compiler


class CompilationError(RuntimeError): ...
