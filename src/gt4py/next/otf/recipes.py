# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses

from typing import Any

from gt4py.next.otf import definitions, stages, workflow


@dataclasses.dataclass(frozen=True)
class OTFCompileWorkflow(workflow.NamedStepSequence):
    """The typical compiled backend steps composed into a workflow."""

    translation: definitions.TranslationStep
    bindings: workflow.Workflow[stages.ProgramSource, stages.CompilableProject]
    compilation: workflow.Workflow[stages.CompilableProject, stages.ExecutableProgram]
    decoration: workflow.Workflow[stages.ExecutableProgram, stages.ExecutableProgram]

    def build_artifact(self, inp: definitions.CompilableProgramDef) -> Any:
        """Run translation -> bindings -> compilation.build and stop.

        Used by the process-pool compile path: the returned descriptor is picklable and can be
        handed back to the main process, which then calls :meth:`finalize_artifact` to import
        the built module / reload the compiled SDFG and apply backend decoration.

        The shape of the returned "artifact" is backend-specific — a GTFN
        :class:`~gt4py.next.otf.stages.BuildArtifact` (src_dir + module + entry point) for
        C++-based backends, a DaCe-specific descriptor for DaCe — as long as it's pickle-safe
        and the same ``compilation`` step's :meth:`load` knows how to consume it.
        """
        compilable = self.bindings(self.translation(inp))
        if not hasattr(self.compilation, "build"):
            raise RuntimeError(
                f"Compilation step {type(self.compilation).__name__} does not support "
                "'build_artifact'; process-pool compilation requires a splittable compiler."
            )
        return self.compilation.build(compilable)  # type: ignore[attr-defined]

    def finalize_artifact(self, artifact: Any) -> stages.ExecutableProgram:
        """Rehydrate a built artifact in the current process and apply the backend's
        decoration step.
        """
        if not hasattr(self.compilation, "load"):
            raise RuntimeError(
                f"Compilation step {type(self.compilation).__name__} does not support "
                "'load'; process-pool compilation requires a splittable compiler."
            )
        return self.decoration(self.compilation.load(artifact))  # type: ignore[attr-defined]
