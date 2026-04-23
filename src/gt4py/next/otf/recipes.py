# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses

from gt4py.next.otf import definitions, stages, workflow


@dataclasses.dataclass(frozen=True)
class OTFBuildWorkflow(
    workflow.NamedStepSequence[definitions.CompilableProgramDef, stages.BuildArtifact]
):
    """Phase 1 of OTF compilation: run everything that produces an on-disk artifact.

    The output type of this phase is deliberately picklable so that async-compile
    strategies (e.g. a process pool) can run the whole phase in a worker and ship
    only the artifact descriptor back to the caller.
    """

    translation: definitions.TranslationStep
    bindings: workflow.Workflow[stages.ProgramSource, stages.CompilableProject]
    compilation: workflow.Workflow[stages.CompilableProject, stages.BuildArtifact]


@dataclasses.dataclass(frozen=True)
class OTFFinalizeWorkflow(
    workflow.NamedStepSequence[stages.BuildArtifact, stages.ExecutableProgram]
):
    """Phase 2 of OTF compilation: rehydrate a built artifact into a callable program.

    Runs in the process that will ultimately invoke the compiled program — dynamic
    module imports and live ``ctypes`` handles can't cross a process boundary.
    """

    load: workflow.Workflow[stages.BuildArtifact, stages.ExecutableProgram]
    decoration: workflow.Workflow[stages.ExecutableProgram, stages.ExecutableProgram]


@dataclasses.dataclass(frozen=True)
class OTFCompileWorkflow(
    workflow.NamedStepSequence[definitions.CompilableProgramDef, stages.ExecutableProgram]
):
    """Full OTF pipeline: ``build`` then ``finalize``.

    The split is a pipeline boundary, not a pair of alternate methods: callers that
    want to run only the heavy build can call ``.build(inp)`` directly; the
    inherited ``__call__`` composes both phases for the common synchronous case.
    """

    build: workflow.Workflow[definitions.CompilableProgramDef, stages.BuildArtifact]
    finalize: workflow.Workflow[stages.BuildArtifact, stages.ExecutableProgram]
