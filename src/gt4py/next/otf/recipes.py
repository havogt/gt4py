# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses

from gt4py.next import config
from gt4py.next.otf import stages, step_types, workflow


@dataclasses.dataclass(frozen=True)
class OTFCompileWorkflow(workflow.NamedStepSequence):
    """The typical compiled backend steps composed into a workflow."""

    translation: step_types.TranslationStep
    bindings: workflow.Workflow[stages.ProgramSource, stages.CompilableSource]
    compilation: workflow.Workflow[stages.CompilableSource, stages.CompiledProgram]
    decoration: workflow.Workflow[stages.CompiledProgram, stages.CompiledProgram]

    def __post_init__(self) -> None:
        from gt4py.next import metrics

        if config.COLLECT_METRICS:
            for key in dataclasses.fields(self):
                # TODO we don't know the program name...
                object.__setattr__(
                    self,
                    key.name,
                    workflow.TimedStep(
                        metrics.global_metric_container["foo"][key.name], getattr(self, key.name)
                    ),
                )
