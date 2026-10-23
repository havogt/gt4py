# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses

from gt4py.next.otf import artifacts, stages, workflow


@dataclasses.dataclass(frozen=True)
class OTFSourceWorkflow(
    workflow.NamedStepSequence[stages.CompilableProgramDef, artifacts.ExtensionSource]
):
    """The steps producing the source of a compiled backend, cached as one unit."""

    translation: stages.TranslationStep
    bindings: workflow.Workflow[artifacts.ProgramSource, artifacts.ExtensionSource]

    def __call__(self, inp: stages.CompilableProgramDef) -> artifacts.ExtensionSource:
        source = super().__call__(inp)
        # Computed before the source is cached, so it is stored with it.
        source.fingerprint  # noqa: B018 [useless-expression]
        return source


@dataclasses.dataclass(frozen=True)
class OTFCompileWorkflow(
    workflow.NamedStepSequence[stages.CompilableProgramDef, artifacts.CompilationArtifact]
):
    """The typical compiled backend steps composed into a workflow.

    ``source`` is an `OTFSourceWorkflow`, possibly wrapped in a `workflow.CachedStep`.
    """

    source: workflow.Workflow[stages.CompilableProgramDef, artifacts.ExtensionSource]
    compilation: workflow.Workflow[artifacts.ExtensionSource, artifacts.CompilationArtifact]

    def lookup(self, inp: stages.CompilableProgramDef) -> artifacts.CompilationArtifact | None:
        """Return the artifact of a completed build of `inp`, or None, without building.

        Requires the `source` and `compilation` steps to support `lookup`.
        """
        if not (hasattr(self.source, "lookup") and hasattr(self.compilation, "lookup")):
            return None
        source = self.source.lookup(inp)
        if source is None:
            return None
        return self.compilation.lookup(source)

    @property
    def _source_workflow(self) -> OTFSourceWorkflow:
        source = self.source.step if isinstance(self.source, workflow.CachedStep) else self.source
        assert isinstance(source, OTFSourceWorkflow)
        return source

    @property
    def translation(self) -> stages.TranslationStep:
        return self._source_workflow.translation

    @property
    def bindings(self) -> workflow.Workflow[artifacts.ProgramSource, artifacts.ExtensionSource]:
        return self._source_workflow.bindings
