# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

import pytest

from gt4py._core import filecache
from gt4py.next import fingerprinting
from gt4py.next.otf import artifacts, recipes, workflow
from gt4py.next.otf.binding import interface


calls: list[str] = []


def _translate(name: str) -> artifacts.ProgramSource:
    calls.append("translation")
    return artifacts.ProgramSource(
        entry_point=interface.Function(name=name, parameters=()),
        source_code=f"void {name}() {{}}",
        library_deps=(),
        code_spec=artifacts.CPPCodeSpec(),
    )


def _bind(program_source: artifacts.ProgramSource) -> artifacts.ExtensionSource:
    calls.append("bindings")
    return artifacts.ExtensionSource(
        program_source=program_source,
        binding_source=artifacts.BindingSource(source_code="", library_deps=()),
    )


@pytest.fixture
def cached_source(tmp_path):
    calls.clear()
    return workflow.CachedStep.persistent(
        recipes.OTFSourceWorkflow(translation=_translate, bindings=_bind),
        input_fingerprinter=fingerprinting.strict_fingerprinter,
        cache=filecache.FileCache(tmp_path),
    )


def test_source_hit_skips_translation_and_bindings(cached_source):
    first = cached_source("prog")
    second = cached_source("prog")

    assert calls == ["translation", "bindings"]
    assert second == first


def test_cached_source_carries_its_fingerprint(cached_source, monkeypatch):
    expected = fingerprinting.strict_fingerprinter(cached_source("prog"))
    key = cached_source.cache_key("prog")

    def no_fingerprint(obj):
        raise AssertionError("the cached source must not be fingerprinted again")

    monkeypatch.setattr(fingerprinting, "strict_fingerprinter", no_fingerprint)
    reloaded = cached_source.cache[key]

    assert reloaded.fingerprint == expected


@dataclasses.dataclass(frozen=True)
class _LookupCompiler:
    built: tuple[str, ...]

    def __call__(self, source: artifacts.ExtensionSource) -> str:
        return source.program_source.entry_point.name

    def lookup(self, source: artifacts.ExtensionSource) -> str | None:
        name = source.program_source.entry_point.name
        return name if name in self.built else None


def test_compile_workflow_lookup_composes_step_lookups(cached_source):
    compile_workflow = recipes.OTFCompileWorkflow(
        source=cached_source, compilation=_LookupCompiler(built=("built", "unbuilt_source"))
    )
    cached_source("built")
    cached_source("not_built")

    assert compile_workflow.lookup("built") == "built"
    assert compile_workflow.lookup("not_built") is None
    assert compile_workflow.lookup("unbuilt_source") is None
    assert calls == ["translation", "bindings"] * 2


def test_compile_workflow_lookup_needs_lookup_on_every_step(cached_source):
    cached_source("built")
    compile_workflow = recipes.OTFCompileWorkflow(
        source=cached_source, compilation=lambda source: source.program_source.entry_point.name
    )

    assert compile_workflow.lookup("built") is None
