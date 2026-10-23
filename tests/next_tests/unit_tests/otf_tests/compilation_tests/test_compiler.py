# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for ``CPPCompilationArtifact`` and ``CPPCompiler``."""

import dataclasses
import pathlib
import pickle
import uuid

import pytest

from gt4py._core import definitions as core_defs
from gt4py.next import config
from gt4py.next.otf.compilation import build_data, cache, compiler

from next_tests.unit_tests.otf_tests.compilation_tests.build_systems_tests.conftest import (
    extension_source_example,
    program_source_example,
)


def test_cpp_compilation_artifact_pickle_round_trip(tmp_path: pathlib.Path):
    artifact = compiler.CPPCompilationArtifact(
        src_dir=tmp_path,
        module=pathlib.Path("entry.so"),
        entry_point_name="entry",
        device_type=core_defs.DeviceType.CPU,
    )
    restored = pickle.loads(pickle.dumps(artifact))
    assert restored == artifact


@dataclasses.dataclass
class _FakeProject:
    root: pathlib.Path
    builds: list[pathlib.Path]

    def build(self) -> None:
        self.builds.append(self.root)
        (self.root / "entry.so").write_bytes(b"")
        build_data.write_data(
            build_data.BuildData(
                status=build_data.BuildStatus.COMPILED,
                module=pathlib.Path("entry.so"),
                entry_point_name="entry",
            ),
            self.root,
        )


def _fake_compiler(builds: list[pathlib.Path], **kwargs) -> compiler.CPPCompiler:
    def builder_factory(source, lifetime):
        return _FakeProject(cache.get_cache_folder(source, lifetime), builds)

    return compiler.CPPCompiler(
        cache_lifetime=config.BuildCacheLifetime.SESSION,
        builder_factory=builder_factory,
        device_type=core_defs.DeviceType.CPU,
        fingerprint_builder_factory=False,
        **kwargs,
    )


@pytest.fixture
def unique_extension_source(extension_source_example):
    source = dataclasses.replace(
        extension_source_example.program_source,
        entry_point=dataclasses.replace(
            extension_source_example.program_source.entry_point, name=f"fake_{uuid.uuid4().hex}"
        ),
    )
    return dataclasses.replace(extension_source_example, program_source=source)


def test_completed_build_is_reused_without_lock(unique_extension_source, monkeypatch):
    builds: list[pathlib.Path] = []
    comp = _fake_compiler(builds)
    first = comp(unique_extension_source)

    def no_lock(*args, **kwargs):
        raise AssertionError("a completed build must be reused without the lock")

    monkeypatch.setattr(compiler.locking, "lock", no_lock)
    second = comp(unique_extension_source)

    assert len(builds) == 1
    assert second.src_dir / second.module == first.src_dir / first.module


def test_build_without_module_is_rebuilt(unique_extension_source):
    builds: list[pathlib.Path] = []
    comp = _fake_compiler(builds)
    artifact = comp(unique_extension_source)
    (artifact.src_dir / artifact.module).unlink()

    comp(unique_extension_source)

    assert len(builds) == 2


def test_force_recompile_rebuilds_under_lock(unique_extension_source, monkeypatch):
    builds: list[pathlib.Path] = []
    _fake_compiler(builds)(unique_extension_source)
    locked: list[pathlib.Path] = []
    real_lock = compiler.locking.lock

    def recording_lock(directory):
        locked.append(pathlib.Path(directory))
        return real_lock(directory)

    monkeypatch.setattr(compiler.locking, "lock", recording_lock)
    _fake_compiler(builds, force_recompile=True)(unique_extension_source)

    assert len(builds) == 2
    assert locked == [builds[-1]]
