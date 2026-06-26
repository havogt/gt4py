# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal contract tests for ``stages.CompilationArtifact``."""

import pathlib
import pickle

from gt4py.next.otf import stages


def test_compilation_artifact_pickle_round_trip(tmp_path: pathlib.Path):
    artifact = stages.CompilationArtifact(src_dir=tmp_path)
    restored = pickle.loads(pickle.dumps(artifact))
    assert restored == artifact


def test_compilation_artifact_load_execs_loader_file(tmp_path: pathlib.Path):
    (tmp_path / stages._LOADER_MODULE_FILENAME).write_text(
        "def load(src_dir):\n    return ('loaded', src_dir)\n"
    )
    artifact = stages.CompilationArtifact(src_dir=tmp_path)
    assert artifact.load() == ("loaded", tmp_path)
