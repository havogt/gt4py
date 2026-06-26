# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import datetime
import enum
import os
import pathlib
import warnings
from typing import Final


class BuildCacheLifetime(enum.Enum):
    SESSION = 1
    PERSISTENT = 2


class CMakeBuildType(enum.Enum):
    """
    CMake build types enum.

    Member values have to be valid CMake syntax.
    """

    DEBUG = "Debug"
    RELEASE = "Release"
    REL_WITH_DEB_INFO = "RelWithDebInfo"
    MIN_SIZE_REL = "MinSizeRel"


def env_flag_to_bool(name: str, default: bool) -> bool:
    """Convert environment variable string variable to a bool value."""
    flag_value = os.environ.get(name, None)
    if flag_value is None:
        return default
    match flag_value.lower():
        case "0" | "false" | "off":
            return False
        case "1" | "true" | "on":
            return True
        case _:
            raise ValueError(
                "Invalid GT4Py environment flag value: use '0 | false | off' or '1 | true | on'."
            )


def env_flag_to_int(name: str, default: int) -> int:
    """Convert environment variable string variable to an int value."""
    flag_value = os.environ.get(name, None)
    if flag_value is None:
        return default
    try:
        return int(flag_value)
    except ValueError:
        raise ValueError(
            f"Invalid GT4Py environment flag value: {flag_value} is not an integer."
        ) from None


#: Master debug flag
#: Changes defaults for all the other options to be as helpful for debugging as possible.
#: Does not override values set in environment variables.
DEBUG: Final[bool] = env_flag_to_bool("GT4PY_DEBUG", default=False)


#: Verbose flag for DSL compilation errors
VERBOSE_EXCEPTIONS: bool = env_flag_to_bool(
    "GT4PY_VERBOSE_EXCEPTIONS", default=True if DEBUG else False
)


#: Where generated code projects should be persisted.
#: Only active if BUILD_CACHE_LIFETIME is set to PERSISTENT
BUILD_CACHE_DIR: pathlib.Path = (
    pathlib.Path(os.environ.get("GT4PY_BUILD_CACHE_DIR", pathlib.Path.cwd())) / ".gt4py_cache"
)


#: Whether generated code projects should be kept around between runs.
#: - SESSION: generated code projects get destroyed when the interpreter shuts down
#: - PERSISTENT: generated code projects are written to BUILD_CACHE_DIR and persist between runs
BUILD_CACHE_LIFETIME: BuildCacheLifetime = BuildCacheLifetime[
    os.environ.get("GT4PY_BUILD_CACHE_LIFETIME", "persistent" if DEBUG else "session").upper()
]


#: Build type to be used when CMake is used to compile generated code.
#: Might have no effect when CMake is not used as part of the toolchain.
# FIXME[#2447](egparedes): compile-time setting, should be included in the build cache key.
CMAKE_BUILD_TYPE: CMakeBuildType = CMakeBuildType[
    os.environ.get("GT4PY_CMAKE_BUILD_TYPE", "debug" if DEBUG else "release").upper()
]


#: Experimental, use at your own risk: assume horizontal dimension has stride 1
# FIXME[#2447](egparedes): compile-time setting, should be included in the build cache key.
UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE: bool = env_flag_to_bool(
    "GT4PY_UNSTRUCTURED_HORIZONTAL_HAS_UNIT_STRIDE", default=False
)


#: Add GPU trace markers (NVTX, ROC-TX) to the generated code, at compile time.
# FIXME[#2447](egparedes): compile-time setting, should be included in the build cache key.
ADD_GPU_TRACE_MARKERS: bool = env_flag_to_bool("GT4PY_ADD_GPU_TRACE_MARKERS", default=False)


#: Experimental: lower skip-value sum-reductions to branchless load-then-mask
#: (dace-style) instead of a per-neighbor `can_deref` branch. The gathered field's
#: K-row-base is then hoisted once across all neighbors. Requires the matching
#: clamped `deref` in the gridtools fn header (compile macro GT4PY_FN_BRANCHLESS_SKIP_REDUCE).
# FIXME[#2447](egparedes): compile-time setting, should be included in the build cache key.
FN_BRANCHLESS_SKIP_REDUCE: bool = env_flag_to_bool(
    "GT4PY_FN_BRANCHLESS_SKIP_REDUCE", default=False
)


#: Experimental: on top of the branchless skip-reduce lowering, hoist the neighbor
#: row of each gathered field out of the unrolled reduction fold (resolve the row base
#: once via `gtfn::neighbor_row`, then offset per neighbor via `gtfn::horizontal_shift_to`),
#: instead of re-deriving the row base per neighbor. Requires FN_BRANCHLESS_SKIP_REDUCE
#: AND the matching `neighbor_row`/`horizontal_shift_to` helpers in the gridtools fn header.
# FIXME[#2447](egparedes): compile-time setting, should be included in the build cache key.
FN_HOIST_NEIGHBOR_ROW: bool = env_flag_to_bool("GT4PY_FN_HOIST_NEIGHBOR_ROW", default=False)


#: Experimental: hoist the neighbor row of a NON-SKIP (non-branchless) connectivity
#: sum-reduction. The plain unrolled fold `reduce(plus, 0)(neighbors)` re-derives the
#: row base (m_index * index_stride) once per neighbor via `shift(field, Conn, _i)`.
#: This resolves the row once via `gtfn::neighbor_row` and offsets per neighbor with
#: `gtfn::horizontal_shift_to`, collapsing the per-neighbor integer/addressing SASS on
#: standalone gather kernels (e.g. hmom's C2E `_fun_0`). Requires the row helpers in the
#: gridtools fn header (compile macro GT4PY_FN_BRANCHLESS_SKIP_REDUCE, which the venv
#: header exposes them under). Independent of the branchless skip-value lowering.
# FIXME[#2447](egparedes): compile-time setting, should be included in the build cache key.
FN_HOIST_NEIGHBOR_ROW_NONSKIP: bool = env_flag_to_bool(
    "GT4PY_FN_HOIST_NEIGHBOR_ROW_NONSKIP", default=False
)


#: Experimental: fuse 2+ sibling unrolled reduction folds in one stencil body that gather
#: over the SAME (field, connectivity) into ONE fold with a tuple accumulator, so the shared
#: per-neighbor `deref(shift(field, Conn, _i))` is emitted once instead of re-gathered per
#: sibling. Removes the redundant LDG + per-neighbor address math the gtfn fold otherwise
#: re-derives (nvcc/ptxas cannot CSE them — distinct iteration vars). No extra kernel/temp →
#: identical DRAM traffic, pure in-kernel compute cut. Default OFF, codegen byte-identical
#: when unset.
# FIXME[#2447](egparedes): compile-time setting, should be included in the build cache key.
FN_FUSE_SIBLING_REDUCE: bool = env_flag_to_bool(
    "GT4PY_FN_FUSE_SIBLING_REDUCE", default=False
)


#: Experimental: restrict the branchless skip-value reduction REWRITE so it fires only on
#: small standalone gather kernels (executors below this stencil-body node-count threshold),
#: NOT on a large co-resident fused kernel where the added load-then-mask compute regresses
#: it. 0 (default) = no size gate (apply globally, the original behavior). A positive value
#: enables selectivity: an executor whose stencil expression has more than this many IR nodes
#: keeps the original branched skip-reduce; smaller ones get the branchless+hoist rewrite.
# FIXME[#2447](egparedes): compile-time setting, should be included in the build cache key.
FN_BRANCHLESS_SKIP_REDUCE_MAX_NODES: int = int(
    os.environ.get("GT4PY_FN_BRANCHLESS_SKIP_REDUCE_MAX_NODES", "0")
)


#: Number of threads to use to use for compilation (0 = synchronous compilation).
#: Default:
#: - use os.cpu_count(), TODO(havogt): in Python >= 3.13 use `process_cpu_count()`
#: - if os.cpu_count() is None we are conservative and use 1 job,
#: - if the number is huge (e.g. HPC system) we limit to a smaller number
BUILD_JOBS: int = int(os.environ.get("GT4PY_BUILD_JOBS", min(os.cpu_count() or 1, 32)))


#: User-defined level to enable metrics at lower or equal level.
#: Enabling metrics collection will do extra synchronization and will have
#: impact on runtime performance.
COLLECT_METRICS_LEVEL: int = env_flag_to_int("GT4PY_COLLECT_METRICS_LEVEL", default=0)


#: File path to dump collected metrics at exit, if COLLECT_METRICS_LEVEL is enabled.
#: If set to a True value, it defaults to "gt4py_metrics_YYYYMMDD_HHMMSS.json" in
#: the current folder.
DUMP_METRICS_AT_EXIT: str | None = None


#: Filter out DaCe related warnings. If not set warnings will be suppressed if the
#: code runs in no debug mode.
SKIP_DACE_WARNINGS: bool = env_flag_to_bool("GT4PY_SKIP_DACE_WARNINGS", default=not __debug__)


if SKIP_DACE_WARNINGS:
    # NOTE: Ideally we would suppress the warnings using context managers directly in
    #   the backend. However, because this is not thread safe in Python versions before
    #   3.14, we have to do it here.
    warnings.filterwarnings(action="ignore", module="^dace(\..+)?")
    warnings.filterwarnings(
        action="ignore", module="^gt4py.next.program_processors.runners.dace.transformations(\..+)?"
    )


def _init_dump_metrics_filename() -> str:
    return f"gt4py_metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


_dump_metrics_at_exit_env = os.environ.get("GT4PY_DUMP_METRICS_AT_EXIT", None)
if _dump_metrics_at_exit_env is not None:
    try:
        if env_flag_to_bool("GT4PY_DUMP_METRICS_AT_EXIT", default=False):
            DUMP_METRICS_AT_EXIT = _init_dump_metrics_filename()
    except ValueError:
        DUMP_METRICS_AT_EXIT = _dump_metrics_at_exit_env


#: The default for whether to allow jit-compilation for a compiled program.
#: This default can be overriden per program.
ENABLE_JIT_DEFAULT: bool = env_flag_to_bool("GT4PY_ENABLE_JIT_DEFAULT", default=True)
