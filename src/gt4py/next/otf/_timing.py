# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Temporary timing instrumentation for the process-pool prototype. NOT FOR MERGING.

Every record is a single stderr line:

    GT4PY_TIMING pid=1234 wall=1751551234.567 <event> dt=1.234 program=foo

Events emitted (dt in seconds):

- ``pool_created``: process pool constructed (max_workers).
- ``worker_ready``: pool worker finished initialization; ``spawn_dt`` is the
  time since pool creation (includes spawn + interpreter start + gt4py import).
- ``submit``: full main-side cost of submitting one variant (includes
  ``frontend_lowering`` and ``executor_pickle``).
- ``frontend_lowering``: DSL -> ITIR lowering in the main process (serial!).
- ``connectivity_dump``: one-time dump of a connectivity table to file.
- ``executor_pickle``: pickling the executor for the worker.
- ``worker_job_received``: job arrived in a worker; ``queue_dt`` is the time
  since submission (queue wait + payload transfer + possibly worker spawn).
- ``worker_unpickle_executor`` / ``worker_execute``: worker-side unpickle and
  the actual translation+codegen+build.
- ``compile_in_calling_thread``: serial fallback (should not appear).
- ``artifact_load``: dlopen/load of the compiled artifact in the main process.
- ``wait_for_compilation``: duration of the barrier.

Aggregate per event, e.g.:

    grep -h GT4PY_TIMING *.log | awk '{split($4,e," "); ev=$4; for(i=5;i<=NF;i++) if ($i ~ /^dt=/) {sub("dt=","",$i); s[ev]+=$i; n[ev]++}} END {for (ev in s) printf "%-28s n=%-5d total=%8.1fs\n", ev, n[ev], s[ev]}'
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from typing import Any, Generator


def log(event: str, **fields: Any) -> None:
    parts = " ".join(f"{k}={v}" for k, v in fields.items())
    print(
        f"GT4PY_TIMING pid={os.getpid()} wall={time.time():.3f} {event} {parts}",
        file=sys.stderr,
        flush=True,
    )


@contextlib.contextmanager
def span(event: str, **fields: Any) -> Generator[None, None, None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        log(event, dt=f"{time.perf_counter() - start:.3f}", **fields)
