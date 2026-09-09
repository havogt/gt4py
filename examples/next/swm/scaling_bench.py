# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Timing-only scaling of the sharded SWM over (size, layout, transport, mode).

    cd <gt4py>
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        python examples/next/swm/scaling_bench.py \
            --sizes 32,64,128 --layouts 1x1,2x1,2x2,4x1 --transports padded,allgather \
            --modes fwd,grad,ref --out scaling.jsonl

    python examples/next/swm/scaling_bench.py --table scaling.jsonl

``--sizes`` are global square edges (strong scaling); with ``--weak`` they are the
per-device block edge, ``M = s*Rx``, ``N = s*Ry``. Modes: ``fwd`` is the jitted sharded
forward, ``grad`` the ``jax.grad`` of ``cost`` through the same program, ``ref`` the
single-device ``reference_program`` (1x1 only) and its gradient ``ref_grad``.

Sizes are processed ascending; after an out-of-memory failure the larger sizes of that
(layout, transport, mode) are skipped. ``--distributed`` as in ``bench_transports.py``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import time

import jax
import numpy as np

from bench_transports import _environment, _init_distributed
from halo_transports import Layout, available_transports, block, get_transport
from initial_conditions import initialize_interior
from swm_sharded import (
    N_STEPS,
    cost,
    dx,
    dy,
    put_global,
    radius,
    reference_program,
    sharded_program,
)

MODES = ("fwd", "grad", "ref", "ref_grad")


def _grad_of(program):
    return jax.jit(jax.grad(lambda *f: cost(program(*f)), argnums=(0, 1, 2)))


def _case(transport, layout, mode, n_steps):
    """``(jitted fn, device args)`` for one measurement."""
    fields = initialize_interior(np, layout.M, layout.N, dx, dy, radius)
    if mode.startswith("ref"):
        prog = reference_program(n_steps, layout.M, layout.N)
        args = tuple(jax.device_put(a) for a in fields)
    else:
        prog = sharded_program(transport, layout, n_steps)
        args = tuple(put_global(block(a, layout), layout) for a in fields)
    return (_grad_of(prog) if mode.endswith("grad") else prog), args


def _time(fn, args, repeats):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    compile_s = time.perf_counter() - t0
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - t0)
    return compile_s, samples


def _peak_mem():
    # process-lifetime peak: no API resets it, so it is monotone across rows
    stats = jax.local_devices()[0].memory_stats()
    return None if stats is None else stats.get("peak_bytes_in_use")


def measure(transport, layout, mode, n_steps, repeats):
    try:
        fn, args = _case(transport, layout, mode, n_steps)
        compile_s, samples = _time(fn, args, repeats)
    except Exception as e:  # noqa: BLE001 - only OOM is recoverable
        if "RESOURCE_EXHAUSTED" not in str(e):
            raise
        return {"status": "oom"}
    cells = layout.M * layout.N
    best = min(samples)
    return {
        "status": "ok",
        "compile_s": compile_s,
        "per_step_ms": 1e3 * best / n_steps,
        "samples_step_ms": [1e3 * t / n_steps for t in samples],
        "cells": cells,
        "cell_updates_per_s": cells * n_steps / best,
        "peak_mem_cumulative_bytes": _peak_mem(),
    }


def _layout(size, spec, weak):
    rx, ry = (int(v) for v in spec.lower().split("x"))
    return Layout(size * rx, size * ry, rx, ry) if weak else Layout(size, size, rx, ry)


def _cases(sizes, layout_specs, transports, modes, weak):
    for spec in layout_specs:
        for mode in modes:
            for transport in [None] if mode.startswith("ref") else transports:
                for size in sorted(sizes):
                    layout = _layout(size, spec, weak)
                    if transport is None and layout.P != 1:
                        continue
                    yield size, spec, layout, transport, mode


def run(sizes, layout_specs, transports, modes, n_steps, repeats, weak, out_path, distributed):
    if transports == ["all"]:
        transports = available_transports()
    if "ref" in modes:
        modes = modes + ["ref_grad"]
    env = _environment(distributed)
    write = env["process_index"] == 0
    n_dev = jax.device_count()
    oom = set()
    rows = []
    for size, spec, layout, transport, mode in _cases(sizes, layout_specs, transports, modes, weak):
        row = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            **env,
            "size": size,
            "weak": weak,
            "M": layout.M,
            "N": layout.N,
            "layout": spec,
            "P": layout.P,
            "MLOC": layout.MLOC,
            "NLOC": layout.NLOC,
            "transport": transport,
            "mode": mode,
            "n_steps": n_steps,
            "repeats": repeats,
        }
        key = (spec, transport, mode)
        sharded = transport is not None
        if key in oom or (sharded and (layout.P > n_dev or (distributed and layout.P != n_dev))):
            row["status"] = "skipped"
        else:
            tr = get_transport(transport) if sharded else None
            row.update(measure(tr, layout, mode, n_steps, repeats))
            if row["status"] == "oom":
                oom.add(key)
        rows.append(row)
        print(
            f"{layout.M}x{layout.N:<6} {spec:<5} {transport or '-':<12} {mode:<9} "
            f"{row['status']:<8} {row.get('per_step_ms', float('nan')):.4f} ms/step",
            flush=True,
        )
        if write:
            with open(out_path, "a") as f:
                f.write(json.dumps(row) + "\n")
    return rows


COLUMNS = (
    "size",
    "layout",
    "transport",
    "mode",
    "per_step_ms",
    "cell_updates_per_s",
    "peak_mem_cum_GB",
    "status",
)


def table(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    out = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in rows:
        ms = r.get("per_step_ms")
        cps = r.get("cell_updates_per_s")
        mem = r.get("peak_mem_cumulative_bytes")
        out.append(
            f"| {r['M']}x{r['N']} | {r['layout']} | {r['transport'] or '-'} | {r['mode']} | "
            f"{'-' if ms is None else f'{ms:.4f}'} | {'-' if cps is None else f'{cps:.3e}'} | "
            f"{'-' if mem is None else f'{mem / 1e9:.2f}'} | {r['status']} |"
        )
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sizes", default="32,64,128", help="global M=N, or block edge with --weak")
    ap.add_argument("--weak", action="store_true", help="sizes are per-device: M=s*Rx, N=s*Ry")
    ap.add_argument("--layouts", default="1x1,2x1,2x2,4x1")
    ap.add_argument("--transports", default="all", help="'all' or a comma-separated list")
    ap.add_argument("--modes", default="fwd,grad,ref", help=f"subset of {','.join(MODES)}")
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--repeats", type=int, default=5, help="timed calls after the compile call")
    ap.add_argument("--out", default="scaling.jsonl")
    ap.add_argument("--distributed", action="store_true", help="jax.distributed.initialize()")
    ap.add_argument("--table", metavar="SCALING.JSONL", help="print the markdown table and exit")
    a = ap.parse_args(argv)
    if a.table:
        print(table(a.table))
        return 0
    modes = [m.strip() for m in a.modes.split(",")]
    bad = set(modes) - set(MODES)
    if bad:
        raise SystemExit(f"unknown modes {sorted(bad)}; choose from {MODES}")
    distributed = _init_distributed(a.distributed)
    rows = run(
        [int(s) for s in a.sizes.split(",")],
        [s.strip() for s in a.layouts.split(",")],
        [t.strip() for t in a.transports.split(",")],
        modes,
        a.steps,
        a.repeats,
        a.weak,
        a.out,
        distributed,
    )
    if rows and rows[0]["process_index"] == 0:
        print(f"\n{len(rows)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
