# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Run the transport battery over a (transport, layout) grid and write one JSON row each.

    cd <gt4py>
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        python examples/next/swm/bench_transports.py \
            --transports all --layouts 1x1,2x1,1x2,2x2,4x2,2x4 --out results.jsonl

    # rewrite the summary table from a results file (any machine)
    python examples/next/swm/bench_transports.py --table results.jsonl

Rows are appended and carry the environment they were measured in, so laptop and cluster
files can be concatenated.

Multi-process runs (``--distributed``) call ``jax.distributed.initialize()`` before anything
touches a device; JAX auto-detects Slurm, every layout must use *all* devices, and only
process 0 writes rows. Tested with two real processes over gloo on CPU; untested on more
than one node, on NCCL and on GPU. See ``santis_bench.sbatch``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys

import jax

import swm_battery as B
from halo_transports import available_transports, get_transport
from jax_compat import shard_map_api
from swm_sharded import GT4PY_TRACER_DISPATCH, N_STEPS

CORRECTNESS = B.TESTS[:7]


def _init_distributed(enabled: bool) -> bool:
    """``jax.distributed.initialize()`` under Slurm, before any device is created."""
    if not enabled:
        return False
    if int(os.environ.get("SLURM_NTASKS", "1")) <= 1:
        print("--distributed: SLURM_NTASKS <= 1, staying single-process", file=sys.stderr)
        return False
    jax.distributed.initialize()  # picks up SLURM_* itself
    return True


def _scalars(res, n_steps, vol):
    """Flatten the battery result dict to one flat row of scalars.

    ``vol`` comes from the prepared tables, not from T7, so a transport that does not
    compile still reports the volume its tables imply.
    """

    def g(test, key):
        return res[test].get(key)

    row = {
        "n_steps": n_steps,
        "all_pass": all(res[t].get("pass") for t in B.TESTS),
        **{f"{t}_pass": bool(res[t].get("pass")) for t in B.TESTS},
        **{f"{t}_status": res[t].get("status", "ok") for t in B.TESTS},
        "T0_fwd_array_equal": g("T0", "fwd_array_equal"),
        "T0_grad_max_rel_diff": g("T0", "grad_max_rel_diff"),
        "T1_max_abs_diff": g("T1", "max_abs_diff"),
        "T2_rel_diff": g("T2", "rel_diff"),
        "T2_dense_fwd_diff": g("T2", "dense_fwd_diff"),
        "T2_dense_vjp_diff": g("T2", "dense_vjp_diff"),
        "T3_max_abs_diff": g("T3", "max_abs_diff"),
        "T3_max_ulp": g("T3", "max_ulp"),
        "T4_max_abs_diff": g("T4", "max_abs_diff"),
        "T4_max_rel_diff": g("T4", "max_rel_diff"),
        "T4_indep_max_rel_diff": g("T4", "indep_max_rel_diff"),
        "T5_max_abs_diff": g("T5", "max_abs_diff"),
        "T5_max_rel_diff": g("T5", "max_rel_diff"),
        "T5_max_rel_diff_common": g("T5", "max_rel_diff_common"),
        "T5_noise_floor_rel": g("T5", "noise_floor_rel"),
        "T6_rate2_last": g("T6", "rate2_last"),
        **vol,
        "hlo_fwd_bytes": g("T7", "hlo_fwd_bytes"),
        "hlo_grad_bytes": g("T7", "hlo_grad_bytes"),
        "hlo_bwd_bytes": g("T7", "hlo_bwd_bytes"),
        "collectives_fwd": g("T7", "hlo_fwd_n"),
        "collectives_bwd": g("T7", "hlo_bwd_n"),
        "collectives_grad": g("T7", "hlo_grad_n"),
        "repeats": g("T8", "repeats"),
        "fwd_ms_per_step": g("T8", "fwd_per_step_ms"),
        "grad_ms_per_step": g("T8", "grad_per_step_ms"),
        "grad_over_fwd": g("T8", "grad_over_fwd"),
    }
    row["hlo_over_table"] = (
        None if row["hlo_fwd_bytes"] is None else row["hlo_fwd_bytes"] / row["table_bytes"]
    )
    return row


def _environment(distributed: bool):
    d0 = jax.devices()[0]
    return {
        "jax": jax.__version__,
        "python": platform.python_version(),
        "platform": d0.platform,
        "device_kind": d0.device_kind,
        "n_devices": jax.device_count(),
        "n_local_devices": jax.local_device_count(),
        "n_processes": jax.process_count(),
        "process_index": jax.process_index(),
        "distributed": distributed,
        "shard_map_api": shard_map_api(),
        "gt4py_tracer_dispatch": GT4PY_TRACER_DISPATCH,
        "host": platform.node(),
        "machine": platform.machine(),
    }


def _verdict(row):
    bad = [t for t in CORRECTNESS if not row[f"{t}_pass"]]
    if not bad:
        return "PASS"
    if any(row[f"{t}_status"].startswith("compile_error") for t in CORRECTNESS):
        return "compile_error"
    return "FAIL " + ",".join(bad)


def run(transports, layout_specs, size, n_steps, out_path, distributed, repeats):
    if transports == ["all"]:
        transports = available_transports()
    env = _environment(distributed)
    write = env["process_index"] == 0
    rows = []
    for spec in layout_specs:
        layout = B.parse_layout(spec, size)
        if distributed and layout.P != jax.device_count():
            raise SystemExit(
                f"--distributed: layout {spec} uses {layout.P} of {jax.device_count()} devices; "
                "every device must take part"
            )
        if layout.P > jax.device_count():
            print(f"skipping {spec}: needs {layout.P} devices, have {jax.device_count()}")
            continue
        for name in transports:
            res = B.run_battery(name, layout, n_steps, repeats)
            row = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "transport": name,
                "layout": spec,
                "Rx": layout.Rx,
                "Ry": layout.Ry,
                "P": layout.P,
                "M": layout.M,
                "N": layout.N,
                "MLOC": layout.MLOC,
                "NLOC": layout.NLOC,
                **env,
                **_scalars(res, n_steps, B.volume(get_transport(name), layout)),
            }
            rows.append(row)
            print(
                f"{name:<12} {spec:<5} {_verdict(row):<16} "
                f"table_bytes {row['table_bytes']}  hlo_fwd_bytes {row['hlo_fwd_bytes']}  "
                f"fwd {row['fwd_ms_per_step']} ms  grad {row['grad_ms_per_step']} ms",
                flush=True,
            )
            if write:
                with open(out_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
    return rows


# --- reporting -------------------------------------------------------------------------------
COLUMNS = (
    "env",
    "transport",
    "layout",
    "P",
    "MLOC x NLOC",
    "T0-T6",
    "table_bytes",
    "hlo_fwd_bytes",
    "true_halo_bytes",
    "K/field",
    "fwd ms",
    "grad ms",
    "grad/fwd",
)


def _cell(r, key, spec="{:.4f}"):
    return "-" if r.get(key) is None else spec.format(r[key])


def table(path):
    """Markdown summary of a results file, one row per (transport, layout)."""
    rows = [json.loads(line) for line in open(path) if line.strip()]
    out = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in rows:
        k = r.get("collectives_fwd")
        env = (
            f"jax {r.get('jax', '?')} {r.get('device_kind', '?')}"
            f" x{r.get('n_devices', '?')}/{r.get('n_processes', '?')}p"
        )
        out.append(
            f"| {env} | `{r['transport']}` | {r['layout']} | {r['P']} | {r['MLOC']}x{r['NLOC']} | "
            f"{_verdict(r)} | {_cell(r, 'table_bytes', '{:d}')} | {_cell(r, 'hlo_fwd_bytes', '{:d}')} | "
            f"{_cell(r, 'true_halo_bytes', '{:d}')} | {'-' if k is None else k // 3} | "
            f"{_cell(r, 'fwd_ms_per_step')} | {_cell(r, 'grad_ms_per_step')} | "
            f"{_cell(r, 'grad_over_fwd', '{:.2f}')} |"
        )
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--transports", default="all", help="'all' or a comma-separated list")
    ap.add_argument("--layouts", default="1x1,2x1,1x2,2x2,4x2,2x4")
    ap.add_argument("--size", type=int, default=B.M, help="global grid edge, M = N = size")
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="timed repeats per measurement; the minimum is reported",
    )
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--distributed", action="store_true", help="jax.distributed.initialize()")
    ap.add_argument("--table", metavar="RESULTS.JSONL", help="print the markdown table and exit")
    a = ap.parse_args(argv)
    if a.table:
        print(table(a.table))
        return 0
    distributed = _init_distributed(a.distributed)
    rows = run(
        [t.strip() for t in a.transports.split(",")],
        [s.strip() for s in a.layouts.split(",")],
        a.size,
        a.steps,
        a.out,
        distributed,
        a.repeats,
    )
    if rows and rows[0]["process_index"] == 0:
        print(f"\n{len(rows)} rows -> {a.out}")
    return 0 if rows and all(r["all_pass"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
