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

Rows are appended, so laptop and cluster results can be concatenated and compared. Every
row carries the environment it was measured in (jax version, platform, device kind, device
and process counts) as well as the numbers.

Multi-process runs (`--distributed`) call ``jax.distributed.initialize()`` before anything
else touches a device; JAX auto-detects Slurm, every layout must use *all* devices, and
only process 0 writes rows. Inputs are built with ``jax.make_array_from_process_local_data``
and read back with ``multihost_utils.process_allgather``; T3 (a P=1 mesh) is skipped when
``jax.process_count() > 1`` because such a mesh does not span the processes. Tested with
two real processes over gloo on CPU (1 and 2 devices each, layouts 2x1/4x1/2x2/1x4);
**untested on more than one node, on NCCL and on GPU**. See ``santis_bench.sbatch``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _init_distributed(enabled: bool) -> bool:
    """``jax.distributed.initialize()`` under Slurm, before any device is created."""
    if not enabled:
        return False
    import jax

    if int(os.environ.get("SLURM_NTASKS", "1")) <= 1:
        print("--distributed: SLURM_NTASKS <= 1, staying single-process", file=sys.stderr)
        return False
    jax.distributed.initialize()  # picks up SLURM_* itself
    return True


def _scalars(res, layout, n_steps, cells, true_bytes):
    """Flatten the battery result dict to one flat row of scalars.

    ``cells``/``true_bytes`` come from the prepared tables, not from T7, so a transport
    that does not compile still reports the volume its tables imply.
    """
    t = {k: res[k] for k in ("T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")}
    g = lambda k, f, d=None: t[k].get(f, d)  # noqa: E731
    row = {
        "n_steps": n_steps,
        "all_pass": all(t[k].get("pass") for k in t),
        **{f"{k}_pass": bool(t[k].get("pass")) for k in t},
        **{f"{k}_status": t[k].get("status", "ok") for k in t},
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
        "T5_noise_floor_rel": g("T5", "noise_floor_rel"),
        "T6_rate2_last": g("T6", "rate2_last"),
        "wire_cells": cells,
        "table_bytes": None if cells is None else 3 * cells * 8,
        "true_halo_bytes": true_bytes,
        "table_over_true": None if cells is None else 3 * cells * 8 / true_bytes,
        "hlo_fwd_bytes": g("T7", "fwd_fields_bytes"),
        "hlo_grad_bytes": g("T7", "grad_bytes"),
        "hlo_bwd_bytes": g("T7", "bwd_only_bytes"),
        "collectives_fwd": g("T7", "fwd_fields_n"),
        "collectives_bwd": g("T7", "bwd_only_n"),
        "collectives_grad": g("T7", "grad_n"),
        "repeats": g("T8", "repeats"),
        "fwd_ms_per_step": g("T8", "fwd_per_step_ms"),
        "grad_ms_per_step": g("T8", "grad_per_step_ms"),
        "grad_over_fwd": g("T8", "grad_over_fwd"),
    }
    row["hlo_over_table"] = (
        None
        if not row["table_bytes"] or row["hlo_fwd_bytes"] is None
        else row["hlo_fwd_bytes"] / row["table_bytes"]
    )
    return row


def _environment(distributed: bool):
    import jax

    import halo_transports as ht

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
        "shard_map_api": ht.shard_map_api(),
        "gt4py_tracer_dispatch": ht.GT4PY_TRACER_DISPATCH,
        "host": platform.node(),
        "machine": platform.machine(),
    }


def run(transports, layout_specs, n_steps, out_path, distributed, repeats=5):
    import jax

    import swm_sharded as S
    from halo_transports import (
        Layout,
        available_transports,
        get_transport,
        true_halo_cells,
        wire_cells,
    )

    if transports == ["all"]:
        transports = available_transports()
    env = _environment(distributed)
    write = env["process_index"] == 0
    rows = []
    for spec in layout_specs:
        rx, ry = (int(v) for v in spec.lower().split("x"))
        if distributed and rx * ry != jax.device_count():
            raise SystemExit(
                f"--distributed: layout {spec} uses {rx * ry} of {jax.device_count()} devices; "
                "every device must take part"
            )
        if rx * ry > jax.device_count():
            print(f"skipping {spec}: needs {rx * ry} devices, have {jax.device_count()}")
            continue
        layout = Layout(S.M, S.N, rx, ry)
        true_bytes = 3 * true_halo_cells(layout) * 8
        for name in transports:
            tr = get_transport(name)
            cells = wire_cells(tr, tr.prepare(layout))
            res = S.run_battery(name, layout, n_steps, repeats)
            row = {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "transport": name,
                "layout": f"{rx}x{ry}",
                "Rx": rx,
                "Ry": ry,
                "P": rx * ry,
                "M": S.M,
                "N": S.N,
                "MLOC": layout.MLOC,
                "NLOC": layout.NLOC,
                **env,
                **_scalars(res, layout, n_steps, cells, true_bytes),
            }
            rows.append(row)
            print(
                f"{name:<12} {spec:<5} "
                f"{'ALL PASS' if row['all_pass'] else 'FAILED: ' + ','.join(k for k in 'T0 T1 T2 T3 T4 T5 T6 T7 T8'.split() if not row[k + '_pass'])}"
                f"   table {row['table_bytes']} B  hlo {row['hlo_fwd_bytes']} B  "
                f"fwd {row['fwd_ms_per_step']} ms  grad {row['grad_ms_per_step']} ms",
                flush=True,
            )
            if write:
                with open(out_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
    return rows


# --- reporting -------------------------------------------------------------------------------
TESTS = ("T0", "T1", "T2", "T3", "T4", "T5", "T6")


def _fmt(r, key, spec="{:.4f}"):
    return "-" if r.get(key) is None else spec.format(r[key])


def table(path):
    """Markdown summary of a results file, one row per (transport, layout)."""
    rows = [json.loads(line) for line in open(path) if line.strip()]
    hdr = ("| env | transport | layout | P | MLOC x NLOC | T0-T6 | table B | HLO B |"
           " true B | K/field | fwd ms | grad ms | grad/fwd |")
    out = [hdr, "|" + "---|" * 13]
    for r in rows:
        bad = [t for t in TESTS if not r.get(f"{t}_pass")]
        st = r.get("T1_status", "ok")
        verdict = "PASS" if not bad else (
            "compile_error" if st.startswith("compile_error") else "FAIL " + ",".join(bad)
        )
        k = r.get("collectives_fwd")
        env = (f"jax {r.get('jax', '?')} {r.get('device_kind', '?')}"
               f" x{r.get('n_devices', '?')}/{r.get('n_processes', '?')}p")
        out.append(
            f"| {env} | `{r['transport']}` | {r['layout']} | {r['P']} | {r['MLOC']}x{r['NLOC']} | "
            f"{verdict} | {_fmt(r, 'table_bytes', '{:d}')} | {_fmt(r, 'hlo_fwd_bytes', '{:d}')} | "
            f"{_fmt(r, 'true_halo_bytes', '{:d}')} | {'-' if k is None else k // 3} | "
            f"{_fmt(r, 'fwd_ms_per_step')} | {_fmt(r, 'grad_ms_per_step')} | "
            f"{_fmt(r, 'grad_over_fwd', '{:.2f}')} |"
        )
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--transports", default="all", help="'all' or a comma-separated list")
    ap.add_argument("--layouts", default="1x1,2x1,1x2,2x2,4x2,2x4")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=5,
                    help="timed repeats per measurement; the minimum is reported")
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
        a.steps,
        a.out,
        distributed,
        a.repeats,
    )
    if rows and rows[0]["process_index"] == 0:
        print(f"\n{len(rows)} rows -> {a.out}")
    return 0 if all(r["all_pass"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
