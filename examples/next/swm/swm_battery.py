# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test battery for one halo transport on one layout of the sharded SWM.

    cd <gt4py>
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        python examples/next/swm/swm_battery.py --transport allgather --layout 2x2

T0-T6 are pass/fail correctness tests, T7 and T8 report collectives and timings.
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

import halo_lib
from halo_transports import (
    Layout,
    available_transports,
    block,
    block_halo,
    get_transport,
    true_halo_cells,
    unblock,
    unblock_halo,
)
from hlo_accounting import bytes_moved, collectives_in
from initial_conditions import initialize_interior
from jax_compat import shard_map_api
from swm_sharded import (
    GT4PY_TRACER_DISPATCH,
    M,
    N,
    N_STEPS,
    cost,
    dx,
    dy,
    exchange_program,
    get_global,
    put_global,
    radius,
    reference_program,
    roll_reference_program,
    sharded_forward,
    sharded_program,
    tables,
    wrap_reference_program,
)


def _initial_state(layout: Layout):
    """``(host fields, sharded rank-major blocks)`` of the nb01 initial condition."""
    fields = initialize_interior(np, M, N, dx, dy, radius)
    return fields, tuple(put_global(block(a, layout), layout) for a in fields)


def _cost_of(program):
    return lambda *fields: cost(program(*fields))


def _sharded_cost(transport, layout: Layout, n_steps: int):
    return _cost_of(sharded_program(transport, layout, n_steps))


def _random_halo_stack(layout: Layout, rng):
    return put_global(
        rng.standard_normal((layout.P * layout.local_shape[0], layout.local_shape[1])), layout
    )


def oracle_gate(transport, layout: Layout):
    """FESOM's gate: any transport must match ``allgather`` exactly on the forward
    and to 1e-12 relative on the adjoint of ``sum(w * exchange(x))``."""
    ref = get_transport("allgather")
    fn = exchange_program(transport, layout)
    rf = exchange_program(ref, layout)
    rng = np.random.default_rng(7)
    x = _random_halo_stack(layout, rng)
    w = _random_halo_stack(layout, rng)
    ya, yb = get_global(fn(x)), get_global(rf(x))
    ga = get_global(jax.grad(lambda z: jnp.sum(w * fn(z)))(x))
    gb = get_global(jax.grad(lambda z: jnp.sum(w * rf(z)))(x))
    fwd_eq = bool(np.array_equal(ya, yb))
    grel = float(np.max(np.abs(ga - gb))) / float(np.max(np.abs(gb)))
    return {
        "oracle": ref.name,
        "self_comparison": transport.name == ref.name,
        "fwd_array_equal": fwd_eq,
        "fwd_max_abs_diff": float(np.max(np.abs(ya - yb))),
        "grad_max_rel_diff": grel,
        "pass": fwd_eq and grel < 1e-12,
    }


def t1_exchange_forward(transport, layout: Layout):
    L = layout
    g = np.random.default_rng(0).standard_normal((L.M, L.N))
    # halos start as garbage so that anything left unrefreshed shows up
    blocks = np.stack([np.pad(g[L.block_slice(r)], L.h, constant_values=-99.0) for r in range(L.P)])
    out = get_global(exchange_program(transport, L)(put_global(block_halo(blocks, L), L)))
    out = unblock_halo(out, L)
    padded = np.pad(g, L.h, mode="wrap")
    exp = np.stack(
        [
            padded[
                L.rx[r] * L.MLOC : L.rx[r] * L.MLOC + L.MLOC + 2 * L.h,
                L.ry[r] * L.NLOC : L.ry[r] * L.NLOC + L.NLOC + 2 * L.h,
            ]
            for r in range(L.P)
        ]
    )
    d = float(np.max(np.abs(out - exp)))
    return {"max_abs_diff": d, "pass": d == 0.0}


def t2_exchange_dotproduct(transport, layout: Layout):
    L = layout
    fn = exchange_program(transport, L)
    rng = np.random.default_rng(1)
    x = _random_halo_stack(L, rng)
    y = _random_halo_stack(L, rng)
    val, vjp = jax.vjp(fn, x)
    (xbar,) = vjp(y)
    lhs = float(jnp.sum(val * y))
    rhs = float(jnp.sum(x * xbar))
    rel = abs(lhs - rhs) / abs(lhs)
    res = {"lhs": lhs, "rhs": rhs, "rel_diff": rel, "pass": rel <= 1e-14}

    # halo_lib's dense transpose is an independent oracle only if its rank ordering agrees
    d = halo_lib.Decomposition(L.M, L.N, L.Rx, L.Ry, L.h)
    same_order = all(d.coords(r) == L.coords(r) for r in range(L.P)) and all(
        int(d.rank(a, b)) == int(L.rank(a, b))
        for a in range(-1, L.Rx + 1)
        for b in range(-1, L.Ry + 1)
    )
    res["halo_lib_order_matches"] = same_order
    if same_order:
        A = halo_lib.exchange_matrix(d, d.single_phase_pattern())
        fwd_dense = (A @ get_global(x).ravel()).reshape(x.shape)
        adj_dense = (A.T @ get_global(y).ravel()).reshape(x.shape)
        res["dense_fwd_diff"] = float(np.max(np.abs(get_global(val) - fwd_dense)))
        res["dense_vjp_diff"] = float(np.max(np.abs(get_global(xbar) - adj_dense)))
        res["pass"] = res["pass"] and res["dense_vjp_diff"] <= 1e-12
    else:
        res["dense_fwd_diff"] = float("nan")
        res["dense_vjp_diff"] = float("nan")
    return res


def t3_bit_identity_p1(transport, n_steps=N_STEPS):
    if jax.process_count() > 1:
        return {
            "status": "skipped (multi-process): a P=1 mesh does not span the processes",
            "pass": True,
        }
    return t4_forward_p(transport, Layout(M, N, 1, 1), n_steps, exact=True)


def t4_forward_p(transport, layout: Layout, n_steps=N_STEPS, exact=False):
    fields, args = _initial_state(layout)
    got = sharded_forward(transport, layout, *args, n_steps=n_steps)
    ref = reference_program(n_steps)(*fields)
    ind = roll_reference_program(n_steps)(*fields)
    out = {}
    ulps, ind_rels = [], []
    for name, g, r, q in zip("uvp", got, ref, ind):
        gg, rr, qq = unblock(get_global(g), layout), np.asarray(r), np.asarray(q)
        out[f"max_abs_diff_{name}"] = float(np.max(np.abs(gg - rr)))
        out[f"max_rel_diff_{name}"] = out[f"max_abs_diff_{name}"] / float(np.max(np.abs(rr)))
        ulps.append(float(np.max(np.abs(gg - rr) / np.spacing(np.abs(rr)))))
        ind_rels.append(float(np.max(np.abs(gg - qq))) / float(np.max(np.abs(qq))))
    out["max_abs_diff"] = max(out[f"max_abs_diff_{k}"] for k in "uvp")
    out["max_rel_diff"] = max(out[f"max_rel_diff_{k}"] for k in "uvp")
    out["max_ulp"] = max(ulps)
    out["indep_max_rel_diff"] = max(ind_rels)
    out["pass"] = out["max_abs_diff"] == 0.0 if exact else out["max_rel_diff"] <= 1e-12
    return out


def t5_gradient(transport, layout: Layout, n_steps=N_STEPS):
    fields, args = _initial_state(layout)
    grad_sharded = jax.grad(_sharded_cost(transport, layout, n_steps), argnums=(0, 1, 2))(*args)
    grad_ref = jax.grad(_cost_of(reference_program(n_steps)), argnums=(0, 1, 2))(*fields)
    grad_wrap = jax.grad(_cost_of(wrap_reference_program(n_steps)), argnums=(0, 1, 2))(*fields)
    out = {}
    worst_abs = worst_rel = floor_abs = floor_rel = 0.0
    for name, s, r, w in zip("uvp", grad_sharded, grad_ref, grad_wrap):
        sharded, ref, wrap = unblock(get_global(s), layout), np.asarray(r), np.asarray(w)
        scale = float(np.max(np.abs(ref)))
        diff = float(np.max(np.abs(sharded - ref)))
        floor = float(np.max(np.abs(wrap - ref)))
        out[f"max_abs_diff_{name}"] = diff
        out[f"max_rel_diff_{name}"] = diff / scale
        worst_abs, worst_rel = max(worst_abs, diff), max(worst_rel, diff / scale)
        floor_abs, floor_rel = max(floor_abs, floor), max(floor_rel, floor / scale)
    out["max_abs_diff"] = worst_abs
    out["max_rel_diff"] = worst_rel
    out["noise_floor_abs"] = floor_abs
    out["noise_floor_rel"] = floor_rel
    out["pass"] = worst_rel <= 1e-10
    return out


def t6_taylor(transport, layout: Layout, n_steps=N_STEPS):
    h0, n_halvings = 1e-2, 6
    fields, args = _initial_state(layout)
    rng = np.random.default_rng(3)
    dirs = tuple(
        put_global(rng.standard_normal(x.shape) * float(np.sqrt(np.mean(a**2))), layout)
        for a, x in zip(fields, args)
    )
    f = jax.jit(_sharded_cost(transport, layout, n_steps))
    J0 = float(f(*args))
    g = jax.grad(_sharded_cost(transport, layout, n_steps), argnums=(0, 1, 2))(*args)
    slope = float(sum(jnp.sum(gi * di) for gi, di in zip(g, dirs)))
    hs, errs = [], []
    for k in range(n_halvings + 1):
        h = h0 * 0.5**k
        Jh = float(f(*(a + h * d for a, d in zip(args, dirs))))
        hs.append(h)
        errs.append(abs(Jh - J0 - h * slope))
    rates = [
        float(np.log2(errs[k] / errs[k + 1])) if errs[k + 1] > 0 else float("inf")
        for k in range(n_halvings)
    ]
    return {
        "J0": J0,
        "slope": slope,
        "h": hs,
        "err2": errs,
        "rate2": rates,
        "rate2_last": rates[-1],
        "pass": abs(rates[-1] - 2.0) <= 0.1,
    }


def volume(transport, layout: Layout):
    """Bytes per device per step for the three fields: what the tables put on the wire
    and the true halo rim."""
    cells = transport.wire_cells(tables(transport, layout))
    table_bytes = 3 * cells * 8
    true_bytes = 3 * true_halo_cells(layout) * 8
    return {
        "wire_cells": cells,
        "table_bytes": table_bytes,
        "true_halo_bytes": true_bytes,
        "table_over_true": table_bytes / true_bytes,
    }


def t7_hlo(transport, layout: Layout):
    _, args = _initial_state(layout)
    costf = _sharded_cost(transport, layout, 1)

    def collectives(fn):
        return collectives_in(jax.jit(fn).lower(*args).compile().as_text())

    fwd = collectives(sharded_program(transport, layout, 1))
    cost_cs = collectives(costf)
    grad_cs = collectives(jax.value_and_grad(costf, argnums=(0, 1, 2)))
    bwd = list((Counter(grad_cs) - Counter(cost_cs)).elements())
    n_a2a = sum(c.startswith("all-to-all") for c in grad_cs)
    return {
        "n_steps": 1,
        "messages_note": (
            f"{n_a2a} all-to-all x (P-1) = {n_a2a * (layout.P - 1)} messages/rank"
            if n_a2a
            else "no all-to-all; one collective == one message here"
        ),
        **volume(transport, layout),
        "hlo_fwd": fwd,
        "hlo_fwd_n": len(fwd),
        "hlo_fwd_bytes": bytes_moved(fwd),
        "hlo_cost": cost_cs,
        "hlo_cost_n": len(cost_cs),
        "hlo_cost_bytes": bytes_moved(cost_cs),
        "hlo_grad": grad_cs,
        "hlo_grad_n": len(grad_cs),
        "hlo_grad_bytes": bytes_moved(grad_cs),
        "hlo_bwd": bwd,
        "hlo_bwd_n": len(bwd),
        "hlo_bwd_bytes": bytes_moved(bwd),
        "pass": True,
    }


def _bench(fn, args, repeats):
    jax.block_until_ready(fn(*args))
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - t0)
    return best


def t8_timing(transport, layout: Layout, n_steps=N_STEPS, repeats=5):
    _, args = _initial_state(layout)
    prog = sharded_program(transport, layout, n_steps)
    gradf = jax.jit(
        jax.value_and_grad(_sharded_cost(transport, layout, n_steps), argnums=(0, 1, 2))
    )
    tf = _bench(prog, args, repeats)
    tg = _bench(gradf, args, repeats)
    return {
        "n_steps": n_steps,
        "repeats": repeats,
        "fwd_total_s": tf,
        "fwd_per_step_ms": 1e3 * tf / n_steps,
        "grad_total_s": tg,
        "grad_per_step_ms": 1e3 * tg / n_steps,
        "grad_over_fwd": tg / tf,
        "pass": True,
    }


TESTS = ("T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
REPORTS = ("T7", "T8")

LABELS = {
    "T0": "oracle_gate        (vs allgather: bit-equal forward, 1e-12 adjoint)",
    "T1": "exchange_forward   (exchange == wrap-pad, per rank)",
    "T2": "exchange_dotproduct(<Lx,y> == <x,L^T y>, + halo_lib dense L^T)",
    "T3": "bit_identity_P1    (sharded P=1 == single-device reference)",
    "T4": "forward_P          (sharded vs reference)",
    "T5": "gradient           (sharded grad vs reference grad)",
    "T6": "taylor             (2nd-order remainder rate -> 2)",
    "T7": "hlo                (collectives, 1 step)",
    "T8": "timing             (min of --repeats)",
}


def _guarded(fn, *a, **kw):
    """Run one test; a transport that will not compile reports instead of aborting."""
    try:
        r = fn(*a, **kw)
        r.setdefault("status", "ok")
        return r
    except Exception as e:  # noqa: BLE001 - any failure must leave a printable row
        first = str(e).strip().splitlines()
        first = first[0] if first else ""
        return {"status": f"compile_error: {type(e).__name__}: {first[:200]}", "pass": False}


def run_battery(transport_name: str, layout: Layout, n_steps=N_STEPS, repeats=5):
    tr = get_transport(transport_name)
    return {
        "T0": _guarded(oracle_gate, tr, layout),
        "T1": _guarded(t1_exchange_forward, tr, layout),
        "T2": _guarded(t2_exchange_dotproduct, tr, layout),
        "T3": _guarded(t3_bit_identity_p1, tr, n_steps),
        "T4": _guarded(t4_forward_p, tr, layout, n_steps),
        "T5": _guarded(t5_gradient, tr, layout, n_steps),
        "T6": _guarded(t6_taylor, tr, layout, n_steps),
        "T7": _guarded(t7_hlo, tr, layout),
        "T8": _guarded(t8_timing, tr, layout, n_steps, repeats),
    }


def _fmt_value(v):
    if isinstance(v, float):
        return f"{v:.6e}" if (v != 0.0 and (abs(v) < 1e-3 or abs(v) >= 1e4)) else f"{v:.6f}"
    if isinstance(v, list) and v and isinstance(v[0], float):
        return "[" + ", ".join(f"{x:.3g}" for x in v) + "]"
    return str(v)


def print_battery(name, layout: Layout, n_steps, res):
    print("=" * 78)
    print(f"transport {name!r}   {layout}   n_steps={n_steps}")
    print(
        f"jax {jax.__version__}   platform {jax.devices()[0].platform}   "
        f"devices {jax.device_count()}   x64 {jax.config.jax_enable_x64}"
    )
    print(f"shard_map via {shard_map_api()}   gt4py tracer dispatch: {GT4PY_TRACER_DISPATCH}")
    print("=" * 78)
    for t in TESTS:
        r = res[t]
        verdict = "PASS" if r.get("pass") else "FAIL"
        info = "" if (t in REPORTS and r.get("status") == "ok") else f"  [{verdict}]"
        print(f"\n{t} {LABELS[t]}{info}")
        for k, v in r.items():
            if k == "pass" or (k == "status" and v == "ok"):
                continue
            print(f"      {k:<22s} {_fmt_value(v)}")
    bad = [t for t in TESTS if not res[t].get("pass")]
    print("\n" + "-" * 78)
    print(f"summary: {'ALL PASS' if not bad else 'FAILED: ' + ','.join(bad)}")
    print("-" * 78)


def parse_layout(spec: str) -> Layout:
    """``"RxxRy"`` -> ``Layout(M, N, Rx, Ry)``."""
    rx, ry = (int(v) for v in spec.lower().split("x"))
    return Layout(M, N, rx, ry)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--transport",
        default="allgather",
        choices=available_transports(),
        help="registered transport name",
    )
    ap.add_argument("--layout", default="2x2", help="RxxRy, Rx*Ry <= device count")
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--repeats", type=int, default=5, help="timed repeats; min is reported")
    a = ap.parse_args(argv)
    try:
        layout = parse_layout(a.layout)
    except ValueError as e:
        raise SystemExit(e) from None
    if layout.P > jax.device_count():
        raise SystemExit(f"layout {a.layout} needs {layout.P} devices, have {jax.device_count()}")
    res = run_battery(a.transport, layout, a.steps, a.repeats)
    print_battery(a.transport, layout, a.steps, res)
    return 0 if all(res[t].get("pass") for t in TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
