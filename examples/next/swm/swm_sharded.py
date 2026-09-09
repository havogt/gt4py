# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Sharded, differentiable shallow water model and the transport comparison battery.

    cd <gt4py>
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        python examples/next/swm/swm_sharded.py --transport allgather --layout 2x2

One jitted program per (transport, layout, n_steps): a ``shard_map`` over the 1-D
device axis ``"d"`` whose body zero-pads this device's interior blocks to the halo
shape, refreshes the halos with the transport, and runs the nb01 leapfrog loop as a
``jax.lax.scan`` of GT4Py ``operators.timestep`` on JAX-backed local fields.

``timestep`` calls ``make_periodic`` with the *local* sizes, so for P > 1 the halos it
writes are wrong -- and never read: the next iteration's exchange overwrites them
before anything touches them, and the ``old`` fields' halos never reach an interior
cell (``uold_new = u + alpha*(unew - 2u + uold)`` is evaluated on the pre-periodic
``unew``, whose domain is the interior). Same argument as ``swm_ghex_2d.py``.
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import halo_lib
from halo_transports import (  # noqa: F401  (re-exported for the transport modules)
    GT4PY_TRACER_DISPATCH,
    Layout,
    available_transports,
    block,
    block_halo,
    bytes_moved,
    collectives_in,
    get_transport,
    shard_map,
    shard_map_api,
    true_halo_cells,
    unblock,
    unblock_halo,
    wire_cells,
)

# imported after halo_transports: it registers gt4py's field constructor for jax tracers
from gt4py import next as gtx
from initial_conditions import initialize_interior
from operators import I, J, make_periodic
from operators import timestep as gtx_timestep

timestep = gtx_timestep.definition

M = N = 16
dx = dy = 100000.0
dt, radius, alpha = 90.0, 1000000.0, 0.001
N_STEPS = 10

AXIS = "d"


# --- the model ---------------------------------------------------------------------------------
def _halo_domain(mloc, nloc):
    return gtx.domain({I: (-1, mloc + 1), J: (-1, nloc + 1)})


def _gt_step(arrays, dt_, alpha_, mloc, nloc):
    """One ``operators.timestep`` on raw ``(mloc+2, nloc+2)`` jnp arrays."""
    dom = _halo_domain(mloc, nloc)
    u, v, p, uo, vo, po = (gtx.as_field(dom, a, allocator=jnp) for a in arrays)
    out = timestep(u, v, p, dx, dy, dt_, uo, vo, po, alpha_, mloc, nloc)
    return tuple(o.ndarray for o in out)


def make_mesh(p: int) -> Mesh:
    """1-D mesh over the first ``p`` *global* devices; identical on every process."""
    devices = jax.devices()
    if len(devices) < p:
        raise ValueError(
            f"a {p}-device mesh was requested but jax.devices() has {len(devices)}: "
            "raise the device count (XLA_FLAGS=--xla_force_host_platform_device_count=N "
            "on CPU) or use a smaller layout"
        )
    if jax.process_count() > 1 and p != len(devices):
        raise ValueError(
            f"multi-process run: a {p}-device mesh does not span all {len(devices)} "
            "devices, so it does not span all processes -- the collectives would hang. "
            "Every layout must satisfy Rx*Ry == jax.device_count()."
        )
    return Mesh(np.array(devices[:p]), (AXIS,))


def _sharding(layout: Layout) -> NamedSharding:
    return NamedSharding(make_mesh(layout.P), P(AXIS))


def put_global(a_host, layout: Layout):
    """Rank-major host array ``(P*rows, cols)`` -> global ``jax.Array`` sharded over ``"d"``.

    Every process computes the whole (tiny) array redundantly and contributes only the
    blocks its own devices own. Single-process behaviour is a plain ``device_put`` and is
    bit-identical to what the battery did before.
    """
    a_host = np.asarray(a_host)
    sh = _sharding(layout)
    if jax.process_count() == 1:
        return jax.device_put(jnp.asarray(a_host), sh)
    rows = a_host.shape[0] // layout.P
    me = jax.process_index()
    mine = [k for k, d in enumerate(make_mesh(layout.P).devices.flat) if d.process_index == me]
    local = np.concatenate([a_host[k * rows : (k + 1) * rows] for k in mine])
    return jax.make_array_from_process_local_data(sh, local)


def get_global(x):
    """Global ``jax.Array`` -> numpy on every process.

    A multi-process array is not fully addressable, so ``np.asarray`` on it raises; the
    battery's comparisons need the whole thing on every process.
    """
    if jax.process_count() == 1:
        return np.asarray(x)
    from jax.experimental import multihost_utils

    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


def _sharded_program(transport, layout: Layout, n_steps: int):
    L = layout
    tables = transport.prepare(L)
    mloc, nloc = L.MLOC, L.NLOC

    def exchange(a):
        return transport.exchange(a, tables, AXIS)

    def body(u0, v0, p0):
        u, v, p = (exchange(jnp.pad(a, L.h)) for a in (u0, v0, p0))
        state = _gt_step((u, v, p, u, v, p), dt, 0.0, mloc, nloc)

        def scan_step(carry, _):
            u, v, p, uo, vo, po = carry
            u, v, p = (exchange(a) for a in (u, v, p))
            return _gt_step((u, v, p, uo, vo, po), 2.0 * dt, alpha, mloc, nloc), None

        final, _ = jax.lax.scan(scan_step, state, None, length=n_steps - 1)
        h = L.h
        return tuple(f[h:-h, h:-h] for f in final[:3])

    return jax.jit(
        shard_map(body, mesh=make_mesh(L.P), in_specs=(P(AXIS),) * 3, out_specs=(P(AXIS),) * 3)
    )


@functools.lru_cache(maxsize=None)
def _cached_program(name, key, n_steps):
    M_, N_, Rx, Ry, h = key
    return _sharded_program(get_transport(name), Layout(M_, N_, Rx, Ry, h), n_steps)


def _key(layout: Layout):
    return (layout.M, layout.N, layout.Rx, layout.Ry, layout.h)


def sharded_forward(transport, layout: Layout, u0, v0, p0, n_steps=N_STEPS):
    """u0, v0, p0: rank-major stacks of interior blocks, ``(P*MLOC, NLOC)``.

    Returns the same layout; use ``unblock`` for the global ``(M, N)`` fields.
    """
    return _cached_program(transport.name, _key(layout), n_steps)(u0, v0, p0)


def _reference_program(n_steps: int):
    dom_int = gtx.domain({I: (0, M), J: (0, N)})

    def run(u0, v0, p0):
        u, v, p = (
            make_periodic(gtx.as_field(dom_int, a, allocator=jnp), M, N).ndarray
            for a in (u0, v0, p0)
        )
        state = _gt_step((u, v, p, u, v, p), dt, 0.0, M, N)

        def scan_step(carry, _):
            u, v, p, uo, vo, po = carry
            return _gt_step((u, v, p, uo, vo, po), 2.0 * dt, alpha, M, N), None

        final, _ = jax.lax.scan(scan_step, state, None, length=n_steps - 1)
        return tuple(f[1:-1, 1:-1] for f in final[:3])

    return jax.jit(run)


@functools.lru_cache(maxsize=None)
def _cached_reference(n_steps):
    return _reference_program(n_steps)


def reference_forward(u0, v0, p0, n_steps=N_STEPS):
    """Single-device model on the global ``(M, N)`` grid, same scan structure."""
    return _cached_reference(n_steps)(u0, v0, p0)


# --- an independent single-device reference ------------------------------------------------
# Pure jnp.roll on the global torus, sharing no code with the GT4Py path -- deliberately
# associating the arithmetic differently (``x/dx`` rather than ``(1/dx)*x``, one fused
# 0.25*(...) rather than two nested 0.5*(...)), so that agreement is evidence about the
# model and not about a shared expression tree.
def _rp(a, k, axis):
    return jnp.roll(a, k, axis=axis)


def _roll_step(u, v, p, uo, vo, po, dt_, alpha_):
    cu = 0.5 * (_rp(p, -1, 0) + p) * u
    cv = 0.5 * (_rp(p, -1, 1) + p) * v
    denom = 0.25 * (_rp(_rp(p, -1, 0), -1, 1) + _rp(p, -1, 0) + _rp(p, -1, 1) + p)
    z = ((_rp(v, -1, 0) - v) / dx - (_rp(u, -1, 1) - u) / dy) / denom
    h = p + 0.5 * (0.5 * (_rp(u, 1, 0) ** 2 + u**2) + 0.5 * (_rp(v, 1, 1) ** 2 + v**2))
    acv = 0.5 * (_rp(cv, -1, 0) + cv)
    acu = 0.5 * (_rp(cu, -1, 1) + cu)
    unew = uo + 0.5 * (_rp(z, 1, 1) + z) * 0.5 * (_rp(acv, 1, 1) + acv) * dt_ - (
        _rp(h, -1, 0) - h
    ) / dx * dt_
    vnew = vo - 0.5 * (_rp(z, 1, 0) + z) * 0.5 * (_rp(acu, 1, 0) + acu) * dt_ - (
        _rp(h, -1, 1) - h
    ) / dy * dt_
    pnew = po - (cu - _rp(cu, 1, 0)) / dx * dt_ - (cv - _rp(cv, 1, 1)) / dy * dt_
    return (
        unew,
        vnew,
        pnew,
        u + alpha_ * (unew - 2.0 * u + uo),
        v + alpha_ * (vnew - 2.0 * v + vo),
        p + alpha_ * (pnew - 2.0 * p + po),
    )


@functools.lru_cache(maxsize=None)
def _roll_reference_program(n_steps: int):
    def run(u0, v0, p0):
        state = _roll_step(u0, v0, p0, u0, v0, p0, dt, 0.0)

        def scan_step(carry, _):
            return _roll_step(*carry, 2.0 * dt, alpha), None

        final, _ = jax.lax.scan(scan_step, state, None, length=n_steps - 1)
        return final[:3]

    return jax.jit(run)


def roll_reference_forward(u0, v0, p0, n_steps=N_STEPS):
    """Independent single-device model on the global ``(M, N)`` interior; no GT4Py."""
    return _roll_reference_program(n_steps)(u0, v0, p0)


def cost(fields):
    """Scalar objective: sum of squares of p over the global interior."""
    return jnp.sum(fields[2] * fields[2])


# --- test battery ------------------------------------------------------------------------------
def _initial_fields():
    return initialize_interior(np, M, N, dx, dy, radius)


def _exchange_program(transport, layout: Layout):
    tables = transport.prepare(layout)
    return jax.jit(
        shard_map(
            lambda a: transport.exchange(a, tables, AXIS),
            mesh=make_mesh(layout.P),
            in_specs=P(AXIS),
            out_specs=P(AXIS),
        )
    )


def t1_exchange_forward(transport, layout: Layout, seed=0):
    L = layout
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((L.M, L.N))
    # halos start as garbage so that anything left unrefreshed shows up
    blocks = np.stack(
        [np.pad(g[L.block_slice(r)], L.h, constant_values=-99.0) for r in range(L.P)]
    )
    out = get_global(_exchange_program(transport, L)(put_global(block_halo(blocks, L), L)))
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
    return {"max_abs_diff": d, "exact": d == 0.0, "pass": d == 0.0}


def t2_exchange_dotproduct(transport, layout: Layout, seed=1):
    L = layout
    fn = _exchange_program(transport, L)
    rng = np.random.default_rng(seed)
    shape = (L.P * L.local_shape[0], L.local_shape[1])
    x = put_global(rng.standard_normal(shape), L)
    y = put_global(rng.standard_normal(shape), L)
    val, vjp = jax.vjp(fn, x)
    (xbar,) = vjp(y)
    lhs = float(jnp.sum(val * y))
    rhs = float(jnp.sum(x * xbar))
    rel = abs(lhs - rhs) / abs(lhs)
    res = {"lhs": lhs, "rhs": rhs, "rel_diff": rel, "pass": rel <= 1e-14}

    # independent dense transpose from halo_lib (nb04). Only meaningful if the rank
    # ordering agrees; halo_lib uses coords = divmod(rank, Ry) and origin = (rx*mloc, ry*nloc).
    d = halo_lib.Decomposition(L.M, L.N, L.Rx, L.Ry, L.h)
    same_order = all(d.coords(r) == L.coords(r) for r in range(L.P)) and all(
        int(d.rank(a, b)) == int(L.rank(a, b))
        for a in range(-1, L.Rx + 1)
        for b in range(-1, L.Ry + 1)
    )
    res["halo_lib_order_matches"] = same_order
    if same_order:
        A = halo_lib.exchange_matrix(d, d.single_phase_pattern())
        fwd_dense = (A @ get_global(x).ravel()).reshape(shape)
        adj_dense = (A.T @ get_global(y).ravel()).reshape(shape)
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
            "skipped": True,
            "pass": True,
        }
    L = Layout(M, N, 1, 1)
    u0, v0, p0 = _initial_fields()
    args = tuple(put_global(block(a, L), L) for a in (u0, v0, p0))
    got = sharded_forward(transport, L, *args, n_steps=n_steps)
    ref = reference_forward(*(jnp.asarray(a) for a in (u0, v0, p0)), n_steps=n_steps)
    diffs, ulps = {}, {}
    for name, g, r in zip("uvp", got, ref):
        gg, rr = unblock(get_global(g), L), np.asarray(r)
        diffs[name] = float(np.max(np.abs(gg - rr)))
        ulps[name] = float(np.max(np.abs(gg - rr) / np.spacing(np.abs(rr))))
    worst, worst_ulp = max(diffs.values()), max(ulps.values())
    return {
        **{f"max_abs_diff_{k}": v for k, v in diffs.items()},
        "max_abs_diff": worst,
        "max_ulp": worst_ulp,
        "exact": worst == 0.0,
        "note": ("SPMD partitioning at P=1 must change nothing: both sides run the same "
                 "GT4Py step, so this is a partitioner invariant, not model reproducibility "
                 "(T4's indep_* fields are the independent statement). A non-zero max_ulp "
                 "of a few is "
                 "XLA instruction selection under SPMD partitioning, not the transport: "
                 "it reproduces with a shard_map body containing no collective at all. "
                 "Report it, do not chase it."),
        "pass": worst == 0.0,
    }


def t4_forward_p(transport, layout: Layout, n_steps=N_STEPS):
    u0, v0, p0 = _initial_fields()
    args = tuple(put_global(block(a, layout), layout) for a in (u0, v0, p0))
    got = sharded_forward(transport, layout, *args, n_steps=n_steps)
    ref = reference_forward(*(jnp.asarray(a) for a in (u0, v0, p0)), n_steps=n_steps)
    ind = roll_reference_forward(*(jnp.asarray(a) for a in (u0, v0, p0)), n_steps=n_steps)
    diffs = {}
    rels = {}
    ind_rels = {}
    for name, g, r, q in zip("uvp", got, ref, ind):
        gg, rr, qq = unblock(get_global(g), layout), np.asarray(r), np.asarray(q)
        diffs[name] = float(np.max(np.abs(gg - rr)))
        rels[name] = diffs[name] / float(np.max(np.abs(rr)))
        ind_rels[name] = float(np.max(np.abs(gg - qq))) / float(np.max(np.abs(qq)))
    worst = max(diffs.values())
    return {
        "indep_max_rel_diff": max(ind_rels.values()),
        "indep_note": ("vs roll_reference_forward, a pure-jnp single-device model that "
                       "shares no code with the GT4Py path -- the independent statement "
                       "T3 cannot make"),
        **{f"max_abs_diff_{k}": v for k, v in diffs.items()},
        **{f"max_rel_diff_{k}": v for k, v in rels.items()},
        "max_abs_diff": worst,
        "max_rel_diff": max(rels.values()),
        "exact": worst == 0.0,
        "pass": max(rels.values()) <= 1e-12,
    }


def _sharded_cost(transport, layout, n_steps):
    def f(ub, vb, pb):
        return cost(sharded_forward(transport, layout, ub, vb, pb, n_steps))

    return f


def _reference_cost(n_steps):
    def f(u, v, p):
        return cost(reference_forward(u, v, p, n_steps))

    return f


@functools.lru_cache(maxsize=None)
def _alt_reference_program(n_steps):
    """Single-device model whose halos are refreshed by an explicit wrap-gather.

    Forward bit-identical to ``reference_forward`` (both are copies of the same
    interior values), but the reverse mode accumulates in a different order. The
    spread between the two gradients is the noise floor any distributed gradient
    has to be judged against.
    """

    def run(u0, v0, p0):
        wrap = lambda a: jnp.pad(a, 1, mode="wrap")  # noqa: E731
        u, v, p = (wrap(a) for a in (u0, v0, p0))
        state = _gt_step((u, v, p, u, v, p), dt, 0.0, M, N)

        def scan_step(carry, _):
            u, v, p, uo, vo, po = carry
            u, v, p = (wrap(a[1:-1, 1:-1]) for a in (u, v, p))
            return _gt_step((u, v, p, uo, vo, po), 2.0 * dt, alpha, M, N), None

        final, _ = jax.lax.scan(scan_step, state, None, length=n_steps - 1)
        return tuple(f[1:-1, 1:-1] for f in final[:3])

    return jax.jit(run)


def _alt_reference_cost(n_steps):
    def f(u, v, p):
        return cost(_alt_reference_program(n_steps)(u, v, p))

    return f


def t5_gradient(transport, layout: Layout, n_steps=N_STEPS):
    u0, v0, p0 = _initial_fields()
    args = tuple(put_global(block(a, layout), layout) for a in (u0, v0, p0))
    gargs = tuple(jnp.asarray(a) for a in (u0, v0, p0))
    gs = jax.grad(_sharded_cost(transport, layout, n_steps), argnums=(0, 1, 2))(*args)
    gr = jax.grad(_reference_cost(n_steps), argnums=(0, 1, 2))(*gargs)
    ga = jax.grad(_alt_reference_cost(n_steps), argnums=(0, 1, 2))(*gargs)
    out = {}
    worst_abs = worst_rel = floor_abs = floor_rel = 0.0
    for name, a, b, c in zip("uvp", gs, gr, ga):
        aa, bb, cc = unblock(get_global(a), layout), np.asarray(b), np.asarray(c)
        scale = float(np.max(np.abs(bb)))
        d = float(np.max(np.abs(aa - bb)))
        f = float(np.max(np.abs(cc - bb)))
        out[f"max_abs_diff_{name}"] = d
        out[f"max_rel_diff_{name}"] = d / scale
        worst_abs, worst_rel = max(worst_abs, d), max(worst_rel, d / scale)
        floor_abs, floor_rel = max(floor_abs, f), max(floor_rel, f / scale)
    out["max_abs_diff"] = worst_abs
    out["max_rel_diff"] = worst_rel
    out["noise_floor_abs"] = floor_abs
    out["noise_floor_rel"] = floor_rel
    out["note"] = ("noise_floor_* is the same comparison between two single-device "
                   "implementations that are forward bit-identical; the gradient is "
                   "cancellation-limited, not transport-limited")
    out["pass"] = worst_rel <= 1e-10
    return out


def t6_taylor(transport, layout: Layout, n_steps=N_STEPS, h0=1e-2, n_halvings=6, seed=3):
    u0, v0, p0 = _initial_fields()
    blocks = tuple(block(a, layout) for a in (u0, v0, p0))
    args = tuple(put_global(a, layout) for a in blocks)
    rng = np.random.default_rng(seed)
    dirs = tuple(
        put_global(rng.standard_normal(a.shape) * float(np.sqrt(np.mean(a**2))), layout)
        for a in blocks
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


def _shard_inputs(layout: Layout, arrays):
    return tuple(put_global(a, layout) for a in arrays)


def t7_hlo(transport, layout: Layout, n_steps=1):
    u0, v0, p0 = _initial_fields()
    args = _shard_inputs(layout, [block(a, layout) for a in (u0, v0, p0)])
    prog = _cached_program(transport.name, _key(layout), n_steps)
    fwd = collectives_in(prog.lower(*args).compile().as_text())
    costf = jax.jit(_sharded_cost(transport, layout, n_steps))
    cost_cs = collectives_in(costf.lower(*args).compile().as_text())
    gradf = jax.jit(
        jax.value_and_grad(_sharded_cost(transport, layout, n_steps), argnums=(0, 1, 2))
    )
    grad_cs = collectives_in(gradf.lower(*args).compile().as_text())
    rest = list(grad_cs)
    missing = []
    for c in cost_cs:
        if c in rest:
            rest.remove(c)
        else:
            missing.append(c)
    bwd = rest
    a2a = [c for c in grad_cs if c.startswith("all-to-all")]
    msgs = (f"{len(a2a)} all-to-all x (P-1) = {len(a2a) * (layout.P - 1)} messages/rank"
            if a2a else "no all-to-all; one collective == one message here")
    cells = wire_cells(transport, transport.prepare(layout))
    true_bytes = 3 * true_halo_cells(layout) * 8
    table_bytes = None if cells is None else 3 * cells * 8
    return {
        "n_steps": n_steps,
        "messages_note": msgs,
        "wire_cells": cells,
        "table_bytes_per_step": table_bytes,
        "true_halo_bytes_per_step": true_bytes,
        "table_over_true": None if table_bytes is None else table_bytes / true_bytes,
        "fwd_fields": fwd,
        "fwd_fields_n": len(fwd),
        "fwd_fields_bytes": bytes_moved(fwd),
        "fwd_cost": cost_cs,
        "fwd_cost_n": len(cost_cs),
        "fwd_cost_bytes": bytes_moved(cost_cs),
        "grad": grad_cs,
        "grad_n": len(grad_cs),
        "grad_bytes": bytes_moved(grad_cs),
        "bwd_only": bwd,
        "bwd_only_n": len(bwd),
        "bwd_only_bytes": bytes_moved(bwd),
        "missing_in_grad": missing,
        "pass": True,
    }


def _bench(fn, args, repeats=5):
    out = fn(*args)
    jax.block_until_ready(out)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        best = min(best, time.perf_counter() - t0)
    return best


def t8_timing(transport, layout: Layout, n_steps=N_STEPS, repeats=5):
    u0, v0, p0 = _initial_fields()
    args = _shard_inputs(layout, [block(a, layout) for a in (u0, v0, p0)])
    prog = _cached_program(transport.name, _key(layout), n_steps)
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


def oracle_gate(transport, layout: Layout, P=None, seed=7):
    """FESOM's gate: any transport must match ``allgather`` exactly on the forward
    and to 1e-12 relative on the adjoint of ``sum(w * exchange(x))``."""
    L = layout
    if P is not None and int(P) != L.P:
        raise ValueError(f"oracle_gate: P={P} does not match {L}")
    ref = get_transport("allgather")
    fn = _exchange_program(transport, L)
    rf = _exchange_program(ref, L)
    rng = np.random.default_rng(seed)
    shape = (L.P * L.local_shape[0], L.local_shape[1])
    x = put_global(rng.standard_normal(shape), L)
    w = put_global(rng.standard_normal(shape), L)
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


TESTS = ("T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")


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
        "T7": _guarded(t7_hlo, tr, layout, 1),
        "T8": _guarded(t8_timing, tr, layout, n_steps, repeats),
    }


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6e}" if (v != 0.0 and (abs(v) < 1e-3 or abs(v) >= 1e4)) else f"{v:.6f}"
    if isinstance(v, list):
        if v and isinstance(v[0], float):
            return "[" + ", ".join(f"{x:.3g}" for x in v) + "]"
        return str(v)
    return str(v)


LABELS = {
    "T0": "oracle_gate        (vs allgather: bit-equal forward, 1e-12 adjoint)",
    "T1": "exchange_forward   (exchange == wrap-pad, per rank)",
    "T2": "exchange_dotproduct(<Lx,y> == <x,L^T y>, + halo_lib dense L^T)",
    "T3": "bit_identity_P1    (sharded P=1 == single-device reference)",
    "T4": "forward_P          (sharded vs reference)",
    "T5": "gradient           (sharded grad vs reference grad)",
    "T6": "taylor             (2nd-order remainder rate -> 2)",
    "T7": "hlo                (collectives, 1 step)",
    "T8": "timing             (CPU, 8 fake devices in one process)",
}


def print_battery(name, layout: Layout, n_steps, res):
    print("=" * 78)
    print(f"transport {name!r}   {layout}   n_steps={n_steps}")
    print(f"jax {jax.__version__}   platform {jax.devices()[0].platform}   "
          f"devices {jax.device_count()}   x64 {jax.config.jax_enable_x64}")
    print(f"shard_map via {shard_map_api()}   gt4py tracer dispatch: {GT4PY_TRACER_DISPATCH}")
    print("=" * 78)
    for t in TESTS:
        r = res[t]
        verdict = "PASS" if r.get("pass") else "FAIL"
        info = "" if (t in ("T7", "T8") and r.get("status") == "ok") else f"  [{verdict}]"
        print(f"\n{t} {LABELS[t]}{info}")
        for k, v in r.items():
            if k == "pass" or (k == "status" and v == "ok"):
                continue
            print(f"      {k:<22s} {_fmt(v)}")
    bad = [t for t in TESTS if not res[t].get("pass")]
    print("\n" + "-" * 78)
    print(f"summary: {'ALL PASS' if not bad else 'FAILED: ' + ','.join(bad)}")
    print("-" * 78)


def parse_layout(s: str) -> Layout:
    rx, ry = (int(x) for x in s.lower().split("x"))
    if rx * ry > jax.device_count():
        raise SystemExit(f"layout {s} needs {rx * ry} devices, have {jax.device_count()}")
    if M % rx or N % ry:
        raise SystemExit(f"M={M} N={N} not divisible by the {s} layout")
    return Layout(M, N, rx, ry)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--transport", default="allgather")
    ap.add_argument("--layout", default="2x2")
    ap.add_argument("--steps", type=int, default=N_STEPS)
    ap.add_argument("--repeats", type=int, default=5, help="timed repeats; min is reported")
    a = ap.parse_args(argv)
    layout = parse_layout(a.layout)
    try:
        res = run_battery(a.transport, layout, a.steps, a.repeats)
    except KeyError as e:
        raise SystemExit(str(e).strip('"')) from None
    print_battery(a.transport, layout, a.steps, res)
    return 0 if all(res[t].get("pass") for t in TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
