# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Sharded, differentiable shallow water model with a pluggable halo transport.

One jitted program per (transport, layout, n_steps, remat): a ``shard_map`` over the 1-D
device axis ``"d"`` whose body zero-pads this device's interior blocks to the halo
shape, refreshes the halos with the transport, and runs the nb01 leapfrog loop as a
``jax.lax.scan`` of GT4Py ``operators.timestep`` on JAX-backed local fields.

``timestep`` calls ``make_periodic`` with the *local* sizes, so for P > 1 the halos it
writes are wrong -- and never read: the next iteration's exchange overwrites them
before anything touches them, and the ``old`` fields' halos never reach an interior
cell (``uold_new = u + alpha*(unew - 2u + uold)`` is evaluated on the pre-periodic
``unew``, whose domain is the interior). Same argument as ``swm_ghex_2d.py``.

With ``remat=True`` the scan step is wrapped in ``jax.checkpoint``: the reverse pass
recomputes each step instead of storing its residuals. The forward is unchanged.

The test battery and CLI live in ``swm_battery.py``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from halo_transports import Layout
from jax_compat import patch_gt4py_tracer_dispatch, shard_map

GT4PY_TRACER_DISPATCH = patch_gt4py_tracer_dispatch()

from gt4py import next as gtx
from operators import I, J, make_periodic
from operators import timestep as gtx_timestep

timestep = gtx_timestep.definition

M = N = 16
dx = dy = 100000.0
dt, radius, alpha = 90.0, 1000000.0, 0.001
N_STEPS = 10

AXIS = "d"


def _gt_step(arrays, dt_, alpha_, mloc, nloc):
    """One ``operators.timestep`` on raw ``(mloc+2, nloc+2)`` jnp arrays."""
    dom = gtx.domain({I: (-1, mloc + 1), J: (-1, nloc + 1)})
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

    Under multiple processes every process computes the whole array and contributes
    only the blocks its own devices own.
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
    """Global ``jax.Array`` -> numpy on every process (a multi-process array is not
    fully addressable, so ``np.asarray`` on it raises)."""
    if jax.process_count() == 1:
        return np.asarray(x)
    from jax.experimental import multihost_utils

    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


@functools.lru_cache(maxsize=None)
def tables(transport, layout: Layout):
    return transport.prepare(layout)


@functools.lru_cache(maxsize=None)
def exchange_program(transport, layout: Layout):
    """One exchange on the rank-major halo stack ``(P*(MLOC+2h), NLOC+2h)``, jitted."""
    t = tables(transport, layout)
    return jax.jit(
        shard_map(
            lambda a: transport.exchange(a, t, AXIS),
            mesh=make_mesh(layout.P),
            in_specs=P(AXIS),
            out_specs=P(AXIS),
        )
    )


@functools.lru_cache(maxsize=None)
def sharded_program(transport, layout: Layout, n_steps: int, remat: bool = False):
    L = layout
    t = tables(transport, L)
    mloc, nloc = L.MLOC, L.NLOC

    def exchange(a):
        return transport.exchange(a, t, AXIS)

    def body(u0, v0, p0):
        u, v, p = (exchange(jnp.pad(a, L.h)) for a in (u0, v0, p0))
        state = _gt_step((u, v, p, u, v, p), dt, 0.0, mloc, nloc)

        def scan_step(carry, _):
            u, v, p, uo, vo, po = carry
            u, v, p = (exchange(a) for a in (u, v, p))
            return _gt_step((u, v, p, uo, vo, po), 2.0 * dt, alpha, mloc, nloc), None

        step = jax.checkpoint(scan_step) if remat else scan_step
        final, _ = jax.lax.scan(step, state, None, length=n_steps - 1)
        h = L.h
        return tuple(f[h:-h, h:-h] for f in final[:3])

    return jax.jit(
        shard_map(body, mesh=make_mesh(L.P), in_specs=(P(AXIS),) * 3, out_specs=(P(AXIS),) * 3)
    )


def sharded_forward(transport, layout: Layout, u0, v0, p0, n_steps=N_STEPS):
    """u0, v0, p0: rank-major stacks of interior blocks, ``(P*MLOC, NLOC)``.

    Returns the same layout; use ``unblock`` for the global ``(M, N)`` fields.
    """
    return sharded_program(transport, layout, n_steps)(u0, v0, p0)


def cost(fields):
    """Scalar objective: sum of squares of p over the global interior."""
    return jnp.sum(fields[2] * fields[2])


# --- single-device references on the global (M, N) grid -----------------------------------------
@functools.lru_cache(maxsize=None)
def reference_program(n_steps: int, M: int = M, N: int = N, remat: bool = False):
    """The same GT4Py step and scan structure, halos from ``make_periodic``."""
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

        step = jax.checkpoint(scan_step) if remat else scan_step
        final, _ = jax.lax.scan(step, state, None, length=n_steps - 1)
        return tuple(f[1:-1, 1:-1] for f in final[:3])

    return jax.jit(run)


@functools.lru_cache(maxsize=None)
def wrap_reference_program(n_steps: int):
    """Halos from an explicit wrap-pad: forward bit-identical to ``reference_program``,
    adjoint accumulated in a different order."""

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


# Pure jnp.roll, sharing no code with the GT4Py path. The arithmetic is deliberately
# associated differently from operators.py; do not "simplify" it into the same form.
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
    unew = (
        uo
        + 0.5 * (_rp(z, 1, 1) + z) * 0.5 * (_rp(acv, 1, 1) + acv) * dt_
        - (_rp(h, -1, 0) - h) / dx * dt_
    )
    vnew = (
        vo
        - 0.5 * (_rp(z, 1, 0) + z) * 0.5 * (_rp(acu, 1, 0) + acu) * dt_
        - (_rp(h, -1, 1) - h) / dy * dt_
    )
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
def roll_reference_program(n_steps: int):
    def run(u0, v0, p0):
        state = _roll_step(u0, v0, p0, u0, v0, p0, dt, 0.0)

        def scan_step(carry, _):
            return _roll_step(*carry, 2.0 * dt, alpha), None

        final, _ = jax.lax.scan(scan_step, state, None, length=n_steps - 1)
        return final[:3]

    return jax.jit(run)
