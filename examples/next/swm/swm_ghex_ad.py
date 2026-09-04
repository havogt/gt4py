# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Reverse-mode AD of the GHEX-distributed shallow water model.

    mpirun -n R python swm_ghex_ad.py        # R must divide M, MLOC >= 2

Installs the backward rule of ``swm_ghex.ghex_exchange``. On the 1-D ring the
adjoint of the exchange reuses the *forward* GHEX pattern on a scratch buffer:
the halo cotangents are placed in the two boundary interior rows, exchanged,
and the halo rows that come back are accumulated into the owning interior
rows. The incoming halo cotangent itself is dropped, since the forward
exchange overwrote the halo.

Checks, in order: a distributed dot-product test of the exchange alone; the
gradient of ``J = sum over ranks of sum_interior p(N_STEPS)**2`` against
``jax.grad`` of the single-process reference model; a distributed Taylor test
on ``J``.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI

from initial_conditions import initialize_interior
from swm_ghex import (
    COMM,
    DOM_G_INT,
    DOM_L_INT,
    I0,
    M,
    MLOC,
    N,
    RANK,
    SIZE,
    _exchange_impl,
    _fwd,
    dx,
    dy,
    ghex_exchange,
    radius,
    run_forward,
    run_forward_reference,
)

N_STEPS = 5

if MLOC < 2:
    raise SystemExit(f"MLOC={MLOC}: the scratch-buffer adjoint needs two distinct boundary rows")


def _exchange_adjoint_impl(g):
    t = np.zeros_like(g)
    t[1] = g[0]
    t[-2] = g[-1]
    t = _exchange_impl(t)
    a_bar = np.zeros_like(g)
    a_bar[1:-1] = g[1:-1]
    a_bar[-2] += t[-1]
    a_bar[1] += t[0]
    return a_bar


def _bwd(_, g):
    return (jax.pure_callback(_exchange_adjoint_impl, jax.ShapeDtypeStruct(g.shape, g.dtype), g),)


ghex_exchange.defvjp(_fwd, _bwd)


def allsum(x):
    return COMM.allreduce(float(x), op=MPI.SUM)


def dot_product_test(rng):
    x = jnp.asarray(rng.standard_normal((MLOC + 2, N + 2)))
    y = jnp.asarray(rng.standard_normal((MLOC + 2, N + 2)))
    lx, vjp = jax.vjp(ghex_exchange, x)
    (x_bar,) = vjp(y)
    lhs = allsum(jnp.sum(lx * y))
    rhs = allsum(jnp.sum(x * x_bar))
    return lhs, rhs, abs(lhs - rhs) / abs(lhs)


def cost_local(u0, v0, p0):
    _, _, p = run_forward(u0, v0, p0, N_STEPS)
    return jnp.sum(p[DOM_L_INT].ndarray ** 2)


def cost(u0, v0, p0):
    return allsum(cost_local(u0, v0, p0))


def cost_reference(u0, v0, p0):
    _, _, p = run_forward_reference(u0, v0, p0, N_STEPS)
    return jnp.sum(p[DOM_G_INT].ndarray ** 2)


def taylor_test(x, grad, direction):
    j0 = cost(*x)
    dj = allsum(sum(jnp.sum(g * d) for g, d in zip(grad, direction)))
    rows, h, prev = [], 1e-2, None
    for _ in range(7):
        jh = cost(*(xi + h * di for xi, di in zip(x, direction)))
        r2 = abs(jh - j0 - h * dj)
        rows.append((h, r2, np.log2(prev / r2) if prev else float("nan")))
        prev, h = r2, h / 2
    return j0, dj, rows


def main():
    rng = np.random.default_rng(RANK)

    lhs, rhs, rel = dot_product_test(rng)
    if RANK == 0:
        print(f"dot-product test of ghex_exchange on {SIZE} ranks")
        print(f"  <Lx, y>    = {lhs:.16e}")
        print(f"  <x, L^T y> = {rhs:.16e}")
        print(f"  relative difference {rel:.2e} -> {'PASS' if rel < 1e-12 else 'FAIL'}")

    u_g, v_g, p_g = initialize_interior(np, M, N, dx, dy, radius)
    loc = slice(I0, I0 + MLOC)
    x = tuple(jnp.asarray(a[loc]) for a in (u_g, v_g, p_g))
    grad = jax.grad(cost_local, argnums=(0, 1, 2))(*x)
    blocks = [COMM.gather(np.asarray(g), root=0) for g in grad]

    if RANK == 0:
        grad_ref = jax.grad(cost_reference, argnums=(0, 1, 2))(
            *(jnp.asarray(a) for a in (u_g, v_g, p_g))
        )
        print(f"gradient of J = sum p({N_STEPS} steps)^2, {SIZE} ranks vs single-process reference")
        # p_bar = 2p ~ 1e5 at the final step while the stencil transpose forms
        # differences of it ~10, so the reordered accumulation at rank boundaries
        # costs ~1e-12 relative, not 1e-16.
        worst = 0.0
        for name, b, r in zip("uvp", blocks, grad_ref):
            r = np.asarray(r)
            diff = float(np.max(np.abs(np.concatenate(b) - r)))
            rel = diff / float(np.max(np.abs(r)))
            worst = max(worst, rel)
            print(f"  dJ/d{name}0: max |ghex - reference| = {diff:.3e}, relative {rel:.3e}")
        print(f"  max relative diff {worst:.3e} -> {'PASS' if worst < 1e-10 else 'FAIL'}")

    direction = tuple(jnp.asarray(rng.standard_normal(a.shape) * float(jnp.std(a))) for a in x)
    j0, dj, rows = taylor_test(x, grad, direction)
    if RANK == 0:
        print(f"Taylor test on {SIZE} ranks: J(x) = {j0:.8e}, <grad J, d> = {dj:.8e}")
        print(f"  {'h':>10} {'r2':>14} {'rate2':>7}")
        for h, r2, rate in rows:
            print(f"  {h:10.2e} {r2:14.6e} {rate:7.2f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
