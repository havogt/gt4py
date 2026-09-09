# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Reverse-mode AD of the GHEX-distributed shallow water model.

    HWLOC_COMPONENTS=-gl mpirun -n R python swm_ghex_ad.py    # R must divide M, MLOC >= 2

The 1-D ring of ``swm_ghex.py`` with a backward rule for the exchange. Checks, in
order: a distributed dot-product test of the exchange alone; the gradient of
``J = sum over ranks of sum_interior p(N_STEPS)**2`` against ``jax.grad`` of the
single-process reference; a distributed Taylor test on ``J``.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI

from halo_operators import halo_domain, interior_domain
from initial_conditions import initialize_interior
from swm_ghex import (
    M,
    N,
    dx,
    dy,
    jax_field,
    make_exchange,
    periodic_j,
    radius,
    run_forward,
    run_forward_reference,
)

N_STEPS = 5

LO_HALO, LO_INT, HI_INT, HI_HALO = 0, 1, -2, -1  # rows of the (MLOC+2, N+2) exchange buffer


def exchange_adjoint(exchange_np, g):
    # On the ring the forward pattern is its own mirror: a halo cotangent placed on the
    # boundary interior row arrives, after a forward exchange, in the halo row of the
    # neighbour that owns it. Needs MLOC >= 2 so the two boundary rows are distinct.
    t = np.zeros_like(g)
    t[LO_INT], t[HI_INT] = g[LO_HALO], g[HI_HALO]
    t = exchange_np(t)
    a_bar = np.zeros_like(g)  # the halo rows get nothing: the forward overwrote them
    a_bar[LO_INT:HI_HALO] = g[LO_INT:HI_HALO]
    a_bar[LO_INT] += t[LO_HALO]
    a_bar[HI_INT] += t[HI_HALO]
    return a_bar


def allsum(comm, x):
    return comm.allreduce(float(x), op=MPI.SUM)


def dot_product_test(comm, exchange, shape, rng):
    x = jnp.asarray(rng.standard_normal(shape))
    y = jnp.asarray(rng.standard_normal(shape))
    lx, vjp = jax.vjp(exchange, x)
    (x_bar,) = vjp(y)
    return allsum(comm, jnp.sum(lx * y)), allsum(comm, jnp.sum(x * x_bar))


def cost_reference(u0, v0, p0):
    _, _, p = run_forward_reference(u0, v0, p0, N_STEPS)
    return jnp.sum(p[interior_domain(M, N)].ndarray ** 2)


def compare_gradients(blocks, grad_ref):
    worst = 0.0
    for name, b, r in zip("uvp", blocks, grad_ref):
        r = np.asarray(r)
        diff = float(np.max(np.abs(np.concatenate(b) - r)))
        rel = diff / float(np.max(np.abs(r)))
        worst = max(worst, rel)
        print(f"  dJ/d{name}0: max |ghex - reference| = {diff:.3e}, relative {rel:.3e}")
    # Not roundoff: p_bar ~ 1e5 enters the stencil transpose as differences ~10, so the
    # reordered accumulation at rank boundaries costs ~1e-12 relative.
    print(f"  max relative diff {worst:.3e} -> {'PASS' if worst < 1e-10 else 'FAIL'}")


def taylor_test(comm, cost, x, grad, direction):
    j0 = cost(*x)
    dj = allsum(comm, sum(jnp.sum(g * d) for g, d in zip(grad, direction)))
    hs = 1e-2 / 2.0 ** np.arange(7)
    remainders = [
        abs(cost(*(xi + h * di for xi, di in zip(x, direction))) - j0 - h * dj) for h in hs
    ]
    return j0, dj, hs, remainders


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if M % size or M // size < 2:
        raise SystemExit(f"M={M} on {size} ranks: M must be divisible by the ranks with MLOC >= 2")
    MLOC = M // size
    I0 = rank * MLOC

    exchange = make_exchange(
        comm, (M, N + 2), ((I0, I0 + MLOC), (0, N + 2)), (1, 0), bwd=exchange_adjoint
    )
    dom = halo_domain(MLOC, N)

    def refresh_halos(f):
        return periodic_j(jax_field(dom, exchange(f.ndarray)), N)

    def cost_local(u0, v0, p0):
        _, _, p = run_forward(refresh_halos, u0, v0, p0, N_STEPS)
        return jnp.sum(p[interior_domain(MLOC, N)].ndarray ** 2)

    def cost(u0, v0, p0):
        return allsum(comm, cost_local(u0, v0, p0))

    rng = np.random.default_rng(rank)
    lhs, rhs = dot_product_test(comm, exchange, (MLOC + 2, N + 2), rng)
    if rank == 0:
        rel = abs(lhs - rhs) / abs(lhs)
        print(f"dot-product test of ghex_exchange on {size} ranks")
        print(f"  <Lx, y>    = {lhs:.16e}")
        print(f"  <x, L^T y> = {rhs:.16e}")
        print(f"  relative difference {rel:.2e} -> {'PASS' if rel < 1e-12 else 'FAIL'}")

    u_g, v_g, p_g = initialize_interior(np, M, N, dx, dy, radius)
    x = tuple(jnp.asarray(a[I0 : I0 + MLOC]) for a in (u_g, v_g, p_g))
    # jax.grad of cost_local, run on every rank at once, is the gradient of the global
    # cost: the allreduce's adjoint only seeds every rank with the cotangent 1, and the
    # other ranks' contributions dJ_s/dx_r arrive through the exchange adjoint, which all
    # ranks execute in lockstep. cost itself ends in float() and cannot be traced.
    grad = jax.grad(cost_local, argnums=(0, 1, 2))(*x)
    blocks = [comm.gather(np.asarray(g), root=0) for g in grad]
    if rank == 0:
        x_g = tuple(jnp.asarray(a) for a in (u_g, v_g, p_g))
        grad_ref = jax.grad(cost_reference, argnums=(0, 1, 2))(*x_g)
        print(f"gradient of J = sum p({N_STEPS} steps)^2, {size} ranks vs single-process reference")
        compare_gradients(blocks, grad_ref)

    direction = tuple(jnp.asarray(rng.standard_normal(a.shape) * float(jnp.std(a))) for a in x)
    j0, dj, hs, remainders = taylor_test(comm, cost, x, grad, direction)
    if rank == 0:
        print(f"Taylor test on {size} ranks: J(x) = {j0:.8e}, <grad J, d> = {dj:.8e}")
        print(f"  {'h':>10} {'r2':>14} {'rate2':>7}")
        for k, (h, r2) in enumerate(zip(hs, remainders)):
            rate = np.log2(remainders[k - 1] / r2) if k else float("nan")
            print(f"  {h:10.2e} {r2:14.6e} {rate:7.2f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
