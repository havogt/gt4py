# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Shallow water model distributed with GHEX, forward mode.

    HWLOC_COMPONENTS=-gl mpirun -n R python swm_ghex.py    # R must divide M
    python swm_ghex.py                                       # single rank, no hydra

``HWLOC_COMPONENTS=-gl`` is needed for every hydra launch on this machine, ``-n 1``
included (see README_ghex.md).

1-D ring decomposition along ``I``: rank r owns global rows ``[r*MLOC, (r+1)*MLOC)``
and all ``N`` columns. Periodicity in ``I`` is a GHEX halo exchange (width 1 in ``I``,
0 in ``J``); periodicity in ``J`` is applied locally by ``periodic_j``. The local state
lives on the halo domain ``{I: (-1, MLOC+1), J: (-1, N+1)}`` as JAX-backed gt4py fields.

``make_exchange`` wraps the GHEX exchange in ``jax.custom_vjp`` around
``jax.pure_callback`` so JAX treats it as a pure function.

``operators.timestep`` calls ``make_periodic`` on its outputs, which for ``M = MLOC``
writes *locally* periodic ``I`` halos. Those rows are wrong for R > 1 but are never
read: every step begins by overwriting them with the exchange.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from ghex.context import make_context
from ghex.structured.cartesian_sets import UnitRange
from ghex.structured.regular import (
    DomainDescriptor,
    HaloGenerator,
    make_communication_object,
    make_field_descriptor,
    make_pattern,
)
from ghex.util import Architecture
from mpi4py import MPI

from gt4py import next as gtx
from gt4py.next.experimental import concat_where

from halo_operators import halo_domain, halo_exchange, interior_domain
from initial_conditions import initialize_interior
from operators import IJField, J
from operators import timestep as gtx_timestep

jax.config.update("jax_enable_x64", True)

# The field-operator wrapper dispatches to a backend and cannot be traced; its Python
# definition runs the stencils eagerly on the JAX-backed fields, which JAX can see through.
timestep = gtx_timestep.definition

M = N = 16
dx = dy = 100000.0
dt, radius, alpha = 90.0, 1000000.0, 0.001
N_STEPS = 10


def jax_field(domain, array):
    return gtx.as_field(domain, jnp.asarray(array, dtype=jnp.float64), allocator=jnp)


@gtx.field_operator
def periodic_j(f: IJField, N: gtx.int32) -> IJField:
    f = concat_where(J == -1, f(J + N), f)
    f = concat_where(J == N, f(J - N), f)
    return f


def make_exchange(comm, global_shape, interior, halo, bwd=None):
    """GHEX halo exchange of this rank's block as a ``jax.custom_vjp`` function.

    ``interior`` is the block's ``((i_lo, i_hi), (j_lo, j_hi))`` in the periodic
    ``global_shape`` grid; the exchanged array is the block padded by ``halo`` cells per
    dimension. ``bwd(exchange_np, g)`` is the NumPy backward rule; it receives the raw
    GHEX exchange so it can reuse the forward pattern.
    """
    ctx = make_context(comm, False)
    domain = DomainDescriptor(ctx.rank(), UnitRange(*interior[0]) * UnitRange(*interior[1]))
    halo_gen = HaloGenerator(
        UnitRange(0, global_shape[0]) * UnitRange(0, global_shape[1]),
        tuple((h, h) for h in halo),
        tuple(h > 0 for h in halo),
    )
    pattern = make_pattern(ctx, halo_gen, [domain])
    co = make_communication_object(ctx)
    buf = np.empty(tuple(hi - lo + 2 * h for (lo, hi), h in zip(interior, halo)))
    fdesc = make_field_descriptor(domain, buf, halo, buf.shape, arch=Architecture.CPU)

    def exchange_np(a):
        np.copyto(buf, a)
        co.exchange([pattern(fdesc)]).wait()
        return buf.copy()

    @jax.custom_vjp
    def exchange(a):
        return jax.pure_callback(exchange_np, jax.ShapeDtypeStruct(a.shape, a.dtype), a)

    def fwd(a):
        return exchange(a), None

    def bwd_rule(_, g):
        if bwd is None:
            raise NotImplementedError("adjoint of the GHEX exchange")
        out = jax.ShapeDtypeStruct(g.shape, g.dtype)
        return (jax.pure_callback(lambda g: bwd(exchange_np, g), out, g),)

    exchange.defvjp(fwd, bwd_rule)
    return exchange


def run_forward(refresh_halos, u0, v0, p0, n_steps):
    """u0, v0, p0: interior blocks. refresh_halos maps a halo-domain field to itself.
    Returns fields on the halo domain."""
    mloc, nloc = u0.shape
    dom = halo_domain(mloc, nloc)
    u, v, p = (refresh_halos(jax_field(dom, jnp.pad(a, 1))) for a in (u0, v0, p0))
    state = timestep(u, v, p, dx, dy, dt, u, v, p, 0.0, mloc, nloc)
    for _ in range(n_steps - 1):
        u, v, p, uo, vo, po = state
        u, v, p = (refresh_halos(f) for f in (u, v, p))
        state = timestep(u, v, p, dx, dy, 2.0 * dt, uo, vo, po, alpha, mloc, nloc)
    return state[0], state[1], state[2]


def run_forward_reference(u0, v0, p0, n_steps):
    """The whole domain in one process; the halo refresh is the local periodic copy."""
    return run_forward(lambda f: halo_exchange(f, M, N), u0, v0, p0, n_steps)


def gather_to_root(comm, block, ry=1):
    """Interior blocks, rank r at (r // ry, r % ry) -> global (M, N) array on rank 0, None elsewhere."""
    blocks = comm.gather(np.asarray(block), root=0)
    if comm.Get_rank() != 0:
        return None
    mloc, nloc = block.shape
    out = np.empty((M, N))
    for r, b in enumerate(blocks):
        bx, by = divmod(r, ry)
        out[bx * mloc : (bx + 1) * mloc, by * nloc : (by + 1) * nloc] = b
    return out


def report(layout, gathered, u_g, v_g, p_g):
    ref = run_forward_reference(u_g, v_g, p_g, N_STEPS)
    worst = 0.0
    for name, got, r in zip("uvp", gathered, ref):
        diff = float(np.max(np.abs(got - np.asarray(r[interior_domain(M, N)].ndarray))))
        worst = max(worst, diff)
        print(f"{name}: max |ghex - reference| = {diff:.3e}")
    print(
        f"{layout}, M={M} N={N}, {N_STEPS} steps: "
        f"max abs diff {worst:.3e} -> {'PASS' if worst < 1e-9 else 'FAIL'}"
    )
    sys.stdout.flush()


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if M % size:
        raise SystemExit(f"M={M} is not divisible by {size} ranks")
    MLOC = M // size
    I0 = rank * MLOC

    # The J halo columns ride along as ordinary columns of an M x (N+2) global grid.
    exchange = make_exchange(comm, (M, N + 2), ((I0, I0 + MLOC), (0, N + 2)), (1, 0))
    dom = halo_domain(MLOC, N)

    def refresh_halos(f):
        return periodic_j(jax_field(dom, exchange(f.ndarray)), N)

    u_g, v_g, p_g = initialize_interior(np, M, N, dx, dy, radius)
    rows = slice(I0, I0 + MLOC)
    u, v, p = run_forward(refresh_halos, u_g[rows], v_g[rows], p_g[rows], N_STEPS)
    gathered = [gather_to_root(comm, f[interior_domain(MLOC, N)].ndarray) for f in (u, v, p)]
    if rank == 0:
        report(f"ranks {size}, MLOC={MLOC}", gathered, u_g, v_g, p_g)


if __name__ == "__main__":
    main()
