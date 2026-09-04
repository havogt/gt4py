# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Differentiable MPI halo exchange, verified by a distributed dot-product test.

    mpirun -n 4 python mpi_halo_exchange.py

A 1-D ring decomposition along ``I``. The exchange is opaque to JAX, so its
adjoint is supplied by hand through ``jax.custom_vjp``: the backward pass sends
halo cotangents back the way they came and accumulates them into the owners.

The MPI calls have to run through ``jax.pure_callback`` because they need
concrete buffers, which also means they cannot be placed under ``jax.jit``.
``mpi4jax`` avoids that by registering MPI operations as XLA custom calls.
"""

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI

jax.config.update("jax_enable_x64", True)

COMM = MPI.COMM_WORLD
RANK, SIZE = COMM.Get_rank(), COMM.Get_size()
LEFT, RIGHT = (RANK - 1) % SIZE, (RANK + 1) % SIZE

MLOC, N = 3, 5


def _exchange_impl(a_loc):
    a_loc = np.asarray(a_loc)
    out = np.zeros((MLOC + 2, N), dtype=a_loc.dtype)
    out[1:-1] = a_loc
    # my last row fills the right neighbour's left halo, and symmetrically
    COMM.Sendrecv(np.ascontiguousarray(a_loc[-1:]), dest=RIGHT, recvbuf=out[0:1], source=LEFT)
    COMM.Sendrecv(np.ascontiguousarray(a_loc[0:1]), dest=LEFT, recvbuf=out[-1:], source=RIGHT)
    return out


def _exchange_adjoint_impl(g):
    g = np.asarray(g)
    own = np.array(g[1:-1])
    recv = np.zeros((1, N), dtype=g.dtype)
    # cotangent of my left halo belongs to the left neighbour's last row
    COMM.Sendrecv(np.ascontiguousarray(g[0:1]), dest=LEFT, recvbuf=recv, source=RIGHT)
    own[-1:] += recv
    # cotangent of my right halo belongs to the right neighbour's first row
    COMM.Sendrecv(np.ascontiguousarray(g[-1:]), dest=RIGHT, recvbuf=recv, source=LEFT)
    own[0:1] += recv
    return own  # the halo cotangent is dropped, i.e. zeroed


@jax.custom_vjp
def mpi_exchange(a_loc):
    return jax.pure_callback(
        _exchange_impl, jax.ShapeDtypeStruct((MLOC + 2, N), a_loc.dtype), a_loc
    )


def _fwd(a_loc):
    return mpi_exchange(a_loc), None


def _bwd(_, g):
    return (
        jax.pure_callback(_exchange_adjoint_impl, jax.ShapeDtypeStruct((MLOC, N), g.dtype), g),
    )


mpi_exchange.defvjp(_fwd, _bwd)


def main():
    rng = np.random.default_rng(RANK)
    x = jnp.asarray(rng.standard_normal((MLOC, N)))
    y = jnp.asarray(rng.standard_normal((MLOC + 2, N)))

    lhs_local = float(jnp.sum(mpi_exchange(x) * y))
    (x_bar,) = _bwd(None, y)
    rhs_local = float(jnp.sum(x * x_bar))

    lhs = COMM.allreduce(lhs_local, op=MPI.SUM)
    rhs = COMM.allreduce(rhs_local, op=MPI.SUM)

    if RANK == 0:
        rel = abs(lhs - rhs) / abs(lhs)
        print(f"ranks              : {SIZE}")
        print(f"<Lx, y>            : {lhs:.16e}")
        print(f"<x, L^T y>         : {rhs:.16e}")
        print(f"relative difference: {rel:.2e}")
        print("PASS" if rel < 1e-12 else "FAIL")


if __name__ == "__main__":
    main()
