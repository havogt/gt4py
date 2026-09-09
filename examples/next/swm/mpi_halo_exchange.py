# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Differentiable MPI halo exchange, verified by a distributed dot-product test.

    HWLOC_COMPONENTS=-gl mpirun -n 4 python mpi_halo_exchange.py

A 1-D ring decomposition along ``I``; the local buffer ``(MLOC + 2, N)`` carries the
``I`` halo rows, ``J`` is untouched. The exchange is opaque to JAX, so its adjoint is
supplied by hand through ``jax.custom_vjp``: the backward pass sends halo cotangents
back the way they came and accumulates them into the owners. The MPI calls run through
``jax.pure_callback`` because they need concrete buffers.
"""

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI

jax.config.update("jax_enable_x64", True)

COMM = MPI.COMM_WORLD
RANK, SIZE = COMM.Get_rank(), COMM.Get_size()
I_LO, I_HI = (RANK - 1) % SIZE, (RANK + 1) % SIZE  # neighbours on the ring along I

MLOC, N = 3, 5
LO_HALO, LO_INT, HI_INT, HI_HALO = 0, 1, -2, -1  # rows of the local buffer


def _exchange_impl(a):
    out = np.array(a)  # the callback's array has a byte-order-tagged buffer mpi4py rejects
    COMM.Sendrecv(out[HI_INT], dest=I_HI, recvbuf=out[LO_HALO], source=I_LO)
    COMM.Sendrecv(out[LO_INT], dest=I_LO, recvbuf=out[HI_HALO], source=I_HI)
    return out


def _exchange_adjoint_impl(g):
    g = np.array(g)
    a_bar = np.zeros_like(g)  # the halo rows get nothing: the forward overwrote them
    a_bar[LO_INT:HI_HALO] = g[LO_INT:HI_HALO]
    recv = np.empty(N)
    COMM.Sendrecv(g[LO_HALO], dest=I_LO, recvbuf=recv, source=I_HI)
    a_bar[HI_INT] += recv
    COMM.Sendrecv(g[HI_HALO], dest=I_HI, recvbuf=recv, source=I_LO)
    a_bar[LO_INT] += recv
    return a_bar


@jax.custom_vjp
def mpi_exchange(a):
    return jax.pure_callback(_exchange_impl, jax.ShapeDtypeStruct(a.shape, a.dtype), a)


def _fwd(a):
    return mpi_exchange(a), None


def _bwd(_, g):
    return (jax.pure_callback(_exchange_adjoint_impl, jax.ShapeDtypeStruct(g.shape, g.dtype), g),)


mpi_exchange.defvjp(_fwd, _bwd)


def main():
    rng = np.random.default_rng(RANK)
    x = jnp.asarray(rng.standard_normal((MLOC + 2, N)))
    y = jnp.asarray(rng.standard_normal((MLOC + 2, N)))

    lx, vjp = jax.vjp(mpi_exchange, x)
    (x_bar,) = vjp(y)
    lhs = COMM.allreduce(float(jnp.sum(lx * y)), op=MPI.SUM)
    rhs = COMM.allreduce(float(jnp.sum(x * x_bar)), op=MPI.SUM)

    if RANK == 0:
        rel = abs(lhs - rhs) / abs(lhs)
        print(f"dot-product test of mpi_exchange on {SIZE} ranks")
        print(f"  <Lx, y>    = {lhs:.16e}")
        print(f"  <x, L^T y> = {rhs:.16e}")
        print(f"  relative difference {rel:.2e} -> {'PASS' if rel < 1e-12 else 'FAIL'}")


if __name__ == "__main__":
    main()
