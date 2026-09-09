# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Shallow water model distributed with GHEX on a 2-D decomposition, forward mode.

    HWLOC_COMPONENTS=-gl mpirun -n 4 python swm_ghex_2d.py 2 2   # RX RY, RX*RY == ranks
    python swm_ghex_2d.py 1 1                                     # single rank, no hydra

Without arguments the layout is ``SIZE x 1``. ``HWLOC_COMPONENTS=-gl`` is needed
for every hydra launch on this machine (see README_ghex.md).

Rank ``r`` owns block ``(BX, BY) = divmod(r, RY)``: global rows ``[BX*MLOC, (BX+1)*MLOC)``
and columns ``[BY*NLOC, (BY+1)*NLOC)`` with ``MLOC = M // RX``, ``NLOC = N // RY``.
Periodicity in both directions is a single GHEX halo exchange (width 1 in ``I`` and
``J``); GHEX's structured halo generator fills the four corner cells as well, so there
is no local periodic step. The local state lives on the halo domain
``{I: (-1, MLOC+1), J: (-1, NLOC+1)}`` as JAX-backed gt4py fields.

``operators.timestep`` calls ``make_periodic`` on its outputs, which for
``M = MLOC, N = NLOC`` writes *locally* periodic halos. Those cells are wrong
whenever ``RX > 1`` or ``RY > 1`` but are never read: every step begins by
overwriting them with the exchange.
"""

import sys

import numpy as np
from mpi4py import MPI

from halo_operators import halo_domain, interior_domain
from initial_conditions import initialize_interior
from swm_ghex import (
    M,
    N,
    N_STEPS,
    dx,
    dy,
    gather_to_root,
    jax_field,
    make_exchange,
    radius,
    report,
    run_forward,
)


def main():
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    RX, RY = map(int, sys.argv[1:3]) if len(sys.argv) == 3 else (size, 1)
    if RX * RY != size:
        raise SystemExit(f"layout {RX}x{RY} needs {RX * RY} ranks, launched with {size}")
    if M % RX or N % RY:
        raise SystemExit(f"M={M} N={N} is not divisible by the {RX}x{RY} layout")
    MLOC, NLOC = M // RX, N // RY
    BX, BY = divmod(rank, RY)
    I0, J0 = BX * MLOC, BY * NLOC

    exchange = make_exchange(comm, (M, N), ((I0, I0 + MLOC), (J0, J0 + NLOC)), (1, 1))
    dom = halo_domain(MLOC, NLOC)

    def refresh_halos(f):
        return jax_field(dom, exchange(f.ndarray))

    u_g, v_g, p_g = initialize_interior(np, M, N, dx, dy, radius)
    block = (slice(I0, I0 + MLOC), slice(J0, J0 + NLOC))
    u, v, p = run_forward(refresh_halos, u_g[block], v_g[block], p_g[block], N_STEPS)
    gathered = [gather_to_root(comm, f[interior_domain(MLOC, NLOC)].ndarray, RY) for f in (u, v, p)]
    if rank == 0:
        report(f"ranks {size} ({RX}x{RY}), MLOC={MLOC} NLOC={NLOC}", gathered, u_g, v_g, p_g)


if __name__ == "__main__":
    main()
