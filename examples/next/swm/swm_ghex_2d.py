# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Shallow water model distributed with GHEX on a 2-D decomposition, forward mode.

    HWLOC_COMPONENTS=-gl mpirun -n 4 python swm_ghex_2d.py 2 2   # Rx Ry, Rx*Ry == ranks
    python swm_ghex_2d.py 1 1                                     # single rank, no hydra

Without arguments the layout is ``SIZE x 1``. ``HWLOC_COMPONENTS=-gl`` is needed
for every hydra launch on this machine (see README_ghex.md).

Rank ``r`` owns block ``(rx, ry) = divmod(r, Ry)``: global rows
``[rx*MLOC, (rx+1)*MLOC)`` and columns ``[ry*NLOC, (ry+1)*NLOC)`` with
``MLOC = M // Rx``, ``NLOC = N // Ry``. Periodicity in both directions is a single
GHEX halo exchange (width 1 in ``I`` and ``J``, periodicity ``(True, True)``);
GHEX's structured halo generator fills the four corner cells as well, so there is
no local periodic step. The local state lives on the halo domain
``{I: (-1, MLOC+1), J: (-1, NLOC+1)}`` as JAX-backed gt4py fields.

``operators.timestep`` calls ``make_periodic`` on its outputs, which for
``M = MLOC, N = NLOC`` writes *locally* periodic halos. Those cells are wrong
whenever ``Rx > 1`` or ``Ry > 1`` but are never read: every step begins by
overwriting them with the exchange.

Parameters and the single-process reference match ``swm_ghex.py`` (duplicated so this
module is standalone: importing that file would run its 1-D GHEX setup and rank check).
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI

jax.config.update("jax_enable_x64", True)

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

from gt4py import next as gtx

from initial_conditions import initialize_interior
from operators import I, J, make_periodic
from operators import timestep as gtx_timestep

timestep = gtx_timestep.definition

M = N = 16
dx = dy = 100000.0
dt, radius, alpha = 90.0, 1000000.0, 0.001
N_STEPS = 10
DOM_G_INT = gtx.domain({I: (0, M), J: (0, N)})


def fld(domain, array):
    return gtx.as_field(domain, jnp.asarray(array, dtype=jnp.float64), allocator=jnp)


def run_forward_reference(u0, v0, p0, n_steps):
    """nb01 forward model on the full global domain, single process."""
    u, v, p = (make_periodic(fld(DOM_G_INT, a), M, N) for a in (u0, v0, p0))
    state = timestep(u, v, p, dx, dy, dt, u, v, p, 0.0, M, N)
    for _ in range(n_steps - 1):
        u, v, p, uo, vo, po = state
        state = timestep(u, v, p, dx, dy, 2.0 * dt, uo, vo, po, alpha, M, N)
    return state[0], state[1], state[2]

COMM = MPI.COMM_WORLD
RANK, SIZE = COMM.Get_rank(), COMM.Get_size()
RX, RY = (int(a) for a in sys.argv[1:3]) if len(sys.argv) == 3 else (SIZE, 1)
if RX * RY != SIZE:
    raise SystemExit(f"layout {RX}x{RY} needs {RX * RY} ranks, launched with {SIZE}")
if M % RX or N % RY:
    raise SystemExit(f"M={M} N={N} is not divisible by the {RX}x{RY} layout")
MLOC, NLOC = M // RX, N // RY
BX, BY = divmod(RANK, RY)
I0, J0 = BX * MLOC, BY * NLOC

DOM_L_HALO = gtx.domain({I: (-1, MLOC + 1), J: (-1, NLOC + 1)})
DOM_L_INT = gtx.domain({I: (0, MLOC), J: (0, NLOC)})


# --- GHEX setup: the exchanged array is the full local halo array (MLOC+2, NLOC+2).
_ctx = make_context(COMM, False)
_owned = UnitRange(I0, I0 + MLOC) * UnitRange(J0, J0 + NLOC)
_domain = DomainDescriptor(_ctx.rank(), _owned)
_halo_gen = HaloGenerator(UnitRange(0, M) * UnitRange(0, N), ((1, 1), (1, 1)), (True, True))
_pattern = make_pattern(_ctx, _halo_gen, [_domain])
_co = make_communication_object(_ctx)
_buf = np.empty((MLOC + 2, NLOC + 2), dtype=np.float64)
_fdesc = make_field_descriptor(_domain, _buf, (1, 1), _buf.shape, arch=Architecture.CPU)


def _exchange_impl(a):
    np.copyto(_buf, a)
    _co.exchange([_pattern(_fdesc)]).wait()
    return _buf.copy()


@jax.custom_vjp
def ghex_exchange(a):
    if a.shape != _buf.shape or a.dtype != _buf.dtype:
        raise ValueError(f"ghex_exchange expects {_buf.shape} {_buf.dtype}, got {a.shape} {a.dtype}")
    return jax.pure_callback(_exchange_impl, jax.ShapeDtypeStruct(a.shape, a.dtype), a)


def _fwd(a):
    return ghex_exchange(a), None


def _bwd(_, g):
    raise NotImplementedError("adjoint of ghex_exchange")


ghex_exchange.defvjp(_fwd, _bwd)


def refresh_halos(f):
    return fld(DOM_L_HALO, ghex_exchange(f.ndarray))


def run_forward(u0, v0, p0, n_steps):
    """u0, v0, p0: this rank's interior blocks (MLOC, NLOC). Returns fields on DOM_L_HALO."""
    u, v, p = (refresh_halos(fld(DOM_L_HALO, jnp.pad(a, 1))) for a in (u0, v0, p0))
    state = timestep(u, v, p, dx, dy, dt, u, v, p, 0.0, MLOC, NLOC)
    for _ in range(n_steps - 1):
        u, v, p, uo, vo, po = state
        u, v, p = (refresh_halos(f) for f in (u, v, p))
        state = timestep(u, v, p, dx, dy, 2.0 * dt, uo, vo, po, alpha, MLOC, NLOC)
    return state[0], state[1], state[2]


def run_forward_scan(u0, v0, p0, n_steps):
    """nb01 forward model: make_periodic on the global domain, time loop as jax.lax.scan."""
    u, v, p = (make_periodic(fld(DOM_G_INT, a), M, N) for a in (u0, v0, p0))
    state = timestep(u, v, p, dx, dy, dt, u, v, p, 0.0, M, N)

    def step(carry, _):
        u, v, p, uo, vo, po = carry
        return timestep(u, v, p, dx, dy, 2.0 * dt, uo, vo, po, alpha, M, N), None

    final, _ = jax.lax.scan(step, state, None, length=n_steps - 1)
    return final[0], final[1], final[2]


def gather_to_root(f):
    """Interior of a DOM_L_HALO field -> global (M, N) numpy array on rank 0, None elsewhere."""
    blocks = COMM.gather(np.asarray(f[DOM_L_INT].ndarray), root=0)
    if RANK != 0:
        return None
    out = np.empty((M, N))
    for r, b in enumerate(blocks):
        bx, by = divmod(r, RY)
        out[bx * MLOC : (bx + 1) * MLOC, by * NLOC : (by + 1) * NLOC] = b
    return out


def main():
    u_g, v_g, p_g = initialize_interior(np, M, N, dx, dy, radius)
    loc = (slice(I0, I0 + MLOC), slice(J0, J0 + NLOC))
    u, v, p = run_forward(u_g[loc], v_g[loc], p_g[loc], N_STEPS)
    gathered = [gather_to_root(f) for f in (u, v, p)]

    if RANK == 0:
        ref = run_forward_reference(u_g, v_g, p_g, N_STEPS)
        scan = run_forward_scan(u_g, v_g, p_g, N_STEPS)
        worst = 0.0
        for name, got, r, s in zip("uvp", gathered, ref, scan):
            diff = float(np.max(np.abs(got - np.asarray(r[DOM_G_INT].ndarray))))
            diff_scan = float(np.max(np.abs(got - np.asarray(s[DOM_G_INT].ndarray))))
            worst = max(worst, diff)
            print(f"{name}: max |ghex - reference| = {diff:.3e}   max |ghex - nb01 scan| = {diff_scan:.3e}")
        print(f"ranks {SIZE} ({RX}x{RY}), M={M} N={N} MLOC={MLOC} NLOC={NLOC}, {N_STEPS} steps: "
              f"max abs diff {worst:.3e} -> {'PASS' if worst < 1e-9 else 'FAIL'}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
