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

1-D ring decomposition along ``I``: rank r owns global rows
``[r*MLOC, (r+1)*MLOC)`` and all ``N`` columns. Periodicity in ``I`` is a GHEX
halo exchange (width 1 in ``I``, 0 in ``J``); periodicity in ``J`` is applied
locally by ``periodic_j``. The local state lives on the halo domain
``{I: (-1, MLOC+1), J: (-1, N+1)}`` as JAX-backed gt4py fields.

``ghex_exchange`` is a ``jax.custom_vjp`` around ``jax.pure_callback`` so JAX
treats it as a pure function; the backward rule is not implemented yet.

``operators.timestep`` calls ``make_periodic`` on its outputs, which for
``M = MLOC`` writes *locally* periodic ``I`` halos. Those rows are wrong for
R > 1 but are never read: every step begins by overwriting them with the
exchange.
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
from gt4py.next.experimental import concat_where

from initial_conditions import initialize_interior
from operators import I, J, IJField, make_periodic
from operators import timestep as gtx_timestep

timestep = gtx_timestep.definition

M = N = 16
dx = dy = 100000.0
dt, radius, alpha = 90.0, 1000000.0, 0.001
N_STEPS = 10

COMM = MPI.COMM_WORLD
RANK, SIZE = COMM.Get_rank(), COMM.Get_size()
if M % SIZE:
    raise SystemExit(f"M={M} is not divisible by {SIZE} ranks")
MLOC = M // SIZE
I0 = RANK * MLOC

DOM_L_HALO = gtx.domain({I: (-1, MLOC + 1), J: (-1, N + 1)})
DOM_L_INT = gtx.domain({I: (0, MLOC), J: (0, N)})
DOM_G_INT = gtx.domain({I: (0, M), J: (0, N)})


def fld(domain, array):
    return gtx.as_field(domain, jnp.asarray(array, dtype=jnp.float64), allocator=jnp)


@gtx.field_operator
def periodic_j(f: IJField, N: gtx.int32) -> IJField:
    f = concat_where(J == -1, f(J + N), f)
    f = concat_where(J == N, f(J - N), f)
    return f


# --- GHEX setup: the exchanged array is the full local halo array (MLOC+2, N+2);
# the J halo columns ride along as ordinary columns of an M x (N+2) global grid.
_ctx = make_context(COMM, False)
_owned = UnitRange(I0, I0 + MLOC) * UnitRange(0, N + 2)
_domain = DomainDescriptor(_ctx.rank(), _owned)
_halo_gen = HaloGenerator(UnitRange(0, M) * UnitRange(0, N + 2), ((1, 1), (0, 0)), (True, False))
_pattern = make_pattern(_ctx, _halo_gen, [_domain])
_co = make_communication_object(_ctx)
_buf = np.empty((MLOC + 2, N + 2), dtype=np.float64)
_fdesc = make_field_descriptor(_domain, _buf, (1, 0), _buf.shape, arch=Architecture.CPU)


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
    return periodic_j(fld(DOM_L_HALO, ghex_exchange(f.ndarray)), N)


def run_forward(u0, v0, p0, n_steps):
    """u0, v0, p0: this rank's interior blocks (MLOC, N). Returns fields on DOM_L_HALO."""
    u, v, p = (refresh_halos(fld(DOM_L_HALO, jnp.pad(a, 1))) for a in (u0, v0, p0))
    state = timestep(u, v, p, dx, dy, dt, u, v, p, 0.0, MLOC, N)
    for _ in range(n_steps - 1):
        u, v, p, uo, vo, po = state
        u, v, p = (refresh_halos(f) for f in (u, v, p))
        state = timestep(u, v, p, dx, dy, 2.0 * dt, uo, vo, po, alpha, MLOC, N)
    return state[0], state[1], state[2]


def run_forward_reference(u0, v0, p0, n_steps):
    """nb01 forward model on the full global domain, single process."""
    u, v, p = (make_periodic(fld(DOM_G_INT, a), M, N) for a in (u0, v0, p0))
    state = timestep(u, v, p, dx, dy, dt, u, v, p, 0.0, M, N)
    for _ in range(n_steps - 1):
        u, v, p, uo, vo, po = state
        state = timestep(u, v, p, dx, dy, 2.0 * dt, uo, vo, po, alpha, M, N)
    return state[0], state[1], state[2]


def gather_to_root(f):
    """Interior of a DOM_L_HALO field -> global (M, N) numpy array on rank 0, None elsewhere."""
    blocks = COMM.gather(np.asarray(f[DOM_L_INT].ndarray), root=0)
    return np.concatenate(blocks, axis=0) if RANK == 0 else None


def main():
    u_g, v_g, p_g = initialize_interior(np, M, N, dx, dy, radius)
    loc = slice(I0, I0 + MLOC)
    u, v, p = run_forward(u_g[loc], v_g[loc], p_g[loc], N_STEPS)
    gathered = [gather_to_root(f) for f in (u, v, p)]

    if RANK == 0:
        ref = run_forward_reference(u_g, v_g, p_g, N_STEPS)
        worst = 0.0
        for name, got, r in zip("uvp", gathered, ref):
            diff = float(np.max(np.abs(got - np.asarray(r[DOM_G_INT].ndarray))))
            worst = max(worst, diff)
            print(f"{name}: max |ghex - reference| = {diff:.3e}")
        print(f"ranks {SIZE}, M={M} N={N} MLOC={MLOC}, {N_STEPS} steps: "
              f"max abs diff {worst:.3e} -> {'PASS' if worst < 1e-9 else 'FAIL'}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
