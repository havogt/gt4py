# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Fitting a diffusivity to observations, with gradients through gt4py stencils.

A tracer is advected by a cellular flow and diffused on a doubly periodic mesh. The
finite-volume step is a gt4py field operator on the edges and cells of the mesh. Everything
around it is plain JAX: the time step from the CFL condition, the time loop, the misfit to
observations of the tracer, and its minimization with BFGS, which recovers the diffusivity
that produced the observations. The whole fit runs under one `jax.jit`, and the gradient of
the misfit comes from `jax.grad` through the stencils.

None of the JAX part can be written in gt4py.next today. Its reductions (`neighbor_sum`,
`max_over`, `min_over`) reduce a local dimension, the neighbors of each element, never the
mesh, so neither the maximum wind speed nor the misfit is a field operator. A program is a
sequence of field operator calls without loops or conditions, so neither the time loop nor
the minimization is a program. And gt4py computes no derivatives.

Run it with `python examples/next/jax_diffusivity_fit.py`.
"""

from __future__ import annotations

import functools
from types import ModuleType
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.optimize
import numpy as np

import gt4py.next as gtx
from gt4py.next import neighbor_sum, where


Cell = gtx.Dimension("Cell")
Edge = gtx.Dimension("Edge")
E2CDim = gtx.Dimension("E2C", kind=gtx.DimensionKind.LOCAL)
C2EDim = gtx.Dimension("C2E", kind=gtx.DimensionKind.LOCAL)
E2C = gtx.FieldOffset("E2C", source=Cell, target=(Edge, E2CDim))
C2E = gtx.FieldOffset("C2E", source=Edge, target=(Cell, C2EDim))

CFL = 0.5


# --- gt4py: one finite-volume step ---------------------------------------------------------


@gtx.field_operator
def normal_flux(
    q: gtx.Field[[Cell], float], vn: gtx.Field[[Edge], float], kappa: float, h: float
) -> gtx.Field[[Edge], float]:
    q_upwind = where(vn > 0.0, q(E2C[0]), q(E2C[1]))
    return vn * q_upwind - kappa * (q(E2C[1]) - q(E2C[0])) / h


@gtx.field_operator
def advect_diffuse(
    q: gtx.Field[[Cell], float],
    vn: gtx.Field[[Edge], float],
    orientation: gtx.Field[[Cell, C2EDim], float],
    kappa: float,
    dt: float,
    h: float,
) -> gtx.Field[[Cell], float]:
    flux = normal_flux(q, vn, kappa, h)
    return q - dt / h * neighbor_sum(orientation * flux(C2E), axis=C2EDim)


# --- the mesh, as an unstructured one ------------------------------------------------------


class Mesh(NamedTuple):
    """
    `n` x `n` square cells of size `h` on the doubly periodic unit square.

    Edge `k` is the east face of cell `k`, edge `n * n + k` its north face; the normal of an
    edge points from its first cell in `e2c` to its second. `orientation` is +1 where it
    points out of the cell.
    """

    c2e: gtx.Connectivity
    e2c: gtx.Connectivity
    orientation: gtx.Field
    h: float


def _cell_index(n: int, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    return (j % n) * n + i % n


def make_mesh(n: int, xp: ModuleType = jnp) -> Mesh:
    j, i = np.divmod(np.arange(n * n), n)
    cell = _cell_index(n, i, j)
    east = np.stack([cell, _cell_index(n, i + 1, j)], axis=1)
    north = np.stack([cell, _cell_index(n, i, j + 1)], axis=1)
    c2e = np.stack(
        [cell, _cell_index(n, i - 1, j), n * n + cell, n * n + _cell_index(n, i, j - 1)], axis=1
    )
    return Mesh(
        c2e=gtx.as_connectivity(
            [Cell, C2EDim], Edge, xp.asarray(c2e.astype(np.int32)), allocator=xp
        ),
        e2c=gtx.as_connectivity(
            [Edge, E2CDim],
            Cell,
            xp.asarray(np.concatenate([east, north]).astype(np.int32)),
            allocator=xp,
        ),
        orientation=gtx.as_field(
            [Cell, C2EDim], xp.asarray(np.tile([1.0, -1.0, 1.0, -1.0], (n * n, 1))), allocator=xp
        ),
        h=1.0 / n,
    )


def cellular_flow(n: int, xp: ModuleType = jnp) -> gtx.Field:
    """Normal wind on the edges, from a stream function on the vertices: free of divergence."""
    h = 1.0 / n
    j, i = np.divmod(np.arange(n * n), n)

    def psi(i: np.ndarray, j: np.ndarray) -> np.ndarray:
        return np.sin(2 * np.pi * i * h) * np.sin(2 * np.pi * j * h) / (2 * np.pi)

    vn_east = -(psi(i + 1, j + 1) - psi(i + 1, j)) / h
    vn_north = (psi(i + 1, j + 1) - psi(i, j + 1)) / h
    return gtx.as_field([Edge], xp.asarray(np.concatenate([vn_east, vn_north])), allocator=xp)


def gaussian_blob(n: int, xp: ModuleType = jnp) -> gtx.Field:
    j, i = np.divmod(np.arange(n * n), n)
    x, y = (i + 0.5) / n, (j + 0.5) / n
    return gtx.as_field(
        [Cell], xp.asarray(np.exp(-((x - 0.5) ** 2 + (y - 0.3) ** 2) / (2 * 0.1**2))), allocator=xp
    )


# --- JAX: time loop, misfit, fit -----------------------------------------------------------


def time_step(vn: gtx.Field, h: float) -> jax.Array:
    return CFL * h / jnp.max(jnp.abs(vn.ndarray))


def advance(q: gtx.Field, vn: gtx.Field, kappa: float, dt: float, mesh: Mesh) -> gtx.Field:
    q_next = gtx.empty(q.domain, allocator=q.array_ns)
    advect_diffuse(
        q,
        vn,
        mesh.orientation,
        kappa,
        dt,
        mesh.h,
        out=q_next,
        offset_provider={"C2E": mesh.c2e, "E2C": mesh.e2c},
    )
    return q_next


def forecast(q0: gtx.Field, vn: gtx.Field, kappa: float, mesh: Mesh, n_steps: int) -> gtx.Field:
    dt = time_step(vn, mesh.h)
    q, _ = jax.lax.scan(lambda q, _: (advance(q, vn, kappa, dt, mesh), None), q0, length=n_steps)
    return q


def misfit(
    kappa: float,
    q0: gtx.Field,
    vn: gtx.Field,
    mesh: Mesh,
    obs_cells: jax.Array,
    observed: jax.Array,
    n_steps: int,
) -> jax.Array:
    q = forecast(q0, vn, kappa, mesh, n_steps)
    return 0.5 * jnp.mean((q.ndarray[obs_cells] - observed) ** 2) / jnp.var(observed)


@functools.partial(jax.jit, static_argnames="n_steps")
def fit_diffusivity(
    kappa_guess: float,
    q0: gtx.Field,
    vn: gtx.Field,
    mesh: Mesh,
    obs_cells: jax.Array,
    observed: jax.Array,
    n_steps: int,
) -> tuple[jax.Array, jax.scipy.optimize.OptimizeResults]:
    """The diffusivity minimizing the misfit, searched for in its logarithm."""
    result = jax.scipy.optimize.minimize(
        lambda log_kappa: misfit(jnp.exp(log_kappa[0]), q0, vn, mesh, obs_cells, observed, n_steps),
        jnp.log(jnp.asarray([kappa_guess])),
        method="BFGS",
    )
    return jnp.exp(result.x[0]), result


if __name__ == "__main__":
    n, n_steps = 32, 64
    mesh, vn, q0 = make_mesh(n), cellular_flow(n), gaussian_blob(n)
    kappa_true, kappa_guess = 5e-3, 5e-4

    obs_cells = jnp.arange(0, n * n, 7)
    truth = jax.jit(forecast, static_argnames="n_steps")(q0, vn, kappa_true, mesh, n_steps)
    observed = truth.ndarray[obs_cells]

    kappa, result = fit_diffusivity(kappa_guess, q0, vn, mesh, obs_cells, observed, n_steps)
    print(
        f"diffusivity: true {kappa_true:.4e}, first guess {kappa_guess:.4e}, "
        f"fitted {float(kappa):.4e} in {int(result.nit)} BFGS iterations "
        f"(misfit {float(result.fun):.2e}, converged: {bool(result.success)})"
    )
