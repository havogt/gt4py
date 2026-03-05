# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Sketch: Shallow Water Model in unstructured notation on a Cartesian grid (dx=dy).

This is a simplified specialization of swm_unstructured_sketch.py that exploits
the regularity of a Cartesian quad mesh with equal grid spacing dx = dy = d.

Simplifications compared to the general unstructured version:
    1. All geometric fields collapse to a single scalar `inv_dx` (= 1/d):
       - edge_length = d for every edge
       - dual_edge_length = d for every edge (dual of x-edge has length dy=d)
       - cell_area = dual_cell_area = d²
       ==> div, curl, grad all simplify to: inv_dx * neighbor_sum(sign * field)

    2. Kinetic energy weights become a constant 0.25 (4 edges per vertex).

    3. TRiSK reconstruction weights have constant magnitude 0.25; only the
       sign varies between x-edges (+1) and y-edges (-1).  This means the
       full Field[[Edge, E2E_TRiSKDim], float] weight array reduces to
       a simple EdgeField `trisk_sign` (= +1 or -1 per edge).

    What does NOT simplify (topological, not geometric):
    - The connectivities themselves (E2V, V2E, E2C, C2E, C2V, E2E_TRiSK)
    - sign_v2e (divergence orientation)
    - sign_c2e (curl orientation)
    - trisk_sign (edge-type sign for Coriolis)

    Net effect: The timestep function goes from 13 field parameters + 2 scalars
    to 4 field parameters + 2 scalars (plus the connectivities, which are the
    same in both versions).  Each operator becomes a one-liner.
"""

import gt4py.next as gtx
from gt4py.next import neighbor_sum
import numpy as np

# ============================================================================
# 1. Dimensions and connectivities  (same as general version)
# ============================================================================

Vertex = gtx.Dimension("Vertex")
Edge = gtx.Dimension("Edge")
Cell = gtx.Dimension("Cell")

E2VDim = gtx.Dimension("E2V", kind=gtx.DimensionKind.LOCAL)
V2EDim = gtx.Dimension("V2E", kind=gtx.DimensionKind.LOCAL)
E2CDim = gtx.Dimension("E2C", kind=gtx.DimensionKind.LOCAL)
C2EDim = gtx.Dimension("C2E", kind=gtx.DimensionKind.LOCAL)
C2VDim = gtx.Dimension("C2V", kind=gtx.DimensionKind.LOCAL)
E2E_TRiSKDim = gtx.Dimension("E2E_TRiSK", kind=gtx.DimensionKind.LOCAL)

E2V = gtx.FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))
V2E = gtx.FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
E2C = gtx.FieldOffset("E2C", source=Cell, target=(Edge, E2CDim))
C2E = gtx.FieldOffset("C2E", source=Edge, target=(Cell, C2EDim))
C2V = gtx.FieldOffset("C2V", source=Vertex, target=(Cell, C2VDim))
E2E_TRiSK = gtx.FieldOffset("E2E_TRiSK", source=Edge, target=(Edge, E2E_TRiSKDim))

VertexField = gtx.Field[[Vertex], float]
EdgeField = gtx.Field[[Edge], float]
CellField = gtx.Field[[Cell], float]


# ============================================================================
# 2. Simplified building-block operators (dx = dy)
# ============================================================================
#
# For dx = dy = d, all geometric factors are the same for every edge/cell/vertex:
#     edge_length = d              (all edges)
#     dual_edge_length = d         (all dual edges)
#     cell_area = d²               (all cells)
#     dual_cell_area = d²          (all dual cells / vertex volumes)
#
# This means:
#     div  = (1/d²) * Σ sign * flux * d = (1/d) * Σ sign * flux
#     curl = (1/d²) * Σ sign * vel  * d = (1/d) * Σ sign * vel
#     grad = (1/d) * (p₁ - p₀)
#
# Compare with the general version where these factors are per-element fields.


@gtx.field_operator
def avg_e2v(p: VertexField) -> EdgeField:
    """Average vertex values to edge (Vertex -> Edge).  Unchanged from general."""
    return 0.5 * (p(E2V[0]) + p(E2V[1]))


@gtx.field_operator
def grad_e(p: VertexField, inv_dx: float) -> EdgeField:
    """Gradient along each edge (Vertex -> Edge).

    General version: inv_edge_length(e) * (p(E2V[1]) - p(E2V[0]))
    Cartesian:       inv_dx            * (p(E2V[1]) - p(E2V[0]))
                     ^^^^^^ scalar replaces per-edge field
    """
    return inv_dx * (p(E2V[1]) - p(E2V[0]))


@gtx.field_operator
def div_v(
    flux: EdgeField,
    sign_v2e: gtx.Field[[Vertex, V2EDim], float],
    inv_dx: float,
) -> VertexField:
    """Divergence at vertices (Edge -> Vertex).

    General version:
        neighbor_sum(sign * flux(V2E) * dual_edge_length(V2E)) * inv_dual_area
    Cartesian (dual_edge_length = d, inv_dual_area = 1/d²):
        (1/d) * neighbor_sum(sign * flux(V2E))

    Proof: (1/d²) * Σ sign * flux * d = (1/d) * Σ sign * flux  ✓
    """
    return inv_dx * neighbor_sum(sign_v2e * flux(V2E), axis=V2EDim)


@gtx.field_operator
def curl_c(
    vel: EdgeField,
    sign_c2e: gtx.Field[[Cell, C2EDim], float],
    inv_dx: float,
) -> CellField:
    """Curl at cell centers (Edge -> Cell).

    General version:
        neighbor_sum(sign * vel(C2E) * edge_length(C2E)) * inv_cell_area
    Cartesian (edge_length = d, inv_cell_area = 1/d²):
        (1/d) * neighbor_sum(sign * vel(C2E))

    Proof: (1/d²) * Σ sign * vel * d = (1/d) * Σ sign * vel  ✓
    """
    return inv_dx * neighbor_sum(sign_c2e * vel(C2E), axis=C2EDim)


@gtx.field_operator
def avg_c2v(p: VertexField) -> CellField:
    """Average vertex values to cell (Vertex -> Cell).  Unchanged."""
    return 0.25 * neighbor_sum(p(C2V), axis=C2VDim)


@gtx.field_operator
def avg_e2c(z: CellField) -> EdgeField:
    """Average cell values to edge (Cell -> Edge).  Unchanged."""
    return 0.5 * (z(E2C[0]) + z(E2C[1]))


# ============================================================================
# 3. Simplified Coriolis term (dx = dy)
# ============================================================================
#
# In the general version, trisk_weights is a Field[[Edge, E2E_TRiSKDim], float]
# with geometry-dependent entries.
#
# On a Cartesian grid (dx = dy), all TRiSK weights have the SAME magnitude 0.25.
# Only the sign varies: +1 for x-edges, -1 for y-edges.  Crucially, this sign
# is constant per edge (not per neighbor slot), so it factors out of the sum:
#
#   General:   Σ_e' trisk_weights(e,e') * flux(e')
#   Cartesian: trisk_sign(e) * 0.25 * Σ_e' flux(e')
#                              ^^^^   ^^^^ simple unweighted sum
#                              const   all neighbors equally weighted
#
# This replaces a Field[[Edge, E2E_TRiSKDim], float] with a single EdgeField.
#
# The sign encodes the cross-product orientation (curl of velocity):
#   x-edges carry the u-velocity → Coriolis from v-fluxes → +1
#   y-edges carry the v-velocity → Coriolis from u-fluxes → -1


@gtx.field_operator
def coriolis(
    z: CellField,
    flux: EdgeField,
    trisk_sign: EdgeField,   # +1 for x-edges, -1 for y-edges
) -> EdgeField:
    """Coriolis acceleration at each edge (Cartesian simplified).

    General version:
        avg_e2c(z) * neighbor_sum(trisk_weights * flux(E2E_TRiSK))
    Cartesian:
        avg_e2c(z) * trisk_sign * 0.25 * neighbor_sum(flux(E2E_TRiSK))

    For x-edge at (i+1/2, j), trisk_sign = +1:
        result = 0.5*(z_below + z_above) * (+1) * 0.25*(cv_NE+cv_NW+cv_SE+cv_SW)
               = avg_y_staggered(z) * avg_y_staggered(avg_x(cv))
        This matches the structured u-equation Coriolis term.  ✓

    For y-edge at (i, j+1/2), trisk_sign = -1:
        result = 0.5*(z_left + z_right) * (-1) * 0.25*(cu_NE+cu_NW+cu_SE+cu_SW)
               = -avg_x_staggered(z) * avg_x_staggered(avg_y(cu))
        This matches the structured v-equation Coriolis term.  ✓
    """
    z_at_edge = avg_e2c(z)
    perp_flux = 0.25 * neighbor_sum(flux(E2E_TRiSK), axis=E2E_TRiSKDim)
    return z_at_edge * trisk_sign * perp_flux


# ============================================================================
# 4. The complete Cartesian-simplified timestep
# ============================================================================


@gtx.field_operator
def timestep_cartesian(
    # --- Prognostic fields ---
    vel: EdgeField,              # normal velocity on all edges (merges u, v)
    p: VertexField,              # pressure/height at vertices
    vel_old: EdgeField,
    p_old: VertexField,
    # --- Sign / orientation fields (topological, still needed) ---
    sign_v2e: gtx.Field[[Vertex, V2EDim], float],   # for divergence
    sign_c2e: gtx.Field[[Cell, C2EDim], float],      # for curl
    trisk_sign: EdgeField,                            # +1 x-edges, -1 y-edges
    # --- Scalars ---
    inv_dx: float,               # 1/dx  (replaces 5 geometric fields!)
    dt: float,
    alpha: float,
) -> tuple[EdgeField, VertexField, EdgeField, VertexField]:
    """One timestep of the SWM on a Cartesian grid in unstructured notation.

    Compare with the general unstructured version which takes:
        inv_edge_length, edge_length, dual_edge_length,
        inv_cell_area, inv_dual_area, ke_weights, trisk_weights
    All replaced here by the single scalar inv_dx.

    Structured equivalent (from operators.py, lines 78-120):
        cu = avg_x(p) * u
        cv = avg_y(p) * v
        z = (delta_x(v) - delta_y(u)) / avg_x(avg_y(p))
        h = p + 0.5 * (avg_x_staggered(u*u) + avg_y_staggered(v*v))
        unew = uold + avg_y_staggered(z) * avg_y_staggered(avg_x(cv)) * dt
                    - delta_x(dx, h) * dt
        vnew = vold - avg_x_staggered(z) * avg_x_staggered(avg_y(cu)) * dt
                    - delta_y(dy, h) * dt
        pnew = pold - delta_x_staggered(dx, cu) * dt
                    - delta_y_staggered(dy, cv) * dt
    """

    # --- Step 1: Mass flux at edges ---
    # cu = avg_x(p) * u;  cv = avg_y(p) * v  →  unified
    flux = avg_e2v(p) * vel

    # --- Step 2: Vorticity at cells ---
    # z = (delta_x(v) - delta_y(u)) / avg_x(avg_y(p))
    # = curl(vel) / avg(p)
    z = curl_c(vel, sign_c2e, inv_dx) / avg_c2v(p)

    # --- Step 3: Bernoulli function at vertices ---
    # h = p + 0.5 * (avg_x_staggered(u²) + avg_y_staggered(v²))
    # = p + 0.25 * Σ vel²  (4 edges per vertex, equal weights)
    h = p + 0.25 * neighbor_sum(vel(V2E) * vel(V2E), axis=V2EDim)

    # --- Step 4: Coriolis (momentum equation RHS, vorticity term) ---
    coriolis_acc = coriolis(z, flux, trisk_sign)

    # --- Step 5: Pressure gradient ---
    grad_h = grad_e(h, inv_dx)

    # --- Step 6: Velocity update ---
    vel_new = vel_old + coriolis_acc * dt - grad_h * dt

    # --- Step 7: Pressure update ---
    p_new = p_old - div_v(flux, sign_v2e, inv_dx) * dt

    # --- Step 8: Asselin time filter ---
    vel_old_new = vel + alpha * (vel_new - 2.0 * vel + vel_old)
    p_old_new = p + alpha * (p_new - 2.0 * p + p_old)

    return vel_new, p_new, vel_old_new, p_old_new


# ============================================================================
# 5. Side-by-side comparison: structured vs Cartesian-unstructured
# ============================================================================
#
# The table below maps each line of the structured timestep (operators.py:92-103)
# to its Cartesian-unstructured equivalent, showing how the code reads almost
# as concisely once the geometric constants are factored out.
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │  Structured (operators.py)          │  Cartesian unstructured              │
# ├─────────────────────────────────────┼──────────────────────────────────────┤
# │  cu = avg_x(p) * u                 │                                      │
# │  cv = avg_y(p) * v                 │  flux = avg_e2v(p) * vel             │
# ├─────────────────────────────────────┼──────────────────────────────────────┤
# │  z = (delta_x(dx,v) - delta_y(dy,u)│  z = curl_c(vel, sign_c2e, inv_dx)  │
# │      / avg_x(avg_y(p))             │      / avg_c2v(p)                    │
# ├─────────────────────────────────────┼──────────────────────────────────────┤
# │  h = p + 0.5*(avg_x_stag(u*u)      │  h = p + 0.25 * neighbor_sum(       │
# │            + avg_y_stag(v*v))       │          vel(V2E)**2, axis=V2EDim)  │
# ├─────────────────────────────────────┼──────────────────────────────────────┤
# │  unew = uold                        │                                      │
# │    + avg_y_stag(z)*avg_y_stag(      │                                      │
# │        avg_x(cv))*dt                │  vel_new = vel_old                   │
# │    - delta_x(dx,h)*dt               │    + coriolis(z,flux,trisk_sign)*dt  │
# │  vnew = vold                        │    - grad_e(h, inv_dx) * dt          │
# │    - avg_x_stag(z)*avg_x_stag(      │                                      │
# │        avg_y(cu))*dt                │                                      │
# │    - delta_y(dy,h)*dt               │                                      │
# ├─────────────────────────────────────┼──────────────────────────────────────┤
# │  pnew = pold                        │  p_new = p_old                       │
# │    - delta_x_stag(dx,cu)*dt         │    - div_v(flux,sign_v2e,inv_dx)*dt  │
# │    - delta_y_stag(dy,cv)*dt         │                                      │
# └─────────────────────────────────────┴──────────────────────────────────────┘
#
# The unstructured version is actually MORE concise because:
# 1. u and v equations merge into one (vel_new = ...)
# 2. cu and cv merge into one (flux = ...)
# 3. delta_x_stag + delta_y_stag merge into div_v(...)
# 4. delta_x(v) - delta_y(u) merges into curl_c(...)
#
# The only "cost" is the sign fields and the TRiSK connectivity for Coriolis,
# but these are precomputed once from the mesh topology.


# ============================================================================
# 6. Mesh setup helper
# ============================================================================
#
# Reuses build_quad_mesh_connectivities from swm_unstructured_sketch.py.
# The Cartesian version only needs a subset of what it returns:

def build_cartesian_fields(M: int, N: int, dx: float):
    """Build the fields needed for the Cartesian-simplified timestep.

    Returns GT4Py fields ready to pass to timestep_cartesian.
    """
    from swm_unstructured_sketch import build_quad_mesh_connectivities

    mesh = build_quad_mesh_connectivities(M, N, dx, dx)

    # Connectivities
    e2v_conn = gtx.as_connectivity(
        domain={Edge: mesh["num_edges"], E2VDim: 2},
        codomain=Vertex, data=mesh["e2v"],
    )
    v2e_conn = gtx.as_connectivity(
        domain={Vertex: mesh["num_vertices"], V2EDim: 4},
        codomain=Edge, data=mesh["v2e"],
    )
    e2c_conn = gtx.as_connectivity(
        domain={Edge: mesh["num_edges"], E2CDim: 2},
        codomain=Cell, data=mesh["e2c"],
    )
    c2e_conn = gtx.as_connectivity(
        domain={Cell: mesh["num_cells"], C2EDim: 4},
        codomain=Edge, data=mesh["c2e"],
    )
    c2v_conn = gtx.as_connectivity(
        domain={Cell: mesh["num_cells"], C2VDim: 4},
        codomain=Vertex, data=mesh["c2v"],
    )
    e2e_trisk_conn = gtx.as_connectivity(
        domain={Edge: mesh["num_edges"], E2E_TRiSKDim: 4},
        codomain=Edge, data=mesh["e2e_trisk"],
    )

    # Sign fields (topological — cannot be simplified away)
    sign_v2e = gtx.as_field([Vertex, V2EDim], mesh["sign_v2e"])
    sign_c2e = gtx.as_field([Cell, C2EDim], mesh["sign_c2e"])

    # TRiSK sign: +1 for x-edges (first M*N), -1 for y-edges (last M*N)
    trisk_sign_arr = np.ones(mesh["num_edges"], dtype=np.float64)
    trisk_sign_arr[M * N:] = -1.0
    trisk_sign = gtx.as_field([Edge], trisk_sign_arr)

    return {
        "sign_v2e": sign_v2e,
        "sign_c2e": sign_c2e,
        "trisk_sign": trisk_sign,
        "inv_dx": 1.0 / dx,
        "offset_provider": {
            "E2V": e2v_conn,
            "V2E": v2e_conn,
            "E2C": e2c_conn,
            "C2E": c2e_conn,
            "C2V": c2v_conn,
            "E2E_TRiSK": e2e_trisk_conn,
        },
    }
