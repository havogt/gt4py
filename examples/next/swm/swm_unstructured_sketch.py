# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Sketch: Shallow Water Model on an unstructured mesh using GT4Py.

This maps the Sadourny (1975) Arakawa C-grid scheme from the structured
Cartesian implementation (operators.py / swm.py) to unstructured
Vertex/Edge/Cell notation.

Mapping of grid point types:
    Structured (C-grid)         Unstructured
    -----------------------------------------
    p-point  at (i, j)         Vertex       — pressure/height
    u-point  at (i+1/2, j)    Edge         — velocity (x-component on structured)
    v-point  at (i, j+1/2)    Edge         — velocity (y-component on structured)
    zeta-point at (i+1/2,j+1/2) Cell       — vorticity

Key insight: On a structured grid u and v are separate fields on x-edges and
y-edges respectively.  On an unstructured mesh they merge into a single field
`vel` on all edges, representing the velocity component *along the edge
direction* (from E2V[0] toward E2V[1]).

Required connectivities:
    E2V  — Edge to its 2 Vertices    (for gradient, edge-averaging)
    V2E  — Vertex to its Edges       (for divergence, kinetic energy)
    E2C  — Edge to its 2 Cells       (for averaging vorticity to edges)
    C2E  — Cell to its Edges         (for curl / vorticity)
    C2V  — Cell to its Vertices      (for averaging p to cells)

Required sign fields:
    sign_v2e — Field[[Vertex, V2EDim], float]
        For divergence: +1 if edge points away from vertex, -1 if toward.
        Constructed as: +1 when vertex == E2V(edge, 0), else -1.
    sign_c2e — Field[[Cell, C2EDim], float]
        For curl: +1 if edge tangent aligns with CCW traversal of the cell
        boundary, -1 otherwise.

The Coriolis/vorticity term in the momentum equation is the hardest part.
On a structured grid the "perpendicular flux" reconstruction is implicit
(x-edges see y-fluxes and vice versa).  On an unstructured mesh this
requires the TRiSK tangential reconstruction (Thuburn et al., 2009), which
introduces an additional Edge-to-Edge connectivity with geometric weights.
For a regular quad mesh the TRiSK weights reduce to simple 1/4 factors and
the scheme reproduces Sadourny exactly.
"""

import gt4py.next as gtx
from gt4py.next import neighbor_sum
import numpy as np


# ============================================================================
# 1. Dimensions
# ============================================================================

Vertex = gtx.Dimension("Vertex")
Edge = gtx.Dimension("Edge")
Cell = gtx.Dimension("Cell")

# Local (connectivity-table) dimensions
E2VDim = gtx.Dimension("E2V", kind=gtx.DimensionKind.LOCAL)   # always 2
V2EDim = gtx.Dimension("V2E", kind=gtx.DimensionKind.LOCAL)   # 4 for quad mesh
E2CDim = gtx.Dimension("E2C", kind=gtx.DimensionKind.LOCAL)   # always 2
C2EDim = gtx.Dimension("C2E", kind=gtx.DimensionKind.LOCAL)   # 4 for quad mesh
C2VDim = gtx.Dimension("C2V", kind=gtx.DimensionKind.LOCAL)   # 4 for quad mesh

# TRiSK: edge-to-edge connectivity for tangential reconstruction
E2E_TRiSKDim = gtx.Dimension("E2E_TRiSK", kind=gtx.DimensionKind.LOCAL)

# Connectivity offsets
E2V = gtx.FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))
V2E = gtx.FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
E2C = gtx.FieldOffset("E2C", source=Cell, target=(Edge, E2CDim))
C2E = gtx.FieldOffset("C2E", source=Edge, target=(Cell, C2EDim))
C2V = gtx.FieldOffset("C2V", source=Vertex, target=(Cell, C2VDim))
E2E_TRiSK = gtx.FieldOffset("E2E_TRiSK", source=Edge, target=(Edge, E2E_TRiSKDim))


# ============================================================================
# 2. Type aliases
# ============================================================================

VertexField = gtx.Field[[Vertex], float]
EdgeField = gtx.Field[[Edge], float]
CellField = gtx.Field[[Cell], float]


# ============================================================================
# 3. Building-block operators
# ============================================================================


@gtx.field_operator
def avg_e2v(p: VertexField) -> EdgeField:
    """Average vertex values to edge midpoints (Vertex -> Edge).

    Structured equivalents:
        avg_x(p)  for x-edges
        avg_y(p)  for y-edges

    Proof for structured quad mesh:
        For x-edge from vertex (i,j) to (i+1,j):
            E2V[0] = (i,j),  E2V[1] = (i+1,j)
            result = 0.5 * (p[i,j] + p[i+1,j]) = avg_x(p)[i,j]  OK
    """
    return 0.5 * (p(E2V[0]) + p(E2V[1]))


@gtx.field_operator
def grad_e(p: VertexField, inv_edge_length: EdgeField) -> EdgeField:
    """Normal gradient at edge midpoints (Vertex -> Edge).

    Structured equivalents:
        delta_x(dx, p)  for x-edges
        delta_y(dy, p)  for y-edges

    The direction is from E2V[0] toward E2V[1].

    Proof for structured quad mesh:
        For x-edge from (i,j) to (i+1,j), inv_edge_length = 1/dx:
            result = (1/dx) * (p[i+1,j] - p[i,j]) = delta_x(dx, p)[i,j]  OK
        For y-edge from (i,j) to (i,j+1), inv_edge_length = 1/dy:
            result = (1/dy) * (p[i,j+1] - p[i,j]) = delta_y(dy, p)[i,j]  OK
    """
    return inv_edge_length * (p(E2V[1]) - p(E2V[0]))


@gtx.field_operator
def div_v(
    flux: EdgeField,
    sign_v2e: gtx.Field[[Vertex, V2EDim], float],
    dual_edge_length: EdgeField,
    inv_dual_area: VertexField,
) -> VertexField:
    """Divergence of edge fluxes at vertices (Edge -> Vertex).

    Structured equivalent:
        delta_x_staggered(dx, cu) + delta_y_staggered(dy, cv)

    sign_v2e[v, local_e] = +1 if edge points away from v (v == E2V[edge, 0]),
                           -1 if edge points toward v  (v == E2V[edge, 1]).

    Proof for structured quad mesh (dx = dy, dual_area = dx*dy):
        Vertex (i,j) has 4 edges:
            East  (i,j)->(i+1,j): v == E2V[0], sign = +1, flux = cu[i,j],   dual_len = dy
            West  (i-1,j)->(i,j): v == E2V[1], sign = -1, flux = cu[i-1,j], dual_len = dy
            North (i,j)->(i,j+1): v == E2V[0], sign = +1, flux = cv[i,j],   dual_len = dx
            South (i,j-1)->(i,j): v == E2V[1], sign = -1, flux = cv[i,j-1], dual_len = dx

        result = 1/(dx*dy) * [(+1)*cu[i,j]*dy + (-1)*cu[i-1,j]*dy
                             + (+1)*cv[i,j]*dx + (-1)*cv[i,j-1]*dx]
               = (cu[i,j] - cu[i-1,j])/dx + (cv[i,j] - cv[i,j-1])/dy
               = delta_x_staggered(dx, cu) + delta_y_staggered(dy, cv)  OK
    """
    return (
        neighbor_sum(sign_v2e * flux(V2E) * dual_edge_length(V2E), axis=V2EDim)
        * inv_dual_area
    )


@gtx.field_operator
def curl_c(
    vel: EdgeField,
    sign_c2e: gtx.Field[[Cell, C2EDim], float],
    edge_length: EdgeField,
    inv_cell_area: CellField,
) -> CellField:
    """Curl of velocity at cell centers (Edge -> Cell).

    Structured equivalent:
        delta_x(dx, v) - delta_y(dy, u)

    sign_c2e[c, local_e] = +1 if the edge's E2V direction matches counter-
    clockwise traversal around cell c, -1 otherwise.

    Proof for structured quad mesh (dx = dy, cell_area = dx*dy):
        Cell at (i+1/2, j+1/2) has 4 edges, CCW from bottom:
            Bottom (i,j)->(i+1,j):     edge_dir = rightward
                CCW at bottom = rightward -> sign = +1
                vel = u[i,j], edge_length = dx
            Right  (i+1,j)->(i+1,j+1): edge_dir = upward
                CCW at right = upward -> sign = +1
                vel = v[i+1,j], edge_length = dy
            Top    (i,j+1)->(i+1,j+1): edge_dir = rightward
                CCW at top = leftward -> sign = -1
                vel = u[i,j+1], edge_length = dx
            Left   (i,j)->(i,j+1):     edge_dir = upward
                CCW at left = downward -> sign = -1
                vel = v[i,j], edge_length = dy

        result = 1/(dx*dy) * [(+1)*u[i,j]*dx + (+1)*v[i+1,j]*dy
                             + (-1)*u[i,j+1]*dx + (-1)*v[i,j]*dy]
               = (v[i+1,j] - v[i,j])/dx - (u[i,j+1] - u[i,j])/dy
               = delta_x(dx, v) - delta_y(dy, u)  OK
    """
    return (
        neighbor_sum(sign_c2e * vel(C2E) * edge_length(C2E), axis=C2EDim)
        * inv_cell_area
    )


@gtx.field_operator
def avg_c2v(p: VertexField) -> CellField:
    """Average vertex values around a cell (Vertex -> Cell).

    Structured equivalent:
        avg_x(avg_y(p))

    Proof for quad mesh (4 vertices per cell):
        Cell (i+1/2, j+1/2) has vertices (i,j), (i+1,j), (i,j+1), (i+1,j+1):
            result = 0.25 * (p[i,j] + p[i+1,j] + p[i,j+1] + p[i+1,j+1])
                   = avg_x(avg_y(p))[i,j]  OK
    """
    # 0.25 is specific to quad meshes; generalize with a weight field if needed.
    return 0.25 * neighbor_sum(p(C2V), axis=C2VDim)


@gtx.field_operator
def kinetic_energy_v(
    vel: EdgeField,
    ke_weights: gtx.Field[[Vertex, V2EDim], float],
) -> VertexField:
    """Kinetic energy at vertices from edge velocities (Edge -> Vertex).

    Structured equivalent:
        0.5 * (avg_x_staggered(u*u) + avg_y_staggered(v*v))

    ke_weights encodes the geometric averaging.  For a regular quad mesh
    with 4 edges per vertex, ke_weights = 0.25 everywhere, giving:

        KE = sum(0.25 * vel_e^2 for all 4 edges)
           = 0.25 * (u_E^2 + u_W^2 + v_N^2 + v_S^2)

    Note the factor 0.5 in the Bernoulli function is applied separately.

    Proof for structured quad mesh:
        avg_x_staggered(u*u)[i,j] = 0.5 * (u[i-1,j]^2 + u[i,j]^2)
        avg_y_staggered(v*v)[i,j] = 0.5 * (v[i,j-1]^2 + v[i,j]^2)

        Sum = 0.5 * (u_W^2 + u_E^2 + v_S^2 + v_N^2)

        With ke_weights = 0.25:
            result = 0.25*(u_E^2 + u_W^2 + v_N^2 + v_S^2)
                   = 0.5 * (0.5*(u_W^2 + u_E^2) + 0.5*(v_S^2 + v_N^2))
                   = 0.5 * (avg_x_staggered(u*u) + avg_y_staggered(v*v))  OK

    The extra factor of 0.5 comes from the Bernoulli function definition:
    h = p + 0.5 * (avg_x_staggered(u*u) + avg_y_staggered(v*v))
      = p + 0.5 * 2 * KE_here  => h = p + KE_here  ... which seems wrong.

    Let's redo: with ke_weights = 0.5 (= 1/num_edges_per_direction):
        result = 0.5*(u_E^2 + u_W^2 + v_N^2 + v_S^2)
    Then h = p + 0.5 * result = p + 0.25*(u_E^2 + u_W^2 + v_N^2 + v_S^2)
    But the structured code gives:
        h = p + 0.5*(0.5*(u_W^2+u_E^2) + 0.5*(v_S^2+v_N^2))
          = p + 0.25*(u_W^2+u_E^2+v_S^2+v_N^2)
    So we need: h = p + 0.5 * neighbor_sum(ke_weights * vel^2) with ke_weights=0.5.
    Alternatively: h = p + neighbor_sum(ke_weights * vel^2) with ke_weights=0.25.
    We use the latter:
    """
    return neighbor_sum(ke_weights * vel(V2E) * vel(V2E), axis=V2EDim)


# ============================================================================
# 4. The Coriolis / vorticity term (the hard part)
# ============================================================================
#
# Structured code for the u-equation:
#     unew = uold + avg_y_staggered(z) * avg_y_staggered(avg_x(cv)) * dt
#                 - delta_x(dx, h) * dt
#
# At x-edge (i+1/2, j):
#     avg_y_staggered(z) = 0.5 * (z[i,j-1] + z[i,j])
#         -> average vorticity over the 2 cells adjacent to this edge
#     avg_x(cv) at cell (i+1/2, j+1/2) = 0.5 * (cv[i,j] + cv[i+1,j])
#         -> average of 2 y-edge fluxes of the cell
#     avg_y_staggered(avg_x(cv)) -> average those cell values over 2 cells
#         -> effectively averages over 4 perpendicular edge fluxes
#
# Similarly for v-edge:
#     vnew = vold - avg_x_staggered(z) * avg_x_staggered(avg_y(cu)) * dt
#                 - delta_y(dy, h) * dt
#
# On an unstructured mesh, this becomes the TRiSK reconstruction.
# For each edge e, we reconstruct the "perpendicular flux" from fluxes on
# neighboring edges using geometric weights:
#
#     perp_flux(e) = sum_{e' in TRiSK_neighbors(e)} w(e,e') * flux(e')
#
# The vorticity is averaged from cells to edges:
#     z_edge(e) = 0.5 * (z(E2C[0]) + z(E2C[1]))
#
# Then: coriolis(e) = z_edge(e) * perp_flux(e)
#
# For a regular quad mesh, TRiSK neighbors of an x-edge are the 4 y-edges
# of its 2 adjacent cells (and vice versa), each with weight 0.25.
# This reproduces Sadourny's enstrophy-conserving scheme exactly.
#
# For the full enstrophy-conserving form (Ringler et al., 2010), the
# vorticity is coupled inside the sum:
#     coriolis(e) = sum_{e'} w(e,e') * z_at_e'(e,e') * flux(e')
# where z_at_e'(e,e') is the vorticity averaged over cells shared by e and e'.


@gtx.field_operator
def avg_e2c(z: CellField) -> EdgeField:
    """Average cell values to edge midpoints (Cell -> Edge).

    For each edge, average the values from the two adjacent cells.

    Structured equivalent at x-edge: avg_y_staggered(z)
    Structured equivalent at y-edge: avg_x_staggered(z)
    """
    return 0.5 * (z(E2C[0]) + z(E2C[1]))


@gtx.field_operator
def trisk_reconstruct(
    flux: EdgeField,
    trisk_weights: gtx.Field[[Edge, E2E_TRiSKDim], float],
) -> EdgeField:
    """TRiSK tangential reconstruction (Edge -> Edge).

    Reconstructs the tangential/perpendicular flux at each edge from
    normal fluxes on neighboring edges using precomputed geometric weights.

    For a regular quad mesh (dx = dy):
        An x-edge has 4 TRiSK neighbors (the 4 y-edges of its 2 cells),
        each with weight 0.25:
            result = 0.25 * (cv_NE + cv_SE + cv_NW + cv_SW)

        But the structured code computes:
            avg_y_staggered(avg_x(cv)):
                avg_x(cv) at cell above = 0.5*(cv_right_above + cv_left_above)
                avg_x(cv) at cell below = 0.5*(cv_right_below + cv_left_below)
                avg_y_staggered = 0.5*(cell_above + cell_below)
                = 0.25*(cv_NE + cv_NW + cv_SE + cv_SW)  OK

    Proof:
        For x-edge at (i+1/2, j):
            Cell above: (i+1/2, j+1/2), y-edges: cv[i,j] and cv[i+1,j]
            Cell below: (i+1/2, j-1/2), y-edges: cv[i,j-1] and cv[i+1,j-1]
            avg_x(cv) at cell above = 0.5*(cv[i,j] + cv[i+1,j])
            avg_x(cv) at cell below = 0.5*(cv[i,j-1] + cv[i+1,j-1])
            avg_y_staggered = 0.5*(above + below)
                = 0.25*(cv[i,j] + cv[i+1,j] + cv[i,j-1] + cv[i+1,j-1])
            = 0.25 * sum of 4 perpendicular-edge fluxes  OK
    """
    return neighbor_sum(trisk_weights * flux(E2E_TRiSK), axis=E2E_TRiSKDim)


@gtx.field_operator
def coriolis(
    z: CellField,
    flux: EdgeField,
    trisk_weights: gtx.Field[[Edge, E2E_TRiSKDim], float],
) -> EdgeField:
    """Coriolis acceleration at each edge.

    This is the simple form: z_avg(e) * perp_flux(e).
    The enstrophy-conserving form would couple z inside the TRiSK sum.

    Structured equivalent:
        For x-edges: +avg_y_staggered(z) * avg_y_staggered(avg_x(cv))
        For y-edges: -avg_x_staggered(z) * avg_x_staggered(avg_y(cu))

    Note the sign difference between u and v equations:  On the structured
    grid the u-equation has +z*cv_perp and the v-equation has -z*cu_perp.
    On the unstructured grid this sign is absorbed into the TRiSK weights
    (which are signed, encoding the cross-product orientation).
    """
    z_at_edge = avg_e2c(z)
    perp_flux = trisk_reconstruct(flux, trisk_weights)
    return z_at_edge * perp_flux


# ============================================================================
# 5. The complete unstructured timestep
# ============================================================================


@gtx.field_operator
def timestep_unstructured(
    # Prognostic fields
    vel: EdgeField,                # normal velocity at all edges (merges u and v)
    p: VertexField,                # pressure/height at vertices
    vel_old: EdgeField,
    p_old: VertexField,
    # Geometric fields (precomputed from mesh)
    inv_edge_length: EdgeField,    # 1/edge_length  (for gradient)
    edge_length: EdgeField,        # primal edge length (for curl)
    dual_edge_length: EdgeField,   # dual edge length (for divergence)
    inv_cell_area: CellField,      # 1/cell_area (for curl)
    inv_dual_area: VertexField,    # 1/dual_cell_area (for divergence)
    ke_weights: gtx.Field[[Vertex, V2EDim], float],   # kinetic energy weights
    # Sign/orientation fields (precomputed from mesh + E2V convention)
    sign_v2e: gtx.Field[[Vertex, V2EDim], float],     # divergence orientation
    sign_c2e: gtx.Field[[Cell, C2EDim], float],        # curl orientation
    # TRiSK reconstruction weights
    trisk_weights: gtx.Field[[Edge, E2E_TRiSKDim], float],
    # Scalars
    dt: float,
    alpha: float,
) -> tuple[EdgeField, VertexField, EdgeField, VertexField]:
    """One timestep of the shallow water equations on an unstructured mesh.

    Structured equivalent (from operators.py):
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
    # Structured: cu = avg_x(p) * u,  cv = avg_y(p) * v
    # Unified: both become the same operation on all edges.
    flux = avg_e2v(p) * vel

    # --- Step 2: Vorticity at cell centers ---
    # Structured: z = (delta_x(v) - delta_y(u)) / avg_x(avg_y(p))
    # Unstructured: curl of velocity / averaged pressure at cells
    raw_curl = curl_c(vel, sign_c2e, edge_length, inv_cell_area)
    z = raw_curl / avg_c2v(p)

    # --- Step 3: Bernoulli function at vertices ---
    # Structured: h = p + 0.5 * (avg_x_staggered(u*u) + avg_y_staggered(v*v))
    # Unstructured: h = p + KE(vel)  where KE uses geometric weights
    h = p + kinetic_energy_v(vel, ke_weights)

    # --- Step 4: Coriolis term at edges ---
    # Structured u-eqn: + avg_y_staggered(z) * avg_y_staggered(avg_x(cv)) * dt
    # Structured v-eqn: - avg_x_staggered(z) * avg_x_staggered(avg_y(cu)) * dt
    # Unstructured: TRiSK reconstruction absorbs both cases + sign into weights
    coriolis_acc = coriolis(z, flux, trisk_weights)

    # --- Step 5: Pressure gradient at edges ---
    # Structured: delta_x(dx, h) for x-edges,  delta_y(dy, h) for y-edges
    # Unstructured: gradient along each edge
    grad_h = grad_e(h, inv_edge_length)

    # --- Step 6: Velocity update (momentum equation) ---
    # Structured:
    #   unew = uold + coriolis_x * dt - delta_x(dx, h) * dt
    #   vnew = vold + coriolis_y * dt - delta_y(dy, h) * dt
    # Unstructured: single equation for all edges
    vel_new = vel_old + coriolis_acc * dt - grad_h * dt

    # --- Step 7: Pressure update (continuity equation) ---
    # Structured: pnew = pold - (delta_x_staggered(cu) + delta_y_staggered(cv)) * dt
    # Unstructured: divergence of mass flux
    div_flux = div_v(flux, sign_v2e, dual_edge_length, inv_dual_area)
    p_new = p_old - div_flux * dt

    # --- Step 8: Asselin time filter ---
    vel_old_new = vel + alpha * (vel_new - 2.0 * vel + vel_old)
    p_old_new = p + alpha * (p_new - 2.0 * p + p_old)

    return vel_new, p_new, vel_old_new, p_old_new


# ============================================================================
# 6. Mesh construction for a periodic quad grid
# ============================================================================
# Below is the connectivity setup that makes a regular M x N periodic quad
# mesh look like an unstructured mesh.  This allows validating the unstructured
# timestep against the structured reference data.


def build_quad_mesh_connectivities(M: int, N: int, dx: float, dy: float):
    """Build unstructured-style connectivities for a periodic M x N quad mesh.

    Vertex numbering: v(i,j) = i * N + j   for i in [0,M), j in [0,N)
    Edge numbering:
        x-edges: e_x(i,j) = i * N + j          for i in [0,M), j in [0,N)
            connects v(i,j) -> v((i+1)%M, j)
        y-edges: e_y(i,j) = M*N + i * N + j     for i in [0,M), j in [0,N)
            connects v(i,j) -> v(i, (j+1)%N)
    Cell numbering: c(i,j) = i * N + j   for i in [0,M), j in [0,N)
        cell (i,j) has corners v(i,j), v(i+1,j), v(i,j+1), v(i+1,j+1) (mod M,N)

    Returns a dict with all connectivity arrays and geometric fields.
    """
    num_vertices = M * N
    num_x_edges = M * N   # one x-edge per vertex (periodic)
    num_y_edges = M * N   # one y-edge per vertex (periodic)
    num_edges = num_x_edges + num_y_edges
    num_cells = M * N

    def vid(i, j):
        return (i % M) * N + (j % N)

    def xe_id(i, j):
        return (i % M) * N + (j % N)

    def ye_id(i, j):
        return M * N + (i % M) * N + (j % N)

    def cid(i, j):
        return (i % M) * N + (j % N)

    # --- E2V: each edge connects 2 vertices ---
    e2v = np.zeros((num_edges, 2), dtype=np.int32)
    for i in range(M):
        for j in range(N):
            # x-edge from v(i,j) to v(i+1,j)
            e2v[xe_id(i, j)] = [vid(i, j), vid(i + 1, j)]
            # y-edge from v(i,j) to v(i,j+1)
            e2v[ye_id(i, j)] = [vid(i, j), vid(i, j + 1)]

    # --- V2E: each vertex has 4 edges (E, N, W, S on a quad mesh) ---
    # Order: east x-edge, north y-edge, west x-edge, south y-edge
    v2e = np.zeros((num_vertices, 4), dtype=np.int32)
    for i in range(M):
        for j in range(N):
            v2e[vid(i, j)] = [
                xe_id(i, j),       # east: x-edge leaving this vertex
                ye_id(i, j),       # north: y-edge leaving this vertex
                xe_id(i - 1, j),   # west: x-edge arriving at this vertex
                ye_id(i, j - 1),   # south: y-edge arriving at this vertex
            ]

    # --- E2C: each edge has 2 adjacent cells ---
    # For x-edge at (i, j) [connecting v(i,j) -> v(i+1,j)]:
    #   cell below = c(i, j-1) [at (i+1/2, j-1/2)]
    #   cell above = c(i, j)   [at (i+1/2, j+1/2)]
    # For y-edge at (i, j) [connecting v(i,j) -> v(i,j+1)]:
    #   cell left  = c(i-1, j) [at (i-1/2, j+1/2)]
    #   cell right = c(i, j)   [at (i+1/2, j+1/2)]
    e2c = np.zeros((num_edges, 2), dtype=np.int32)
    for i in range(M):
        for j in range(N):
            e2c[xe_id(i, j)] = [cid(i, j - 1), cid(i, j)]
            e2c[ye_id(i, j)] = [cid(i - 1, j), cid(i, j)]

    # --- C2E: each cell has 4 edges (bottom, right, top, left) ---
    # Cell (i,j) at position (i+1/2, j+1/2):
    c2e = np.zeros((num_cells, 4), dtype=np.int32)
    for i in range(M):
        for j in range(N):
            c2e[cid(i, j)] = [
                xe_id(i, j),       # bottom: x-edge at y=j
                ye_id(i + 1, j),   # right: y-edge at x=i+1
                xe_id(i, j + 1),   # top: x-edge at y=j+1
                ye_id(i, j),       # left: y-edge at x=i
            ]

    # --- C2V: each cell has 4 vertices ---
    c2v = np.zeros((num_cells, 4), dtype=np.int32)
    for i in range(M):
        for j in range(N):
            c2v[cid(i, j)] = [
                vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            ]

    # --- sign_v2e: +1 if edge leaves vertex (v == E2V[0]), -1 if arrives ---
    sign_v2e_arr = np.zeros((num_vertices, 4), dtype=np.float64)
    for v in range(num_vertices):
        for local_e in range(4):
            edge = v2e[v, local_e]
            if e2v[edge, 0] == v:
                sign_v2e_arr[v, local_e] = +1.0  # edge leaves this vertex
            else:
                sign_v2e_arr[v, local_e] = -1.0  # edge arrives at this vertex

    # --- sign_c2e: +1 if edge direction matches CCW traversal ---
    # For cell (i,j), CCW traversal: bottom(->), right(^), top(<-), left(v)
    sign_c2e_arr = np.zeros((num_cells, 4), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            c = cid(i, j)
            # Bottom edge: (i,j)->(i+1,j), direction=right. CCW at bottom=right. +1
            sign_c2e_arr[c, 0] = +1.0
            # Right edge: (i+1,j)->(i+1,j+1), direction=up. CCW at right=up. +1
            sign_c2e_arr[c, 1] = +1.0
            # Top edge: (i,j+1)->(i+1,j+1), direction=right. CCW at top=left. -1
            sign_c2e_arr[c, 2] = -1.0
            # Left edge: (i,j)->(i,j+1), direction=up. CCW at left=down. -1
            sign_c2e_arr[c, 3] = -1.0

    # --- Geometric fields ---
    edge_length_arr = np.zeros(num_edges, dtype=np.float64)
    edge_length_arr[:num_x_edges] = dx   # x-edges have length dx
    edge_length_arr[num_x_edges:] = dy   # y-edges have length dy

    # Dual edge length: perpendicular extent of the dual face crossing each edge
    dual_edge_length_arr = np.zeros(num_edges, dtype=np.float64)
    dual_edge_length_arr[:num_x_edges] = dy  # dual of x-edge has length dy
    dual_edge_length_arr[num_x_edges:] = dx  # dual of y-edge has length dx

    inv_edge_length_arr = 1.0 / edge_length_arr
    inv_cell_area_arr = np.full(num_cells, 1.0 / (dx * dy), dtype=np.float64)
    inv_dual_area_arr = np.full(num_vertices, 1.0 / (dx * dy), dtype=np.float64)

    # KE weights: 0.25 for quad mesh (see kinetic_energy_v docstring)
    ke_weights_arr = np.full((num_vertices, 4), 0.25, dtype=np.float64)

    # --- TRiSK weights ---
    # For each edge, the "perpendicular" edges and their reconstruction weights.
    # On a quad mesh, each edge has exactly 4 TRiSK neighbors (the orthogonal
    # edges of the 2 adjacent cells), each with weight +/- 0.25.
    # The sign encodes the cross-product orientation:
    #   For x-edge: perpendicular edges are y-edges, weight = +0.25
    #   For y-edge: perpendicular edges are x-edges, weight = -0.25
    # (The sign difference captures u-eqn having +z*cv and v-eqn having -z*cu.)
    e2e_trisk = np.zeros((num_edges, 4), dtype=np.int32)
    trisk_weights_arr = np.zeros((num_edges, 4), dtype=np.float64)

    for i in range(M):
        for j in range(N):
            # x-edge at (i,j): neighbors are the 4 y-edges of its 2 cells
            xe = xe_id(i, j)
            # Cell below (i, j-1): y-edges at left=ye(i,j-1), right=ye(i+1,j-1)
            # Cell above (i, j):   y-edges at left=ye(i,j),   right=ye(i+1,j)
            e2e_trisk[xe] = [ye_id(i + 1, j), ye_id(i, j),
                             ye_id(i + 1, j - 1), ye_id(i, j - 1)]
            trisk_weights_arr[xe] = [0.25, 0.25, 0.25, 0.25]

            # y-edge at (i,j): neighbors are the 4 x-edges of its 2 cells
            ye = ye_id(i, j)
            # Cell left (i-1, j):  x-edges at bottom=xe(i-1,j), top=xe(i-1,j+1)
            # Cell right (i, j):   x-edges at bottom=xe(i,j),   top=xe(i,j+1)
            e2e_trisk[ye] = [xe_id(i, j), xe_id(i - 1, j),
                             xe_id(i, j + 1), xe_id(i - 1, j + 1)]
            trisk_weights_arr[ye] = [-0.25, -0.25, -0.25, -0.25]

    return {
        "e2v": e2v,
        "v2e": v2e,
        "e2c": e2c,
        "c2e": c2e,
        "c2v": c2v,
        "e2e_trisk": e2e_trisk,
        "sign_v2e": sign_v2e_arr,
        "sign_c2e": sign_c2e_arr,
        "edge_length": edge_length_arr,
        "dual_edge_length": dual_edge_length_arr,
        "inv_edge_length": inv_edge_length_arr,
        "inv_cell_area": inv_cell_area_arr,
        "inv_dual_area": inv_dual_area_arr,
        "ke_weights": ke_weights_arr,
        "trisk_weights": trisk_weights_arr,
        "num_vertices": num_vertices,
        "num_edges": num_edges,
        "num_cells": num_cells,
    }
