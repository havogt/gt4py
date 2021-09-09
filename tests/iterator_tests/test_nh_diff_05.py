# GT4Py New Semantic Model - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.  GT4Py
# New Semantic Model is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or any later version.
# See the LICENSE.txt file at the top-level directory of this distribution for
# a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from iterator.atlas_utils import AtlasTable
from iterator.embedded import NeighborTableOffsetProvider, np_as_located_field
from iterator.runtime import *
from iterator.builtins import *
from iterator import library
import numpy as np


Vertex = CartesianAxis("Vertex")
Edge = CartesianAxis("Edge")

V2E = offset("V2E")
E2V = offset("E2V")
Diamond = offset("Diamond")

# def mo_nh_diffusion_stencil_05(
#     z_nabla4_e2: Field[Edge, K],
#     u_vert: Field[Vertex, K],
#     v_vert: Field[Vertex, K],
#     primal_normal_vert_v1: Field[Edge > Cell > Vertex],
#     primal_normal_vert_v2: Field[Edge > Cell > Vertex],
#     z_nabla2_e: Field[Edge, K],
#     inv_vert_vert_length: Field[Edge],
#     inv_primal_edge_length: Field[Edge],
# ):
#     nabv_tang: Field[Edge, K]
#     nabv_norm: Field[Edge, K]
#     with domain.upward.across[nudging:halo]:
#         nabv_tang = sum_over(
#             Edge > Cell > Vertex,
#             u_vert * primal_normal_vert_v1 + v_vert * primal_normal_vert_v2,
#             weights=[1.0, 1.0, 0.0, 0.0],
#         )
#         nabv_norm = sum_over(
#             Edge > Cell > Vertex,
#             u_vert * primal_normal_vert_v1 + v_vert * primal_normal_vert_v2,
#             weights=[0.0, 0.0, 1.0, 1.0],
#         )
#         z_nabla4_e2 = 4.0 * (
#             (nabv_norm - 2.0 * z_nabla2_e) * inv_vert_vert_length ** 2
#             + (nabv_tang - 2.0 * z_nabla2_e) * inv_primal_edge_length ** 2
#         )


# @fundef
# def foo(sparse_field):
#     return deref(sparse_field)


# @fundef
# def deref_sparse_field(field):
#     sparse_field = shift(E2V)(field)
#     return deref(shift(0)(lift(foo)(sparse_field)))


@fundef
def whatever_computation(
    u_vert_neighs, v_vert_neighs, primal_normal_vert_v1, primal_normal_vert_v2
):
    return deref(u_vert_neighs) * deref(primal_normal_vert_v1) + deref(v_vert_neighs) * deref(
        primal_normal_vert_v2
    )


@fundef
def for_first_neighbor(u_vert_neighs, v_vert_neighs, primal_normal_vert_v1, primal_normal_vert_v2):
    # is this possible?
    return deref(
        shift(0)(
            lift(whatever_computation)(
                u_vert_neighs, v_vert_neighs, primal_normal_vert_v1, primal_normal_vert_v2
            )
        )
    )


# @fundef
# def close_sum(fun):
#     return lift(whatever_computation)


@fundef
def nabv_tang(
    u_vert,
    v_vert,
    primal_normal_vert_v1,
    primal_normal_vert_v2,
):
    def body(i, u_vert_neighs, v_vert_neighs, primal_normal_vert_v1, primal_normal_vert_v2):
        return deref(shift(i)(u_vert_neighs)) * deref(shift(i)(primal_normal_vert_v1)) + deref(
            shift(i)(v_vert_neighs)
        ) * deref(shift(i)(primal_normal_vert_v2))

    u_vert_neighs = shift(Diamond)(u_vert)
    v_vert_neighs = shift(Diamond)(v_vert)
    return body(
        0, u_vert_neighs, v_vert_neighs, primal_normal_vert_v1, primal_normal_vert_v2
    ) + body(1, u_vert_neighs, v_vert_neighs, primal_normal_vert_v1, primal_normal_vert_v2)


@fendef
def test_nabv_tang(
    nabv_tang_out,
    u_vert,
    v_vert,
    primal_normal_vert_v1,
    primal_normal_vert_v2,
):
    closure(
        domain(named_range(Edge, 0, 0)),
        nabv_tang,
        [nabv_tang_out],
        [u_vert, v_vert, primal_normal_vert_v1, primal_normal_vert_v2],
    )


test_nabv_tang(None, None, None, None, None, backend="cpptoy")


@fundef
def nh_diff_05(
    u_vert,
    v_vert,
    primal_normal_vert_v1,
    primal_normal_vert_v2,
    z_nabla2_e,
    inv_vert_vert_length,
    inv_primal_edge_length,
):
    ...


@fundef
def compute_zavgS(pp, S_M):
    zavg = 0.5 * (deref(shift(E2V, 0)(pp)) + deref(shift(E2V, 1)(pp)))
    # zavg = 0.5 * reduce(lambda a, b: a + b, 0)(shift(E2V)(pp))
    # zavg = 0.5 * library.sum()(shift(E2V)(pp))
    return deref(S_M) * zavg


@fendef
def compute_zavgS_fencil(
    n_edges,
    out,
    pp,
    S_M,
):
    closure(
        domain(named_range(Edge, 0, n_edges)),
        compute_zavgS,
        [out],
        [pp, S_M],
    )


@fundef
def compute_pnabla(pp, S_M, sign, vol):
    zavgS = lift(compute_zavgS)(pp, S_M)
    # pnabla_M = reduce(lambda a, b, c: a + b * c, 0)(shift(V2E)(zavgS), sign)
    # pnabla_M = library.sum(lambda a, b: a * b)(shift(V2E)(zavgS), sign)
    pnabla_M = library.dot(shift(V2E)(zavgS), sign)
    return pnabla_M / deref(vol)


@fendef
def nabla(
    n_nodes,
    out_MXX,
    out_MYY,
    pp,
    S_MXX,
    S_MYY,
    sign,
    vol,
):
    # TODO replace by single stencil which returns tuple
    closure(
        domain(named_range(Vertex, 0, n_nodes)),
        compute_pnabla,
        [out_MXX],
        [pp, S_MXX, sign, vol],
    )
    closure(
        domain(named_range(Vertex, 0, n_nodes)),
        compute_pnabla,
        [out_MYY],
        [pp, S_MYY, sign, vol],
    )
