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

Diamond = offset("Diamond")

# periodic
#
# 0v---0e-- 1v---3e-- 2v---6e-- 0v
# |  \ 0c   |  \ 1c   |  \
# |   \1e   |   \4e   |   \7e
# |2e   \   |5e   \   |8e   \
# |      \  |      \  |      \
# 3v---9e-- 4v--12e-- 5v--15e-- 3v
# |  \ 2c   |  \ 3c   |  \
# |   \10e  |   \13e  |   \16e
# |11e  \   |14e  \   |17e  \
# |      \  |      \  |      \
# 6v--18e-- 7v--21e-- 8v--24e-- 6v
# |  \      |  \      |  \
# |   \19e  |   \22e  |   \25e
# |20e  \   |23e  \   |26e  \
# |      \  |      \  |      \
# 0v       1v         2v        0v

diamond_arr = np.asarray(
    [
        [0, 1, 4, 6],  # 0
        [0, 4, 1, 3],
        [0, 3, 4, 2],
        [1, 2, 5, 7],  # 3
        [1, 5, 2, 4],
        [1, 4, 5, 0],
        [2, 0, 3, 8],  # 6
        [2, 3, 0, 5],
        [2, 5, 1, 3],
        [3, 4, 0, 7],  # 9
        [3, 7, 4, 6],
        [3, 6, 5, 7],
        [4, 5, 1, 8],  # 12
        [4, 8, 5, 7],
        [4, 7, 3, 8],
        [5, 3, 2, 6],  # 15
        [5, 6, 3, 8],
        [5, 8, 4, 6],
        [6, 7, 3, 1],  # 18
        [6, 1, 7, 0],
        [6, 0, 1, 8],
        [7, 8, 4, 2],  # 21
        [7, 2, 8, 1],
        [7, 1, 6, 2],
        [8, 6, 5, 0],  # 24
        [8, 0, 6, 2],
        [8, 2, 7, 0],
    ]
)

n_vertices = 9
n_edges = 27

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


def nabv_ref(weights, u_vert, v_vert, primal_normal_vert_v1, primal_normal_vert_v2):
    return np.sum(
        (u_vert[diamond_arr] * primal_normal_vert_v1 + v_vert[diamond_arr] * primal_normal_vert_v2)
        * weights,
        axis=-1,
    )


def nabv_tang_ref(u_vert, v_vert, primal_normal_vert_v1, primal_normal_vert_v2):
    weights = np.asarray([[1.0, 1.0, 0.0, 0.0]] * n_edges)
    return nabv_ref(weights, u_vert, v_vert, primal_normal_vert_v1, primal_normal_vert_v2)


def nabv_norm_ref(u_vert, v_vert, primal_normal_vert_v1, primal_normal_vert_v2):
    weights = np.asarray([[0.0, 0.0, 1.0, 1.0]] * n_edges)
    return nabv_ref(weights, u_vert, v_vert, primal_normal_vert_v1, primal_normal_vert_v2)


def z_nabla4_e2_ref(nabv_norm, nabv_tang, z_nabla2_e, inv_vert_vert_length, inv_primal_edge_length):
    return 4.0 * (
        (nabv_norm - 2.0 * z_nabla2_e) * inv_vert_vert_length ** 2
        + (nabv_tang - 2.0 * z_nabla2_e) * inv_primal_edge_length ** 2
    )


class Setup:
    def __init__(self) -> None:
        self.u_vert = np.random.rand(n_vertices)
        self.v_vert = np.random.rand(n_vertices)
        self.primal_normal_vert_v1 = np.random.rand(n_edges, 4)
        self.primal_normal_vert_v2 = np.random.rand(n_edges, 4)
        self.z_nabla2_e = np.random.rand(n_edges)
        self.inv_vert_vert_length = np.random.rand(n_edges)
        self.inv_primal_edge_length = np.random.rand(n_edges)


# @fundef
# def foo(sparse_field):
#     return deref(sparse_field)


# @fundef
# def deref_sparse_field(field):
#     sparse_field = shift(Diamond)(field)
#     return deref(shift(0)(lift(foo)(sparse_field)))


# @fendef
# def deref_sparse_field_fencil(n_edges, out, inp):
#     closure(
#         domain(named_range(Edge, 0, n_edges)),
#         deref_sparse_field,
#         [out],
#         [inp],
#     )


# def test_deref_sparse_field():
#     n_vertices = 9
#     n_edges = 27
#     vert = np_as_located_field(Vertex)(np.random.rand(n_vertices))

#     out = np_as_located_field(Edge)(np.zeros([n_edges]))

#     diamond = NeighborTableOffsetProvider(diamond_arr, Edge, Vertex, 4)

#     deref_sparse_field_fencil(
#         n_edges,
#         out,
#         vert,
#         offset_provider={"Diamond": diamond},
#     )


# test_deref_sparse_field()
# # exit(1)


# @fundef
# def whatever_computation(
#     u_vert_neighs, v_vert_neighs, primal_normal_vert_v1, primal_normal_vert_v2
# ):
#     return deref(u_vert_neighs) * deref(primal_normal_vert_v1) + deref(v_vert_neighs) * deref(
#         primal_normal_vert_v2
#     )


# @fundef
# def for_first_neighbor(u_vert, v_vert_neighs, primal_normal_vert_v1, primal_normal_vert_v2):
#     # is this possible?
#     u_vert_neighs = shift(Diamond)(u_vert)
#     return deref(
#         shift(0)(
#             lift(whatever_computation)(
#                 u_vert_neighs, v_vert_neighs, primal_normal_vert_v1, primal_normal_vert_v2
#             )
#         )
#     )


# @fendef
# def neigh_fencil(
#     nabv_tang_out,
#     n_edges,
#     u_vert,
#     v_vert,
#     primal_normal_vert_v1,
#     primal_normal_vert_v2,
# ):
#     closure(
#         domain(named_range(Edge, 0, n_edges)),
#         for_first_neighbor,
#         [nabv_tang_out],
#         [u_vert, v_vert, primal_normal_vert_v1, primal_normal_vert_v2],
#     )


# neigh_fencil(None, 0, None, None, None, None, backend="cpptoy")


# def test_neigh_fencil():
#     n_vertices = 9
#     n_edges = 27
#     u_vert = np_as_located_field(Vertex)(np.random.rand(n_vertices))
#     v_vert = np_as_located_field(Vertex)(np.random.rand(n_vertices))
#     primal_normal_vert_v1 = np_as_located_field(Vertex, Diamond)(np.random.rand(n_edges, 4))
#     primal_normal_vert_v2 = np_as_located_field(Vertex, Diamond)(np.random.rand(n_edges, 4))

#     out = np_as_located_field(Edge)(np.zeros([n_edges]))

#     diamond = NeighborTableOffsetProvider(diamond_arr, Edge, Vertex, 4)

#     neigh_fencil(
#         out,
#         n_edges,
#         u_vert,
#         v_vert,
#         primal_normal_vert_v1,
#         primal_normal_vert_v2,
#         offset_provider={"Diamond": diamond},
#     )


# test_neigh_fencil()
# exit(1)
# @fundef
# def close_sum(fun):
#     return lift(whatever_computation)


# @fundef
# def nabv_tang(
#     u_vert,
#     v_vert,
#     primal_normal_vert_v1,
#     primal_normal_vert_v2,
# ):
#     def body(i, u_vert_neighs, v_vert_neighs, primal_normal_vert_v1, primal_normal_vert_v2):

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


@fundef
def nabv_norm(
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
        2, u_vert_neighs, v_vert_neighs, primal_normal_vert_v1, primal_normal_vert_v2
    ) + body(3, u_vert_neighs, v_vert_neighs, primal_normal_vert_v1, primal_normal_vert_v2)


@fundef
def z_nabla4_e2(
    u_vert,
    v_vert,
    primal_normal_vert_v1,
    primal_normal_vert_v2,
    z_nabla2_e,
    inv_vert_vert_length,
    inv_primal_edge_length,
):
    nabv_norm_v = nabv_norm(u_vert, v_vert, primal_normal_vert_v1, primal_normal_vert_v2)
    nabv_tang_v = nabv_tang(u_vert, v_vert, primal_normal_vert_v1, primal_normal_vert_v2)

    return 4.0 * (
        (nabv_norm_v - 2.0 * deref(z_nabla2_e))
        * (deref(inv_vert_vert_length) * deref(inv_vert_vert_length))
        + (nabv_tang_v - 2.0 * deref(z_nabla2_e))
        * (deref(inv_primal_edge_length) * deref(inv_primal_edge_length))
    )


@fendef
def nh_diff_05(
    z_nabla4_e2_out,
    u_vert,
    v_vert,
    primal_normal_vert_v1,
    primal_normal_vert_v2,
    z_nabla2_e,
    inv_vert_vert_length,
    inv_primal_edge_length,
):
    closure(
        domain(named_range(Edge, 0, n_edges)),
        z_nabla4_e2,
        [z_nabla4_e2_out],
        [
            u_vert,
            v_vert,
            primal_normal_vert_v1,
            primal_normal_vert_v2,
            z_nabla2_e,
            inv_vert_vert_length,
            inv_primal_edge_length,
        ],
    )


@fendef
def nabv_tang_fencil(
    nabv_tang_out,
    u_vert,
    v_vert,
    primal_normal_vert_v1,
    primal_normal_vert_v2,
):
    closure(
        domain(named_range(Edge, 0, n_edges)),
        nabv_tang,
        [nabv_tang_out],
        [u_vert, v_vert, primal_normal_vert_v1, primal_normal_vert_v2],
    )


# nabv_tang_fencil(None, None, None, None, None, backend="cpptoy")


def test_nabv_tang():
    s = Setup()
    ref = nabv_tang_ref(s.u_vert, s.v_vert, s.primal_normal_vert_v1, s.primal_normal_vert_v2)

    diamond = NeighborTableOffsetProvider(diamond_arr, Edge, Vertex, 4)
    out = np_as_located_field(Edge)(np.zeros(n_edges))
    nabv_tang_fencil(
        out,
        np_as_located_field(Vertex)(s.u_vert),
        np_as_located_field(Vertex)(s.v_vert),
        np_as_located_field(Edge, Diamond)(s.primal_normal_vert_v1),
        np_as_located_field(Edge, Diamond)(s.primal_normal_vert_v2),
        offset_provider={"Diamond": diamond},
    )
    assert np.allclose(out, ref)


test_nabv_tang()


def test_nh_diff05():
    s = Setup()
    nabv_tang = nabv_tang_ref(s.u_vert, s.v_vert, s.primal_normal_vert_v1, s.primal_normal_vert_v2)
    nabv_norm = nabv_norm_ref(s.u_vert, s.v_vert, s.primal_normal_vert_v1, s.primal_normal_vert_v2)
    ref = z_nabla4_e2_ref(
        nabv_norm, nabv_tang, s.z_nabla2_e, s.inv_vert_vert_length, s.inv_primal_edge_length
    )

    diamond = NeighborTableOffsetProvider(diamond_arr, Edge, Vertex, 4)
    out = np_as_located_field(Edge)(np.zeros(n_edges))
    nh_diff_05(
        out,
        np_as_located_field(Vertex)(s.u_vert),
        np_as_located_field(Vertex)(s.v_vert),
        np_as_located_field(Edge, Diamond)(s.primal_normal_vert_v1),
        np_as_located_field(Edge, Diamond)(s.primal_normal_vert_v2),
        np_as_located_field(Edge)(s.z_nabla2_e),
        np_as_located_field(Edge)(s.inv_vert_vert_length),
        np_as_located_field(Edge)(s.inv_primal_edge_length),
        offset_provider={"Diamond": diamond},
    )
    assert np.allclose(out, ref)


test_nh_diff05()
