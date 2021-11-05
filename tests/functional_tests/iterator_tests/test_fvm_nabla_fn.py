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

import numpy as np
import pytest


pytest.importorskip("atlas4py")

from functional.iterator import library
from functional.iterator.atlas_utils import AtlasTable
from functional.iterator.builtins import *
from functional.iterator.embedded import (
    NeighborTableOffsetProvider,
    NeighborTableOffsetProviderBase,
    index_field,
    np_as_located_field,
)
from functional.iterator.runtime import *


Vertex = CartesianAxis("Vertex")
Edge = CartesianAxis("Edge")
K = CartesianAxis("K")

V2E = offset("V2E")
E2V = offset("E2V")


@fundef
def compute_zavgS(pp, S_M):
    # zavg = 0.5 * (deref(shift(E2V, 0)(pp)) + deref(shift(E2V, 1)(pp)))
    # zavg = 0.5 * (deref(shift(E2V, 0)(pp)) + deref(shift(E2V, 1)(pp)))
    zavg = 0.5 * reduce(lambda a, b: a + b, 0.0)(shift(E2V)(pp))
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
    pnabla_M = library.dot(shift(V2E)(zavgS), deref(sign))
    return pnabla_M / deref(vol)


@fendef
def nabla(
    n_nodes,
    out_MXX,
    # out_MYY,
    pp,
    S_MXX,
    # S_MYY,
    sign,
    vol,
):
    # TODO replace by single stencil which returns tuple
    closure(
        domain(named_range(Vertex, 0, n_nodes), named_range(K, 0, 3)),
        compute_pnabla,
        [out_MXX],
        [pp, S_MXX, sign, vol],
    )
    # closure(
    #     domain(named_range(Vertex, 0, n_nodes)),
    #     compute_pnabla,
    #     [out_MYY],
    #     [pp, S_MYY, sign, vol],
    # )


def test_nabla():
    # setup = nabla_setup()

    # sign = np_as_located_field(Vertex, V2E)(setup.sign_field)
    # pp = np_as_located_field(Vertex)(setup.input_field)
    # S_MXX, S_MYY = tuple(map(np_as_located_field(Edge), setup.S_fields))
    # vol = np_as_located_field(Vertex)(setup.vol_field)

    # pnabla_MXX = np_as_located_field(Vertex)(np.zeros((setup.nodes_size)))
    # pnabla_MYY = np_as_located_field(Vertex)(np.zeros((setup.nodes_size)))

    e2v = NeighborTableOffsetProviderBase(2)
    v2e = NeighborTableOffsetProviderBase(6)

    params = [None] * 6
    nabla(
        *params,
        offset_provider={"E2V": e2v, "V2E": v2e},
        backend="gtfn",
        file="/home/vogtha/git/gridtools/tests/regression/fn/fn_nabla_gen.hpp",
    )

    # if validate:
    #     assert_close(-3.5455427772566003e-003, min(pnabla_MXX))
    #     assert_close(3.5455427772565435e-003, max(pnabla_MXX))
    #     assert_close(-3.3540113705465301e-003, min(pnabla_MYY))
    #     assert_close(3.3540113705465301e-003, max(pnabla_MYY))


test_nabla()
