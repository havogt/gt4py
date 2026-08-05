# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import common


Vertex = common.Dimension("Vertex")
Edge = common.Dimension("Edge")
V2EDim = common.Dimension("V2E", kind=common.DimensionKind.LOCAL)


@dataclasses.dataclass(frozen=True)
class Mesh:
    V2E: common.Connectivity


def make_mesh(table) -> Mesh:
    return Mesh(
        V2E=gtx.as_connectivity(
            domain={Vertex: 2, V2EDim: 2},
            codomain=Edge,
            data=np.asarray(table, dtype=np.int32),
            skip_value=None,
        )
    )


def test_nothing_bound_yields_empty_offset_provider():
    assert gtx.ambient.offset_provider() == {}


def test_bound_namespace_provides_its_connectivities():
    mesh = gtx.Namespace("mesh")
    m = make_mesh([[0, 1], [1, 2]])
    with gtx.bind(mesh, m):
        assert gtx.ambient.offset_provider() == {"V2E": m.V2E}
    assert gtx.ambient.offset_provider() == {}


def test_bindings_nest_and_unwind():
    mesh = gtx.Namespace("mesh")
    m1, m2 = make_mesh([[0, 1], [1, 2]]), make_mesh([[1, 2], [0, 1]])
    with gtx.bind(mesh, m1):
        with gtx.bind(mesh, m2):
            assert gtx.ambient.offset_provider()["V2E"] is m2.V2E
        assert gtx.ambient.offset_provider()["V2E"] is m1.V2E


def test_colliding_offset_names_are_rejected():
    a, b = gtx.Namespace("a"), gtx.Namespace("b")
    with gtx.bind(a, make_mesh([[0, 1], [1, 2]])):
        with gtx.bind(b, make_mesh([[1, 2], [0, 1]])):
            with pytest.raises(ValueError, match="provided by more than one namespace"):
                gtx.ambient.offset_provider()


def test_freeze_gives_a_content_hash():
    conn = make_mesh([[0, 1], [1, 2]]).V2E
    assert common.frozen_content_hash(conn) is None
    gtx.freeze(conn)
    assert common.frozen_content_hash(conn) is not None


def test_equal_frozen_connectivities_share_a_cache_key():
    m1, m2 = make_mesh([[0, 1], [1, 2]]), make_mesh([[0, 1], [1, 2]])
    unfrozen = (
        common.hash_offset_provider_items_by_id({"V2E": m1.V2E}),
        common.hash_offset_provider_items_by_id({"V2E": m2.V2E}),
    )
    assert unfrozen[0] != unfrozen[1], "distinct objects are keyed apart before freezing"

    gtx.freeze(m1.V2E)
    gtx.freeze(m2.V2E)
    assert common.hash_offset_provider_items_by_id(
        {"V2E": m1.V2E}
    ) == common.hash_offset_provider_items_by_id({"V2E": m2.V2E})


def test_differing_frozen_connectivities_are_keyed_apart():
    m1 = gtx.freeze(make_mesh([[0, 1], [1, 2]]).V2E)
    m2 = gtx.freeze(make_mesh([[1, 2], [0, 1]]).V2E)
    assert common.hash_offset_provider_items_by_id(
        {"V2E": m1}
    ) != common.hash_offset_provider_items_by_id({"V2E": m2})


def test_readonly_freeze_marks_the_buffer_immutable():
    conn = gtx.freeze(make_mesh([[0, 1], [1, 2]]).V2E, readonly=True)
    with pytest.raises(ValueError):
        conn.ndarray[0, 0] = 7
