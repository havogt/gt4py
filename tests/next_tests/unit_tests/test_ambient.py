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
from gt4py.next import common, neighbor_sum


Vertex = common.Dimension("Vertex")
Edge = common.Dimension("Edge")
V2EDim = common.Dimension("V2E", kind=common.DimensionKind.LOCAL)


V2E = gtx.FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))


class Mesh:
    """A container declaring what it supplies; the class attribute is the declaration."""

    V2E = V2E

    def __init__(self, connectivity):
        self.V2E = connectivity


def make_mesh(table) -> Mesh:
    return Mesh(
        gtx.as_connectivity(
            domain={Vertex: 2, V2EDim: 2},
            codomain=Edge,
            data=np.asarray(table, dtype=np.int32),
            skip_value=None,
        )
    )


def test_nothing_bound_yields_empty_offset_provider():
    assert gtx.ambient.offset_provider() == {}


def test_bound_offset_declaration_provides_its_connectivity():
    m = make_mesh([[0, 1], [1, 2]])
    with gtx.bind(V2E, m.V2E):
        assert gtx.ambient.offset_provider() == {"V2E": m.V2E}
    assert gtx.ambient.offset_provider() == {}


def test_container_binds_by_declaration_not_by_name():
    """The container's class attribute names the declaration it supplies."""
    m = make_mesh([[0, 1], [1, 2]])
    with gtx.bindings(gtx.ambient.as_bindings(m)):
        assert gtx.ambient.offset_provider() == {"V2E": m.V2E}


def test_bindings_nest_and_unwind():
    m1, m2 = make_mesh([[0, 1], [1, 2]]), make_mesh([[1, 2], [0, 1]])
    with gtx.bind(V2E, m1.V2E):
        with gtx.bind(V2E, m2.V2E):
            assert gtx.ambient.offset_provider()["V2E"] is m2.V2E
        assert gtx.ambient.offset_provider()["V2E"] is m1.V2E


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


# --- embedded end-to-end (backend-free) --------------------------------------


@gtx.field_operator
def sum_edges(a: gtx.Field[gtx.Dims[Edge], gtx.int32]) -> gtx.Field[gtx.Dims[Vertex], gtx.int32]:
    return neighbor_sum(a(V2E), axis=V2EDim)


@gtx.program
def run(
    a: gtx.Field[gtx.Dims[Edge], gtx.int32], out: gtx.Field[gtx.Dims[Vertex], gtx.int32]
) -> None:
    sum_edges(a, out=out)


@pytest.fixture
def inputs():
    return (
        gtx.as_field([Edge], np.arange(3, dtype=np.int32)),
        make_mesh([[0, 1], [1, 2]]),
        np.asarray([1, 3], dtype=np.int32),
    )


@pytest.mark.parametrize("entry_point", ["program", "field_operator"])
@pytest.mark.parametrize("mechanism", ["offset_provider", "context_manager", "bind_kwarg"])
def test_embedded_execution_via_every_mechanism(inputs, entry_point, mechanism):
    a, m, expected = inputs
    out = gtx.zeros(gtx.domain({Vertex: 2}), dtype=np.int32)
    callee = run if entry_point == "program" else sum_edges
    kwargs = {"out": out} if entry_point == "field_operator" else {}
    args = (a,) if entry_point == "field_operator" else (a, out)
    if mechanism == "offset_provider":
        callee(*args, **kwargs, offset_provider={"V2E": m.V2E})
    elif mechanism == "context_manager":
        with gtx.bind(V2E, m.V2E):
            callee(*args, **kwargs)
    else:
        callee(*args, **kwargs, bind=m)

    np.testing.assert_array_equal(out.asnumpy(), expected)


def test_bind_kwarg_is_scoped_to_the_call(inputs):
    a, m, _ = inputs
    out = gtx.zeros(gtx.domain({Vertex: 2}), dtype=np.int32)
    run(a, out, bind=m)
    assert gtx.ambient.offset_provider() == {}


def test_bindings_are_context_local():
    """The binding must not leak between contexts (the fingerprint reads it, not a mirror)."""
    import contextvars

    decl = gtx.Static[gtx.float64]
    seen = {}

    def bind_and_record():
        with gtx.bind(decl, 0.25):
            seen["inner"] = decl.value

    with gtx.bind(decl, 0.5):
        contextvars.copy_context().run(bind_and_record)
        seen["outer"] = decl.value

    assert seen == {"inner": 0.25, "outer": 0.5}
