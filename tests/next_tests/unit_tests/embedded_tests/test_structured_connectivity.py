# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pickle

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import common, neighbor_sum
from gt4py.next.embedded import structured_connectivity as structured
from gt4py.next.embedded.structured_connectivity import StructuredConnectivity


Cell = gtx.Dimension("Cell")
Edge = gtx.Dimension("Edge")
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)
C2EDim = gtx.Dimension("C2E", kind=gtx.DimensionKind.LOCAL)
C2E = gtx.FieldOffset("C2E", source=Edge, target=(Cell, C2EDim))

I = gtx.Dimension("I")
J = gtx.Dimension("J")
X = gtx.Dimension("X")

# a row of cell pairs (X in {0, 1}) with three edges per cell index (X in {0, 1, 2})
STRUCTURED_C2E = StructuredConnectivity(
    source_dim=Cell,
    codomain=Edge,
    color_dim=X,
    local_dim=C2EDim,
    offsets={0: [{}, {X: 1}, {X: 2}], 1: [{X: -1}, {I: 1}, {X: 1}]},
)
N = 5


@gtx.field_operator
def weighted_sum_plus_first(
    e: gtx.Field[[Edge, K], float], w: gtx.Field[[Cell, C2EDim], float]
) -> gtx.Field[[Cell, K], float]:
    return neighbor_sum(e(C2E) * w, axis=C2EDim) + e(C2E[0])


def _reference(edges: np.ndarray, weights: np.ndarray) -> np.ndarray:
    ref = np.empty((N, 2, edges.shape[-1]))
    neighbors = {
        0: [edges[:N, 0], edges[:N, 1], edges[:N, 2]],
        1: [edges[:N, 0], edges[1 : N + 1, 1], edges[:N, 2]],
    }
    for color, per_k in neighbors.items():
        ref[:, color] = (
            sum(nb * weights[:, color, k, None] for k, nb in enumerate(per_k)) + per_k[0]
        )
    return ref


def _call(edges, weights):
    return weighted_sum_plus_first(
        edges,
        weights,
        domain={I: (0, N), X: (0, 2), K: (0, 3)},
        offset_provider={"C2E": STRUCTURED_C2E},
    )


def _inputs(xp):
    rng = np.random.default_rng(0)
    edges = gtx.as_field([I, X, K], xp.asarray(rng.random((N + 1, 3, 3))), allocator=xp)
    weights = gtx.as_field([I, X, C2EDim], xp.asarray(rng.random((N, 2, 3))), allocator=xp)
    return edges, weights


def test_structured_neighbor_sum_with_vertical_dimension():
    edges, weights = _inputs(np)

    result = _call(edges, weights)

    assert result.domain == gtx.domain({I: (0, N), X: (0, 2), K: (0, 3)})
    np.testing.assert_allclose(result.asnumpy(), _reference(edges.asnumpy(), weights.asnumpy()))


@pytest.mark.requires_jax
def test_structured_neighbor_sum_under_jax_jit():
    jax = pytest.importorskip("jax")
    edges, weights = _inputs(jax.numpy)

    result = jax.jit(_call)(edges, weights)

    np.testing.assert_allclose(result.asnumpy(), _reference(edges.asnumpy(), weights.asnumpy()))


GATHER_CASES = {
    "one_color": (
        {0: [{}, {I: 1}, {J: -1}, {I: -1, J: 1}]},
        {I: (0, 6), J: (-1, 5), X: (0, 1), K: (0, 3)},
    ),
    "two_colors": (
        {0: [{}, {X: 1}, {J: 1, X: 2}], 1: [{X: -1}, {I: 1}, {X: 1}]},
        {I: (0, 6), J: (0, 5), X: (0, 3), K: (0, 3)},
    ),
    "three_colors": (
        {0: [{}, {J: -1, X: 1}], 1: [{X: -1}, {}], 2: [{X: -2}, {I: 1, X: -1}]},
        {I: (-2, 5), J: (-1, 4), X: (0, 2), K: (0, 2)},
    ),
    "three_colors_without_vertical": (
        {0: [{}, {J: -1, X: 1}], 1: [{X: -1}, {}], 2: [{X: -2}, {I: 1, X: -1}]},
        {I: (0, 5), J: (-1, 4), X: (0, 2)},
    ),
    "color_dim_between_horizontal_dims": (
        {0: [{I: 1}, {X: 1}], 1: [{J: -1}, {X: -1}]},
        {I: (1, 5), X: (0, 2), J: (0, 4), K: (0, 2)},
    ),
}


def _connectivity(offsets) -> StructuredConnectivity:
    return StructuredConnectivity(
        source_dim=Cell, codomain=Edge, color_dim=X, local_dim=C2EDim, offsets=offsets
    )


def _input(input_ranges, xp=np):
    shape = tuple(stop - start for start, stop in input_ranges.values())
    data = np.random.default_rng(0).random(shape)
    return gtx.as_field(gtx.domain(input_ranges), xp.asarray(data), allocator=xp)


def _gather_reference(offsets, input_ranges, data, ks):
    """Plain-numpy gather of the neighbors `ks`, one point at a time."""
    dims = list(input_ranges)
    multi_color = len(offsets) > 1
    out_ranges = {}
    for d in dims:
        if multi_color and d == X:
            out_ranges[d] = (min(offsets), max(offsets) + 1)
        else:
            shifted = [
                (input_ranges[d][0] - off.get(d, 0), input_ranges[d][1] - off.get(d, 0))
                for per_k in offsets.values()
                for off in (per_k[k] for k in ks)
            ]
            out_ranges[d] = (max(s for s, _ in shifted), min(e for _, e in shifted))

    result = np.empty((*(stop - start for start, stop in out_ranges.values()), len(ks)))
    for idx in np.ndindex(result.shape[:-1]):
        point = {d: out_ranges[d][0] + i for d, i in zip(dims, idx)}
        color = point[X] if multi_color else next(iter(offsets))
        for local, k in enumerate(ks):
            off = offsets[color][k]
            result[(*idx, local)] = data[
                tuple(point[d] + off.get(d, 0) - input_ranges[d][0] for d in dims)
            ]
    return out_ranges, result


@pytest.mark.parametrize("case", GATHER_CASES.keys())
def test_expand_k(case):
    offsets, input_ranges = GATHER_CASES[case]
    conn = _connectivity(offsets)
    inp = _input(input_ranges)

    for k in range(conn.num_neighbors):
        result = structured.expand_k(inp, conn, k)

        out_ranges, ref = _gather_reference(offsets, input_ranges, inp.asnumpy(), [k])
        assert result.domain == gtx.domain(out_ranges)
        np.testing.assert_array_equal(result.asnumpy(), ref[..., 0])


@pytest.mark.parametrize("case", GATHER_CASES.keys())
def test_expand_stacked(case):
    offsets, input_ranges = GATHER_CASES[case]
    conn = _connectivity(offsets)
    inp = _input(input_ranges)

    result = structured.expand_stacked(inp, conn)

    out_ranges, ref = _gather_reference(
        offsets, input_ranges, inp.asnumpy(), range(conn.num_neighbors)
    )
    dims = list(out_ranges)
    local_axis = dims.index(K) if K in dims else len(dims)
    expected = {d: out_ranges[d] for d in dims[:local_axis]}
    expected[C2EDim] = (0, conn.num_neighbors)
    expected |= {d: out_ranges[d] for d in dims[local_axis:]}
    assert result.domain == gtx.domain(expected)
    np.testing.assert_array_equal(result.asnumpy(), np.moveaxis(ref, -1, local_axis))


@pytest.mark.requires_jax
@pytest.mark.parametrize("case", GATHER_CASES.keys())
def test_expand_under_jax_jit(case):
    jax = pytest.importorskip("jax")
    offsets, input_ranges = GATHER_CASES[case]
    conn = _connectivity(offsets)
    inp = _input(input_ranges, jax.numpy)
    np_inp = _input(input_ranges)

    stacked = jax.jit(lambda f: structured.expand_stacked(f, conn))(inp)
    first = jax.jit(lambda f: structured.expand_k(f, conn, 0))(inp)

    np_stacked = structured.expand_stacked(np_inp, conn)
    np_first = structured.expand_k(np_inp, conn, 0)
    assert stacked.domain == np_stacked.domain
    assert first.domain == np_first.domain
    np.testing.assert_array_equal(stacked.asnumpy(), np_stacked.asnumpy())
    np.testing.assert_array_equal(first.asnumpy(), np_first.asnumpy())


def test_gt_type_is_hashable_and_equal_by_value():
    reordered_with_zeros = StructuredConnectivity(
        source_dim=Cell,
        codomain=Edge,
        color_dim=X,
        local_dim=C2EDim,
        offsets={1: [{X: -1}, {I: 1, X: 0}, {X: 1}], 0: [{I: 0}, {X: 1}, {X: 2}]},
    )

    first = common.offset_provider_to_type({"C2E": STRUCTURED_C2E})
    second = common.offset_provider_to_type({"C2E": reordered_with_zeros})

    assert first == second
    assert hash(first["C2E"]) == hash(second["C2E"])
    assert first["C2E"] == common.StructuredConnectivityType(
        source_dim=Cell,
        codomain=Edge,
        color_dim=X,
        local_dim=C2EDim,
        offsets=(
            (0, ((), ((X, 1),), ((X, 2),))),
            (1, (((X, -1),), ((I, 1),), ((X, 1),))),
        ),
    )
    assert first["C2E"].max_neighbors == STRUCTURED_C2E.max_neighbors == 3
    assert first["C2E"].neighbor_dim == STRUCTURED_C2E.neighbor_dim == C2EDim
    assert not first["C2E"].has_skip_values and not STRUCTURED_C2E.has_skip_values
    assert first["C2E"].colors == (0, 1)
    assert first["C2E"].neighbor_offset(1, 1) == {I: 1}


def test_structured_offset_provider_is_accepted():
    provider = {"C2E": STRUCTURED_C2E}

    assert common.is_offset_provider(provider)
    assert common.is_offset_provider_type(common.offset_provider_to_type(provider))


def test_structured_offset_provider_pickles():
    from gt4py.next.otf import compilation_tasks

    provider = compilation_tasks._offset_provider_with_file_refs({"C2E": STRUCTURED_C2E})

    restored = pickle.loads(pickle.dumps(provider))

    assert restored == {"C2E": STRUCTURED_C2E}
    assert common.offset_provider_to_type(restored) == common.offset_provider_to_type(provider)
