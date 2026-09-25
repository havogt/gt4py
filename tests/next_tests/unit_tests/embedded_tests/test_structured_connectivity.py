# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import neighbor_sum
from gt4py.next.embedded.structured_connectivity import StructuredConnectivity


Cell = gtx.Dimension("Cell")
Edge = gtx.Dimension("Edge")
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)
C2EDim = gtx.Dimension("C2E", kind=gtx.DimensionKind.LOCAL)
C2E = gtx.FieldOffset("C2E", source=Edge, target=(Cell, C2EDim))

I = gtx.Dimension("I")
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
