# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Field operators on a structured torus: unstructured entities stored on an `(I, J, X)` lattice."""

import dataclasses

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import common, neighbor_sum
from gt4py.next.embedded.structured_connectivity import StructuredConnectivity

from next_tests import definitions
from next_tests.integration_tests.cases_utils import exec_alloc_descriptor


pytestmark = [pytest.mark.uses_structured_connectivity, pytest.mark.uses_unstructured_shift]

Cell = gtx.Dimension("Cell")
Edge = gtx.Dimension("Edge")
Vertex = gtx.Dimension("Vertex")
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)
C2EDim = gtx.Dimension("C2E", kind=gtx.DimensionKind.LOCAL)
E2CDim = gtx.Dimension("E2C", kind=gtx.DimensionKind.LOCAL)
V2EDim = gtx.Dimension("V2E", kind=gtx.DimensionKind.LOCAL)
E2C2VDim = gtx.Dimension("E2C2V", kind=gtx.DimensionKind.LOCAL)
C2E = gtx.FieldOffset("C2E", source=Edge, target=(Cell, C2EDim))
E2C = gtx.FieldOffset("E2C", source=Cell, target=(Edge, E2CDim))
V2E = gtx.FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))
E2C2V = gtx.FieldOffset("E2C2V", source=Vertex, target=(Edge, E2C2VDim))
I = gtx.Dimension("I")  # noqa: E741 [ambiguous-variable-name]
J = gtx.Dimension("J")
X = gtx.Dimension("X")

N_COLORS = {Cell: 2, Edge: 3, Vertex: 1}
#: per source colour and neighbor: the `(I, J, X)` offset of the neighbor
OFFSETS = {
    "C2E": {0: [{}, {X: 1}, {J: 1, X: 2}], 1: [{X: -1}, {I: 1}, {X: 1}]},
    "E2C": {0: [{}, {J: -1, X: 1}], 1: [{X: -1}, {}], 2: [{X: -2}, {I: 1, X: -1}]},
    "V2E": {0: [{}, {X: 1}, {X: 2}, {I: -1}, {J: -1, X: 1}, {I: -1, J: -1, X: 2}]},
    "E2C2V": {
        0: [{}, {I: 1}, {J: 1}, {I: -1}],
        1: [{X: -1}, {I: 1, X: -1}, {J: -1, X: -1}, {I: 1, J: 1, X: -1}],
        2: [{X: -2}, {J: 1, X: -2}, {I: -1, X: -2}, {I: -1, J: 1, X: -2}],
    },
}
_ENTITIES = {
    "C2E": (Cell, Edge),
    "E2C": (Edge, Cell),
    "V2E": (Vertex, Edge),
    "E2C2V": (Edge, Vertex),
}
_LOCAL_DIMS = {"C2E": C2EDim, "E2C": E2CDim, "V2E": V2EDim, "E2C2V": E2C2VDim}
OFFSET_PROVIDER = {
    tag: StructuredConnectivity(
        source_dim=_ENTITIES[tag][0],
        codomain=_ENTITIES[tag][1],
        color_dim=X,
        local_dim=_LOCAL_DIMS[tag],
        offsets=offsets,
    )
    for tag, offsets in OFFSETS.items()
}
LAYOUT = tuple((entity, (I, J, X)) for entity in N_COLORS)

NI, NJ, NK = 5, 4, 3
#: periodic halo, enough for two chained neighbor accesses
HALO = 2


@gtx.field_operator
def c2e_sum(
    e: gtx.Field[[Edge, K], float], w: gtx.Field[[Cell, C2EDim], float]
) -> gtx.Field[[Cell, K], float]:
    return neighbor_sum(e(C2E) * w, axis=C2EDim) + e(C2E[1])


@gtx.field_operator
def sparse_slot(
    e: gtx.Field[[Edge, K], float], w: gtx.Field[[Cell, C2EDim], float]
) -> gtx.Field[[Cell, K], float]:
    return e(C2E[2]) * w[C2EDim(1)]


@gtx.field_operator
def e2c2v_sum(v: gtx.Field[[Vertex, K], float]) -> gtx.Field[[Edge, K], float]:
    return neighbor_sum(v(E2C2V), axis=E2C2VDim) - v(E2C2V[3])


@gtx.field_operator
def chained(c: gtx.Field[[Cell, K], float]) -> gtx.Field[[Cell, K], float]:
    return c(E2C[1])(C2E[0]) + c(E2C[0])(C2E[2])


@gtx.field_operator
def v2e_sum(e: gtx.Field[[Edge, K], float]) -> gtx.Field[[Vertex, K], float]:
    return neighbor_sum(e(V2E), axis=V2EDim)


def _torus_field(rng: np.random.Generator, entity: common.Dimension) -> np.ndarray:
    """Periodic halo `HALO` in I and J, then a NaN ring in I, J and X that no access may reach."""
    core = rng.random((NI, NJ, N_COLORS[entity], NK))
    padded = np.pad(core, ((HALO, HALO), (HALO, HALO), (0, 0), (0, 0)), mode="wrap")
    return np.pad(padded, ((1, 1), (1, 1), (1, 1), (0, 0)), constant_values=np.nan)


def _torus_domain(entity: common.Dimension) -> common.Domain:
    pad = HALO + 1
    return gtx.domain(
        {I: (-pad, NI + pad), J: (-pad, NJ + pad), X: (-1, N_COLORS[entity] + 1), K: (0, NK)}
    )


def _neighbor(data: np.ndarray, shifts: list[dict], colors: range, shape_i=NI, shape_j=NJ):
    """`data` read along the chain `shifts` (first applied first) for each output colour, by hand."""
    pad = HALO + 1
    out = np.empty((shape_i, shape_j, len(colors), data.shape[-1]))
    for c in colors:
        di = dj = 0
        x = c
        for tag, k in shifts:
            offset = OFFSETS[tag][x][k]
            di += offset.get(I, 0)
            dj += offset.get(J, 0)
            x += offset.get(X, 0)
        out[:, :, c] = data[pad + di : pad + di + shape_i, pad + dj : pad + dj + shape_j, 1 + x]
    return out


def _call(program, backend, *args, out_entity: common.Dimension, offset_provider=OFFSET_PROVIDER):
    out_domain = gtx.domain({I: (0, NI), J: (0, NJ), X: (0, N_COLORS[out_entity]), K: (0, NK)})
    out = gtx.as_field(out_domain, np.full(out_domain.shape, -1.0), allocator=backend.allocator)
    if isinstance(backend, definitions.EmbeddedDummyBackend):
        operator = program
    else:
        operator = (
            dataclasses.replace(
                program,
                definition_stage=dataclasses.replace(
                    program.definition_stage, _structured_layout=LAYOUT
                ),
            )
            .with_grid_type(common.GridType.UNSTRUCTURED)
            .with_backend(backend)
            .with_compilation_options(static_domains=True)
        )
    operator(*args, out=out, offset_provider=offset_provider)
    return out.asnumpy()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _as_field(backend, entity, data):
    return gtx.as_field(_torus_domain(entity), data, allocator=backend.allocator)


def _sparse(backend, rng, entity, local_dim, n):
    data = rng.random((NI, NJ, N_COLORS[entity], n))
    domain = gtx.domain({I: (0, NI), J: (0, NJ), X: (0, N_COLORS[entity]), local_dim: (0, n)})
    return data, gtx.as_field(domain, data, allocator=backend.allocator)


def test_c2e_sum(exec_alloc_descriptor, rng):
    e = _torus_field(rng, Edge)
    w, w_field = _sparse(exec_alloc_descriptor, rng, Cell, C2EDim, 3)
    colors = range(N_COLORS[Cell])
    reference = sum(
        _neighbor(e, [("C2E", k)], colors) * w[:, :, :, k, None] for k in range(3)
    ) + _neighbor(e, [("C2E", 1)], colors)

    actual = _call(
        c2e_sum,
        exec_alloc_descriptor,
        _as_field(exec_alloc_descriptor, Edge, e),
        w_field,
        out_entity=Cell,
    )

    np.testing.assert_allclose(actual, reference, rtol=1e-14)


def test_sparse_slot(exec_alloc_descriptor, rng):
    e = _torus_field(rng, Edge)
    w, w_field = _sparse(exec_alloc_descriptor, rng, Cell, C2EDim, 3)
    reference = _neighbor(e, [("C2E", 2)], range(N_COLORS[Cell])) * w[:, :, :, 1, None]

    actual = _call(
        sparse_slot,
        exec_alloc_descriptor,
        _as_field(exec_alloc_descriptor, Edge, e),
        w_field,
        out_entity=Cell,
    )

    np.testing.assert_allclose(actual, reference, rtol=1e-14)


def test_e2c2v_sum(exec_alloc_descriptor, rng):
    v = _torus_field(rng, Vertex)
    colors = range(N_COLORS[Edge])
    reference = sum(_neighbor(v, [("E2C2V", k)], colors) for k in range(4)) - _neighbor(
        v, [("E2C2V", 3)], colors
    )

    actual = _call(
        e2c2v_sum,
        exec_alloc_descriptor,
        _as_field(exec_alloc_descriptor, Vertex, v),
        out_entity=Edge,
    )

    np.testing.assert_allclose(actual, reference, rtol=1e-14)


def test_chained(exec_alloc_descriptor, rng):
    c = _torus_field(rng, Cell)
    colors = range(N_COLORS[Cell])
    reference = _neighbor(c, [("C2E", 0), ("E2C", 1)], colors) + _neighbor(
        c, [("C2E", 2), ("E2C", 0)], colors
    )

    actual = _call(
        chained, exec_alloc_descriptor, _as_field(exec_alloc_descriptor, Cell, c), out_entity=Cell
    )

    np.testing.assert_allclose(actual, reference, rtol=1e-14)


def test_single_color_output(exec_alloc_descriptor, rng):
    e = _torus_field(rng, Edge)
    reference = sum(_neighbor(e, [("V2E", k)], range(1)) for k in range(6))

    actual = _call(
        v2e_sum, exec_alloc_descriptor, _as_field(exec_alloc_descriptor, Edge, e), out_entity=Vertex
    )

    np.testing.assert_allclose(actual, reference, rtol=1e-14)


def test_unused_neighbor_table_is_ignored(exec_alloc_descriptor, rng):
    E2VDim = gtx.Dimension("E2V", kind=gtx.DimensionKind.LOCAL)
    e2v_table = gtx.as_connectivity(
        [Edge, E2VDim],
        Vertex,
        np.zeros((3, 2), dtype=gtx.IndexType),
        allocator=exec_alloc_descriptor.allocator,
    )
    e = _torus_field(rng, Edge)
    reference = sum(_neighbor(e, [("V2E", k)], range(1)) for k in range(6))

    actual = _call(
        v2e_sum,
        exec_alloc_descriptor,
        _as_field(exec_alloc_descriptor, Edge, e),
        out_entity=Vertex,
        offset_provider={**OFFSET_PROVIDER, "E2V": e2v_table},
    )

    np.testing.assert_allclose(actual, reference, rtol=1e-14)
