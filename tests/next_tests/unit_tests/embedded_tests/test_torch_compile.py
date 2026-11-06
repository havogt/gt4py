# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import NamedTuple

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import astype, neighbor_sum
from gt4py.next.experimental import as_offset


pytestmark = pytest.mark.requires_torch

K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)
Vertex = gtx.Dimension("Vertex")
Edge = gtx.Dimension("Edge")
E2VDim = gtx.Dimension("E2V", kind=gtx.DimensionKind.LOCAL)
E2V = gtx.FieldOffset("E2V", source=Vertex, target=(Edge, E2VDim))
Koff = gtx.FieldOffset("Koff", source=K, target=(K,))


@gtx.field_operator
def _scale_shift(a: gtx.Field[gtx.Dims[K], float], n: gtx.int32) -> gtx.Field[gtx.Dims[K], float]:
    return a(K + 1) * 2.0 + astype(n, float)


@gtx.program
def _scale_shift_program(
    a: gtx.Field[gtx.Dims[K], float], n: gtx.int32, out: gtx.Field[gtx.Dims[K], float]
):
    _scale_shift(a, n, out=out, domain={K: (0, 3)})


@gtx.field_operator
def _edge_sum(v: gtx.Field[gtx.Dims[Vertex], float]) -> gtx.Field[gtx.Dims[Edge], float]:
    return neighbor_sum(v(E2V), axis=E2VDim) + v(E2V[0])


@gtx.program
def _edge_sum_program(v: gtx.Field[gtx.Dims[Vertex], float], out: gtx.Field[gtx.Dims[Edge], float]):
    _edge_sum(v, out=out)


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)])
def torch_device(request):
    import torch

    return torch.device(request.param)


def test_program_fullgraph(torch_device):
    import torch

    with torch_device:
        a = gtx.as_field([K], np.arange(4.0), allocator=torch)
        out, out2, expected, expected2 = (gtx.zeros({K: 4}, allocator=torch) for _ in range(4))

    def step(a, out, out2):
        _scale_shift_program(a, 3, out, offset_provider={})
        _scale_shift(out, 1, out=out2, offset_provider={}, domain={K: (0, 2)})
        return out2

    step(a, expected, expected2)
    torch._dynamo.reset()
    result = torch.compile(step, fullgraph=True)(a, out, out2)

    assert result.ndarray.device.type == torch_device.type
    np.testing.assert_array_equal(out.asnumpy(), expected.asnumpy())
    np.testing.assert_array_equal(result.asnumpy(), expected2.asnumpy())


def test_unstructured_fullgraph(torch_device):
    import torch

    table = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [1, 3]], dtype=np.int32)
    with torch_device:
        e2v = gtx.as_connectivity([Edge, E2VDim], Vertex, torch.asarray(table), allocator=torch)
        v = gtx.as_field([Vertex], np.arange(4.0) + 1.0, allocator=torch)
        out = gtx.zeros({Edge: 5}, allocator=torch)
    offset_provider = {"E2V": e2v}

    def step(v, out):
        _edge_sum_program(v, out, offset_provider=offset_provider)
        return out

    torch._dynamo.reset()
    result = torch.compile(step, fullgraph=True)(v, out)

    expected = (np.arange(4.0) + 1.0)[table].sum(axis=1) + (np.arange(4.0) + 1.0)[table[:, 0]]
    np.testing.assert_array_equal(result.asnumpy(), expected)


def test_int_index_fullgraph(torch_device):
    import torch

    with torch_device:
        f = gtx.as_field([Edge, K], np.arange(6.0).reshape(2, 3), allocator=torch)

    def last_level(f):
        return f[K(2)]

    torch._dynamo.reset()
    result = torch.compile(last_level, fullgraph=True)(f)

    np.testing.assert_array_equal(result.asnumpy(), last_level(f).asnumpy())


class _Pair(NamedTuple):
    a: gtx.Field[gtx.Dims[K], float]
    b: gtx.Field[gtx.Dims[K], float]


@gtx.field_operator
def _pair_sum(
    p: tuple[gtx.Field[gtx.Dims[K], float], gtx.Field[gtx.Dims[K], float]],
) -> gtx.Field[gtx.Dims[K], float]:
    return p[0] + p[1]


def test_namedtuple_arg_fullgraph(torch_device):
    import torch

    with torch_device:
        a = gtx.as_field([K], np.arange(3.0), allocator=torch)
        b = gtx.as_field([K], np.arange(3.0) * 10.0, allocator=torch)
        out = gtx.zeros({K: 3}, allocator=torch)

    def step(a, b, out):
        _pair_sum(_Pair(a, b), out=out, offset_provider={})
        return out

    torch._dynamo.reset()
    result = torch.compile(step, fullgraph=True)(a, b, out)

    np.testing.assert_array_equal(result.asnumpy(), np.arange(3.0) * 11.0)


@gtx.field_operator
def _shift_by_field(
    a: gtx.Field[gtx.Dims[Edge, K], float], offset: gtx.Field[gtx.Dims[Edge, K], gtx.int32]
) -> gtx.Field[gtx.Dims[Edge, K], float]:
    return a(as_offset(Koff, offset))


def test_as_offset_field_fullgraph(torch_device):
    import torch

    values = np.arange(8.0).reshape(2, 4)
    offsets = np.asarray([[1, 1, 0, -1], [0, 2, 1, 0]], dtype=np.int32)
    with torch_device:
        a = gtx.as_field([Edge, K], values, allocator=torch)
        offset = gtx.as_field([Edge, K], offsets, allocator=torch)
        beyond = gtx.as_field([Edge, K], np.full((2, 4), 3, dtype=np.int32), allocator=torch)
        below = gtx.as_field([Edge, K], np.full((2, 4), -3, dtype=np.int32), allocator=torch)
        out = gtx.zeros({Edge: 2, K: 4}, allocator=torch)

    def step(a, offset, out):
        _shift_by_field(a, offset, out=out, offset_provider={})
        return out

    torch._dynamo.reset()
    compiled = torch.compile(step, fullgraph=True)

    np.testing.assert_array_equal(
        compiled(a, offset, out).asnumpy(),
        np.take_along_axis(values, np.arange(4) + offsets, axis=1),
    )
    # out-of-range targets read the boundary value under a trace
    np.testing.assert_array_equal(
        compiled(a, beyond, out).asnumpy(), np.repeat(values[:, -1:], 4, axis=1)
    )
    np.testing.assert_array_equal(
        compiled(a, below, out).asnumpy(),
        np.take_along_axis(
            values, np.maximum(np.arange(4) - 3, 0)[None, :].repeat(2, axis=0), axis=1
        ),
    )


@gtx.field_operator
def _neighbor_sum(v: gtx.Field[gtx.Dims[Vertex], float]) -> gtx.Field[gtx.Dims[Edge], float]:
    return neighbor_sum(v(E2V), axis=E2VDim)


@gtx.field_operator
def _first_neighbor(v: gtx.Field[gtx.Dims[Vertex], float]) -> gtx.Field[gtx.Dims[Edge], float]:
    return v(E2V[0])


def _skip_value_e2v_table(num_vertices: int) -> np.ndarray:
    """Two edges per vertex, both entries equal, every seventh second entry a skip value."""
    vertex = np.arange(4 * num_vertices) // 2
    table = np.stack([vertex, vertex], axis=1).astype(np.int32)
    table[::7, 1] = -1
    return table


class _GraphRecorder:
    def __init__(self):
        self.graphs = []

    def __call__(self, gm, example_inputs):
        self.graphs.append(gm)
        return gm.forward


def _reference_neighbor_sum(values: np.ndarray, table: np.ndarray) -> np.ndarray:
    return np.where(table != -1, values[table], 0.0).sum(axis=1)


@pytest.mark.parametrize("num_field_vertices", [300, 600], ids=["narrowing", "covering"])
def test_premap_with_connectivity_argument_fullgraph(torch_device, num_field_vertices):
    import torch

    table = _skip_value_e2v_table(300)
    num_edges = 2 * num_field_vertices
    values = np.arange(float(num_field_vertices))
    with torch_device:
        e2v = gtx.as_connectivity(
            [Edge, E2VDim], Vertex, torch.asarray(table), skip_value=-1, allocator=torch
        )
        v = gtx.as_field([Vertex], values, allocator=torch)
        out, expected = (gtx.zeros({Edge: num_edges}, allocator=torch) for _ in range(2))

    def step(v, e2v, out):
        _neighbor_sum(v, out=out, offset_provider={"E2V": e2v})
        return out

    step(v, e2v, expected)
    torch._dynamo.reset()
    recorder = _GraphRecorder()
    result = torch.compile(step, fullgraph=True, backend=recorder)(v, e2v, out)

    np.testing.assert_allclose(result.asnumpy(), expected.asnumpy())
    np.testing.assert_allclose(result.asnumpy(), _reference_neighbor_sum(values, table[:num_edges]))
    (graph,) = recorder.graphs
    input_shapes = [
        tuple(node.meta["example_value"].shape) for node in graph.graph.find_nodes(op="placeholder")
    ]
    assert table.shape in input_shapes
    constants = [getattr(graph, node.target) for node in graph.graph.find_nodes(op="get_attr")]
    assert not any(isinstance(c, torch.Tensor) for c in constants)


def test_premap_with_restricted_connectivity_fullgraph(torch_device):
    import torch

    table = _skip_value_e2v_table(8)
    with torch_device:
        e2v = gtx.as_connectivity(
            [Edge, E2VDim], Vertex, torch.asarray(table), skip_value=-1, allocator=torch
        )
        v = gtx.as_field([Vertex], np.arange(4.0), allocator=torch)
        out = gtx.zeros({Edge: 8}, allocator=torch)

    def step(v, e2v, out):
        _first_neighbor(v, out=out, offset_provider={"E2V": e2v})
        return out

    torch._dynamo.reset()
    result = torch.compile(step, fullgraph=True)(v, e2v, out)

    np.testing.assert_array_equal(result.asnumpy(), np.arange(4.0)[table[:8, 0]])


def test_premap_non_contiguous_inverse_image_raises_fullgraph(torch_device):
    import torch

    with torch_device:
        e2v = gtx.as_connectivity(
            [Edge, E2VDim],
            Vertex,
            torch.asarray([[0, 0], [5, 5], [1, 1]], dtype=torch.int32),
            allocator=torch,
        )
        v = gtx.as_field([Vertex], np.arange(3.0), allocator=torch)
        out = gtx.zeros({Edge: 3}, allocator=torch)

    def step(v, e2v, out):
        _first_neighbor(v, out=out, offset_provider={"E2V": e2v})
        return out

    torch._dynamo.reset()
    with pytest.raises(Exception) as excinfo:
        torch.compile(step, fullgraph=True)(v, e2v, out)

    # Dynamo reports the failed fake run, the error is further down the chain
    error = excinfo.value
    while error is not None and not isinstance(error, ValueError):
        error = error.__cause__ or error.__context__
    assert error is not None and "non-contiguous" in str(error)


def test_recompiles_per_connectivity(torch_device):
    import torch

    table = _skip_value_e2v_table(8)
    with torch_device:
        tensor = torch.asarray(table)
        v = gtx.as_field([Vertex], np.arange(8.0), allocator=torch)
        out = gtx.zeros({Edge: 16}, allocator=torch)

    def connectivity(tensor):
        return gtx.as_connectivity([Edge, E2VDim], Vertex, tensor, skip_value=-1, allocator=torch)

    def step(v, e2v, out):
        _neighbor_sum(v, out=out, offset_provider={"E2V": e2v})
        return out

    torch._dynamo.reset()
    recorder = _GraphRecorder()
    compiled = torch.compile(step, fullgraph=True, backend=recorder)
    e2v = connectivity(tensor)
    compiled(v, e2v, out)
    compiled(v, e2v, out)
    assert len(recorder.graphs) == 1

    compiled(v, connectivity(tensor), out)
    assert len(recorder.graphs) == 2

    np.testing.assert_allclose(
        compiled(v, connectivity(tensor.clone()), out).asnumpy(),
        _reference_neighbor_sum(np.arange(8.0), table[:16]),
    )
    assert len(recorder.graphs) == 3
