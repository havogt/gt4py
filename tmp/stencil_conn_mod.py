"""Unstructured-mesh stencil for the connectivity-in-offset-provider smoke test."""

import numpy as np

import gt4py.next as gtx
from gt4py.next import neighbor_sum
from gt4py.next.program_processors.runners import gtfn


Vertex = gtx.Dimension("Vertex")
Edge = gtx.Dimension("Edge")
V2EDim = gtx.Dimension("V2E", kind=gtx.DimensionKind.LOCAL)
V2E = gtx.FieldOffset("V2E", source=Edge, target=(Vertex, V2EDim))


@gtx.field_operator
def sum_neighbors(
    edges: gtx.Field[[Edge], gtx.float64],
) -> gtx.Field[[Vertex], gtx.float64]:
    return neighbor_sum(edges(V2E), axis=V2EDim)


@gtx.program(backend=gtfn.run_gtfn)
def run_sum_neighbors(
    edges: gtx.Field[[Edge], gtx.float64],
    out: gtx.Field[[Vertex], gtx.float64],
):
    sum_neighbors(edges, out=out)


def build_connectivity(n_vertex: int = 6, n_neighbors: int = 3) -> gtx.Field:
    table = np.arange(n_vertex * n_neighbors, dtype=np.int32).reshape(n_vertex, n_neighbors)
    return gtx.as_connectivity([Vertex, V2EDim], codomain=Edge, data=table)
