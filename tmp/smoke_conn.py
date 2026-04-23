"""Unstructured connectivity offset-provider smoke: verify that a NeighborTable survives
the spawn pickle round-trip and that the compiled program produces correct results."""

import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

import gt4py.next as gtx
from gt4py.next.otf import compiled_program

from stencil_conn_mod import Edge, Vertex, run_sum_neighbors, build_connectivity


def main() -> None:
    n_vertex, n_neighbors = 6, 3
    print(
        f"[main] pid={os.getpid()} "
        f"mode={os.environ.get('GT4PY_BUILD_JOBS_MODE', '<default>')} "
        f"pool={type(compiled_program._async_compilation_pool).__name__}"
    )

    conn = build_connectivity(n_vertex=n_vertex, n_neighbors=n_neighbors)
    n_edges = int(np.asarray(conn.ndarray).max()) + 1
    edges = gtx.as_field([Edge], np.arange(n_edges, dtype=np.float64))
    out = gtx.as_field([Vertex], np.zeros(n_vertex, dtype=np.float64))

    run_sum_neighbors(edges, out, offset_provider={"V2E": conn})

    expected = np.asarray(conn.ndarray).astype(np.float64).sum(axis=1)
    got = out.asnumpy()
    print(f"[main] out = {got.tolist()}")
    print(f"[main] exp = {expected.tolist()}")
    assert np.allclose(got, expected)
    print("[main] OK")


if __name__ == "__main__":
    main()
