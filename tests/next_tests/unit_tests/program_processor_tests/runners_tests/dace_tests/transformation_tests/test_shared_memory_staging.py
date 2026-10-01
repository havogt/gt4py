# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import dace
import numpy as np
import pytest
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations

from . import util


NI, NJ, NK, NX = 40, 20, 50, 2
TILE = (32, 16)
K_CHUNK = 30
K, X, J, I = "i_K_gtx_vertical", "i_X_gtx_horizontal", "i_J_gtx_horizontal", "i_I_gtx_horizontal"


def _make_sdfg() -> dace.SDFG:
    """Producer `t = c * f` on `[-1, NI] x [-1, NJ]`, consumer reading `t` at offsets.

    `t[I + 1, J + 1, 0, K]` holds the value at `(I, J)`. The consumer reads `t` at `(I + 1, J)`
    through a memlet and, per colour, at `(I, J - 1)` or `(I - 1, J + 1)` through tasklet
    subscripts. `out` is larger than the consumer domain by one in I and J.
    """
    sdfg = dace.SDFG(util.unique_name("staging"))
    state = sdfg.add_state(is_start_block=True)
    gpu = dace.StorageType.GPU_Global
    sdfg.add_array("c", (NI + 2, NJ + 2), dace.float64, storage=gpu)
    sdfg.add_array("f", (NI + 2, NJ + 2, NK), dace.float64, storage=gpu)
    sdfg.add_array("t", (NI + 2, NJ + 2, 1, NK), dace.float64, storage=gpu, transient=True)
    sdfg.add_array("out", (NI + 1, NJ + 1, NX, NK), dace.float64, storage=gpu)
    t = state.add_access("t")

    state.add_mapped_tasklet(
        "producer",
        map_ranges={K: f"0:{NK}", X: "0:1", J: f"-1:{NJ + 1}", I: f"-1:{NI + 1}"},
        inputs={
            "__c": dace.Memlet(f"c[{I} + 1, {J} + 1]"),
            "__f": dace.Memlet(f"f[{I} + 1, {J} + 1, {K}]"),
        },
        code="__t = __c * __f",
        outputs={"__t": dace.Memlet(f"t[{I} + 1, {J} + 1, 0, {K}]")},
        output_nodes={t},
        external_edges=True,
    )
    tasklet, _, _ = state.add_mapped_tasklet(
        "consumer",
        map_ranges={K: f"0:{NK}", X: f"0:{NX}", J: f"0:{NJ}", I: f"0:{NI}"},
        inputs={
            "__a": dace.Memlet(f"t[{I} + 2, {J} + 1, 0, {K}]"),
            "__in0": dace.Memlet(f"t[0:{NI + 2}, 0:{NJ + 2}, 0, 0:{NK}]"),
        },
        code=(
            f"__out = __a + (__in0[{I} + 1, {J}, {K}] if {X} == 0 else __in0[{I}, {J} + 2, {K}])"
        ),
        outputs={"__out": dace.Memlet(f"out[{I}, {J}, {X}, {K}]")},
        input_nodes={t},
        external_edges=True,
    )
    tasklet.in_connectors["__in0"] = dace.pointer(dace.float64)
    for node in state.nodes():
        if isinstance(node, dace_nodes.MapEntry):
            node.map.schedule = dace.ScheduleType.GPU_Device
            node.map.gpu_block_size = [32, 8, 1]
    sdfg.validate()
    return sdfg


def _stage(sdfg: dace.SDFG) -> int:
    return gtx_transformations.gt_stage_in_shared_memory(
        sdfg, pairs=[("producer", "consumer", ["t"])], tile=TILE, k_chunk=K_CHUNK
    )


def test_shared_memory_staging_structure():
    sdfg = _make_sdfg()
    assert _stage(sdfg) == 1
    sdfg.validate()
    state = sdfg.states()[0]

    kernels = [
        node
        for node in state.nodes()
        if isinstance(node, dace_nodes.MapEntry)
        and node.map.schedule == dace.ScheduleType.GPU_Device
    ]
    assert len(kernels) == 1
    assert kernels[0].map.gpu_block_size == [*TILE, 1]
    assert "t" not in sdfg.arrays

    tiles = [desc for desc in sdfg.arrays.values() if desc.storage == dace.StorageType.GPU_Shared]
    assert len(tiles) == 1 and tuple(tiles[0].shape) == (*TILE, 1, 2)

    maps = [node.map for node in state.nodes() if isinstance(node, dace_nodes.MapEntry)]
    blocks = [m for m in maps if m.schedule == dace.ScheduleType.GPU_ThreadBlock]
    assert len(blocks) == 2
    assert all([int(r[1] - r[0] + 1) for r in m.range] == [TILE[1], TILE[0]] for m in blocks)
    (k_map,) = [m for m in maps if m.params == [K]]
    assert k_map.schedule == dace.ScheduleType.Sequential
    assert kernels[0].map.range[0][2] == K_CHUNK

    code = [c for c in sdfg.generate_code() if c.language == "cu"][0].clean_code
    assert "__shared__ double" in code
    k_loop = code[code.index(f"for (auto {K}") :]
    assert k_loop.count("__syncthreads();") == 2


@pytest.mark.requires_gpu
def test_shared_memory_staging_result():
    import cupy as cp

    rng = np.random.default_rng(0)
    c = rng.uniform(0.5, 1.5, (NI + 2, NJ + 2))
    f = rng.uniform(0.5, 1.5, (NI + 2, NJ + 2, NK))

    def run(sdfg: dace.SDFG) -> np.ndarray:
        out = cp.full((NI + 1, NJ + 1, NX, NK), np.nan)
        sdfg.compile()(c=cp.asarray(c), f=cp.asarray(f), out=out)
        return cp.asnumpy(out)

    reference = run(_make_sdfg())
    staged_sdfg = _make_sdfg()
    assert _stage(staged_sdfg) == 1
    staged = run(staged_sdfg)

    assert np.isnan(reference[NI, :]).all() and np.isnan(reference[:, NJ]).all()
    assert np.isfinite(reference[:NI, :NJ]).all()
    np.testing.assert_allclose(staged, reference, rtol=1e-15, equal_nan=True)
