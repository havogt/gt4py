# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import re

import dace
import numpy as np
import pytest
from dace.sdfg import nodes as dace_nodes

from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations
from gt4py.next.program_processors.runners.dace.transformations import (
    shared_memory_staging as gtx_staging,
)

from . import util


NI, NJ, NK, NX = 40, 20, 50, 2
TILE = (32, 16)
K_CHUNK = 30
K, X, J, I = "i_K_gtx_vertical", "i_X_gtx_horizontal", "i_J_gtx_horizontal", "i_I_gtx_horizontal"
GPU = dace.StorageType.GPU_Global


def _make_sdfg(
    *,
    other_output: bool = False,
    shared_input: bool = False,
    k_offset: int = 0,
    second_writer: bool = False,
    squeezed_read: bool = False,
) -> dace.SDFG:
    """Producer `t = c * f` on `[-1, NI] x [-1, NJ]`, consumer reading `t` at offsets.

    `t[I + 1, J + 1, 0, K]` holds the value at `(I, J)`. The consumer reads `t`
    at `(I + 1, J)` through a memlet and, per colour, at `(I, J - 1)` or
    `(I - 1, J + 1)` through tasklet subscripts. `out` is larger than the
    consumer domain by one in I and J.
    """
    sdfg = dace.SDFG(util.unique_name("staging"))
    state = sdfg.add_state(is_start_block=True)
    sdfg.add_array("c", (NI + 2, NJ + 2), dace.float64, storage=GPU)
    sdfg.add_array("f", (NI + 2, NJ + 2, NK), dace.float64, storage=GPU)
    sdfg.add_array("t", (NI + 2, NJ + 2, 1, NK), dace.float64, storage=GPU, transient=True)
    sdfg.add_array("out", (NI + 1, NJ + 1, NX, NK), dace.float64, storage=GPU)
    t = state.add_access("t")

    outputs = {"__t": dace.Memlet(f"t[{I} + 1, {J} + 1, 0, {K}]")}
    code = "__t = __c * __f"
    if other_output:
        sdfg.add_array("u", (NI + 2, NJ + 2, NK), dace.float64, storage=GPU)
        outputs["__u"] = dace.Memlet(f"u[{I} + 1, {J} + 1, {K}]")
        code += "\n__u = __t + __c"
    state.add_mapped_tasklet(
        "producer",
        map_ranges={K: f"0:{NK}", X: "0:1", J: f"-1:{NJ + 1}", I: f"-1:{NI + 1}"},
        inputs={
            "__c": dace.Memlet(f"c[{I} + 1, {J} + 1]"),
            "__f": dace.Memlet(f"f[{I} + 1, {J} + 1, {K}]"),
        },
        code=code,
        outputs=outputs,
        output_nodes={t},
        external_edges=True,
    )
    if second_writer:
        state.add_mapped_tasklet(
            "second_writer",
            map_ranges={K: f"0:{NK}", J: "0:1", I: "0:1"},
            inputs={},
            code="__t = 0.0",
            outputs={"__t": dace.Memlet(f"t[{I}, {J}, 0, {K}]")},
            output_nodes={t},
            external_edges=True,
        )
    inputs = {
        "__a": dace.Memlet(f"t[{I} + 2, {J} + 1, 0, {K} + {k_offset}]"),
        "__in0": dace.Memlet(
            f"t[0:{NI + 2}, 3, 0, 0:{NK}]"
            if squeezed_read
            else f"t[0:{NI + 2}, 0:{NJ + 2}, 0, 0:{NK}]"
        ),
    }
    code = f"__out = __a + (__in0[{I} + 1, {J}, {K}] if {X} == 0 else __in0[{I}, {J} + 2, {K}])"
    if squeezed_read:
        code = f"__out = __a + __in0[{I} + 1, {K}]"
    if shared_input:
        inputs["__c"] = dace.Memlet(f"c[{I} + 1, {J} + 1]")
        code += " + __c"
    tasklet, _, _ = state.add_mapped_tasklet(
        "consumer",
        map_ranges={K: f"0:{NK}", X: f"0:{NX}", J: f"0:{NJ}", I: f"0:{NI}"},
        inputs=inputs,
        code=code,
        outputs={"__out": dace.Memlet(f"out[{I}, {J}, {X}, {K}]")},
        input_nodes={t},
        external_edges=True,
    )
    tasklet.in_connectors["__in0"] = dace.pointer(dace.float64)
    for node in state.nodes():
        if isinstance(node, dace_nodes.MapEntry):
            node.map.schedule = dace.ScheduleType.GPU_Device
            node.map.gpu_block_size = [32, 8, 1]
    return sdfg


def _labels(sdfg: dace.SDFG) -> tuple[str, str]:
    labels = [n.map.label for n in sdfg.states()[0].nodes() if isinstance(n, dace_nodes.MapEntry)]
    return next(lb for lb in labels if "producer" in lb), next(
        lb for lb in labels if "consumer" in lb
    )


def _stage(sdfg: dace.SDFG, **kwargs) -> int:
    producer, consumer = _labels(sdfg)
    return gtx_transformations.gt_stage_in_shared_memory(
        sdfg, pairs=[(producer, consumer, ["t"])], tile=TILE, k_chunk=K_CHUNK, **kwargs
    )


def test_shared_memory_staging_structure():
    sdfg = _make_sdfg()
    sdfg.validate()
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

    tiles = [name for name, d in sdfg.arrays.items() if d.storage == dace.StorageType.GPU_Shared]
    assert len(tiles) == 1 and tuple(sdfg.arrays[tiles[0]].shape) == (*TILE, 1, 2)

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
    barriers = [m.start() for m in re.finditer(r"__syncthreads\(\);", k_loop)]
    accesses = [m.start() for m in re.finditer(re.escape(tiles[0]) + r"\[", k_loop)]
    assert len(barriers) == 2
    # tile write, barrier, tile read, barrier
    assert accesses[0] < barriers[0] < accesses[-1] < barriers[1]
    assert any(barriers[0] < a < barriers[1] for a in accesses)


def test_shared_memory_staging_keeps_other_outputs():
    sdfg = _make_sdfg(other_output=True, shared_input=True)
    sdfg.validate()
    assert _stage(sdfg) == 1
    sdfg.validate()
    state = sdfg.states()[0]
    kernels = [
        node
        for node in state.nodes()
        if isinstance(node, dace_nodes.MapEntry)
        and node.map.schedule == dace.ScheduleType.GPU_Device
    ]
    assert len(kernels) == 2
    (producer,) = [k for k in kernels if "producer" in k.map.label]
    assert {e.data.data for e in state.out_edges(state.exit_node(producer))} == {"u"}
    assert sorted(sdfg.arglist()) == ["c", "f", "out", "u"]


@pytest.mark.parametrize(
    ["kwargs", "message"],
    [
        ({"k_offset": -1}, "vertical offset"),
        ({"second_writer": True}, "writers other than the producer"),
        ({"squeezed_read": True}, "squeezed"),
    ],
)
def test_shared_memory_staging_refuses(kwargs, message):
    sdfg = _make_sdfg(**kwargs)
    before = sdfg.hash_sdfg()
    with pytest.raises(gtx_staging.StagingRefusedError, match=message):
        _stage(sdfg)
    assert sdfg.hash_sdfg() == before


def test_shared_memory_staging_rejects_ambiguous_label():
    sdfg = _make_sdfg()
    producer, _ = _labels(sdfg)
    for node in sdfg.states()[0].nodes():
        if isinstance(node, dace_nodes.MapEntry):
            node.map.label = producer
    with pytest.raises(ValueError, match="matches 2 maps"):
        gtx_transformations.gt_stage_in_shared_memory(sdfg, pairs=[(producer, producer, ["t"])])


@pytest.mark.requires_gpu
@pytest.mark.parametrize(
    ["sdfg_kwargs", "stage_kwargs"],
    [
        ({}, {}),
        ({"other_output": True, "shared_input": True}, {}),
        ({}, {"double_buffer": False}),
    ],
    ids=["base", "other_output", "single_buffer"],
)
def test_shared_memory_staging_result(sdfg_kwargs, stage_kwargs):
    import cupy as cp

    rng = np.random.default_rng(0)
    c = rng.uniform(0.5, 1.5, (NI + 2, NJ + 2))
    f = rng.uniform(0.5, 1.5, (NI + 2, NJ + 2, NK))

    def run(sdfg: dace.SDFG) -> dict[str, np.ndarray]:
        outs = {"out": cp.full((NI + 1, NJ + 1, NX, NK), np.nan)}
        if "u" in sdfg.arrays:
            outs["u"] = cp.full((NI + 2, NJ + 2, NK), np.nan)
        sdfg.compile()(c=cp.asarray(c), f=cp.asarray(f), **outs)
        return {name: cp.asnumpy(value) for name, value in outs.items()}

    reference = run(_make_sdfg(**sdfg_kwargs))
    staged_sdfg = _make_sdfg(**sdfg_kwargs)
    assert _stage(staged_sdfg, **stage_kwargs) == 1
    staged = run(staged_sdfg)

    out = reference["out"]
    assert np.isnan(out[NI, :]).all() and np.isnan(out[:, NJ]).all()
    assert np.isfinite(out[:NI, :NJ]).all()
    for name in reference:
        np.testing.assert_allclose(staged[name], reference[name], rtol=1e-15, equal_nan=True)
