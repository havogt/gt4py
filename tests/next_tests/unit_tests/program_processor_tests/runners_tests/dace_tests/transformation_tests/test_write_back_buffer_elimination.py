# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

dace = pytest.importorskip("dace")

from dace.transformation import pass_pipeline as dace_ppl

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util


def _mk_write_back_buffer_sdfg(
    write_back_subset: str,
) -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    """`tmp` is written back into `b` and additionally read by a second consumer."""
    sdfg = dace.SDFG(util.unique_name("write_back_buffer_sdfg"))

    for name in ["a", "c"]:
        sdfg.add_array(name, shape=(10, 10), dtype=dace.float64, transient=False)
    sdfg.add_array("b", shape=(100, 100), dtype=dace.float64, transient=False)
    sdfg.add_array("tmp", shape=(10, 10), dtype=dace.float64, transient=True)

    state1: dace.SDFGState = sdfg.add_state(is_start_block=True)
    state1.add_mapped_tasklet(
        "producer",
        map_ranges={"__i1": "0:10", "__i2": "0:10"},
        inputs={"__in": dace.Memlet("a[__i1, __i2]")},
        code="__out = __in + 10.0",
        outputs={"__out": dace.Memlet("tmp[__i1, __i2]")},
        external_edges=True,
    )

    state2 = sdfg.add_state_after(state1)
    state2.add_edge(
        state2.add_access("tmp"),
        None,
        state2.add_access("b"),
        None,
        dace.Memlet(write_back_subset),
    )
    # This second consumer is what makes `DistributedBufferRelocator` and
    #  `GT4PyMapBufferElimination` bail out.
    state2.add_mapped_tasklet(
        "consumer",
        map_ranges={"__i1": "0:10", "__i2": "0:10"},
        inputs={"__in": dace.Memlet("tmp[__i1, __i2]")},
        code="__out = __in * 2.0",
        outputs={"__out": dace.Memlet("c[__i1, __i2]")},
        external_edges=True,
    )
    sdfg.validate()

    return sdfg, state1, state2


def _apply(sdfg: dace.SDFG) -> bool:
    res = dace_ppl.Pipeline(
        [gtx_transformations.GT4PyWriteBackBufferElimination(assume_pointwise=True)]
    ).apply_pass(sdfg, {})
    return bool(res and res.get("GT4PyWriteBackBufferElimination"))


def test_write_back_buffer_elimination():
    sdfg, state1, state2 = _mk_write_back_buffer_sdfg("tmp[0:10, 0:10] -> [11:21, 22:32]")
    ref, res = util.make_sdfg_args(sdfg)
    util.compile_and_run_sdfg(sdfg, **ref)

    assert _apply(sdfg)

    assert "tmp" not in sdfg.arrays
    assert not any(dnode.data == "tmp" for dnode in state1.data_nodes())
    assert not any(dnode.data == "tmp" for dnode in state2.data_nodes())
    # The producer now writes `b` directly, at the offset of the former write back.
    (producer_out,) = [
        edge for edge in state1.edges() if isinstance(edge.dst, dace.sdfg.nodes.AccessNode)
    ]
    assert producer_out.dst.data == "b"
    assert str(producer_out.data.subset) == "11:21, 22:32"

    util.compile_and_run_sdfg(sdfg, **res)
    assert util.compare_sdfg_res(ref, res)


def test_write_back_buffer_elimination_partial_write_back():
    """A partially copied buffer must be kept.

    Writing `b` directly would touch `b[16:21, 22:32]`, which is outside the range
    that the write back covers.
    """
    sdfg, _, _ = _mk_write_back_buffer_sdfg("tmp[0:5, 0:10] -> [11:16, 22:32]")
    assert not _apply(sdfg)
    assert "tmp" in sdfg.arrays
