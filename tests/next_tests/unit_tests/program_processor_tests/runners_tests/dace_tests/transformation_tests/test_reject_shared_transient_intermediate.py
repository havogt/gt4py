# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dace
import pytest

from gt4py.next.program_processors.runners.dace import (
    transformations as gtx_transformations,
)

from . import util


def _make_sdfg(transient_intermediate: bool, extra_consumer: bool):
    sdfg = dace.SDFG(util.unique_name("reject_shared_transient_intermediate"))
    state = sdfg.add_state(is_start_block=True)
    for name in "abc":
        sdfg.add_array(name, shape=(10,), dtype=dace.float64, transient=False)
    sdfg.add_array("t", shape=(10,), dtype=dace.float64, transient=transient_intermediate)
    a, b, t = (state.add_access(name) for name in "abt")

    _, _, first_map_exit = state.add_mapped_tasklet(
        "first",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("a[__i]")},
        code="__out = __in + 1.0",
        outputs={"__out": dace.Memlet("t[__i]")},
        input_nodes={a},
        output_nodes={t},
        external_edges=True,
    )
    _, second_map_entry, _ = state.add_mapped_tasklet(
        "second",
        map_ranges={"__i": "0:10"},
        inputs={"__in": dace.Memlet("t[__i]")},
        code="__out = __in * 2.0",
        outputs={"__out": dace.Memlet("b[__i]")},
        input_nodes={t},
        output_nodes={b},
        external_edges=True,
    )
    if extra_consumer:
        state.add_nedge(t, state.add_access("c"), dace.Memlet("t[0:10] -> [0:10]"))
    sdfg.validate()
    return sdfg, state, first_map_exit, second_map_entry


@pytest.mark.parametrize(
    "transient_intermediate, extra_consumer, expected",
    [(True, False, True), (True, True, False), (False, True, True)],
)
def test_reject_shared_transient_intermediate(
    transient_intermediate: bool, extra_consumer: bool, expected: bool
):
    sdfg, state, first_map_exit, second_map_entry = _make_sdfg(
        transient_intermediate, extra_consumer
    )
    assert (
        gtx_transformations.gt_reject_shared_transient_intermediate(
            None, first_map_exit, second_map_entry, state, sdfg
        )
        is expected
    )
