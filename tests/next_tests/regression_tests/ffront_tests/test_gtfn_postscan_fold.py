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
from gt4py.next import common

from next_tests import definitions as test_definitions
from next_tests.integration_tests import cases
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)


pytestmark = [pytest.mark.uses_unstructured_shift, pytest.mark.uses_scan]


Cell = gtx.Dimension("Cell")
KDim = gtx.Dimension("KDim", kind=gtx.DimensionKind.VERTICAL)
Koff = gtx.FieldOffset("Koff", KDim, (KDim,))


@gtx.scan_operator(axis=KDim, forward=False, init=0.0)
def _wscan(w_above: float, a: float, b: float) -> float:
    return a * w_above + b


@gtx.field_operator
def _solve_and_consume(
    a: gtx.Field[[Cell, KDim], float],
    b: gtx.Field[[Cell, KDim], float],
    g: gtx.Field[[Cell, KDim], float],
) -> gtx.Field[[Cell, KDim], float]:
    # `w` is an internal temporary: a backward scan output consumed cell-locally at K and Koff[1].
    # On the GTFN backend the post-scan fold folds this consumer into the scan's tail and drops the
    # `w` temp's SID write; other backends just run it as a separate scan + field-op.
    w = _wscan(a, b)
    return w * g + w(Koff[1])


@gtx.program
def postscan_fold_program(
    a: gtx.Field[[Cell, KDim], float],
    b: gtx.Field[[Cell, KDim], float],
    g: gtx.Field[[Cell, KDim], float],
    out0: gtx.Field[[Cell, KDim], float],
):
    _solve_and_consume(a, b, g, out=out0[:, :-1])


def _reference(a, b, g):
    cell, n = a.shape
    w = np.zeros((cell, n + 1))
    for k in range(n - 1, -1, -1):
        w[:, k] = a[:, k] * w[:, k + 1] + b[:, k]
    out0 = np.zeros((cell, n))
    for k in range(n - 1):
        out0[:, k] = w[:, k] * g[:, k] + w[:, k + 1]
    return out0


@pytest.mark.uses_program_with_sliced_out_arguments
def test_postscan_fold(exec_alloc_descriptor):
    cell_size, k_size = 14, 10
    test_case = cases.Case(
        None
        if isinstance(exec_alloc_descriptor, test_definitions.EmbeddedDummyBackend)
        else exec_alloc_descriptor,
        offset_provider={"Koff": KDim},
        default_sizes={Cell: cell_size, KDim: k_size},
        grid_type=common.GridType.UNSTRUCTURED,
        allocator=exec_alloc_descriptor.allocator,
    )

    rng = np.random.default_rng(42)
    a = test_case.as_field([Cell, KDim], rng.uniform(size=(cell_size, k_size)))
    b = test_case.as_field([Cell, KDim], rng.uniform(size=(cell_size, k_size)))
    g = test_case.as_field([Cell, KDim], rng.uniform(size=(cell_size, k_size)))
    out0 = test_case.as_field([Cell, KDim], np.zeros((cell_size, k_size)))

    ref = _reference(a.asnumpy(), b.asnumpy(), g.asnumpy())

    cases.verify(
        test_case,
        postscan_fold_program,
        a,
        b,
        g,
        out0,
        inout=out0,
        ref=ref,
        comparison=lambda r, o: np.allclose(r[:, :-1], o[:, :-1]),
    )
