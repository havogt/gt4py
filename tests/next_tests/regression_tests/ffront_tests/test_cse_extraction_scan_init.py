# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import gt4py.next as gtx
from gt4py.next import scan
import numpy as np

from next_tests.integration_tests.cases import KDim, cartesian_case
from next_tests.integration_tests import cases
from next_tests.integration_tests.cases_utils import (
    exec_alloc_descriptor,
)


@pytest.mark.uses_scan
def test_scan_init_duplicated(cartesian_case):
    """
    Tests that a non-trivial duplicated expression in the `init` argument of a scan operator works.

    GTFN currently doesn't like if the expression gets cse-extracted.
    """

    @gtx.field_operator
    def testee_pass(
        state: tuple[tuple[float], tuple[float]], inp: float
    ) -> tuple[tuple[float], tuple[float]]:
        return (state[0][0] + inp,), (state[1][0] + inp,)

    @gtx.field_operator
    def testee(
        inp: gtx.Field[[KDim], float],
    ) -> tuple[tuple[gtx.Field[[KDim], float]], tuple[gtx.Field[[KDim], float]]]:
        return scan(testee_pass, axis=KDim, forward=True, init=((1.0,), (1.0,)))(inp)

    inp = cases.allocate(cartesian_case, testee, "inp")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN).zeros()()

    cases.verify(
        cartesian_case,
        testee,
        inp,
        out=out,
        ref=(
            (np.cumsum(inp.asnumpy(), axis=0) + 1.0,),
            (np.cumsum(inp.asnumpy(), axis=0) + 1.0,),
        ),
    )
