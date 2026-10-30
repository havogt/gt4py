# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from gt4py.next import field_operator, scan_operator
from gt4py.next.ffront.foast_to_func import field_operator_from_foast

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    IFloatField,
    IJKFloatField,
    KDim,
    cartesian_case,
)
from next_tests.integration_tests.cases_utils import exec_alloc_descriptor


def test_field_operator_with_statements(cartesian_case):
    @field_operator
    def original(a: IFloatField, b: IFloatField) -> tuple[IFloatField, IFloatField]:
        c = a + 1.0
        c = c * 2.0
        d, e = c, b + a
        if True:
            d = d - b
        return d, e

    testee = field_operator_from_foast(original.foast_stage)

    cases.verify_with_default_data(
        cartesian_case, testee, ref=lambda a, b: ((a + 1.0) * 2.0 - b, b + a)
    )


def test_call_from_field_operator(cartesian_case):
    @field_operator
    def double(a: IFloatField) -> IFloatField:
        return 2.0 * a

    # a field operator is only found under the name of its definition
    double = field_operator_from_foast(double.foast_stage)

    @field_operator
    def testee(a: IFloatField) -> IFloatField:
        return double(a) + 1.0

    cases.verify_with_default_data(cartesian_case, testee, ref=lambda a: 2.0 * a + 1.0)


@pytest.mark.uses_scan
@pytest.mark.uses_scan_in_field_operator
def test_scan_operator(cartesian_case):
    @scan_operator(axis=KDim, forward=True, init=0.0)
    def accumulate(state: float, x: float) -> float:
        return state + x

    # a scan operator is only found under the name of its definition
    accumulate = field_operator_from_foast(accumulate.foast_stage)

    @field_operator
    def testee(x: IJKFloatField) -> IJKFloatField:
        return accumulate(x)

    cases.verify_with_default_data(
        cartesian_case, testee, ref=lambda x: np.add.accumulate(x, axis=2)
    )
