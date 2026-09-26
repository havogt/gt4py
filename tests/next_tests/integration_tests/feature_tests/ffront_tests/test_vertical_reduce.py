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
from gt4py.next import int32, maximum, reduce

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, KDim, cartesian_case
from next_tests.integration_tests.cases_utils import exec_alloc_descriptor


pytestmark = pytest.mark.uses_vertical_reduce


@gtx.field_operator
def add(a: float, b: float) -> float:
    return a + b


def test_reduce_sum(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IKFloatField) -> cases.IFloatField:
        return reduce(add, range=(KDim, 1, 4))(a)

    a = cases.allocate(cartesian_case, testee, "a")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, out=out, ref=a.asnumpy()[:, 1:4].sum(axis=1))


def test_reduce_runtime_range(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IKFloatField, start: int32, stop: int32) -> cases.IFloatField:
        return reduce(add, range=(KDim, start, stop))(a)

    ksize = cartesian_case.default_sizes[KDim]
    a = cases.allocate(cartesian_case, testee, "a")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(
        cartesian_case,
        testee,
        a,
        int32(2),
        int32(ksize),
        out=out,
        ref=a.asnumpy()[:, 2:].sum(axis=1),
    )


def test_reduce_single_level(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IKFloatField) -> cases.IFloatField:
        return reduce(add, range=(KDim, 3, 4))(a)

    a = cases.allocate(cartesian_case, testee, "a")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, out=out, ref=a.asnumpy()[:, 3])


def test_reduce_max(cartesian_case):
    @gtx.field_operator
    def max_op(a: int32, b: int32) -> int32:
        return maximum(a, b)

    @gtx.field_operator
    def testee(a: cases.IKField) -> cases.IField:
        return reduce(max_op, range=(KDim, 0, 5))(a)

    a = cases.allocate(cartesian_case, testee, "a", strategy=cases.UniqueInitializer())()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, out=out, ref=a.asnumpy()[:, 0:5].max(axis=1))


def test_reduce_of_expression(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IKFloatField, b: cases.IKFloatField) -> cases.IFloatField:
        return reduce(add, range=(KDim, 0, 5))(a * b) + 1.0

    a = cases.allocate(cartesian_case, testee, "a")()
    b = cases.allocate(cartesian_case, testee, "b")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(
        cartesian_case,
        testee,
        a,
        b,
        out=out,
        ref=(a.asnumpy() * b.asnumpy())[:, 0:5].sum(axis=1) + 1.0,
    )


@pytest.mark.uses_tuple_returns
def test_reduce_tuple(cartesian_case):
    @gtx.field_operator
    def add_and_max(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
        return a[0] + b[0], maximum(a[1], b[1])

    @gtx.field_operator
    def testee(
        a: cases.IKFloatField, b: cases.IKFloatField
    ) -> tuple[cases.IFloatField, cases.IFloatField]:
        return reduce(add_and_max, range=(KDim, 0, 5))((a, b))

    a = cases.allocate(cartesian_case, testee, "a")()
    b = cases.allocate(cartesian_case, testee, "b")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(
        cartesian_case,
        testee,
        a,
        b,
        out=out,
        ref=(a.asnumpy()[:, 0:5].sum(axis=1), b.asnumpy()[:, 0:5].max(axis=1)),
    )
