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
from gt4py.next import broadcast
from gt4py.next.ffront.experimental import as_offset, concat_where

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    IDim,
    KDim,
    cartesian_case,
)
from next_tests.integration_tests.cases_utils import (
    Ioff,
    Koff,
    exec_alloc_descriptor,
)


@pytest.mark.uses_cartesian_shift
def test_cartesian_shift(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IJKField) -> cases.IJKField:
        return a(IDim + 1)

    a = cases.allocate(cartesian_case, testee, "a").extend({IDim: (0, 1)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, out=out, ref=a[1:])


@pytest.mark.uses_cartesian_shift
def test_fold_shifts(cartesian_case):
    """Shifting the result of an addition should work."""

    @gtx.field_operator
    def testee(a: cases.IJKField, b: cases.IJKField) -> cases.IJKField:
        tmp = a + b(IDim + 1)
        return tmp(IDim + 1)

    a = cases.allocate(cartesian_case, testee, "a").extend({cases.IDim: (0, 1)})()
    b = cases.allocate(cartesian_case, testee, "b").extend({cases.IDim: (0, 2)})()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    cases.verify(cartesian_case, testee, a, b, out=out, ref=a.ndarray[1:] + b.ndarray[2:])


@pytest.mark.uses_cartesian_shift
@pytest.mark.uses_dynamic_offsets
def test_offset_field(cartesian_case):
    ref = np.full(
        (cartesian_case.default_sizes[IDim], cartesian_case.default_sizes[KDim]), True, dtype=bool
    )

    @gtx.field_operator
    def testee(a: cases.IKField, offset_field: cases.IKField) -> gtx.Field[[IDim, KDim], bool]:
        a_i = a(as_offset(Ioff, offset_field))
        # note: this leads to an access to offset_field in
        # IDim: (0, out.size[I]), KDim: (0, out.size[K]+1)
        a_i_k = a_i(as_offset(Koff, offset_field))
        b_i = a(IDim + 1)
        b_i_k = b_i(KDim + 1)
        return a_i_k == b_i_k

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    a = cases.allocate(cartesian_case, testee, "a").extend({IDim: (0, 1), KDim: (0, 1)})()
    offset_field = (
        cases.allocate(cartesian_case, testee, "offset_field")
        .strategy(cases.ConstInitializer(1))
        .extend({KDim: (0, 1)})()
    )  # see comment at a_i_k for domain bounds

    cases.verify(
        cartesian_case,
        testee,
        a,
        offset_field,
        out=out,
        ref=ref,
        comparison=lambda out, ref: np.all(out == ref),
    )


@pytest.mark.uses_dynamic_offsets
def test_offset_field_of_chained_ops(cartesian_case):
    """A dynamic offset on top of a chain of operations must be fused past all of them."""

    @gtx.field_operator
    def testee(a: cases.IKField, offset_field: cases.IKField) -> cases.IKField:
        b = a + 1
        c = b * 2
        return c(as_offset(Koff, offset_field))

    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    a = cases.allocate(cartesian_case, testee, "a").extend({KDim: (0, 1)})()
    offset_field = cases.allocate(
        cartesian_case, testee, "offset_field", strategy=cases.ConstInitializer(1)
    )()

    cases.verify(
        cartesian_case,
        testee,
        a,
        offset_field,
        out=out,
        ref=(a.asnumpy()[:, 1:] + 1) * 2,
    )


def _offsets_crossing_k_level_2(k_size: int) -> np.ndarray:
    return np.where(np.arange(k_size) < 2, 2, -2).astype(np.int32)


@pytest.mark.uses_dynamic_offsets
@pytest.mark.uses_concat_where
def test_offset_field_of_concat_where(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IKField, b: cases.IKField, offset_field: cases.IKField) -> cases.IKField:
        return concat_where(KDim < 2, a, b)(as_offset(Koff, offset_field))

    i_size, k_size = cartesian_case.default_sizes[IDim], cartesian_case.default_sizes[KDim]
    k_offsets = _offsets_crossing_k_level_2(k_size)
    offset_field = cartesian_case.as_field(
        [IDim, KDim], np.broadcast_to(k_offsets, (i_size, k_size)).copy()
    )
    a = cases.allocate(cartesian_case, testee, "a")()
    b = cases.allocate(cartesian_case, testee, "b")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    shifted_k = np.arange(k_size) + k_offsets
    ref = np.where(shifted_k < 2, a.asnumpy()[:, shifted_k], b.asnumpy()[:, shifted_k])

    cases.verify(cartesian_case, testee, a, b, offset_field, out=out, ref=ref)


@pytest.mark.uses_dynamic_offsets
@pytest.mark.uses_broadcast_with_dynamic_offsets
def test_offset_field_of_broadcast(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IField, b: cases.IKField, offset_field: cases.IKField) -> cases.IKField:
        return (broadcast(a, (IDim, KDim)) + b)(as_offset(Koff, offset_field))

    i_size, k_size = cartesian_case.default_sizes[IDim], cartesian_case.default_sizes[KDim]
    k_offsets = _offsets_crossing_k_level_2(k_size)
    offset_field = cartesian_case.as_field(
        [IDim, KDim], np.broadcast_to(k_offsets, (i_size, k_size)).copy()
    )
    a = cases.allocate(cartesian_case, testee, "a")()
    b = cases.allocate(cartesian_case, testee, "b")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    shifted_k = np.arange(k_size) + k_offsets
    ref = a.asnumpy()[:, np.newaxis] + b.asnumpy()[:, shifted_k]

    cases.verify(cartesian_case, testee, a, b, offset_field, out=out, ref=ref)


@pytest.mark.uses_dynamic_offsets
@pytest.mark.uses_if_stmts
@pytest.mark.parametrize("cond", [True, False])
def test_offset_field_of_if_stmt_tuple_element(cartesian_case, cond):
    @gtx.field_operator
    def testee(
        a: cases.IKField, b: cases.IKField, offset_field: cases.IKField, cond: bool
    ) -> cases.IKField:
        if cond:
            x, y = a, b * 2
        else:
            x, y = b, a + 1
        return y(as_offset(Koff, offset_field)) + x

    i_size, k_size = cartesian_case.default_sizes[IDim], cartesian_case.default_sizes[KDim]
    k_offsets = _offsets_crossing_k_level_2(k_size)
    offset_field = cartesian_case.as_field(
        [IDim, KDim], np.broadcast_to(k_offsets, (i_size, k_size)).copy()
    )
    a = cases.allocate(cartesian_case, testee, "a")()
    b = cases.allocate(cartesian_case, testee, "b")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    shifted_k = np.arange(k_size) + k_offsets
    a_np, b_np = a.asnumpy(), b.asnumpy()
    ref = b_np[:, shifted_k] * 2 + a_np if cond else a_np[:, shifted_k] + 1 + b_np

    cases.verify(cartesian_case, testee, a, b, offset_field, cond, out=out, ref=ref)


@pytest.mark.uses_dynamic_offsets
def test_offset_field_of_shared_producer(cartesian_case):
    @gtx.field_operator
    def testee(a: cases.IKField, b: cases.IKField, offset_field: cases.IKField) -> cases.IKField:
        x = b * 3
        p = (a + x) * 2
        q = (b + x) * 4
        return p(as_offset(Koff, offset_field)) + q(as_offset(Koff, offset_field))

    i_size, k_size = cartesian_case.default_sizes[IDim], cartesian_case.default_sizes[KDim]
    k_offsets = _offsets_crossing_k_level_2(k_size)
    offset_field = cartesian_case.as_field(
        [IDim, KDim], np.broadcast_to(k_offsets, (i_size, k_size)).copy()
    )
    a = cases.allocate(cartesian_case, testee, "a")()
    b = cases.allocate(cartesian_case, testee, "b")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    shifted_k = np.arange(k_size) + k_offsets
    a_s, b_s = a.asnumpy()[:, shifted_k], b.asnumpy()[:, shifted_k]
    ref = (a_s + 3 * b_s) * 2 + (b_s + 3 * b_s) * 4

    cases.verify(cartesian_case, testee, a, b, offset_field, out=out, ref=ref)


@pytest.mark.uses_dynamic_offsets
@pytest.mark.parametrize("mode", [1, 3])
def test_offset_field_of_ternary_with_scalar_cond_expr(cartesian_case, mode):
    @gtx.field_operator
    def testee(
        a: cases.IKField, b: cases.IKField, offset_field: cases.IKField, mode: gtx.int32
    ) -> cases.IKField:
        return (a if mode == 3 else b * 2)(as_offset(Koff, offset_field))

    i_size, k_size = cartesian_case.default_sizes[IDim], cartesian_case.default_sizes[KDim]
    k_offsets = _offsets_crossing_k_level_2(k_size)
    offset_field = cartesian_case.as_field(
        [IDim, KDim], np.broadcast_to(k_offsets, (i_size, k_size)).copy()
    )
    a = cases.allocate(cartesian_case, testee, "a")()
    b = cases.allocate(cartesian_case, testee, "b")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()

    shifted_k = np.arange(k_size) + k_offsets
    ref = a.asnumpy()[:, shifted_k] if mode == 3 else b.asnumpy()[:, shifted_k] * 2

    cases.verify(cartesian_case, testee, a, b, offset_field, gtx.int32(mode), out=out, ref=ref)
