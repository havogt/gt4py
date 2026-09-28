# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from functools import reduce

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import common, errors, float64, int32, maximum, neighbor_sum, scan
from gt4py.next.experimental import concat_where

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    Edge,
    IDim,
    JDim,
    KDim,
    KHalfDim,
    V2E,
    V2EDim,
    Vertex,
    cartesian_case,
    unstructured_case,
    unstructured_case_3d,
)
from next_tests.integration_tests.cases_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)


@pytest.mark.uses_scan
def test_scalar_scan(cartesian_case):
    @gtx.field_operator
    def testee_pass(state: float, qc_in: float, scalar: float) -> float:
        qc = qc_in + state + scalar
        return qc

    @gtx.field_operator
    def testee_scan(qc: cases.IKFloatField, scalar: float) -> cases.IKFloatField:
        return scan(testee_pass, range=(KDim, 0, 9), forward=True, init=0.0)(qc, scalar)

    @gtx.program
    def testee(qc: cases.IKFloatField, scalar: float):
        testee_scan(qc, scalar, out=qc)

    qc = cases.allocate(cartesian_case, testee, "qc").zeros()()
    scalar = 1.0
    isize = cartesian_case.default_sizes[IDim]
    ksize = cartesian_case.default_sizes[KDim]
    expected = np.full((isize, ksize), np.arange(start=1, stop=ksize + 1, step=1).astype(float64))

    cases.verify(cartesian_case, testee, qc, scalar, inout=qc, ref=expected)


@pytest.mark.uses_scan
@pytest.mark.uses_tuple_args
def test_tuple_scalar_scan(cartesian_case):
    @gtx.field_operator
    def testee_pass(
        state: float, qc_in: float, tuple_scalar: tuple[float, tuple[float, float]]
    ) -> float:
        return (qc_in + state + tuple_scalar[1][0] + tuple_scalar[1][1]) / tuple_scalar[0]

    @gtx.field_operator
    def testee_op(
        qc: cases.IKFloatField, tuple_scalar: tuple[float, tuple[float, float]]
    ) -> cases.IKFloatField:
        return scan(testee_pass, range=(KDim, 0, 9), forward=True, init=0.0)(qc, tuple_scalar)

    qc = cases.allocate(cartesian_case, testee_op, "qc").zeros()()
    tuple_scalar = (1.0, (1.0, 0.0))
    isize = cartesian_case.default_sizes[IDim]
    ksize = cartesian_case.default_sizes[KDim]
    expected = np.full((isize, ksize), np.arange(start=1.0, stop=ksize + 1), dtype=float)
    cases.verify(cartesian_case, testee_op, qc, tuple_scalar, out=qc, ref=expected)


@pytest.mark.uses_cartesian_shift
@pytest.mark.uses_scan
def test_scalar_scan_vertical_offset(cartesian_case):
    @gtx.field_operator
    def testee_pass(state: float, inp: float) -> float:
        return inp

    @gtx.field_operator
    def testee(inp: gtx.Field[[KDim], float]) -> gtx.Field[[KDim], float]:
        return scan(testee_pass, range=(KDim, 0, 9), forward=True, init=0.0)(inp(KDim + 1))

    inp = cases.allocate(
        cartesian_case,
        testee,
        "inp",
        extend={KDim: (0, 1)},
        strategy=cases.UniqueInitializer(start=2),
    )()
    out = cases.allocate(cartesian_case, testee, "inp").zeros()()
    ksize = cartesian_case.default_sizes[KDim]
    expected = np.full((ksize), np.arange(start=3, stop=ksize + 3, step=1).astype(float64))

    cases.verify(cartesian_case, testee, inp, out=out, ref=expected)


@pytest.mark.uses_scan
def test_scan_unused_parameter(cartesian_case):
    @gtx.field_operator
    def testee_pass(state: float, inp: float, unused: float) -> float:
        return state + inp

    @gtx.field_operator
    def testee(
        inp: gtx.Field[[KDim], float], unused: gtx.Field[[KDim], float]
    ) -> gtx.Field[[KDim], float]:
        return scan(testee_pass, range=(KDim, 0, 9), forward=True, init=0.0)(inp, unused)

    inp = cases.allocate(cartesian_case, testee, "inp")()
    unused = cases.allocate(cartesian_case, testee, "unused")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN).zeros()()

    cases.verify(
        cartesian_case,
        testee,
        inp,
        unused,
        out=out,
        ref=np.cumsum(inp.asnumpy(), axis=0),
    )


@pytest.mark.uses_scan
@pytest.mark.uses_scan_without_field_args
@pytest.mark.parametrize("forward", [True, False])
def test_fieldop_from_scan(cartesian_case, forward):
    expected = np.arange(2.0, 2.0 + cartesian_case.default_sizes[KDim], 1)
    out = cartesian_case.as_field([KDim], np.zeros((cartesian_case.default_sizes[KDim],)))

    if not forward:
        expected = np.flip(expected)

    @gtx.field_operator
    def add(carry: float, foo: float) -> float:
        return carry + foo

    @gtx.field_operator
    def scan_pass(carry: float) -> float:
        return add(carry, 1.0)

    @gtx.field_operator
    def forward_scan() -> gtx.Field[[KDim], float]:
        return scan(scan_pass, range=(KDim, 0, 9), forward=True, init=1.0)()

    @gtx.field_operator
    def backward_scan() -> gtx.Field[[KDim], float]:
        return scan(scan_pass, range=(KDim, 0, 9), forward=False, init=1.0)()

    cases.verify(cartesian_case, forward_scan if forward else backward_scan, out=out, ref=expected)


@pytest.mark.uses_scan
@pytest.mark.uses_scan_nested
def test_solve_triag(cartesian_case):
    @gtx.field_operator
    def tridiag_forward(
        state: tuple[float, float], a: float, b: float, c: float, d: float
    ) -> tuple[float, float]:
        return (c / (b - a * state[0]), (d - a * state[1]) / (b - a * state[0]))

    @gtx.field_operator
    def tridiag_backward(x_kp1: float, cp: float, dp: float) -> float:
        return dp - cp * x_kp1

    @gtx.field_operator
    def solve_tridiag(
        a: cases.IJKFloatField,
        b: cases.IJKFloatField,
        c: cases.IJKFloatField,
        d: cases.IJKFloatField,
    ) -> cases.IJKFloatField:
        cp, dp = scan(tridiag_forward, range=(KDim, 0, 9), forward=True, init=(0.0, 0.0))(
            a, b, c, d
        )
        return scan(tridiag_backward, range=(KDim, 0, 9), forward=False, init=0.0)(cp, dp)

    def expected(a, b, c, d):
        shape = tuple(cartesian_case.default_sizes[dim] for dim in [IDim, JDim, KDim])
        matrices = np.zeros(shape + shape[-1:])
        i = np.arange(shape[2])
        matrices[:, :, i[1:], i[:-1]] = a[:, :, 1:]
        matrices[:, :, i, i] = b
        matrices[:, :, i[:-1], i[1:]] = c[:, :, :-1]
        # Changed in NumPY version 2.0: In a linear matrix equation ax = b, the b array
        # is only treated as a shape (M,) column vector if it is exactly 1-dimensional.
        # In all other instances it is treated as a stack of (M, K) matrices. Therefore
        # below we add an extra dimension (K) of size 1. Previously b would be treated
        # as a stack of (M,) vectors if b.ndim was equal to a.ndim - 1.
        # Refer to https://numpy.org/doc/2.0/reference/generated/numpy.linalg.solve.html
        d_ext = np.empty(shape=(*shape, 1))
        d_ext[:, :, :, 0] = d
        x = np.linalg.solve(matrices, d_ext)
        return x[:, :, :, 0]

    cases.verify_with_default_data(cartesian_case, solve_tridiag, ref=expected)


@pytest.mark.uses_scan
def test_ternary_scan(cartesian_case):
    @gtx.field_operator
    def scan_pass(carry: float, a: float) -> float:
        return carry if carry > a else carry + 1.0

    @gtx.field_operator
    def simple_scan_operator(a: gtx.Field[[KDim], float]) -> gtx.Field[[KDim], float]:
        return scan(scan_pass, range=(KDim, 0, 9), forward=True, init=0.0)(a)

    k_size = cartesian_case.default_sizes[KDim]
    a = cartesian_case.as_field([KDim], 4.0 * np.ones((k_size,)))
    out = cartesian_case.as_field([KDim], np.zeros((k_size,)))

    cases.verify(
        cartesian_case,
        simple_scan_operator,
        a,
        out=out,
        ref=np.asarray([i if i <= 4.0 else 4.0 + 1 for i in range(1, k_size + 1)]),
    )


@pytest.mark.parametrize("forward", [True, False])
@pytest.mark.uses_scan
@pytest.mark.uses_scan_without_field_args
@pytest.mark.uses_tuple_returns
def test_scan_nested_tuple_output(forward, cartesian_case):
    k_size = cartesian_case.default_sizes[KDim]
    expected = np.arange(1, 1 + k_size, 1, dtype=int32)
    if not forward:
        expected = np.flip(expected)

    @gtx.field_operator
    def scan_pass(
        carry: tuple[int32, tuple[int32, int32]],
    ) -> tuple[int32, tuple[int32, int32]]:
        return (carry[0] + 1, (carry[1][0] + 1, carry[1][1] + 1))

    @gtx.field_operator
    def forward_scan() -> tuple[cases.KField, tuple[cases.KField, cases.KField]]:
        return scan(scan_pass, range=(KDim, 0, 9), forward=True, init=(1, (2, 3)))()

    @gtx.field_operator
    def backward_scan() -> tuple[cases.KField, tuple[cases.KField, cases.KField]]:
        return scan(scan_pass, range=(KDim, 0, 9), forward=False, init=(1, (2, 3)))()

    @gtx.program
    def testee_forward(out: tuple[cases.KField, tuple[cases.KField, cases.KField]]):
        forward_scan(out=out)

    @gtx.program
    def testee_backward(out: tuple[cases.KField, tuple[cases.KField, cases.KField]]):
        backward_scan(out=out)

    cases.verify_with_default_data(
        cartesian_case,
        testee_forward if forward else testee_backward,
        ref=lambda: (expected + 1.0, (expected + 2.0, expected + 3.0)),
        comparison=lambda ref, out: (
            np.all(out[0] == ref[0])
            and np.all(out[1][0] == ref[1][0])
            and np.all(out[1][1] == ref[1][1])
        ),
    )


@pytest.mark.uses_scan
@pytest.mark.uses_tuple_args
def test_scan_nested_tuple_input(cartesian_case):
    init = 1.0
    k_size = cartesian_case.default_sizes[KDim]

    inp1_np = np.ones((k_size,))
    inp2_np = np.arange(0.0, k_size, 1)
    inp1 = cartesian_case.as_field([KDim], inp1_np)
    inp2 = cartesian_case.as_field([KDim], inp2_np)
    out = cartesian_case.as_field([KDim], np.zeros((k_size,)))

    def prev_levels_iterator(i):
        return range(i + 1)

    expected = np.asarray(
        [
            reduce(lambda prev, i: prev + inp1_np[i] + inp2_np[i], prev_levels_iterator(i), init)
            for i in range(k_size)
        ]
    )

    @gtx.field_operator
    def scan_pass(carry: float, a: tuple[float, float]) -> float:
        return carry + a[0] + a[1]

    @gtx.field_operator
    def simple_scan_operator(
        a: tuple[gtx.Field[[KDim], float], gtx.Field[[KDim], float]],
    ) -> gtx.Field[[KDim], float]:
        return scan(scan_pass, range=(KDim, 0, 9), forward=True, init=1.0)(a)

    cases.verify(cartesian_case, simple_scan_operator, (inp1, inp2), out=out, ref=expected)


@pytest.mark.uses_scan
@pytest.mark.uses_tuple_args
def test_scan_different_domain_in_tuple(cartesian_case):
    init = 1.0
    i_size = cartesian_case.default_sizes[IDim]
    k_size = cartesian_case.default_sizes[KDim]

    inp1_np = np.ones((i_size + 1, k_size))  # i_size bigger than in the other argument
    inp2_np = np.fromfunction(lambda i, k: k, shape=(i_size, k_size), dtype=float)
    inp1 = cartesian_case.as_field([IDim, KDim], inp1_np)
    inp2 = cartesian_case.as_field([IDim, KDim], inp2_np)
    out = cartesian_case.as_field([IDim, KDim], np.zeros((i_size, k_size)))

    def prev_levels_iterator(i):
        return range(i + 1)

    expected = np.asarray(
        [
            reduce(
                lambda prev, k: prev + inp1_np[:-1, k] + inp2_np[:, k],
                prev_levels_iterator(k),
                init,
            )
            for k in range(k_size)
        ]
    ).transpose()

    @gtx.field_operator
    def scan_pass(carry: float, a: tuple[float, float]) -> float:
        return carry + a[0] + a[1]

    @gtx.field_operator
    def foo(
        inp1: gtx.Field[[IDim, KDim], float], inp2: gtx.Field[[IDim, KDim], float]
    ) -> gtx.Field[[IDim, KDim], float]:
        return scan(scan_pass, range=(KDim, 0, 9), forward=True, init=1.0)((inp1, inp2))

    cases.verify(cartesian_case, foo, inp1, inp2, out=out, ref=expected)


@pytest.mark.uses_scan
@pytest.mark.uses_tuple_args
def test_scan_tuple_field_scalar_mixed(cartesian_case):
    init = 1.0
    i_size = cartesian_case.default_sizes[IDim]
    k_size = cartesian_case.default_sizes[KDim]

    inp2_np = np.fromfunction(lambda i, k: k, shape=(i_size, k_size), dtype=float)
    inp2 = cartesian_case.as_field([IDim, KDim], inp2_np)
    out = cartesian_case.as_field([IDim, KDim], np.zeros((i_size, k_size)))

    def prev_levels_iterator(i):
        return range(i + 1)

    expected = np.asarray(
        [
            reduce(lambda prev, k: prev + 1.0 + inp2_np[:, k], prev_levels_iterator(k), init)
            for k in range(k_size)
        ]
    ).transpose()

    @gtx.field_operator
    def scan_pass(carry: float, a: tuple[float, float]) -> float:
        return carry + a[0] + a[1]

    @gtx.field_operator
    def foo(inp1: float, inp2: gtx.Field[[IDim, KDim], float]) -> gtx.Field[[IDim, KDim], float]:
        return scan(scan_pass, range=(KDim, 0, 9), forward=True, init=1.0)((inp1, inp2))

    cases.verify(cartesian_case, foo, 1.0, inp2, out=out, ref=expected)


@pytest.mark.uses_scan
def test_scan_wrong_return_type(cartesian_case):
    with pytest.raises(
        errors.DSLError,
        match=(r"Argument 'state' to scan pass 'testee_pass' must have same type as its return"),
    ):

        @gtx.field_operator
        def testee_pass(state: int32) -> float:
            return 1.0

        @gtx.field_operator
        def testee(qc: cases.IKFloatField) -> cases.IKFloatField:
            return scan(testee_pass, range=(KDim, 0, 9), forward=True, init=0)()


@pytest.mark.uses_scan
def test_scan_wrong_init_type(cartesian_case):
    with pytest.raises(
        errors.DSLError,
        match=(
            r"Argument 'init' to scan pass 'testee_pass' must have same type as 'state' argument"
        ),
    ):

        @gtx.field_operator
        def testee_pass(state: float) -> float:
            return 1.0

        @gtx.field_operator
        def testee(qc: cases.IKFloatField) -> cases.IKFloatField:
            return scan(testee_pass, range=(KDim, 0, 9), forward=True, init=0)()


@pytest.mark.uses_scan
def test_scan_without_carry(cartesian_case):
    with pytest.raises(
        errors.DSLError,
        match=r"Scan pass 'testee_pass' must have at least one argument",
    ):

        @gtx.field_operator
        def testee_pass() -> float:
            return 1.0

        @gtx.field_operator
        def testee(qc: cases.IKFloatField) -> cases.IKFloatField:
            return scan(testee_pass, range=(KDim, 0, 9), forward=True, init=0.0)()


@pytest.mark.uses_scan
def test_scan_call(cartesian_case):
    @gtx.field_operator
    def add(carry: float, inp: float) -> float:
        return carry + inp

    @gtx.field_operator
    def testee(inp: cases.IKFloatField) -> cases.IKFloatField:
        return scan(add, range=(KDim, 0, 9), forward=True, init=1.0)(inp)

    inp = cases.allocate(cartesian_case, testee, "inp")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN).zeros()()

    cases.verify(cartesian_case, testee, inp, out=out, ref=1.0 + np.cumsum(inp.asnumpy(), axis=1))


@pytest.mark.uses_scan
def test_scan_call_backward(cartesian_case):
    @gtx.field_operator
    def add(carry: float, inp: float, scalar: float) -> float:
        return carry + inp * scalar

    @gtx.field_operator
    def testee(inp: cases.IKFloatField, scalar: float) -> cases.IKFloatField:
        return scan(add, range=(KDim, 0, 9), forward=False, init=-1.0)(inp, scalar) + inp

    inp = cases.allocate(cartesian_case, testee, "inp")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN).zeros()()
    inp_np = inp.asnumpy()
    backward_cumsum = np.flip(np.cumsum(np.flip(2.0 * inp_np, axis=1), axis=1), axis=1)

    cases.verify(cartesian_case, testee, inp, 2.0, out=out, ref=backward_cumsum - 1.0 + inp_np)


@pytest.mark.uses_scan
@pytest.mark.uses_tuple_returns
def test_scan_call_tuple_carry(cartesian_case):
    @gtx.field_operator
    def sum_and_count(carry: tuple[float, int32], inp: float) -> tuple[float, int32]:
        return carry[0] + inp, carry[1] + int32(1)

    @gtx.field_operator
    def testee(inp: cases.IKFloatField) -> tuple[cases.IKFloatField, cases.IKField]:
        return scan(sum_and_count, range=(KDim, 0, 9), init=(0.0, int32(0)))(inp)

    inp = cases.allocate(cartesian_case, testee, "inp")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN).zeros()()
    isize = cartesian_case.default_sizes[IDim]
    ksize = cartesian_case.default_sizes[KDim]

    cases.verify(
        cartesian_case,
        testee,
        inp,
        out=out,
        ref=(
            np.cumsum(inp.asnumpy(), axis=1),
            np.full((isize, ksize), np.arange(1, ksize + 1, dtype=np.int32)),
        ),
    )


@pytest.mark.uses_scan
def test_scan_call_scalar_init(cartesian_case):
    @gtx.field_operator
    def add(carry: float, inp: float) -> float:
        return carry + inp

    @gtx.field_operator
    def testee(inp: cases.IKFloatField, init: float) -> cases.IKFloatField:
        return scan(add, range=(KDim, 0, 9), forward=True, init=init)(inp)

    inp = cases.allocate(cartesian_case, testee, "inp")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN).zeros()()

    cases.verify(
        cartesian_case, testee, inp, 0.5, out=out, ref=0.5 + np.cumsum(inp.asnumpy(), axis=1)
    )


@pytest.mark.uses_scan
def test_scan_call_column_init(cartesian_case):
    @gtx.field_operator
    def add(carry: float, inp: float) -> float:
        return carry + inp

    @gtx.field_operator
    def testee(inp: cases.IKFloatField, init: cases.IFloatField) -> cases.IKFloatField:
        return scan(add, range=(KDim, 0, 9), forward=False, init=init)(inp)

    inp = cases.allocate(cartesian_case, testee, "inp")()
    init = cartesian_case.as_field(
        [IDim], np.arange(cartesian_case.default_sizes[IDim], dtype=float64) + 0.25
    )
    out = cases.allocate(cartesian_case, testee, cases.RETURN).zeros()()
    backward_cumsum = np.flip(np.cumsum(np.flip(inp.asnumpy(), axis=1), axis=1), axis=1)

    cases.verify(
        cartesian_case,
        testee,
        inp,
        init,
        out=out,
        ref=init.asnumpy()[:, np.newaxis] + backward_cumsum,
    )


@pytest.mark.uses_scan
@pytest.mark.uses_tuple_returns
def test_scan_call_tuple_init_mixed(cartesian_case):
    @gtx.field_operator
    def sum_and_max(carry: tuple[float, float], inp: float) -> tuple[float, float]:
        return carry[0] + inp, maximum(carry[1], inp)

    @gtx.field_operator
    def testee(
        inp: cases.IKFloatField, init: cases.IFloatField
    ) -> tuple[cases.IKFloatField, cases.IKFloatField]:
        return scan(sum_and_max, range=(KDim, 0, 9), forward=True, init=(1.0, init))(inp)

    inp = cases.allocate(cartesian_case, testee, "inp")()
    init = cartesian_case.as_field(
        [IDim], np.arange(cartesian_case.default_sizes[IDim], dtype=float64) + 0.25
    )
    out = cases.allocate(cartesian_case, testee, cases.RETURN).zeros()()
    inp_np = inp.asnumpy()

    cases.verify(
        cartesian_case,
        testee,
        inp,
        init,
        out=out,
        ref=(
            1.0 + np.cumsum(inp_np, axis=1),
            np.maximum.accumulate(
                np.concatenate([init.asnumpy()[:, np.newaxis], inp_np], axis=1), axis=1
            )[:, 1:],
        ),
    )


@pytest.mark.uses_scan
@pytest.mark.uses_concat_where
@pytest.mark.parametrize("k_start", [0, 2])
def test_scan_range_under_concat_where(cartesian_case, k_start):
    """The scan covers exactly its range, whatever part of it the consumer reads."""

    @gtx.field_operator
    def add(carry: float, inp: float) -> float:
        return carry + inp

    @gtx.field_operator
    def testee(inp: cases.IKFloatField, k_start: int32) -> cases.IKFloatField:
        return concat_where(
            KDim >= 2, scan(add, range=(KDim, k_start, 9), forward=True, init=0.0)(inp), -inp
        )

    inp = cases.allocate(cartesian_case, testee, "inp")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN).zeros()()
    inp_np = inp.asnumpy()
    ref = -inp_np.copy()
    ref[:, 2:] = np.cumsum(inp_np[:, k_start:], axis=1)[:, 2 - k_start :]

    cases.verify(cartesian_case, testee, inp, k_start, out=out, ref=ref)


@pytest.mark.uses_scan
@pytest.mark.uses_concat_where
def test_scan_range_runtime_bounds_backward(cartesian_case):
    @gtx.field_operator
    def add(carry: float, inp: float) -> float:
        return carry + inp

    @gtx.field_operator
    def testee(inp: cases.IKFloatField, k_start: int32, k_end: int32) -> cases.IKFloatField:
        return concat_where(
            (KDim >= k_start) & (KDim < k_end),
            scan(add, range=(KDim, k_start, k_end), forward=False, init=0.0)(inp),
            inp,
        )

    inp = cases.allocate(cartesian_case, testee, "inp")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN).zeros()()
    inp_np = inp.asnumpy()
    ref = inp_np.copy()
    ref[:, 3:7] = np.flip(np.cumsum(np.flip(inp_np[:, 3:7], axis=1), axis=1), axis=1)

    cases.verify(cartesian_case, testee, inp, 3, 7, out=out, ref=ref)


@pytest.mark.uses_scan
def test_scan_range_larger_than_output(cartesian_case):
    @gtx.field_operator
    def add(carry: float, inp: float) -> float:
        return carry + inp

    @gtx.field_operator
    def testee(inp: cases.IKFloatField) -> cases.IKFloatField:
        return scan(add, range=(KDim, 0, 9), forward=True, init=0.0)(inp)

    @gtx.program
    def prog(inp: cases.IKFloatField, out: cases.IKFloatField):
        testee(inp, out=out, domain={IDim: (0, 5), KDim: (2, 5)})

    inp = cases.allocate(cartesian_case, prog, "inp")()
    out = cases.allocate(cartesian_case, prog, "out").zeros()()
    ref = np.zeros_like(inp.asnumpy())
    ref[:, 2:5] = np.cumsum(inp.asnumpy(), axis=1)[:, 2:5]

    cases.verify(cartesian_case, prog, inp, out, inout=out, ref=ref)


@pytest.mark.uses_scan
@pytest.mark.uses_tuple_returns
@pytest.mark.uses_program_with_sliced_out_arguments
def test_scan_range_tuple_outputs_on_different_domains(cartesian_case):
    @gtx.field_operator
    def sum_and_double(carry: tuple[float, float], inp: float) -> tuple[float, float]:
        return carry[0] + inp, carry[1] + 2.0 * inp

    @gtx.field_operator
    def testee(inp: cases.IKFloatField) -> tuple[cases.IKFloatField, cases.IKFloatField]:
        return scan(sum_and_double, range=(KDim, 0, 9), forward=True, init=(0.0, 0.0))(inp)

    @gtx.program
    def prog(inp: cases.IKFloatField, out1: cases.IKFloatField, out2: cases.IKFloatField):
        testee(inp, out=(out1, out2[:, 2:7]))

    inp = cases.allocate(cartesian_case, prog, "inp")()
    out1 = cases.allocate(cartesian_case, prog, "out1").zeros()()
    out2 = cases.allocate(cartesian_case, prog, "out2").zeros()()
    cumsum = np.cumsum(inp.asnumpy(), axis=1)
    ref2 = np.zeros_like(cumsum)
    ref2[:, 2:7] = 2.0 * cumsum[:, 2:7]

    cases.verify(
        cartesian_case,
        prog,
        inp,
        out1,
        out2,
        inout=(out1, out2),
        ref=(cumsum, ref2),
        comparison=lambda ref, out: all(np.allclose(r, o) for r, o in zip(ref, out)),
    )


@pytest.mark.uses_scan
@pytest.mark.uses_unstructured_shift
@pytest.mark.uses_scan_with_unstructured_shift
def test_scan_range_unstructured_backward(unstructured_case_3d):
    @gtx.field_operator
    def add(carry: float, inp: float) -> float:
        return carry + inp

    @gtx.field_operator
    def testee(
        e: gtx.Field[[Edge, KDim], float64], k_start: int32, k_end: int32
    ) -> gtx.Field[[Vertex, KDim], float64]:
        s = scan(add, range=(KDim, k_start, k_end), forward=False, init=0.0)(e)
        return neighbor_sum(s(V2E), axis=V2EDim)

    e = cases.allocate(unstructured_case_3d, testee, "e")()
    out = cases.allocate(unstructured_case_3d, testee, cases.RETURN).zeros()()
    v2e_table = unstructured_case_3d.offset_provider["V2E"].asnumpy()
    backward_cumsum = np.flip(np.cumsum(np.flip(e.asnumpy(), axis=1), axis=1), axis=1)
    ref = np.sum(
        backward_cumsum[v2e_table],
        axis=1,
        initial=0,
        where=(v2e_table != common._DEFAULT_SKIP_VALUE)[:, :, np.newaxis],
    )

    cases.verify(unstructured_case_3d, testee, e, 0, 10, out=out, ref=ref)


@pytest.mark.uses_scan
@pytest.mark.uses_tuple_returns
def test_scans_over_different_vertical_dims(cartesian_case):
    @gtx.field_operator
    def add(carry: float, inp: float) -> float:
        return carry + inp

    @gtx.field_operator
    def testee(
        a: cases.IKFloatField, b: gtx.Field[[IDim, KHalfDim], float64], nk: int32, nhalf: int32
    ) -> tuple[cases.IKFloatField, gtx.Field[[IDim, KHalfDim], float64]]:
        return (
            scan(add, range=(KDim, 0, nk), init=0.0)(a),
            scan(add, range=(KHalfDim, 0, nhalf), forward=False, init=0.0)(b),
        )

    a = cases.allocate(cartesian_case, testee, "a")()
    b = cases.allocate(cartesian_case, testee, "b")()
    out = cases.allocate(cartesian_case, testee, cases.RETURN)()
    backward_cumsum = np.flip(np.cumsum(np.flip(b.asnumpy(), axis=1), axis=1), axis=1)

    cases.verify(
        cartesian_case,
        testee,
        a,
        b,
        int32(cartesian_case.default_sizes[KDim]),
        int32(cartesian_case.default_sizes[KHalfDim]),
        out=out,
        ref=(np.cumsum(a.asnumpy(), axis=1), backward_cumsum),
    )
