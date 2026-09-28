# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import NamedTuple

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import common, scan
from gt4py.next.embedded import operators
from gt4py.next.ffront import fbuiltins


IDim = common.Dimension("IDim")
JDim = common.Dimension("JDim")
KDim = common.Dimension("KDim", kind=common.DimensionKind.VERTICAL)


def _field(array, domain):
    return common._field(array, domain=common.domain(domain))


def _weighted_sum(carry, inp, weight):
    return carry * 0.5 + inp * weight


def _sum_and_max(carry, inp):
    return carry[0] + inp, carry[1] * 0.0 + (carry[1] + inp) * 0.5


class _State(NamedTuple):
    total: float
    mean: float


def _running_mean(carry, inp):
    return _State(total=carry.total + inp, mean=(carry.mean + inp) * 0.5)


def _scan(fun, forward, init, k_range, per_column, *args, **kwargs):
    scan_range = common.NamedRange(KDim, common.UnitRange(*k_range))
    return operators.ScanOperator(fun, forward, init, scan_range, per_column=per_column)(
        *args, **kwargs
    )


def _as_numpy(result):
    return tuple(_as_numpy(r) for r in result) if isinstance(result, tuple) else result.asnumpy()


@pytest.fixture
def ijk_fields():
    rng = np.random.default_rng(42)
    inp = _field(rng.normal(size=(4, 3, 6)), {IDim: (0, 4), JDim: (0, 3), KDim: (0, 6)})
    weight = _field(rng.normal(size=(5, 3)), {IDim: (-1, 4), JDim: (0, 3)})
    column_init = _field(rng.normal(size=(3,)), {JDim: (0, 3)})
    return inp, weight, column_init


@pytest.mark.parametrize("forward", [True, False])
@pytest.mark.parametrize("k_range", [(0, 6), (1, 4)])
@pytest.mark.parametrize("init", ["scalar", "column"])
def test_sliced_matches_per_column(ijk_fields, forward, k_range, init):
    inp, weight, column_init = ijk_fields
    init_value = 0.25 if init == "scalar" else column_init

    results = [
        _scan(_weighted_sum, forward, init_value, k_range, per_column, inp, weight=weight)
        for per_column in (True, False)
    ]

    assert results[0].domain == results[1].domain
    np.testing.assert_allclose(results[1].asnumpy(), results[0].asnumpy())


@pytest.mark.parametrize("forward", [True, False])
def test_sliced_matches_per_column_tuple_carry(ijk_fields, forward):
    inp, _, column_init = ijk_fields

    results = [
        _scan(_sum_and_max, forward, (column_init, 1.0), (0, 6), per_column, inp)
        for per_column in (True, False)
    ]

    for per_column_result, sliced_result in zip(*map(_as_numpy, results), strict=True):
        np.testing.assert_allclose(sliced_result, per_column_result)


def test_sliced_matches_per_column_named_collection_carry(ijk_fields):
    inp, _, column_init = ijk_fields

    results = [
        _scan(_running_mean, True, _State(total=column_init, mean=0.0), (0, 6), per_column, inp)
        for per_column in (True, False)
    ]

    assert isinstance(results[1], _State)
    for per_column_result, sliced_result in zip(*results, strict=True):
        np.testing.assert_allclose(sliced_result.asnumpy(), per_column_result.asnumpy())


def test_sliced_without_field_args():
    result = _scan(lambda carry, x: carry + x, True, 1.0, (2, 5), False, 2.0)

    assert result.domain == common.domain({KDim: (2, 5)})
    np.testing.assert_allclose(result.asnumpy(), [3.0, 5.0, 7.0])


@pytest.mark.requires_jax
@pytest.mark.parametrize("forward", [True, False])
@pytest.mark.parametrize("init", ["scalar", "column"])
def test_jax_matches_numpy(ijk_fields, forward, init):
    import jax.numpy as jnp

    inp, weight, column_init = ijk_fields
    to_jax = lambda f: common._field(jnp.asarray(f.ndarray), domain=f.domain)
    init_value = 0.25 if init == "scalar" else column_init

    expected = _scan(_weighted_sum, forward, init_value, (1, 5), True, inp, weight=weight)
    result = _scan(
        _weighted_sum,
        forward,
        init_value if init == "scalar" else to_jax(column_init),
        (1, 5),
        False,
        to_jax(inp),
        weight=to_jax(weight),
    )

    assert result.domain == expected.domain
    np.testing.assert_allclose(result.asnumpy(), expected.asnumpy())


@pytest.mark.requires_jax
def test_jax_tuple_carry(ijk_fields):
    import jax.numpy as jnp

    inp, _, column_init = ijk_fields
    to_jax = lambda f: common._field(jnp.asarray(f.ndarray), domain=f.domain)

    expected = _scan(_sum_and_max, False, (column_init, 1.0), (0, 6), True, inp)
    result = _scan(_sum_and_max, False, (to_jax(column_init), 1.0), (0, 6), False, to_jax(inp))

    for expected_el, result_el in zip(_as_numpy(expected), _as_numpy(result), strict=True):
        np.testing.assert_allclose(result_el, expected_el)


@pytest.mark.requires_jax
def test_jax_named_collection_carry(ijk_fields):
    import jax.numpy as jnp

    inp, _, column_init = ijk_fields
    to_jax = lambda f: common._field(jnp.asarray(f.ndarray), domain=f.domain)

    expected = _scan(_running_mean, False, _State(column_init, 0.0), (1, 5), True, inp)
    result = _scan(
        _running_mean, False, _State(to_jax(column_init), 0.0), (1, 5), False, to_jax(inp)
    )

    assert isinstance(result, _State)
    for expected_el, result_el in zip(expected, result, strict=True):
        np.testing.assert_allclose(result_el.asnumpy(), expected_el.asnumpy())


@pytest.mark.requires_jax
def test_jax_jit_and_grad():
    import jax
    import jax.numpy as jnp

    domain = {IDim: (0, 2), KDim: (0, 5)}
    array = jnp.asarray(np.arange(10, dtype=np.float64).reshape(2, 5))

    def column_sum(array):
        inp = _field(array, domain)
        return _scan(lambda carry, x: carry + x, True, 0.0, (0, 5), False, inp).ndarray.sum()

    np.testing.assert_allclose(
        jax.jit(column_sum)(array), np.cumsum(np.asarray(array), axis=1).sum()
    )
    # every input level contributes to all levels at and above it
    np.testing.assert_allclose(
        jax.jit(jax.grad(column_sum))(array), np.broadcast_to([5.0, 4.0, 3.0, 2.0, 1.0], (2, 5))
    )


@pytest.mark.requires_jax
def test_jax_value_branches_raise():
    import jax.numpy as jnp

    @gtx.field_operator
    def ternary(carry: float, x: float) -> float:
        return carry if carry > x else x

    @gtx.field_operator
    def testee(a: gtx.Field[[IDim, KDim], float]) -> gtx.Field[[IDim, KDim], float]:
        return scan(ternary, range=(KDim, 0, 3), init=0.0)(a)

    domain = {IDim: (0, 2), KDim: (0, 3)}
    with pytest.raises(ValueError, match="branches on values"):
        testee(
            _field(jnp.ones((2, 3)), domain),
            out=_field(jnp.zeros((2, 3)), domain),
            offset_provider={},
        )


def test_branches_on_values():
    @gtx.field_operator
    def arithmetic(carry: float, x: float) -> float:
        return carry + x

    @gtx.field_operator
    def ternary(carry: float, x: float) -> float:
        return carry if carry > x else x

    @gtx.field_operator
    def if_stmt(carry: float, x: float) -> float:
        if carry > x:
            result = carry
        else:
            result = x
        return result

    @gtx.field_operator
    def calls_ternary(carry: float, x: float) -> float:
        return ternary(carry, x) + 1.0

    assert not fbuiltins._branches_on_values(arithmetic)
    assert not fbuiltins._branches_on_values(lambda carry, x: carry + x)
    assert fbuiltins._branches_on_values(ternary)
    assert fbuiltins._branches_on_values(if_stmt)
    assert fbuiltins._branches_on_values(calls_ternary)


@pytest.mark.requires_jax
def test_jax_jit_and_grad_field_operator():
    import jax
    import jax.numpy as jnp

    @gtx.field_operator
    def damped_sum(carry: float, x: float) -> float:
        return 0.5 * carry + x

    @gtx.field_operator
    def testee(a: gtx.Field[[IDim, KDim], float]) -> gtx.Field[[IDim, KDim], float]:
        return scan(damped_sum, range=(KDim, 0, 4), forward=False, init=0.0)(a)

    def loss(array):
        domain = {IDim: (0, 3), KDim: (0, 4)}
        out = _field(jnp.zeros((3, 4)), domain)
        testee(_field(array, domain), out=out, offset_provider={})
        return out.ndarray.sum()

    array = jnp.asarray(np.arange(12, dtype=np.float64).reshape(3, 4))
    expected = np.zeros((3, 4))
    carry = np.zeros(3)
    for k in reversed(range(4)):
        carry = 0.5 * carry + np.asarray(array)[:, k]
        expected[:, k] = carry

    np.testing.assert_allclose(jax.jit(loss)(array), expected.sum())
    # d(sum)/d(x_k) = sum over levels l <= k of 0.5**(k - l)
    grad = [sum(0.5 ** (k - l) for l in range(k + 1)) for k in range(4)]
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(array), np.broadcast_to(grad, (3, 4)))
