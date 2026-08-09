# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.transforms import concat_where
from gt4py.next.type_system import type_specifications as ts


float_type = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
KDim = common.Dimension(value="KDim", kind=common.DimensionKind.VERTICAL)
IDim = common.Dimension(value="IDim", kind=common.DimensionKind.HORIZONTAL)
k_field = ts.FieldType(dims=[KDim], dtype=float_type)

Koff = im.cartesian_offset(KDim, KDim)


def _a():
    return im.ref("a", k_field)


def _b():
    return im.ref("b", k_field)


Ioff = im.cartesian_offset(IDim, IDim)


def _shift(offset, distance, arg):
    """`as_fieldop(lambda it: deref(shift(offset, distance)(it)))(arg)`."""
    return im.as_fieldop(im.lambda_("it")(im.deref(im.shift(offset, distance)("it"))))(arg)


def _k_domain(start, stop):
    return im.domain(common.GridType.CARTESIAN, {KDim: (start, stop)})


def _apply(expr):
    """Runs the pass on a program whose single statement is `expr`."""
    program = itir.Program(
        id="testee",
        function_definitions=[],
        params=[im.sym("out", k_field), im.sym("a", k_field), im.sym("b", k_field)],
        declarations=[],
        body=[itir.SetAt(expr=expr, domain=_k_domain(0, 10), target=im.ref("out", k_field))],
    )
    return concat_where.push_shifts(program).body[0].expr


def test_shift_is_distributed_and_condition_translated():
    testee = _shift(
        Koff, 1, im.concat_where(_k_domain(itir.InfinityLiteral.NEGATIVE, 5), _a(), _b())
    )
    result = _apply(testee)

    assert cpm.is_call_to(result, "concat_where")
    # The boundary moves by `-distance`, so the shifted branch is only selected where
    #  the original branch was valid at the shifted position.
    new_range = domain_utils.SymbolicDomain.from_expr(result.args[0]).ranges[KDim]
    assert new_range.stop == im.plus(im.literal_from_value(5), -1)
    # Both branches carry the shift.
    for branch in result.args[1:]:
        assert cpm.is_applied_as_fieldop(branch)


def test_shift_orthogonal_to_condition_leaves_condition_alone():
    cond = im.domain(common.GridType.CARTESIAN, {IDim: (0, 5)})
    testee = _shift(Koff, 1, im.concat_where(cond, _a(), _b()))
    result = _apply(testee)

    assert cpm.is_call_to(result, "concat_where")
    assert result.args[0] == cond


def test_shift_is_sunk_to_the_leaves():
    """The shift must not come to rest on an intermediate, that is the fusion barrier."""
    branch = im.as_fieldop(im.lambda_("x", "y")(im.plus(im.deref("x"), im.deref("y"))))(_a(), _b())
    branch.type = k_field
    testee = _shift(
        Koff, 1, im.concat_where(_k_domain(itir.InfinityLiteral.NEGATIVE, 5), branch, _b())
    )
    result = _apply(testee)

    shifted_branch = result.args[1]
    # the stencil of the branch is preserved and the shift moved onto its arguments
    assert shifted_branch.fun == branch.fun
    assert all(cpm.is_applied_as_fieldop(arg) for arg in shifted_branch.args)


def test_zero_distance_is_not_pushed():
    testee = _shift(
        Koff, 0, im.concat_where(_k_domain(itir.InfinityLiteral.NEGATIVE, 5), _a(), _b())
    )
    assert _apply(testee) == testee


def test_non_concat_where_argument_is_untouched():
    testee = _shift(Koff, 1, im.ref("a", k_field))
    assert _apply(testee) == testee


def test_shift_is_looked_through_a_let_binding():
    testee = im.let(
        "tmp", im.concat_where(_k_domain(itir.InfinityLiteral.NEGATIVE, 5), _a(), _b())
    )(_shift(Koff, 1, im.ref("tmp", k_field)))
    result = _apply(testee)

    assert cpm.is_call_to(result.fun.expr, "concat_where")


def test_shadowing_parameter_is_not_looked_through():
    """A parameter shadowing an outer `concat_where` binding must not be substituted."""
    testee = im.let(
        "tmp", im.concat_where(_k_domain(itir.InfinityLiteral.NEGATIVE, 5), _a(), _b())
    )(
        im.call(im.lambda_(im.sym("tmp", k_field))(_shift(Koff, 1, im.ref("tmp", k_field))))(
            im.ref("b", k_field)
        )
    )
    result = _apply(testee)

    # The shift sits inside the inner lambda, whose `tmp` is the lambda parameter and
    #  not the outer `concat_where`, so nothing may be pushed there.
    inner_lambda_body = result.fun.expr.fun.expr
    assert not cpm.is_call_to(inner_lambda_body, "concat_where")
