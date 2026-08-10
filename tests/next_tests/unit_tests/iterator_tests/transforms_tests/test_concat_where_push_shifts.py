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
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas
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


def test_binding_read_through_shift_is_inlined():
    """A multi use binding must be exposed; that is the shape the pass exists for."""
    testee = im.let(
        "tmp", im.concat_where(_k_domain(itir.InfinityLiteral.NEGATIVE, 5), _a(), _b())
    )(
        im.as_fieldop(im.lambda_("x", "y")(im.minus(im.deref("x"), im.deref("y"))))(
            im.ref("tmp", k_field), _shift(Koff, 1, im.ref("tmp", k_field))
        )
    )
    result = _apply(testee)

    # the binding is gone, the unshifted use keeps the original condition and the
    #  shifted one carries the translated condition
    assert not cpm.is_let(result)
    unshifted, shifted = result.args
    assert cpm.is_call_to(unshifted, "concat_where")
    assert cpm.is_call_to(shifted, "concat_where")
    assert domain_utils.SymbolicDomain.from_expr(shifted.args[0]).ranges[KDim].stop == im.plus(
        im.literal_from_value(5), -1
    )


def test_inlining_does_not_capture():
    """The free variables of the inlined `concat_where` must not be captured."""
    testee = im.let(
        "tmp", im.concat_where(_k_domain(itir.InfinityLiteral.NEGATIVE, 5), _a(), _b())
    )(
        im.call(im.lambda_(im.sym("a", k_field))(_shift(Koff, 1, im.ref("tmp", k_field))))(
            im.ref("b", k_field)
        )
    )
    result = _apply(testee)

    # The inner parameter shadowed `a`, which the moved `concat_where` refers to, so
    #  the inliner must have renamed it rather than capturing the reference.
    assert isinstance(result.fun, itir.Lambda)
    assert str(result.fun.params[0].id) != "a"


def test_scan_branch_declines():
    """Sinking into a scan would run the fold over a different range."""
    scan = im.call("scan")(
        im.lambda_("acc", "x")(im.plus("acc", im.deref("x"))),
        im.literal_from_value(True),
        im.literal_from_value(0.0),
    )
    branch = im.as_fieldop(scan)(_a())
    branch.type = k_field
    testee = _shift(
        Koff, 1, im.concat_where(_k_domain(itir.InfinityLiteral.NEGATIVE, 5), branch, _b())
    )
    assert _apply(testee) == testee


def test_explicit_domain_branch_declines():
    """An explicit domain argument would be left stale by the rewrite."""
    branch = im.as_fieldop(im.lambda_("x")(im.deref("x")), _k_domain(0, 10))(_a())
    branch.type = k_field
    testee = _shift(
        Koff, 1, im.concat_where(_k_domain(itir.InfinityLiteral.NEGATIVE, 5), branch, _b())
    )
    assert _apply(testee) == testee


def test_untyped_branch_declines():
    """Without a type the pass cannot tell whether the dimension is present."""
    testee = _shift(Koff, 1, im.concat_where(_k_domain(itir.InfinityLiteral.NEGATIVE, 5), "a", "b"))
    assert _apply(testee) == testee


def test_compound_leaf_declines():
    """Leaving the shift on a compound intermediate is the shape the pass removes."""
    leaf = im.tuple_get(0, im.make_tuple(_a(), _b()))
    leaf.type = k_field
    branch = im.as_fieldop(im.lambda_("x")(im.deref("x")))(leaf)
    branch.type = k_field
    testee = _shift(
        Koff, 1, im.concat_where(_k_domain(itir.InfinityLiteral.NEGATIVE, 5), branch, _b())
    )
    assert _apply(testee) == testee
