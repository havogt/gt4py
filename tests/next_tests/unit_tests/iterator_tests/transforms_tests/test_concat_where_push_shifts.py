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


def test_binding_read_through_shift_is_exposed():
    """A multi use binding must be exposed; that is the shape the pass exists for."""
    testee = im.let(
        "tmp", im.concat_where(_k_domain(itir.InfinityLiteral.NEGATIVE, 5), _a(), _b())
    )(
        im.as_fieldop(im.lambda_("x", "y")(im.minus(im.deref("x"), im.deref("y"))))(
            im.ref("tmp", k_field), _shift(Koff, 1, im.ref("tmp", k_field))
        )
    )
    result = _apply(testee)

    # one read each, so neither of the two needs a binding: the unshifted use keeps the
    #  original condition and the shifted one carries the translated condition
    assert not cpm.is_let(result)
    unshifted, shifted = result.args
    assert cpm.is_call_to(unshifted, "concat_where")
    assert cpm.is_call_to(shifted, "concat_where")
    assert domain_utils.SymbolicDomain.from_expr(shifted.args[0]).ranges[KDim].stop == im.plus(
        im.literal_from_value(5), -1
    )


def test_one_binding_per_distinct_distance():
    """Duplication follows the number of distinct offsets, not the number of uses."""
    tmp = im.ref("tmp", k_field)
    testee = im.let(
        "tmp", im.concat_where(_k_domain(itir.InfinityLiteral.NEGATIVE, 5), _a(), _b())
    )(
        im.as_fieldop(im.lambda_("x", "y", "z")(im.plus(im.deref("x"), im.deref("y"))))(
            _shift(Koff, 1, tmp), _shift(Koff, 1, tmp), _shift(Koff, -1, tmp)
        )
    )
    result = _apply(testee)

    # the two reads at `+1` share one binding, the single read at `-1` needs none
    assert len(result.fun.params) == 1
    first, second, third = result.fun.expr.args
    assert cpm.is_ref_to(first, result.fun.params[0].id)
    assert first == second
    assert cpm.is_call_to(third, "concat_where")

    # so the three reads leave one copy of the `concat_where` per distinct distance
    assert sum(cpm.is_call_to(node, "concat_where") for node in result.pre_walk_values()) == 2


def test_shared_binding_stays_in_the_outer_scope():
    """A `concat_where` kept for several reads must not move under a shadowing binder."""
    tmp = im.ref("tmp", k_field)
    inner = im.lambda_(im.sym("a", k_field))(
        im.as_fieldop(im.lambda_("x", "y")(im.plus(im.deref("x"), im.deref("y"))))(
            _shift(Koff, 1, tmp), _shift(Koff, 1, tmp)
        )
    )
    testee = im.let(
        "tmp", im.concat_where(_k_domain(itir.InfinityLiteral.NEGATIVE, 5), _a(), _b())
    )(im.call(inner)(im.ref("b", k_field)))
    result = _apply(testee)

    # The binding sits where the original one was, outside the lambda that rebinds `a`,
    #  so its reference to `a` still resolves to the outer one.
    assert len(result.fun.params) == 1
    assert cpm.is_call_to(result.args[0], "concat_where")
    assert cpm.is_ref_to(result.args[0].args[1].args[0], "a")
    assert str(result.fun.expr.fun.params[0].id) != "a"
    # both reads share it
    assert all(cpm.is_ref_to(arg, result.fun.params[0].id) for arg in result.fun.expr.fun.expr.args)


def test_moving_a_binding_to_its_single_read_does_not_capture():
    """The free variables of a `concat_where` moved to its read must not be captured."""
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


def test_rebound_parameter_declines():
    """A body that shadows the parameter is left alone rather than substituted into."""
    testee = im.let(
        "tmp", im.concat_where(_k_domain(itir.InfinityLiteral.NEGATIVE, 5), _a(), _b())
    )(
        im.call(im.lambda_(im.sym("tmp", k_field))(_shift(Koff, 1, im.ref("tmp", k_field))))(
            im.ref("b", k_field)
        )
    )
    assert _apply(testee) == testee


def _scan(*args):
    scan = im.call("scan")(
        im.lambda_("acc", *(f"x{i}" for i in range(len(args))))(im.plus("acc", im.deref("x0"))),
        im.literal_from_value(True),
        im.literal_from_value(0.0),
    )
    result = im.as_fieldop(scan)(*args)
    result.type = k_field
    return result


def test_binding_feeding_a_scan_declines():
    """A `scan` argument is materialized either way, so the copies buy nothing."""
    tmp = im.ref("tmp", k_field)
    testee = im.let(
        "tmp", im.concat_where(_k_domain(itir.InfinityLiteral.NEGATIVE, 5), _a(), _b())
    )(_scan(tmp, _shift(Koff, 1, tmp)))
    assert _apply(testee) == testee


def test_binding_feeding_a_scan_through_a_binding_declines():
    """The use reaching the `scan` may be several bindings away."""
    tmp = im.ref("tmp", k_field)
    testee = im.let(
        "tmp", im.concat_where(_k_domain(itir.InfinityLiteral.NEGATIVE, 5), _a(), _b())
    )(
        im.let("q", im.as_fieldop(im.lambda_("x")(im.deref("x")))(tmp))(
            _scan(im.ref("q", k_field), _shift(Koff, 1, tmp))
        )
    )
    assert _apply(testee) == testee


def test_binding_beside_an_unrelated_scan_still_fires():
    """Only a `scan` the binding actually reaches may block the rewrite."""
    tmp = im.ref("tmp", k_field)
    testee = im.let(
        "tmp", im.concat_where(_k_domain(itir.InfinityLiteral.NEGATIVE, 5), _a(), _b())
    )(
        im.as_fieldop(im.lambda_("x", "y")(im.minus(im.deref("x"), im.deref("y"))))(
            _scan(_b()), _shift(Koff, 1, tmp)
        )
    )
    result = _apply(testee)
    assert cpm.is_call_to(result.args[1], "concat_where")


def _shift_chain(depth):
    """`depth` nested `let`s, each binding a `concat_where` reading the previous twice."""
    values = [im.concat_where(_k_domain(itir.InfinityLiteral.NEGATIVE, 5), _a(), _b())]
    for i in range(1, depth + 1):
        previous = im.ref(f"t{i - 1}", k_field)
        shifted = [_shift(Koff, 1, previous) for _ in range(2)]
        for expr in shifted:
            expr.type = k_field
        branch = im.as_fieldop(im.lambda_("x", "y")(im.plus(im.deref("x"), im.deref("y"))))(
            *shifted
        )
        branch.type = k_field
        values.append(im.concat_where(_k_domain(itir.InfinityLiteral.NEGATIVE, 5), branch, _b()))
    expr = im.ref(f"t{depth}", k_field)
    for i in reversed(range(depth + 1)):
        expr = im.let(f"t{i}", values[i])(expr)
    return expr


def _node_count(node):
    return sum(1 for _ in node.pre_walk_values().if_isinstance(itir.Node))


def test_deep_shift_chain_stays_linear():
    """Every level of the chain doubles the shifted reads, so per use duplication is exponential.

    A linear implementation stays below a factor of two here; the bound is set at three
    so that only a genuine change of complexity trips it.
    """
    for depth in (1, 5, 10):
        testee = _shift_chain(depth)
        assert _node_count(_apply(testee)) <= 3 * _node_count(testee)


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
