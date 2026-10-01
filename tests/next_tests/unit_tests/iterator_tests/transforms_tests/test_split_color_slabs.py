# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.next import common
from gt4py.next.embedded.structured_connectivity import StructuredConnectivity
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.split_color_slabs import split_color_slabs


Cell = common.Dimension("Cell")
Edge = common.Dimension("Edge")
I = common.Dimension("I")  # noqa: E741 [ambiguous-variable-name]
X = common.Dimension("X")
K = common.Dimension("K", kind=common.DimensionKind.VERTICAL)
C2EDim = common.Dimension("C2E", kind=common.DimensionKind.LOCAL)

PROVIDER_TYPE = common.offset_provider_to_type(
    {
        "C2E": StructuredConnectivity(
            source_dim=Cell,
            codomain=Edge,
            color_dim=X,
            local_dim=C2EDim,
            offsets={0: [{}, {X: 1}], 1: [{X: -1}, {I: 1}]},
        )
    }
)


def _domain(x: tuple[int, int] | tuple[itir.Expr, itir.Expr]) -> itir.FunCall:
    return im.domain(common.GridType.CARTESIAN, {I: (0, 4), X: x, K: (0, 3)})


def _shifted(it: str, dim: common.Dimension, offset: int) -> itir.Expr:
    return im.deref(im.shift(im.cartesian_offset(dim), offset)(it))


def _program(stencil: itir.Lambda, *args: itir.Expr, x=(0, 2)) -> itir.Program:
    return itir.Program(
        id="testee",
        function_definitions=[],
        params=[im.sym("e"), im.sym("n"), im.sym("out")],
        declarations=[],
        body=[
            itir.SetAt(
                expr=im.as_fieldop(stencil, _domain(x))(*args),
                domain=_domain(x),
                target=im.ref("out"),
            )
        ],
    )


#: colour 0 reads its own edge, colour 1 the edge one colour down, as `StructuredToCartesian` emits it
COLOR_CHAIN = im.lambda_("x", "e")(
    im.if_(
        im.and_(True, im.less(im.deref("x"), 1)),
        im.deref("e"),
        im.plus(_shifted("e", X, -1), _shifted("e", I, 1)),
    )
)


def test_one_statement_per_color():
    testee = _program(COLOR_CHAIN, im.index(X), im.ref("e"))
    expected = [
        itir.SetAt(
            expr=im.as_fieldop(stencil, _domain((c, c + 1)))(im.ref("e")),
            domain=_domain((c, c + 1)),
            target=im.ref("out"),
        )
        for c, stencil in [
            (0, im.lambda_("e")(im.deref("e"))),
            (1, im.lambda_("e")(im.plus(_shifted("e", X, -1), _shifted("e", I, 1)))),
        ]
    ]

    actual = split_color_slabs(testee, offset_provider_type=PROVIDER_TYPE)

    assert actual.body == expected


def test_let_bound_condition_is_folded():
    # the shape CSE leaves: the colour test bound once, the shared read hoisted out of the branches
    stencil = im.lambda_("x", "e")(
        im.let(("c", im.less(im.deref("x"), 1)), ("s", _shifted("e", I, 1)))(
            im.if_("c", im.deref("e"), im.plus(_shifted("e", X, -1), "s"))
        )
    )
    testee = _program(stencil, im.index(X), im.ref("e"))

    actual = split_color_slabs(testee, offset_provider_type=PROVIDER_TYPE)

    assert [stmt.expr.fun.args[0] for stmt in actual.body] == [
        im.lambda_("e")(im.deref("e")),
        im.lambda_("e")(im.plus(_shifted("e", X, -1), _shifted("e", I, 1))),
    ]


def test_without_structured_connectivity_unchanged():
    testee = _program(COLOR_CHAIN, im.index(X), im.ref("e"))

    assert split_color_slabs(testee, offset_provider_type={}) is testee


def test_without_index_unchanged():
    testee = _program(im.lambda_("e")(_shifted("e", I, 1)), im.ref("e"))

    assert split_color_slabs(testee, offset_provider_type=PROVIDER_TYPE) is testee


def test_symbolic_color_range_unchanged():
    testee = _program(COLOR_CHAIN, im.index(X), im.ref("e"), x=(0, im.ref("n")))

    assert split_color_slabs(testee, offset_provider_type=PROVIDER_TYPE) is testee


def test_shifted_index_unchanged():
    # `·⟪X, 1⟫(x)` has no single value per colour slab
    stencil = im.lambda_("x", "e")(
        im.if_(im.less(_shifted("x", X, 1), 2), im.deref("e"), _shifted("e", I, 1))
    )
    testee = _program(stencil, im.index(X), im.ref("e"))

    assert split_color_slabs(testee, offset_provider_type=PROVIDER_TYPE) is testee


def test_index_of_other_dimension_unchanged():
    stencil = im.lambda_("k", "e")(
        im.if_(im.less(im.deref("k"), 1), im.deref("e"), _shifted("e", I, 1))
    )
    testee = _program(stencil, im.index(K), im.ref("e"))

    assert split_color_slabs(testee, offset_provider_type=PROVIDER_TYPE) is testee


def test_statements_in_if_are_split():
    inner = _program(COLOR_CHAIN, im.index(X), im.ref("e"))
    testee = itir.Program(
        id="testee",
        function_definitions=[],
        params=[*inner.params, im.sym("flag")],
        declarations=[],
        body=[itir.IfStmt(cond=im.ref("flag"), true_branch=inner.body, false_branch=[])],
    )

    actual = split_color_slabs(testee, offset_provider_type=PROVIDER_TYPE)

    assert (
        actual.body[0].true_branch
        == split_color_slabs(inner, offset_provider_type=PROVIDER_TYPE).body
    )
    assert len(actual.body[0].true_branch) == 2
