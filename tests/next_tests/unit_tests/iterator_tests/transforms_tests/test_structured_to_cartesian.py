# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py.next import common, utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import inline_dynamic_shifts, pass_manager
from gt4py.next.iterator.transforms.structured_to_cartesian import StructuredToCartesian
from gt4py.next.embedded.structured_connectivity import StructuredConnectivity
from gt4py.next.type_system import type_specifications as ts


Cell = common.Dimension("Cell")
Edge = common.Dimension("Edge")
Vertex = common.Dimension("Vertex")
I = common.Dimension("I")  # noqa: E741 [ambiguous-variable-name]
X = common.Dimension("X")
K = common.Dimension("K", kind=common.DimensionKind.VERTICAL)
C2EDim = common.Dimension("C2E", kind=common.DimensionKind.LOCAL)
E2CDim = common.Dimension("E2C", kind=common.DimensionKind.LOCAL)
V2EDim = common.Dimension("V2E", kind=common.DimensionKind.LOCAL)

# cells have colours 0, 1 and edges 0, 1, 2 on each `I`
C2E = StructuredConnectivity(
    source_dim=Cell,
    codomain=Edge,
    color_dim=X,
    local_dim=C2EDim,
    offsets={0: [{}, {X: 1}, {X: 2}], 1: [{X: -1}, {I: 1}, {X: 1}]},
)
E2C = StructuredConnectivity(
    source_dim=Edge,
    codomain=Cell,
    color_dim=X,
    local_dim=E2CDim,
    offsets={0: [{}, {X: 1}], 1: [{X: -1}, {I: -1}], 2: [{X: -2}, {X: -1}]},
)
# a single vertex colour
V2E = StructuredConnectivity(
    source_dim=Vertex,
    codomain=Edge,
    color_dim=X,
    local_dim=V2EDim,
    offsets={0: [{}, {X: 1}, {I: -1, X: 2}]},
)
PROVIDER_TYPE = common.offset_provider_to_type({"C2E": C2E, "E2C": E2C, "V2E": V2E})

f64 = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
i32 = ts.ScalarType(kind=ts.ScalarKind.INT32)
cell_field = ts.FieldType(dims=[I, X, K], dtype=f64)
edge_field = ts.FieldType(dims=[I, X, K], dtype=f64)
c2e_field = ts.FieldType(dims=[I, X, C2EDim, K], dtype=f64)
offset_field = ts.FieldType(dims=[I, X, K], dtype=i32)

cell_domain = {I: (0, 4), X: (0, 2), K: (0, 3)}


def _program(expr: itir.Expr, params: dict[str, ts.TypeSpec], domain=cell_domain) -> itir.Program:
    return itir.Program(
        id="testee",
        function_definitions=[],
        params=[im.sym(name, type_) for name, type_ in {**params, "out": cell_field}.items()],
        declarations=[],
        body=[
            itir.SetAt(
                expr=expr,
                domain=im.domain(common.GridType.UNSTRUCTURED, domain),
                target=im.ref("out"),
            )
        ],
    )


def _apply(program: itir.Program, uids: utils.IDGeneratorPool, provider_type=PROVIDER_TYPE):
    return StructuredToCartesian.apply(program, offset_provider_type=provider_type, uids=uids)


def _off(value: int) -> itir.OffsetLiteral:
    return itir.OffsetLiteral(value=value)


def _shifted(it: str, *pairs) -> itir.Expr:
    """`deref(shift(D0, o0, D1, o1, ...)(it))` with Cartesian offsets."""
    args: list[itir.Expr] = []
    for dim, offset in pairs:
        args.extend((im.cartesian_offset(dim), _off(offset) if isinstance(offset, int) else offset))
    return im.deref(_shift(*args)(it)) if args else im.deref(it)


def _shift(*args) -> im.call:
    return im.call(
        im.call("shift")(*(im.ensure_offset(a) if isinstance(a, (str, int)) else a for a in args))
    )


def _per_color(stencils: list[itir.Lambda], *args: str) -> itir.Expr:
    expr = im.as_fieldop(stencils[-1])(*args)
    for color in reversed(range(len(stencils) - 1)):
        expr = im.concat_where(
            im.less(im.axis_literal(X), color + 1), im.as_fieldop(stencils[color])(*args), expr
        )
    return expr


def test_domains_become_cartesian(uids):
    testee = _program(
        im.as_fieldop(im.lambda_("it")(im.deref(im.shift("C2E", 0)("it"))))("e"), {"e": edge_field}
    )

    actual = _apply(testee, uids)

    assert actual.body[0].domain == im.domain(common.GridType.CARTESIAN, cell_domain)


def test_single_shift(uids):
    testee = _program(
        im.as_fieldop(im.lambda_("it")(im.deref(im.shift("C2E", 1)("it"))))("e"), {"e": edge_field}
    )
    # colour 0: C2E[0][1] = {X: 1}; colour 1: C2E[1][1] = {I: 1}
    expected = _per_color(
        [
            im.lambda_("it")(_shifted("it", (X, 1))),
            im.lambda_("it")(_shifted("it", (I, 1))),
        ],
        "e",
    )

    actual = _apply(testee, uids)

    assert actual.body[0].expr == expected
    assert actual.body[0].expr.type == cell_field


def test_chained_shift(uids):
    testee = _program(
        im.as_fieldop(im.lambda_("it")(im.deref(_shift("C2E", 1, "E2C", 1)("it"))))("c"),
        {"c": cell_field},
    )
    # colour 0: C2E[0][1] = {X: 1} -> edge colour 1, E2C[1][1] = {I: -1}: (I -1, X +1)
    # colour 1: C2E[1][1] = {I: 1} -> edge colour 1, E2C[1][1] = {I: -1}: no shift
    expected = _per_color(
        [
            im.lambda_("it")(_shifted("it", (I, -1), (X, 1))),
            im.lambda_("it")(_shifted("it")),
        ],
        "c",
    )

    actual = _apply(testee, uids)

    assert actual.body[0].expr == expected


def test_nested_shift_is_flattened(uids):
    testee = _program(
        im.as_fieldop(im.lambda_("it")(im.deref(im.shift("E2C", 1)(im.shift("C2E", 1)("it")))))(
            "c"
        ),
        {"c": cell_field},
    )
    expected = _per_color(
        [
            im.lambda_("it")(_shifted("it", (I, -1), (X, 1))),
            im.lambda_("it")(_shifted("it")),
        ],
        "c",
    )

    actual = _apply(testee, uids)

    assert actual.body[0].expr == expected


def test_iterator_let_used_twice_is_inlined(uids):
    testee = _program(
        im.as_fieldop(
            im.lambda_("it")(
                im.let("x", im.shift("C2E", 1)("it"))(
                    im.plus(im.deref("x"), im.deref(im.shift(im.cartesian_offset(K), 1)("x")))
                )
            )
        )("e"),
        {"e": edge_field},
    )
    expected = _per_color(
        [
            im.lambda_("it")(im.plus(_shifted("it", (X, 1)), _shifted("it", (X, 1), (K, 1)))),
            im.lambda_("it")(im.plus(_shifted("it", (I, 1)), _shifted("it", (I, 1), (K, 1)))),
        ],
        "e",
    )

    actual = _apply(testee, uids)

    assert actual.body[0].expr == expected


def test_list_producer_read_twice_in_one_consumer(uids):
    neighbors = im.as_fieldop(im.lambda_("it")(im.neighbors("C2E", "it")))("e")
    testee = _program(
        im.as_fieldop(
            im.lambda_("l")(im.plus(im.list_get(0, im.deref("l")), im.list_get(1, im.deref("l"))))
        )(neighbors),
        {"e": edge_field},
    )
    expected = _per_color(
        [
            im.lambda_("e")(im.plus(_shifted("e"), _shifted("e", (X, 1)))),
            im.lambda_("e")(im.plus(_shifted("e", (X, -1)), _shifted("e", (I, 1)))),
        ],
        "e",
    )

    actual = _apply(testee, uids)

    assert actual.body[0].expr == expected


def _weighted_sum(e: str, w: str, color_offsets: list[list[tuple]]) -> list[itir.Lambda]:
    stencils = []
    for per_neighbor in color_offsets:
        acc: itir.Expr = im.literal_from_value(0.0)
        for k, pairs in enumerate(per_neighbor):
            acc = im.plus(
                acc, im.multiplies_(_shifted(e, *pairs), im.list_get(_off(k), im.deref(w)))
            )
        stencils.append(im.lambda_(e, w)(acc))
    return stencils


C2E_BY_COLOR = [[(), ((X, 1),), ((X, 2),)], [((X, -1),), ((I, 1),), ((X, 1),)]]


def test_neighbor_sum_is_unrolled(uids):
    neighbors = im.as_fieldop(im.lambda_("it")(im.neighbors("C2E", "it")))("e")
    products = im.as_fieldop(
        im.lambda_("a", "b")(im.map_list("multiplies")(im.deref("a"), im.deref("b")))
    )(neighbors, "w")
    testee = _program(
        im.as_fieldop(
            im.lambda_("l")(im.reduce("plus", im.literal_from_value(0.0))(im.deref("l")))
        )(products),
        {"e": edge_field, "w": c2e_field},
    )
    expected = _per_color(_weighted_sum("e", "w", C2E_BY_COLOR), "e", "w")

    actual = _apply(testee, uids)

    assert actual.body[0].expr == expected


def test_let_bound_list_field_read_at_several_slots(uids):
    # `v = e(C2E) * w; v[C2EDim(0)] + v[C2EDim(2)]`
    products = im.as_fieldop(
        im.lambda_("a", "b")(im.map_list("multiplies")(im.deref("a"), im.deref("b")))
    )(im.as_fieldop(im.lambda_("it")(im.neighbors("C2E", "it")))("e"), "w")

    def slot(k: int) -> itir.Expr:
        return im.as_fieldop(im.lambda_("l")(im.list_get(k, im.deref("l"))))("v")

    testee = _program(
        im.let("v", products)(
            im.as_fieldop(im.lambda_("x", "y")(im.plus(im.deref("x"), im.deref("y"))))(
                slot(0), slot(2)
            )
        ),
        {"e": edge_field, "w": c2e_field},
    )

    def read(k: int) -> itir.Expr:
        return _per_color(
            [
                im.lambda_("e", "w")(
                    im.multiplies_(
                        _shifted("e", *C2E_BY_COLOR[c][k]), im.list_get(k, im.deref("w"))
                    )
                )
                for c in range(2)
            ],
            "e",
            "w",
        )

    expected = im.as_fieldop(im.lambda_("x", "y")(im.plus(im.deref("x"), im.deref("y"))))(
        read(0), read(2)
    )

    actual = _apply(testee, uids)

    assert actual.body[0].expr == expected


def test_sparse_list_get_is_kept(uids):
    testee = _program(
        im.as_fieldop(im.lambda_("w")(im.list_get(1, im.deref("w"))))("w"), {"w": c2e_field}
    )

    actual = _apply(testee, uids)

    assert actual.body[0].expr == testee.body[0].expr


def test_sparse_reduce_without_shift_is_unrolled_once(uids):
    testee = _program(
        im.as_fieldop(
            im.lambda_("w")(im.reduce("plus", im.literal_from_value(0.0))(im.deref("w")))
        )("w"),
        {"w": c2e_field},
    )
    acc: itir.Expr = im.literal_from_value(0.0)
    for k in range(3):
        acc = im.plus(acc, im.list_get(_off(k), im.deref("w")))
    expected = im.as_fieldop(im.lambda_("w")(acc))("w")

    actual = _apply(testee, uids)

    assert actual.body[0].expr == expected


def test_static_vertical_shift_is_kept(uids):
    testee = _program(
        im.as_fieldop(
            im.lambda_("it")(im.deref(_shift("C2E", 1, im.cartesian_offset(K), 1)("it")))
        )("e"),
        {"e": edge_field},
    )
    expected = _per_color(
        [
            im.lambda_("it")(_shifted("it", (X, 1), (K, 1))),
            im.lambda_("it")(_shifted("it", (I, 1), (K, 1))),
        ],
        "e",
    )

    actual = _apply(testee, uids)

    assert actual.body[0].expr == expected


def test_dynamic_vertical_shift_after_inline_dynamic_shifts(uids):
    # `e(C2E[1])(as_offset(Koff, o))` as lowered from the frontend
    testee = _program(
        im.as_fieldop(
            im.lambda_("it", "o")(im.deref(im.shift(im.cartesian_offset(K), im.deref("o"))("it")))
        )(im.as_fieldop(im.lambda_("it")(im.deref(im.shift("C2E", 1)("it"))))("e"), "o"),
        {"e": edge_field, "o": offset_field},
    )
    testee = inline_dynamic_shifts.InlineDynamicShifts.apply(
        testee, offset_provider_type=PROVIDER_TYPE, uids=uids
    )

    actual = _apply(testee, uids)

    (stencil_params,) = {
        tuple(p.id for p in branch.fun.args[0].params)
        for branch in (actual.body[0].expr.args[1], actual.body[0].expr.args[2])
    }
    it, o = stencil_params
    dynamic = (K, im.deref(o))
    expected = _per_color(
        [
            im.lambda_(it, o)(_shifted(it, (X, 1), dynamic)),
            im.lambda_(it, o)(_shifted(it, (I, 1), dynamic)),
        ],
        "e",
        "o",
    )
    assert actual.body[0].expr == expected


def test_vertical_concat_where_around_edge_stencil(uids):
    shifted = im.as_fieldop(im.lambda_("it")(im.deref(im.shift("C2E", 1)("it"))))("e")
    testee = _program(
        im.concat_where(im.less(im.axis_literal(K), 1), shifted, "c"),
        {"e": edge_field, "c": cell_field},
    )
    expected = im.concat_where(
        im.less(im.axis_literal(K), 1),
        _per_color(
            [
                im.lambda_("it")(_shifted("it", (X, 1))),
                im.lambda_("it")(_shifted("it", (I, 1))),
            ],
            "e",
        ),
        "c",
    )

    actual = _apply(testee, uids)

    assert actual.body[0].expr == expected


def test_single_color_output_has_no_concat_where(uids):
    testee = _program(
        im.as_fieldop(im.lambda_("it")(im.deref(im.shift("V2E", 2)("it"))))("e"),
        {"e": edge_field},
        domain={I: (0, 4), X: (0, 1), K: (0, 3)},
    )
    expected = im.as_fieldop(im.lambda_("it")(_shifted("it", (I, -1), (X, 2))))("e")

    actual = _apply(testee, uids)

    assert actual.body[0].expr == expected


def test_non_trivial_arguments_are_let_bound_once(uids):
    scaled = im.as_fieldop(im.lambda_("a")(im.multiplies_(im.deref("a"), 2.0)))("e")
    testee = _program(
        im.as_fieldop(im.lambda_("it")(im.deref(im.shift("C2E", 1)("it"))))(scaled),
        {"e": edge_field},
    )

    actual = _apply(testee, uids)

    expr = actual.body[0].expr
    assert len(expr.fun.params) == 1 and expr.args == [scaled]
    (name,) = [p.id for p in expr.fun.params]
    assert expr.fun.expr == _per_color(
        [
            im.lambda_("it")(_shifted("it", (X, 1))),
            im.lambda_("it")(_shifted("it", (I, 1))),
        ],
        name,
    )


def test_broadcast_axes_become_lattice_dims(uids):
    testee = _program(
        im.as_fieldop(
            im.lambda_("a", "it")(im.plus(im.deref("a"), im.deref(im.shift("C2E", 0)("it"))))
        )(
            im.call("broadcast")(1.0, im.make_tuple(im.axis_literal(Cell), im.axis_literal(K))),
            "e",
        ),
        {"e": edge_field},
    )

    actual = _apply(testee, uids)

    broadcasts = [
        call
        for call in actual.pre_walk_values().if_isinstance(itir.FunCall)
        if call.fun == im.ref("broadcast")
    ]
    assert broadcasts
    for call in broadcasts:
        assert call.args[1] == im.make_tuple(
            im.axis_literal(I), im.axis_literal(X), im.axis_literal(K)
        )


def test_idempotent(uids):
    testee = _program(
        im.as_fieldop(im.lambda_("it")(im.deref(_shift("C2E", 1, "E2C", 1)("it"))))("c"),
        {"c": cell_field},
    )

    once = _apply(testee, uids)
    twice = _apply(once, uids)

    assert twice == once


def test_noop_without_structured_connectivity(uids):
    table_type = common.NeighborConnectivityType(
        domain=(Cell, C2EDim), codomain=Edge, skip_value=None, dtype=None, max_neighbors=3
    )
    testee = _program(
        im.as_fieldop(im.lambda_("it")(im.deref(im.shift("C2E", 1)("it"))))("e"),
        {"e": ts.FieldType(dims=[Edge, K], dtype=f64)},
    )

    assert _apply(testee, uids, provider_type={"C2E": table_type}) is testee


def test_referenced_table_next_to_structured_raises(uids):
    table_type = common.NeighborConnectivityType(
        domain=(Cell, C2EDim), codomain=Edge, skip_value=None, dtype=None, max_neighbors=3
    )
    testee = _program(
        im.as_fieldop(
            im.lambda_("a", "b")(
                im.plus(im.deref(im.shift("E2C", 1)("a")), im.deref(im.shift("C2E", 0)("b")))
            )
        )("c", "e"),
        {"c": cell_field, "e": edge_field},
    )

    with pytest.raises(ValueError, match="Neighbor tables \\['C2E'\\]"):
        _apply(
            testee,
            uids,
            provider_type={**PROVIDER_TYPE, "C2E": table_type, "Koff": K},
        )


def test_unreferenced_table_and_vertical_dimension_are_ignored(uids):
    table_type = common.NeighborConnectivityType(
        domain=(Cell, C2EDim), codomain=Edge, skip_value=None, dtype=None, max_neighbors=3
    )
    testee = _program(
        im.as_fieldop(im.lambda_("it")(im.deref(im.shift("E2C", 1)("it"))))("c"),
        {"c": cell_field},
        domain={I: (0, 4), X: (0, 3), K: (0, 3)},
    )
    # edge colours 0, 1, 2: E2C[c][1] = {X: 1}, {I: -1}, {X: -1}
    expected = _per_color(
        [
            im.lambda_("it")(_shifted("it", (X, 1))),
            im.lambda_("it")(_shifted("it", (I, -1))),
            im.lambda_("it")(_shifted("it", (X, -1))),
        ],
        "c",
    )

    actual = _apply(testee, uids, provider_type={**PROVIDER_TYPE, "C2E": table_type, "Koff": K})

    assert actual.body[0].expr == expected


def test_list_valued_output_raises(uids):
    testee = _program(
        im.as_fieldop(im.lambda_("it")(im.neighbors("C2E", "it")))("e"), {"e": edge_field}
    )
    testee.params[-1].type = ts.FieldType(
        dims=[I, X, K], dtype=ts.ListType(element_type=f64, offset_type=C2EDim)
    )

    with pytest.raises(ValueError, match="remain"):
        _apply(testee, uids)


def test_dynamic_structured_index_raises(uids):
    testee = _program(
        im.as_fieldop(im.lambda_("it", "o")(im.deref(im.shift("C2E", im.deref("o"))("it"))))(
            "e", "o"
        ),
        {"e": edge_field, "o": offset_field},
    )

    with pytest.raises(ValueError, match="non-constant index"):
        _apply(testee, uids)


def test_chains_from_different_sources_raise(uids):
    testee = _program(
        im.as_fieldop(
            im.lambda_("a", "b")(
                im.plus(im.deref(im.shift("C2E", 0)("a")), im.deref(im.shift("E2C", 0)("b")))
            )
        )("e", "c"),
        {"e": edge_field, "c": cell_field},
    )

    with pytest.raises(ValueError, match="different source"):
        _apply(testee, uids)


def test_tuple_valued_stencil_raises(uids):
    testee = _program(
        im.as_fieldop(
            im.lambda_("it")(
                im.make_tuple(
                    im.deref(im.shift("C2E", 0)("it")), im.deref(im.shift("C2E", 1)("it"))
                )
            )
        )("e"),
        {"e": edge_field},
    )
    testee.params[-1].type = ts.TupleType(types=[cell_field, cell_field])

    with pytest.raises(NotImplementedError, match="Tuple"):
        _apply(testee, uids)


@pytest.mark.parametrize(
    "driver",
    [pass_manager.apply_common_transforms, pass_manager.apply_fieldview_transforms],
)
def test_wired_in_driver(driver):
    testee = _program(
        im.as_fieldop(im.lambda_("it")(im.deref(im.shift("C2E", 1)("it"))))("e"),
        {"e": edge_field},
    )

    actual = driver(testee, offset_provider={"C2E": C2E, "E2C": E2C, "V2E": V2E})

    literals = actual.pre_walk_values().if_isinstance(itir.OffsetLiteral).getattr("value").to_list()
    assert not {"C2E", "E2C", "V2E"} & set(literals)
    assert actual.body[0].domain == im.domain(common.GridType.CARTESIAN, cell_domain)
