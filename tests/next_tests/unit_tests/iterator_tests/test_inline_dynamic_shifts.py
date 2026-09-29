# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from gt4py import next as gtx
from gt4py.next import utils
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import inline_dynamic_shifts
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_specifications as ts

IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")
int_type = ts.ScalarType(kind=ts.ScalarKind.INT32)
field_type = ts.FieldType(dims=[IDim], dtype=int_type)
IOff = im.cartesian_offset(IDim, IDim)
dynamic_shift_stencil = im.lambda_("a", "b")(im.deref(im.shift(IOff, im.deref("b"))("a")))


def test_inline_dynamic_shift_as_fieldop_arg(uids):
    testee = im.as_fieldop(im.lambda_("a", "b")(im.deref(im.shift(IOff, im.deref("b"))("a"))))(
        im.as_fieldop("deref")("inp"), "offset_field"
    )
    expected = im.as_fieldop(
        im.lambda_("inp", "offset_field")(im.deref(im.shift(IOff, im.deref("offset_field"))("inp")))
    )("inp", "offset_field")

    actual = inline_dynamic_shifts.InlineDynamicShifts.apply(
        testee, offset_provider_type={}, uids=uids
    )
    assert actual == expected


def test_inline_dynamic_shift_nested_as_fieldop_args(uids):
    testee = im.as_fieldop(im.lambda_("a", "b")(im.deref(im.shift(IOff, im.deref("b"))("a"))))(
        im.as_fieldop(im.lambda_("a")(im.deref(im.shift(IOff, 1)("a"))))(
            im.as_fieldop("deref")("inp")
        ),
        "offset_field",
    )
    expected = im.as_fieldop(
        im.lambda_("inp", "offset_field")(
            im.deref(im.shift(IOff, 1)(im.shift(IOff, im.deref("offset_field"))("inp")))
        )
    )("inp", "offset_field")

    actual = inline_dynamic_shifts.InlineDynamicShifts.apply(
        testee, offset_provider_type={}, uids=uids
    )
    assert actual == expected


def test_inline_dynamic_shift_let_var(uids):
    testee = im.let("tmp", im.as_fieldop("deref")("inp"))(
        im.as_fieldop(im.lambda_("a", "b")(im.deref(im.shift(IOff, im.deref("b"))("a"))))(
            "tmp", "offset_field"
        )
    )

    expected = im.as_fieldop(
        im.lambda_("inp", "offset_field")(im.deref(im.shift(IOff, im.deref("offset_field"))("inp")))
    )("inp", "offset_field")

    actual = inline_dynamic_shifts.InlineDynamicShifts.apply(
        testee, offset_provider_type={}, uids=uids
    )
    assert actual == expected


def test_inline_dynamic_shift_broadcast_arg(uids):
    testee = im.as_fieldop(dynamic_shift_stencil)(
        im.call("broadcast")(
            im.ref("inp", ts.FieldType(dims=[JDim], dtype=int_type)),
            im.make_tuple(im.axis_literal(IDim), im.axis_literal(JDim)),
        ),
        "offset_field",
    )
    expected = im.as_fieldop(
        im.lambda_("inp", "offset_field")(im.deref(im.shift(IOff, im.deref("offset_field"))("inp")))
    )("inp", "offset_field")

    actual = inline_dynamic_shifts.InlineDynamicShifts.apply(
        testee, offset_provider_type={}, uids=uids
    )
    assert actual == expected


def test_inline_dynamic_shift_concat_where_arg(uids):
    cond = im.less(im.axis_literal(IDim), im.ref("n", int_type))
    type_inference.reinfer(cond)
    testee = im.as_fieldop(dynamic_shift_stencil)(
        im.concat_where(cond, "inp1", "inp2"), "offset_field"
    )

    def shifted(it: str):
        return im.shift(IOff, "_cs_0")(it)

    expected = im.as_fieldop(
        im.lambda_("__iasfop_0", "inp1", "inp2", "n", "offset_field")(
            im.let("_cs_0", im.deref("offset_field"))(
                im.if_(
                    im.less(
                        im.tuple_get(0, im.make_tuple(im.deref(shifted("__iasfop_0")))),
                        im.deref(shifted("n")),
                    ),
                    im.deref(shifted("inp1")),
                    im.deref(shifted("inp2")),
                )
            )
        )
    )(im.index(IDim), "inp1", "inp2", "n", "offset_field")

    actual = inline_dynamic_shifts.InlineDynamicShifts.apply(
        testee, offset_provider_type={}, uids=uids
    )
    assert actual == expected


def test_inline_dynamic_shift_let_bound_tuple_element(uids):
    testee = im.let(
        "t", im.if_("cond", im.make_tuple("inp1", "inp2"), im.make_tuple("inp2", "inp1"))
    )(
        im.op_as_fieldop("plus")(
            im.as_fieldop(dynamic_shift_stencil)(im.tuple_get(1, "t"), "offset_field"),
            im.tuple_get(0, "t"),
        )
    )

    def shifted(it: str):
        return im.shift(IOff, "_cs_0")(it)

    expected = im.op_as_fieldop("plus")(
        im.as_fieldop(
            im.lambda_("cond", "inp2", "inp1", "offset_field")(
                im.let("_cs_0", im.deref("offset_field"))(
                    im.if_(
                        im.deref(shifted("cond")),
                        im.deref(shifted("inp2")),
                        im.deref(shifted("inp1")),
                    )
                )
            )
        )("cond", "inp2", "inp1", "offset_field"),
        im.if_("cond", "inp1", "inp2"),
    )

    actual = inline_dynamic_shifts.InlineDynamicShifts.apply(
        testee, offset_provider_type={}, uids=uids
    )
    assert actual == expected


def test_inline_dynamic_shift_let_var_shared_between_consumers(uids):
    testee = im.let("tmp", im.as_fieldop(im.lambda_("x")(im.multiplies_(im.deref("x"), 3)))("inp"))(
        im.op_as_fieldop("plus")(
            im.as_fieldop(dynamic_shift_stencil)("tmp", "offset_field1"),
            im.as_fieldop(dynamic_shift_stencil)("tmp", "offset_field2"),
        )
    )

    def expected_consumer(offset_field: str):
        return im.as_fieldop(
            im.lambda_("inp", offset_field)(
                im.multiplies_(im.deref(im.shift(IOff, im.deref(offset_field))("inp")), 3)
            )
        )("inp", offset_field)

    expected = im.op_as_fieldop("plus")(
        expected_consumer("offset_field1"), expected_consumer("offset_field2")
    )

    actual = inline_dynamic_shifts.InlineDynamicShifts.apply(
        testee, offset_provider_type={}, uids=uids
    )
    assert actual == expected


def test_inline_dynamic_shift_if_with_scalar_cond_expr(uids):
    cond = im.eq(im.ref("m", int_type), 3)
    testee = im.as_fieldop(dynamic_shift_stencil)(im.if_(cond, "inp1", "inp2"), "offset_field")

    def shifted(it: str):
        return im.shift(IOff, "_cs_0")(it)

    expected = im.as_fieldop(
        im.lambda_("__iasfop_0", "inp1", "inp2", "offset_field")(
            im.let("_cs_0", im.deref("offset_field"))(
                im.if_(
                    im.deref(shifted("__iasfop_0")),
                    im.deref(shifted("inp1")),
                    im.deref(shifted("inp2")),
                )
            )
        )
    )(im.eq("m", 3), "inp1", "inp2", "offset_field")

    actual = inline_dynamic_shifts.InlineDynamicShifts.apply(
        testee, offset_provider_type={}, uids=uids
    )
    assert actual == expected


def test_inline_dynamic_shift_let_arg(uids):
    testee = im.as_fieldop(dynamic_shift_stencil)(
        im.let("x", im.op_as_fieldop("plus")("inp1", "inp2"))(
            im.op_as_fieldop("multiplies")("x", "x")
        ),
        "offset_field",
    )

    def shifted(it: str):
        return im.shift(IOff, "_cs_1")(it)

    expected = im.as_fieldop(
        im.lambda_("inp1", "inp2", "offset_field")(
            im.let("_cs_1", im.deref("offset_field"))(
                im.let("_cs_0", im.plus(im.deref(shifted("inp1")), im.deref(shifted("inp2"))))(
                    im.multiplies_("_cs_0", "_cs_0")
                )
            )
        )
    )("inp1", "inp2", "offset_field")

    actual = inline_dynamic_shifts.InlineDynamicShifts.apply(
        testee, offset_provider_type={}, uids=uids
    )
    assert actual == expected
