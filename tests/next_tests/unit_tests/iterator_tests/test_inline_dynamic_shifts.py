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
                        im.tuple_get(0, im.deref(shifted("__iasfop_0"))),
                        im.deref(shifted("n")),
                    ),
                    im.deref(shifted("inp1")),
                    im.deref(shifted("inp2")),
                )
            )
        )
    )(im.make_tuple(im.index(IDim)), "inp1", "inp2", "n", "offset_field")

    actual = inline_dynamic_shifts.InlineDynamicShifts.apply(
        testee, offset_provider_type={}, uids=uids
    )
    assert actual == expected
