# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from gt4py import next as gtx
from gt4py.next import errors
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms.normalize_shifts import CanonicalizeShiftOffsets


IDim = gtx.Dimension("IDim")
IOff = im.cartesian_offset(IDim)


def _shift_stencil(offset: itir.Expr) -> itir.Expr:
    return im.as_fieldop(
        im.lambda_("it", "off")(im.deref(im.call(im.call("shift")(IOff, offset))("it")))
    )("inp", "offset_field")


def test_literal_offset_becomes_offset_literal():
    actual = CanonicalizeShiftOffsets().visit(_shift_stencil(im.literal_from_value(3)))

    assert actual == _shift_stencil(itir.OffsetLiteral(value=3))


def test_offset_read_in_the_stencil_is_kept():
    testee = _shift_stencil(im.deref("off"))

    assert CanonicalizeShiftOffsets().visit(testee) == testee


def test_offset_from_outside_the_stencil_is_rejected():
    testee = im.let("m", im.literal_from_value(1))(_shift_stencil(im.minus("m", 1)))

    with pytest.raises(errors.NonStaticShiftError) as exc_info:
        CanonicalizeShiftOffsets().visit(testee)

    assert exc_info.value.symbols == ["m"]
