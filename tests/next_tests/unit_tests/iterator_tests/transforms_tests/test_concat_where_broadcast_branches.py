# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pytest

from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import concat_where
from gt4py.next.type_system import type_specifications as ts

Vertex = common.Dimension(value="Vertex", kind=common.DimensionKind.HORIZONTAL)
K = common.Dimension(value="K", kind=common.DimensionKind.VERTICAL)

float64 = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
vertex_k_field = ts.FieldType(dims=[Vertex, K], dtype=float64)
vertex_field = ts.FieldType(dims=[Vertex], dtype=float64)
k_field = ts.FieldType(dims=[K], dtype=float64)

cond = im.domain(common.GridType.UNSTRUCTURED, {K: (itir.InfinityLiteral.NEGATIVE, 0)})


def _broadcast(expr):
    return im.call("broadcast")(
        expr,
        im.make_tuple(
            *(itir.AxisLiteral(value=dim.value, kind=dim.kind) for dim in (Vertex, K)),
        ),
    )


@pytest.mark.parametrize(
    "true_branch_type, false_branch_type",
    [
        (vertex_field, vertex_k_field),
        (vertex_k_field, k_field),
        (k_field, vertex_field),
    ],
)
def test_broadcast_branches(true_branch_type, false_branch_type):
    testee = im.concat_where(cond, im.ref("a", true_branch_type), im.ref("b", false_branch_type))

    expected = im.concat_where(
        cond,
        im.ref("a") if true_branch_type == vertex_k_field else _broadcast(im.ref("a")),
        im.ref("b") if false_branch_type == vertex_k_field else _broadcast(im.ref("b")),
    )

    assert concat_where.broadcast_branches(testee) == expected


def test_no_broadcast_when_branches_span_all_dimensions():
    testee = im.concat_where(cond, im.ref("a", vertex_k_field), im.ref("b", vertex_k_field))

    assert concat_where.broadcast_branches(testee) == testee


def test_no_broadcast_without_type_information():
    testee = im.concat_where(cond, "a", "b")

    assert concat_where.broadcast_branches(testee) == testee
