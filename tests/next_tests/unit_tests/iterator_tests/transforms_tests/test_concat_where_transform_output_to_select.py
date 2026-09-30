# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py import eve
from gt4py.next.experimental import concat_where
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.iterator.transforms import pass_manager


Cell = gtx.Dimension("Cell")
KDim = gtx.Dimension("KDim", kind=gtx.DimensionKind.VERTICAL)
CKField = gtx.Field[gtx.Dims[Cell, KDim], float]


@gtx.field_operator
def _output_selection(a: CKField, b: CKField, n: gtx.int32, m: gtx.int32) -> CKField:
    c = a + b
    c = concat_where((KDim >= 1) & (KDim < n), c * 2.0, c)
    return concat_where(KDim < m, c, a)


@gtx.program
def output_selection(a: CKField, b: CKField, n: gtx.int32, m: gtx.int32, out: CKField):
    _output_selection(a, b, n, m, out=out, domain={Cell: (0, 10), KDim: (0, 8)})


@gtx.field_operator
def _intermediate_selection(a: CKField, b: CKField, n: gtx.int32) -> CKField:
    return concat_where(KDim < n, a, b) * 2.0


@gtx.program
def intermediate_selection(a: CKField, b: CKField, n: gtx.int32, out: CKField):
    _intermediate_selection(a, b, n, out=out, domain={Cell: (0, 10), KDim: (0, 8)})


@gtx.field_operator
def _no_selection(a: CKField, b: CKField) -> CKField:
    return a + b


@gtx.program
def no_selection(a: CKField, b: CKField, out: CKField):
    _no_selection(a, b, out=out, domain={Cell: (0, 10), KDim: (0, 8)})


@gtx.field_operator
def _tuple_in_condition(a: CKField, shape: tuple[gtx.int32, gtx.int32]) -> CKField:
    return concat_where(KDim == shape[1] - 1, 2.0, a)


@gtx.program
def tuple_in_condition(a: CKField, shape: tuple[gtx.int32, gtx.int32], out: CKField):
    _tuple_in_condition(a, shape, out=out, domain={Cell: (0, 10), KDim: (0, 8)})


def _concat_wheres(node: itir.Node) -> list[itir.FunCall]:
    return eve.walk_values(node).filter(lambda n: cpm.is_call_to(n, "concat_where")).to_list()


def _applied_as_fieldops(node: itir.Node) -> list[itir.FunCall]:
    return eve.walk_values(node).filter(cpm.is_applied_as_fieldop).to_list()


def _transformed(program: gtx.typing.Program) -> itir.Program:
    return pass_manager.apply_fieldview_transforms(program.gtir, offset_provider={})


def test_output_concat_where_becomes_one_selection():
    result = _transformed(output_selection)

    assert not _concat_wheres(result)
    stmt = result.body[0]
    assert isinstance(stmt, itir.SetAt)
    # the selection and the producer of `c`, which both branches read; the `c * 2.0` branch is fused
    as_fieldops = _applied_as_fieldops(stmt.expr)
    assert len(as_fieldops) == 2
    assert all(hasattr(node.annex, "domain") for node in as_fieldops)


def test_intermediate_concat_where_is_kept():
    result = _transformed(intermediate_selection)

    assert len(_concat_wheres(result)) == 1


def test_program_without_concat_where_is_unchanged():
    program = no_selection.gtir
    reference = pass_manager.apply_fieldview_transforms(program, offset_provider={})
    assert not _concat_wheres(reference)
    assert len(_applied_as_fieldops(reference.body[0].expr)) == 1


def test_concat_where_with_tuple_in_condition_is_kept():
    result = _transformed(tuple_in_condition)

    assert _concat_wheres(result)
    assert not _applied_as_fieldops(result.body[0].expr)
