# -*- coding: utf-8 -*-
from typing import List

from gtc.common import ArithmeticOperator, ComparisonOperator, DataType, ExprKind, NativeFunction
from gtc.gtir import BinaryOp, Cast, Expr, NativeFuncCall, ParAssignStmt, TernaryOp
from gtc.passes.gtir_upcaster import _GTIRUpcasting

from .gtir_utils import FieldAccessBuilder, make_Literal


class Placeholder(Expr):
    """Used as placeholder for an expression in comparison."""

    kind = ExprKind.SCALAR  # unused
    dtype = DataType.AUTO  # unused


A_BOOL_LITERAL = make_Literal("", dtype=DataType.BOOL)
A_INT64_LITERAL = make_Literal("", dtype=DataType.INT64)
A_FLOAT64_LITERAL = make_Literal("", dtype=DataType.FLOAT64)
AN_UNIMPORTANT_LITERAL = make_Literal("", dtype=DataType.DEFAULT)


def contains_cast_node(cast_node, expr):
    # Checks if `expr` contains `cast_node`. If `cast_node` contains `expr=Placeholder()`
    # we skip equality check of `expr` and `cast_node.expr`
    return (
        len(
            expr.iter_tree()
            .if_isinstance(Cast)
            .filter(
                lambda node: node.dtype == cast_node.dtype
                and (isinstance(cast_node.expr, Placeholder) or node.expr == cast_node.expr)
            )
            .to_list()
        )
        == 1
    )


def upcast_and_validate(expr, expected_cast_nodes):

    assert isinstance(expected_cast_nodes, List)
    assert all([isinstance(cast, Cast) for cast in expected_cast_nodes])
    assert all([not contains_cast_node(cast, expr) for cast in expected_cast_nodes])

    result: Expr = _GTIRUpcasting().visit(expr)

    for cast in expected_cast_nodes:
        assert contains_cast_node(cast, result)


def test_upcast_BinaryOp_BOOL_to_FLOAT():
    testee = BinaryOp(op=ArithmeticOperator.ADD, left=A_BOOL_LITERAL, right=A_FLOAT64_LITERAL)
    upcast_and_validate(testee, [Cast(dtype=DataType.FLOAT64, expr=A_BOOL_LITERAL)])


def test_upcast_BinaryOp_INT_to_FLOAT():
    testee = BinaryOp(op=ArithmeticOperator.ADD, left=A_INT64_LITERAL, right=A_FLOAT64_LITERAL)
    upcast_and_validate(testee, [Cast(dtype=DataType.FLOAT64, expr=A_INT64_LITERAL)])


def test_upcast_BinaryOp_nested_casting():
    outer_expr = BinaryOp(op=ArithmeticOperator.ADD, left=A_BOOL_LITERAL, right=A_INT64_LITERAL)
    testee = BinaryOp(op=ArithmeticOperator.ADD, left=outer_expr, right=A_FLOAT64_LITERAL)
    expected = [
        Cast(dtype=DataType.INT64, expr=A_BOOL_LITERAL),
        Cast(dtype=DataType.FLOAT64, expr=Placeholder()),
    ]
    upcast_and_validate(testee, expected)


def test_upcast_NativeFuncCall():
    testee = NativeFuncCall(func=NativeFunction.MAX, args=[A_INT64_LITERAL, A_FLOAT64_LITERAL])
    upcast_and_validate(testee, [Cast(dtype=DataType.FLOAT64, expr=A_INT64_LITERAL)])


def test_upcast_ParAssignStmt():
    testee = ParAssignStmt(
        left=FieldAccessBuilder("out").dtype(DataType.FLOAT64).build(), right=A_INT64_LITERAL
    )
    upcast_and_validate(testee, [Cast(dtype=DataType.FLOAT64, expr=A_INT64_LITERAL)])


def test_upcast_TernaryOp():
    testee = TernaryOp(
        cond=A_BOOL_LITERAL,
        true_expr=A_INT64_LITERAL,
        false_expr=A_FLOAT64_LITERAL,
    )
    upcast_and_validate(testee, [Cast(dtype=DataType.FLOAT64, expr=A_INT64_LITERAL)])


def test_upcast_in_cond_of_TernaryOp():
    testee = TernaryOp(
        cond=BinaryOp(op=ComparisonOperator.GE, left=A_INT64_LITERAL, right=A_FLOAT64_LITERAL),
        true_expr=AN_UNIMPORTANT_LITERAL,
        false_expr=AN_UNIMPORTANT_LITERAL,
    )
    upcast_and_validate(testee, [Cast(dtype=DataType.FLOAT64, expr=A_INT64_LITERAL)])
