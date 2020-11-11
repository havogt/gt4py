import ast
import pydantic
from pydantic.error_wrappers import ValidationError

import pytest
from devtools import debug
from eve import SourceLocation

from gt4py.gtc.common import BinaryOperator, DataType, LevelMarker, LoopOrder
from gt4py.gtc.gtir import (
    AxisBound,
    BinaryOp,
    CartesianOffset,
    Computation,
    FieldAccess,
    FieldDecl,
    IfStmt,
    Stmt,
    Literal,
    VerticalInterval,
    VerticalLoop,
    ParAssignStmt,
    Expr,
)
from gt4py.gtc.python.python_naive_codegen import PythonNaiveCodegen


@pytest.fixture
def copy_assign():
    yield ParAssignStmt(
        loc=SourceLocation(line=3, column=2, source="copy_gtir"),
        left=FieldAccess.centered(
            name="a", loc=SourceLocation(line=3, column=1, source="copy_gtir")
        ),
        right=FieldAccess.centered(
            name="b", loc=SourceLocation(line=3, column=3, source="copy_gtir")
        ),
    )


@pytest.fixture
def copy_interval(copy_assign):
    yield VerticalInterval(
        loc=SourceLocation(line=2, column=11, source="copy_gtir"),
        start=AxisBound(level=LevelMarker.START, offset=0),
        end=AxisBound(level=LevelMarker.END, offset=0),
        body=[copy_assign],
    )


@pytest.fixture
def copy_v_loop(copy_interval):
    yield VerticalLoop(
        loc=SourceLocation(line=2, column=1, source="copy_gtir"),
        loop_order=LoopOrder.FORWARD,
        vertical_intervals=[copy_interval],
    )


@pytest.fixture
def copy_computation(copy_v_loop):
    yield Computation(
        name="copy_gtir",
        loc=SourceLocation(line=1, column=1, source="copy_gtir"),
        params=[
            FieldDecl(name="a", dtype=DataType.FLOAT32),
            FieldDecl(name="b", dtype=DataType.FLOAT32),
        ],
        vertical_loops=[copy_v_loop],
    )


def test_copy(copy_computation):
    print(debug(copy_computation))
    assert copy_computation
    assert copy_computation.param_names == ["a", "b"]


def test_naive_python_copy(copy_computation):
    assert ast.parse(PythonNaiveCodegen.apply(copy_computation))


def test_naive_python_avg():
    horizontal_avg = Computation(
        name="horizontal_avg",
        params=[
            FieldDecl(name="a", dtype=DataType.FLOAT32),
            FieldDecl(name="b", dtype=DataType.FLOAT32),
        ],
        vertical_loops=[
            VerticalLoop(
                loop_order=LoopOrder.FORWARD,
                vertical_intervals=[
                    VerticalInterval(
                        start=AxisBound(level=LevelMarker.START, offset=0),
                        end=AxisBound(level=LevelMarker.END, offset=0),
                        body=[
                            ParAssignStmt(
                                left=FieldAccess.centered(name="a"),
                                right=BinaryOp(
                                    left=FieldAccess(
                                        name="b",
                                        offset=CartesianOffset(i=-1, j=0, k=0),
                                    ),
                                    right=FieldAccess(
                                        name="b", offset=CartesianOffset(i=1, j=0, k=0)
                                    ),
                                    op=BinaryOperator.ADD,
                                ),
                            )
                        ],
                    )
                ],
            )
        ],
    )
    assert ast.parse(PythonNaiveCodegen.apply(horizontal_avg))


def test_ExprBaseclassIsNotInstantiatable():
    with pytest.raises(TypeError):
        Expr()


def test_StmtBaseclassIsNotInstantiatable():
    with pytest.raises(TypeError):
        Stmt()


class DummyExpr(Expr):
    """Fake expression for cases where a concrete expression is not needed."""


# Validation tests
def test_ParAssignStmtWithVerticalOffsetIsOk():
    ParAssignStmt(
        left=FieldAccess(name="foo", offset=CartesianOffset(i=0, j=0, k=1)), right=DummyExpr()
    )


def test_ParAssignStmtWithHorizontalOffsetIsError():
    with pytest.raises(ValidationError):
        ParAssignStmt(
            left=FieldAccess(name="foo", offset=CartesianOffset(i=1, j=0, k=0)), right=DummyExpr()
        )


def test_NonBooleanIfStmtConditionIsError():
    with pytest.raises(ValidationError):
        IfStmt(cond=DummyExpr(dtype=DataType.INT32), true_branch=[], false_branch=[])


def test_LiteralRequiresDtype():
    with pytest.raises(ValidationError):
        Literal(value="foo")


def test_BinaryOpErrorsForIncompatibleTypes():
    with pytest.raises(ValidationError):
        BinaryOp(left=DummyExpr(dtype=DataType.INT32), right=DummyExpr(dtype=DataType.INT16))
