from typing import List

from gt4py.gtc.common import DataType, ExprKind, LoopOrder
from gt4py.gtc.gtir import (
    AxisBound,
    BlockStmt,
    CartesianOffset,
    Decl,
    Expr,
    FieldAccess,
    FieldIfStmt,
    Interval,
    ParAssignStmt,
    Stencil,
    Stmt,
    VerticalLoop,
)


class DummyExpr(Expr):
    """Fake expression for cases where a concrete expression is not needed."""

    dtype: DataType = DataType.FLOAT32
    kind: ExprKind = ExprKind.FIELD


class FieldAccessBuilder:
    def __init__(self, name) -> None:
        self._name = name
        self._offset = CartesianOffset.zero()
        self._kind = ExprKind.FIELD
        self._dtype = DataType.FLOAT32

    def offset(self, offset: CartesianOffset) -> "FieldAccessBuilder":
        self._offset = offset
        return self

    def dtype(self, dtype: DataType) -> "FieldAccessBuilder":
        self._dtype = dtype
        return self

    def build(self) -> FieldAccess:
        return FieldAccess(
            name=self._name, offset=self._offset, dtype=self._dtype, kind=self._kind
        )


class ParAssignStmtBuilder:
    def __init__(self, left_name=None, right_name=None) -> None:
        self._left = FieldAccessBuilder(left_name) if left_name else None
        self._right = FieldAccessBuilder(right_name) if right_name else None

    def left(self, left: FieldAccess) -> "ParAssignStmtBuilder":
        self._left = left
        return self

    def right(self, right: Expr) -> "ParAssignStmtBuilder":
        self._right = right
        return self

    def build(self) -> ParAssignStmt:
        return ParAssignStmt(left=self._left, right=self._right)


class FieldIfStmtBuilder:
    def __init__(self) -> None:
        self._cond = None
        self._true_branch = []
        self._false_branch = None

    def cond(self, cond: Expr) -> "FieldIfStmtBuilder":
        self._cond = cond
        return self

    def true_branch(self, true_branch: List[Stmt]) -> "FieldIfStmtBuilder":
        self._true_branch = true_branch
        return self

    def false_branch(self, false_branch: List[Stmt]) -> "FieldIfStmtBuilder":
        self._false_branch = false_branch
        return self

    def add_true_stmt(self, stmt: Stmt) -> "FieldIfStmtBuilder":
        self._true_branch.append(stmt)
        return self

    def add_false_stmt(self, stmt: Stmt) -> "FieldIfStmtBuilder":
        if not self._false_branch:
            self._false_branch = []
        self._false_branch.append(stmt)
        return self

    def build(self) -> FieldIfStmt:
        return FieldIfStmt(
            cond=self._cond,
            true_branch=BlockStmt(body=self._true_branch),
            false_branch=BlockStmt(body=self._false_branch) if self._false_branch else None,
        )


class StencilBuilder:
    def __init__(self, name="foo") -> None:
        self._name = name
        self._params = []
        self._vertical_loops = []

    def add_param(self, param: Decl) -> "StencilBuilder":
        self._params.append(param),
        return self

    def add_vertical_loop(self, vertical_loop: VerticalLoop) -> "StencilBuilder":
        self._vertical_loops.append(vertical_loop)
        return self

    def add_par_assign_stmt(self, par_assign_stmt: ParAssignStmt) -> "StencilBuilder":
        if len(self._vertical_loops) == 0:
            self._vertical_loops.append(  # TODO builder
                VerticalLoop(
                    interval=Interval(start=AxisBound.start(), end=AxisBound.end()),
                    loop_order=LoopOrder.FORWARD,
                    body=[],
                    temporaries=[],
                )
            )

        self._vertical_loops[-1].body.append(par_assign_stmt)
        return self

    def build(self) -> Stencil:
        return Stencil(
            name=self._name,
            params=self._params,
            vertical_loops=self._vertical_loops,
        )
