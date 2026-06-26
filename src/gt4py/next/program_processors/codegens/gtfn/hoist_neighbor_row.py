# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Combined branchless skip-reduce + neighbor-row hoist (gtfn codegen pass).

When ``config.FN_BRANCHLESS_SKIP_REDUCE`` lowers a skip-value sum-reduction to the
branchless load-then-mask form (``unroll_reduce.py``), the unrolled ``_step`` fold
still contains a per-neighbor ``shift(field, Conn, _i)`` for both the ``deref`` and
the ``can_deref`` mask. Each ``shift`` re-runs ``neighbor_table_neighbors`` and
re-derives the row base ``m_index * index_stride`` (see ``tmp/iteraddr/addressing.md``).

This pass makes the branchless lowering (A) USE the hoisted row (B): it resolves the
neighbor row ONCE outside the fold via ``gtfn::neighbor_row(field, Conn)`` and replaces
every per-neighbor ``gtfn::shift(field, Conn, _i)`` with
``gtfn::horizontal_shift_to(field, __row, _i)`` (offset from the pre-resolved row).
Combined, the per-neighbor row re-derivation collapses to a single row resolution + N
element offsets, and the branchless deref hoists the K-row-base once.

Gated by ``config.FN_BRANCHLESS_SKIP_REDUCE`` and only fires on a reduction fold whose
step body already carries the branchless mask (a ``can_deref(shift(...))`` ternary over
the same per-neighbor ``shift`` it derefs). Default OFF, regression-safe.
"""

from __future__ import annotations

from typing import Any

from gt4py.eve import NodeTranslator
from gt4py.next.program_processors.codegens.gtfn import gtfn_ir, gtfn_ir_common


def _is_call_to(node: Any, name: str) -> bool:
    return (
        isinstance(node, gtfn_ir.FunCall)
        and isinstance(node.fun, gtfn_ir_common.SymRef)
        and node.fun.id == name
    )


def _is_horizontal_shift_over(node: Any, idx_id: str) -> bool:
    """``shift(field, Conn, _i)`` whose last arg is the per-neighbor index symbol."""
    if not _is_call_to(node, "shift"):
        return False
    if len(node.args) != 3:
        return False
    conn, idx = node.args[1], node.args[2]
    return (
        isinstance(conn, gtfn_ir.OffsetLiteral)
        and isinstance(conn.value, str)
        and isinstance(idx, gtfn_ir_common.SymRef)
        and idx.id == idx_id
    )


def _shift_key(shift: gtfn_ir.FunCall) -> tuple[str, str]:
    """(field-expr-repr, connectivity-tag) identifying one hoistable row."""
    return (str(shift.args[0]), shift.args[1].value)  # type: ignore[union-attr]


class _CollectShifts(NodeTranslator):
    def __init__(self, idx_id: str) -> None:
        self.idx_id = idx_id
        self.shifts: dict[tuple[str, str], gtfn_ir.FunCall] = {}

    def visit_FunCall(self, node: gtfn_ir.FunCall) -> gtfn_ir.FunCall:
        if _is_horizontal_shift_over(node, self.idx_id):
            self.shifts.setdefault(_shift_key(node), node)
        return self.generic_visit(node)  # type: ignore[return-value]


class _RewriteShifts(NodeTranslator):
    """Replace ``shift(field, Conn, _i)`` -> ``horizontal_shift_to(field, __row, _i)``."""

    def __init__(self, idx_id: str, row_of: dict[tuple[str, str], str]) -> None:
        self.idx_id = idx_id
        self.row_of = row_of

    def visit_FunCall(self, node: gtfn_ir.FunCall) -> gtfn_ir.Expr:
        if _is_horizontal_shift_over(node, self.idx_id):
            row_sym = self.row_of[_shift_key(node)]
            return gtfn_ir.FunCall(
                fun=gtfn_ir_common.SymRef(id="horizontal_shift_to"),
                args=[node.args[0], gtfn_ir_common.SymRef(id=row_sym), node.args[2]],
            )
        return self.generic_visit(node)  # type: ignore[return-value]


class HoistNeighborRow(NodeTranslator):
    """gtfn_ir -> gtfn_ir: hoist the neighbor row out of unrolled reduction folds.

    By default only the *branchless skip-reduce* fold is hoisted (it carries a
    ``can_deref`` value mask). With ``hoist_nonskip=True`` a plain non-skip fold
    (``reduce(plus, 0)(neighbors)`` over a non-skip connectivity, NO mask) is also
    hoisted: its per-neighbor ``shift(field, Conn, _i)`` re-derives the row base
    ``m_index * index_stride`` once per neighbor, which the row hoist collapses to a
    single ``neighbor_row`` resolution + N element picks.
    """

    def __init__(self, hoist_nonskip: bool = False) -> None:
        self._counter = 0
        self._hoist_nonskip = hoist_nonskip

    @classmethod
    def apply(cls, node: gtfn_ir.Program, hoist_nonskip: bool = False) -> gtfn_ir.Program:
        return cls(hoist_nonskip=hoist_nonskip).visit(node)

    def _is_branchless_reduce_fold(self, node: gtfn_ir.FunCall) -> gtfn_ir.Lambda | None:
        """If ``node`` is a reduction fold ``(lambda _step: _step(...))(step_fun)`` whose
        ``step_fun`` carries the branchless mask, return the ``step_fun`` lambda; else None.
        """
        if not (isinstance(node.fun, gtfn_ir.Lambda) and len(node.args) == 1):
            return None
        step_fun = node.args[0]
        if not (isinstance(step_fun, gtfn_ir.Lambda) and len(step_fun.params) == 2):
            return None
        idx_id = step_fun.params[1].id
        # Branchless signature: a `can_deref(shift(field, Conn, _i))` over the per-neighbor
        # index (the value mask emitted by unroll_reduce's branchless path).
        has_branchless_mask = any(
            _is_call_to(n, "can_deref")
            and len(n.args) == 1
            and _is_horizontal_shift_over(n.args[0], idx_id)
            for n in step_fun.pre_walk_values()
        )
        return step_fun if has_branchless_mask else None

    def _is_nonskip_reduce_fold(self, node: gtfn_ir.FunCall) -> gtfn_ir.Lambda | None:
        """A plain non-skip reduction fold ``(lambda _step: _step(...))(step_fun)`` whose
        ``step_fun`` derefs a per-neighbor ``shift(field, Conn, _i)`` but carries NO
        ``can_deref`` mask (no skip-value handling). Return ``step_fun`` else None.
        """
        if not (isinstance(node.fun, gtfn_ir.Lambda) and len(node.args) == 1):
            return None
        step_fun = node.args[0]
        if not (isinstance(step_fun, gtfn_ir.Lambda) and len(step_fun.params) == 2):
            return None
        idx_id = step_fun.params[1].id
        has_can_deref = any(_is_call_to(n, "can_deref") for n in step_fun.pre_walk_values())
        if has_can_deref:
            return None  # skip-value fold; handled by the branchless path only
        has_horizontal_shift = any(
            _is_horizontal_shift_over(n, idx_id) for n in step_fun.pre_walk_values()
        )
        return step_fun if has_horizontal_shift else None

    def visit_FunCall(self, node: gtfn_ir.FunCall) -> gtfn_ir.Expr:
        node = self.generic_visit(node)
        step_fun = self._is_branchless_reduce_fold(node)
        if step_fun is None and self._hoist_nonskip:
            step_fun = self._is_nonskip_reduce_fold(node)
        if step_fun is None:
            return node

        idx_id = step_fun.params[1].id
        collector = _CollectShifts(idx_id)
        collector.visit(step_fun.expr)
        if not collector.shifts:
            return node

        # One hoisted row per distinct (field, connectivity).
        row_of: dict[tuple[str, str], str] = {}
        row_args: list[gtfn_ir.Expr] = []
        row_params: list[gtfn_ir_common.Sym] = []
        for key, shift in collector.shifts.items():
            row_sym = f"__nrow_{self._counter}"
            self._counter += 1
            row_of[key] = row_sym
            row_params.append(gtfn_ir_common.Sym(id=row_sym))
            row_args.append(
                gtfn_ir.FunCall(
                    fun=gtfn_ir_common.SymRef(id="neighbor_row"),
                    args=[shift.args[0], shift.args[1]],
                )
            )

        new_step_fun = gtfn_ir.Lambda(
            params=step_fun.params,
            expr=_RewriteShifts(idx_id, row_of).visit(step_fun.expr),
        )
        new_fold = gtfn_ir.FunCall(fun=node.fun, args=[new_step_fun])

        # Wrap: `(lambda __nrow_k...: <fold>)(neighbor_row(field, Conn)...)`. The `[=]`
        # capture makes the rows available inside the inner step lambda.
        return gtfn_ir.FunCall(
            fun=gtfn_ir.Lambda(params=row_params, expr=new_fold),
            args=row_args,
        )
