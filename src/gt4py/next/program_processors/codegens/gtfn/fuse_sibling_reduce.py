# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Sibling-reduction fold-fusion (gtfn codegen pass).

When 2+ unrolled reduction folds in one stencil body reduce over the SAME
``(field, connectivity)`` gather (e.g. ``dwdx`` and ``dwdy`` both folding over
``w_old(C2E2CO)``), gtfn emits each as a separate unrolled fold that INDEPENDENTLY
re-gathers the same neighbor list — a redundant ``deref(shift(field, Conn, _i))``
per neighbor (extra LDG + duplicated per-neighbor address math). nvcc/ptxas cannot
CSE them: the folds have distinct iteration vars and gridtools::fn re-derives the
neighbor row per shift.

This pass fuses such siblings into ONE fold with a tuple accumulator, so the shared
``deref(shift(field, Conn, _i))`` is emitted once and threaded into each reducer's
own weight. No extra kernel/temp → identical DRAM traffic, a pure in-kernel compute
cut. It is the inverse of the vertical-shift/recompute fusion lever.

Gated by ``config.FN_FUSE_SIBLING_REDUCE`` / the ``enable_sibling_reduce_fusion``
translation-step flag. Default OFF, codegen byte-identical when unset.
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


def _shift_key(shift: gtfn_ir.FunCall) -> tuple[str, str]:
    """(field-expr-repr, connectivity-tag) identifying one gather. Shared with
    ``hoist_neighbor_row._shift_key``."""
    return (str(shift.args[0]), shift.args[1].value)  # type: ignore[union-attr]


def _is_reduce_fold(node: Any) -> gtfn_ir.Lambda | None:
    """If ``node`` is an unrolled reduction fold
    ``(λ _step: _step(_step(…, 0ₒ) …, Nₒ))(step_fun)`` whose ``step_fun`` is
    ``λ(_acc, _i): _acc + body``, return the ``step_fun`` lambda; else ``None``."""
    if not (
        isinstance(node, gtfn_ir.FunCall)
        and isinstance(node.fun, gtfn_ir.Lambda)
        and len(node.fun.params) == 1
        and len(node.args) == 1
    ):
        return None
    step_fun = node.args[0]
    if not (isinstance(step_fun, gtfn_ir.Lambda) and len(step_fun.params) == 2):
        return None
    return step_fun


def _fold_offset_literals(node: gtfn_ir.FunCall) -> list[gtfn_ir.OffsetLiteral] | None:
    """Walk the ``_step(_step(…, 0ₒ), …, Nₒ)`` chain in the fold's outer lambda body and
    return the per-neighbor ``OffsetLiteral`` indices in order; ``None`` if the body is
    not a clean nested ``_step`` chain (e.g. a non-trivial init)."""
    step_id = node.fun.params[0].id  # type: ignore[attr-defined]
    offsets: list[gtfn_ir.OffsetLiteral] = []
    cur: Any = node.fun.expr  # type: ignore[attr-defined]
    while (
        isinstance(cur, gtfn_ir.FunCall)
        and isinstance(cur.fun, gtfn_ir_common.SymRef)
        and cur.fun.id == step_id
        and len(cur.args) == 2
        and isinstance(cur.args[1], gtfn_ir.OffsetLiteral)
    ):
        offsets.append(cur.args[1])
        cur = cur.args[0]
    offsets.reverse()
    # The innermost ``cur`` must be the literal init (0.) that we restart the chain from.
    if not isinstance(cur, gtfn_ir.Literal):
        return None
    return offsets


def _collect_gather_shifts(
    step_fun: gtfn_ir.Lambda,
) -> dict[tuple[str, str], gtfn_ir.FunCall]:
    """All distinct ``shift(field, Conn, _i)`` over the fold index, keyed by gather."""
    idx_id = step_fun.params[1].id
    shifts: dict[tuple[str, str], gtfn_ir.FunCall] = {}
    for n in step_fun.pre_walk_values():
        if (
            _is_call_to(n, "shift")
            and len(n.args) == 3
            and isinstance(n.args[1], gtfn_ir.OffsetLiteral)
            and isinstance(n.args[1].value, str)
            and isinstance(n.args[2], gtfn_ir_common.SymRef)
            and n.args[2].id == idx_id
        ):
            shifts.setdefault(_shift_key(n), n)
    return shifts


class _ReplaceSym(NodeTranslator):
    """Replace every ``SymRef(old)`` with ``new`` (a fresh Expr) in a subtree."""

    def __init__(self, old: str, new: gtfn_ir.Expr) -> None:
        self.old = old
        self.new = new

    def visit_SymRef(self, node: gtfn_ir_common.SymRef) -> gtfn_ir.Expr:
        return self.new if node.id == self.old else node


class FuseSiblingReduce(NodeTranslator):
    """gtfn_ir -> gtfn_ir: fuse sibling reduction folds that share one gather.

    Fires on a ``make_tuple(...)`` whose args contain 2+ fusible siblings, where a
    sibling is either a bare reduction fold, or a ``TernaryExpr(cond, fold, false)``
    with a common ``cond`` (the concat_where guard). Each sibling's fold must:

    - have the SAME per-neighbor offset sequence (same fold arity/order),
    - gather EXACTLY ONE connectivity (the shared one), and
    - share that gather's ``(field, connectivity)`` key with the others.

    The matched siblings are replaced by a single fold over a tuple accumulator that
    derefs the shared gather once, and ``tuple_get(k, ...)`` projections back into the
    original slots (under the same guard, if any).
    """

    def __init__(self) -> None:
        self._counter = 0

    @classmethod
    def apply(cls, node: gtfn_ir.Program) -> gtfn_ir.Program:
        return cls().visit(node)

    @staticmethod
    def _sibling_payload(arg: gtfn_ir.Expr) -> tuple[gtfn_ir.FunCall, gtfn_ir.Expr | None]:
        """Return ``(fold, guard_cond)``: the inner fold and its concat_where guard (or
        ``None`` if the fold is bare). The guard is the ``cond`` of a wrapping ternary
        whose ``true_expr`` is the fold."""
        if isinstance(arg, gtfn_ir.TernaryExpr) and _is_reduce_fold(arg.true_expr) is not None:
            return arg.true_expr, arg.cond  # type: ignore[return-value]
        return arg, None  # type: ignore[return-value]

    def _try_fuse(self, args: list[gtfn_ir.Expr]) -> list[gtfn_ir.Expr] | None:
        """If a fusible sibling group exists among ``args``, return rewritten args (with
        a sentinel ``__fused`` SymRef in the projected slots wired by the caller); else
        ``None``. Returns the rewritten arg list plus binds the fused fold via a closure
        capture passed back through ``self._pending``.
        """
        # Group sibling indices by (gather_key, offset_sequence, guard_repr).
        groups: dict[tuple, list[int]] = {}
        meta: dict[int, tuple[gtfn_ir.Lambda, gtfn_ir.FunCall, gtfn_ir.FunCall, Any]] = {}
        for i, arg in enumerate(args):
            fold, guard = self._sibling_payload(arg)
            step_fun = _is_reduce_fold(fold)
            if step_fun is None:
                continue
            offsets = _fold_offset_literals(fold)
            if offsets is None:
                continue
            shifts = _collect_gather_shifts(step_fun)
            if len(shifts) != 1:
                continue  # fuse only single-gather folds (exactly-once deref semantics)
            (gkey, shift_node), = shifts.items()
            off_seq = tuple(o.value for o in offsets)
            guard_key = None if guard is None else str(guard)
            key = (gkey, off_seq, guard_key)
            groups.setdefault(key, []).append(i)
            meta[i] = (step_fun, fold, shift_node, guard)

        # Pick the first group with 2+ members.
        fuse_group: list[int] | None = None
        for key, members in groups.items():
            if len(members) >= 2:
                fuse_group = members
                break
        if fuse_group is None:
            return None

        # Build the fused fold over a tuple accumulator.
        rep_idx = fuse_group[0]
        rep_step_fun, _rep_fold, rep_shift, guard = meta[rep_idx]
        acc_id = f"__sracc_{self._counter}"
        i_id = f"__sri_{self._counter}"
        w_id = f"__srw_{self._counter}"
        self._counter += 1
        k = len(fuse_group)

        # Per-sibling body with `_acc -> tuple_get(slot, acc)`, `_i -> __sri`,
        # `deref(shift(field, Conn, _i)) -> __srw` (the shared gather, derefed once).
        # The shared gather is the rep's shift with its index renamed to __sri.
        gather_deref = _ReplaceSym(
            rep_step_fun.params[1].id, gtfn_ir_common.SymRef(id=i_id)
        ).visit(gtfn_ir.FunCall(fun=gtfn_ir_common.SymRef(id="deref"), args=[rep_shift]))
        new_step_bodies: list[gtfn_ir.Expr] = []
        for slot, idx in enumerate(fuse_group):
            step_fun, _fold, shift_node, _g = meta[idx]
            old_acc, old_i = step_fun.params[0].id, step_fun.params[1].id
            body = step_fun.expr
            # acc -> tuple_get(slot, __sracc)
            body = _ReplaceSym(
                old_acc,
                gtfn_ir.FunCall(
                    fun=gtfn_ir_common.SymRef(id="tuple_get"),
                    args=[gtfn_ir.OffsetLiteral(value=slot), gtfn_ir_common.SymRef(id=acc_id)],
                ),
            ).visit(body)
            # this sibling's own gather deref -> the shared __srw register
            this_deref = gtfn_ir.FunCall(
                fun=gtfn_ir_common.SymRef(id="deref"), args=[shift_node]
            )
            body = _ReplaceExprByRepr(
                str(this_deref), gtfn_ir_common.SymRef(id=w_id)
            ).visit(body)
            # remaining per-neighbor index _i -> shared __sri
            body = _ReplaceSym(old_i, gtfn_ir_common.SymRef(id=i_id)).visit(body)
            new_step_bodies.append(body)

        # step body: bind __srw once, return make_tuple(body_0, ..., body_{k-1})
        tuple_expr = gtfn_ir.FunCall(
            fun=gtfn_ir_common.SymRef(id="make_tuple"), args=new_step_bodies
        )
        let_body = gtfn_ir.FunCall(
            fun=gtfn_ir.Lambda(
                params=[gtfn_ir_common.Sym(id=w_id)], expr=tuple_expr
            ),
            args=[gather_deref],
        )
        new_step_fun = gtfn_ir.Lambda(
            params=[gtfn_ir_common.Sym(id=acc_id), gtfn_ir_common.Sym(id=i_id)],
            expr=let_body,
        )
        # outer fold: (λ _step: _step(_step(… init, off0) …, offN))(new_step_fun)
        offsets = _fold_offset_literals(meta[rep_idx][1])
        assert offsets is not None
        init_tuple = gtfn_ir.FunCall(
            fun=gtfn_ir_common.SymRef(id="make_tuple"),
            args=[gtfn_ir.Literal(value="0", type="float64") for _ in range(k)],
        )
        step_sym = gtfn_ir_common.SymRef(id=f"__srstep_{self._counter}")
        chain: gtfn_ir.Expr = init_tuple
        for off in offsets:
            chain = gtfn_ir.FunCall(fun=step_sym, args=[chain, off])
        fused_fold = gtfn_ir.FunCall(
            fun=gtfn_ir.Lambda(
                params=[gtfn_ir_common.Sym(id=step_sym.id)], expr=chain
            ),
            args=[new_step_fun],
        )

        # If guarded, wrap: cond ? fused_fold : make_tuple(false_0, ..., false_{k-1}).
        if guard is not None:
            false_tuple = gtfn_ir.FunCall(
                fun=gtfn_ir_common.SymRef(id="make_tuple"),
                args=[args[idx].false_expr for idx in fuse_group],  # type: ignore[union-attr]
            )
            fused_expr: gtfn_ir.Expr = gtfn_ir.TernaryExpr(
                cond=guard, true_expr=fused_fold, false_expr=false_tuple
            )
        else:
            fused_expr = fused_fold

        # Bind the fused tuple once and project into the original slots.
        fused_id = f"__srfused_{self._counter}"
        self._counter += 1
        new_args = list(args)
        for slot, idx in enumerate(fuse_group):
            new_args[idx] = gtfn_ir.FunCall(
                fun=gtfn_ir_common.SymRef(id="tuple_get"),
                args=[
                    gtfn_ir.OffsetLiteral(value=slot),
                    gtfn_ir_common.SymRef(id=fused_id),
                ],
            )
        self._pending = (fused_id, fused_expr)
        return new_args

    def visit_FunCall(self, node: gtfn_ir.FunCall) -> gtfn_ir.Expr:
        node = self.generic_visit(node)  # type: ignore[assignment]
        if not _is_call_to(node, "make_tuple"):
            return node
        self._pending = None  # type: ignore[assignment]
        new_args = self._try_fuse(list(node.args))
        if new_args is None:
            return node
        fused_id, fused_expr = self._pending  # type: ignore[misc]
        new_tuple = gtfn_ir.FunCall(fun=node.fun, args=new_args)
        # let-bind the fused fold around the rewritten make_tuple.
        return gtfn_ir.FunCall(
            fun=gtfn_ir.Lambda(params=[gtfn_ir_common.Sym(id=fused_id)], expr=new_tuple),
            args=[fused_expr],
        )


class _ReplaceExprByRepr(NodeTranslator):
    """Replace any subexpr whose ``str(...)`` equals ``target_repr`` with ``new``."""

    def __init__(self, target_repr: str, new: gtfn_ir.Expr) -> None:
        self.target_repr = target_repr
        self.new = new

    def generic_visit(self, node: Any, **kwargs: Any) -> Any:
        if isinstance(node, gtfn_ir_common.Node) and str(node) == self.target_repr:
            return self.new
        return super().generic_visit(node, **kwargs)
