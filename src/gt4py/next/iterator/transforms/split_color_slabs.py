# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Split the colour range of structured-layout statements into one statement per colour."""

from __future__ import annotations

from gt4py import eve
from gt4py.next import common
from gt4py.next.iterator import builtins, ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.transforms import symbol_ref_utils
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas


#: bound on fold rounds per colour; each round strictly shrinks the stencil or stops
_MAX_FOLD_ROUNDS = 10


class _ReplaceDerefs(eve.PreserveLocationVisitor, eve.NodeTranslator):
    def visit_FunCall(self, node: itir.FunCall, *, values: dict[str, itir.Expr]) -> itir.Expr:
        if cpm.is_call_to(node, "deref") and isinstance(sym := node.args[0], itir.SymRef):
            if sym.id in values:
                return values[sym.id]
        return self.generic_visit(node, values=values)


def _literal_bounds(domain: domain_utils.SymbolicDomain, dim: common.Dimension) -> range | None:
    r = domain.ranges.get(dim)
    if r is None or not all(isinstance(b, itir.Literal) for b in (r.start, r.stop)):
        return None
    return range(int(r.start.value), int(r.stop.value))  # type: ignore[attr-defined]  # literals, checked above


def _index_params(stencil: itir.Lambda, args: list[itir.Expr], dim: common.Dimension) -> list[int]:
    """Positions of `index(dim)` arguments whose parameter is only ever dereferenced."""
    positions = []
    for i, (param, arg) in enumerate(zip(stencil.params, args, strict=True)):
        if not (
            cpm.is_call_to(arg, "index")
            and isinstance(axis := arg.args[0], itir.AxisLiteral)
            and axis.value == dim.value
        ):
            continue
        refs = symbol_ref_utils.collect_symbol_refs(stencil.expr, [param.id])
        derefs = [
            node
            for node in stencil.expr.pre_walk_values()
            if cpm.is_call_to(node, "deref") and node.args[0] == im.ref(param.id)
        ]
        shadowed = any(
            isinstance(node, itir.Lambda) and param.id in (p.id for p in node.params)
            for node in stencil.expr.pre_walk_values()
        )
        # legality: a shifted or forwarded index iterator has no single value per slab
        if not shadowed and len(refs) == len(derefs):
            positions.append(i)
    return positions


def _fold(expr: itir.Expr) -> itir.Expr:
    for _ in range(_MAX_FOLD_ROUNDS):
        folded = ConstantFolding.apply(InlineLambdas.apply(expr, opcount_preserving=True))
        assert isinstance(folded, itir.Expr)
        if folded == expr:
            break
        expr = folded
    return expr


def _split_set_at(stmt: itir.SetAt, color_dims: list[common.Dimension]) -> list[itir.Stmt]:
    expr = stmt.expr
    if not (
        cpm.is_applied_as_fieldop(expr)
        and isinstance(stencil := expr.fun.args[0], itir.Lambda)
        and cpm.is_call_to(stmt.domain, "cartesian_domain")
    ):
        return [stmt]
    domain = domain_utils.SymbolicDomain.from_expr(stmt.domain)
    for dim in color_dims:
        colors = _literal_bounds(domain, dim)
        positions = _index_params(stencil, expr.args, dim)
        if colors is None or not positions:
            continue
        params = [p for i, p in enumerate(stencil.params) if i not in positions]
        args = [a for i, a in enumerate(expr.args) if i not in positions]
        slabs: list[itir.Stmt] = []
        for color in colors:
            value = im.literal(str(color), builtins.INTEGER_INDEX_BUILTIN)
            body = _ReplaceDerefs().visit(
                stencil.expr, values={stencil.params[i].id: value for i in positions}
            )
            slab = domain_utils.SymbolicDomain(
                domain.grid_type,
                {
                    **domain.ranges,
                    dim: domain_utils.SymbolicRange(
                        im.literal(str(color), builtins.INTEGER_INDEX_BUILTIN),
                        im.literal(str(color + 1), builtins.INTEGER_INDEX_BUILTIN),
                    ),
                },
            ).as_expr()
            slabs.append(
                itir.SetAt(
                    expr=im.as_fieldop(im.lambda_(*params)(_fold(body)), slab)(*args),
                    domain=slab,
                    target=stmt.target,
                    location=stmt.location,
                )
            )
        return slabs
    return [stmt]


def _split_stmts(stmts: list[itir.Stmt], color_dims: list[common.Dimension]) -> list[itir.Stmt]:
    result: list[itir.Stmt] = []
    for stmt in stmts:
        if isinstance(stmt, itir.SetAt):
            result.extend(_split_set_at(stmt, color_dims))
        elif isinstance(stmt, itir.IfStmt):
            result.append(
                itir.IfStmt(
                    cond=stmt.cond,
                    true_branch=_split_stmts(stmt.true_branch, color_dims),
                    false_branch=_split_stmts(stmt.false_branch, color_dims),
                    location=stmt.location,
                )
            )
        else:
            result.append(stmt)
    return result


def split_color_slabs(
    program: itir.Program, *, offset_provider_type: common.OffsetProviderType
) -> itir.Program:
    """
    Write each colour of a structured-layout statement with its own statement.

    A statement over the colour dimension `X` of a structured connectivity whose stencil reads
    `index(X)` becomes one statement per colour `c`, on the domain slice `X: [c, c+1[`, with
    every `·it` of that index iterator replaced by `c` and the stencil folded::

        out @ c⟨ Iₕ: [0, 5[, Xₕ: [0, 2[ ⟩ ← as_fieldop(λ(x, e) → if ·x < 1 then ·e else ·⟪Iₕ, 1ₒ⟫(e), d)(index(Xₕ), e)
        →
        out @ c⟨ Iₕ: [0, 5[, Xₕ: [0, 1[ ⟩ ← as_fieldop(λ(e) → ·e, d₀)(e)
        out @ c⟨ Iₕ: [0, 5[, Xₕ: [1, 2[ ⟩ ← as_fieldop(λ(e) → ·⟪Iₕ, 1ₒ⟫(e), d₁)(e)

    The per-colour `if_` chains `StructuredToCartesian` builds then disappear, and with them the
    registers each kernel spends keeping every colour's operands live.

    Preconditions: runs after the temporaries are extracted (one applied `as_fieldop` per
    statement); the colour range must be a literal (static domains). Statements whose index
    iterator is shifted or passed on are left alone (legality). Growth: the statement count by
    the number of colours (at most a handful); each copy is folded, so it is not larger than the
    original stencil.
    """
    color_dims = list(
        dict.fromkeys(  # ordered set for reproducibility
            t.color_dim
            for t in offset_provider_type.values()
            if isinstance(t, common.StructuredConnectivityType)
        )
    )
    if not color_dims:
        return program
    body = _split_stmts(program.body, color_dims)
    if body == program.body:
        return program
    return itir.Program(
        id=program.id,
        function_definitions=program.function_definitions,
        params=program.params,
        declarations=program.declarations,
        body=body,
        location=program.location,
    )
