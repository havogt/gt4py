# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""PROTOTYPE (feasibility): split a full-column cell SetAt whose body is a K-index-masked
concat_where ternary into disjoint K-band SetAts (gtfn analog of dace's per-sub-interval map).

After `transform_to_as_fieldop` collapses a concat_where into a single masked
`if_(_in(index(K), ...), tb, fb)` over the union domain, the band structure survives only as
runtime predicates on `index(K)`. This pass recovers the bands: it finds the literal K
thresholds in those predicates, partitions the SetAt's vertical domain on them, and for each
band emits a SetAt over that disjoint K-sub-range with every K-index predicate constant-folded
to its band-constant truth value. gtfn lowers each SetAt to one executor with its vertical
range, so the bands become disjoint-K kernels that read shared inputs once (no full-column
re-stream). Env-gated, opt-in, prototype.
"""

from __future__ import annotations

from gt4py import eve
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.iterator.transforms import constant_folding


def _vertical_range(domain_expr: itir.Expr) -> tuple[common.Dimension, int, int] | None:
    try:
        dom = domain_utils.SymbolicDomain.from_expr(domain_expr)
    except Exception:
        return None
    for dim, rng in dom.ranges.items():
        if dim.kind == common.DimensionKind.VERTICAL:
            if isinstance(rng.start, itir.Literal) and isinstance(rng.stop, itir.Literal):
                return (dim, int(rng.start.value), int(rng.stop.value))
    return None


def _set_vertical_range(domain_expr: itir.Expr, dim: common.Dimension, start: int, stop: int):
    dom = domain_utils.SymbolicDomain.from_expr(domain_expr)
    dom.ranges[dim] = domain_utils.SymbolicRange(
        start=im.literal_from_value(start), stop=im.literal_from_value(stop)
    )
    return dom.as_expr()


def _is_raw_index_k(node: itir.Node, vdim: common.Dimension) -> bool:
    return (
        isinstance(node, itir.FunCall)
        and cpm.is_call_to(node, "index")
        and isinstance(node.args[0], itir.AxisLiteral)
        and node.args[0].value == vdim.value
    )


class KAliases:
    """Names whose value equals the K index.

    `transform_to_as_fieldop` passes `index(K)` as an as_fieldop SID argument, so the
    corresponding stencil param `p` is an *iterator* over the index field and the K value is
    `deref(p)`. That deref is then let-bound to another name (e.g. `_cs_109`) that the band
    predicates reference. We therefore track both:
      - `iters`: names `p` whose `deref(p)` is the K index (param fed an `index(K)` arg);
      - `vals` : names that directly *are* the K index value (bound to `deref(p)`, p∈iters,
                 or to a raw `index(K)`).
    """

    def __init__(self) -> None:
        self.iters: set[str] = set()
        self.vals: set[str] = set()


def _is_k_value(node: itir.Node, vdim: common.Dimension, ka: KAliases) -> bool:
    """True if `node` evaluates to the K index value (a scalar usable in K-predicates)."""
    if _is_raw_index_k(node, vdim):
        return True
    if isinstance(node, itir.SymRef) and node.id in ka.vals:
        return True
    # deref of an iterator over the index(K) field
    if cpm.is_call_to(node, "deref"):
        arg = node.args[0]
        if isinstance(arg, itir.SymRef) and arg.id in ka.iters:
            return True
    return False


def _collect_index_k_aliases(node: itir.Node, vdim: common.Dimension) -> KAliases:
    ka = KAliases()
    # fixpoint: each lambda application can introduce new iter/val aliases from already-known ones
    for _ in range(8):
        before = (len(ka.iters), len(ka.vals))
        for n in node.pre_walk_values():
            # plain lambda application: `(λ(params) → body)(args)`
            if (
                isinstance(n, itir.FunCall)
                and isinstance(n.fun, itir.Lambda)
                and len(n.fun.params) == len(n.args)
            ):
                for param, arg in zip(n.fun.params, n.args, strict=True):
                    if _is_raw_index_k(arg, vdim) or (
                        isinstance(arg, itir.SymRef) and arg.id in ka.iters
                    ):
                        ka.iters.add(param.id)  # iterator over index(K) field
                    elif _is_k_value(arg, vdim, ka):
                        ka.vals.add(param.id)  # K value passed through
            # applied as_fieldop: `as_fieldop(stencil, dom)(args)` — args bind stencil params
            elif cpm.is_applied_as_fieldop(n):
                stencil = n.fun.args[0]
                if isinstance(stencil, itir.Lambda) and len(stencil.params) == len(n.args):
                    for param, arg in zip(stencil.params, n.args, strict=True):
                        if _is_raw_index_k(arg, vdim) or (
                            isinstance(arg, itir.SymRef) and arg.id in ka.iters
                        ):
                            ka.iters.add(param.id)
        if (len(ka.iters), len(ka.vals)) == before:
            break
    return ka


def _is_index_k(node: itir.Node, vdim: common.Dimension, ka: KAliases) -> bool:
    return _is_k_value(node, vdim, ka)


class _CollectThresholds(eve.NodeVisitor):
    """Collect K-index comparison thresholds: `c <= index(K)`, `index(K) < c`, etc."""

    def __init__(self, vdim: common.Dimension, aliases: "KAliases"):
        self.vdim = vdim
        self.aliases = aliases
        self.thresholds: set[int] = set()

    def visit_FunCall(self, node: itir.FunCall) -> None:
        self.generic_visit(node)
        if cpm.is_call_to(node, ("less", "less_equal", "greater", "greater_equal", "eq")):
            a, b = node.args
            for x, y in ((a, b), (b, a)):
                if _is_index_k(x, self.vdim, self.aliases) and isinstance(y, itir.Literal):
                    v = int(y.value)
                    # boundary where a `<`/`<=`/`>=` flips: split at v and v+1 to be safe.
                    self.thresholds.add(v)
                    self.thresholds.add(v + 1)


class _FoldKPredicates(eve.PreserveLocationVisitor, eve.NodeTranslator):
    """Within a band [lo, hi) (hi exclusive, band non-empty), replace every K-index comparison
    with its constant truth value. K ∈ [lo, hi-1]."""

    def __init__(self, vdim: common.Dimension, lo: int, hi: int, aliases: "KAliases"):
        self.vdim = vdim
        self.lo = lo
        self.hi = hi  # exclusive
        self.aliases = aliases

    def visit_FunCall(self, node: itir.FunCall) -> itir.Node:
        node = self.generic_visit(node)
        if cpm.is_call_to(node, ("less", "less_equal", "greater", "greater_equal", "eq")):
            a, b = node.args
            kmin, kmax = self.lo, self.hi - 1
            # normalize to (index(K) OP literal)
            if _is_index_k(a, self.vdim, self.aliases) and isinstance(b, itir.Literal):
                c = int(b.value)
                op = node.fun.id
                lo_true = {
                    "less": kmin < c,
                    "less_equal": kmin <= c,
                    "greater": kmin > c,
                    "greater_equal": kmin >= c,
                    "eq": kmin == c,
                }[op]
                hi_true = {
                    "less": kmax < c,
                    "less_equal": kmax <= c,
                    "greater": kmax > c,
                    "greater_equal": kmax >= c,
                    "eq": kmax == c,
                }[op]
                if lo_true == hi_true:
                    return im.literal_from_value(lo_true)
            elif _is_index_k(b, self.vdim, self.aliases) and isinstance(a, itir.Literal):
                c = int(a.value)
                op = node.fun.id  # literal OP index(K)  ==  index(K) (flipped op) literal
                lo_true = {
                    "less": c < kmin,
                    "less_equal": c <= kmin,
                    "greater": c > kmin,
                    "greater_equal": c >= kmin,
                    "eq": c == kmin,
                }[op]
                hi_true = {
                    "less": c < kmax,
                    "less_equal": c <= kmax,
                    "greater": c > kmax,
                    "greater_equal": c >= kmax,
                    "eq": c == kmax,
                }[op]
                # NOTE: monotone in K only for inequalities; eq handled by lo==hi check
                if lo_true == hi_true:
                    return im.literal_from_value(lo_true)
        return node


def _has_k_index_predicate(expr: itir.Node, vdim: common.Dimension, aliases: "KAliases") -> bool:
    c = _CollectThresholds(vdim, aliases)
    c.visit(expr)
    return len(c.thresholds) > 0


class SplitConcatWhereKBands(eve.PreserveLocationVisitor, eve.NodeTranslator):
    @classmethod
    def apply(cls, program: itir.Program) -> itir.Program:
        new_body: list[itir.Stmt] = []
        for stmt in program.body:
            new_body.extend(cls._split_stmt(stmt))
        return itir.Program(
            id=program.id,
            function_definitions=program.function_definitions,
            params=program.params,
            declarations=program.declarations,
            body=new_body,
        )

    @staticmethod
    def _split_stmt(stmt: itir.Stmt) -> list[itir.Stmt]:
        if not isinstance(stmt, itir.SetAt):
            return [stmt]
        vr = _vertical_range(stmt.domain)
        if vr is None:
            return [stmt]
        vdim, vstart, vstop = vr
        aliases = _collect_index_k_aliases(stmt.expr, vdim)
        if not _has_k_index_predicate(stmt.expr, vdim, aliases):
            return [stmt]
        collector = _CollectThresholds(vdim, aliases)
        collector.visit(stmt.expr)
        cuts = sorted({vstart, vstop} | {t for t in collector.thresholds if vstart < t < vstop})
        if len(cuts) <= 2:
            return [stmt]
        bands: list[tuple[int, int, itir.Expr]] = []
        for lo, hi in zip(cuts[:-1], cuts[1:], strict=False):
            band_expr = _FoldKPredicates(vdim, lo, hi, aliases).visit(stmt.expr)
            band_expr = constant_folding.ConstantFolding.apply(band_expr)
            # coalesce adjacent bands whose folded body is structurally identical
            if bands and bands[-1][2] == band_expr:
                bands[-1] = (bands[-1][0], hi, bands[-1][2])
            else:
                bands.append((lo, hi, band_expr))
        if len(bands) <= 1:
            return [stmt]
        out: list[itir.Stmt] = []
        for lo, hi, band_expr in bands:
            band_dom = _set_vertical_range(stmt.domain, vdim, lo, hi)
            new = itir.SetAt(expr=band_expr, domain=band_dom, target=stmt.target)
            new.expr.annex.domain = domain_utils.SymbolicDomain.from_expr(band_dom)
            out.append(new)
        return out
