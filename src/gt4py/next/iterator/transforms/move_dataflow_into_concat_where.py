# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""Move branch-only (concat_where sub-domain) pointwise dataflow into its consumers.

gtfn analog of dace's ``MoveDataflowIntoIfBody``: a ``concat_where`` over a vertical sub-range
(e.g. the CFL-clip damping layer ``K[10,37[``) makes ``infer_domain`` assign the branch producers a
sub-domain. ``FuseAsFieldOp`` then refuses to fuse those (multi-use / sub-domain) into the full-K
consumers, so ``global_tmps`` materializes them and the surrounding full-K cell chain is split into
several kernels that each re-read the shared full-K fields (the dominant traffic excess vs dace).

This pass force-inlines such *pointwise, vertically-restricted* let-bound producers into their uses.
Inlining is safe because every consumer already reads the producer only within its (smaller) domain,
so the producer's values outside are never observed; computing them on the larger consumer domain
only wastes (cheap, pointwise) FLOPs, which is free on these memory-bound kernels. ``FuseAsFieldOp``
then folds the inlined producer into the consumer (computed on the consumer's domain) and CSE shares
it, collapsing the cell chain into one kernel and removing the re-reads. Env-gated, opt-in.
"""

from __future__ import annotations

from gt4py import eve
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, domain_utils
from gt4py.next.iterator.transforms import inline_lambdas, trace_shifts


def _vertical_range(node: itir.Node) -> tuple[int, int] | None:
    """(start, stop) of the vertical range of an applied as_fieldop with a literal domain, else None."""
    if not (cpm.is_applied_as_fieldop(node) and len(node.fun.args) > 1):  # type: ignore[attr-defined]
        return None
    try:
        dom = domain_utils.SymbolicDomain.from_expr(node.fun.args[1])  # type: ignore[attr-defined]
    except Exception:
        return None
    for dim, rng in dom.ranges.items():
        if dim.kind == common.DimensionKind.VERTICAL:
            if isinstance(rng.start, itir.Literal) and isinstance(rng.stop, itir.Literal):
                return (int(rng.start.value), int(rng.stop.value))
    return None


def _vertical_start_expr(domain: itir.Node) -> itir.Expr | None:
    """The vertical-range start expression of a domain literal, else None."""
    try:
        dom = domain_utils.SymbolicDomain.from_expr(domain)
    except Exception:
        return None
    for dim, rng in dom.ranges.items():
        if dim.kind == common.DimensionKind.VERTICAL:
            return rng.start
    return None


def _is_below_start(start: itir.Expr, valid_starts: set[str], min_literal_start: int | None) -> bool:
    """True if `start` extends a producer below the program's valid vertical domain start.

    The fusion satisfies a ``Koff[-1]`` consumer by extending a producer's vertical domain down
    one level. When that producer dereferences an input field, the extra level reads past the
    field's allocation (the field only exists on the program's declared vertical range), an OOB
    gather. Detect the two shapes the extension takes: a literal start below the smallest declared
    literal start (compile-time domain, e.g. ``-1`` vs ``0``), or a ``minus(s, +lit)`` of a declared
    start symbol (dynamic domain, e.g. ``vertical_start - 1``).
    """
    if isinstance(start, itir.Literal):
        return min_literal_start is not None and int(start.value) < min_literal_start
    if cpm.is_call_to(start, "minus"):
        lhs, rhs = start.args  # type: ignore[attr-defined]
        if (
            str(lhs) in valid_starts
            and isinstance(rhs, itir.Literal)
            and int(rhs.value) > 0
        ):
            return True
    return False


def fusion_overextends_below_vertical_start(program: itir.Program) -> bool:
    """True if the (post-fusion, domain-inferred) program extends a producer below the program's
    valid vertical start — an OOB field gather. Used by the validate-or-revert gate to fall back to
    the unfused (correct) lowering for such a program.
    """
    valid_starts, min_literal_start = _program_valid_starts(program)
    if not valid_starts:
        return False
    for node in program.pre_walk_values():
        if cpm.is_applied_as_fieldop(node) and len(node.fun.args) > 1:  # type: ignore[attr-defined]
            start = _vertical_start_expr(node.fun.args[1])  # type: ignore[attr-defined]
            if start is not None and _is_below_start(start, valid_starts, min_literal_start):
                return True
    return False


def _clamp_below_start(
    start: itir.Expr, valid_starts: set[str], min_literal_start: int | None
) -> itir.Expr | None:
    """If `start` extends below the program's valid vertical start, the in-bounds start to clamp it
    to (the smallest declared literal start, or the declared start symbol of a ``minus(s, +lit)``);
    else None (no clamp needed).
    """
    if isinstance(start, itir.Literal):
        if min_literal_start is not None and int(start.value) < min_literal_start:
            return itir.Literal(value=str(min_literal_start), type=start.type)
        return None
    if cpm.is_call_to(start, "minus"):
        lhs, rhs = start.args  # type: ignore[attr-defined]
        if str(lhs) in valid_starts and isinstance(rhs, itir.Literal) and int(rhs.value) > 0:
            return lhs
    return None


def _domain_exprs(domain: itir.Node) -> list[itir.Node]:
    """The per-output domain exprs of a SetAt domain: the domain itself, or — for a multi-output
    (tuple) SetAt whose domain is a ``make_tuple`` of domains — each element domain.
    """
    if cpm.is_call_to(domain, "make_tuple"):
        return list(domain.args)  # type: ignore[attr-defined]
    return [domain]


def _program_valid_starts(program: itir.Program) -> tuple[set[str], int | None]:
    valid_starts: set[str] = set()
    min_literal_start: int | None = None
    for stmt in program.body:
        if isinstance(stmt, itir.SetAt):
            for dom in _domain_exprs(stmt.domain):
                start = _vertical_start_expr(dom)
                if start is None:
                    continue
                valid_starts.add(str(start))
                if isinstance(start, itir.Literal):
                    v = int(start.value)
                    min_literal_start = (
                        v if min_literal_start is None else min(min_literal_start, v)
                    )
    return valid_starts, min_literal_start


def _clamp_domain_expr(
    domain: itir.Expr, valid_starts: set[str], min_literal_start: int | None
) -> itir.Expr | None:
    """Return `domain` with any below-start vertical start raised to vertical_start, else None."""
    try:
        dom = domain_utils.SymbolicDomain.from_expr(domain)
    except Exception:
        return None
    changed = False
    new_ranges = dict(dom.ranges)
    for dim, rng in dom.ranges.items():
        if dim.kind != common.DimensionKind.VERTICAL:
            continue
        clamp = _clamp_below_start(rng.start, valid_starts, min_literal_start)
        if clamp is not None:
            new_ranges[dim] = domain_utils.SymbolicRange(clamp, rng.stop)
            changed = True
    if not changed:
        return None
    return domain_utils.SymbolicDomain(dom.grid_type, new_ranges).as_expr()


def below_start_temporaries(program: itir.Program) -> list[str]:
    """Names of post-`global_tmps` temporaries whose declared vertical-domain start is below the
    program's valid vertical start.
    """
    valid_starts, min_literal_start = _program_valid_starts(program)
    names: list[str] = []
    if not valid_starts:
        return names
    for decl in program.declarations:
        if isinstance(decl, itir.Temporary) and decl.domain is not None:
            start = _vertical_start_expr(decl.domain)
            if start is not None and _is_below_start(start, valid_starts, min_literal_start):
                names.append(decl.id)
    return names


def _negative_koff_shift_of(arg: itir.Node, param: str) -> bool:
    """True if `arg` is a vertical shift with a negative literal offset applied to `param`:
    ``(shift(Koff, neg))(param)`` — the curried form ``FunCall(fun=shift(...), args=[param])``.
    """
    if not (
        isinstance(arg, itir.FunCall)
        and cpm.is_call_to(arg.fun, "shift")
        and len(arg.args) == 1
        and isinstance(arg.args[0], itir.SymRef)
        and arg.args[0].id == param
    ):
        return False
    offsets = arg.fun.args  # type: ignore[attr-defined]
    return (
        len(offsets) == 2
        and isinstance(offsets[1], itir.OffsetLiteral)
        and isinstance(offsets[1].value, int)
        and offsets[1].value < 0
    )


def _param_below_start_access(stencil: itir.Lambda, param: str) -> tuple[bool, bool]:
    """Classify how `param` is accessed inside `stencil`, returning ``(safe, has_guarded_shift)``.

    ``safe`` is False if any access of `param` could reach *below* its field's own start without a
    guard: an *unguarded* negative-``Koff`` shifted deref, or any unknown/dynamic shift of `param`.
    Center / positive reads never go below start (consumer domains are ``>= vertical_start``), so
    they are always safe. ``has_guarded_shift`` records whether a guarded negative-``Koff`` shift was
    seen — that is the access that drove the over-extension and that the clamp restores to safety.

    A negative-``Koff`` shift is "guarded" if it lies in the ``then``/``else`` branch of some ``if_``
    (the lowered ``concat_where`` boundary that short-circuits the below-start access).
    """
    # Collect, for each `if_` node, the set of node-ids in its branch subtrees.
    guarded_ids: set[int] = set()
    for node in stencil.pre_walk_values():
        if cpm.is_call_to(node, "if_") and len(node.args) == 3:  # type: ignore[attr-defined]
            for branch in node.args[1:]:  # type: ignore[attr-defined]
                for sub in branch.pre_walk_values():
                    guarded_ids.add(id(sub))

    safe = True
    has_guarded_shift = False
    for node in stencil.pre_walk_values():
        if not cpm.is_call_to(node, "deref"):
            continue
        arg = node.args[0]  # type: ignore[attr-defined]
        if _negative_koff_shift_of(arg, param):
            if id(node) in guarded_ids:
                has_guarded_shift = True
            else:
                safe = False
            continue
        if (
            isinstance(arg, itir.FunCall)
            and cpm.is_call_to(arg.fun, "shift")
            and any(isinstance(a, itir.SymRef) and a.id == param for a in arg.args)
        ):
            safe = False  # non-negative / dynamic shift of param
    return safe, has_guarded_shift


def _temp_reads_only_at_negative_shift(program: itir.Program, name: str) -> bool:
    """True if temporary `name` can be safely start-clamped: every below-start (negative-``Koff``)
    access of `name` across all its consumers is guarded by an enclosing ``if_`` (the ``concat_where``
    boundary), no access uses an unguarded / dynamic below-start shift, and at least one guarded
    negative-``Koff`` shift exists (the access that drove the over-extension). Center / positive reads
    are allowed. At the post-`global_tmps` stage `name` is passed as an arg to consumer
    ``as_fieldop``s; map it to each stencil param to classify the accesses.
    """
    any_guarded_shift = False
    seen = False
    for node in program.pre_walk_values():
        if not cpm.is_applied_as_fieldop(node):
            continue
        stencil = node.fun.args[0]  # type: ignore[attr-defined]
        for pos, arg in enumerate(node.args):
            if isinstance(arg, itir.SymRef) and arg.id == name:
                seen = True
                if not isinstance(stencil, itir.Lambda) or pos >= len(stencil.params):
                    return False
                safe, has_guarded_shift = _param_below_start_access(
                    stencil, stencil.params[pos].id
                )
                if not safe:
                    return False
                any_guarded_shift = any_guarded_shift or has_guarded_shift
    return seen and any_guarded_shift


class _ClampMaterializedTempsBelowStart(eve.PreserveLocationVisitor, eve.NodeTranslator):
    """Raise a materialized (post-``global_tmps``) temporary's vertical-domain start that the fused
    program pushed below the program's valid vertical start back up to that start.

    The fusion satisfies a ``Koff[-1]`` consumer of a connectivity-reduction producer by extending
    the (now materialized) producer temp's domain one level below ``vertical_start``; its producer
    kernel then dereferences the gathered input field at that level — an OOB gather. The below-start
    level is only ever *read* by the consumer through a ``concat_where`` boundary guard (a C++
    ternary that short-circuits, e.g. ``1 <= K ? ... shift(t, Koff, -1) : ...``), so the temp's
    value there is never observed. Clamping the temp's start up to ``vertical_start`` is therefore
    value-preserving: the buffer stores ``f(k)`` at absolute index ``k`` regardless of its start, so
    every in-bounds read ``temp[K-1]`` for the guarded ``K >= vertical_start+1`` is retained, while
    the dropped (``K-1 < vertical_start``) read is dead. It keeps the producer fused (the kernel-
    count reduction that motivates the fusion), unlike reverting to the unfused lowering.
    """

    def __init__(self, valid_starts: set[str], min_literal_start: int | None):
        self.valid_starts = valid_starts
        self.min_literal_start = min_literal_start

    def visit_Temporary(self, node: itir.Temporary, **kwargs: object) -> itir.Temporary:
        if node.domain is None:
            return node
        new_domain = _clamp_domain_expr(node.domain, self.valid_starts, self.min_literal_start)
        if new_domain is None:
            return node
        return itir.Temporary(id=node.id, domain=new_domain, dtype=node.dtype)

    def visit_SetAt(self, node: itir.SetAt, **kwargs: object) -> itir.SetAt:
        new_domain = _clamp_domain_expr(node.domain, self.valid_starts, self.min_literal_start)
        # Clamp every as_fieldop domain in the producer expression (the gather/reduce chain) too,
        # so the whole producer column-range is raised consistently with the temp/SetAt domain.
        new_expr = _ClampAsFieldopDomains(self.valid_starts, self.min_literal_start).visit(
            node.expr
        )
        if new_domain is None and new_expr is node.expr:
            return node
        return itir.SetAt(
            target=node.target,
            domain=new_domain if new_domain is not None else node.domain,
            expr=new_expr,
        )

    @classmethod
    def apply(cls, program: itir.Program) -> itir.Program:
        valid_starts, min_literal_start = _program_valid_starts(program)
        if not valid_starts:
            return program
        return cls(valid_starts, min_literal_start).visit(program)


class _ClampAsFieldopDomains(eve.PreserveLocationVisitor, eve.NodeTranslator):
    def __init__(self, valid_starts: set[str], min_literal_start: int | None):
        self.valid_starts = valid_starts
        self.min_literal_start = min_literal_start

    def visit_FunCall(self, node: itir.FunCall, **kwargs: object) -> itir.Node:
        node = self.generic_visit(node, **kwargs)
        if cpm.is_applied_as_fieldop(node) and len(node.fun.args) > 1:  # type: ignore[attr-defined]
            new_domain = _clamp_domain_expr(
                node.fun.args[1], self.valid_starts, self.min_literal_start  # type: ignore[attr-defined]
            )
            if new_domain is not None:
                return itir.FunCall(
                    fun=itir.FunCall(fun=node.fun.fun, args=[node.fun.args[0], new_domain]),  # type: ignore[attr-defined]
                    args=node.args,
                )
        return node


def _applied_fieldop_is_below_start(
    node: itir.Node, valid_starts: set[str], min_literal_start: int | None
) -> bool:
    if not (cpm.is_applied_as_fieldop(node) and len(node.fun.args) > 1):  # type: ignore[attr-defined]
        return False
    start = _vertical_start_expr(node.fun.args[1])  # type: ignore[attr-defined]
    return start is not None and _is_below_start(start, valid_starts, min_literal_start)


def overextension_is_clampable(program: itir.Program) -> bool:
    """True if every below-start over-extension in the (pre-`global_tmps`) inferred `program` is the
    clampable shape: each below-start ``as_fieldop`` flows *only* into a vertical ``Koff[neg]``-shift
    ``as_fieldop`` that is lexically inside an ``if_`` (the ``concat_where`` boundary guard). Such an
    over-extension is dropped value-preservingly by the post-`global_tmps` start clamp; any other
    shape (an unguarded or non-``Koff[neg]`` use of a below-start producer) is *not* safe to clamp,
    so the caller reverts that program to the unfused lowering instead.
    """
    valid_starts, min_literal_start = _program_valid_starts(program)
    if not valid_starts:
        return True

    def is_below(n: itir.Node) -> bool:
        return _applied_fieldop_is_below_start(n, valid_starts, min_literal_start)

    # All below-start as_fieldop node-ids that must be accounted for.
    below_ids: set[int] = {
        id(n) for n in program.pre_walk_values() if isinstance(n, itir.Node) and is_below(n)
    }
    if not below_ids:
        return True

    # The node-ids covered by a *valid root*: a below-start as_fieldop that is the single arg of an
    # `if_`-guarded Koff[neg]-shift as_fieldop. The whole below-start chain under such a root (every
    # node in its subtree) is clamp-safe.
    covered: set[int] = set()
    for node in program.pre_walk_values():
        if not isinstance(node, itir.Node):
            continue
        if not cpm.is_call_to(node, "if_") or len(node.args) != 3:  # type: ignore[attr-defined]
            continue
        for branch in node.args[1:]:  # type: ignore[attr-defined]
            if not isinstance(branch, itir.Node):
                continue
            for sub in branch.pre_walk_values():
                if not (isinstance(sub, itir.Node) and cpm.is_applied_as_fieldop(sub)):
                    continue
                if not _stencil_is_single_negative_koff_shift(sub.fun.args[0]):  # type: ignore[attr-defined]
                    continue
                if len(sub.args) != 1:  # type: ignore[attr-defined]
                    continue
                root = sub.args[0]  # type: ignore[attr-defined]
                if isinstance(root, itir.Node) and is_below(root):
                    for r in root.pre_walk_values():
                        if isinstance(r, itir.Node):
                            covered.add(id(r))

    # Clampable iff every below-start as_fieldop is covered by some valid guarded root.
    return below_ids <= covered


def _stencil_is_single_negative_koff_shift(stencil: itir.Node) -> bool:
    """True if `stencil` is ``λ(x) → deref(shift(Koff, neg)(x))`` — a single negative vertical shift."""
    if not (isinstance(stencil, itir.Lambda) and len(stencil.params) == 1):
        return False
    body = stencil.expr
    if not cpm.is_call_to(body, "deref"):
        return False
    return _negative_koff_shift_of(body.args[0], stencil.params[0].id)  # type: ignore[attr-defined]


def _is_pointwise_as_fieldop(node: itir.Node) -> bool:
    """True for an applied as_fieldop whose stencil only accesses its inputs at the center."""
    if not cpm.is_applied_as_fieldop(node):
        return False
    stencil = node.fun.args[0]  # type: ignore[attr-defined]
    if not isinstance(stencil, itir.Lambda):
        return False
    try:
        shifts = trace_shifts.trace_stencil(stencil, num_args=len(node.args))  # type: ignore[attr-defined]
    except Exception:
        return False
    return all(len(seq) == 0 for arg_shifts in shifts for seq in arg_shifts)


class MoveDataflowIntoConcatWhere(eve.PreserveLocationVisitor, eve.NodeTranslator):
    def __init__(self, full_range: tuple[int, int] | None):
        self.full_range = full_range

    def _is_restricted(self, node: itir.Node) -> bool:
        kr = _vertical_range(node)
        return (
            kr is not None
            and self.full_range is not None
            and (kr[0] > self.full_range[0] or kr[1] < self.full_range[1])
        )

    def visit_FunCall(self, node: itir.FunCall, **kwargs):
        node = self.generic_visit(node, **kwargs)
        if isinstance(node.fun, itir.Lambda) and len(node.fun.params) == len(node.args):
            eligible = [
                self._is_restricted(arg) and _is_pointwise_as_fieldop(arg) for arg in node.args
            ]
            if any(eligible):
                inlined = inline_lambdas.inline_lambda(node, eligible_params=eligible)
                return self.visit(inlined, **kwargs) if inlined is not node else inlined
        return node

    @classmethod
    def apply(cls, program: itir.Program) -> itir.Program:
        starts: list[int] = []
        stops: list[int] = []
        for n in program.pre_walk_values():
            if isinstance(n, itir.FunCall):
                kr = _vertical_range(n)
                if kr is not None:
                    starts.append(kr[0])
                    stops.append(kr[1])
        full = (min(starts), max(stops)) if starts else None
        return cls(full).visit(program)
