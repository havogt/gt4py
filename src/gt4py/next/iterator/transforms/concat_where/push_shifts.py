# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Distribute a shift over a `concat_where`.

A shifted read of a `concat_where` result forces that result to be materialized:
neither the `as_fieldop` fusion of the gtfn pipeline nor the map fusion of the dace
pipeline can fuse a producer into a consumer that reads it at an offset, and the
`concat_where` blocks the usual escape of moving the shift onto the producer's own
inputs. This pass moves it::

    as_fieldop(lambda it: deref(shift(K, d)(it)))(concat_where(u<K: [lo, hi[>, a, b))
      ->
    concat_where(u<K: [lo - d, hi - d[>, shift_d(a), shift_d(b))

where `shift_d` distributes the shift down to the leaves::

    shift_d(as_fieldop(S)(x0, ..., xn)) == as_fieldop(S)(shift_d(x0), ..., shift_d(xn))

Translating the condition together with the expression is what makes the rewrite
safe: the shifted branch is only selected where the original branch was selected at
the shifted position. This matters because a branch may be readable on a wider index
range than it is selected on -- reading one element past the end of the selected
range can be perfectly in bounds and still meaningless.

Sinking to the leaves is essential rather than an optimization. Stopping at
`as_fieldop(shift)(a)` leaves the shift on the intermediate `a`, which is the very
fusion barrier this pass exists to remove, and merely moves the materialization.

The pass never guesses. It fires only when the shape is recognized syntactically and
every leaf the shift would reach can be classified from an already computed `type`;
anything else declines.

A `let` whose bound value is a `concat_where` read through a shift is handled by
adding one sibling binding per distinct shift distance rather than by inlining the
bound value into its use sites::

    let a = cw in ... a[+1] ... a[+1] ... a ...
      ->
    let a = cw, a_p1 = concat_where(c', shift_1(t), shift_1(f)) in ... a_p1 ... a ...

so the number of copies of `cw` is the number of distinct offsets it is read at, not
the number of uses, and the unshifted uses keep sharing the original binding. The
copies are never folded back together: the dace pipeline
(`apply_fieldview_transforms`) runs neither `fuse_as_fieldop` nor CSE, so whatever
this pass duplicates stays duplicated all the way to the backend.

A new binding that is read only once is then inlined at that read, which costs nothing
and keeps the shifted `concat_where` in the same scope as the expression consuming it.

Substituting a binding by hand risks capturing the free variables of the substituted
expression. The new binding is therefore placed as a sibling of the original -- the
same scope its value was already written in -- under a name that is fresh in the whole
`let`; moving it inward afterwards is left to the project's capture correct
`inline_lambda`; and a parameter that the body rebinds anywhere is skipped rather than
substituted under a binder.

Sinking into a `scan` is not merely unprofitable but wrong -- the fold would run over
a different range -- so a `scan` stencil declines rather than being wrapped.

It must run

* after `concat_where.canonicalize_domain_argument`, so the condition is a plain
  domain with a single range,
* before `infer_domain.infer_program`, so the new branch domains are inferred, and
* before `concat_where.transform_to_as_fieldop`, which replaces the `concat_where`
  by a position dependent `if_` that a shift can no longer be pushed through.

Translating a condition can leave a branch selected on an empty region. Deciding that
requires the inferred domain of the `concat_where`, which does not exist yet at this
position, so it is deliberately left to `prune_empty_concat_where` downstream.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Optional

from gt4py import eve
from gt4py.eve import utils as eve_utils
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
    misc as ir_misc,
)
from gt4py.next.iterator.transforms import inline_lambdas
from gt4py.next.type_system import type_info, type_specifications as ts


def _shift_dim_and_distance(stencil: itir.Expr) -> Optional[tuple[common.Dimension, int]]:
    """If `stencil` is `lambda p: deref(shift(K, d)(p))` return `(K, d)`, else `None`."""
    if not isinstance(stencil, itir.Lambda) or len(stencil.params) != 1:
        return None
    body = stencil.expr
    if not cpm.is_call_to(body, "deref"):
        return None
    shifted = body.args[0]
    if not cpm.is_applied_shift(shifted):
        return None
    if not cpm.is_ref_to(shifted.args[0], stencil.params[0].id):
        return None
    offsets = shifted.fun.args
    if len(offsets) != 2:
        return None
    off, val = offsets
    if not isinstance(off, itir.CartesianOffset):
        return None
    if off.domain != off.codomain:
        return None
    if not (isinstance(val, itir.OffsetLiteral) and isinstance(val.value, int)):
        return None
    return ir_misc.dim_from_axis_literal(off.domain), val.value


def _translate_bound(bound: itir.Expr, distance: int) -> itir.Expr:
    if bound in (itir.InfinityLiteral.POSITIVE, itir.InfinityLiteral.NEGATIVE):
        return bound
    return im.plus(bound, -distance)


def _translate_cond(cond: itir.Expr, shift_dim: common.Dimension, distance: int) -> itir.Expr:
    """Translate the range of `shift_dim` in `cond` by `-distance`."""
    sym_domain = domain_utils.SymbolicDomain.from_expr(cond)
    new_ranges = dict(sym_domain.ranges)
    rng = new_ranges[shift_dim]
    new_ranges[shift_dim] = domain_utils.SymbolicRange(
        _translate_bound(rng.start, distance), _translate_bound(rng.stop, distance)
    )
    return domain_utils.SymbolicDomain(sym_domain.grid_type, new_ranges).as_expr()


class _DimUse(enum.Enum):
    """Whether an expression varies along a dimension."""

    WITHOUT = enum.auto()
    WITH = enum.auto()
    UNKNOWN = enum.auto()


def _dim_use(expr: itir.Expr, dim: common.Dimension) -> _DimUse:
    """Classifies whether `expr` varies along `dim`.

    Only an available type yields a definite answer. Anything else is `UNKNOWN`,
    which makes the caller decline: sinking a shift onto an argument that does not
    have the dimension makes domain inference fail, and not sinking one onto an
    argument that does have it silently changes the result, so neither direction is
    safe to guess. Tuples and lists are `UNKNOWN` too, since a shifted tuple or list
    iterator cannot be dereferenced componentwise here.
    """
    if isinstance(expr, itir.Literal):
        return _DimUse.WITHOUT
    type_ = expr.type
    if type_ is None or isinstance(type_, ts.DeferredType):
        return _DimUse.UNKNOWN
    if isinstance(type_, ts.TupleType):
        return _DimUse.UNKNOWN
    constituents = list(type_info.primitive_constituents(type_))
    if len(constituents) != 1:
        return _DimUse.UNKNOWN
    (constituent,) = constituents
    if isinstance(constituent, ts.ScalarType):
        return _DimUse.WITHOUT
    if not isinstance(constituent, ts.FieldType):
        return _DimUse.UNKNOWN
    if isinstance(constituent.dtype, ts.ListType):
        return _DimUse.UNKNOWN
    return _DimUse.WITH if dim in constituent.dims else _DimUse.WITHOUT


def _shifted_reads(
    body: itir.Expr, param: eve.concepts.SymbolName
) -> dict[tuple[common.Dimension, int], itir.Expr]:
    """Maps each distinct non zero shift `body` applies directly to `param` to its `as_fieldop`.

    Several reads at the same distance share one entry, which is what makes the number
    of copies this pass creates proportional to the number of distinct offsets.
    """
    reads: dict[tuple[common.Dimension, int], itir.Expr] = {}
    for call in body.walk_values().if_isinstance(itir.FunCall):
        if not (_is_plain_as_fieldop(call) and len(call.args) == 1):
            continue
        assert isinstance(call.fun, itir.FunCall)
        shift_info = _shift_dim_and_distance(call.fun.args[0])
        if shift_info is not None and shift_info[1] != 0 and cpm.is_ref_to(call.args[0], param):
            reads.setdefault(shift_info, call.fun)
    return reads


def _is_rebound(body: itir.Expr, param: eve.concepts.SymbolName) -> bool:
    """Whether `body` binds `param` again somewhere, shadowing the outer binding."""
    return any(sym.id == param for sym in body.walk_values().if_isinstance(itir.Sym))


def _bound_names(node: itir.Node) -> set[str]:
    return {str(sym.id) for sym in node.walk_values().if_isinstance(itir.Sym, itir.SymRef)}


@dataclasses.dataclass(frozen=True)
class _ReplaceShiftedReads(eve.PreserveLocationVisitor, eve.NodeTranslator):
    """Replaces `as_fieldop(shift(K, d))(param)` by a reference to a new binding.

    Sound only because the caller has established that `param` is not rebound in the
    visited expression, so every occurrence refers to the same binding.
    """

    param: eve.concepts.SymbolName
    replacements: dict[tuple[common.Dimension, int], itir.Sym]

    def visit_FunCall(self, node: itir.FunCall) -> itir.Expr:
        if _is_plain_as_fieldop(node) and len(node.args) == 1:
            assert isinstance(node.fun, itir.FunCall)
            shift_info = _shift_dim_and_distance(node.fun.args[0])
            if (
                shift_info is not None
                and cpm.is_ref_to(node.args[0], self.param)
                and shift_info in self.replacements
            ):
                return im.ref(self.replacements[shift_info].id, node.type)
        return self.generic_visit(node)


def _is_plain_as_fieldop(expr: itir.Expr) -> bool:
    """An `as_fieldop` applied to a stencil alone, i.e. without an explicit domain."""
    return cpm.is_applied_as_fieldop(expr) and len(expr.fun.args) == 1


def _sink_shift(shift_fun: itir.Expr, dim: common.Dimension, arg: itir.Expr) -> Optional[itir.Expr]:
    """Applies the shift `as_fieldop` `shift_fun` to `arg`, sinking it to the leaves.

    Returns `None` if the shift cannot be sunk soundly, in which case the whole
    rewrite must be abandoned. Leaving the shift on a compound intermediate counts as
    a failure rather than a fallback: that is the shape the pass exists to remove,
    and it measures worse than not rewriting at all.
    """
    if cpm.is_applied_as_fieldop(arg):
        if not _is_plain_as_fieldop(arg):
            return None  # an explicit domain argument would be left stale
        if cpm.is_call_to(arg.fun.args[0], "scan"):
            return None  # sinking would run the fold over a different range
        new_args = []
        for a in arg.args:
            match _dim_use(a, dim):
                case _DimUse.WITHOUT:
                    new_args.append(a)
                case _DimUse.WITH:
                    if (new_arg := _sink_shift(shift_fun, dim, a)) is None:
                        return None
                    new_args.append(new_arg)
                case _DimUse.UNKNOWN:
                    return None
        return itir.FunCall(fun=arg.fun, args=new_args, type=arg.type)

    # Terminal positions: a name, a literal, or a nested `concat_where` that the
    #  visitor pushes into on re-entry.
    if isinstance(arg, (itir.SymRef, itir.Literal)) or cpm.is_call_to(arg, "concat_where"):
        return itir.FunCall(fun=shift_fun, args=[arg], type=arg.type)
    return None


def _shifted_branch(
    shift_fun: itir.Expr, dim: common.Dimension, branch: itir.Expr
) -> Optional[itir.Expr]:
    match _dim_use(branch, dim):
        case _DimUse.WITHOUT:
            return branch
        case _DimUse.WITH:
            return _sink_shift(shift_fun, dim, branch)
        case _:
            return None


def _node_count(node: itir.Node) -> int:
    return sum(1 for _ in node.pre_walk_values().if_isinstance(itir.Node))


#: Heuristic backstop: a `let` may not grow by more than this factor.
#:
#: One binding per distinct offset already bounds the copies structurally, so this is
#: not what makes the rewrite bounded -- it is what makes the bound hold even for a
#: shape the pass does not model. That matters because nothing downstream shares
#: duplicated subtrees back together on the dace path.
#:
#: The value has to clear what a legitimate rewrite costs. A bound `concat_where` can
#: be almost the whole `let`, in which case `n` distinct offsets legitimately reach
#: `n + 1` times the original size; the dycore reads at `+-1`, so up to three times.
#: The observed maximum in `compute_perturbed_quantities_and_interpolation` is 1.98x
#: for a single offset, which is close enough to that ceiling to be worth stating.
_MAX_LET_GROWTH_FACTOR = 4


@dataclasses.dataclass
class PushShiftIntoConcatWhere(eve.PreserveLocationVisitor, eve.NodeTranslator):
    _uids: eve_utils.SequentialIDGenerator = dataclasses.field(
        init=False,
        repr=False,
        default_factory=lambda: eve_utils.SequentialIDGenerator(prefix="_psh"),
    )

    @classmethod
    def apply(cls, node: itir.Program) -> itir.Program:
        return cls().visit(node)

    def _fresh_sym(self, taken: set[str], type_: Optional[ts.TypeSpec]) -> itir.Sym:
        while (name := next(self._uids)) in taken:
            pass
        return im.sym(name, type_)

    def _push_into_concat_where(
        self, shift_fun: itir.Expr, dim: common.Dimension, distance: int, concat_where: itir.FunCall
    ) -> Optional[itir.Expr]:
        """Distributes the shift `as_fieldop` `shift_fun` over `concat_where`, or declines."""
        cond, true_branch, false_branch = concat_where.args
        sym_domain = domain_utils.SymbolicDomain.from_expr(cond)
        # A shift orthogonal to the condition leaves the condition untouched.
        new_cond = _translate_cond(cond, dim, distance) if dim in sym_domain.ranges else cond

        new_branches = [
            _shifted_branch(shift_fun, dim, branch) for branch in (true_branch, false_branch)
        ]
        if any(branch is None for branch in new_branches):
            return None

        # The branches may themselves be `concat_where`s.
        return self.visit(im.concat_where(new_cond, *new_branches))

    def _bind_shifted_reads(self, node: itir.FunCall) -> itir.Expr:
        """Binds every distinct shifted read of a `concat_where` parameter to a new sibling.

        Nothing is committed before the rewrite has succeeded, so a parameter whose
        shift cannot be sunk leaves no duplicated material behind.
        """
        assert isinstance(node.fun, itir.Lambda)
        if not any(cpm.is_call_to(arg, "concat_where") for arg in node.args):
            return node

        body = node.fun.expr
        taken = _bound_names(node)
        new_params, new_args = list(node.fun.params), list(node.args)
        rewritten: set[eve.concepts.SymbolName] = set()

        for param, arg in zip(node.fun.params, node.args, strict=True):
            if not cpm.is_call_to(arg, "concat_where"):
                continue
            # Substituting under a binder that shadows `param` would rewrite reads of a
            #  different value; such a body is left alone rather than analyzed.
            if _is_rebound(body, param.id):
                continue
            replacements: dict[tuple[common.Dimension, int], itir.Sym] = {}
            for (dim, distance), shift_fun in _shifted_reads(body, param.id).items():
                shifted = self._push_into_concat_where(shift_fun, dim, distance, arg)
                if shifted is None:
                    continue
                sym = self._fresh_sym(taken, arg.type)
                taken.add(str(sym.id))
                replacements[(dim, distance)] = sym
                new_params.append(sym)
                new_args.append(shifted)
            if replacements:
                body = _ReplaceShiftedReads(param.id, replacements).visit(body)
                rewritten.add(param.id)

        if (added := len(new_params) - len(node.fun.params)) == 0:
            return node

        new_node = itir.FunCall(
            fun=itir.Lambda(params=new_params, expr=body), args=new_args, type=node.type
        )
        # A new binding read only once is placed at that read instead, so the shifted
        #  `concat_where` ends up in the same scope as the expression consuming it, and a
        #  rewritten parameter whose last read was a shift is dropped. `inline_lambda`
        #  renames the binders it moves under; substituting here would capture.
        new_node = inline_lambdas.inline_lambda(
            new_node,
            opcount_preserving=True,
            eligible_params=[param.id in rewritten for param in node.fun.params] + [True] * added,
        )
        if _node_count(new_node) > _MAX_LET_GROWTH_FACTOR * _node_count(node):
            return node
        return new_node

    def visit_FunCall(self, node: itir.FunCall) -> itir.Expr:
        node = self.generic_visit(node)

        if cpm.is_let(node):
            return self._bind_shifted_reads(node)

        if not _is_plain_as_fieldop(node) or len(node.args) != 1:
            return node
        assert isinstance(node.fun, itir.FunCall)  # `_is_plain_as_fieldop`
        shift_info = _shift_dim_and_distance(node.fun.args[0])
        if shift_info is None:
            return node
        dim, distance = shift_info
        if distance == 0:
            return node

        # Deliberately not looking through a binding, see the module docstring.
        arg = node.args[0]
        if not cpm.is_call_to(arg, "concat_where"):
            return node

        return self._push_into_concat_where(node.fun, dim, distance, arg) or node


push_shifts = PushShiftIntoConcatWhere.apply
