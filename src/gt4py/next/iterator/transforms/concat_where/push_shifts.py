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

It never substitutes a binding itself -- doing so by hand risks capturing the free
variables of the substituted expression. Instead, a `let` whose bound value is a
`concat_where` that the body reads through a shift is inlined with the project's
capture correct `inline_lambda`, restricted to exactly those parameters. This does
duplicate the `concat_where` when the binding has several uses, which is the normal
case here (`(H[k] - H[k+1]) * c` reads `H` twice). The duplication is intended: it is
what exposes the `concat_where` to the rewrite, and the duplicate leaves are folded
back together downstream by `fuse_as_fieldop`, which always inlines a single argument
`as_fieldop`, and by CSE.

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


def _reads_through_shift(body: itir.Expr, param: eve.concepts.SymbolName) -> bool:
    """Whether `body` applies a non zero shift directly to `param`."""
    for call in body.walk_values().if_isinstance(itir.FunCall):
        if not (_is_plain_as_fieldop(call) and len(call.args) == 1):
            continue
        assert isinstance(call.fun, itir.FunCall)
        shift_info = _shift_dim_and_distance(call.fun.args[0])
        if shift_info is not None and shift_info[1] != 0 and cpm.is_ref_to(call.args[0], param):
            return True
    return False


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


@dataclasses.dataclass
class PushShiftIntoConcatWhere(eve.PreserveLocationVisitor, eve.NodeTranslator):
    @classmethod
    def apply(cls, node: itir.Program) -> itir.Program:
        return cls().visit(node)

    def visit_FunCall(self, node: itir.FunCall) -> itir.Expr:
        node = self.generic_visit(node)

        if cpm.is_let(node):
            # Expose a `concat_where` that is only read through a shift, using the
            #  capture correct inliner rather than substituting here.
            eligible = [
                cpm.is_call_to(arg, "concat_where")
                and _reads_through_shift(node.fun.expr, param.id)
                for param, arg in zip(node.fun.params, node.args, strict=True)
            ]
            if any(eligible):
                return self.visit(inline_lambdas.inline_lambda(node, eligible_params=eligible))
            return node

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

        cond, true_branch, false_branch = arg.args
        sym_domain = domain_utils.SymbolicDomain.from_expr(cond)
        # A shift orthogonal to the condition leaves the condition untouched.
        new_cond = _translate_cond(cond, dim, distance) if dim in sym_domain.ranges else cond

        new_branches = [
            _shifted_branch(node.fun, dim, branch) for branch in (true_branch, false_branch)
        ]
        if any(branch is None for branch in new_branches):
            return node

        # The branches may themselves be `concat_where`s.
        return self.visit(im.concat_where(new_cond, *new_branches))


push_shifts = PushShiftIntoConcatWhere.apply
