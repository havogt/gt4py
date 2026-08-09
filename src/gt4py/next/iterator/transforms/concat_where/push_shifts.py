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

where `shift_d` distributes the shift all the way down to the leaves::

    shift_d(as_fieldop(S)(x0, ..., xn)) == as_fieldop(S)(shift_d(x0), ..., shift_d(xn))

Sinking to the leaves is essential. Stopping at `as_fieldop(shift)(a)` leaves the
shift on the intermediate `a`, which is the very fusion barrier this pass removes;
the materialization would merely move from the `concat_where` result to `a`.

Translating the condition together with the expression makes the rewrite safe by
construction: the shifted branch is only evaluated where the original branch was
valid at the shifted position, so no out-of-branch access can be introduced. This
matters because a branch may be valid on a wider index range than it is selected
on -- reading one element past the end of the selected range can be perfectly in
bounds yet semantically meaningless.

The rewrite is only performed when it can be established, for every leaf the shift
would reach, whether that leaf varies along the shifted dimension. Sinking a shift
onto an argument that does not have the dimension makes domain inference fail, and
failing to sink one onto an argument that does have it silently changes the result,
so an unknown type makes the pass decline rather than guess.

Where it does apply it applies unconditionally, mirroring how pointwise producers
are already recomputed into every consumer (see
`fuse_as_fieldop._arg_inline_predicate`); the shift is the only reason this case
was treated differently. Should a program ever be hurt by that, the decision
belongs in a cost model rather than in an arbitrary limit here.

It must run

* after `concat_where.canonicalize_domain_argument`, so the condition is a plain
  domain,
* before `infer_domain.infer_program`, so the new branch domains are inferred, and
* before `concat_where.transform_to_as_fieldop`, which replaces the `concat_where`
  by a position dependent `if_` that a shift can no longer be pushed through.
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

    Only a type that is actually available yields a definite answer. Anything else
    is `UNKNOWN`, which makes the caller decline the rewrite: sinking a shift onto
    an argument that turns out not to have the dimension makes domain inference
    fail, and not sinking one onto an argument that does have it silently changes
    the result, so neither direction is safe to guess.
    """
    if isinstance(expr, itir.Literal):
        return _DimUse.WITHOUT
    type_ = expr.type
    if type_ is None or isinstance(type_, ts.DeferredType):
        return _DimUse.UNKNOWN
    constituents = list(type_info.primitive_constituents(type_))
    if all(isinstance(t, ts.ScalarType) for t in constituents):
        return _DimUse.WITHOUT
    fields = [t for t in constituents if isinstance(t, ts.FieldType)]
    if len(fields) != len(constituents):
        return _DimUse.UNKNOWN
    if all(dim not in t.dims for t in fields):
        return _DimUse.WITHOUT
    if all(dim in t.dims for t in fields):
        return _DimUse.WITH
    return _DimUse.UNKNOWN


def _sink_shift(shift_fun: itir.Expr, dim: common.Dimension, arg: itir.Expr) -> Optional[itir.Expr]:
    """Applies the shift `as_fieldop` `shift_fun` to `arg`, sinking it to the leaves.

    `as_fieldop(shift)(as_fieldop(S)(a0, ..., an))` becomes
    `as_fieldop(S)(shift(a0), ..., shift(an))`, so that no new shared intermediate is
    created. Arguments that do not have the shifted dimension are left alone.

    Returns `None` if it cannot be established for every leaf whether it varies
    along `dim`, in which case the rewrite must not be performed at all.
    """
    if cpm.is_applied_as_fieldop(arg) and isinstance(arg.fun.args[0], itir.Lambda):
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
    return itir.FunCall(fun=shift_fun, args=[arg], type=arg.type)


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
    PRESERVED_ANNEX_ATTRS = ("type",)

    @classmethod
    def apply(cls, node: itir.Program) -> itir.Program:
        return cls().visit(node, env={})

    def visit_Lambda(self, node: itir.Lambda, *, env: dict[str, itir.Expr]) -> itir.Lambda:
        # The parameters shadow anything of the same name bound further out.
        shadowed = {str(param.id) for param in node.params}
        inner_env = {name: expr for name, expr in env.items() if name not in shadowed}
        return itir.Lambda(
            params=node.params, expr=self.visit(node.expr, env=inner_env), type=node.type
        )

    def visit_FunCall(self, node: itir.FunCall, *, env: dict[str, itir.Expr]) -> itir.Expr:
        if cpm.is_let(node):
            # Record which parameters are bound to a `concat_where`, so that a shift
            #  applied to a reference can look through the binding. Parameters bound
            #  to anything else shadow an outer binding of the same name.
            new_args = [self.visit(arg, env=env) for arg in node.args]
            inner_env = dict(env)
            for param, arg in zip(node.fun.params, new_args, strict=True):
                if cpm.is_call_to(arg, "concat_where"):
                    inner_env[str(param.id)] = arg
                else:
                    inner_env.pop(str(param.id), None)
            new_body = self.visit(node.fun.expr, env=inner_env)
            return itir.FunCall(
                fun=itir.Lambda(params=node.fun.params, expr=new_body, type=node.fun.type),
                args=new_args,
                type=node.type,
            )

        # `type` must be carried along: the decision whether a shift may be sunk onto
        #  an argument is taken from its type, and a reconstruction that drops it would
        #  silently turn every case into `UNKNOWN`.
        node = itir.FunCall(
            fun=self.visit(node.fun, env=env),
            args=[self.visit(a, env=env) for a in node.args],
            type=node.type,
        )

        if not cpm.is_applied_as_fieldop(node) or len(node.args) != 1:
            return node
        shift_info = _shift_dim_and_distance(node.fun.args[0])
        if shift_info is None:
            return node
        dim, distance = shift_info
        if distance == 0:
            return node

        arg = node.args[0]
        if isinstance(arg, itir.SymRef) and str(arg.id) in env:
            arg = env[str(arg.id)]
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
            # Not provable that the rewrite is meaning preserving, leave it alone.
            return node

        # The branches may themselves be `concat_where`s.
        return self.visit(im.concat_where(new_cond, *new_branches), env=env)


push_shifts = PushShiftIntoConcatWhere.apply
