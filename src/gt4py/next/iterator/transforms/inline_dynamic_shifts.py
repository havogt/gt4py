# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses

import gt4py.next.iterator.ir_utils.common_pattern_matcher as cpm
from gt4py import eve
from gt4py.next import common, utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.iterator.transforms import (
    collapse_tuple,
    fuse_as_fieldop,
    inline_lambdas,
    trace_shifts,
)
from gt4py.next.iterator.transforms.symbol_ref_utils import collect_symbol_refs
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_specifications as ts


def _dynamic_shift_args(node: itir.Expr) -> list[bool] | None:
    if not cpm.is_applied_as_fieldop(node):
        return None
    params_shifts = trace_shifts.trace_stencil(
        node.fun.args[0],
        num_args=len(node.args),
        save_to_annex=True,
    )
    dynamic_shifts = [
        any(trace_shifts.Sentinel.VALUE in shifts for shifts in param_shifts)
        for param_shifts in params_shifts
    ]
    return dynamic_shifts


def _is_scalar(node: itir.Expr) -> bool:
    type_inference.reinfer(node)
    return isinstance(node.type, ts.ScalarType)


def _is_symbol_element(node: itir.Expr) -> bool:
    if cpm.is_call_to(node, "tuple_get"):
        return _is_symbol_element(node.args[1])
    return isinstance(node, itir.SymRef)


def _dynamically_shifted_refs(node: itir.Expr) -> set[str]:
    refs: set[str] = set()
    for fieldop in node.pre_walk_values().filter(cpm.is_applied_as_fieldop):
        if dynamic_shift_args := _dynamic_shift_args(fieldop):
            for inp, is_dynamic_shift_arg in zip(fieldop.args, dynamic_shift_args, strict=True):
                if is_dynamic_shift_arg:
                    refs.update(collect_symbol_refs(inp))
    return refs


@dataclasses.dataclass
class InlineDynamicShifts(eve.NodeTranslator, eve.VisitorWithSymbolTableTrait):
    offset_provider_type: common.OffsetProviderType
    uids: utils.IDGeneratorPool

    @classmethod
    def apply(
        cls,
        node: itir.Program,
        offset_provider_type: common.OffsetProviderType,
        uids: utils.IDGeneratorPool,
    ):
        return cls(offset_provider_type=offset_provider_type, uids=uids).visit(node)

    def _collapse_tuple_get(self, node: itir.Expr) -> itir.Expr:
        result = collapse_tuple.CollapseTuple.apply(
            node,
            enabled_transformations=collapse_tuple.CollapseTuple.Transformation.COLLAPSE_TUPLE_GET_MAKE_TUPLE
            | collapse_tuple.CollapseTuple.Transformation.PROPAGATE_TUPLE_GET,
            within_stencil=False,
            allow_undeclared_symbols=True,
            uids=self.uids,
        )
        assert isinstance(result, itir.Expr)
        return result

    def _inline_let(self, node: itir.FunCall, eligible_params: list[bool]) -> itir.Expr:
        # also remove the `tuple_get`s on the inlined producers at non-shifted uses
        return self._collapse_tuple_get(
            inline_lambdas.inline_lambda(node, eligible_params=eligible_params)
        )

    def _simplify_producer(self, node: itir.Expr) -> itir.Expr:
        if cpm.is_let(node):
            return self._inline_let(node, [True] * len(node.fun.params))
        if cpm.is_call_to(node, "tuple_get"):
            return self._collapse_tuple_get(node)
        return node

    def visit_FunCall(self, node: itir.FunCall, **kwargs):
        node = self.generic_visit(node, **kwargs)

        # Inline let-bound producers of dynamically shifted arguments, also when shared between
        # multiple consumers or read through `tuple_get`, and fuse them into the consumers.
        if cpm.is_let(node):
            shifted_refs = _dynamically_shifted_refs(node.fun.expr)
            inline_let_params = [param.id in shifted_refs for param in node.fun.params]
            if any(inline_let_params):
                return self.visit(self._inline_let(node, inline_let_params), **kwargs)

        # Fusing one producer can expose another one behind it (e.g. a chain of shifts split
        # across multiple `as_fieldop`s), so repeat until every dynamically shifted argument is a
        # symbol (or an element of one), whose producer is inlined at its let, an `index` field or
        # a scalar.
        # This terminates: each iteration either replaces an `as_fieldop`, `if_`, `broadcast` or
        # `concat_where` argument by strict subterms of itself and `index` fields, inlines a `let`
        # argument, propagates a `tuple_get` argument into its tuple expression, or drops a
        # tuple-of-literals argument entirely.
        expr: itir.Expr = node
        while dynamic_shift_args := _dynamic_shift_args(expr):
            assert isinstance(expr, itir.FunCall) and len(expr.fun.args) in [1, 2]  # type: ignore[attr-defined]  # ensured by is_applied_as_fieldop in _dynamic_shift_args
            fuse_args = [
                dynamic_shift_arg
                and not _is_symbol_element(inp)
                and not cpm.is_call_to(inp, "index")
                and not _is_scalar(inp)
                for inp, dynamic_shift_arg in zip(expr.args, dynamic_shift_args, strict=True)
            ]
            if not any(fuse_args):
                break
            simplified_args = [
                self._simplify_producer(inp) if fuse else inp
                for inp, fuse in zip(expr.args, fuse_args, strict=True)
            ]
            if simplified_args != expr.args:
                expr = im.call(expr.fun)(*simplified_args)
                continue
            expr = fuse_as_fieldop.fuse_as_fieldop(
                expr,
                fuse_args,
                uids=self.uids,
                offset_provider_type=self.offset_provider_type,
                enable_cse=True,
            )

        return expr
