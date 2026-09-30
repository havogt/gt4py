# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional

from gt4py import eve
from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next import common, utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.iterator.transforms import fuse_as_fieldop, infer_domain
from gt4py.next.iterator.transforms.concat_where.transform_to_as_fieldop import (
    concat_where_to_as_fieldop,
)
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_specifications as ts


def _output_concat_wheres(program: itir.Program) -> set[int]:
    """
    Ids of the `concat_where` calls whose value is stored in a program output and whose condition
    only refers to scalars.

    Starting at the expression of each `SetAt`, the search descends into tuple elements, `let`
    bodies, the values bound to `let` symbols the body returns, and the branches of the found
    `concat_where` calls.
    """
    found: set[int] = set()
    non_scalar_params = {
        str(param.id) for param in program.params if not isinstance(param.type, ts.ScalarType)
    }

    def visit(expr: itir.Expr, bound: dict[str, itir.Expr]) -> None:
        if cpm.is_call_to(expr, "concat_where"):
            # the symbols of the condition become arguments of the selection's `as_fieldop`,
            # which the dace lowering only accepts as scalars
            if not any(
                str(ref.id) in non_scalar_params
                for ref in eve.walk_values(expr.args[0]).if_isinstance(itir.SymRef)
            ):
                found.add(id(expr))
            visit(expr.args[1], bound)
            visit(expr.args[2], bound)
        elif cpm.is_call_to(expr, "make_tuple"):
            for arg in expr.args:
                visit(arg, bound)
        elif cpm.is_let(expr):
            visit(
                expr.fun.expr,
                {**bound, **{str(p.id): arg for p, arg in zip(expr.fun.params, expr.args)}},
            )
        elif isinstance(expr, itir.SymRef) and str(expr.id) in bound:
            visit(bound[str(expr.id)], {k: v for k, v in bound.items() if k != str(expr.id)})

    for stmt in program.body:
        if isinstance(stmt, itir.SetAt):
            visit(stmt.expr, {})
    return found


class _TransformSelected(PreserveLocationVisitor, NodeTranslator):
    PRESERVED_ANNEX_ATTRS = ("type", "domain")

    def __init__(self, selected: set[int]) -> None:
        self.selected = selected

    def visit_FunCall(self, node: itir.FunCall) -> itir.Expr:
        is_selected = id(node) in self.selected
        node = self.generic_visit(node)
        if is_selected:
            domains = utils.flatten_nested_tuple(node.annex.domain)
            return concat_where_to_as_fieldop(node, domains[0].as_expr())
        return node


class _FuseIntoSelects(PreserveLocationVisitor, NodeTranslator):
    PRESERVED_ANNEX_ATTRS = ("domain",)

    def __init__(
        self, offset_provider_type: common.OffsetProviderType, uids: utils.IDGeneratorPool
    ) -> None:
        self.offset_provider_type = offset_provider_type
        self.uids = uids

    def visit_FunCall(self, node: itir.FunCall) -> itir.Expr:
        node = self.generic_visit(node)
        if (
            cpm.is_applied_as_fieldop(node)
            and isinstance(stencil := node.fun.args[0], itir.Lambda)
            and stencil.params
            and str(stencil.params[0].id).startswith("__tcw_pos")
        ):
            return fuse_as_fieldop.FuseAsFieldOp.apply(
                node,  # type: ignore[arg-type]  # `apply` is annotated for programs but accepts any node
                uids=self.uids,
                offset_provider_type=self.offset_provider_type,
                allow_undeclared_symbols=True,
                within_set_at_expr=True,
            )
        return node


def transform_output_to_select(
    program: itir.Program,
    *,
    offset_provider: common.OffsetProvider,
    symbolic_domain_sizes: Optional[dict[str, itir.Expr]],
    uids: utils.IDGeneratorPool,
) -> itir.Program:
    """
    Replace the `concat_where` calls that produce program outputs by a pointwise selection.

    Each such `concat_where` becomes an `as_fieldop` whose stencil selects between the two
    branches by position (see `concat_where_to_as_fieldop`), and the `as_fieldop` producers of its
    branches are fused into that stencil, so that a branch is only evaluated where it is taken.
    `concat_where` calls that produce intermediate values are left unchanged.

    Requires inferred domains and returns a program with inferred domains.
    """
    selected = _output_concat_wheres(program)
    if not selected:
        return program
    offset_provider_type = common.offset_provider_to_type(offset_provider)
    program = _TransformSelected(selected).visit(program)
    program = type_inference.SanitizeTypes().visit(program)
    # The conditions contain infinite bounds (e.g. `pos < ∞`) which must be folded away.
    program = ConstantFolding.apply(program)  # type: ignore[assignment]  # always an itir.Program
    program = type_inference.infer(program, offset_provider_type=offset_provider_type)
    program = _FuseIntoSelects(offset_provider_type, uids).visit(program)
    return infer_domain.infer_program(
        program, offset_provider=offset_provider, symbolic_domain_sizes=symbolic_domain_sizes
    )
