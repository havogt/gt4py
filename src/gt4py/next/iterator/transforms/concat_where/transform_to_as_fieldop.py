# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
from typing import Optional, Sequence

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next import common, utils
from gt4py.next.iterator import builtins, ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
    misc as ir_misc,
)
from gt4py.next.iterator.transforms import symbol_ref_utils
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_specifications as ts


def _in(pos: itir.Expr, dims: Sequence[common.Dimension], domain: itir.Expr) -> itir.Expr:
    """
    Given a position and a domain return an expression that evaluates to `True` if the position is inside the domain.

    The i-th element of `pos` is the index in `dims[i]`. Besides a domain expression, `domain`
    can be a comparison of a dimension with a value, e.g. `Kᵥ < 1`, or a union or intersection
    (`or_`, `and_`) of such expressions.

    pos = `{i, j, k}`, domain = `u⟨ Iₕ: [i0, i1[, Iₕ: [j0, j1[, Iₕ: [k0, k1[ ⟩`
    -> `((i0 <= i) & (i < i1)) & ((j0 <= j) & (j < j1)) & ((k0 <= k)l & (k < k1))`
    """
    if cpm.is_call_to(domain, ("and_", "or_")):
        return im.call(domain.fun.id)(*(_in(pos, dims, arg) for arg in domain.args))
    if cpm.is_call_to(domain, builtins.BINARY_MATH_COMPARISON_BUILTINS):
        return im.call(domain.fun.id)(
            *(
                im.tuple_get(dims.index(ir_misc.dim_from_axis_literal(arg)), pos)
                if isinstance(arg, itir.AxisLiteral)
                else arg
                for arg in domain.args
            )
        )
    ret = [
        im.and_(
            im.less_equal(v.start, im.tuple_get(dims.index(dim), pos)),
            im.less(im.tuple_get(dims.index(dim), pos), v.stop),
        )
        for dim, v in domain_utils.SymbolicDomain.from_expr(domain).ranges.items()
    ]
    return functools.reduce(im.and_, ret)


def concat_where_to_as_fieldop(
    node: itir.FunCall, domain: Optional[itir.Expr] = None
) -> itir.FunCall:
    """
    Transform a `concat_where` call into an `as_fieldop` selecting the branch by position.

    The position is passed to the `as_fieldop` as one `index` field per dimension, hence when
    the result is shifted the condition is evaluated at the shifted position.
    """
    assert cpm.is_call_to(node, "concat_where")
    cond, true_branch, false_branch = node.args
    assert isinstance(cond.type, ts.DomainType)
    dims = cond.type.dims
    position_params = [f"__tcw_pos_{dim.value}" for dim in dims]
    refs = symbol_ref_utils.collect_symbol_refs(cond)

    return im.as_fieldop(
        im.lambda_(*position_params, "__tcw_arg0", "__tcw_arg1", *refs)(
            im.let(*zip(refs, map(im.deref, refs), strict=True))(
                im.if_(
                    _in(im.make_tuple(*map(im.deref, position_params)), dims, cond),
                    im.deref("__tcw_arg0"),
                    im.deref("__tcw_arg1"),
                )
            )
        ),
        domain,
    )(*(im.index(dim) for dim in dims), true_branch, false_branch, *refs)


class _TransformToAsFieldop(PreserveLocationVisitor, NodeTranslator):
    PRESERVED_ANNEX_ATTRS = (
        "type",
        "domain",
    )

    @classmethod
    def apply(cls, node: itir.Node):
        """
        Transform `concat_where` expressions into equivalent `as_fieldop` expressions.

        Note that (backward) domain inference may not be executed after this pass as it can not
        correctly infer the accessed domains when the value selection is represented as an `if_`
        inside the `as_fieldop.
        """
        node = cls().visit(node)
        node = type_inference.SanitizeTypes().visit(node)
        return node

    def visit_FunCall(self, node: itir.FunCall) -> itir.FunCall:
        node = self.generic_visit(node)
        if cpm.is_call_to(node, "concat_where"):
            domains: tuple[domain_utils.SymbolicDomain, ...] = utils.flatten_nested_tuple(
                node.annex.domain
            )
            assert all(domain == domains[0] for domain in domains), (
                "At this point all `concat_where` arguments should be posed on the same domain."
            )
            assert isinstance(domains[0], domain_utils.SymbolicDomain)
            return concat_where_to_as_fieldop(node, domains[0].as_expr())

        return node


transform_to_as_fieldop = _TransformToAsFieldop.apply
