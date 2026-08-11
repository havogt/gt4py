# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Optional, TypeVar

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_info, type_specifications as ts


NODE = TypeVar("NODE", bound=itir.Node)


def _dims(type_: Optional[ts.TypeSpec]) -> Optional[list[common.Dimension]]:
    if isinstance(type_, (ts.FieldType, ts.ScalarType)):
        return type_info.extract_dims(type_)
    return None


class _BroadcastBranches(PreserveLocationVisitor, NodeTranslator):
    """
    Make the implicit broadcast of `concat_where` branches explicit.

    A `concat_where` broadcasts a branch to the dimensions that only occur in the condition or in
     the other branch, e.g.

    >>> from gt4py.next.iterator.ir_utils import ir_makers as im
    >>> from gt4py.next.type_system import type_specifications as ts
    >>> Vertex = common.Dimension("Vertex")
    >>> K = common.Dimension("K", kind=common.DimensionKind.VERTICAL)
    >>> float64 = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    >>> expr = im.concat_where(
    ...     im.domain(common.GridType.UNSTRUCTURED, {K: (itir.InfinityLiteral.NEGATIVE, 0)}),
    ...     im.ref("a", ts.FieldType(dims=[Vertex], dtype=float64)),
    ...     im.ref("b", ts.FieldType(dims=[Vertex, K], dtype=float64)),
    ... )
    >>> print(broadcast_branches(expr))
    concat_where(u⟨ Kᵥ: [-∞, 0[ ⟩, broadcast(a, {Vertexₕ, Kᵥ}), b)

    Domain inference restricts the domain of an expression to the dimensions of its type, so
     without the explicit `broadcast` the domain a branch is selected on is lost in exactly the
     dimensions that make it empty.
    """

    @classmethod
    def apply(
        cls, node: NODE, *, offset_provider_type: Optional[common.OffsetProviderType] = None
    ) -> NODE:
        node = type_inference.reinfer(node, offset_provider_type=offset_provider_type)
        return cls().visit(node)

    def visit_FunCall(self, node: itir.FunCall) -> itir.Expr:
        node = self.generic_visit(node)

        if cpm.is_call_to(node, "concat_where"):
            dims = _dims(node.type)
            if dims is None:
                return node
            cond, *branches = node.args
            new_branches = [
                im.call("broadcast")(
                    branch,
                    im.make_tuple(
                        *(itir.AxisLiteral(value=dim.value, kind=dim.kind) for dim in dims)
                    ),
                )
                if _dims(branch.type) not in (None, dims)
                else branch
                for branch in branches
            ]
            if new_branches != branches:
                return im.call(node.fun)(cond, *new_branches)

        return node


broadcast_branches = _BroadcastBranches.apply
