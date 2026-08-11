# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
from typing import TypeVar

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.eve.extended_typing import Self
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, domain_utils


PRG = TypeVar("PRG", bound=itir.Program | itir.Expr)


@dataclasses.dataclass
class _PruneEmptyConcatWhere(PreserveLocationVisitor, NodeTranslator):
    """
    Prune `concat_where` expression with one branch never being accessed.

    Whether a branch is ever selected is decided from the domains of the branches themselves: a
     branch is selected on its own domain restricted to the condition, or to the complement of the
     condition for the false branch.

    This pass requires domain inference to be executed before. In particular it relies on the
     condition being in the canonical form that domain inference already requires, i.e. bounded on
     exactly one side, as the complement of the condition is not defined otherwise.

    This pass the true and false branch values to be fields, not tuples of fields. Execute
     `gt4py.next.iterator.transforms.concat_where.expand_tuple_args` before.

    A branch that does not span all dimensions of the `concat_where` is neither recognized as
     never selected, since its domain has no range in the missing dimension, nor usable as a
     replacement for the entire expression, since that would drop a dimension. Execute
     `gt4py.next.iterator.transforms.concat_where.broadcast_branches` before to give every branch
     the dimensions of the `concat_where`.

    >>> from gt4py.next.iterator.ir_utils import domain_utils, ir_makers as im
    >>> from gt4py.next.iterator.transforms import concat_where, infer_domain
    >>> IDim = common.Dimension("IDim")
    >>> expr = im.concat_where(
    ...     im.domain(common.GridType.CARTESIAN, {IDim: (10, itir.InfinityLiteral.POSITIVE)}),
    ...     "a",
    ...     "b",
    ... )
    >>> expr, _ = infer_domain.infer_expr(
    ...     concat_where.canonicalize_domain_argument(expr),
    ...     domain_utils.SymbolicDomain.from_expr(
    ...         im.domain(common.GridType.CARTESIAN, {IDim: (0, 10)})
    ...     ),
    ...     offset_provider={},
    ... )
    >>> assert prune_empty_concat_where(expr) == im.ref("b")
    """

    PRESERVED_ANNEX_ATTRS = ("domain",)

    @classmethod
    def apply(cls: type[Self], node: PRG) -> PRG:
        return cls().visit(node)

    def visit_FunCall(self, node: itir.FunCall) -> itir.Expr:
        node = self.generic_visit(node)

        if cpm.is_call_to(node, "concat_where"):
            cond_expr, tb, fb = node.args

            if tb == fb:
                # note: as long as we visited the args we have a copy here, so no need to copy again
                tb.annex.domain = node.annex.domain
                return tb

            cond = domain_utils.SymbolicDomain.from_expr(cond_expr)
            branch_domains = [arg.annex.domain for arg in (tb, fb)]
            if not all(
                isinstance(domain, domain_utils.SymbolicDomain) for domain in branch_domains
            ):
                return node

            # a branch is implicitly broadcast to the dimensions it does not have itself, so the
            #  `concat_where` spans the dimensions of the condition and of both branches
            dims: dict[common.Dimension, None] = dict.fromkeys(
                [*cond.ranges, *(dim for domain in branch_domains for dim in domain.ranges)]
            )

            for is_true_branch in (True, False):
                pruned, kept = (tb, fb) if is_true_branch else (fb, tb)
                selected = domain_utils.concat_where_branch_domain(
                    domain_utils.promote_domain(pruned.annex.domain, dims), cond, is_true_branch
                )
                # the kept branch replaces the entire expression as is, so pruning to it would
                #  drop the dimensions it is implicitly broadcast to
                if selected.empty() and kept.annex.domain.ranges.keys() == dims.keys():
                    return kept

        return node


prune_empty_concat_where = _PruneEmptyConcatWhere.apply
