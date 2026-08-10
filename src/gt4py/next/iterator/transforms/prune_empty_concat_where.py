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
from gt4py.eve.extended_typing import Container, Self
from gt4py.next import common
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, domain_utils


def _selected_domain(
    cw_domain: domain_utils.SymbolicDomain,
    cond: domain_utils.SymbolicDomain,
    is_true_branch: bool,
) -> domain_utils.SymbolicDomain:
    """The part of `cw_domain` on which one branch of a `concat_where` is selected.

    Note:
        `domain_complement` requires each range to be infinite on exactly one side, so
        the complement has to be taken before promoting to the full set of dimensions,
        the same order `infer_domain._infer_concat_where` uses.
    """
    region = cond if is_true_branch else domain_utils.domain_complement(cond)
    region = domain_utils.promote_domain(region, cw_domain.ranges.keys())
    return domain_utils.domain_intersection(cw_domain, region)


def _filter_domain(
    domain: domain_utils.SymbolicDomain, dims: Container[common.Dimension]
) -> domain_utils.SymbolicDomain:
    return domain_utils.SymbolicDomain(
        grid_type=domain.grid_type,
        ranges={d: r for d, r in domain.ranges.items() if d in dims},
    )


PRG = TypeVar("PRG", bound=itir.Program | itir.Expr)


@dataclasses.dataclass
class _PruneEmptyConcatWhere(PreserveLocationVisitor, NodeTranslator):
    """
    Prune `concat_where` expression with one branch never being accessed.

    This pass requires domain inference to be executed before.

    This pass the true and false branch values to be fields, not tuples of fields. Execute
     `gt4py.next.iterator.transforms.concat_where.expand_tuple_args` before.

    >>> from gt4py.next.iterator.ir_utils import ir_makers as im
    >>> IDim = common.Dimension("IDim")
    >>> expr = im.concat_where(im.domain(common.GridType.UNSTRUCTURED, {IDim: (0, 0)}), "a", "b")
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
            if cond.empty():
                return node.args[2]

            # Derive the selected region from the `concat_where`'s own domain rather
            #  than from the branch's. `infer_expr` restricts a branch's domain to the
            #  dimensions of the branch's own type, so for a branch that does not have
            #  the concat dimension the filtered domain has no ranges at all and
            #  `empty()` is vacuously `False` -- a never selected branch would survive.
            cw_domain = node.annex.domain
            if isinstance(cw_domain, domain_utils.SymbolicDomain) and set(cond.ranges) <= set(
                cw_domain.ranges
            ):
                # Only prune when the surviving branch has the same dimensions, else it
                #  would have to be broadcast back to the `concat_where`'s type.
                if _selected_domain(cw_domain, cond, True).empty() and fb.type == node.type:
                    fb.annex.domain = cw_domain
                    return fb
                if _selected_domain(cw_domain, cond, False).empty() and tb.type == node.type:
                    tb.annex.domain = cw_domain
                    return tb

            tb_domain, fb_domain = (
                _filter_domain(arg.annex.domain, cond.ranges.keys()) for arg in node.args[1:]
            )
            assert all(isinstance(d, domain_utils.SymbolicDomain) for d in (tb_domain, fb_domain))
            if tb_domain.empty():
                return node.args[2]
            if fb_domain.empty():
                return node.args[1]

        return node


prune_empty_concat_where = _PruneEmptyConcatWhere.apply
