# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py.eve import NodeTranslator, PreserveLocationVisitor
from gt4py.next import errors
from gt4py.next.iterator import ir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.iterator.transforms.symbol_ref_utils import collect_symbol_refs
from gt4py.next.type_system import type_info


class NormalizeShifts(PreserveLocationVisitor, NodeTranslator):
    def visit_FunCall(self, node: ir.FunCall):
        node = self.generic_visit(node)
        if (
            isinstance(node.fun, ir.FunCall)
            and isinstance(node.fun.fun, ir.SymRef)
            and node.fun.fun.id == "shift"
            and node.args
            and isinstance(node.args[0], ir.FunCall)
            and isinstance(node.args[0].fun, ir.FunCall)
            and isinstance(node.args[0].fun.fun, ir.SymRef)
            and node.args[0].fun.fun.id == "shift"
        ):
            # shift(args1...)(shift(args2...)(it)) -> shift(args2..., args1...)(it)
            assert len(node.args) == 1
            return ir.FunCall(
                fun=ir.FunCall(
                    fun=ir.SymRef(id="shift"), args=node.args[0].fun.args + node.fun.args
                ),
                args=node.args[0].args,
            )
        return node


class CanonicalizeShiftOffsets(PreserveLocationVisitor, NodeTranslator):
    """
    Turn integral `Literal` shift offsets into `OffsetLiteral`s; reject Cartesian shifts by a value
    that is not known at compile time.

    A shift offset that is a static program argument, e.g. `m` in `field(IDim + m)`, reaches the
    shift as an `ir.Literal` once the argument is substituted and constant folded. Domain inference
    requires the `ir.OffsetLiteral` that literal offsets have from the start.

    A Cartesian shift whose offset still refers to a symbol from outside its stencil is a shift by
    a runtime scalar: it is the same at every point, but domain inference can not translate by it.
    Offsets read from inside the stencil, e.g. `deref` of an offset field, are left to
    `InlineDynamicShifts`.
    """

    def visit_FunCall(self, node: ir.FunCall, *, stencil_params: frozenset[str] = frozenset()):
        if cpm.is_call_to(node, "as_fieldop"):
            stencil_params = frozenset()
        node = self.generic_visit(node, stencil_params=stencil_params)
        if not cpm.is_call_to(node, "shift"):
            return node

        args = [
            ir.OffsetLiteral(value=int(arg.value))
            if isinstance(arg, ir.Literal) and type_info.is_integral(arg.type)
            else arg
            for arg in node.args
        ]
        for tag, offset in zip(args[::2], args[1::2]):
            if isinstance(tag, ir.CartesianOffset) and not isinstance(offset, ir.OffsetLiteral):
                if free_symbols := sorted(set(collect_symbol_refs(offset)) - stencil_params):
                    raise errors.NonStaticShiftError(
                        node.location, str(tag.domain.value), free_symbols
                    )
        return ir.FunCall(fun=node.fun, args=args)

    def visit_Lambda(self, node: ir.Lambda, *, stencil_params: frozenset[str] = frozenset()):
        return self.generic_visit(
            node, stencil_params=stencil_params | {str(p.id) for p in node.params}
        )
