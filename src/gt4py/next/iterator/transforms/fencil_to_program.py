# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py import eve
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms import global_tmps


class FencilToProgram(eve.NodeTranslator):
    @classmethod
    def apply(
        cls, node: itir.FencilDefinition | global_tmps.FencilWithTemporaries | itir.Program
    ) -> itir.Program:
        return cls().visit(node)

    def visit_StencilClosure(self, node: itir.StencilClosure) -> itir.SetAt:
        as_fieldop = im.call(im.call("as_fieldop")(node.stencil, node.domain))(*node.inputs)
        return itir.SetAt(expr=as_fieldop, domain=node.domain, target=node.output)

    def visit_FencilDefinition(self, node: itir.FencilDefinition) -> itir.Program:
        return itir.Program(
            id=node.id,
            function_definitions=node.function_definitions,
            params=node.params,
            declarations=[],
            body=self.visit(node.closures),
        )

    def visit_FencilWithTemporaries(self, node: global_tmps.FencilWithTemporaries) -> itir.Program:
        return itir.Program(
            id=node.fencil.id,
            function_definitions=node.fencil.function_definitions,
            params=node.params,
            declarations=node.tmps,
            body=self.visit(node.fencil.closures),
        )


class ProgramToFencil(eve.NodeTranslator):
    @classmethod
    def apply(cls, node: itir.Program) -> itir.FencilDefinition:
        return cls().visit(node)

    def visit_Program(self, node: itir.Program) -> itir.FencilDefinition:
        return itir.FencilDefinition(
            id=node.id,
            function_definitions=node.function_definitions,
            params=node.params,
            closures=self.visit(node.body),
        )

    def visit_SetAt(self, node: itir.SetAt) -> itir.StencilClosure:
        assert isinstance(node.expr, itir.FunCall) and cpm.is_call_to(node.expr.fun, "as_fieldop")
        return itir.StencilClosure(
            stencil=node.expr.fun.args[0],
            domain=node.domain,
            inputs=node.expr.args,
            output=node.target,
        )
