from typing import Any, Dict

from eve import NodeTranslator
from iterator import ir


class _MapSymbolRefs(NodeTranslator):
    def visit_SymRef(self, node: ir.SymRef, *, symbol_map: Dict[str, Any]):
        return symbol_map.get(node.id, node)

    def visit_Lambda(self, node: ir.Lambda, *, symbol_map: Dict[str, Any]):
        params = {str(p.id) for p in node.params}
        new_symbol_map = {k: v for k, v in symbol_map.items() if k not in params}
        return ir.Lambda(
            params=node.params,
            expr=self.generic_visit(node.expr, symbol_map=new_symbol_map),
        )

    def generic_visit(self, node: ir.Node, **kwargs: Any):
        assert isinstance(node, ir.SymbolTableTrait) == isinstance(
            node, ir.Lambda
        ), "found unexpected new symbol scope"
        return super().generic_visit(node, **kwargs)


class InlineLambdas(NodeTranslator):
    def visit_FunCall(self, node: ir.FunCall):
        node = self.generic_visit(node)
        if isinstance(node.fun, ir.Lambda):
            assert len(node.fun.params) == len(node.args)
            symbol_map = {param.id: arg for param, arg in zip(node.fun.params, node.args)}
            return _MapSymbolRefs().visit(node.fun.expr, symbol_map=symbol_map)
        return node
