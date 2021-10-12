from typing import Any, Dict

from eve import NodeTranslator
from iterator import ir


class RemapSymbolRefs(NodeTranslator):
    def visit_SymRef(self, node: ir.SymRef, *, symbol_map: Dict[str, ir.Node]):
        return symbol_map.get(node.id, node)

    def visit_Lambda(self, node: ir.Lambda, *, symbol_map: Dict[str, ir.Node]):
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
