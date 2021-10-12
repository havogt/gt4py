from eve import NodeTranslator
from iterator import ir
from iterator.transforms.remap_symbols import RemapSymbolRefs


class InlineLambdas(NodeTranslator):
    def visit_FunCall(self, node: ir.FunCall):
        node = self.generic_visit(node)
        if isinstance(node.fun, ir.Lambda):
            assert len(node.fun.params) == len(node.args)
            symbol_map = {param.id: arg for param, arg in zip(node.fun.params, node.args)}
            return RemapSymbolRefs().visit(node.fun.expr, symbol_map=symbol_map)
        return node
