import eve  # noqa: F401
from gt4py.gtc import gtir, stageir


class GTIRToStageIR(eve.NodeTranslator):
    # def visit_BinaryOp(self, node: gtir.BinaryOp, **kwargs):
    #     return stageir.BinaryOp(op=node.op, left=self.visit(node.left), right=self.visit(node.right))

    def visit_VerticalInterval(self, node: gtir.VerticalInterval, **kwargs):
        return stageir.StageInterval(
            start=self.visit(node.start), end=self.visit(node.end), body=[]
        )

    def visit_VerticalLoop(self, node: gtir.VerticalLoop, **kwargs):
        return stageir.MultiStage(
            stages=[stageir.Stage(intervals=self.visit(node.vertical_intervals))],
            loop_order=node.loop_order,
        )

    def visit_Stencil(self, node: gtir.Stencil, **kwargs):
        return stageir.Stencil(
            name=node.name, params=[], multi_stages=self.visit(node.vertical_loops)
        )
