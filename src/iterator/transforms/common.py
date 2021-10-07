from iterator.transforms.inline_fundefs import InlineFundefs, PruneUnreferencedFundefs
from iterator.transforms.inline_lambdas import InlineLambdas
from iterator.transforms.inline_lifts import InlineLifts
from iterator.transforms.normalize_shifts import NormalizeShifts


def apply_common_transforms(ir):
    ir = InlineFundefs().visit(ir)
    ir = PruneUnreferencedFundefs().visit(ir)
    ir = NormalizeShifts().visit(ir)
    ir = InlineLifts().visit(ir)
    ir = InlineLambdas().visit(ir)
    ir = NormalizeShifts().visit(ir)
    return ir
