from iterator.transforms.inline_fundefs import InlineFundefs, PruneUnreferencedFundefs
from iterator.transforms.normalize_shifts import NormalizeShifts


def apply_common_transforms(ir):
    ir = InlineFundefs().visit(ir)
    ir = PruneUnreferencedFundefs().visit(ir)
    ir = NormalizeShifts().visit(ir)
    return ir
