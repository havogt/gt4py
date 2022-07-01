from functional.iterator import type_inference
from functional.iterator.processor_interface import fencil_formatter
from functional.iterator.transforms import apply_common_transforms


@fencil_formatter
def check(root, *args, **kwargs):
    type_inference.pprint(type_inference.infer(root))
    transformed = apply_common_transforms(
        root, use_tmps=kwargs.get("use_tmps", False), offset_provider=kwargs["offset_provider"]
    )
    type_inference.pprint(type_inference.infer(transformed))
