# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.ffront.fbuiltins import BuiltInFunction, FieldOffset, WhereBuiltinFunction


@BuiltInFunction
def as_offset(offset_: FieldOffset, field: common.Field, /) -> common.Connectivity:
    raise NotImplementedError()


@WhereBuiltinFunction
def concat_where(
    mask: common.Domain | tuple[common.Domain, ...],
    true_field: common.Field | core_defs.ScalarT | Tuple,
    false_field: common.Field | core_defs.ScalarT | Tuple,
    /,
) -> common.Field | Tuple:
    """Assemble a field by selecting from ``true_field`` where ``mask`` applies and from ``false_field`` elsewhere.

    Unlike ``where`` (element-wise selection via a boolean mask field), ``concat_where``
    works on **domain regions**: the mask is a ``Domain`` (not a ``Field``), and the
    result is the concatenation of slices from the two fields along the mask dimension(s).
    Each field only needs to cover its own region — they may be non-overlapping.

    The mask can be:
    - A 1D ``Domain`` (e.g. ``I < 5``): selects a contiguous region along one dimension.
    - A multi-dimensional ``Domain`` (e.g. ``(I < 2) & (J < 3)``): decomposed into
      nested 1D calls automatically.
    - A ``tuple[Domain, ...]`` (e.g. from ``I != 3`` or ``(I > 0) | (J > 0)``):
      handles same-dim tuples directly and different-dim tuples via nesting.

    Args:
        mask: Domain or tuple of Domains specifying the "true" region.
        true_field: Field (or scalar) providing values inside the mask region.
        false_field: Field (or scalar) providing values outside the mask region.

    Returns:
        A new field whose domain is the concatenation of the contributed regions.

    Raises:
        NonContiguousDomain: If the resulting domain has interior gaps.
    """
    raise NotImplementedError()


EXPERIMENTAL_FUN_BUILTIN_NAMES = ["as_offset", "concat_where"]
