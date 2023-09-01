# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

from typing import Optional, Sequence

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.storage import allocators as next_allocators
from gt4py.storage.cartesian import utils as storage_utils


# Public interface
def empty(
    domain: common.Domain,
    dtype: core_defs.DTypeLike = core_defs.Float64DType(()),
    allocator: next_allocators.FieldAllocatorInterface = next_allocators.DefaultCPUAllocator(),
    *,
    device_id: int = 0,
    aligned_index: Optional[Sequence[common.NamedIndex]] = None,
) -> common.Field:
    """Allocate an array of uninitialized (undefined) values with performance-optimal strides and alignment.

    !!!TODO!!!

    Parameters
    ----------
        shape : `Sequence` of `int`
            The shape of the resulting `ndarray`
        dtype :  DTypeLike, optional
            The dtype of the resulting `ndarray`

    Keyword Arguments
    -----------------
        backend : `str`
            The target backend for which the allocation is optimized.
        aligned_index: `Sequence` of `int`, optional
            Indicate the index of the resulting array that most commonly corresponds to the origin of the compute
            domain. If not passed, it is aligned at the data origin.
        dimensions: `Sequence` of `str`, optional
            Indicate the semantic meaning of the dimensions in the provided array. Only used for determining optimal
            strides, the information is not stored.

    Returns
    -------
        NumPy or CuPy ndarray
            With uninitialized values, padded and aligned to provide optimal performance for the given `backend` and
            `aligned_index`

    Raises
    -------
        TypeError
            If arguments of an unexpected type are specified.
        ValueError
            If illegal or inconsistent arguments are specified.
    """
    dtype = core_defs.dtype(dtype)
    buffer = allocator.__gt_allocate__(domain, dtype, device_id, aligned_index)
    return common.field(buffer, domain=domain)


def zeros(
    domain: common.Domain,
    dtype: core_defs.DTypeLike = core_defs.Float64DType(()),
    allocator: next_allocators.FieldAllocatorInterface = next_allocators.DefaultCPUAllocator(),
    *,
    device_id: int = 0,
    aligned_index: Optional[Sequence[common.NamedIndex]] = None,
) -> common.Field:
    field = empty(
        domain=domain,
        dtype=dtype,
        allocator=allocator,
        device_id=device_id,
        aligned_index=aligned_index,
    )
    field.ndarray[...] = field.value_type(0)  # TODO dtype.scalar_type
    return field


def ones(
    domain: common.Domain,
    dtype: core_defs.DTypeLike = core_defs.Float64DType(()),
    allocator: next_allocators.FieldAllocatorInterface = next_allocators.DefaultCPUAllocator(),
    *,
    device_id: int = 0,
    aligned_index: Optional[Sequence[common.NamedIndex]] = None,
) -> common.Field:
    field = empty(
        domain=domain,
        dtype=dtype,
        allocator=allocator,
        device_id=device_id,
        aligned_index=aligned_index,
    )
    field.ndarray[...] = field.value_type(1)  # TODO dtype.scalar_type
    return field


def full(
    domain: common.Domain,
    fill_value: core_defs.Scalar,
    dtype: core_defs.DTypeLike = core_defs.Float64DType(()),
    allocator: next_allocators.FieldAllocatorInterface = next_allocators.DefaultCPUAllocator(),
    *,
    device_id: int = 0,
    aligned_index: Optional[Sequence[common.NamedIndex]] = None,
) -> common.Field:
    field = empty(
        domain=domain,
        dtype=dtype,
        allocator=allocator,
        device_id=device_id,
        aligned_index=aligned_index,
    )
    field.ndarray[...] = field.value_type(fill_value)  # TODO dtype.scalar_type
    return field


def asfield(
    domain: common.Domain,
    data: core_defs.NDArrayObject,
    dtype: core_defs.DTypeLike = core_defs.Float64DType(()),
    allocator: next_allocators.FieldAllocatorInterface = next_allocators.DefaultCPUAllocator(),
    *,
    device_id: int = 0,
    aligned_index: Optional[Sequence[common.NamedIndex]] = None,
    # copy=False, TODO
) -> common.Field:
    # TODO make sure we don't reallocate if its in correct layout and device
    shape = storage_utils.asarray(data).shape
    if shape != domain.shape:
        raise ValueError(f"Cannot construct `Field` from array of shape `{shape}` ")
    if dtype is None:
        dtype = storage_utils.asarray(data).dtype
    dtype = core_defs.dtype(dtype)
    assert dtype.tensor_shape == ()  # TODO
    field = empty(
        domain=domain,
        dtype=dtype,
        allocator=allocator,
        device_id=device_id,
        aligned_index=aligned_index,
    )

    field.ndarray[...] = field.array_ns.asarray(data)

    return field
