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

import abc
import dataclasses

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.eve.extended_typing import (
    TYPE_CHECKING,
    Callable,
    Final,
    Optional,
    Protocol,
    Sequence,
    TypeAlias,
)
from gt4py.next import common
from gt4py.storage import allocators as core_allocators


try:
    import cupy as cp
except ImportError:
    cp = None


class FieldAllocatorInterface(Protocol[core_defs.DeviceTypeT]):
    @property
    @abc.abstractmethod
    def __gt_device_type__(self) -> core_defs.DeviceTypeT:
        ...

    @abc.abstractmethod
    def __gt_allocate__(
        self,
        domain: common.Domain,
        dtype: core_defs.DType[core_defs.ScalarT],
        device_id: int = 0,
        aligned_index: Optional[Sequence[common.NamedIndex]] = None,  # absolute position
    ) -> core_allocators.TensorBuffer[core_defs.DeviceTypeT, core_defs.ScalarT]:
        ...


FieldLayoutMapper: TypeAlias = Callable[
    [Sequence[common.Dimension]], core_allocators.BufferLayoutMap
]


@dataclasses.dataclass(frozen=True)
class FieldAllocator(FieldAllocatorInterface[core_defs.DeviceTypeT]):
    """Parametrizable FieldAllocator."""

    device_type: core_defs.DeviceTypeT
    array_ns: core_allocators.ValidNumPyLikeAllocationNS
    layout_mapper: FieldLayoutMapper
    byte_alignment: int

    @property
    def __gt_device_type__(self) -> core_defs.DeviceTypeT:
        return self.device_type

    def __gt_allocate__(
        self,
        domain: common.Domain,
        dtype: core_defs.DType[core_defs.ScalarT],
        device_id: int = 0,
        aligned_index: Optional[Sequence[common.NamedIndex]] = None,  # absolute position
    ) -> core_allocators.TensorBuffer[core_defs.DeviceTypeT, core_defs.ScalarT]:
        shape = domain.shape
        layout_map = self.layout_mapper(domain.dims)
        assert aligned_index is None  # TODO

        return core_allocators.NDArrayBufferAllocator(self.device_type, self.array_ns).allocate(
            shape, dtype, device_id, layout_map, self.byte_alignment, aligned_index
        )

    __call__ = __gt_allocate__


if TYPE_CHECKING:
    __TensorFieldAllocatorAsFieldAllocatorInterfaceT: type[FieldAllocatorInterface] = FieldAllocator


def horizontal_first_layout_mapper(
    dims: Sequence[common.Dimension],
) -> core_allocators.BufferLayoutMap:
    def pos_of_kind(kind: common.DimensionKind) -> list[int]:
        return [i for i, dim in enumerate(dims) if dim.kind == kind]

    horizontals = pos_of_kind(common.DimensionKind.HORIZONTAL)
    verticals = pos_of_kind(common.DimensionKind.VERTICAL)
    locals_ = pos_of_kind(common.DimensionKind.LOCAL)

    layout_map = [0] * len(dims)
    for i, pos in enumerate(horizontals + verticals + locals_):
        layout_map[pos] = len(dims) - 1 - i
    valid_layout_map = tuple(layout_map)
    assert core_allocators.is_valid_layout_map(valid_layout_map)
    return valid_layout_map


if TYPE_CHECKING:
    __horizontal_first_layout_mapper: FieldLayoutMapper = horizontal_first_layout_mapper


#: Registry of default allocators for each device type.
device_allocators: dict[core_defs.DeviceType, FieldAllocatorInterface] = {}

assert core_allocators.is_valid_nplike_allocation_ns(np)

DefaultCPUAllocator: Final[FieldAllocatorInterface] = FieldAllocator(
    device_type=core_defs.DeviceType.CPU,
    array_ns=np,
    layout_mapper=horizontal_first_layout_mapper,
    byte_alignment=64,
)
device_allocators[core_defs.DeviceType.CPU] = DefaultCPUAllocator

if cp:
    cp_device_type = (
        core_defs.DeviceType.ROCM if cp.cuda.get_hipcc_path() else core_defs.DeviceType.CUDA
    )

    assert core_allocators.is_valid_nplike_allocation_ns(cp)

    DefaultGPUAllocator: Final[FieldAllocatorInterface] = FieldAllocator(
        device_type=core_defs.DeviceType.CPU,
        array_ns=np,
        layout_mapper=horizontal_first_layout_mapper,
        byte_alignment=128,
    )

    device_allocators[cp_device_type] = DefaultGPUAllocator
else:
    DefaultGPUAllocator: Final[Optional[FieldAllocatorInterface]] = None


def allocate(
    domain: common.Domain,
    dtype: core_defs.DType[core_defs.ScalarT],
    *,
    aligned_index: Optional[Sequence[common.NamedIndex]] = None,
    allocator: Optional[FieldAllocatorInterface] = None,
    device: Optional[core_defs.Device] = None,
) -> core_allocators.TensorBuffer:
    """
    Allocate a TensorBuffer with the given settings on the given device.

    The arguments `device` and `allocator` are mutually exclusive.
    If `device` is specified, the corresponding default allocator
    (defined in :data:`device_allocators`) is used.

    Args: TODO
        domain:
        dtype: Data type descriptor as defined in :meth:`BufferAllocator.allocate`.

        aligned_index: N-dimensional index of the first aligned element as defined
            in :meth:`BufferAllocator.allocate`.
    """
    if device is None and allocator is None:
        raise ValueError("No 'device' or 'allocator' specified")
    if device is None:
        assert allocator is not None  # for mypy
        device = core_defs.Device(allocator.__gt_device_type__, 0)
    assert device is not None  # for mypy
    field_allocator = allocator or device_allocators[device.device_type]
    if device.device_type != field_allocator.__gt_device_type__:
        raise ValueError(f"Device {device} and allocator {allocator} are incompatible")

    return field_allocator.__gt_allocate__(
        domain=domain,
        dtype=dtype,
        device_id=device.device_id,
        aligned_index=aligned_index,
    )
