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

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable
from types import ModuleType
from typing import ClassVar, Optional, ParamSpec, TypeAlias, TypeVar, overload

import numpy as np
from numpy import typing as npt

from gt4py._core import definitions as core_defs
from gt4py.next import common

try:
    import cupy as cp
except ImportError:
    cp: Optional[ModuleType] = None  # type:ignore[no-redef]

try:
    from jax import numpy as jnp
except ImportError:
    jnp: Optional[ModuleType] = None  # type:ignore[no-redef]

from gt4py.next.ffront import fbuiltins


def _make_unary_array_field_intrinsic_func(builtin_name: str, array_builtin_name: str) -> Callable:
    def _builtin_unary_op(a: _BaseNdArrayField) -> common.Field:
        xp = a.__class__.array_ns
        op = getattr(xp, array_builtin_name)
        new_data = op(a.ndarray)

        return a.__class__.from_array(new_data, domain=a.domain)

    _builtin_unary_op.__name__ = builtin_name
    return _builtin_unary_op


def _make_binary_array_field_intrinsic_func(builtin_name: str, array_builtin_name: str) -> Callable:
    def _builtin_binary_op(a: _BaseNdArrayField, b: common.Field) -> common.Field:
        xp = a.__class__.array_ns
        op = getattr(xp, array_builtin_name)
        if hasattr(b, "__gt_builtin_func__"):  # isinstance(b, common.Field):
            if not a.domain == b.domain:
                domain_intersection = a.domain & b.domain
                a_broadcasted = broadcast(a, domain_intersection.dims)
                b_broadcasted = broadcast(b, domain_intersection.dims)
                a_slices = _get_slices_from_domain_slice(a_broadcasted.domain, domain_intersection)
                b_slices = _get_slices_from_domain_slice(b_broadcasted.domain, domain_intersection)
                new_data = op(a_broadcasted.ndarray[a_slices], b_broadcasted.ndarray[b_slices])
                return a.__class__.from_array(new_data, domain=domain_intersection)
            new_data = op(a.ndarray, xp.asarray(b.ndarray))
        else:
            # assert isinstance(b, core_defs.SCALAR_TYPES) # TODO reenable this assert (if b is an array it should be wrapped into a field)
            new_data = op(a.ndarray, b)

        return a.__class__.from_array(new_data, domain=a.domain)

    _builtin_binary_op.__name__ = builtin_name
    return _builtin_binary_op


_Value: TypeAlias = common.Field | core_defs.ScalarT
_P = ParamSpec("_P")
_R = TypeVar("_R", _Value, tuple[_Value, ...])


@dataclasses.dataclass(frozen=True)
class _BaseNdArrayField(common.FieldABC[common.DimsT, core_defs.ScalarT]):
    """
    Shared field implementation for NumPy-like fields.

    Builtin function implementations are registered in a dictionary.
    Note: Currently, all concrete NdArray-implementations share
    the same implementation, dispatching is handled inside of the registered
    function via its namespace.
    """

    _domain: common.Domain
    _ndarray: core_defs.NDArrayObject
    _value_type: type[core_defs.ScalarT]

    array_ns: ClassVar[
        ModuleType
    ]  # TODO(havogt) after storage PR is merged, update to the NDArrayNamespace protocol

    _builtin_func_map: ClassVar[dict[fbuiltins.BuiltInFunction, Callable]] = {}

    @classmethod
    def __gt_builtin_func__(cls, func: fbuiltins.BuiltInFunction[_R, _P], /) -> Callable[_P, _R]:
        return cls._builtin_func_map.get(func, NotImplemented)

    @overload
    @classmethod
    def register_builtin_func(
        cls, op: fbuiltins.BuiltInFunction[_R, _P], op_func: None
    ) -> functools.partial[Callable[_P, _R]]:
        ...

    @overload
    @classmethod
    def register_builtin_func(
        cls, op: fbuiltins.BuiltInFunction[_R, _P], op_func: Callable[_P, _R]
    ) -> Callable[_P, _R]:
        ...

    @classmethod
    def register_builtin_func(
        cls, op: fbuiltins.BuiltInFunction[_R, _P], op_func: Optional[Callable[_P, _R]] = None
    ) -> Callable[_P, _R] | functools.partial[Callable[_P, _R]]:
        assert op not in cls._builtin_func_map
        if op_func is None:  # when used as a decorator
            return functools.partial(cls.register_builtin_func, op)  # type: ignore[arg-type]
        return cls._builtin_func_map.setdefault(op, op_func)

    @property
    def domain(self) -> common.Domain:
        return self._domain

    @property
    def __gt_dims__(self) -> common.DimsT:
        return self._domain.dims

    @property
    def __gt_origin__(self) -> tuple[int, ...]:
        return tuple(-r.start for _, r in self._domain)

    @property
    def ndarray(self) -> core_defs.NDArrayObject:
        return self._ndarray

    def __array__(self) -> np.ndarray:
        return np.asarray(self._ndarray)

    @property
    def value_type(self) -> type[core_defs.ScalarT]:
        return self._value_type

    @property
    def dtype(self):
        return self.array_ns.dtype(self._value_type)

    @classmethod
    def from_array(
        cls,
        data: npt.ArrayLike,
        /,
        *,
        domain: common.Domain,
        value_type: Optional[type] = None,
    ) -> _BaseNdArrayField:
        xp = cls.array_ns
        dtype = None
        if value_type is not None:
            dtype = xp.dtype(value_type)
        array = xp.asarray(data, dtype=dtype)

        value_type = array.dtype.type  # TODO add support for Dimensions as value_type

        assert issubclass(array.dtype.type, core_defs.SCALAR_TYPES)

        assert all(isinstance(d, common.Dimension) for d in domain.dims), domain
        assert len(domain) == array.ndim
        assert all(
            len(r) == s or (s == 1 and r == common.UnitRange.infinity)
            for r, s in zip(domain.ranges, array.shape)
        )

        assert value_type is not None  # for mypy
        return cls(domain, array, value_type)

    def remap(self: _BaseNdArrayField, connectivity) -> _BaseNdArrayField:
        raise NotImplementedError()

    # def restrict(
    #     self: _BaseNdArrayField, domain_slice: common.Domain | common.DomainSlice | common.Position
    # ) -> _BaseNdArrayField | _Value:
    #     if common.is_domain_slice(domain_slice):
    #         return self._getitem_domain_slice(domain_slice)
    #     else:
    #         return self.ndarray[domain_slice]  # TODO should return field

    def __setitem__(self, domain, value):
        slices = _get_slices_from_domain_slice(self.domain, domain)
        self.ndarray[slices] = value

    # def _getitem_domain_slice(self, index: common.DomainSlice) -> common.Field:
    #     slices = _get_slices_from_domain_slice(self.domain, index)

    #     dims = []
    #     ranges = []
    #     for k, v in index:
    #         if not common.is_int_index(v):
    #             dims.append(k)
    #             ranges.append(v)

    #     new = self.ndarray[slices]
    #     if len(dims) == 0:
    #         return new  # scalar
    #     else:
    #         new_domain = common.Domain(tuple(dims), tuple(ranges))
    #         return self.__class__(new_domain, new, self.value_type)

    __call__ = None  # type: ignore[assignment]  # TODO: remap

    # __getitem__ = restrict

    __abs__ = _make_unary_array_field_intrinsic_func("abs", "abs")

    __neg__ = _make_unary_array_field_intrinsic_func("neg", "negative")

    def __pos__(self):
        return self

    __add__ = __radd__ = _make_binary_array_field_intrinsic_func("add", "add")

    __sub__ = __rsub__ = _make_binary_array_field_intrinsic_func("sub", "subtract")

    __mul__ = __rmul__ = _make_binary_array_field_intrinsic_func("mul", "multiply")

    __truediv__ = __rtruediv__ = _make_binary_array_field_intrinsic_func("div", "divide")

    __floordiv__ = __rfloordiv__ = _make_binary_array_field_intrinsic_func(
        "floordiv", "floor_divide"
    )

    __pow__ = _make_binary_array_field_intrinsic_func("pow", "power")

    __mod__ = __rmod__ = _make_binary_array_field_intrinsic_func("mod", "mod")

    __and__ = __rand__ = _make_binary_array_field_intrinsic_func("bitwise_and", "bitwise_and")
    __or__ = __ror__ = _make_binary_array_field_intrinsic_func("bitwise_or", "bitwise_or")
    __xor__ = __rxor__ = _make_binary_array_field_intrinsic_func("bitwise_xor", "bitwise_xor")

    __invert__ = _make_unary_array_field_intrinsic_func("invert", "invert")

    @overload
    def __getitem__(self, index: common.DomainSlice) -> common.Field:
        """Absolute slicing with dimension names."""
        ...

    @overload
    def __getitem__(self, index: tuple[slice | int, ...]) -> common.Field:
        """Relative slicing with ordered dimension access."""
        ...

    @overload
    def __getitem__(
        self, index: Sequence[common.NamedIndex]
    ) -> common.Field | core_defs.DType[core_defs.ScalarT]:
        # Value in case len(i) == len(self.domain)
        ...

    def __getitem__(
        self, index: common.DomainSlice | Sequence[common.NamedIndex] | tuple[int, ...]
    ) -> common.Field | core_defs.DType[core_defs.ScalarT]:
        if not (isinstance(index, tuple) or common.is_domain_slice(index)):
            index = (index,)

        # if isinstance(index[0], common.Domain):
        #     index = index[0]

        if common.is_domain_slice(index):
            return self._getitem_absolute_slice(index)

        if all(isinstance(idx, slice) or common.is_int_index(idx) for idx in index):
            return self._getitem_relative_slice(index)

        raise IndexError(f"Unsupported index type: {index}")

    restrict = __getitem__

    def _getitem_absolute_slice(self, index: common.DomainSlice) -> common.Field:
        # all_named_range = all(isinstance(idx[0], Dimension) and isinstance(idx[1], UnitRange) for idx in index)
        # all_named_index = all(isinstance(idx[0], Dimension) and isinstance(idx[1], int) for idx in index)
        slices = _get_slices_from_domain_slice(self.domain, index)

        # if all_named_range or all_named_index:
        new_ranges = []
        new_dims = []
        new = self.ndarray[slices]

        for i, dim in enumerate(self.domain.dims):
            if (pos := _find_index_of_dim(dim, index)) is not None:
                index_or_range = index[pos][1]
                if isinstance(index_or_range, common.UnitRange):
                    new_ranges.append(index_or_range)
                    new_dims.append(dim)
            else:
                # dimension not mentioned in slice
                new_ranges.append(self.domain.ranges[i])
                new_dims.append(dim)

        new_domain = common.Domain(dims=tuple(new_dims), ranges=tuple(new_ranges))
        # elif :
        # idx = self._get_new_domain_indices(index)
        # new = self.ndarray[self._create_new_index_tuple(slices, index)]
        # new_domain = self._create_new_domain_with_indices(idx)

        return new if new.ndim == 0 else common.field(new, domain=new_domain)

    def _get_new_domain_indices(self, index: common.NamedIndex) -> tuple[int]:
        ndarray_shape = self.ndarray.shape
        dim_indices_to_exclude = [self.domain.dims.index(dim[0]) for dim in index]
        new_domain_indices = [
            i for i in range(len(ndarray_shape)) if i not in dim_indices_to_exclude
        ]
        return tuple(new_domain_indices)

    def _create_new_index_tuple(
        self, slices: tuple[int], index: common.NamedIndex
    ) -> tuple[int | slice]:
        all_dims = self.domain.dims
        subset_dims = [dim for dim, _ in index]
        missing_dim_indices = [i for i, dim in enumerate(all_dims) if dim not in subset_dims]

        new_index_list = []
        slices_index = 0
        for i in range(len(all_dims)):
            if i in missing_dim_indices:
                new_index_list.append(slice(None))
            else:
                new_index_list.append(slices[slices_index])
                slices_index += 1
        return tuple(new_index_list)

    def _create_new_domain_with_indices(self, indices: tuple[int]) -> common.Domain:
        new_dims = index_tuple_with_indices(self.domain.dims, indices)
        new_ranges = index_tuple_with_indices(self.domain.ranges, indices)
        return common.Domain(dims=new_dims, ranges=new_ranges)

    def _getitem_relative_slice(self, index: tuple[slice | int, ...]) -> common.Field:
        new = self.ndarray[index]

        if len(new.shape) == 0:
            return new
        new_dims = []
        new_ranges = []

        dim_diff = len(self.domain) - len(index)

        if dim_diff > 0:
            new_index = tuple([*index] + [Ellipsis] * dim_diff)
        else:
            new_index = index

        for i, elem in enumerate(new_index):
            if isinstance(elem, slice):
                new_dims.append(self.domain.dims[i])
                new_ranges.append(self._slice_range(self.domain.ranges[i], elem))
            elif common.is_int_index(elem):
                ...
                # new_dims.append(self.domain.dims[elem])
                # new_ranges.append(self.domain.ranges[elem])
            elif isinstance(elem, type(Ellipsis)):
                new_dims.append(self.domain.dims[i])
                new_ranges.append(self.domain.ranges[i])

        new_domain = common.Domain(dims=new_dims, ranges=tuple(new_ranges))
        return common.field(new, domain=new_domain)

    def _slice_range(self, input_range: common.UnitRange, slice_obj: slice) -> common.UnitRange:
        start = (input_range.start if (slice_obj.start or 0) >= 0 else input_range.stop) + (
            slice_obj.start or 0
        )
        stop = (
            input_range.start if (slice_obj.stop or len(input_range)) >= 0 else input_range.stop
        ) + (slice_obj.stop or len(input_range))
        return common.UnitRange(start, stop)
        # if slice_obj.start is None:
        #     slice_start = 0
        # else:
        #     slice_start = (
        #         slice_obj.start if slice_obj.start >= 0 else input_range.stop + slice_obj.start
        #     )

        # if slice_obj.stop is None:
        #     slice_stop = 0
        # else:
        #     slice_stop = (
        #         slice_obj.stop if slice_obj.stop >= 0 else input_range.stop + slice_obj.stop
        #     )

        # start = input_range.start + slice_start
        # stop = input_range.start + slice_stop

        # return common.UnitRange(start, stop)


# -- Specialized implementations for intrinsic operations on array fields --

_BaseNdArrayField.register_builtin_func(fbuiltins.abs, _BaseNdArrayField.__abs__)  # type: ignore[attr-defined]
_BaseNdArrayField.register_builtin_func(fbuiltins.power, _BaseNdArrayField.__pow__)  # type: ignore[attr-defined]
# TODO gamma

for name in (
    fbuiltins.UNARY_MATH_FP_BUILTIN_NAMES
    + fbuiltins.UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES
    + fbuiltins.UNARY_MATH_NUMBER_BUILTIN_NAMES
):
    if name in ["abs", "power", "gamma"]:
        continue
    _BaseNdArrayField.register_builtin_func(
        getattr(fbuiltins, name), _make_unary_array_field_intrinsic_func(name, name)
    )

_BaseNdArrayField.register_builtin_func(
    fbuiltins.minimum, _make_binary_array_field_intrinsic_func("minimum", "minimum")  # type: ignore[attr-defined]
)
_BaseNdArrayField.register_builtin_func(
    fbuiltins.maximum, _make_binary_array_field_intrinsic_func("maximum", "maximum")  # type: ignore[attr-defined]
)
_BaseNdArrayField.register_builtin_func(
    fbuiltins.fmod, _make_binary_array_field_intrinsic_func("fmod", "fmod")  # type: ignore[attr-defined]
)

# -- Concrete array implementations --
# NumPy
_nd_array_implementations = [np]


@dataclasses.dataclass(frozen=True)
class NumPyArrayField(_BaseNdArrayField):
    array_ns: ClassVar[ModuleType] = np


common.field.register(np.ndarray, NumPyArrayField.from_array)

# CuPy
if cp:
    _nd_array_implementations.append(cp)

    @dataclasses.dataclass(frozen=True)
    class CuPyArrayField(_BaseNdArrayField):
        array_ns: ClassVar[ModuleType] = cp

    common.field.register(cp.ndarray, CuPyArrayField.from_array)

# JAX
if jnp:
    _nd_array_implementations.append(jnp)

    @dataclasses.dataclass(frozen=True)
    class JaxArrayField(_BaseNdArrayField):
        array_ns: ClassVar[ModuleType] = jnp

    common.field.register(jnp.ndarray, JaxArrayField.from_array)


def _find_index_of_dim(dim: Dimension, domain_slice: common.DomainSlice) -> Optional[int]:
    for i, (d, _) in enumerate(domain_slice):
        if dim == d:
            return i
    return None


def broadcast(field: common.Field, new_dimensions: tuple[common.Dimension, ...]):
    # assert all(dim in new_domain for dim in domain)
    # assert len(domain) <= len(new_domain)
    # domain and new_domain are ordered with `promote_dims`

    # slice or broadcast

    domain_slice = []

    new_domain_dims = []
    new_domain_ranges = []
    for dim in new_dimensions:
        if (pos := _find_index_of_dim(dim, field.domain)) is not None:
            domain_slice.append(slice(None))
            new_domain_dims.append(dim)
            new_domain_ranges.append(field.domain[pos][1])
        else:
            domain_slice.append(np.newaxis)
            new_domain_dims.append(dim)
            new_domain_ranges.append(
                common.UnitRange(common.Infinity.negative(), common.Infinity.positive())
            )
    return common.field(
        field.ndarray[tuple(domain_slice)],
        domain=common.Domain(tuple(new_domain_dims), tuple(new_domain_ranges)),
    )


# def _get_slices_from_domain(domain, new_domain):
#     # assert all(dim in new_domain for dim in domain)
#     # assert len(domain) <= len(new_domain)
#     # domain and new_domain are ordered with `promote_dims`

#     # slice or broadcast

#     domain_slice = []

#     new_domain_dims =[]
#     new_domain_ranges=[]
#     for dim, rng in new_domain:
#         if (pos := _find_index_of_dim(dim, domain)) is not None:
#             domain_slice.append((dim, rng))
#             new_domain_dims.append(dim)
#             new_domain_ranges.append(domain[pos][1])
#         else:
#             domain_slice.append((dim, np.newaxis))
#             new_domain_dims.append(dim)
#             new_domain_ranges.append(UnitRange(common.Infinity.negative(), common.Infinity.positive()))
#     return _get_slices_from_domain_slice(Domain(tuple(new_domain_dims), tuple(new_domain_ranges)), domain_slice)


def _get_slices_from_domain_slice(
    domain: common.Domain, domain_slice: common.DomainSlice
) -> tuple[slice | int | None, ...]:
    """Generate slices for sub-array extraction based on named ranges or named indices within a Domain.

    This function generates a tuple of slices that can be used to extract sub-arrays from a field. The provided
    named ranges or indices specify the dimensions and ranges of the sub-arrays to be extracted.

    Args:
        domain (common.Domain): The Domain object representing the original field.
        domain_slice (DomainSlice): A sequence of dimension names and associated ranges.

    Returns:
        tuple[slice | int | None, ...]: A tuple of slices representing the sub-array extraction along each dimension
                                       specified in the Domain. If a dimension is not included in the named indices
                                       or ranges, a None is used to indicate expansion along that axis.
    """
    # assert all(dim in domain for dim in domain_slice)
    # assert len(domain) >= len(domain_slice)
    # no ordering of domain_slice dimensions

    slice_indices: list[slice | int | None] = []

    # all_dims = ... # ordered such that

    # for dim in all_dims:
    #     if dim in domain_slice and dim in domain:
    #         slice_indices.append(_compute_slice(index_or_range, domain, pos_old)) # slice dimension in
    #     elif dim in domain_slice:
    #         slice_indices.append(None) # np.newaxis
    #     else:
    #         slice_indices.append(slice(None)) # ellipsis (take whole dimension)

    for dim, rng in domain:
        if (pos := _find_index_of_dim(dim, domain_slice)) is not None:
            index_or_range = domain_slice[pos][1]
            # if index_or_range is np.newaxis:
            #     slice_indices.append(index_or_range)
            # else:
            shifted = index_or_range - rng.start
            slice_indices.append(_to_slice(shifted))
        else:
            slice_indices.append(slice(None))

    # for new_dim, new_rng in domain_slice:
    #     pos_new = next(index for index, (dim, _) in enumerate(domain_slice) if dim == new_dim)

    #     if new_dim in domain.dims:
    #         pos_old = domain.dims.index(new_dim)
    #         slice_indices.append(_compute_slice(new_rng, domain, pos_old))
    #     else:
    #         slice_indices.insert(pos_new, None)  # None is equal to np.newaxis

    return tuple(slice_indices)


def _to_slice(value: common.IntIndex | common.UnitRange) -> common.IntIndex | slice:
    if isinstance(value, common.UnitRange):
        return slice(
            None if value.start == common.Infinity.negative() else value.start,
            None if value.stop == common.Infinity.positive() else value.stop,
        )
    else:
        return value
