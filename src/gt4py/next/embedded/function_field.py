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
import operator
from typing import Any, Callable, TypeGuard, overload

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next import common
from gt4py.next.embedded import (
    common as embedded_common,
    exceptions as embedded_exceptions,
    nd_array_field as nd,
)
from gt4py.next.ffront import fbuiltins


@dataclasses.dataclass(frozen=True)
class FunctionField(common.Field[common.DimsT, core_defs.ScalarT], common.FieldBuiltinFuncRegistry):
    func: Callable
    domain: common.Domain = common.Domain()
    _skip_invariant: bool = False

    def __post_init__(self):
        if not self._skip_invariant:
            num_params = len(self.domain)
            try:
                func_params = self.func.__code__.co_argcount
                if func_params != num_params:
                    raise ValueError(
                        f"Invariant violation: len(self.domain) ({num_params}) does not match the number of parameters of the self.func ({func_params})."
                    )
            except AttributeError:
                raise ValueError(f"Must pass a function as an argument to self.func.")

    def restrict(self, index: common.AnyIndexSpec) -> FunctionField:
        new_domain = embedded_common.sub_domain(self.domain, index)
        return self.__class__(self.func, new_domain)

    __getitem__ = restrict

    @property
    def ndarray(self) -> core_defs.NDArrayObject:
        if not self.domain.is_finite():
            embedded_exceptions.InfiniteRangeNdarrayError(self.__class__.__name__, self.domain)

        shape = [len(rng) for rng in self.domain.ranges]

        return np.fromfunction(self.func, shape)

    def _handle_function_field_op(self, other: FunctionField, op: Callable) -> FunctionField:
        domain_intersection = self.domain & other.domain
        broadcasted_self = _broadcast(self, domain_intersection.dims)
        broadcasted_other = _broadcast(other, domain_intersection.dims)
        return self.__class__(
            _compose(op, broadcasted_self, broadcasted_other),
            domain_intersection,
            _skip_invariant=True,
        )

    def _handle_scalar_op(self, other: FunctionField, op: Callable) -> FunctionField:
        new_func = lambda *args: op(self.func(*args), other)
        return self.__class__(
            new_func, self.domain, _skip_invariant=True
        )  # skip invariant as we cannot deduce number of args

    @overload
    def _binary_operation(self, op: Callable, other: core_defs.ScalarT) -> common.Field:
        ...

    @overload
    def _binary_operation(self, op: Callable, other: common.Field) -> common.Field:
        ...

    def _binary_operation(self, op, other):
        if isinstance(other, self.__class__):
            return self._handle_function_field_op(other, op)
        elif isinstance(other, (int, float)):  # Handle scalar values
            return self._handle_scalar_op(other, op)
        else:
            return op(other, self)

    def _unary_op(self, op: Callable) -> FunctionField:
        return self.__class__(_compose(op, self), self.domain, _skip_invariant=True)

    def __add__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        return self._binary_operation(operator.add, other)

    def __sub__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        return self._binary_operation(operator.sub, other)

    def __mul__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        return self._binary_operation(operator.mul, other)

    def __truediv__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        return self._binary_operation(operator.truediv, other)

    def __floordiv__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        return self._binary_operation(operator.floordiv, other)

    def __mod__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        return self._binary_operation(operator.mod, other)

    def __pow__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        return self._binary_operation(operator.pow, other)

    def __lt__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        return self._binary_operation(operator.lt, other)

    def __le__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        return self._binary_operation(operator.le, other)

    def __gt__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        return self._binary_operation(operator.gt, other)

    def __ge__(self, other: common.Field | core_defs.ScalarT) -> common.Field:
        return self._binary_operation(operator.ge, other)

    def __pos__(self) -> common.Field:
        return self._unary_op(operator.pos)

    def __neg__(self) -> common.Field:
        return self._unary_op(operator.neg)

    def __invert__(self) -> common.Field:
        return self._unary_op(operator.invert)

    def __abs__(self) -> common.Field:
        return self._unary_op(abs)

    def __and__(self, other) -> common.Field:
        raise NotImplementedError("Method __and__ not implemented")

    def __call__(self, *args, **kwargs) -> common.Field:
        raise NotImplementedError("Method __call__ not implemented")

    def __or__(self, other) -> common.Field:
        raise NotImplementedError("Method __or__ not implemented")

    def __radd__(self, other) -> common.Field:
        raise NotImplementedError("Method __radd__ not implemented")

    def __rfloordiv__(self, other) -> common.Field:
        raise NotImplementedError("Method __rfloordiv__ not implemented")

    def __rmul__(self, other) -> common.Field:
        raise NotImplementedError("Method __rmul__ not implemented")

    def __rsub__(self, other) -> common.Field:
        raise NotImplementedError("Method __rsub__ not implemented")

    def __rtruediv__(self, other) -> common.Field:
        raise NotImplementedError("Method __rtruediv__ not implemented")

    def __xor__(self, other) -> common.Field:
        raise NotImplementedError("Method __xor__ not implemented")

    def remap(self, *args, **kwargs) -> common.Field:
        raise NotImplementedError("Method remap not implemented")


def _compose(operation: Callable, *fields: FunctionField) -> Callable:
    return lambda *args: operation(*[f.func(*args) for f in fields])


def _broadcast(field: FunctionField, dims: tuple[common.Dimension, ...]) -> FunctionField:
    def broadcasted_func(*args: int):
        selected_args = [args[i] for i, dim in enumerate(dims) if dim in field.domain.dims]
        return field.func(*selected_args)

    named_ranges = embedded_common._compute_named_ranges(field, dims)
    return FunctionField(broadcasted_func, common.Domain(*named_ranges), _skip_invariant=True)


def _is_nd_array(other: Any) -> TypeGuard[nd._BaseNdArrayField]:
    return isinstance(other, nd._BaseNdArrayField)


def constant_field(
    value: core_defs.ScalarT, domain: common.Domain = common.Domain()
) -> common.Field:
    return FunctionField(lambda *args: value, domain, _skip_invariant=True)


FunctionField.register_builtin_func(fbuiltins.broadcast, _broadcast)
