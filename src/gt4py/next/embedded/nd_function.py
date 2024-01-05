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
import types
from typing import Any, Callable, Optional

import numpy as np

from gt4py._core import definitions as core_defs
from gt4py.next.ffront import fbuiltins


float32 = np.float32


@dataclasses.dataclass(frozen=True)
class _DType:
    type: type


# @dataclasses.dataclass(frozen=True)
# class _Function:
#     fun: Callable
#     dtype: _DType
#     shape: tuple[int,...]

#     @property
#     def ndim(self):
#         return len(self.shape)

#     def __call__(self, *args: Any, **kwargs: Any) -> Any:
#         return self.fun(*args, **kwargs)

#     def __array_namespace__(self: _Function, /, *, api_version: Optional[str] = None) -> types.ModuleType:
#         from gt4py.next.embedded import nd_function
#         return nd_function


@dataclasses.dataclass(frozen=True)
class _LazyFunction:
    fun: Callable | str
    args: tuple[_LazyFunction, ...]
    dtype: _DType
    shape: tuple[int, ...]

    @property
    def ndim(self):
        return len(self.shape)

    def eval(self, array_ns):
        def impl(
            *args,
        ):
            if isinstance(self.fun, str):
                fun = getattr(array_ns, self.fun)
                return fun(*[a.eval(array_ns)(*args) for a in self.args])
            else:
                assert len(self.args) == 0
                return self.fun(*args)

        return impl


def asarray(a, dtype=None):
    if isinstance(a, _LazyFunction):
        return a
    if hasattr(a, "shape"):
        dtype = _DType(a.dtype.type)
        shape = a.shape
        array = a
    elif hasattr(a, "__len__"):
        shape = (len(a),)
        array = np.asarray(a)
        dtype = _DType(type(array[0]))
    else:
        raise NotImplementedError()
    return _LazyFunction(
        lambda *args: array.__getitem__(*list(a.astype(int) for a in args)),
        args=(),
        dtype=dtype,
        shape=shape,
    )


def asnumpy(a):
    return np.fromfunction(a.eval(np), a.shape, dtype=np.dtype(a.dtype.type))


def convert_array(a, xp):
    return xp.fromfunction(a.eval(xp), a.shape, dtype=np.dtype(a.dtype.type))


UNARY = (
    fbuiltins.UNARY_MATH_FP_BUILTIN_NAMES
    + fbuiltins.UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES
    + fbuiltins.UNARY_MATH_NUMBER_BUILTIN_NAMES
)


def __getattr__(name):
    if (
        name
        in [
            "where",
            "negative",
            "positive",
            "invert",
            "add",
            "subtract",
            "multiply",
            "divide",
            "floor_divide",
            "mod",
            "logical_xor",
            "logical_and",
            "logical_or",
            "abs",
            "minimum",
            "maximum",
            "fmod",
            "power",
        ]
        + UNARY
    ):
        return lambda *args: _LazyFunction(name, args, dtype=args[0].dtype, shape=args[0].shape)
    raise AttributeError(name)
