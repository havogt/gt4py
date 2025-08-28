# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, Generic, Type, TypeAlias, TypeVar, overload


class DimensionMeta(type):
    def __repr__(cls) -> str:
        return cls.__name__


class Z(metaclass=DimensionMeta):
    pass


class K(Z):
    pass


class Khalf(Z):
    pass


D = TypeVar("D")


class Half(Generic[D]): ...


class MetaDual(type):
    @overload
    def __getitem__(cls, arg: Type[Half[D]]) -> Type[D]: ...  # type: ignore[overload-overlap]
    @overload
    def __getitem__(cls, arg: Type[D]) -> Type[Half[D]]: ...
    def __getitem__(cls, arg: TypeAlias) -> type:
        if hasattr(arg, "__origin__") and arg.__origin__ is Half:
            return arg.__args__[0]
        else:
            return Half[arg]


class ShiftHalf(metaclass=MetaDual):
    """Represents the dual of dimension D"""

    pass


# reveal_type(ShiftHalf[K])
# reveal_type(ShiftHalf[ShiftHalf[K]])


# Field type
class Field(Generic[D]):
    def __init__(self, dim_type: Type[D]):
        self.dim_type = dim_type

    def __repr__(self) -> str:
        return f"Field[{self.dim_type}]"

    @overload
    def __call__(self, shift: int) -> Field[D]: ...
    @overload
    def __call__(self, shift: float) -> Field[ShiftHalf[D]]: ...
    def __call__(self, shift: Any) -> Field[Any]:
        return Field(ShiftHalf[self.dim_type] if isinstance(shift, float) else self.dim_type)

    def __add__(self, other: Field[D]) -> Field[D]:
        return self  # placeholder


def interpolate(f: Field[Z]) -> Field[ShiftHalf[Z]]:
    return f(0.5) + f(-0.5)
    # return Field(result.data, dual_type)  # type: ignore


#  reveal_type(Field(K))
# reveal_type(Field(Dual[K]))
# reveal_type(Field(Dual[Dual[K]]))

f = Field(K)
print(f)
# shifted = f(0.5)
# # reveal_type(shifted)
# shifted2 = shifted(0.5)
# # reveal_type(shifted2)
# shifted3 = f(1)
# reveal_type(shifted3)
# reveal_type(f)

# print(f)

# g = Field(ShiftHalf[K])
# print(g)


def fun(f: Field[K]) -> None: ...


fun(Field(K))
fun(f)

print(interpolate(f))
print(interpolate(interpolate(f)))
