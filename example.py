# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections import namedtuple
from typing import Generic, Type, TypeVar


Shift = namedtuple("Shift", ["dimension", "offset"])


class DimensionMeta(type):
    def __add__(cls, offset: float) -> Shift:
        return Shift(cls, offset)

    def __sub__(cls, offset: float) -> Shift:
        return Shift(cls, -offset)


class VerticalDimension(metaclass=DimensionMeta):
    pass


class K(VerticalDimension):
    pass


class Khalf(VerticalDimension):
    pass


# Generic dual mapping
D = TypeVar("D")


class Dual(Generic[D]):
    """Represents the dual of dimension D"""

    pass


# Field type
class Field(Generic[D]):
    def __init__(self, data: str, dim_type: Type[D]):
        self.data = data
        self.dim_type = dim_type

    def __call__(self, shift: Shift) -> "Field[Dual[D]]":
        return Field(self.data, Dual[D])

    def __add__(self, other: "Field[D]") -> "Field[D]":
        return Field(f"({self.data} + {other.data})", self.dim_type)

    def __mul__(self, scalar: float) -> "Field[D]":
        return Field(f"{scalar} * {self.data}", self.dim_type)

    def __rmul__(self, scalar: float) -> "Field[D]":
        return self.__mul__(scalar)


# The typing-clean interpolate function!
def interpolate[Z: VerticalDimension](f: Field[Z]) -> Field[Dual[Z]]:
    # Implementation maps Dual[Z] to actual dual type
    # dual_type: type[VerticalDimension]
    # if f.dim_type is K:
    #     dual_type = Dual[K]
    # elif f.dim_type is Khalf:
    #     dual_type = Dual[Khalf]
    # else:
    #     assert False
    # else:
    #     dual_type = f.dim_type
    # dual_type = Dual[f.dim_type]

    Z = f.dim_type
    left = f(Z - 0.5)
    right = f(Z + 0.5)
    result = 0.5 * (left + right)
    return result


# Usage
def main() -> None:
    field_K: Field[K] = Field("data", K)
    field_Khalf: Field[Khalf] = Field("data", Khalf)

    # These have the clean generic signature you wanted!
    result1 = interpolate(field_K)  # Type: Field[Dual[Type[K]]]
    result2 = interpolate(field_Khalf)  # Type: Field[Dual[Type[Khalf]]]

    reveal_type(result1)  # Field[Dual[Type[K]]]
    reveal_type(result2)  # Field[Dual[Type[Khalf]]]


if __name__ == "__main__":
    main()
