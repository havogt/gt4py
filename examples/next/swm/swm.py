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
from typing import Any, TypeAlias

import initial_conditions
import numpy as np
import utils

from gt4py import next as gtx
from gt4py.next import common, fbuiltins, float64


# Thoughts
# - don't hard-code staggering in parsing: tracing augmented parsing
# - keep frontend beautification to execution, then it's easier to change

TopologicalDimension: TypeAlias = common.Dimension


@dataclasses.dataclass
class FooCartesianOffset:
    from_: TopologicalDimension
    to_: TopologicalDimension
    offset: int


# Offset needs to be a callable that takes field dimensions and returns a ConnectivityType


@dataclasses.dataclass
class Half:
    _value: int = 0

    @property
    def value(self):
        return self._value

    def __neg__(self):
        return Half(self._value + 1)


# TODO think about:
# probably GeometricalDimension is a pure frontend feature
# after parsing we should only deal with the original dimensions (TopologicalDimensions)
@dataclasses.dataclass(frozen=True)
class StaggeredDimension:
    _d1: common.Dimension
    _d2: common.Dimension
    _value: int = 0

    def __add__(self, value: Any) -> StaggeredDimension:
        return dataclasses.replace(self, _value=value)

    def __sub__(self, value: Any) -> StaggeredDimension:
        return dataclasses.replace(self, _value=-value)

    def _get_dual_dim(self, dim):
        if dim == self._d1:
            return (self._d2, 0)
        else:
            assert dim == self._d2
            return (self._d1, -1)

    def as_connectivity_field(self, dims):
        dim_source = [d for d in dims if d in [self._d1, self._d2]][0]
        if isinstance(self._value, int):
            return common.CartesianConnectivity(dim_source, self._value)
        elif isinstance(self._value, Half):
            dual_dim, offset = self._get_dual_dim(dim_source)
            return common.CartesianConnectivity(dim_source, self._value.value + offset, dual_dim)
        else:
            assert isinstance(self._value, float)
            if self._value == 0.5:
                offset = 0
            elif self._value == -0.5:
                offset = 1
            else:
                assert False
            dual_dim, dir_offset = self._get_dual_dim(dim_source)
            return common.CartesianConnectivity(dim_source, offset + dir_offset, dual_dim)


half = Half()

I = TopologicalDimension("I")
î = TopologicalDimension("î")

J = TopologicalDimension("J")
Ĵ = TopologicalDimension("Ĵ")

X = StaggeredDimension(I, î)
Y = StaggeredDimension(J, Ĵ)


@gtx.field_operator
def avg_x(q: gtx.Field[[X, Y], float]) -> gtx.Field[[X + half, X], float]:
    return 0.5 * (q(X + 0.5) + q(X - 0.5))


@gtx.field_operator
def avg_y(q: gtx.Field[[X, Y], float]) -> gtx.Field[[Y + half, Y], float]:
    return 0.5 * (q(Y + half) + q(Y - half))


@gtx.field_operator
def delta_x(fsdx: float, q: gtx.Field[[X, Y], float]) -> gtx.Field[[X + Half, Y], float]:
    return fsdx * (q(X + 0.5) - q(X - 0.5))


@gtx.field_operator
def delta_y(fsdy: float, q: gtx.Field[[X, Y], float]) -> gtx.Field[[X, Y + Half], float]:
    return fsdy * (q(Y + 0.5) - q(Y - 0.5))


@gtx.field_operator
def calc_cucvzh(
    p: gtx.Field[I, J], u: gtx.Field[î, J], v: gtx.Field[I, Ĵ], fsdx: float, fsdy: float
) -> gtx.Field[I, J]:
    cu = avg_x(p) * u
    cv = avg_y(p) * v
    h = p + 0.5 * (avg_x(u * u) + avg_y(v * v))
    z = (
        0.25 * (-delta_x(fsdx, v) + delta_y(fsdy, u)) / avg_x(avg_y(p))
    )  # TODO: why is the sign wrong?
    return cu, cv, z, h


def calculate_gt4py(u, v, p, fsdx, fsdy):
    M = u.shape[0] - 1
    N = u.shape[1] - 1

    u_field = gtx.as_field(common.domain({î: (-1, M), J: N + 1}), u)
    v_field = gtx.as_field(common.domain({I: M + 1, Ĵ: (-1, N)}), v)
    p_field = gtx.as_field((I, J), p)

    cu, cv, z, h = calc_cucvzh(p_field, u_field, v_field, fsdx, fsdy)

    return cu, cv, z, h


def main():
    M = 16
    N = 16

    a = 1000000.0
    dx = 100000.0
    dy = 100000.0
    fsdx = 4.0 / dx
    fsdy = 4.0 / dy

    u, v, p = initial_conditions.initialize(M, N, dx, dy, a)

    u_ref, v_ref, p_ref = utils.read_uvp(0, "init", M, N)
    np.testing.assert_allclose(u, u_ref)
    np.testing.assert_allclose(v, v_ref)
    np.testing.assert_allclose(p, p_ref)
    print("init passed")

    cu_gt4py, cv_gt4py, z_gt4py, h_gt4py = calculate_gt4py(u, v, p, fsdx, fsdy)

    cu, cv, z, h = utils.read_cucvzh(0, "t100", M, N)

    np.testing.assert_allclose(cu_gt4py.asnumpy(), cu[1:, :])
    np.testing.assert_allclose(cv_gt4py.asnumpy(), cv[:, 1:])
    np.testing.assert_allclose(h_gt4py.asnumpy(), h[:-1, :-1])
    np.testing.assert_allclose(z_gt4py.asnumpy(), z[1:, 1:])

    print("t100 passed")


if __name__ == "__main__":
    main()
