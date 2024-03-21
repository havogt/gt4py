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

import dataclasses
from typing import Any

import initial_conditions
import numpy as np
import utils

from gt4py import next as gtx
from gt4py.next import common, fbuiltins, float64


# @dataclasses.dataclass
# class FooOffset:
#     dim: common.Dimension
#     value: Any
#     mapping: dict = dataclasses.field(default_factory=dict)

#     def __call__(self, dims):
#         dim_source = [d for d in dims if d in self.dim.dims][0]
#         cart_offset = self.mapping[(dim_source, type(self.value))]
#         return common.CartesianConnectivity(
#             cart_offset.from_,
#             self.value if isinstance(self.value, int) else self.value.value,
#             cart_offset.to_,
#         )


TopologicalDimension = common.Dimension


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

    def __add__(self, value: Any):
        return dataclasses.replace(self, _value=value)

    def __sub__(self, value: Any):
        return dataclasses.replace(self, _value=-value)

    def _get_dual_dim(self, dim):
        if dim == self._d1:
            return self._d2
        else:
            assert dim == self._d2
            return self._d1

    def as_connectivity_field(self, dims):
        dim_source = [d for d in dims if d in [self._d1, self._d2]][0]
        if isinstance(self._value, int):
            return common.CartesianConnectivity(dim_source, self._value)
        elif isinstance(self._value, Half):
            return common.CartesianConnectivity(
                dim_source, self._value.value, self._get_dual_dim(dim_source)
            )
        else:
            assert isinstance(self._value, float)
            if self._value == 0.5:
                offset = 0
            elif self._value == -0.5:
                offset = 1
            else:
                assert False
            return common.CartesianConnectivity(dim_source, offset, self._get_dual_dim(dim_source))


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
def calc_cu(p: gtx.Field[I, J], u: gtx.Field[î, J]) -> gtx.Field[î, J]:
    return avg_x(p) * u


@gtx.field_operator
def avg_y(q: gtx.Field[[X, Y], float]) -> gtx.Field[[Y + half, Y], float]:
    return 0.5 * (q(Y + half) + q(Y - half))


@gtx.field_operator
def calc_cv(p: gtx.Field[I, J], v: gtx.Field[I, Ĵ]) -> gtx.Field[I, Ĵ]:
    return avg_y(p) * v


@gtx.field_operator
def calc_h(p: gtx.Field[I, J], u: gtx.Field[î, J], v: gtx.Field[I, Ĵ]) -> gtx.Field[I, J]:
    return p + 0.5 * (avg_x(u * u) + avg_y(v * v))


def calculate_gt4py(u, v, p, fsdx, fsdy):
    M = u.shape[0] - 1
    N = u.shape[1] - 1

    u_field = gtx.as_field(common.domain({î: (-1, M), J: N + 1}), u)
    v_field = gtx.as_field(common.domain({I: M + 1, Ĵ: (-1, N)}), v)
    p_field = gtx.as_field((I, J), p)

    # print(u_field)
    # print(u_field(X - 1))
    # # v_field = gtx.as_field((I, Js), v[:-1, 1:])
    # print(p_field)
    # print(p_field(X + half))
    # print(p_field(X - half))
    # print(p_field(X - half)(X + half))
    # print(p_field(X + half)(X - half))

    # exit(1)

    cu = np.zeros_like(u_field.ndarray)
    cu_field = gtx.as_field((î, J), cu[:-1, :])

    calc_cu(p_field, u_field, out=cu_field, offset_provider={})

    cv = np.zeros_like(v_field.ndarray)
    cv_field = gtx.as_field((I, Ĵ), cv[:, :-1])

    print(avg_y(p_field))
    calc_cv(p_field, v_field, out=cv_field, offset_provider={})

    print(p_field.domain)
    print(avg_x(u_field).domain)
    print(avg_y(v_field).domain)
    # h_field = gtx.as_field(common.domain({I: (0, 16), J: (0, 16)}), np.zeros((M, N)))
    # calc_h(p_field, u_field, v_field, out=h_field, offset_provider={})

    cu = np.zeros_like(u)
    cu[1:, :] = cu_field.ndarray

    cv = np.zeros_like(v)
    cv[:, 1:] = cv_field.ndarray
    # cv = np.zeros_like(v)
    # cv[:-1, 1:] = cv_field.ndarray.reshape((M, N))
    # z = np.zeros_like(u)
    # z[1:, 1:] = z_field.ndarray.reshape((M, N))
    # h = np.zeros_like(u)
    # h[:-1, :-1] = h_field.ndarray.reshape((M, N))

    return cu, cv  # , z, h


def main():
    M = 16
    N = 16
    # u = np.random.rand(M + 1, N + 1)
    # v = np.random.rand(M + 1, N + 1)
    # p = np.random.rand(M + 1, N + 1)
    a = 1000000.0
    dx = 100000.0
    dy = 100000.0
    fsdx = 4.0 / dx
    fsdy = 4.0 / dy

    u, v, p = initial_conditions.initialize(M, N, dx, dy, a)

    # u = np.fromfile("u_init.dat").reshape(M + 1, N + 1)
    # v = np.fromfile("v_init.dat").reshape(M + 1, N + 1)
    # p = np.fromfile("p_init.dat").reshape(M + 1, N + 1)

    cu_gt4py, cv_gt4py = calculate_gt4py(u, v, p, fsdx, fsdy)

    cu, cv, z, h = utils.read_cucvzh(0, "t100", M, N)

    np.testing.assert_allclose(cu_gt4py[1:, :], cu[1:, :])
    np.testing.assert_allclose(cv_gt4py[:, 1:], cv[:, 1:])

    print("t100 passed")

    # cu_step0 = np.fromfile("cu_step0.dat").reshape(M + 1, N + 1)
    # cv_step0 = np.fromfile("cv_step0.dat").reshape(M + 1, N + 1)
    # z_step0 = np.fromfile("z_step0.dat").reshape(M + 1, N + 1)
    # h_step0 = np.fromfile("h_step0.dat").reshape(M + 1, N + 1)

    # assert np.allclose(cu_step0[1:, :-1], cu_gt4py[1:, :-1])
    # assert np.allclose(cv_step0[:-1, 1:], cv_gt4py[:-1, 1:])
    # assert np.allclose(h_step0[:-1, :-1], h_gt4py[:-1, :-1])
    # # assert np.allclose(z_step0[1:, 1:], z_gt4py[1:, 1:])
    # np.testing.assert_allclose(z_step0[1:, 1:], z_gt4py[1:, 1:])


if __name__ == "__main__":
    main()
    main()
