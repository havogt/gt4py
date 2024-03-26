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
def delta_x(dx: float, q: gtx.Field[[X, Y], float]) -> gtx.Field[[X + Half, Y], float]:
    return (1.0 / dx) * (q(X + 0.5) - q(X - 0.5))


@gtx.field_operator
def delta_y(dy: float, q: gtx.Field[[X, Y], float]) -> gtx.Field[[X, Y + Half], float]:
    return (1.0 / dy) * (q(Y + 0.5) - q(Y - 0.5))


@gtx.field_operator
def calc_cucvzh(
    p: gtx.Field[I, J], u: gtx.Field[î, J], v: gtx.Field[I, Ĵ], dx: float, dy: float
) -> gtx.Field[I, J]:
    cu = avg_x(p) * u
    cv = avg_y(p) * v
    h = p + 0.5 * (avg_x(u * u) + avg_y(v * v))
    z = (-delta_x(dx, v) + delta_y(dy, u)) / avg_x(avg_y(p))  # TODO: why is the sign wrong?
    return cu, cv, z, h


# @gtx.field_operator
# def calc_uvp()


def apply_periodic_boundary_u(u: gtx.Field):
    u = u[
        J(0) : J(16)
    ]  # TODO e.g. for cu we compute all J's including halo, so we need to slice the halo away
    M = u.shape[0]
    N = u.shape[1]
    res = np.empty((M + 1, N + 1))
    res[1:, :-1] = u.asnumpy()[...]
    res[0, :-1] = u.asnumpy()[-1, :]
    res[1:, N] = u.asnumpy()[:, 0]
    res[0, N] = u.asnumpy()[-1, 0]

    return gtx.as_field(gtx.domain({î: (-1, N), J: N + 1}), res)


def apply_periodic_boundary_v(v: gtx.Field):
    v = v[I(0) : I(16)]
    M = v.shape[0]
    N = v.shape[1]
    res = np.empty((M + 1, N + 1))
    res[:-1, 1:] = v.asnumpy()[...]
    res[:-1, 0] = v.asnumpy()[:, -1]
    res[M, 1:] = v.asnumpy()[0, :]
    res[M, 0] = v.asnumpy()[0, -1]

    return gtx.as_field(gtx.domain({I: M + 1, Ĵ: (-1, N)}), res)


def apply_periodic_boundary_p(h: gtx.Field):
    h = h[I(0) : I(16), J(0) : J(16)]
    M = h.shape[0]
    N = h.shape[1]
    res = np.empty((M + 1, N + 1))
    res[:-1, :-1] = h.asnumpy()[...]
    res[-1, :-1] = h.asnumpy()[0, :]
    res[:-1, -1] = h.asnumpy()[:, 0]
    res[-1, -1] = h.asnumpy()[0, 0]

    return gtx.as_field(gtx.domain({I: M + 1, J: N + 1}), res)


def apply_periodic_boundary_z(z: gtx.Field):
    z = z[I(0) : I(16), J(0) : J(16)]
    M = z.shape[0]
    N = z.shape[1]
    res = np.empty((M + 1, N + 1))
    res[1:, 1:] = z.asnumpy()[...]
    res[0, 1:] = z.asnumpy()[-1, :]
    res[1:, 0] = z.asnumpy()[:, -1]
    res[0, 0] = z.asnumpy()[-1, -1]

    return gtx.as_field(gtx.domain({î: (-1, M), Ĵ: (-1, N)}), res)

    # h[M, :] = h[0, :]
    # cv[M, 1:] = cv[0, 1:]
    # z[0, 1:] = z[M, 1:]

    # cv[:, 0] = cv[:, N]
    # h[:, N] = h[:, 0]
    # z[1:, N] = z[1:, 0]

    # cv[M, 0] = cv[0, N]
    # z[0, 0] = z[M, N]
    # h[M, N] = h[0, 0]


def calculate_unew(u, cv, z, h, dx, dt):
    return u + avg_y(z) * avg_y(avg_x(cv)) * dt + delta_x(dx, h) * dt  # TODO sign in delta_x


def calculate_pnew(p, cu, cv, dx, dy, dt):
    return p + delta_x(dx, cu) * dt + delta_y(dy, cv) * dt  # TODO signs in delta


def calculate_gt4py(u, v, p, dx, dy, dt):
    M = u.shape[0] - 1
    N = u.shape[1] - 1

    u_field = gtx.as_field(common.domain({î: (-1, M), J: N + 1}), u)
    v_field = gtx.as_field(common.domain({I: M + 1, Ĵ: (-1, N)}), v)
    p_field = gtx.as_field((I, J), p)

    cu, cv, z, h = calc_cucvzh(p_field, u_field, v_field, dx, dy)
    cu = apply_periodic_boundary_u(cu)
    cv = apply_periodic_boundary_v(cv)
    h = apply_periodic_boundary_p(h)
    z = apply_periodic_boundary_z(z)

    unew = calculate_unew(u_field, cv, z, h, dx, dt)
    pnew = calculate_pnew(p_field, cu, cv, dx, dy, dt)

    return cu, cv, z, h, unew, pnew


def main():
    M = 16
    N = 16

    a = 1000000.0
    dx = 100000.0
    dy = 100000.0
    fsdx = 4.0 / dx
    fsdy = 4.0 / dy
    dt = 90.0

    u, v, p = initial_conditions.initialize(M, N, dx, dy, a)

    u_ref, v_ref, p_ref = utils.read_uvp(0, "init", M, N)
    np.testing.assert_allclose(u, u_ref)
    np.testing.assert_allclose(v, v_ref)
    np.testing.assert_allclose(p, p_ref)
    print("init passed")

    cu_gt4py, cv_gt4py, z_gt4py, h_gt4py, unew_gt4py, pnew_gt4py = calculate_gt4py(
        u, v, p, dx, dy, dt
    )

    cu, cv, z, h = utils.read_cucvzh(0, "t100", M, N)

    np.testing.assert_allclose(cu_gt4py.asnumpy(), cu)
    np.testing.assert_allclose(cv_gt4py.asnumpy(), cv)
    np.testing.assert_allclose(h_gt4py.asnumpy(), h)
    np.testing.assert_allclose(z_gt4py.asnumpy(), z)

    print("t100 passed")

    unew, vnew, pnew = utils.read_uvp(0, "t200", M, N)
    # print(unew_gt4py)
    np.testing.assert_allclose(unew_gt4py.asnumpy(), unew[1:, :-1])
    np.testing.assert_allclose(pnew_gt4py.asnumpy(), pnew[:-1, :-1])

    print("t200 passed")


if __name__ == "__main__":
    main()
