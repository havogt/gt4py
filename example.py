# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from gt4py import next as gtx


I = gtx.Dimension("I")
J = gtx.Dimension("J")
K = gtx.Dimension("K")

IJKField = gtx.Field[gtx.Dims[I, J, K], float]


def legacy(phi: np.ndarray, eps: float, i: int, j: int, k: int) -> float:
    aT1r = (
        13.0 / 12.0 * (phi[i - 2, j, k] - 2.0 * phi[i - 1, j, k] + phi[i, j, k]) ** 2.0
        + 1.0 / 4.0 * (phi[i - 2, j, k] - 4.0 * phi[i - 1, j, k] + 3.0 * phi[i, j, k]) ** 2.0
    )
    aT2r = (
        13.0 / 12.0 * (phi[i - 1, j, k] - 2.0 * phi[i, j, k] + phi[i + 1, j, k]) ** 2.0
        + 1.0 / 4.0 * (phi[i - 1, j, k] - phi[i + 1, j, k]) ** 2.0
    )
    aT3r = (
        13.0 / 12.0 * (phi[i, j, k] - 2.0 * phi[i + 1, j, k] + phi[i + 2, j, k]) ** 2.0
        + 1.0 / 4.0 * (3.0 * phi[i, j, k] - 4.0 * phi[i + 1, j, k] + phi[i + 2, j, k]) ** 2.0
    )

    aT1r = 1.0 / 10.0 / (eps + aT1r) ** 2.0
    aT2r = 6.0 / 10.0 / (eps + aT2r) ** 2.0
    aT3r = 3.0 / 10.0 / (eps + aT3r) ** 2.0

    fT1r = 2.0 / 6.0 * phi[i - 2, j, k] - 7.0 / 6.0 * phi[i - 1, j, k] + 11.0 / 6.0 * phi[i, j, k]
    fT2r = -1.0 / 6.0 * phi[i - 1, j, k] + 5.0 / 6.0 * phi[i, j, k] + 2.0 / 6.0 * phi[i + 1, j, k]
    fT3r = 2.0 / 6.0 * phi[i, j, k] + 5.0 / 6.0 * phi[i + 1, j, k] - 1.0 / 6.0 * phi[i + 2, j, k]

    a1r = aT1r / (aT1r + aT2r + aT3r)
    a2r = aT2r / (aT1r + aT2r + aT3r)
    a3r = aT3r / (aT1r + aT2r + aT3r)
    frx = a1r * fT1r + a2r * fT2r + a3r * fT3r
    return frx


def reference(phi: np.ndarray, eps: float) -> np.ndarray:
    result = np.empty_like(phi)
    nx, ny, nz = result.shape
    for i in range(2, nx - 2):
        for j in range(ny):
            for k in range(nz):
                result[i, j, k] = legacy(phi, eps, i, j, k)
    return result


# == v1 ==


@gtx.field_operator
def naive(phi: IJKField, eps: float) -> IJKField:
    aT1r = (
        13.0 / 12.0 * (phi(I - 2) - 2.0 * phi(I - 1) + phi) ** 2.0
        + 1.0 / 4.0 * (phi(I - 2) - 4.0 * phi(I - 1) + 3.0 * phi) ** 2.0
    )
    aT2r = (
        13.0 / 12.0 * (phi(I - 1) - 2.0 * phi + phi(I + 1)) ** 2.0
        + 1.0 / 4.0 * (phi(I - 1) - phi(I + 1)) ** 2.0
    )
    aT3r = (
        13.0 / 12.0 * (phi - 2.0 * phi(I + 1) + phi(I + 2)) ** 2.0
        + 1.0 / 4.0 * (3.0 * phi - 4.0 * phi(I + 1) + phi(I + 2)) ** 2.0
    )

    aT1r = 1.0 / 10.0 / (eps + aT1r) ** 2.0
    aT2r = 6.0 / 10.0 / (eps + aT2r) ** 2.0
    aT3r = 3.0 / 10.0 / (eps + aT3r) ** 2.0

    fT1r = 2.0 / 6.0 * phi(I - 2) - 7.0 / 6.0 * phi(I - 1) + 11.0 / 6.0 * phi
    fT2r = -1.0 / 6.0 * phi(I - 1) + 5.0 / 6.0 * phi + 2.0 / 6.0 * phi(I + 1)
    fT3r = 2.0 / 6.0 * phi + 5.0 / 6.0 * phi(I + 1) - 1.0 / 6.0 * phi(I + 2)

    a1r = aT1r / (aT1r + aT2r + aT3r)
    a2r = aT2r / (aT1r + aT2r + aT3r)
    a3r = aT3r / (aT1r + aT2r + aT3r)
    frx = a1r * fT1r + a2r * fT2r + a3r * fT3r
    return frx


# == v2 ==


@gtx.field_operator
def aT1r(phi: IJKField, eps: float) -> IJKField:
    aT1r = (
        13.0 / 12.0 * (phi(I - 2) - 2.0 * phi(I - 1) + phi) ** 2.0
        + 1.0 / 4.0 * (phi(I - 2) - 4.0 * phi(I - 1) + 3.0 * phi) ** 2.0
    )
    aT1r = 1.0 / 10.0 / (eps + aT1r) ** 2.0
    return aT1r


@gtx.field_operator
def aT2r(phi: IJKField, eps: float) -> IJKField:
    aT2r = (
        13.0 / 12.0 * (phi(I - 1) - 2.0 * phi + phi(I + 1)) ** 2.0
        + 1.0 / 4.0 * (phi(I - 1) - phi(I + 1)) ** 2.0
    )
    aT2r = 6.0 / 10.0 / (eps + aT2r) ** 2.0
    return aT2r


@gtx.field_operator
def aT3r(phi: IJKField, eps: float) -> IJKField:
    aT3r = (
        13.0 / 12.0 * (phi - 2.0 * phi(I + 1) + phi(I + 2)) ** 2.0
        + 1.0 / 4.0 * (3.0 * phi - 4.0 * phi(I + 1) + phi(I + 2)) ** 2.0
    )
    aT3r = 3.0 / 10.0 / (eps + aT3r) ** 2.0
    return aT3r


@gtx.field_operator
def v2(phi: IJKField, eps: float) -> IJKField:
    aT1r_f = aT1r(phi, eps)
    aT2r_f = aT2r(phi, eps)
    aT3r_f = aT3r(phi, eps)

    fT1r = 2.0 / 6.0 * phi(I - 2) - 7.0 / 6.0 * phi(I - 1) + 11.0 / 6.0 * phi
    fT2r = -1.0 / 6.0 * phi(I - 1) + 5.0 / 6.0 * phi + 2.0 / 6.0 * phi(I + 1)
    fT3r = 2.0 / 6.0 * phi + 5.0 / 6.0 * phi(I + 1) - 1.0 / 6.0 * phi(I + 2)

    a1r = aT1r_f / (aT1r_f + aT2r_f + aT3r_f)
    a2r = aT2r_f / (aT1r_f + aT2r_f + aT3r_f)
    a3r = aT3r_f / (aT1r_f + aT2r_f + aT3r_f)
    frx = a1r * fT1r + a2r * fT2r + a3r * fT3r
    return frx


def main():
    backend = gtx.gtfn_cpu

    phi = np.fromfunction(lambda i, j, k: i + j + k, (50, 1, 1))
    eps = 1e-6
    ref = reference(phi, eps)

    phi_field = gtx.as_field(domain=(I, J, K), data=phi, allocator=backend)
    res_field = gtx.full(
        domain={I: phi.shape[0], J: phi.shape[1], K: phi.shape[2]},
        fill_value=np.nan,
        dtype=float,
        allocator=backend,
    )

    naive.with_backend(backend)(phi_field, eps, out=res_field[2:-2, :, :])

    assert np.allclose(ref[2:-2, :, :], res_field.asnumpy()[2:-2, :, :])

    res2_field = gtx.full(
        domain={I: phi.shape[0], J: phi.shape[1], K: phi.shape[2]},
        fill_value=np.nan,
        dtype=float,
        allocator=backend,
    )

    v2.with_backend(backend)(phi_field, eps, out=res2_field[2:-2, :, :])

    assert np.allclose(ref[2:-2, :, :], res2_field.asnumpy()[2:-2, :, :])


if __name__ == "__main__":
    main()
