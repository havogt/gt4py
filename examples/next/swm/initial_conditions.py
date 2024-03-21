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

import numpy as np


def initialize(M, N, dx, dy, a):
    pi = 4.0 * np.arctan(1.0)
    tpi = 2.0 * pi
    d_i = tpi / M
    d_j = tpi / N
    el = N * dx
    pcf = (pi * pi * a * a) / (el * el)

    M_LEN = M + 1
    N_LEN = N + 1

    psi = (
        a
        * np.sin((np.arange(0, M_LEN)[:, np.newaxis] + 0.5) * d_i)
        * np.sin((np.arange(0, N_LEN) + 0.5) * d_j)
    )
    p = (
        pcf
        * (
            np.cos(2.0 * np.arange(0, M_LEN)[:, np.newaxis] * d_i)
            + np.cos(2.0 * np.arange(0, N_LEN) * d_j)
        )
        + 50000.0
    )

    u = np.zeros((M_LEN, N_LEN))
    v = np.zeros((M_LEN, N_LEN))
    u[1:, :-1] = -(psi[1:, 1:] - psi[1:, :-1]) / dy
    v[:-1, 1:] = (psi[1:, 1:] - psi[:-1, 1:]) / dx

    # # Periodic Boundary conditions
    u[0, :] = u[M, :]
    v[M, 1:] = v[0, 1:]
    u[1:, N] = u[1:, 0]
    v[:, 0] = v[:, N]

    u[0, N] = u[M, 0]
    v[M, 0] = v[0, N]

    return u, v, p
