# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import array_api_compat


def initialize_interior(xp, M, N, dx, dy, a, ic="default"):
    pi = 4.0 * xp.arctan(1.0)
    tpi = 2.0 * pi
    d_i = tpi / M
    d_j = tpi / N
    el = N * dx
    pcf = (pi * pi * a * a) / (el * el)

    if ic == "default":
        return _ic_default(xp, M, N, dx, dy, a, pi, tpi, d_i, d_j, el, pcf)
    elif ic == "dam_break":
        return _ic_dam_break(xp, M, N, dx, dy)
    elif ic == "colliding_vortices":
        return _ic_colliding_vortices(xp, M, N, dx, dy, a)
    elif ic == "shear":
        return _ic_shear(xp, M, N, dx, dy)
    elif ic == "gravity_wave":
        return _ic_gravity_wave(xp, M, N, dx, dy)
    else:
        raise ValueError(f"Unknown initial condition: {ic!r}")


def _ic_default(xp, M, N, dx, dy, a, pi, tpi, d_i, d_j, el, pcf):
    """Original balanced sine-wave stream function."""
    psi = (
        a
        * xp.sin((xp.arange(0, M + 1).reshape(-1, 1) + 0.5) * d_i)
        * xp.sin((xp.arange(0, N + 1) + 0.5) * d_j)
    )
    p = (
        pcf
        * (xp.cos(2.0 * xp.arange(0, M).reshape(-1, 1) * d_i) + xp.cos(2.0 * xp.arange(0, N) * d_j))
        + 50000.0
    )

    u = -(psi[1:, 1:] - psi[1:, :-1]) / dy
    v = (psi[1:, 1:] - psi[:-1, 1:]) / dx

    return u, v, p


def _ic_dam_break(xp, M, N, dx, dy):
    """Circular dam break: tall column collapses into radial waves."""
    p0 = 50000.0
    cx, cy = M / 2.0, N / 2.0
    radius = min(M, N) / 6.0

    ii = xp.arange(M).reshape(-1, 1)
    jj = xp.arange(N)
    dist = xp.sqrt((ii - cx) ** 2 + (jj - cy) ** 2)

    bump = p0 * 0.1 * xp.exp(-0.5 * (dist / radius) ** 2)
    p = p0 + bump
    u = xp.zeros((M, N), dtype=xp.float64)
    v = xp.zeros((M, N), dtype=xp.float64)

    return u, v, p


def _ic_colliding_vortices(xp, M, N, dx, dy, a):
    """Two counter-rotating vortices that collide and produce complex flow."""
    pi = 4.0 * xp.arctan(1.0)
    p0 = 50000.0

    ii = xp.arange(M + 1).reshape(-1, 1)
    jj = xp.arange(N + 1)

    # Two vortex centres offset left and right
    cx1, cy1 = M * 0.3, N * 0.5
    cx2, cy2 = M * 0.7, N * 0.5
    r = min(M, N) / 5.0

    dist1 = xp.sqrt((ii - cx1) ** 2 + (jj - cy1) ** 2)
    dist2 = xp.sqrt((ii - cx2) ** 2 + (jj - cy2) ** 2)

    psi = a * (xp.exp(-0.5 * (dist1 / r) ** 2) - xp.exp(-0.5 * (dist2 / r) ** 2))

    u = -(psi[1:, 1:] - psi[1:, :-1]) / dy
    v = (psi[1:, 1:] - psi[:-1, 1:]) / dx

    # Pressure from approximate geostrophic balance
    ii_p = xp.arange(M).reshape(-1, 1)
    jj_p = xp.arange(N)
    dist1_p = xp.sqrt((ii_p - cx1) ** 2 + (jj_p - cy1) ** 2)
    dist2_p = xp.sqrt((ii_p - cx2) ** 2 + (jj_p - cy2) ** 2)
    p = p0 + 500.0 * (xp.exp(-(dist1_p / r) ** 2) + xp.exp(-(dist2_p / r) ** 2))

    return u, v, p


def _ic_shear(xp, M, N, dx, dy):
    """Shear flow with perturbation — develops Kelvin-Helmholtz-like rolls."""
    pi = 4.0 * xp.arctan(1.0)
    p0 = 50000.0

    ii = xp.arange(M).reshape(-1, 1)
    jj = xp.arange(N)

    # Eastward jet in top half, westward in bottom half (smooth tanh profile)
    u_max = 20.0
    u = xp.broadcast_to(u_max * xp.tanh(6.0 * (jj / N - 0.5)), (M, N)).copy()

    # Small sinusoidal perturbation in v to trigger instability
    v = 2.0 * xp.sin(4.0 * pi * ii / M) * xp.exp(-50.0 * (jj / N - 0.5) ** 2)

    # Flat pressure with small perturbation matching the shear
    p = p0 + 200.0 * xp.cos(4.0 * pi * ii / M) * xp.exp(-50.0 * (jj / N - 0.5) ** 2)

    return u, v, p


def _ic_gravity_wave(xp, M, N, dx, dy):
    """Radial gravity wave: unbalanced height pulse with no initial velocity."""
    p0 = 50000.0
    cx, cy = M / 2.0, N / 2.0
    radius = min(M, N) / 8.0

    ii = xp.arange(M).reshape(-1, 1)
    jj = xp.arange(N)
    dist = xp.sqrt((ii - cx) ** 2 + (jj - cy) ** 2)

    # Sharper, taller bump than dam_break — produces clean expanding rings
    bump = p0 * 0.2 * xp.exp(-(dist / radius) ** 2)
    p = p0 + bump
    u = xp.zeros((M, N), dtype=xp.float64)
    v = xp.zeros((M, N), dtype=xp.float64)

    return u, v, p


def apply_periodic_halo(arr, top=0, bottom=0, left=0, right=0):
    """Apply periodic (wrap-around) halo padding to an array.

    Parameters
    ----------
    arr : array
        Input array to pad
    top : int
        Number of rows to add at the top (from bottom of array)
    bottom : int
        Number of rows to add at the bottom (from top of array)
    left : int
        Number of columns to add at the left (from right of array)
    right : int
        Number of columns to add at the right (from left of array)
    """
    xp = array_api_compat.array_namespace(arr)

    # Build vertical padding from the original array before any modification
    parts_v = []
    if top > 0:
        parts_v.append(arr[-top:, :])
    parts_v.append(arr)
    if bottom > 0:
        parts_v.append(arr[:bottom, :])
    arr = xp.concatenate(parts_v, axis=0)

    # Build horizontal padding from the vertically-padded array
    parts_h = []
    if left > 0:
        parts_h.append(arr[:, -left:])
    parts_h.append(arr)
    if right > 0:
        parts_h.append(arr[:, :right])
    arr = xp.concatenate(parts_h, axis=1)

    return arr


def initialize(xp, M, N, dx, dy, a, ic="default"):
    u, v, p = initialize_interior(xp, M, N, dx, dy, a, ic=ic)

    # Apply staggered 1-halo padding
    u = apply_periodic_halo(u, top=1, right=1)
    v = apply_periodic_halo(v, bottom=1, left=1)
    p = apply_periodic_halo(p, bottom=1, right=1)

    return u, v, p


def initialize_2halo(xp, M, N, dx, dy, a, ic="default"):
    u, v, p = initialize_interior(xp, M, N, dx, dy, a, ic=ic)
    return (
        apply_periodic_halo(u, 1, 1, 1, 1),
        apply_periodic_halo(v, 1, 1, 1, 1),
        apply_periodic_halo(p, 1, 1, 1, 1),
    )
