# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from gt4py import next as gtx
from gt4py.next.experimental import concat_where

I = gtx.Dimension("I")
J = gtx.Dimension("J")

dtype = gtx.float64
IJField = gtx.Field[gtx.Dims[I, J], dtype]


@gtx.field_operator
def avg_x(f: IJField):
    """Average field in the x direction."""
    return 0.5 * (f(I + 1) + f)


@gtx.field_operator
def avg_y(f: IJField):
    """Average field in the y direction."""
    return 0.5 * (f(J + 1) + f)


@gtx.field_operator
def avg_x_staggered(f: IJField):
    """Average field which is staggered in x in the x direction."""
    return 0.5 * (f(I - 1) + f)


@gtx.field_operator
def avg_y_staggered(f: IJField):
    """Average field which is staggered in y in the y direction."""
    return 0.5 * (f(J - 1) + f)


@gtx.field_operator
def delta_x(dx: dtype, f: IJField):
    """Calculate the difference in the x direction."""
    return (1.0 / dx) * (f(I + 1) - f)


@gtx.field_operator
def delta_y(dx: dtype, f: IJField):
    """Calculate the difference in the y direction."""
    return (1.0 / dx) * (f(J + 1) - f)


@gtx.field_operator
def delta_x_staggered(dx: dtype, f: IJField):
    """Calculate the difference in the x direction for field staggered in x."""
    return (1.0 / dx) * (f - f(I - 1))


@gtx.field_operator
def delta_y_staggered(dx: dtype, f: IJField):
    """Calculate the difference in the y direction for field staggered in y."""
    return (1.0 / dx) * (f - f(J - 1))


@gtx.field_operator
def make_periodic(f: IJField, M: gtx.int32, N: gtx.int32):
    """Make the field f periodic by copying values from the opposite sides."""
    f = concat_where(I == -1, f(I + M), f)
    f = concat_where(I == M, f(I - M), f)
    f = concat_where(J == -1, f(J + N), f)
    f = concat_where(J == N, f(J - N), f)
    return f


@gtx.field_operator
def timestep(
    u: IJField,
    v: IJField,
    p: IJField,
    dx: dtype,
    dy: dtype,
    dt: dtype,
    uold: IJField,
    vold: IJField,
    pold: IJField,
    alpha: dtype,
    M: gtx.int32,
    N: gtx.int32,
) -> tuple[IJField, IJField, IJField, IJField, IJField, IJField]:
    cu = avg_x(p) * u
    cv = avg_y(p) * v
    z = (delta_x(dx, v) - delta_y(dy, u)) / avg_x(avg_y(p))
    h = p + 0.5 * (avg_x_staggered(u * u) + avg_y_staggered(v * v))

    unew = uold + avg_y_staggered(z) * avg_y_staggered(avg_x(cv)) * dt - delta_x(dx, h) * dt
    vnew = vold - avg_x_staggered(z) * avg_x_staggered(avg_y(cu)) * dt - delta_y(dy, h) * dt
    pnew = pold - delta_x_staggered(dx, cu) * dt - delta_y_staggered(dy, cv) * dt

    uold_new = u + alpha * (unew - 2.0 * u + uold)
    vold_new = v + alpha * (vnew - 2.0 * v + vold)
    pold_new = p + alpha * (pnew - 2.0 * p + pold)

    unew = make_periodic(unew, M, N)
    vnew = make_periodic(vnew, M, N)
    pnew = make_periodic(pnew, M, N)

    uold_new = make_periodic(uold_new, M, N)
    vold_new = make_periodic(vold_new, M, N)
    pold_new = make_periodic(pold_new, M, N)

    return (
        unew,
        vnew,
        pnew,
        uold_new,
        vold_new,
        pold_new,
    )


@gtx.program
def timestep_program(
    u: IJField,
    v: IJField,
    p: IJField,
    dx: dtype,
    dy: dtype,
    dt: dtype,
    uold: IJField,
    vold: IJField,
    pold: IJField,
    alpha: dtype,
    unew: IJField,
    vnew: IJField,
    pnew: IJField,
    M: gtx.int32,
    N: gtx.int32,
):
    timestep(
        u=u,
        v=v,
        p=p,
        dx=dx,
        dy=dy,
        dt=dt,
        uold=uold,
        vold=vold,
        pold=pold,
        alpha=alpha,
        M=M,
        N=N,
        out=(unew, vnew, pnew, uold, vold, pold),
        domain=(
            {I: (-1, M + 1), J: (-1, N + 1)},
            {I: (-1, M + 1), J: (-1, N + 1)},
            {I: (-1, M + 1), J: (-1, N + 1)},
            {I: (0, M), J: (0, N)},
            {I: (0, M), J: (0, N)},
            {I: (0, M), J: (0, N)},
        ),
    )
