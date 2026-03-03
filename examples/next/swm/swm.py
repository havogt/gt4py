# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
This version uses 2 halo lines (1 on each side)

e.g. for M=3, N=3, with 'x' = interior, '0' = periodic halo, the grid is:

for all fields
0 0 0 0 0
0 x x x 0
0 x x x 0
0 x x x 0
0 0 0 0 0
"""

from gt4py import next as gtx
from gt4py.next.experimental import concat_where
from time import perf_counter
import initial_conditions
import utils
import config
from gt4py.next.program_processors.runners.dace import run_dace_gpu_cached, run_dace_cpu_cached

# from gt4py.next.program_processors.runners import jax_jit
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import jax.numpy as jnp
    import jax
except ImportError:
    jnp = None
    jax = None

dtype = gtx.float64

BACKENDS = {
    "gtfn_gpu": (gtx.gtfn_gpu, gtx.gtfn_gpu),
    "gtfn_cpu": (gtx.gtfn_cpu, gtx.gtfn_cpu),
    "dace_gpu": (run_dace_gpu_cached, run_dace_gpu_cached),
    "dace_cpu": (run_dace_cpu_cached, run_dace_cpu_cached),
    "numpy": (None, np),
}
if cp is not None:
    BACKENDS["cupy"] = (None, cp)
if jnp is not None:
    assert jax is not None
    BACKENDS["jax"] = (None, jnp)
    BACKENDS["jax_jit"] = (jax.jit(static_argnames=["M", "N"]), jnp)


allocator = None

if config.backend not in BACKENDS:
    raise ValueError(
        f"Unsupported backend '{config.backend}'. Supported backends are: {list(BACKENDS.keys())}"
    )
backend, allocator = BACKENDS[config.backend]

print(f"Using backend '{getattr(backend, 'name', backend)}'.")

I = gtx.Dimension("I")
J = gtx.Dimension("J")

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
    f = concat_where(I <= -1, f(I + M), f)
    f = concat_where(I >= M, f(I - M), f)
    f = concat_where(J <= -1, f(J + N), f)
    f = concat_where(J >= N, f(J - N), f)
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

    # because GT4Py does not allow slicing and periodic halo is not a supported pattern we need to trick GT4Py to shrink the domain
    unew = make_periodic(unew, M, N) + u * 0.0
    vnew = make_periodic(vnew, M, N) + v * 0.0
    pnew = make_periodic(pnew, M, N) + p * 0.0

    # The following only works in embedded/jax.jit but is not compliant GT4Py DSL (slightly faster than the above with jax.jit)
    # unew = make_periodic(unew, M, N)[gtx.domain({I: (-1, M + 1), J: (-1, N + 1)})]
    # vnew = make_periodic(vnew, M, N)[gtx.domain({I: (-1, M + 1), J: (-1, N + 1)})]
    # pnew = make_periodic(pnew, M, N)[gtx.domain({I: (-1, M + 1), J: (-1, N + 1)})]

    # uold_new = make_periodic(uold_new, M, N)
    # vold_new = make_periodic(vold_new, M, N)
    # pold_new = make_periodic(pold_new, M, N)

    return (
        unew,
        vnew,
        pnew,
        uold_new,
        vold_new,
        pold_new,
    )


@gtx.program(backend=backend)
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
        domain={I: (-1, M + 1), J: (-1, N + 1)},
    )


def main():
    dt0 = 0.0

    M = config.M
    N = config.N

    domain = gtx.domain({I: (-1, M + 1), J: (-1, N + 1)})

    pnew = gtx.empty(domain, dtype=dtype, allocator=allocator)
    unew = gtx.empty(domain, dtype=dtype, allocator=allocator)
    vnew = gtx.empty(domain, dtype=dtype, allocator=allocator)

    # Initialize fields
    _u, _v, _p = initial_conditions.initialize_2halo(np, M, N, config.dx, config.dy, config.a)
    u = gtx.as_field(domain, _u, dtype=dtype, allocator=allocator)
    v = gtx.as_field(domain, _v, dtype=dtype, allocator=allocator)
    p = gtx.as_field(domain, _p, dtype=dtype, allocator=allocator)

    # Initial old fields
    uold = gtx.as_field(domain, _u, dtype=dtype, allocator=allocator)
    vold = gtx.as_field(domain, _v, dtype=dtype, allocator=allocator)
    pold = gtx.as_field(domain, _p, dtype=dtype, allocator=allocator)

    # Print initial conditions
    if config.L_OUT:
        print(" Number of points in the x direction: ", M)
        print(" Number of points in the y direction: ", N)
        print(" grid spacing in the x direction: ", config.dx)
        print(" grid spacing in the y direction: ", config.dy)
        print(" time step: ", config.dt)
        print(" time filter coefficient: ", config.alpha)

        print(" Initial p:\n", p[:, :].ndarray.diagonal()[1:-1])
        print(" Initial u:\n", u[:, :].ndarray.diagonal()[1:-1])
        print(" Initial v:\n", v[:, :].ndarray.diagonal()[1:-1])

    USE_PROGRAM = True

    if backend.__module__.startswith("jax"):
        prog = timestep.with_backend(backend)
    elif backend is not None:
        if USE_PROGRAM:
            prog = timestep_program.with_backend(backend).compile(offset_provider={}, M=[M], N=[N])
        else:
            prog = timestep.with_backend(backend).compile(offset_provider={})
        gtx.wait_for_compilation()
    else:
        prog = timestep_program if USE_PROGRAM else timestep

    t0_start = perf_counter()

    # Main time loop
    for ncycle in range(config.ITMAX):
        if (ncycle % 100 == 0) & (config.VIS == False):
            print(f"cycle number{ncycle}")

        if config.VAL_DEEP and ncycle <= 3:
            print("validating init")
            utils.validate_uvp(
                u.asnumpy()[:-1, 1:],
                v.asnumpy()[1:, :-1],
                p.asnumpy()[1:, 1:],
                M,
                N,
                ncycle,
                "init",
            )

        if backend.__module__.startswith("jax"):
            unew, vnew, pnew, uold, vold, pold = prog(
                u=u,
                v=v,
                p=p,
                dx=config.dx,
                dy=config.dy,
                dt=config.dt if ncycle == 0 else config.dt * 2.0,
                uold=uold,
                vold=vold,
                pold=pold,
                alpha=config.alpha if ncycle > 0 else 0.0,
                M=M,
                N=N,
            )
        elif USE_PROGRAM:
            prog(
                u=u,
                v=v,
                p=p,
                dx=config.dx,
                dy=config.dy,
                dt=config.dt if ncycle == 0 else config.dt * 2.0,
                uold=uold,
                vold=vold,
                pold=pold,
                alpha=config.alpha if ncycle > 0 else 0.0,
                unew=unew,
                vnew=vnew,
                pnew=pnew,
                M=M,
                N=N,
            )
        else:
            prog(
                u=u,
                v=v,
                p=p,
                dx=config.dx,
                dy=config.dy,
                dt=config.dt if ncycle == 0 else config.dt * 2.0,
                uold=uold,
                vold=vold,
                pold=pold,
                alpha=config.alpha if ncycle > 0 else 0.0,
                offset_provider={},
                out=(unew, vnew, pnew, uold, vold, pold),
                domain={I: (0, M), J: (0, N)},
            )

        if hasattr(u.array_ns, "cuda"):
            u.array_ns.cuda.runtime.deviceSynchronize()

        # swap x with xnew fields
        u, unew = unew, u
        v, vnew = vnew, v
        p, pnew = pnew, p

        if (config.VIS) & (ncycle % config.VIS_DT == 0):
            utils.live_plot3(
                u.asnumpy(),
                v.asnumpy(),
                p.asnumpy(),
                "ncycle: " + str(ncycle),
            )

    if cp is not None:
        cp.cuda.runtime.deviceSynchronize()
    t0_stop = perf_counter()
    dt0 = dt0 + (t0_stop - t0_start)
    # Print initial conditions
    if config.L_OUT:
        print("cycle number ", config.ITMAX)
        print(" diagonal elements of p:\n", p[:, :].ndarray.diagonal()[:-1])
        print(" diagonal elements of u:\n", u[:, :].ndarray.diagonal()[:-1])
        print(" diagonal elements of v:\n", v[:, :].ndarray.diagonal()[:-1])
    print("total: ", dt0)

    u = u[gtx.domain({I: (-1, M + 1), J: (-1, N + 1)})]
    v = v[gtx.domain({I: (-1, M + 1), J: (-1, N + 1)})]
    p = p[gtx.domain({I: (-1, M + 1), J: (-1, N + 1)})]

    if config.VAL:
        utils.final_validation(
            u.asnumpy()[:-1, 1:],
            v.asnumpy()[1:, :-1],
            p.asnumpy()[1:, 1:],
            ITMAX=config.ITMAX,
            M=M,
            N=N,
        )


if __name__ == "__main__":
    main()
