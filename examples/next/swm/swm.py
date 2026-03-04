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
import functools
from time import perf_counter
import initial_conditions
import utils
import config

from operators import timestep, I, J, IJField

# from gt4py.next.program_processors.runners import jax_jit
import numpy as np

try:
    from gt4py.next.program_processors.runners.dace import run_dace_gpu_cached, run_dace_cpu_cached
except ImportError:
    run_dace_gpu_cached = None
    run_dace_cpu_cached = None

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
if run_dace_cpu_cached is not None:
    BACKENDS["dace_cpu"] = (run_dace_cpu_cached, run_dace_cpu_cached)
if run_dace_gpu_cached is not None:
    BACKENDS["dace_gpu"] = (run_dace_gpu_cached, run_dace_gpu_cached)
if cp is not None:
    BACKENDS["cupy"] = (None, cp)
if jnp is not None:
    assert jax is not None
    BACKENDS["jax"] = (None, jnp)
    BACKENDS["jax_jit"] = (jax.jit, jnp)


allocator = None

if config.backend not in BACKENDS:
    raise ValueError(
        f"Unsupported backend '{config.backend}'. Supported backends are: {list(BACKENDS.keys())}"
    )
backend, allocator = BACKENDS[config.backend]

print(f"Using backend '{getattr(backend, 'name', backend)}'.")


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


def main():
    dt0 = 0.0

    M = config.M
    N = config.N

    domain = gtx.domain({I: (-1, M + 1), J: (-1, N + 1)})

    pnew = gtx.empty(domain, dtype=dtype, allocator=allocator)
    unew = gtx.empty(domain, dtype=dtype, allocator=allocator)
    vnew = gtx.empty(domain, dtype=dtype, allocator=allocator)

    # Initialize fields
    _u, _v, _p = initial_conditions.initialize_2halo(np, M, N, config.dx, config.dy, config.a, ic=config.ic)
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

    if backend is not None and backend.__module__.startswith("jax"):
        prog = backend(functools.partial(timestep.definition, M=M, N=N))
    elif backend is not None:
        prog = timestep_program.with_backend(backend).compile(offset_provider={}, M=[M], N=[N])
        gtx.wait_for_compilation()
    else:
        prog = timestep_program

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

        if backend is not None and backend.__module__.startswith("jax"):
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
                unew=unew,
                vnew=vnew,
                pnew=pnew,
                M=M,
                N=N,
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

    if config.VAL:
        utils.final_validation(
            u.asnumpy()[:-1, 1:],
            v.asnumpy()[1:, :-1],
            p.asnumpy()[1:, 1:],
            ITMAX=config.ITMAX,
            M=M,
            N=N,
        )

    if config.VIS:
        import os
        anim_path = os.path.join(os.path.dirname(__file__), "swm_animation.mp4")
        utils.create_animation(output_path=anim_path, fps=20)


if __name__ == "__main__":
    main()
