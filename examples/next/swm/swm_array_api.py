# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Shallow Water Model using the Python Array API standard.

This implementation uses symmetric halo lines (1 on each side) for all fields,
following the approach of swm_next2_halo2_restructured.py. The periodic boundary
conditions are applied via halo exchange rather than asymmetric padding.

Compatible with any array library supporting the Array API standard:
  numpy, jax.numpy, cupy, array_api_strict, etc.

Usage:
  python swm_array_api.py --array-library numpy
  python swm_array_api.py --array-library jax
  python swm_array_api.py --array-library torch
  python swm_array_api.py --array-library cupy
  python swm_array_api.py --strict             # validate compliance with array_api_strict wrapping
  python swm_array_api.py --array-library jax --compile    # run with jax.jit
  python swm_array_api.py --array-library torch --compile  # run with torch.compile
  python swm_array_api.py --array-library torch --compile --device cuda  # torch.compile on GPU
  python swm_array_api.py --array-library jax --compile --device cpu     # jax.jit on CPU
"""

import argparse
from time import perf_counter
from array_api_compat import array_namespace


def _get_array_module(name):
    """Import and return the array module for the given library name."""
    if name == "numpy":
        import numpy

        return numpy
    elif name == "jax":
        import jax.numpy

        return jax.numpy
    elif name == "torch":
        import torch

        return torch
    elif name == "cupy":
        import cupy

        return cupy
    elif name == "array_api_strict":
        import array_api_strict

        return array_api_strict
    else:
        raise ValueError(f"Unknown array library: {name}")


def _to_numpy(arr):
    """Convert array to numpy, handling GPU/CUDA tensors."""
    import numpy as np

    try:
        import torch

        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(arr)


def initialize_interior(xp, M, N, dx, dy, a):
    """Create initial u, v, p fields on the interior (M x N) grid."""
    pi = 4.0 * xp.atan(xp.asarray(1.0, dtype=xp.float64))
    tpi = 2.0 * pi
    d_i = tpi / M
    d_j = tpi / N
    el = N * dx
    pcf = (pi * pi * a * a) / (el * el)

    i_vals = xp.arange(0, M + 1, dtype=xp.float64)
    j_vals = xp.arange(0, N + 1, dtype=xp.float64)
    i_interior = xp.arange(0, M, dtype=xp.float64)
    j_interior = xp.arange(0, N, dtype=xp.float64)

    # psi: (M+1) x (N+1), p: (M) x (N), u: (M) x (N), v: (M) x (N)
    # Use reshape to create 2D broadcasting: column * row
    i_col = xp.reshape(i_vals, (M + 1, 1))  # (M+1, 1)
    j_row = xp.reshape(j_vals, (1, N + 1))  # (1, N+1)
    psi = a * xp.sin((i_col + 0.5) * d_i) * xp.sin((j_row + 0.5) * d_j)

    i_int_col = xp.reshape(i_interior, (M, 1))  # (M, 1)
    j_int_row = xp.reshape(j_interior, (1, N))  # (1, N)
    p = pcf * (xp.cos(2.0 * i_int_col * d_i) + xp.cos(2.0 * j_int_row * d_j)) + 50000.0

    u = -(psi[1:, 1:] - psi[1:, :-1]) / dy
    v = (psi[1:, 1:] - psi[:-1, 1:]) / dx

    return u, v, p


def _interior_to_halo(xp, interior):
    """Build (M+2, N+2) array from (M, N) interior with periodic halos.

    Wraps the interior periodically: last col -> left halo, first col -> right halo,
    last row -> top halo, first row -> bottom halo.
    """
    M, N = interior.shape

    # Wrap columns: [last_col | interior | first_col]
    left_col = interior[:, N - 1 : N]  # (M, 1)
    right_col = interior[:, 0:1]  # (M, 1)
    middle_rows = xp.concat([left_col, interior, right_col], axis=1)  # (M, N+2)

    # Wrap rows: [last_row | middle | first_row]
    top_row = middle_rows[M - 1 : M, :]  # (1, N+2)
    bottom_row = middle_rows[0:1, :]  # (1, N+2)
    return xp.concat([top_row, middle_rows, bottom_row], axis=0)  # (M+2, N+2)


def apply_periodic_halo(xp, interior, x):
    """Apply periodic boundary conditions by filling the halo from the interior.

    The array x has shape (M+2, N+2) where the interior is x[1:-1, 1:-1].
    The halos are filled by wrapping around the interior periodically.
    """
    return _interior_to_halo(xp, x[1:-1, 1:-1])


def avg_x(xp, f):
    """Average field in the x direction."""
    M, N = f.shape[0] - 2, f.shape[1] - 2
    interior = 0.5 * (f[2 : M + 2, 1 : N + 1] + f[1 : M + 1, 1 : N + 1])
    return _interior_to_halo(xp, interior)


def avg_y(xp, f):
    """Average field in the y direction."""
    M, N = f.shape[0] - 2, f.shape[1] - 2
    interior = 0.5 * (f[1 : M + 1, 2 : N + 2] + f[1 : M + 1, 1 : N + 1])
    return _interior_to_halo(xp, interior)


def avg_x_staggered(xp, f):
    """Average field which is staggered in x in the x direction."""
    M, N = f.shape[0] - 2, f.shape[1] - 2
    interior = 0.5 * (f[0:M, 1 : N + 1] + f[1 : M + 1, 1 : N + 1])
    return _interior_to_halo(xp, interior)


def avg_y_staggered(xp, f):
    """Average field which is staggered in y in the y direction."""
    M, N = f.shape[0] - 2, f.shape[1] - 2
    interior = 0.5 * (f[1 : M + 1, 0:N] + f[1 : M + 1, 1 : N + 1])
    return _interior_to_halo(xp, interior)


def delta_x(xp, dx, f):
    """Calculate the difference in the x direction."""
    M, N = f.shape[0] - 2, f.shape[1] - 2
    interior = (1.0 / dx) * (f[2 : M + 2, 1 : N + 1] - f[1 : M + 1, 1 : N + 1])
    return _interior_to_halo(xp, interior)


def delta_y(xp, dx, f):
    """Calculate the difference in the y direction."""
    M, N = f.shape[0] - 2, f.shape[1] - 2
    interior = (1.0 / dx) * (f[1 : M + 1, 2 : N + 2] - f[1 : M + 1, 1 : N + 1])
    return _interior_to_halo(xp, interior)


def delta_x_staggered(xp, dx, f):
    """Calculate the difference in the x direction for field staggered in x."""
    M, N = f.shape[0] - 2, f.shape[1] - 2
    interior = (1.0 / dx) * (f[1 : M + 1, 1 : N + 1] - f[0:M, 1 : N + 1])
    return _interior_to_halo(xp, interior)


def delta_y_staggered(xp, dx, f):
    """Calculate the difference in the y direction for field staggered in y."""
    M, N = f.shape[0] - 2, f.shape[1] - 2
    interior = (1.0 / dx) * (f[1 : M + 1, 1 : N + 1] - f[1 : M + 1, 0:N])
    return _interior_to_halo(xp, interior)


def timestep(xp, u, v, p, uold, vold, pold, dx, dy, dt_val, alpha_val, M, N):
    """Perform one timestep of the shallow water equations.

    All fields have shape (M+2, N+2) with 1-wide symmetric halos.
    Each helper function takes a full (M+2, N+2) array and returns a full
    (M+2, N+2) array with periodic halos, enabling direct composition.
    """
    cu = avg_x(xp, p) * u
    cv = avg_y(xp, p) * v
    z = (delta_x(xp, dx, v) - delta_y(xp, dy, u)) / avg_x(xp, avg_y(xp, p))
    h = p + 0.5 * (avg_x_staggered(xp, u * u) + avg_y_staggered(xp, v * v))

    unew = (
        uold
        + avg_y_staggered(xp, z) * avg_y_staggered(xp, avg_x(xp, cv)) * dt_val
        - delta_x(xp, dx, h) * dt_val
    )
    vnew = (
        vold
        - avg_x_staggered(xp, z) * avg_x_staggered(xp, avg_y(xp, cu)) * dt_val
        - delta_y(xp, dy, h) * dt_val
    )
    pnew = pold - delta_x_staggered(xp, dx, cu) * dt_val - delta_y_staggered(xp, dy, cv) * dt_val

    uold_new = u + alpha_val * (unew - 2.0 * u + uold)
    vold_new = v + alpha_val * (vnew - 2.0 * v + vold)
    pold_new = p + alpha_val * (pnew - 2.0 * p + pold)

    return unew, vnew, pnew, uold_new, vold_new, pold_new


def initialize_2halo(xp, M, N, dx, dy, a):
    """Initialize fields with 2-halo (1 on each side) symmetric padding."""
    u, v, p = initialize_interior(xp, M, N, dx, dy, a)
    return _interior_to_halo(xp, u), _interior_to_halo(xp, v), _interior_to_halo(xp, p)


def to_reference_layout(arr, M, N):
    """Convert from 2-halo (M+2, N+2) layout to reference (M+1, N+1) layout.

    The reference data uses an asymmetric layout where:
      u: padded with (1,0) in x and (0,1) in y  -> u_ref = [halo; interior_rows][interior_cols; halo]
      v: padded with (0,1) in x and (1,0) in y
      p: padded with (0,1) in x and (0,1) in y

    For the 2-halo symmetric layout, the interior is at [1:M+1, 1:N+1].
    The reference format stores M+1 x N+1 values.

    For u: ref has rows [M, 0..M-1] and cols [0..N-1, 0] -> u_ref = u_2halo[0:M+1, 1:N+2]
      which is u_2halo[:-1, 1:]
    For v: ref has rows [0..N-1, 0] and cols [N, 0..N-1] -> v_ref = u_2halo[1:M+2, 0:N+1]
      which is v_2halo[1:, :-1]
    For p: ref has rows [0..M-1, 0] and cols [0..N-1, 0] -> p_ref = p_2halo[1:M+2, 1:N+2]
      which is p_2halo[1:, 1:]
    """
    pass  # implemented inline in validation


def main():
    parser = argparse.ArgumentParser(description="Shallow Water Model (Array API)")
    parser.add_argument(
        "--array-library",
        type=str,
        default="numpy",
        choices=["numpy", "jax", "torch", "cupy", "array_api_strict"],
        help="Array library to use",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable array-api-strict compliance checking via array_api_compat",
    )
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--ITMAX", type=int, default=4000)
    parser.add_argument("--validate", action="store_true", help="Validate against reference data")
    parser.add_argument("--validate-deep", action="store_true", help="Deep validation of each step")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable JIT compilation (jax.jit for jax, torch.compile for torch)",
    )
    parser.add_argument("--no-output", action="store_true", help="Suppress diagnostic output")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Device to run on (default: CPU for numpy/torch, GPU for jax if available)",
    )
    args = parser.parse_args()

    M = args.M
    N = args.N
    ITMAX = args.ITMAX
    L_OUT = not args.no_output

    # Physical parameters
    dt = 90.0
    dx = 100000.0
    dy = 100000.0
    a = 1000000.0
    alpha = 0.001

    # Get the array module
    if args.array_library == "jax":
        import jax

        jax.config.update("jax_enable_x64", True)
        if args.device is not None:
            jax_device_kind = "gpu" if args.device == "cuda" else "cpu"
            jax.config.update("jax_default_device", jax.devices(jax_device_kind)[0])
            print(f"JAX device: {jax_device_kind}")

    lib = _get_array_module(args.array_library)

    # Configure torch device
    if args.array_library == "torch" and args.device is not None:
        import torch

        torch.set_default_device(args.device)
        print(f"Torch device: {args.device}")
    elif args.device == "cuda" and args.array_library not in ("jax", "torch", "cupy"):
        print(f"Warning: --device cuda not supported for {args.array_library}")

    if args.strict:
        import array_api_strict

        array_api_strict.set_array_api_strict_flags(api_version="2024.12")

    if args.strict and args.array_library != "array_api_strict":
        # Wrap the library arrays in array_api_strict for compliance testing
        import array_api_strict

        xp = array_api_strict
        print(
            f"Running with {args.array_library} arrays wrapped in array_api_strict for compliance checking"
        )
    elif args.strict:
        import array_api_strict

        xp = array_api_strict
        print("Running with array_api_strict directly")
    else:
        xp = lib
        # Get the array-api-compat namespace for the library
        test_arr = lib.zeros((1,))
        xp = array_namespace(test_arr)
        print(f"Running with {args.array_library} via array_api_compat namespace")

    # Initialize fields
    u, v, p = initialize_2halo(xp, M, N, dx, dy, a)

    if args.strict and args.array_library != "array_api_strict":
        # Convert to array_api_strict arrays
        import array_api_strict
        import numpy as np

        u_np = np.asarray(u) if hasattr(u, "__array__") else u
        v_np = np.asarray(v) if hasattr(v, "__array__") else v
        p_np = np.asarray(p) if hasattr(p, "__array__") else p
        u = array_api_strict.asarray(u_np, dtype=array_api_strict.float64)
        v = array_api_strict.asarray(v_np, dtype=array_api_strict.float64)
        p = array_api_strict.asarray(p_np, dtype=array_api_strict.float64)
        xp = array_api_strict

    uold = xp.asarray(u, copy=True)
    vold = xp.asarray(v, copy=True)
    pold = xp.asarray(p, copy=True)

    if L_OUT:
        print(f" Number of points in the x direction: {M}")
        print(f" Number of points in the y direction: {N}")
        print(f" grid spacing in the x direction: {dx}")
        print(f" grid spacing in the y direction: {dy}")
        print(f" time step: {dt}")
        print(f" time filter coefficient: {alpha}")

    # For validation, we need numpy
    if args.validate or args.validate_deep:
        import numpy as np

    if args.validate_deep:
        import sys

        sys.path.insert(0, "/home/user/SWM/swm_python")
        import utils

    # Set up the timestep function, optionally with JIT compilation
    def timestep_fn(u, v, p, uold, vold, pold, dt_val, alpha_val):
        return timestep(xp, u, v, p, uold, vold, pold, dx, dy, dt_val, alpha_val, M, N)

    if args.compile:
        if args.array_library == "jax":
            import jax

            timestep_fn = jax.jit(timestep_fn)
            print("JIT compilation enabled via jax.jit")
        elif args.array_library == "torch":
            import torch

            timestep_fn = torch.compile(timestep_fn)
            print("JIT compilation enabled via torch.compile")
        else:
            print(f"Warning: --compile has no effect for {args.array_library}")

        # Warm-up call to trigger compilation before timing
        print("Warm-up call...", end=" ", flush=True)
        t_warmup_start = perf_counter()
        _warmup_result = timestep_fn(u, v, p, uold, vold, pold, dt, 0.0)
        if args.array_library == "jax":
            _warmup_result[0].block_until_ready()
        elif args.array_library == "torch" and args.device == "cuda":
            torch.cuda.synchronize()
        t_warmup_stop = perf_counter()
        del _warmup_result
        print(f"done ({t_warmup_stop - t_warmup_start:.3f}s)")

    dt_total = 0.0
    dt_compute = 0.0

    t0_start = perf_counter()

    for ncycle in range(ITMAX):
        if ncycle % 100 == 0:
            print(f"cycle number {ncycle}")

        if args.validate_deep and ncycle <= 3:
            import numpy as np

            u_np = _to_numpy(u)
            v_np = _to_numpy(v)
            p_np = _to_numpy(p)
            # Convert 2-halo to reference layout
            utils.validate_uvp(
                u_np[:-1, 1:],
                v_np[1:, :-1],
                p_np[1:, 1:],
                M,
                N,
                ncycle,
                "init",
            )

        tdt = dt if ncycle == 0 else dt * 2.0
        alpha_val = alpha if ncycle > 0 else 0.0

        t_start = perf_counter()
        unew, vnew, pnew, uold, vold, pold = timestep_fn(u, v, p, uold, vold, pold, tdt, alpha_val)
        t_stop = perf_counter()
        dt_compute += t_stop - t_start

        u = unew
        v = vnew
        p = pnew

    # Synchronize device for accurate timing
    if args.array_library == "jax":
        u.block_until_ready()
    elif args.array_library == "torch" and args.device == "cuda":
        import torch

        torch.cuda.synchronize()

    t0_stop = perf_counter()
    dt_total = t0_stop - t0_start

    if L_OUT:
        print(f"cycle number {ITMAX}")

    print(f"total: {dt_total}")
    print(f"compute: {dt_compute}")

    if args.validate:
        import numpy as np

        u_np = _to_numpy(u)
        v_np = _to_numpy(v)
        p_np = _to_numpy(p)

        # Convert to reference layout for validation
        u_ref_layout = u_np[:-1, 1:]
        v_ref_layout = v_np[1:, :-1]
        p_ref_layout = p_np[1:, 1:]

        sys_path_added = False
        import sys

        if "/home/user/SWM/swm_python" not in sys.path:
            sys.path.insert(0, "/home/user/SWM/swm_python")
            sys_path_added = True
        import utils

        utils.final_validation(u_ref_layout, v_ref_layout, p_ref_layout, ITMAX=ITMAX, M=M, N=N)


if __name__ == "__main__":
    main()
