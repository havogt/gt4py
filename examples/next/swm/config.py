# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import argparse

parser = argparse.ArgumentParser(description="Shallow Water Model")
parser.add_argument("--M", type=int, default=16, help="Number of points in the x direction")
parser.add_argument("--N", type=int, default=16, help="Number of points in the y direction")
parser.add_argument("--L_OUT", type=bool, default=True, help="a boolean for L_OUT")
parser.add_argument("--ITMAX", type=int, default=4000, help="Number of iterations")
parser.add_argument("--VAL_DEEP", type=bool, default=True, help="Do deep validation")
parser.add_argument("--backend", type=str, default="gtfn_cpu", help="Backend to use")


args = parser.parse_args()

# Initialize model parameters
backend = args.backend
M = args.M
N = args.N
M_LEN = M + 1
N_LEN = N + 1
L_OUT = args.L_OUT
VAL_DEEP = False  # args.VAL_DEEP
VIS = True
VIS_DT = 1

# Reference resolution: M=N=16 with dx=dy=100000, dt=90
# Scale dx/dy/dt to keep the domain size and CFL number fixed at any resolution
_M_REF = 16
_N_REF = 16
dx = 100000.0 * _M_REF / M
dy = 100000.0 * _N_REF / N
dt = 90.0 * min(_M_REF / M, _N_REF / N)

ITMAX = args.ITMAX if args.ITMAX != 4000 else int(4000 * max(M / _M_REF, N / _N_REF))
fsdx = 4.0 / (dx)
fsdy = 4.0 / (dy)
a = 1000000.0
alpha = 0.001

# Validation only works at reference resolution
VAL = (M == _M_REF and N == _N_REF)
