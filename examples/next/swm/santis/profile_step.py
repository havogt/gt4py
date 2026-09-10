# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Run a few one-step calls of the model for an external profiler (nsys / ncu).

python profile_step.py MODE SIZE CALLS     MODE in ref, fwd (padded 1x1), grad_remat, ref_grad_remat
"""

import os
import sys

import jax

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import scaling_bench as sb  # noqa: E402
from halo_transports import Layout, get_transport  # noqa: E402

mode, size, calls = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
layout = Layout(size, size, 1, 1)
transport = None if mode.startswith("ref") else get_transport("padded")
fn, args = sb._case(transport, layout, mode, 1)
jax.block_until_ready(fn(*args))
for _ in range(calls):
    jax.block_until_ready(fn(*args))
print(f"done: {mode} {size}x{size}, {calls} timed calls of one step")
