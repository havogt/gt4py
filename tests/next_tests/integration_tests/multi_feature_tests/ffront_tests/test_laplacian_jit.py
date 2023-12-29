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
import pytest

import gt4py.next as gtx
from gt4py.next import allocators
from gt4py.next.ffront import decorator

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, Ioff, JDim, Joff, cartesian_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    fieldview_backend,
)


pytestmark = pytest.mark.uses_cartesian_shift


@gtx.field_operator(jit=True)
def lap(in_field: gtx.Field[[IDim, JDim], float]) -> gtx.Field[[IDim, JDim], float]:
    return (
        -4.0 * in_field
        + in_field(Ioff[1])
        + in_field(Joff[1])
        + in_field(Ioff[-1])
        + in_field(Joff[-1])
    )


@gtx.field_operator(jit=True)
def laplap(in_field: gtx.Field[[IDim, JDim], float]) -> gtx.Field[[IDim, JDim], float]:
    return lap(lap(in_field))


def lap_ref(inp):
    """Compute the laplacian using numpy"""
    return -4.0 * inp[1:-1, 1:-1] + inp[:-2, 1:-1] + inp[2:, 1:-1] + inp[1:-1, :-2] + inp[1:-1, 2:]


def test_ffront_lap(cartesian_case):
    if not isinstance(cartesian_case.allocator, allocators.StandardJAXCPUFieldBufferAllocator):
        pytest.skip("only jax")
    in_field = cases.allocate(cartesian_case, lap, "in_field")()
    out_field = cases.allocate(cartesian_case, lap, cases.RETURN)()

    cases.verify(
        cartesian_case,
        lap,
        in_field,
        out=out_field[1:-1, 1:-1],
        # inout=lambda *args: args[1],
        ref=lap_ref(in_field.ndarray),
    )


def test_ffront_laplap(cartesian_case):
    if not isinstance(cartesian_case.allocator, allocators.StandardJAXCPUFieldBufferAllocator):
        pytest.skip("only jax")

    in_field = cases.allocate(cartesian_case, laplap, "in_field")()
    out_field = cases.allocate(cartesian_case, laplap, cases.RETURN)()

    cases.verify(
        cartesian_case,
        laplap,
        in_field,
        out=out_field[2:-2, 2:-2],
        ref=lap_ref(lap_ref(in_field.array_ns.asarray(in_field.ndarray))),
    )
