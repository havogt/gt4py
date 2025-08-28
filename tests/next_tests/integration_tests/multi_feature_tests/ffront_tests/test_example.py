# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

import gt4py.next as gtx
import dataclasses

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import cartesian_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)
from gt4py.next.type_system import type_specifications as ts

pytestmark = pytest.mark.uses_cartesian_shift

# TODO move to a proper location

# IDim = gtx.Dimension("i")
# JDim = gtx.Dimension("j")

K = gtx.Dimension("K")
Khalf = gtx.Dimension("Khalf")
κ = gtx.Dimension("κ")
Z = gtx.Dimension("Z")


@gtx.field_operator
def interpolate_K_to_Khalf(f: gtx.Field[[K], gtx.float32]) -> gtx.Field[[Khalf], gtx.float32]:
    return 0.5 * (f(K - 1) + f(K))


@gtx.field_operator
def interpolate_Khalf_to_K(f: gtx.Field[[Khalf], gtx.float32]) -> gtx.Field[[K], gtx.float32]:
    return 0.5 * (f(Khalf) + f(Khalf + 1))


@gtx.field_operator
def interpolate_K_to_Khalf(f: gtx.Field[[K], gtx.float32]) -> gtx.Field[[Khalf], gtx.float32]:
    return 0.5 * (f(K - 1 / 2) + f(K + 1 / 2))


@gtx.field_operator
def interpolate_Khalf_to_K(f: gtx.Field[[Khalf], gtx.float32]) -> gtx.Field[[K], gtx.float32]:
    return 0.5 * (f(Khalf - 1 / 2) + f(Khalf + 1 / 2))


@gtx.field_operator
def diff_K(f: gtx.Field[[K], gtx.float32]) -> gtx.Field[[κ], gtx.float32]:
    return 0.5 * (f(K + 1 / 2) - f(K - 1 / 2))


@gtx.field_operator
def diff_κ(f: gtx.Field[[κ], gtx.float32]) -> gtx.Field[[K], gtx.float32]:
    return 0.5 * (f(κ + 1 / 2) - f(κ - 1 / 2))


@gtx.field_operator
def diff(f: gtx.Field[[Z], gtx.float32]) -> gtx.Field[gtx.Dims[Half[Z]], gtx.float32]:
    return 0.5 * (f(Z + 1 / 2) - f(Z - 1 / 2))


@gtx.program
def foo_program(
    vel: Velocity,
    out: gtx.Field[[IDim, JDim], gtx.float32],
) -> gtx.Field[[IDim, JDim], gtx.float32]:
    foo(vel, out=out)


def test_named_tuple_like_constructed_outside(cartesian_case):
    vel = cases.allocate(cartesian_case, foo_program, "vel")()
    out = cases.allocate(cartesian_case, foo_program, "out")()

    cases.verify(
        cartesian_case,
        foo_program,
        vel,
        out,
        inout=out,
        ref=(vel.u + vel.v),
    )
