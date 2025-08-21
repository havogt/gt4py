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
from next_tests.integration_tests.cases import IDim, JDim, cartesian_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)
from gt4py.next.type_system import type_specifications as ts

pytestmark = pytest.mark.uses_cartesian_shift

# TODO move to a proper location

# IDim = gtx.Dimension("i")
# JDim = gtx.Dimension("j")


class VectorTrait:
    def __mul__(self, other):
        print("Custom multiplication")
        if isinstance(other, self.__class__):
            constructor = {
                f.name: getattr(self, f.name) * getattr(other, f.name)
                for f in dataclasses.fields(self)
                if not f.name.startswith("_")
            }
            return self.__class__(**constructor)
        else:  # other is scalar check
            constructor = {
                f.name: getattr(self, f.name) * other
                for f in dataclasses.fields(self)
                if not f.name.startswith("_")
            }
            return self.__class__(**constructor)


@dataclasses.dataclass
class Velocity(VectorTrait):
    u: gtx.Field[[IDim, JDim], gtx.float32]
    v: gtx.Field[[IDim, JDim], gtx.float32]
    __gt_is_vector__: bool = True


# (field_i, field_j, 1)*field_ij

# vel = Velocity(u=vel.u, v=vel.v*3)

# @field_operator
# def foo(vel: tuple[field1, field2])
#     vel*other_field

# (field1,field2,field3)*field
# make_tuple(tuple_)


# class Velocity:
#     def __init__(self, u, v):
#         self.u = u
#         self.v = v

#     def __gt_type__(self):
#         return ts.NamedTupleType(
#             types=[
#                 ts.FieldType(
#                     dims=[IDim, JDim],
#                     dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32),
#                 ),
#                 ts.FieldType(
#                     dims=[IDim, JDim],
#                     dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32),
#                 ),
#             ],
#             keys=["u", "v"],
#         )

assert hasattr(Velocity, "__mul__")


@gtx.field_operator
def foo(
    vel: Velocity,
    # vel: ts.NamedTuple
    # vel:
) -> gtx.Field[[IDim, JDim], gtx.float32]:
    return vel.u + vel.v


@gtx.program
def foo_program(
    vel: Velocity,
    out: gtx.Field[[IDim, JDim], gtx.float32],
) -> gtx.Field[[IDim, JDim], gtx.float32]:
    foo(vel, out=out)


def test_named_tuple_like_constructed_outside(cartesian_case):
    vel = cases.allocate(cartesian_case, foo_program, "vel")()
    vel = Velocity(u=vel[0], v=vel[1])  # TODO make cases construct this directly
    out = cases.allocate(cartesian_case, foo_program, "out")()

    cases.verify(
        cartesian_case,
        foo_program,
        vel,
        out,
        inout=out,
        ref=(vel.u + vel.v),
    )


@gtx.field_operator
def bar(
    vel: Velocity,
) -> tuple[gtx.Field[[IDim, JDim], gtx.float32], gtx.Field[[IDim, JDim], gtx.float32]]:
    # ) -> Velocity:
    tmp = Velocity(v=vel.u - vel.v, u=vel.u + vel.v)  # order swapped to show kwargs work
    return tmp.u, tmp.v  # later return Velocity directly


def test_named_tuple_like_constructed_inside(cartesian_case):
    vel = cases.allocate(cartesian_case, bar, "vel")()
    vel = Velocity(u=vel[0], v=vel[1])  # TODO make cases construct this directly
    out = cases.allocate(cartesian_case, bar, cases.RETURN)()
    out = Velocity(u=out[0], v=out[1])  # TODO make cases construct this directly

    cases.verify(
        cartesian_case,
        bar,
        vel,
        out=out,
        ref=(vel.u + vel.v, vel.u - vel.v),
    )


@gtx.field_operator
def unroll(vel: Velocity, factor: gtx.Field[[IDim, JDim], gtx.float32]) -> Velocity:
    return vel * factor


def test_unroll(cartesian_case):
    vel = cases.allocate(cartesian_case, unroll, "vel")()
    vel = Velocity(u=vel[0], v=vel[1])  # TODO make cases construct this directly
    factor = cases.allocate(cartesian_case, unroll, "factor")()
    out = cases.allocate(cartesian_case, unroll, cases.RETURN)()
    out = Velocity(u=out[0], v=out[1])  # TODO make cases construct this directly

    cases.verify(
        cartesian_case,
        unroll,
        vel,
        factor,
        out=out,
        ref=(vel.u * factor, vel.v * factor),
    )


# - container with only index access and uniform type
# - container with only index access and runtime known size
# - implicit unroll of tuple_type with non-tuple type (could be implemented on standard tuples independently?)


# - advanced tracers example
# @dataclasses.dataclass
# class Tracer:
#     tracer: gtx.Field[[IDim, JDim], gtx.float32]
#     kind: int


# class TracerList:
#     def __init__(self, tracers: list[Tracer]):
#         self.tracers = tracers

#     def __gt_type__(self):
#         return ts.ListType(  # if reusing this makes sense
#             element_type=ts.NamedTupleType(
#                 types=[
#                     ts.FieldType(
#                         dims=[IDim, JDim],
#                         dtype=ts.ScalarType(kind=ts.ScalarKind.FLOAT32),
#                     ),
#                     ts.ScalarType(kind=ts.ScalarKind.INT32),
#                 ],
#                 keys=["tracer", "kind"],
#             )
#         )


# @gtx.field_operator
# def select_tracer(
#     tracers: TracerList,
#     kind: int,
# ) -> gtx.Field[[IDim, JDim], gtx.float32]:
#     return TracerList(tracer for tracer in tracers if tracer.kind == kind)


# or can we avoid the loop-like construct (it's not a bad loop here, but adds a lot of syntax to the language)
