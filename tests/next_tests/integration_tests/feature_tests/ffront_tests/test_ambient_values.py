# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from gt4py import next as gtx
from gt4py.next import common
from gt4py.next.ffront.fbuiltins import neighbor_sum

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    V2E,
    V2EDim,
    cartesian_case,
    mesh_descriptor,
    unstructured_case,
)
from next_tests.integration_tests.cases_utils import exec_alloc_descriptor


class Grid(gtx.Container):
    dx: gtx.Static[float]
    nu: gtx.Extern[float]


grid = Grid()


@gtx.field_operator
def scale_by_dx(a: cases.IFloatField) -> cases.IFloatField:
    return grid.dx * a


@gtx.program
def scale_by_dx_program(a: cases.IFloatField, out: cases.IFloatField):
    scale_by_dx(a, out=out)


@gtx.field_operator
def scale_by_nu(a: cases.IFloatField) -> cases.IFloatField:
    return grid.nu * a


@gtx.program
def scale_by_nu_program(a: cases.IFloatField, out: cases.IFloatField):
    scale_by_nu(a, out=out)


@gtx.field_operator
def scale_by_dx_twice(a: cases.IFloatField) -> cases.IFloatField:
    return scale_by_dx(a) + scale_by_dx(a)


@gtx.program
def nested_program(a: cases.IFloatField, out: cases.IFloatField):
    scale_by_dx_twice(a, out=out)


@gtx.field_operator
def sum_neighbors(a: cases.EField) -> cases.VField:
    return neighbor_sum(a(V2E), axis=V2EDim)


@gtx.program
def sum_neighbors_program(a: cases.EField, out: cases.VField):
    sum_neighbors(a, out=out)


def _inout(case, program):
    return (
        cases.allocate(case, program, "a")(),
        cases.allocate(case, program, "out").zeros()(),
    )


@pytest.mark.parametrize("spacing", [0.5, 2.0])
def test_value_bound_at_call(cartesian_case, spacing):
    a, out = _inout(cartesian_case, scale_by_dx_program)

    scale_by_dx_program.with_backend(cartesian_case.backend)(a, out, bind=Grid(dx=spacing))

    np.testing.assert_allclose(out.asnumpy(), spacing * a.asnumpy())


def test_value_bound_in_region(cartesian_case):
    a, out = _inout(cartesian_case, scale_by_dx_program)

    with gtx.bind(Grid(dx=0.5)):
        cases.verify(cartesian_case, scale_by_dx_program, a, out=out, ref=0.5 * a.asnumpy())


def test_extern_bound_at_call(cartesian_case):
    a, out = _inout(cartesian_case, scale_by_nu_program)

    scale_by_nu_program.with_backend(cartesian_case.backend)(a, out, bind=Grid(nu=1e-3))

    np.testing.assert_allclose(out.asnumpy(), 1e-3 * a.asnumpy())


def test_value_reaches_nested_operator(cartesian_case):
    a, out = _inout(cartesian_case, nested_program)

    nested_program.with_backend(cartesian_case.backend)(a, out, bind=Grid(dx=0.5))

    np.testing.assert_allclose(out.asnumpy(), a.asnumpy())


def test_distinct_values_do_not_share_a_compiled_program(cartesian_case):
    a, out = _inout(cartesian_case, scale_by_dx_program)
    testee = scale_by_dx_program.with_backend(cartesian_case.backend)

    for spacing in (0.5, 2.0):
        testee(a, out, bind=Grid(dx=spacing))
        np.testing.assert_allclose(out.asnumpy(), spacing * a.asnumpy())


@pytest.mark.parametrize(
    "program, variants",
    [(scale_by_dx_program, 2), (scale_by_nu_program, 1)],
    ids=["static", "extern"],
)
def test_static_specialises_and_extern_does_not(cartesian_case, program, variants):
    if cartesian_case.backend is None:
        pytest.skip("Embedded execution does not compile programs.")
    a, out = _inout(cartesian_case, program)
    testee = program.with_backend(cartesian_case.backend)

    for value in (0.5, 2.0):
        # Both declarations are bound although each program reads only one: a static
        # declaration a program does not read must not specialise it either.
        testee(a, out, bind=Grid(dx=value, nu=value))

    assert len(testee._compiled_programs.compiled_programs) == variants


def test_bind_does_not_leak_past_the_call(cartesian_case):
    a, out = _inout(cartesian_case, scale_by_dx_program)

    scale_by_dx_program.with_backend(cartesian_case.backend)(a, out, bind=Grid(dx=0.5))

    with pytest.raises(ValueError, match="not bound"):
        grid.dx


@pytest.mark.uses_unstructured_shift
@pytest.mark.parametrize(
    "testee", [sum_neighbors, sum_neighbors_program], ids=["field-operator", "program"]
)
def test_offset_spellings_agree(unstructured_case, testee):
    connectivity = unstructured_case.offset_provider[V2E.value]
    a = cases.allocate(unstructured_case, sum_neighbors, "a")()
    outs = [
        cases.allocate(unstructured_case, sum_neighbors, cases.RETURN).zeros()() for _ in range(3)
    ]
    testee = testee.with_backend(unstructured_case.backend)

    testee(a, out=outs[0], offset_provider=unstructured_case.offset_provider)
    with gtx.bind({V2E: connectivity}):
        testee(a, out=outs[1])
    testee(a, out=outs[2], bind={V2E: connectivity})

    v2e_table = connectivity.asnumpy()
    ref = np.sum(a.asnumpy()[v2e_table], axis=1, where=v2e_table != common._DEFAULT_SKIP_VALUE)
    for out in outs:
        np.testing.assert_allclose(out.asnumpy(), ref)
