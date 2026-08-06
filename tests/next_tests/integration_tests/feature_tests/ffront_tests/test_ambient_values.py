# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Ambient values referenced by name inside an operator, bound at call time."""

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next.type_system import type_specifications as ts

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import IDim, JDim, cartesian_case
from next_tests.integration_tests.cases_utils import exec_alloc_descriptor


IJFloatField = gtx.Field[gtx.Dims[IDim, JDim], gtx.float64]


class Grid(gtx.Container):
    """Declarations live in a container; `grid.dx` reads the bound value."""

    dx = gtx.Static[gtx.float64]
    dx_extern = gtx.Extern[gtx.float64]


grid = Grid()


@gtx.field_operator
def delta_x(f: IJFloatField) -> IJFloatField:
    """Forward difference in x."""
    return (1.0 / grid.dx) * (f(IDim + 1) - f)


@gtx.field_operator
def delta_x_twice(f: IJFloatField) -> IJFloatField:
    """`dx` is used one level down, and still never appears in a signature."""
    return delta_x(f) + delta_x(f)


@gtx.program
def run_delta_x(f: IJFloatField, out: IJFloatField) -> None:
    delta_x(f, out=out)


@gtx.field_operator
def delta_x_extern(f: IJFloatField) -> IJFloatField:
    """Same, but supplied as a runtime argument instead of folded in."""
    return (1.0 / grid.dx_extern) * (f(IDim + 1) - f)


@gtx.program
def run_delta_x_twice(f: IJFloatField, out: IJFloatField) -> None:
    delta_x_twice(f, out=out)


@gtx.program
def run_delta_x_extern(f: IJFloatField, out: IJFloatField) -> None:
    delta_x_extern(f, out=out)


def _inputs(case):
    data = gtx.as_field([IDim, JDim], np.arange(20.0).reshape(5, 4), allocator=case.allocator)
    out = gtx.zeros(gtx.domain({IDim: 4, JDim: 4}), dtype=np.float64, allocator=case.allocator)
    return data, out


def _binding(spacing):
    """A container is bound as a unit, so every declaration it holds gets a value."""
    return {Grid.dx: spacing, Grid.dx_extern: spacing}


def _reference(data, spacing, factor=1):
    a = data.asnumpy()
    return factor * (1.0 / spacing) * (a[1:5, :] - a[0:4, :])


@pytest.mark.uses_cartesian_shift
@pytest.mark.parametrize("spacing", [0.5, 0.25])
def test_static_value_is_bound_at_call(cartesian_case, spacing):
    data, out = _inputs(cartesian_case)
    run_delta_x.with_backend(cartesian_case.backend)(data, out, bind=_binding(spacing))
    np.testing.assert_allclose(out.asnumpy(), _reference(data, spacing))


@pytest.mark.uses_cartesian_shift
def test_distinct_values_do_not_share_a_compiled_program(cartesian_case):
    """The second binding must not reuse the first one's folded literal."""
    data, out = _inputs(cartesian_case)
    prog = run_delta_x.with_backend(cartesian_case.backend)

    prog(data, out, bind=_binding(0.5))
    np.testing.assert_allclose(out.asnumpy(), _reference(data, 0.5))

    prog(data, out, bind=_binding(0.25))
    np.testing.assert_allclose(out.asnumpy(), _reference(data, 0.25))


@pytest.mark.uses_cartesian_shift
def test_value_reaches_a_nested_operator(cartesian_case):
    """`dx` is referenced two levels down without being passed as an argument."""
    data, out = _inputs(cartesian_case)
    run_delta_x_twice.with_backend(cartesian_case.backend)(data, out, bind=_binding(0.5))
    np.testing.assert_allclose(out.asnumpy(), _reference(data, 0.5, factor=2))


@pytest.mark.uses_cartesian_shift
def test_context_manager_binding(cartesian_case):
    data, out = _inputs(cartesian_case)
    with gtx.bindings(_binding(0.5)):
        run_delta_x.with_backend(cartesian_case.backend)(data, out)
    np.testing.assert_allclose(out.asnumpy(), _reference(data, 0.5))


@pytest.mark.uses_cartesian_shift
@pytest.mark.parametrize("spacing", [0.5, 0.25])
def test_extern_value_is_bound_at_call(cartesian_case, spacing):
    data, out = _inputs(cartesian_case)
    run_delta_x_extern.with_backend(cartesian_case.backend)(data, out, bind=_binding(spacing))
    np.testing.assert_allclose(out.asnumpy(), _reference(data, spacing))


@pytest.mark.uses_cartesian_shift
def test_static_specializes_but_extern_does_not(cartesian_case):
    """The one difference between the two forms: how many programs get compiled."""
    if cartesian_case.backend is None:
        pytest.skip("compiled-program pool only exists for compiled backends")
    data, out = _inputs(cartesian_case)

    static_prog = run_delta_x.with_backend(cartesian_case.backend)
    extern_prog = run_delta_x_extern.with_backend(cartesian_case.backend)
    for spacing in (0.5, 0.25):
        static_prog(data, out, bind=_binding(spacing))
        extern_prog(data, out, bind=_binding(spacing))

    assert len(static_prog._compiled_programs.compiled_programs) == 2
    assert len(extern_prog._compiled_programs.compiled_programs) == 1


def test_declaration_types_itself_without_a_binding():
    """The frontend only needs the type at decoration; the value comes later."""
    expected = ts.ScalarType(kind=ts.ScalarKind.FLOAT64)
    assert gtx.Static[gtx.float64].__gt_type__() == expected
    assert gtx.Extern[gtx.float64].__gt_type__() == expected


def test_unbound_declaration_reports_itself():
    with pytest.raises(ValueError, match="not bound"):
        Grid.dx.value


def test_class_access_is_the_declaration_instance_access_the_value():
    """The descriptor is what lets embedded execution see a plain scalar."""
    assert Grid.dx is not None and not isinstance(Grid.dx, float)
    with gtx.bindings({Grid.dx: 0.5}):
        assert grid.dx == 0.5
        assert 1.0 / grid.dx == 2.0  # no arithmetic protocol on the declaration
