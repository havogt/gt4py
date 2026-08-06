# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import contextvars
import re
import sys
import types

import pytest

from gt4py import next as gtx
from gt4py.next import ambient
from gt4py.next.type_system import type_specifications as ts, type_translation


IDim = gtx.Dimension("IDim")
KDim = gtx.Dimension("KDim", kind=gtx.DimensionKind.VERTICAL)
Koff = gtx.FieldOffset("Koff", source=KDim, target=(KDim,))
Ioff = gtx.FieldOffset("Ioff", source=IDim, target=(IDim,))

KField = gtx.Field[gtx.Dims[KDim], gtx.float64]


class Grid(gtx.Container):
    dx: gtx.Static[float]
    nu: gtx.Extern[float]


grid = Grid()


def container_in_module(module_name: str) -> type:
    """Declare a container in a module of its own, as two unrelated user modules would."""
    module = types.ModuleType(module_name)
    module.gtx = gtx
    sys.modules[module_name] = module

    class Grid(gtx.Container):
        __module__ = module_name

        dx: gtx.Static[float]

    return Grid


def test_class_access_is_the_context_variable():
    assert isinstance(Grid.dx, contextvars.ContextVar)
    assert Grid.dx is not Grid.nu


def test_instance_access_is_the_value():
    with gtx.bind({Grid.dx: 0.5}):
        assert grid.dx == 0.5
        assert 1.0 / grid.dx == 2.0  # a plain scalar, no arithmetic protocol involved


def test_container_binds_all_its_values():
    with gtx.bind(Grid(dx=0.5, nu=1e-3)):
        assert (grid.dx, grid.nu) == (0.5, 1e-3)


def test_partial_container_binds_what_it_carries():
    with gtx.bind(Grid(dx=0.5)):
        assert grid.dx == 0.5
        with pytest.raises(ValueError, match="Grid.nu' is not bound"):
            grid.nu


def test_undeclared_keyword_is_rejected():
    with pytest.raises(TypeError, match="no declaration 'dz'"):
        Grid(dz=0.5)


def test_bindings_nest_and_unwind():
    with gtx.bind({Grid.dx: 0.5}):
        with gtx.bind({Grid.dx: 0.25}):
            assert grid.dx == 0.25
        assert grid.dx == 0.5


def test_bindings_are_context_local():
    def bound():
        with gtx.bind({Grid.dx: 0.5}):
            return grid.dx

    assert contextvars.copy_context().run(bound) == 0.5
    with pytest.raises(ValueError, match="Grid.dx' is not bound"):
        grid.dx


def test_unbound_declaration_names_the_declaration():
    with pytest.raises(ValueError, match=re.escape(f"'{__name__}.Grid.dx'")):
        grid.dx


def test_same_name_in_different_modules_gives_distinct_parameters():
    first = container_in_module("next_tests_ambient_module_a")
    second = container_in_module("next_tests_ambient_module_b")

    assert first.__declarations__["dx"].param_name != second.__declarations__["dx"].param_name


def test_parameter_name_depends_only_on_the_qualified_name():
    assert Grid.__declarations__["dx"].param_name == ambient.parameter_name(
        f"{__name__}.Grid", "dx"
    )


def test_ambiguous_container_is_rejected():
    def declare(**kwargs):
        class Ambiguous(gtx.Container, **kwargs):
            dx: gtx.Static[float]

        return Ambiguous

    first = declare()
    with pytest.raises(TypeError, match="already declared"):
        declare()
    assert declare(name="explicit").__container_id__ != first.__container_id__


def test_subclassing_is_rejected():
    with pytest.raises(TypeError, match="must not derive from container"):

        class Derived(Grid):
            dz: gtx.Static[float]


def test_declaration_without_type_is_rejected():
    with pytest.raises(TypeError, match="Invalid declaration"):

        class Untagged(gtx.Container):
            dx: float


def test_container_types_itself_without_binding():
    type_ = type_translation.from_value(grid)

    assert isinstance(type_, ts.NamespaceType)
    assert type_.dx == ts.ScalarType(kind=ts.ScalarKind.FLOAT64)


def test_offset_provider_is_scoped_to_the_referenced_offsets():
    @gtx.field_operator
    def shift_k(f: KField) -> KField:
        return f(Koff[1])

    @gtx.program
    def testee(f: KField, out: KField) -> None:
        shift_k(f, out=out)

    with gtx.bind({Koff: "k-connectivity", Ioff: "unrelated"}):
        assert ambient.offset_provider(testee.past_stage.closure_vars) == {"Koff": "k-connectivity"}
