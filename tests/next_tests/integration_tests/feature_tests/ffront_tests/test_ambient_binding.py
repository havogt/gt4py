# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Ambient binding of the offset provider, across the backend matrix."""

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import common, neighbor_sum

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import (
    V2E,
    Edge,
    V2EDim,
    Vertex,
    unstructured_case,
)
from next_tests.integration_tests.cases_utils import (
    exec_alloc_descriptor,
    mesh_descriptor,
)


@gtx.field_operator
def sum_edges(edge_f: cases.EField) -> cases.VField:
    return neighbor_sum(edge_f(V2E), axis=V2EDim)


@gtx.program
def sum_edges_program(edge_f: cases.EField, out: cases.VField) -> None:
    sum_edges(edge_f, out=out)


def _reference(case, inp):
    v2e_table = case.offset_provider["V2E"].asnumpy()
    return np.sum(
        inp.asnumpy()[v2e_table],
        axis=1,
        where=v2e_table != common._DEFAULT_SKIP_VALUE,
    )


@pytest.mark.uses_unstructured_shift
@pytest.mark.parametrize("entry_point", ["program", "field_operator"])
@pytest.mark.parametrize("mechanism", ["offset_provider", "context_manager", "bind_kwarg"])
def test_ambient_binding_matches_explicit_offset_provider(
    unstructured_case, entry_point, mechanism
):
    """Every spelling produces the same result on every backend in the matrix."""
    inp = cases.allocate(unstructured_case, sum_edges, "edge_f")()
    out = cases.allocate(unstructured_case, sum_edges, cases.RETURN)()

    # one rule for everything ambient: the declaration is the key. The fixture
    # hands out a name-keyed mapping, so re-key it on the offset declarations.
    bound = {V2E: unstructured_case.offset_provider["V2E"]}

    if entry_point == "program":
        callee = sum_edges_program.with_backend(unstructured_case.backend)
        args, kwargs = (inp, out), {}
    else:
        callee = sum_edges.with_backend(unstructured_case.backend)
        args, kwargs = (inp,), {"out": out}

    if mechanism == "offset_provider":
        callee(*args, **kwargs, offset_provider=unstructured_case.offset_provider)
    elif mechanism == "context_manager":
        with gtx.bind(V2E, bound[V2E]):
            callee(*args, **kwargs)
    else:
        callee(*args, **kwargs, bind=bound)

    np.testing.assert_allclose(out.asnumpy(), _reference(unstructured_case, inp))


@pytest.mark.uses_unstructured_shift
def test_bind_kwarg_does_not_leak_past_the_call(unstructured_case):
    inp = cases.allocate(unstructured_case, sum_edges, "edge_f")()
    out = cases.allocate(unstructured_case, sum_edges, cases.RETURN)()
    bound = {V2E: unstructured_case.offset_provider["V2E"]}

    sum_edges_program.with_backend(unstructured_case.backend)(inp, out, bind=bound)

    assert gtx.ambient.offset_provider() == {}
