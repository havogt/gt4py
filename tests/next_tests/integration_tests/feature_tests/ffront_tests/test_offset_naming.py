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
from gt4py.next import neighbor_sum

from next_tests.integration_tests.cases import unstructured_case
from next_tests.integration_tests.cases_utils import exec_alloc_descriptor, mesh_descriptor


Vertex = gtx.Dimension("Vertex")
Edge = gtx.Dimension("Edge")
#: deliberately *not* named like the offset it belongs to
V2ELocal = gtx.Dimension("V2ELocal", kind=gtx.DimensionKind.LOCAL)
V2E = gtx.FieldOffset("V2E", source=Edge, target=(Vertex, V2ELocal))

_TABLE = np.asarray([[0, 1], [1, 2], [2, 0]], dtype=np.int32)


def _connectivity(local_dim=V2ELocal):
    return gtx.as_connectivity([Vertex, local_dim], codomain=Edge, data=_TABLE, skip_value=None)


@gtx.field_operator
def sum_over_neighbors(
    e: gtx.Field[gtx.Dims[Edge], gtx.float64],
) -> gtx.Field[gtx.Dims[Vertex], gtx.float64]:
    return neighbor_sum(e(V2E), axis=V2ELocal)


@gtx.field_operator
def sum_sparse(
    s: gtx.Field[gtx.Dims[Vertex, V2ELocal], gtx.float64],
) -> gtx.Field[gtx.Dims[Vertex], gtx.float64]:
    return neighbor_sum(s, axis=V2ELocal)


@pytest.mark.uses_unstructured_shift
def test_offset_tag_differs_from_local_dimension(unstructured_case):
    """The offset provider key and the local dimension need not be the same name."""
    case = unstructured_case
    e = gtx.as_field([Edge], np.asarray([1.0, 2.0, 3.0]), allocator=case.allocator)
    out = gtx.zeros(gtx.domain({Vertex: 3}), dtype=np.float64, allocator=case.allocator)

    op = sum_over_neighbors.with_backend(case.backend)
    op(e, out=out, offset_provider={"V2E": _connectivity()})

    np.testing.assert_allclose(out.asnumpy(), np.asarray([3.0, 5.0, 4.0]))


@pytest.mark.uses_sparse_fields
def test_sparse_field_local_dimension_is_not_a_tag(unstructured_case):
    """A sparse argument's local dimension identifies its connectivity without being its key."""
    case = unstructured_case
    s = gtx.as_field([Vertex, V2ELocal], np.ones((3, 2)), allocator=case.allocator)
    out = gtx.zeros(gtx.domain({Vertex: 3}), dtype=np.float64, allocator=case.allocator)

    op = sum_sparse.with_backend(case.backend)
    op(s, out=out, offset_provider={"V2E": _connectivity()})

    np.testing.assert_allclose(out.asnumpy(), np.asarray([2.0, 2.0, 2.0]))


@pytest.mark.uses_unstructured_shift
def test_offset_declaration_checked_against_connectivity(unstructured_case):
    case = unstructured_case
    if case.backend is None:
        pytest.skip("Embedded does not lower the program, so the declaration is not checked.")
    Other = gtx.Dimension("Other", kind=gtx.DimensionKind.LOCAL)
    e = gtx.as_field([Edge], np.asarray([1.0, 2.0, 3.0]), allocator=case.allocator)
    out = gtx.zeros(gtx.domain({Vertex: 3}), dtype=np.float64, allocator=case.allocator)

    op = sum_over_neighbors.with_backend(case.backend)
    with pytest.raises(ValueError, match="is declared as"):
        op(e, out=out, offset_provider={"V2E": _connectivity(local_dim=Other)})
