# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

import gt4py.next as gtx
from gt4py.next import neighbor_sum
from gt4py.next.experimental import as_offset, connectivity_chains, required_indices
from gt4py.next.ffront.connectivity_chains import Access, Unbounded


Cell = gtx.Dimension("Cell")
Edge = gtx.Dimension("Edge")
K = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)
KHalf = gtx.flip_staggered(K)
C2EDim = gtx.Dimension("C2E", kind=gtx.DimensionKind.LOCAL)
E2CDim = gtx.Dimension("E2C", kind=gtx.DimensionKind.LOCAL)
C2E = gtx.FieldOffset("C2E", source=Edge, target=(Cell, C2EDim))
E2C = gtx.FieldOffset("E2C", source=Cell, target=(Edge, E2CDim))
Koff = gtx.FieldOffset("Koff", source=K, target=(K,))

CellField = gtx.Field[[Cell], float]
EdgeField = gtx.Field[[Edge], float]
CellKField = gtx.Field[[Cell, K], float]

# 4 cells in a ring, edge i between cell i and cell i+1; the last edge has a missing neighbor
C2E_TABLE = np.array([[3, 0], [0, 1], [1, 2], [2, 3]], dtype=np.int32)
E2C_TABLE = np.array([[0, 1], [1, 2], [2, 3], [3, -1]], dtype=np.int32)
OFFSET_PROVIDER = {
    "C2E": gtx.as_connectivity([Cell, C2EDim], Edge, C2E_TABLE),
    "E2C": gtx.as_connectivity([Edge, E2CDim], Cell, E2C_TABLE, skip_value=-1),
}


@gtx.field_operator
def cell_to_edge(c: CellField) -> EdgeField:
    return neighbor_sum(c(E2C), axis=E2CDim)


def test_single_neighbor_chain():
    assert connectivity_chains(cell_to_edge, OFFSET_PROVIDER) == {
        (): {"c": frozenset({Access(chain=("E2C",))})}
    }


def test_chain_through_nested_field_operator_is_ordered_from_the_output():
    @gtx.field_operator
    def testee(c: CellField, e: EdgeField) -> CellField:
        return neighbor_sum((cell_to_edge(c) + e)(C2E), axis=C2EDim) + c

    assert connectivity_chains(testee, OFFSET_PROVIDER) == {
        (): {
            "c": frozenset({Access(chain=("C2E", "E2C")), Access()}),
            "e": frozenset({Access(chain=("C2E",))}),
        }
    }


def test_tuple_return_is_reported_per_output():
    @gtx.field_operator
    def testee(c: CellField, e: EdgeField) -> tuple[EdgeField, CellField]:
        return cell_to_edge(c), neighbor_sum(e(C2E), axis=C2EDim)

    assert connectivity_chains(testee, OFFSET_PROVIDER) == {
        (0,): {"c": frozenset({Access(chain=("E2C",))})},
        (1,): {"e": frozenset({Access(chain=("C2E",))})},
    }


def test_scalar_and_unread_parameters():
    @gtx.field_operator
    def testee(c: CellField, s: float, unused: CellField) -> CellField:
        return c * s if s > 0.0 else c

    assert connectivity_chains(testee, OFFSET_PROVIDER) == {
        (): {"c": frozenset({Access()}), "s": frozenset({Access()})}
    }


def test_vertical_displacements():
    @gtx.field_operator
    def testee(
        a: CellKField, w: gtx.Field[[Cell, KHalf], float], off: gtx.Field[[Cell, K], gtx.int32]
    ) -> gtx.Field[[Edge, K], float]:
        column = a(K + 1) + w(K - 0.5) + w(K + 0.5) + a(as_offset(Koff, off))
        return neighbor_sum(column(E2C), axis=E2CDim)

    assert connectivity_chains(testee, OFFSET_PROVIDER) == {
        (): {
            "a": frozenset(
                {
                    Access(chain=("E2C",), displacement=(("K", 1),)),
                    Access(chain=("E2C",), displacement=(("K", Unbounded.DYNAMIC),)),
                }
            ),
            "w": frozenset(
                {
                    Access(chain=("E2C",), displacement=(("K", -0.5),)),
                    Access(chain=("E2C",), displacement=(("K", 0.5),)),
                }
            ),
            "off": frozenset({Access(chain=("E2C",))}),
        }
    }


@gtx.scan_operator(axis=K, forward=True, init=0.0)
def column_sum(state: float, x: float) -> float:
    return state + x


def test_scan_reads_the_whole_column():
    assert connectivity_chains(column_sum, {}) == {
        (): {"x": frozenset({Access(displacement=(("K", Unbounded.COLUMN),))})}
    }


def test_scan_inside_field_operator():
    @gtx.field_operator
    def testee(a: CellKField) -> gtx.Field[[Edge, K], float]:
        return neighbor_sum(column_sum(a)(E2C), axis=E2CDim)

    assert connectivity_chains(testee, OFFSET_PROVIDER) == {
        (): {"a": frozenset({Access(chain=("E2C",), displacement=(("K", Unbounded.COLUMN),))})}
    }


def test_required_indices_follow_tables_and_skip_missing_neighbors():
    @gtx.field_operator
    def testee(c: CellField, e: EdgeField) -> tuple[CellField, EdgeField]:
        return neighbor_sum((cell_to_edge(c) + e)(C2E), axis=C2EDim), cell_to_edge(c)

    result = required_indices(
        testee, OFFSET_PROVIDER, owned={Cell: np.array([0]), Edge: np.array([3])}
    )

    # cell 0 -> edges {3, 0} -> cells {3, 0, 1}; edge 3 -> cell 3 (the skip value is dropped)
    assert result.keys() == {"c", "e"}
    np.testing.assert_array_equal(result["c"], [0, 1, 3])
    np.testing.assert_array_equal(result["e"], [0, 3])
