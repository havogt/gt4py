# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# C2E connectivity for an icosahedral grid (parallelogram of triangles)
#
# The grid is built from upward-pointing (△) and downward-pointing (▽) triangular cells.
#
# Cell kinds:
#   kind 0 : △ (upward-pointing)
#   kind 1 : ▽ (downward-pointing)
#
# Edge kinds:
#   kind 0 (—) : horizontal edges
#   kind 1 (/) : diagonal edges (rising left-to-right)
#   kind 2 (\) : diagonal edges (falling left-to-right)
#
# C2E maps each cell to its 3 bounding edges:
#   - △ (cell kind 0) :  C2E = [bottom —, left /, right \]
#   - ▽ (cell kind 1) :  C2E = [top —, right /, left \]
#
# Parallelogram grid (both rows start with △, row 1 shifted LEFT):
#
#  Each vertex owns exactly 3 edges (one of each kind):
#    — edge to its RIGHT, / edge to its LOWER-LEFT, \ edge to its LOWER-RIGHT
#
#
#           v0———e0———v1———e1———v2——e2—————*
#          / \       / \       / \
#        e6   e12   e7 e13   e8   e14
#        / c0  \ c6/ c1  \ c7/ c2  \ c8
#       /       \ /       \ /       \
#      v3———e3———v4————e4——v5——e5————*
#     / \        / \       / \
#   e9   e15   e10  e16  e11  e17
#   /  c3  \c9 /  c4 \c10/ c5  \ c11
#  /        \ /       \ /       \
# *          *         *         *
#  V2E ownership (vertex → [—, /, \]):
#    v0 → [e0,  e6,  e12]
#    v1 → [e1,  e7,  e13]
#    v2 → [e2,  e8,  e14]
#    v3 → [e3,  e9,  e15]
#    v4 → [e4,  e10, e16]
#    v5 → [e5,  e11, e17]
#
#  Cell layout:
#   - △ (kind 0): c0, c1, c2, c3, c4, c5,
#   - ▽ (kind 1): c6, c7, c8, c9, c10, c11
#
#  Edge kinds:
#    kind 0 (—) : e0,e1,e2,e3,e4,e5           (horizontal)
#    kind 1 (/) : e6,e7,e8,e9,e10,e11         (diagonal /)
#    kind 2 (\) : e12,e13,e14,e15,e16,e17     (diagonal \)
#
#  C2E connectivity table (* = edge outside domain):
#  ┌──────┬────────────────┬────────────────────────────────┐
#  │ Cell │ Kind (type)    │ C2E  [edge0, edge1, edge2]     │
#  ├──────┼────────────────┼────────────────────────────────┤
#  │  c0  │ 0  △ (up)      │ [e3,   e6,   e12]             │
#  │  c1  │ 0  △ (up)      │ [e4,   e7,   e13]             │
#  │  c2  │ 0  △ (up)      │ [e5,   e8,   e14]             │
#  │  c3  │ 0  △ (up)      │ [ *,   e9,   e15]             │
#  │  c4  │ 0  △ (up)      │ [ *,   e10,  e16]             │
#  │  c5  │ 0  △ (up)      │ [ *,   e11,  e17]             │
#  ├──────┼────────────────┼────────────────────────────────┤
#  │  c6  │ 1  ▽ (down)    │ [e0,   e7,   e12]             │
#  │  c7  │ 1  ▽ (down)    │ [e1,   e8,   e13]             │
#  │  c8  │ 1  ▽ (down)    │ [e2,    *,   e14]             │
#  │  c9  │ 1  ▽ (down)    │ [e3,   e10,  e15]             │
#  │  c10 │ 1  ▽ (down)    │ [e4,   e11,  e16]             │
#  │  c11 │ 1  ▽ (down)    │ [e5,    *,   e17]             │
#  └──────┴────────────────┴────────────────────────────────┘
#
# Same diagram with (i,j,kind) labeling:
#
#  Vertices: (i,j,0)  — only one kind of vertex
#  Edges:    (i,j,k)  — k=0 horizontal, k=1 diagonal /, k=2 diagonal \  # noqa: ERA001, RUF100
#  Cells:    (i,j,k)  — k=0 upward △, k=1 downward ▽
#
#  Using condensed notation: ijk means (i,j,k), ij means (i,j)
#
#           00——000——10———100———20———200——*
#          / \       / \       / \
#       001  002  101  102   201  202
#        / 000 \001/ 100 \101/ 200 \201
#       /       \ /       \ /       \
#      01——010——11————110——21———210——*
#     / \       / \       / \
#   011 012   111  112  211  212
#   / 010 \011/ 110 \111/ 210 \211
#  /       \ /       \ /       \
# *         *         *         *
#
#  Mapping to flat indices:
#  ┌──────────────┬──────────────────────────────────────────┐
#  │  Element     │  (i,j,kind) → flat index                │
#  ├──────────────┼──────────────────────────────────────────┤
#  │  Vertex      │  v(i,j) = j * 3 + i                     │
#  │  Edge        │  e(i,j,k) = k * 6 + j * 3 + i           │
#  │  Cell        │  c(i,j,k) = k * 6 + j * 3 + i           │
#  └──────────────┴──────────────────────────────────────────┘
#
#  C2E in (i,j,kind) notation:
#   △ cell (i,j,0): C2E = [(i, j+1, 0), (i, j, 1), (i, j, 2)]
#                           bottom —     left /      right \
#
#   ▽ cell (i,j,1): C2E = [(i, j, 0),  (i+1,j, 1), (i, j, 2)]
#                           top —        right /      left \
#

from gt4py import next as gtx
from gt4py.next.experimental import concat_where


I = gtx.Dimension("I")  # noqa: E741
J = gtx.Dimension("J")
X = gtx.Dimension("X")

C = gtx.Dimension("C")
E = gtx.Dimension("E")

C2EDim = gtx.Dimension("C2EDim", kind=gtx.DimensionKind.LOCAL)

C2E = gtx.FieldOffset("C2E", source=E, target=(C, C2EDim))


@gtx.field_operator
def avg(inp: gtx.Field[[E], float]) -> gtx.Field[[C], float]:
    return inp(C2E[0]) + inp(C2E[1]) + inp(C2E[2])


print(avg.__gt_itir__())


@gtx.field_operator
def avg_C0(inp: gtx.Field[[I, J, X], float]):
    return inp(J + 1) + inp(X + 1) + inp(X + 2)  # c0 -> e3+e6+e12


@gtx.field_operator
def avg_C1(inp: gtx.Field[[I, J, X], float]):
    return inp(X - 1) + inp(I + 1) + inp(X + 1)  # c6 -> e0+e7+e12


@gtx.field_operator
def on_cells(
    c0: gtx.Field[[I, J, X], float], c1: gtx.Field[[I, J, X], float]
) -> gtx.Field[[I, J, X], float]:
    return concat_where(X == 0, c0, c1)


@gtx.field_operator
def avg_cartesian(inp: gtx.Field[[I, J, X], float]) -> gtx.Field[[I, J, X], float]:
    return on_cells(avg_C0(inp), avg_C1(inp))


print(avg_cartesian.__gt_itir__())
