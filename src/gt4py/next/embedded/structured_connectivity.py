# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
"""Structured (Cartesian-offset) connectivity for embedded execution.

A `StructuredConnectivity` replaces an ``(N, max_neighbors)`` integer neighbor
table with a per-source-color table of Cartesian offsets on the target layout.
For a classical unstructured access ``inp(Offset[k])`` the gather is computed
as a ``concat_where``-tree built from the offsets table (one branch per color),
instead of an indirect table lookup.

TODO(havogt): this currently lives under ``embedded/`` because the only
dispatch site is embedded execution. When the API stabilizes, move the
dataclass next to the other connectivity types in ``common`` and keep only the
embedded expansion here.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable

from gt4py.next import common
from gt4py.next.ffront.experimental import concat_where


@dataclasses.dataclass(frozen=True)
class StructuredConnectivity:
    """Per-color Cartesian offset table replacing an integer neighbor table.

    Example — C2E on a parallelogram patch where ``X`` encodes element color::

        StructuredConnectivity(
            source_dim=C,                           # unstructured target dim
            codomain=E,                             # unstructured source dim
            color_dim=X,                            # target dim encoding color
            local_dim=C2EDim,                       # LOCAL dim for neighbor_sum
            offsets={
                0: [{J: 1}, {X: 1}, {X: 2}],        # △: edges at J+1, X+1, X+2
                1: [{X: -1}, {I: 1}, {X: 1}],       # ▽: edges at X-1, I+1, X+1
            },
        )

    Single-color connectivities (e.g. V2E on a patch where vertices are one
    color) use a one-entry ``offsets`` dict; the expansion short-circuits the
    ``concat_where`` in that case.
    """

    source_dim: common.Dimension
    codomain: common.Dimension
    color_dim: common.Dimension
    local_dim: common.Dimension
    offsets: dict[int, list[dict[common.Dimension, int]]]

    def __post_init__(self) -> None:
        lengths = {len(v) for v in self.offsets.values()}
        if len(lengths) != 1:
            raise ValueError(
                f"Inconsistent neighbor counts across colors: {lengths}. "
                f"All colors must provide the same num_neighbors."
            )

    @property
    def num_neighbors(self) -> int:
        return len(next(iter(self.offsets.values())))

    @property
    def colors(self) -> tuple[int, ...]:
        return tuple(sorted(self.offsets.keys()))


@dataclasses.dataclass(frozen=True)
class _StructuredConnectivityK:
    """A `StructuredConnectivity` paired with a specific neighbor index.

    Produced by ``FieldOffset.__getitem__(k)`` when the offset provider entry
    is a `StructuredConnectivity`, and consumed by the embedded ``premap``
    dispatch to build the ``concat_where`` tree for the k-th neighbor.
    """

    connectivity: StructuredConnectivity
    k: int


def _apply_shift(field: common.Field, shift: dict[common.Dimension, int]) -> common.Field:
    """Apply a Cartesian offset dict to a field via chained ``field(dim + offset)``."""
    for dim, offset in shift.items():
        if offset == 0:
            continue
        field = field(dim + offset)
    return field


def expand_k(inp: common.Field, conn: StructuredConnectivity, k: int) -> common.Field:
    """Gather the k-th neighbor per-color.

    Returns a field on the Cartesian layout whose value at each ``color_dim == c``
    slice is ``inp`` shifted by ``conn.offsets[c][k]``.
    """
    if not 0 <= k < conn.num_neighbors:
        raise IndexError(f"k={k} out of range [0, {conn.num_neighbors})")

    colors = conn.colors
    if len(colors) == 1:
        return _apply_shift(inp, conn.offsets[colors[0]][k])

    # Build the concat_where chain from last → first so the outer-most condition
    # pins color_dim == colors[0], matching hand-rolled structured examples.
    result = _apply_shift(inp, conn.offsets[colors[-1]][k])
    for c in reversed(colors[:-1]):
        branch = _apply_shift(inp, conn.offsets[c][k])
        result = concat_where(conn.color_dim == c, branch, result)
    return result


def expand_all(inp: common.Field, conn: StructuredConnectivity) -> Iterable[common.Field]:
    """Yield the per-k gathered fields; useful for stacking or summing externally."""
    for k in range(conn.num_neighbors):
        yield expand_k(inp, conn, k)
