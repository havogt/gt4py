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
For a classical unstructured access ``inp(Offset[k])`` the gather concatenates,
along the color dimension, one slab per color of ``inp`` shifted by that color's
offset, instead of an indirect table lookup.

TODO(havogt): this currently lives under ``embedded/`` because the only
dispatch site is embedded execution. When the API stabilizes, move the
dataclass next to the other connectivity types in ``common`` and keep only the
embedded expansion here.
"""

from __future__ import annotations

import dataclasses
import functools
import operator
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, cast

from gt4py.next import common
from gt4py.next.embedded import exceptions as embedded_exceptions


if TYPE_CHECKING:
    from gt4py.next.embedded import nd_array_field


@dataclasses.dataclass(frozen=True)
class StructuredConnectivity:
    """Per-color Cartesian offset table replacing an integer neighbor table.

    Example — C2E on a parallelogram patch where ``X`` encodes element color::

        StructuredConnectivity(
            source_dim=C,  # unstructured target dim
            codomain=E,  # unstructured source dim
            color_dim=X,  # target dim encoding color
            local_dim=C2EDim,  # LOCAL dim for neighbor_sum
            offsets={
                0: [{J: 1}, {X: 1}, {X: 2}],  # △: edges at J+1, X+1, X+2
                1: [{X: -1}, {I: 1}, {X: 1}],  # ▽: edges at X-1, I+1, X+1
            },
        )

    Single-color connectivities (e.g. V2E on a patch where vertices are one
    color) use a one-entry ``offsets`` dict; the expansion is a plain shift in
    that case.
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

    @property
    def max_neighbors(self) -> int:
        return self.num_neighbors

    @property
    def neighbor_dim(self) -> common.Dimension:
        return self.local_dim

    @property
    def has_skip_values(self) -> bool:
        return False

    @functools.cached_property
    def _gt_type(self) -> common.StructuredConnectivityType:
        return common.StructuredConnectivityType(
            source_dim=self.source_dim,
            codomain=self.codomain,
            color_dim=self.color_dim,
            local_dim=self.local_dim,
            offsets=tuple(
                (
                    color,
                    tuple(
                        tuple(
                            sorted(
                                ((d, o) for d, o in off.items() if o != 0), key=lambda p: p[0].value
                            )
                        )
                        for off in self.offsets[color]
                    ),
                )
                for color in self.colors
            ),
        )

    def __gt_type__(self) -> common.StructuredConnectivityType:
        return self._gt_type


@dataclasses.dataclass(frozen=True)
class _StructuredConnectivityK:
    """A `StructuredConnectivity` paired with a specific neighbor index.

    Produced by ``FieldOffset.__getitem__(k)`` when the offset provider entry
    is a `StructuredConnectivity`, and consumed by the embedded ``premap``
    dispatch to gather the k-th neighbor.
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
    return _gather(cast("nd_array_field.NdArrayField", inp), conn, [k], local_axis=None)


def expand_all(inp: common.Field, conn: StructuredConnectivity) -> Iterable[common.Field]:
    """Yield the per-k gathered fields; useful for stacking or summing externally."""
    for k in range(conn.num_neighbors):
        yield expand_k(inp, conn, k)


def expand_stacked(
    inp: nd_array_field.NdArrayField, conn: StructuredConnectivity
) -> nd_array_field.NdArrayField:
    """Gather all neighbors, stacked along ``conn.local_dim``.

    The local dimension is placed before the first vertical dimension (or last if there is none).
    """
    local_axis = next(
        (i for i, d in enumerate(inp.domain.dims) if d.kind == common.DimensionKind.VERTICAL),
        inp.domain.ndim,
    )
    return _gather(inp, conn, range(conn.num_neighbors), local_axis=local_axis)


def _gather(
    inp: nd_array_field.NdArrayField,
    conn: StructuredConnectivity,
    ks: Sequence[int],
    local_axis: Optional[int],
) -> nd_array_field.NdArrayField:
    """Assemble the neighbors `ks` from per-(color, k) slabs with a single concatenation.

    All shifted fields are intersected in the non-color dimensions. With more than one color,
    the color dimension spans the colors whose shifted field covers them for every k, and each
    color slab is taken from its own shifted field.
    """
    color_dim = conn.color_dim
    dims = inp.domain.dims
    colors = conn.colors
    multi_color = len(colors) > 1
    shifted = [[_apply_shift(inp, conn.offsets[c][k]) for k in ks] for c in colors]

    ranges = {}
    for i, d in enumerate(dims):
        if multi_color and d == color_dim:
            continue
        ranges[d] = functools.reduce(
            operator.and_, (f.domain.ranges[i] for per_c in shifted for f in per_c)
        )
    if multi_color:
        covered = [
            c
            for c, per_c in zip(colors, shifted)
            if all(c in f.domain[color_dim].unit_range for f in per_c)
        ]
        if not covered or covered != list(range(covered[0], covered[-1] + 1)):
            raise embedded_exceptions.NonContiguousDomain(
                f"Colors {covered} covered by all shifted fields are not contiguous along '{color_dim}'."
            )
        ranges[color_dim] = common.UnitRange(covered[0], covered[-1] + 1)
        shifted = [shifted[colors.index(c)] for c in covered]
    domain = common.Domain(*(common.NamedRange(d, ranges[d]) for d in dims))

    xp = inp.array_ns
    if multi_color:
        x_axis = domain.dim_index(color_dim, allow_missing=False)
        pieces = [
            f.restrict(
                domain.replace(color_dim, common.NamedRange(color_dim, common.UnitRange(c, c + 1)))
            ).ndarray
            for c, per_c in zip(covered, shifted)
            for f in per_c
        ]
        buffer = xp.concatenate(pieces, axis=x_axis)
        if local_axis is not None:
            shape = domain.shape
            buffer = xp.reshape(
                buffer, (*shape[:x_axis], len(covered), len(ks), *shape[x_axis + 1 :])
            )
            buffer = xp.moveaxis(buffer, x_axis + 1, local_axis)
    else:
        buffer = xp.stack([f.restrict(domain).ndarray for f in shifted[0]], axis=local_axis)

    if local_axis is not None:
        local_range = common.NamedRange(conn.local_dim, common.UnitRange(0, len(ks)))
        domain = common.Domain(*domain[:local_axis], local_range, *domain[local_axis:])
    return inp.__class__.from_array(buffer, domain=domain, dtype=inp.dtype)
