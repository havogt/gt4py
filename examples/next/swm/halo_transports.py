# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared interface for halo transports on a structured 2-D torus.

A *transport* refreshes the halo rim of one device's local block inside a
``shard_map`` body, built only from ``jax.lax`` collectives and array operations
so that ``jax.vjp`` transposes it automatically.

Decomposition: ``P = Rx * Ry`` devices on a 1-D mesh axis; rank ``r`` owns the
block at ``(rx, ry) = divmod(r, Ry)``, i.e. global rows ``[rx*MLOC, (rx+1)*MLOC)``
and columns ``[ry*NLOC, (ry+1)*NLOC)``. Local arrays are ``(MLOC+2h, NLOC+2h)``
with the interior at ``[h:-h, h:-h]``.

Directions: axis 0 is ``I`` (x, E/W), axis 1 is ``J`` (y, N/S).
``E = (+1, 0)``, ``W = (-1, 0)``, ``N = (0, +1)``, ``S = (0, -1)``, and the four
diagonals accordingly.
"""

from __future__ import annotations

import glob
import importlib
import os
import sys
from typing import Any, Protocol

import numpy as np

# the transport_*.py siblings are imported by bare module name
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


NEIGHBOUR_OFFSETS = {
    "E": (1, 0),
    "W": (-1, 0),
    "N": (0, 1),
    "S": (0, -1),
    "NE": (1, 1),
    "NW": (-1, 1),
    "SE": (1, -1),
    "SW": (-1, -1),
}


class Layout:
    """Static 2-D torus decomposition of an ``(M, N)`` grid over ``Rx * Ry`` ranks."""

    def __init__(self, M: int, N: int, Rx: int, Ry: int, h: int = 1):
        if M % Rx or N % Ry:
            raise ValueError(f"grid {M}x{N} is not divisible by the {Rx}x{Ry} layout")
        self.M, self.N, self.Rx, self.Ry, self.h = M, N, Rx, Ry, h
        self.MLOC, self.NLOC = M // Rx, N // Ry
        if self.MLOC < h or self.NLOC < h:
            raise ValueError(
                f"halo {h} wider than the local block {self.MLOC}x{self.NLOC}: "
                "a nearest-neighbour exchange cannot fill it"
            )
        self.P = Rx * Ry
        self.local_shape = (self.MLOC + 2 * h, self.NLOC + 2 * h)
        rxs, rys = np.divmod(np.arange(self.P, dtype=np.int32), Ry)
        self.rx = rxs.astype(np.int32)
        self.ry = rys.astype(np.int32)
        self.neighbour_tables = {
            name: self.rank((rxs + dx) % Rx, (rys + dy) % Ry).astype(np.int32)
            for name, (dx, dy) in NEIGHBOUR_OFFSETS.items()
        }

    def coords(self, r):
        return divmod(r, self.Ry)

    def rank(self, rx, ry):
        return (np.asarray(rx) % self.Rx) * self.Ry + (np.asarray(ry) % self.Ry)

    def owner_lane(self):
        """Per-rank source maps for the local halo box.

        Returns ``(src_dev, src_lane)``, both ``(P, MLOC+2h, NLOC+2h)`` int32.
        ``src_dev[r, i, j]`` is the rank that owns local cell ``(i-h, j-h)`` of rank
        ``r``, and ``src_lane[r, i, j]`` is that cell's index in the owner's
        row-major flattened ``(MLOC, NLOC)`` interior.
        """
        h, M, N = self.h, self.M, self.N
        nrow, ncol = self.local_shape
        dev = np.empty((self.P, nrow, ncol), dtype=np.int32)
        lane = np.empty((self.P, nrow, ncol), dtype=np.int32)
        for r in range(self.P):
            i0, j0 = self.origin(r)
            gi = (np.arange(i0 - h, i0 + self.MLOC + h) % M).reshape(-1, 1)
            gj = (np.arange(j0 - h, j0 + self.NLOC + h) % N).reshape(1, -1)
            dev[r] = (gi // self.MLOC) * self.Ry + (gj // self.NLOC)
            lane[r] = (gi % self.MLOC) * self.NLOC + (gj % self.NLOC)
        return dev, lane

    def halo_mask(self):
        """The local rim as a flat ``(nrow*ncol,)`` bool; identical on every rank."""
        h = self.h
        mask = np.ones(self.local_shape, dtype=bool)
        mask[h:-h, h:-h] = False
        return mask.reshape(-1)

    def halo_chunks(self):
        """``chunk[d][e]``: ``(sender_idx, recv_idx)`` pairs of the rim cells rank ``e``
        receives from rank ``d``, both flat local-box indices, in receiver row-major
        order."""
        h, ncol = self.h, self.local_shape[1]
        dev, lane = (a.reshape(self.P, -1) for a in self.owner_lane())
        rim = np.flatnonzero(self.halo_mask())
        chunk = [[[] for _ in range(self.P)] for _ in range(self.P)]
        for e in range(self.P):
            for c in rim:
                i, j = divmod(int(lane[e, c]), self.NLOC)
                chunk[int(dev[e, c])][e].append(((i + h) * ncol + j + h, int(c)))
        return chunk

    def origin(self, r):
        rx, ry = self.coords(r)
        return rx * self.MLOC, ry * self.NLOC

    def block_slice(self, r):
        i0, j0 = self.origin(r)
        return (slice(i0, i0 + self.MLOC), slice(j0, j0 + self.NLOC))

    def __repr__(self):
        return (
            f"Layout(M={self.M}, N={self.N}, Rx={self.Rx}, Ry={self.Ry}, h={self.h}, "
            f"MLOC={self.MLOC}, NLOC={self.NLOC}, P={self.P})"
        )


def block(g, layout: Layout):
    """Global ``(M, N)`` -> rank-major stack of interior blocks, ``(P*MLOC, NLOC)``."""
    L = layout
    x = g.reshape(L.Rx, L.MLOC, L.Ry, L.NLOC)
    return x.transpose(0, 2, 1, 3).reshape(L.P * L.MLOC, L.NLOC)


def unblock(x, layout: Layout):
    """Rank-major stack of interior blocks ``(P*MLOC, NLOC)`` -> global ``(M, N)``."""
    L = layout
    y = x.reshape(L.Rx, L.Ry, L.MLOC, L.NLOC)
    return y.transpose(0, 2, 1, 3).reshape(L.M, L.N)


def block_halo(a, layout: Layout):
    """Stack of per-rank halo arrays ``(P, MLOC+2h, NLOC+2h)`` -> ``(P*(MLOC+2h), NLOC+2h)``."""
    L = layout
    return a.reshape(L.P * L.local_shape[0], L.local_shape[1])


def unblock_halo(a, layout: Layout):
    """``(P*(MLOC+2h), NLOC+2h)`` -> ``(P, MLOC+2h, NLOC+2h)``."""
    L = layout
    return a.reshape(L.P, *L.local_shape)


def true_halo_cells(layout: Layout) -> int:
    """Cells in one local halo rim: ``2h(MLOC + NLOC) + 4h^2``."""
    h = layout.h
    return 2 * h * (layout.MLOC + layout.NLOC) + 4 * h * h


class Transport(Protocol):
    """Contract every halo transport implements.

    ``prepare`` runs once, outside jit, and returns the static NumPy/Python tables
    ``exchange`` needs. ``exchange`` runs inside a ``shard_map`` body on this device's
    ``(MLOC+2h, NLOC+2h)`` array and returns the same shape with every halo cell --
    four faces and four corners -- refreshed from its owner, periodic in both
    directions. It must be built only from ``jax.lax`` collectives and array ops (no
    ``pure_callback``, no ``custom_vjp``) so that ``jax.vjp`` transposes it.
    ``wire_cells`` is the number of cells crossing the wire per device per exchange,
    the larger of the send and receive sides.
    """

    name: str

    def prepare(self, layout: Layout) -> Any: ...

    def exchange(self, a_local: Any, tables: Any, axis_name: str) -> Any: ...

    def wire_cells(self, tables: Any) -> int: ...


REGISTRY: dict[str, Transport] = {}


def register(transport: Transport) -> Transport:
    REGISTRY[transport.name] = transport
    return transport


def import_all_transports() -> dict[str, str]:
    """Import every ``transport_*.py`` sibling; return ``{module: "ok" | error}``."""
    seen = {}
    for path in sorted(glob.glob(os.path.join(_HERE, "transport_*.py"))):
        mod = os.path.splitext(os.path.basename(path))[0]
        try:
            importlib.import_module(mod)
            seen[mod] = "ok"
        except Exception as e:  # noqa: BLE001 - a broken sibling must not hide the others
            seen[mod] = f"{type(e).__name__}: {e}"
    return seen


def get_transport(name: str) -> Transport:
    if name not in REGISTRY:
        failed = {m: e for m, e in import_all_transports().items() if e != "ok"}
        if name not in REGISTRY:
            raise KeyError(
                f"unknown transport {name!r}; registered: {sorted(REGISTRY)}"
                + (f"; modules that failed to import: {failed}" if failed else "")
            )
    return REGISTRY[name]


def available_transports() -> list[str]:
    import_all_transports()
    return list(REGISTRY)
