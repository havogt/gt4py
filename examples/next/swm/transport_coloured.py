# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Coloured ``lax.ppermute`` transports: the halo arrives in K permutation rounds.

On a torus each of the 8 directions is a full permutation of the ranks, so each is one
legal ``ppermute``. The rounds of a phase are packed into one send buffer; the received
rounds are concatenated and one gather reads every halo cell out of the result. All
tables are rank-independent; only the perms carry the rank structure.

``coloured8``
    One phase of 8 independent rounds on the original field: E/W carry ``NLOC*h``
    cells, N/S ``MLOC*h``, the four diagonals ``h*h`` (the corners).

``coloured2ph``
    Phase 1 is E, W; phase 2 is N, S on the x-refreshed array, sending the edge columns
    including the halo rows phase 1 filled (``(MLOC+2h)*h`` cells), so the corners
    arrive with the faces. K = 4, two dependent applications of the same body.

Both move exactly the halo rim, ``2h(MLOC+NLOC) + 4h^2`` cells per device per exchange.

Degenerate layouts are not special-cased: ``Rx == 2`` makes E and W the same partner
(two rounds to one peer, each still a permutation); ``Rx == 1`` makes the E neighbour
the device itself, a self-pair ``(i, i)`` in the perm, which ``lax.ppermute`` accepts
forward and reverse.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import lax

from halo_transports import NEIGHBOUR_OFFSETS, Layout, register


def _spans(off, n, h, full):
    """``(send_lo, send_hi), (recv_lo, recv_hi)`` along one axis for a shift ``off``.

    ``full`` widens a zero shift from the interior to the whole local extent.
    """
    if off == 1:
        return (n, n + h), (0, h)
    if off == -1:
        return (h, 2 * h), (n + h, n + 2 * h)
    if full:
        return (0, n + 2 * h), (0, n + 2 * h)
    return (h, n + h), (h, n + h)


def _flat(i0, i1, j0, j1, ncol):
    ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
    return (ii * ncol + jj).reshape(-1).astype(np.int32)


def _round(layout: Layout, name: str, full_x: bool):
    h = layout.h
    dx, dy = NEIGHBOUR_OFFSETS[name]
    (si0, si1), (ri0, ri1) = _spans(dx, layout.MLOC, h, full_x)
    (sj0, sj1), (rj0, rj1) = _spans(dy, layout.NLOC, h, False)
    ncol = layout.local_shape[1]
    perm = tuple((d, int(nbr)) for d, nbr in enumerate(layout.neighbour_tables[name]))
    return perm, _flat(si0, si1, sj0, sj1, ncol), _flat(ri0, ri1, rj0, rj1, ncol)


def _phase(layout: Layout, dirs, full_x=False):
    rounds = [_round(layout, d, full_x) for d in dirs]
    slots = tuple(send.size for _, send, _ in rounds)
    offs = tuple(int(o) for o in np.cumsum((0,) + slots[:-1]))
    ncell = layout.local_shape[0] * layout.local_shape[1]
    recv_pos = np.zeros(ncell, dtype=np.int32)
    halo_mask = np.zeros(ncell, dtype=bool)  # the cells this phase writes
    for (_, _, recv), o, s in zip(rounds, offs, slots):
        recv_pos[recv] = o + np.arange(s, dtype=np.int32)
        halo_mask[recv] = True
    return {
        "perms": tuple(perm for perm, _, _ in rounds),
        "slots": slots,
        "offs": offs,
        "send_idx": np.concatenate([send for _, send, _ in rounds]),
        "recv_pos": recv_pos,
        "halo_mask": halo_mask,
    }


def _exchange_phase(a_flat, ph, axis_name: str):
    buf = a_flat[ph["send_idx"]]
    rounds = [
        lax.ppermute(buf[o : o + s], axis_name, perm=list(p))
        for p, s, o in zip(ph["perms"], ph["slots"], ph["offs"])
    ]
    return jnp.where(ph["halo_mask"], jnp.concatenate(rounds)[ph["recv_pos"]], a_flat)


class ColouredTransport:
    def __init__(self, name, phases):
        self.name = name
        self._phases = phases

    def prepare(self, layout: Layout):
        return {"phases": [_phase(layout, **ph) for ph in self._phases]}

    def wire_cells(self, tables) -> int:
        return sum(sum(ph["slots"]) for ph in tables["phases"])

    def exchange(self, a_local, tables, axis_name: str):
        a = a_local.reshape(-1)
        for ph in tables["phases"]:
            a = _exchange_phase(a, ph, axis_name)
        return a.reshape(a_local.shape)


register(ColouredTransport("coloured8", [dict(dirs=("E", "W", "N", "S", "NE", "NW", "SE", "SW"))]))
register(
    ColouredTransport("coloured2ph", [dict(dirs=("E", "W")), dict(dirs=("N", "S"), full_x=True)])
)
