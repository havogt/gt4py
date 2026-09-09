# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Coloured ``lax.ppermute`` transports: the halo arrives in K permutation rounds.

FESOM2-JAX's fourth transport (``fesom_jax/halo.py::halo_exchange_coloured``): the
neighbour graph is edge-coloured into classes, each class a *partial permutation* of
the devices -- nobody sends or receives twice -- i.e. one legal ``lax.ppermute``. The
rounds are packed into one buffer (round ``r`` at ``[offs[r], offs[r]+slots[r])``), so
the send side is a single gather; the received rounds are re-concatenated and one
gather reads every halo cell out of the result::

    buf      = a_flat[send_idx]
    buf      = where(send_valid, buf, 0)
    rounds   = [ppermute(buf[o:o+s], axis, perm=p) for p, s, o in ...]
    recv     = concatenate(rounds)
    out      = where(halo_mask, recv[colpos], a_flat)

All perms / slot widths / offsets are static Python metadata (``ppermute`` needs a
Python perm). Every step is linear with a registered transpose -- gather, ``where``,
static slice, ``ppermute`` (transposes to the inverse ``ppermute``), concatenate,
gather-back, ``where`` -- so ``jax.vjp`` produces the distributed adjoint.

Two schedules, registered as two transports:

``coloured8``
    FESOM's single-phase form. On a torus each of the 8 directions is already a *full*
    permutation of the ranks (every device has exactly one E neighbour and is the E
    neighbour of exactly one device), so no greedy colouring is needed: K = 8 rounds,
    all reading the original field, all independent. E/W carry the interior edge slice
    of ``NLOC*h`` cells, N/S the interior edge slice of ``MLOC*h`` cells, the four
    diagonals the single interior corner of ``h*h`` cells. Corners come from the
    diagonal rounds.

``coloured2ph``
    The two-phase schedule a structured grid permits and an unstructured mesh cannot.
    Phase 1 is E, W on the original field, giving an x-refreshed array; phase 2 is N, S
    on *that* array, sending the edge rows including its two refreshed halo columns
    (``(MLOC+2h)*h`` cells), so the corners arrive with the faces. K = 4, but phase 2
    depends on phase 1: it is two dependent applications of the same body, the second
    reading the output of the first.

Both move ``2*NLOC*h + 2*MLOC*h + 4*h*h`` cells per device per exchange, which is
exactly the true halo rim -- ratio 1.0.

Degenerate layouts are not special-cased. ``Rx == 2`` makes E and W the same partner
(two rounds to one peer, each still a permutation). ``Rx == 1`` makes the E neighbour
the device itself; jax 0.6.2's ``lax.ppermute`` accepts self-pairs ``(i, i)`` in the
perm (checked forward and reverse), so those rounds stay collectives and the tables
need no branch. At 1x1 all 8 perms are the identity.

The tables are rank-independent -- on a structured torus every device sends and
receives the same local index sets -- so unlike ``transport_allgather`` this module
needs no ``lax.axis_index`` to select a per-device row; only the perms, which are
static Python, carry the rank structure.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import lax

from halo_transports import NEIGHBOUR_OFFSETS, Layout, register


def _spans(off, n, h, full):
    """``(send_lo, send_hi), (recv_lo, recv_hi)`` along one axis for a shift ``off``.

    ``full`` widens a zero shift from the interior to the whole local extent, which is
    how ``coloured2ph``'s phase 2 picks up the halo columns phase 1 refreshed.
    """
    if off == 1:
        return (n, n + h), (0, h)
    if off == -1:
        return (h, 2 * h), (n + h, n + 2 * h)
    if full:
        return (0, n + 2 * h), (0, n + 2 * h)
    return (h, n + h), (h, n + h)


def _flat(i0, i1, j0, j1, width):
    ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
    return (ii * width + jj).reshape(-1).astype(np.int32)


def _round(layout: Layout, name: str, full_x: bool, full_y: bool):
    L, h = layout, layout.h
    dx, dy = NEIGHBOUR_OFFSETS[name]
    (si0, si1), (ri0, ri1) = _spans(dx, L.MLOC, h, full_x)
    (sj0, sj1), (rj0, rj1) = _spans(dy, L.NLOC, h, full_y)
    width = L.local_shape[1]
    send = _flat(si0, si1, sj0, sj1, width)
    recv = _flat(ri0, ri1, rj0, rj1, width)
    assert send.size == recv.size
    nbr = L.neighbour_tables[name]
    return {
        "name": name,
        "perm": tuple((d, int(nbr[d])) for d in range(L.P)),
        "send": send,
        "recv": recv,
    }


def _phase(layout: Layout, rounds):
    L = layout
    slots, offs, acc = [], [], 0
    for r in rounds:
        slots.append(int(r["send"].size))
        offs.append(acc)
        acc += slots[-1]
    hw = L.local_shape[0] * L.local_shape[1]
    colpos = np.zeros(hw, dtype=np.int32)
    halo_mask = np.zeros(hw, dtype=bool)
    for r, o, s in zip(rounds, offs, slots):
        colpos[r["recv"]] = o + np.arange(s, dtype=np.int32)
        halo_mask[r["recv"]] = True
    return {
        "rounds": tuple(rounds),
        "perms": tuple(r["perm"] for r in rounds),
        "slots": tuple(slots),
        "offs": tuple(offs),
        "total": acc,
        "send_idx": np.concatenate([r["send"] for r in rounds]),
        # no round is padded here: every chunk of a round has the same width on every
        # device, so send_valid is identically True. Kept because the mask is what makes
        # the transpose correct once a round *is* padded (FESOM's ragged colour classes).
        "send_valid": np.ones(acc, dtype=bool),
        "colpos": colpos,
        "halo_mask": halo_mask,
    }


def _check_structure(layout: Layout, phases):
    """FESOM's structural gate, run in ``prepare``.

    Per round: no device is a source twice or a destination twice. Per phase: the
    round's ``(src, dst, cell)`` triples hit each destination cell at most once, and
    across phases every halo cell of every rank is written exactly once. Finally the
    provenance composed through the phases -- phase 2 reads cells phase 1 wrote -- is
    compared cell by cell against ``Layout.owner_lane()``, which is the statement that
    the schedule delivers the *right* value and not merely some value.
    """
    L, h = layout, layout.h
    hw = L.local_shape[0] * L.local_shape[1]
    dev, lane = (a.reshape(L.P, hw) for a in L.owner_lane())
    interior = _flat(h, h + L.MLOC, h, h + L.NLOC, L.local_shape[1])
    halo = np.setdiff1d(np.arange(hw, dtype=np.int32), interior)

    pdev = np.full((L.P, hw), -1, dtype=np.int32)
    plane = np.full((L.P, hw), -1, dtype=np.int32)
    pdev[:, interior] = dev[:, interior]
    plane[:, interior] = lane[:, interior]

    covered, n_rounds = set(), 0
    for ph in phases:
        seen, ndev, nlane = set(), pdev.copy(), plane.copy()
        for rnd in ph["rounds"]:
            n_rounds += 1
            srcs = [d for d, _ in rnd["perm"]]
            dsts = [e for _, e in rnd["perm"]]
            assert len(set(srcs)) == len(srcs), f"round {rnd['name']}: duplicate source"
            assert len(set(dsts)) == len(dsts), f"round {rnd['name']}: duplicate destination"
            for d, e in rnd["perm"]:
                for sc, rc in zip(rnd["send"], rnd["recv"]):
                    key = (e, int(rc))
                    assert key not in seen, f"round {rnd['name']}: {key} written twice in phase"
                    assert key not in covered, f"round {rnd['name']}: {key} written in an earlier phase"
                    seen.add(key)
                    ndev[e, rc], nlane[e, rc] = pdev[d, sc], plane[d, sc]
        covered |= seen
        pdev, plane = ndev, nlane

    want = {(r, int(c)) for r in range(L.P) for c in halo}
    assert covered == want, (
        f"halo cover mismatch: {len(want - covered)} uncovered, {len(covered - want)} extra"
    )
    assert np.array_equal(pdev, dev) and np.array_equal(plane, lane), "wrong owner delivered"
    return {
        "rounds": n_rounds,
        "phases": len(phases),
        "partial_permutation": True,
        "halo_cells_covered_once": len(covered),
        "provenance_matches_owner_lane": True,
    }


def _exchange_phase(a_flat, ph, axis_name: str):
    buf = a_flat[ph["send_idx"]]
    buf = jnp.where(ph["send_valid"], buf, jnp.zeros_like(buf))
    rounds = [
        lax.ppermute(buf[o : o + s], axis_name, perm=list(p))
        for p, s, o in zip(ph["perms"], ph["slots"], ph["offs"])
    ]
    recv = jnp.concatenate(rounds) if len(rounds) > 1 else rounds[0]
    return jnp.where(ph["halo_mask"], recv[ph["colpos"]], a_flat)


class ColouredTransport:
    def __init__(self, name, phase_dirs):
        self.name = name
        self._phase_dirs = phase_dirs

    def prepare(self, layout: Layout):
        phases = [
            _phase(layout, [_round(layout, d, fx, fy) for d, fx, fy in ph])
            for ph in self._phase_dirs
        ]
        return {"phases": phases, "checks": _check_structure(layout, phases)}

    def wire_cells(self, tables) -> int:
        """Sum of the round slot widths over all phases; here exactly the halo rim."""
        return sum(sum(ph["slots"]) for ph in tables["phases"])

    def exchange(self, a_local, tables, axis_name: str):
        a = a_local.reshape(-1)
        for ph in tables["phases"]:
            a = _exchange_phase(a, ph, axis_name)
        return a.reshape(a_local.shape)


register(
    ColouredTransport(
        "coloured8",
        ((("E", False, False), ("W", False, False), ("N", False, False), ("S", False, False),
          ("NE", False, False), ("NW", False, False), ("SE", False, False), ("SW", False, False)),),
    )
)

register(
    ColouredTransport(
        "coloured2ph",
        ((("E", False, False), ("W", False, False)),
         (("N", True, False), ("S", True, False))),
    )
)
