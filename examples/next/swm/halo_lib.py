# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""An illustrative halo-exchange library: R = Rx * Ry ranks emulated in one process.

Every message is an explicit object that can be printed, counted and transposed.
The pieces map one-to-one onto an MPI or GHEX implementation:

    Decomposition   the Cartesian communicator plus per-rank domain descriptors
    Message         one Isend/Irecv pair (source buffer slice -> destination buffer slice)
    pattern         the list of phases, each phase a list of messages posted together
    exchange        post every message of a phase, wait, next phase

Local blocks are `(mloc + 2*halo, nloc + 2*halo)` NumPy arrays; index
`[halo:-halo, halo:-halo]` is the interior. Rank `r` has Cartesian coordinates
`(r // Ry, r % Ry)`; both directions are periodic.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Message:
    src: int
    dst: int
    src_slice: tuple
    dst_slice: tuple
    phase: str
    src_names: tuple
    dst_names: tuple

    def __str__(self):
        return (
            f"rank {self.src} -> rank {self.dst}: {_fmt(self.src_slice)} -> {_fmt(self.dst_slice)}"
        )

    @property
    def size(self):
        return _slice_len(self.src_slice[0]) * _slice_len(self.src_slice[1])


def _fmt(sl):
    return f"[rows {sl[0].start}:{sl[0].stop}, cols {sl[1].start}:{sl[1].stop}]"


def _slice_len(s):
    return s.stop - s.start


def _slices(n, halo):
    return {
        "lo_halo": slice(0, halo),
        "lo_int": slice(halo, 2 * halo),
        "int": slice(halo, n + halo),
        "hi_int": slice(n, n + halo),
        "hi_halo": slice(n + halo, n + 2 * halo),
        "all": slice(0, n + 2 * halo),
    }


_MIRROR = {"lo_halo": "lo_int", "lo_int": "lo_halo", "hi_halo": "hi_int", "hi_int": "hi_halo"}


class Decomposition:
    def __init__(self, M, N, Rx, Ry, halo=1):
        assert M % Rx == 0 and N % Ry == 0
        self.M, self.N, self.Rx, self.Ry, self.halo = M, N, Rx, Ry, halo
        self.R = Rx * Ry
        self.mloc, self.nloc = M // Rx, N // Ry
        self.local_shape = (self.mloc + 2 * halo, self.nloc + 2 * halo)
        self.rows, self.cols = _slices(self.mloc, halo), _slices(self.nloc, halo)

    def coords(self, rank):
        return divmod(rank, self.Ry)

    def rank(self, px, py):
        return (px % self.Rx) * self.Ry + (py % self.Ry)

    def origin(self, rank):
        px, py = self.coords(rank)
        return px * self.mloc, py * self.nloc

    def scatter(self, global_arr):
        blocks = []
        for r in range(self.R):
            i0, j0 = self.origin(r)
            b = np.zeros(self.local_shape, dtype=global_arr.dtype)
            b[self.rows["int"], self.cols["int"]] = global_arr[
                i0 : i0 + self.mloc, j0 : j0 + self.nloc
            ]
            blocks.append(b)
        return blocks

    def gather(self, blocks):
        out = np.empty((self.M, self.N), dtype=blocks[0].dtype)
        for r, b in enumerate(blocks):
            i0, j0 = self.origin(r)
            out[i0 : i0 + self.mloc, j0 : j0 + self.nloc] = b[self.rows["int"], self.cols["int"]]
        return out

    def slices(self, rows, cols):
        return (self.rows[rows], self.cols[cols])

    def mirror(self, names):
        """Halo slice <-> the interior slice just across the block boundary; others unchanged."""
        return self.slices(*(_MIRROR.get(n, n) for n in names))

    def _message(self, src, dst, src_names, dst_names, phase):
        return Message(
            src, dst, self.slices(*src_names), self.slices(*dst_names), phase, src_names, dst_names
        )

    # --- message lists ----------------------------------------------------------------------
    def x_messages(self):
        """Face messages along x: last interior rows -> the x+1 neighbour's low halo, and back."""
        msgs = []
        for r in range(self.R):
            px, py = self.coords(r)
            hi, lo = self.rank(px + 1, py), self.rank(px - 1, py)
            msgs.append(self._message(r, hi, ("hi_int", "int"), ("lo_halo", "int"), "x"))
            msgs.append(self._message(r, lo, ("lo_int", "int"), ("hi_halo", "int"), "x"))
        return msgs

    def y_messages(self, with_x_halos=False):
        """Face messages along y; sending the x halos along as well fills the corners."""
        rows = "all" if with_x_halos else "int"
        msgs = []
        for r in range(self.R):
            px, py = self.coords(r)
            hi, lo = self.rank(px, py + 1), self.rank(px, py - 1)
            msgs.append(self._message(r, hi, (rows, "hi_int"), (rows, "lo_halo"), "y"))
            msgs.append(self._message(r, lo, (rows, "lo_int"), (rows, "hi_halo"), "y"))
        return msgs

    def corner_messages(self):
        msgs = []
        for r in range(self.R):
            px, py = self.coords(r)
            for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                src = ("hi_int" if dx > 0 else "lo_int", "hi_int" if dy > 0 else "lo_int")
                dst = ("lo_halo" if dx > 0 else "hi_halo", "lo_halo" if dy > 0 else "hi_halo")
                msgs.append(self._message(r, self.rank(px + dx, py + dy), src, dst, "corner"))
        return msgs

    # --- patterns: a pattern is a list of phases; a phase is a list of messages --------------
    def two_phase_pattern(self):
        return [self.x_messages(), self.y_messages(with_x_halos=True)]

    def single_phase_pattern(self):
        return [self.x_messages() + self.y_messages() + self.corner_messages()]

    def faces_only_pattern(self):
        return [self.x_messages() + self.y_messages()]

    def ring_pattern(self):
        assert self.Ry == 1
        return [self.x_messages()]


# --- forward ---------------------------------------------------------------------------------
def exchange(blocks, pattern, log=None):
    """Fill the halos of `blocks` in place. Within a phase every message reads the state at the
    start of the phase (post all, then wait all); phases run in sequence."""
    for phase in pattern:
        packed = [(m, blocks[m.src][m.src_slice].copy()) for m in phase]
        for m, buf in packed:
            blocks[m.dst][m.dst_slice] = buf
            if log is not None:
                log.append(m)
    return blocks


# --- transpose --------------------------------------------------------------------------------
def exchange_adjoint(blocks_bar, pattern, accumulate=True, zero_halo=True):
    """Transpose of `exchange` with the same pattern, in place.

    Phases in reverse order, every message reversed (dst -> src), assignment replaced by
    accumulation into the owner, and the halo cotangent zeroed once it has been sent.
    The two keyword switches exist only to reproduce the classic mistakes."""
    for phase in reversed(pattern):
        packed = [(m, blocks_bar[m.dst][m.dst_slice].copy()) for m in phase]
        if zero_halo:
            for m, _ in packed:
                blocks_bar[m.dst][m.dst_slice] = 0.0
        for m, buf in packed:
            if accumulate:
                blocks_bar[m.src][m.src_slice] += buf
            else:
                blocks_bar[m.src][m.src_slice] = buf
    return blocks_bar


def exchange_adjoint_scratch(decomp, blocks_bar, pattern):
    """Carry halo cotangents backwards with the FORWARD exchange on a scratch buffer.

    Exact only when every interior slot feeds a single neighbour: a 1-D ring with mloc >= 2*halo."""
    scratch = [np.zeros_like(b) for b in blocks_bar]
    for phase in pattern:
        for m in phase:
            scratch[m.dst][decomp.mirror(m.dst_names)] += blocks_bar[m.dst][m.dst_slice]
            blocks_bar[m.dst][m.dst_slice] = 0.0
    exchange(scratch, pattern)
    for phase in pattern:
        for m in phase:
            blocks_bar[m.src][m.src_slice] += scratch[m.src][decomp.mirror(m.src_names)]
    return blocks_bar


# --- the dense matrix, for tiny cases --------------------------------------------------------
def exchange_matrix(decomp, pattern):
    """Dense matrix A with  stack(exchange(blocks)) == A @ stack(blocks)  (row-major flattening)."""
    n = decomp.R * decomp.local_shape[0] * decomp.local_shape[1]
    A = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = 1.0
        blocks = list(e.reshape(decomp.R, *decomp.local_shape))
        A[:, k] = np.stack(exchange(blocks, pattern)).ravel()
    return A


def unstack(stacked):
    # copies, not views: exchange and its adjoints write into the blocks in place
    return [np.array(a) for a in stacked]
