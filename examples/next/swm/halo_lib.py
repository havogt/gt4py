# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""An illustrative halo-exchange library: R = Rx * Ry ranks emulated in one process.

Support module for `nb04_halo_exchange_patterns_2d.ipynb`. Every message is an
explicit object so the notebook can print, count and transpose them. The pieces
map one-to-one onto an MPI or GHEX implementation:

    Decomposition   the Cartesian communicator plus per-rank domain descriptors
    Message         one Isend/Irecv pair (source buffer slice -> destination buffer slice)
    pattern         the list of phases, each phase a list of messages posted together
    exchange        post every message of a phase, wait, next phase

Local blocks are `(mloc + 2h, nloc + 2h)` NumPy arrays; index `[h:-h, h:-h]` is
the interior. Rank `r` has Cartesian coordinates `(r // Ry, r % Ry)`; both
directions are periodic.
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

    def __str__(self):
        return f"rank {self.src} -> rank {self.dst}: {_fmt(self.src_slice)} -> {_fmt(self.dst_slice)}"

    @property
    def size(self):
        return _slice_len(self.src_slice[0]) * _slice_len(self.src_slice[1])


def _fmt(sl):
    return f"[rows {sl[0].start}:{sl[0].stop}, cols {sl[1].start}:{sl[1].stop}]"


def _slice_len(s):
    return s.stop - s.start


class Decomposition:
    def __init__(self, M, N, Rx, Ry, halo=1):
        assert M % Rx == 0 and N % Ry == 0
        self.M, self.N, self.Rx, self.Ry, self.h = M, N, Rx, Ry, halo
        self.mloc, self.nloc = M // Rx, N // Ry
        assert self.mloc >= halo and self.nloc >= halo, (
            f"halo {halo} wider than the local block {self.mloc} x {self.nloc}: "
            "a nearest-neighbour pattern can only fill a halo of at most one block width")
        self.R = Rx * Ry
        self.local_shape = (self.mloc + 2 * halo, self.nloc + 2 * halo)

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
            b[self.h : -self.h, self.h : -self.h] = global_arr[i0 : i0 + self.mloc, j0 : j0 + self.nloc]
            blocks.append(b)
        return blocks

    def gather(self, blocks):
        out = np.empty((self.M, self.N), dtype=blocks[0].dtype)
        for r, b in enumerate(blocks):
            i0, j0 = self.origin(r)
            out[i0 : i0 + self.mloc, j0 : j0 + self.nloc] = b[self.h : -self.h, self.h : -self.h]
        return out

    # --- slices of the local buffer -------------------------------------------------------
    def rows(self, which):
        h, m = self.h, self.mloc
        return {"lo_halo": slice(0, h), "lo_int": slice(h, 2 * h), "hi_int": slice(m, m + h),
                "hi_halo": slice(m + h, m + 2 * h), "int": slice(h, m + h), "all": slice(0, m + 2 * h)}[which]

    def cols(self, which):
        h, n = self.h, self.nloc
        return {"lo_halo": slice(0, h), "lo_int": slice(h, 2 * h), "hi_int": slice(n, n + h),
                "hi_halo": slice(n + h, n + 2 * h), "int": slice(h, n + h), "all": slice(0, n + 2 * h)}[which]

    # --- message lists ----------------------------------------------------------------------
    def x_messages(self, cols="int"):
        """Face messages along x: my last interior rows -> right neighbour's low halo, and back."""
        msgs = []
        for r in range(self.R):
            px, py = self.coords(r)
            c = self.cols(cols)
            msgs.append(Message(r, self.rank(px + 1, py), (self.rows("hi_int"), c), (self.rows("lo_halo"), c), "x"))
            msgs.append(Message(r, self.rank(px - 1, py), (self.rows("lo_int"), c), (self.rows("hi_halo"), c), "x"))
        return msgs

    def y_messages(self, rows="int"):
        """Face messages along y. With rows="all" the x halos travel too, which fills the corners."""
        msgs = []
        for r in range(self.R):
            px, py = self.coords(r)
            rw = self.rows(rows)
            msgs.append(Message(r, self.rank(px, py + 1), (rw, self.cols("hi_int")), (rw, self.cols("lo_halo")), "y"))
            msgs.append(Message(r, self.rank(px, py - 1), (rw, self.cols("lo_int")), (rw, self.cols("hi_halo")), "y"))
        return msgs

    def corner_messages(self):
        msgs = []
        for r in range(self.R):
            px, py = self.coords(r)
            for dx, dy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                src = (self.rows("hi_int" if dx > 0 else "lo_int"), self.cols("hi_int" if dy > 0 else "lo_int"))
                dst = (self.rows("lo_halo" if dx > 0 else "hi_halo"), self.cols("lo_halo" if dy > 0 else "hi_halo"))
                msgs.append(Message(r, self.rank(px + dx, py + dy), src, dst, "corner"))
        return msgs

    # --- patterns: a pattern is a list of phases; a phase is a list of messages --------------
    def two_phase_pattern(self):
        return [self.x_messages(cols="int"), self.y_messages(rows="all")]

    def single_phase_pattern(self):
        return [self.x_messages(cols="int") + self.y_messages(rows="int") + self.corner_messages()]

    def faces_only_pattern(self):
        return [self.x_messages(cols="int") + self.y_messages(rows="int")]

    def ring_pattern(self):
        """1-D decomposition along x: the y direction is not decomposed and nothing is exchanged there."""
        assert self.Ry == 1
        return [self.x_messages(cols="int")]


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
    """The 'scratch-buffer trick': carry halo cotangents backwards with the FORWARD exchange.

    Reflect every halo cotangent into the interior cell across the block boundary, run the
    ordinary forward exchange on that scratch buffer, and read the result back from the halo
    slots on the other side. Exact when every interior slot feeds a single neighbour, i.e. a
    1-D ring with mloc >= 2h. Wrong on thinner blocks, and wrong in 2-D, where a corner cell
    feeds three."""
    scratch = [np.zeros_like(b) for b in blocks_bar]
    for phase in pattern:
        for m in phase:
            scratch[m.dst][_reflect(decomp, m.dst_slice)] += blocks_bar[m.dst][m.dst_slice]
    exchange(scratch, pattern)
    for phase in pattern:
        for m in phase:
            blocks_bar[m.dst][m.dst_slice] = 0.0
    for phase in pattern:
        for m in phase:
            blocks_bar[m.src][m.src_slice] += scratch[m.src][_reflect(decomp, m.src_slice)]
    return blocks_bar


def _reflect(decomp, sl):
    """Mirror a halo slice into the interior across the block boundary, and vice versa;
    slices that are not halo/edge slices are left alone."""
    pairs = (("lo_halo", "lo_int"), ("lo_int", "lo_halo"), ("hi_halo", "hi_int"), ("hi_int", "hi_halo"))

    def mirror(s, axis):
        for a, b in pairs:
            if s == axis(a):
                return axis(b)
        return s

    return (mirror(sl[0], decomp.rows), mirror(sl[1], decomp.cols))


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


def unstack(arr):
    return [np.array(a) for a in arr]
