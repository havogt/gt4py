# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Slot-padded dense all-to-all transport: one ``all_to_all`` slot per ordered peer pair.

FESOM2-JAX's ``halo_exchange_padded`` (fesom_jax/halo.py:152-178), adapted to the
structured torus. Every ordered device pair ``(d, e)`` gets a fixed-width slot of
``pad_slot`` cells (the largest ``d -> e`` halo chunk over all pairs, zero-padded)
in a ``P * pad_slot`` send/recv buffer, so ONE ``lax.all_to_all`` moves every
device's boundary at once. Unlike ``ragged_all_to_all`` (unimplemented on
XLA:CPU, and with a reverse-mode rule FESOM found broken everywhere), dense
``all_to_all`` runs on every backend and transposes to another ``all_to_all``
(one of JAX's oldest rules), so the gradient is correct by construction. The
``where(pad_valid, ...)`` before the collective is load-bearing for that
transpose: it kills the cotangents of the duplicated pad-slot gathers.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from halo_transports import Layout, register


class PaddedTransport:
    name = "padded"

    def prepare(self, layout: Layout):
        L = layout
        h, P = L.h, L.P
        H, W = L.local_shape
        MLOC, NLOC = L.MLOC, L.NLOC
        src_dev, src_lane = L.owner_lane()  # (P, H, W)

        ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        interior = (ii >= h) & (ii < H - h) & (jj >= h) & (jj < W - h)
        halo = ~interior  # (H, W); the rim is the same shape for every rank

        # owner-interior lane (row-major over the owner's (MLOC, NLOC) interior) ->
        # the owner's OWN full local-box (H, W) flat index. Same map for every rank:
        # every rank's local box has the same shape.
        li, lj = np.divmod(np.arange(MLOC * NLOC), NLOC)
        lane_to_full = (li + h) * W + (lj + h)

        # chunk[d][e]: the (sender-local, receiver-local) full-box flat index pairs
        # of the cells device e's halo needs from device d, in receiver-flat order.
        chunk = [[[] for _ in range(P)] for _ in range(P)]
        for e in range(P):
            for i, j in zip(*np.nonzero(halo)):
                d = int(src_dev[e, i, j])
                sender_idx = int(lane_to_full[int(src_lane[e, i, j])])
                chunk[d][e].append((sender_idx, i * W + j))

        pad_slot = max((len(chunk[d][e]) for d in range(P) for e in range(P)), default=1)
        pad_slot = max(pad_slot, 1)

        pad_src = np.zeros((P, P * pad_slot), dtype=np.int32)
        pad_valid = np.zeros((P, P * pad_slot), dtype=bool)
        pad_slotpos = np.zeros((P, H * W), dtype=np.int32)
        halo_mask = np.tile(halo.reshape(1, -1), (P, 1))

        for d in range(P):
            for e in range(P):
                for k, (sender_idx, recv_idx) in enumerate(chunk[d][e]):
                    pad_src[d, e * pad_slot + k] = sender_idx
                    pad_valid[d, e * pad_slot + k] = True
                    pad_slotpos[e, recv_idx] = d * pad_slot + k

        return {
            "layout": L,
            "pad_src": pad_src,
            "pad_valid": pad_valid,
            "pad_slotpos": pad_slotpos,
            "halo_mask": halo_mask,
            "pad_slot": pad_slot,
        }

    def wire_cells(self, tables) -> int:
        """One padded slot per ordered peer: ``P * pad_slot``, zeros included."""
        return tables["layout"].P * tables["pad_slot"]

    def exchange(self, a_local, tables, axis_name: str):
        L: Layout = tables["layout"]
        H, W = L.local_shape
        r = jax.lax.axis_index(axis_name)
        pad_src = jnp.asarray(tables["pad_src"])[r]
        pad_valid = jnp.asarray(tables["pad_valid"])[r]
        pad_slotpos = jnp.asarray(tables["pad_slotpos"])[r]
        halo_mask = jnp.asarray(tables["halo_mask"])[r]

        a_flat = a_local.reshape(-1)
        buf = a_flat[pad_src]
        buf = jnp.where(pad_valid, buf, 0.0)
        recv = jax.lax.all_to_all(buf, axis_name, split_axis=0, concat_axis=0, tiled=True)
        gathered = recv[pad_slotpos]
        out = jnp.where(halo_mask, gathered, a_flat)
        return out.reshape(H, W)


register(PaddedTransport())
