# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Slot-padded dense all-to-all transport: one ``lax.all_to_all`` slot per ordered peer pair.

Every ordered pair ``(d, e)`` gets a fixed slot of ``pad_slot`` cells (the largest
``d -> e`` chunk over all pairs; unused slots repeat cell 0 and are never read) in a
``P * pad_slot`` buffer, so one ``all_to_all`` moves every device's halo. Wire volume
``P * pad_slot`` per device; runs and transposes on every backend.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import lax

from halo_transports import Layout, register


class PaddedTransport:
    name = "padded"

    def prepare(self, layout: Layout):
        P = layout.P
        chunk = layout.halo_chunks()
        halo_mask = layout.halo_mask()
        pad_slot = max(len(chunk[d][e]) for d in range(P) for e in range(P))

        send_idx = np.zeros((P, P * pad_slot), dtype=np.int32)
        recv_pos = np.zeros((P, halo_mask.size), dtype=np.int32)
        for d in range(P):
            for e in range(P):
                for k, (sender_idx, recv_idx) in enumerate(chunk[d][e]):
                    send_idx[d, e * pad_slot + k] = sender_idx
                    recv_pos[e, recv_idx] = d * pad_slot + k
        return {"send_idx": send_idx, "recv_pos": recv_pos, "halo_mask": halo_mask}

    def wire_cells(self, tables) -> int:
        return tables["send_idx"].shape[1]

    def exchange(self, a_local, tables, axis_name: str):
        r = lax.axis_index(axis_name)
        a_flat = a_local.reshape(-1)
        buf = a_flat[jnp.asarray(tables["send_idx"])[r]]
        recv = lax.all_to_all(buf, axis_name, split_axis=0, concat_axis=0, tiled=True)
        gathered = recv[jnp.asarray(tables["recv_pos"])[r]]
        out = jnp.where(jnp.asarray(tables["halo_mask"]), gathered, a_flat)
        return out.reshape(a_local.shape)


register(PaddedTransport())
