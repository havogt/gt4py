# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Ragged all-to-all transport, and a CPU-runnable emulation of it.

Tables, per device ``d`` and peer ``e``, with ``chunk[d][e]`` from ``Layout.halo_chunks``:
``send_idx[d]`` is the sender indices of ``chunk[d][e]`` concatenated over ``e``;
``send_sizes[d, e] = len(chunk[d][e])`` and ``recv_sizes = send_sizes.T``;
``input_offsets`` is the exclusive row cumsum of ``send_sizes``, ``output_offsets`` the
transpose of that of ``recv_sizes`` (where *my* slice for peer ``e`` lands in ``e``'s
receive buffer); ``rim`` is the flat indices of the rim cells and ``halo_mask`` the same
as a flat bool (both identical on every rank); ``recv_pos[e]`` the slot each rim cell of
device ``e`` is read from.

``ragged``: ``lax.ragged_all_to_all`` ships exactly the cells the peers need, the minimal
wire volume here. Not implemented on XLA:CPU, and its transpose rule is defective, so it
is forward-only.

``ragged_emul``: the same tables and pipeline, with the primitive replaced by an
``all_gather`` plus a static index pair derived only from the six arguments the primitive
would receive, so a wrong table is a wrong result. Runs and transposes on every backend;
its own wire volume is that of ``all_gather``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import lax

from halo_transports import Layout, register, set_rim


def _tables(layout: Layout):
    P = layout.P
    chunk = layout.halo_chunks()
    mask = layout.halo_mask()
    rim = np.flatnonzero(mask).astype(np.int32)

    send_sizes = np.array([[len(chunk[d][e]) for e in range(P)] for d in range(P)], dtype=np.int32)
    recv_sizes = send_sizes.T.copy()
    input_offsets = np.cumsum(send_sizes, axis=1, dtype=np.int32) - send_sizes
    output_offsets = (np.cumsum(recv_sizes, axis=1, dtype=np.int32) - recv_sizes).T.copy()

    send_idx = np.array(
        [[s for e in range(P) for s, _ in chunk[d][e]] for d in range(P)], dtype=np.int32
    )
    pos = np.zeros((P, mask.size), dtype=np.int32)
    for e in range(P):
        recv = [r for d in range(P) for _, r in chunk[d][e]]
        pos[e, recv] = np.arange(len(recv), dtype=np.int32)
    return {
        "send_idx": send_idx,
        "send_sizes": send_sizes,
        "input_offsets": input_offsets,
        "output_offsets": output_offsets,
        "recv_sizes": recv_sizes,
        "recv_pos": pos[:, rim],
        "rim": rim,
        "halo_mask": mask,
        "recv_max": int(recv_sizes.sum(axis=1).max()),
    }


def _move_ragged(operand, tables, axis_name, r):
    return lax.ragged_all_to_all(
        operand,
        jnp.zeros((tables["recv_max"],), operand.dtype),
        jnp.asarray(tables["input_offsets"])[r],
        jnp.asarray(tables["send_sizes"])[r],
        jnp.asarray(tables["output_offsets"])[r],
        jnp.asarray(tables["recv_sizes"])[r],
        axis_name=axis_name,
    )


class RaggedTransport:
    def __init__(self, name, tables, move):
        self.name = name
        self._tables = tables
        self._move = move

    def prepare(self, layout: Layout):
        return self._tables(layout)

    def wire_cells(self, tables) -> int:
        return tables["send_idx"].shape[1]

    def exchange(self, a_local, tables, axis_name: str):
        r = lax.axis_index(axis_name)
        a_flat = a_local.reshape(-1)
        operand = a_flat[jnp.asarray(tables["send_idx"])[r]]
        recv = self._move(operand, tables, axis_name, r)
        vals = recv[jnp.asarray(tables["recv_pos"])[r]]
        return set_rim(a_flat, tables["rim"], tables["halo_mask"], vals).reshape(a_local.shape)


def _emul_tables(layout: Layout):
    """The primitive's receive semantics as a static index pair:
    ``recv[o] = operand_of(src_dev[o])[src_pos[o]]`` where ``valid[o]``."""
    t = _tables(layout)
    P, recv_max = layout.P, t["recv_max"]
    # all_to_all(output_offsets): local_offsets[e, d] == output_offsets[d, e]
    local_offsets = t["output_offsets"].T
    src_dev = np.zeros((P, recv_max), dtype=np.int32)
    src_pos = np.zeros((P, recv_max), dtype=np.int32)
    valid = np.zeros((P, recv_max), dtype=bool)
    for e in range(P):
        for d in range(P):
            k = np.arange(t["recv_sizes"][e, d], dtype=np.int32)
            o = local_offsets[e, d] + k
            src_dev[e, o], src_pos[e, o], valid[e, o] = d, t["input_offsets"][d, e] + k, True
    return {**t, "emul_src_dev": src_dev, "emul_src_pos": src_pos, "emul_valid": valid}


def _move_emul(operand, tables, axis_name, r):
    every = lax.all_gather(operand, axis_name, axis=0, tiled=False)  # (P, send_max)
    picked = every[jnp.asarray(tables["emul_src_dev"])[r], jnp.asarray(tables["emul_src_pos"])[r]]
    return jnp.where(jnp.asarray(tables["emul_valid"])[r], picked, 0.0)


register(RaggedTransport("ragged", _tables, _move_ragged))
register(RaggedTransport("ragged_emul", _emul_tables, _move_emul))
