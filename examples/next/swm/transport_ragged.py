# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Ragged all-to-all halo transport (FESOM2-JAX form), and a CPU-runnable emulation.

Two transports share one set of tables and one pipeline; they differ only in the
collective that moves the bytes.

``"ragged"`` is FESOM's ``halo_exchange_ragged`` on the structured torus::

    operand = a_flat[send_idx]  # [send_max], grouped by destination
    recv = lax.ragged_all_to_all(
        operand,
        jnp.zeros(recv_max),
        input_offsets,
        send_sizes,
        output_offsets,
        recv_sizes,
        axis_name,
    )
    out = jnp.where(halo_mask, recv[recv_gather], a_flat)

It ships exactly the cells the neighbours need and nothing else -- the minimal wire
volume of any of the transports here. It **does not run on XLA:CPU** (``UNIMPLEMENTED:
HLO opcode `ragged-all-to-all` is not supported by XLA:CPU ThunkEmitter``, raised as
``XlaRuntimeError`` on jax 0.6.2 and ``JaxRuntimeError`` on 0.11.1), and where it does run
its reverse-mode
transpose rule is defective: ``jax._src.lax.parallel._ragged_all_to_all_transpose``
violates ``<f(x),y> == <x,f^T(y)>`` by O(1) and returns ``axis_size`` times the correct
``f^T(ones)`` (FESOM2-JAX, ``docs/JAX_RAGGED_A2A_BUG.md``). So this transport is
forward-only in practice; the battery's adjoint gates cannot be met with it.

``"ragged_emul"`` uses the *same* tables and the same pipeline, with the primitive
replaced by a pure-jnp emulation of its documented semantics built from ``all_gather``
plus a static index pair -- both CPU-capable and both correctly transposed by JAX. Its
job is to validate the offset/size tables end to end (forward *and* adjoint) on the
backend that cannot run the primitive itself. The emulation is derived only from the
six arguments the primitive would receive, so a wrong table is a wrong result.

Tables (per device ``d``, peer ``e``; ``chunk[d][e]`` = the halo cells of ``e`` owned by
``d``, in increasing receiver-local row-major order):

* ``send_idx[d]``    -- ``chunk[d][e]``'s sender-local box indices, concatenated over
  ``e`` in rank order, padded to ``send_max`` with a valid interior index.
* ``send_sizes[d,e] = len(chunk[d][e])``, ``input_offsets[d]`` its exclusive cumsum.
* ``recv_sizes[e,d] = len(chunk[d][e])``, ``recv_offsets[e]`` its exclusive cumsum.
* ``output_offsets = recv_offsets.T`` -- the docstring's convention: ``output_offsets[i]``
  is the offset at which *my* slice for peer ``i`` lands in **peer i's** output buffer.
* ``recv_gather[e]`` -- per local cell of ``e``, its slot in ``e``'s receive buffer.

Degenerate layouts need no special case: ``Rx == 2`` makes E and W the same peer, so
that peer's chunk simply holds both strips; ``Rx == 1`` (or ``P == 1``) makes the peer
``e == d`` and the chunk becomes a self-message, which ``ragged_all_to_all`` allows.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import lax

from halo_transports import Layout, register


def _chunks(layout: Layout):
    """``chunk[d][e]`` = ``[(receiver-local box index, sender-local box index), ...]``."""
    L, h = layout, layout.h
    H, W = L.local_shape
    dev, lane = L.owner_lane()
    halo = np.ones((H, W), dtype=bool)
    halo[h : H - h, h : W - h] = False
    chunk = [[[] for _ in range(L.P)] for _ in range(L.P)]
    for e in range(L.P):
        for ii, jj in zip(*np.nonzero(halo)):  # row-major: the canonical order
            i, j = divmod(int(lane[e, ii, jj]), L.NLOC)
            chunk[int(dev[e, ii, jj])][e].append((int(ii) * W + int(jj), (i + h) * W + (j + h)))
    return chunk, halo


def _emul_maps(input_offsets, send_sizes, output_offsets, recv_sizes, recv_max):
    """The primitive's documented receive semantics, as a static index pair.

    ``recv[o] = operand_of(src_dev[o])[src_pos[o]]`` where ``o`` runs over the receive
    buffer, taken straight from the docstring: ``output_offsets_ = all_to_all(
    output_offsets)`` gives the local offsets, slice ``i`` comes from device ``i``, and
    the source is that device's ``operand[input_offsets[i] : +send_sizes[i]]``.
    """
    P = input_offsets.shape[0]
    if not np.array_equal(send_sizes, recv_sizes.T):
        raise ValueError("send_sizes != all_to_all(recv_sizes): the primitive's precondition")
    oo_ = output_offsets.T  # all_to_all(output_offsets): oo_[e, d] == output_offsets[d, e]
    src_dev = np.zeros((P, recv_max), dtype=np.int32)
    src_pos = np.zeros((P, recv_max), dtype=np.int32)
    valid = np.zeros((P, recv_max), dtype=bool)
    for e in range(P):
        for d in range(P):
            n, o, i0 = int(recv_sizes[e, d]), int(oo_[e, d]), int(input_offsets[d, e])
            k = np.arange(n, dtype=np.int32)
            if valid[e, o + k].any():
                raise ValueError(f"receive slots overlap on device {e} for source {d}")
            src_dev[e, o + k], src_pos[e, o + k], valid[e, o + k] = d, i0 + k, True
    return src_dev, src_pos, valid


def _tables(layout: Layout):
    L, h = layout, layout.h
    H, W = L.local_shape
    P = L.P
    chunk, halo = _chunks(L)

    send_sizes = np.zeros((P, P), dtype=np.int32)
    recv_sizes = np.zeros((P, P), dtype=np.int32)
    for d in range(P):
        for e in range(P):
            send_sizes[d, e] = recv_sizes[e, d] = len(chunk[d][e])
    input_offsets = np.zeros((P, P), dtype=np.int32)
    recv_offsets = np.zeros((P, P), dtype=np.int32)
    input_offsets[:, 1:] = np.cumsum(send_sizes, axis=1)[:, :-1]
    recv_offsets[:, 1:] = np.cumsum(recv_sizes, axis=1)[:, :-1]
    output_offsets = recv_offsets.T.copy()

    send_max = int(send_sizes.sum(axis=1).max())
    recv_max = int(recv_sizes.sum(axis=1).max())
    send_idx = np.full((P, send_max), h * W + h, dtype=np.int32)  # pad: a real interior cell
    recv_gather = np.zeros((P, H * W), dtype=np.int32)
    for d in range(P):
        pos = 0
        for e in range(P):
            for _, sbox in chunk[d][e]:
                send_idx[d, pos] = sbox
                pos += 1
    for e in range(P):
        pos = 0
        for d in range(P):
            for rbox, _ in chunk[d][e]:
                recv_gather[e, rbox] = pos
                pos += 1

    src_dev, src_pos, valid = _emul_maps(
        input_offsets, send_sizes, output_offsets, recv_sizes, recv_max
    )
    return {
        "layout": L,
        "send_idx": send_idx,
        "send_sizes": send_sizes,
        "input_offsets": input_offsets,
        "output_offsets": output_offsets,
        "recv_sizes": recv_sizes,
        "recv_offsets": recv_offsets,
        "recv_gather": recv_gather,
        "halo_mask": halo.reshape(-1),
        "send_max": send_max,
        "recv_max": recv_max,
        "emul_src_dev": src_dev,
        "emul_src_pos": src_pos,
        "emul_valid": valid,
    }


class _RaggedBase:
    def prepare(self, layout: Layout):
        return _tables(layout)

    def wire_cells(self, tables) -> int:
        """``send_max``: exactly the cells the peers need, no padding on this torus.

        The number the *ragged* design puts on the wire. ``ragged_emul`` reports the
        same, because it exists to price the design; its own HLO shows the
        ``all_gather`` it stands in with, which is ``P`` times as much.
        """
        return tables["send_max"]

    def exchange(self, a_local, tables, axis_name: str):
        r = lax.axis_index(axis_name)
        a_flat = a_local.reshape(-1)
        operand = a_flat[jnp.asarray(tables["send_idx"])[r]]
        recv = self._move(operand, r, tables, axis_name)
        gathered = recv[jnp.asarray(tables["recv_gather"])[r]]
        out = jnp.where(jnp.asarray(tables["halo_mask"]), gathered, a_flat)
        return out.reshape(a_local.shape)


class RaggedTransport(_RaggedBase):
    name = "ragged"

    def _move(self, operand, r, t, axis_name):
        return lax.ragged_all_to_all(
            operand,
            jnp.zeros((t["recv_max"],), operand.dtype),
            jnp.asarray(t["input_offsets"])[r],
            jnp.asarray(t["send_sizes"])[r],
            jnp.asarray(t["output_offsets"])[r],
            jnp.asarray(t["recv_sizes"])[r],
            axis_name=axis_name,
        )


class RaggedEmulTransport(_RaggedBase):
    name = "ragged_emul"

    def _move(self, operand, r, t, axis_name):
        every = lax.all_gather(operand, axis_name, axis=0, tiled=False)  # (P, send_max)
        picked = every[jnp.asarray(t["emul_src_dev"])[r], jnp.asarray(t["emul_src_pos"])[r]]
        return jnp.where(jnp.asarray(t["emul_valid"])[r], picked, jnp.zeros((), operand.dtype))


register(RaggedTransport())
register(RaggedEmulTransport())
