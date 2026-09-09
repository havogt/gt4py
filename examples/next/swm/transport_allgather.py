# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""All-gather broadcast transport: every device receives the whole field.

FESOM2-JAX's baseline and the oracle the other transports are checked against.
``lax.all_gather`` collects every device's flattened interior lanes into a
``(P, MLOC*NLOC)`` array on every device; a fancy-index gather with two
precomputed int32 maps of the local halo shape then reads every local cell --
interior and halo alike -- from its owner. Volume is O(P) times what is needed,
but both primitives are linear with known transposes (``all_gather`` ->
``psum_scatter``, gather -> scatter-add), so the adjoint is exact and free.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from halo_transports import Layout, register


class AllGatherTransport:
    name = "allgather"

    def prepare(self, layout: Layout):
        dev, lane = layout.owner_lane()
        return {"layout": layout, "src_dev": dev, "src_lane": lane}

    def wire_cells(self, tables) -> int:
        """The **receive** side: every device takes delivery of every block, ``P*MLOC*NLOC``.

        The send side is ``MLOC*NLOC`` and shrinks with P; the receive side is what makes
        this the O(P) transport, and it is the number the comparison quotes.
        """
        L: Layout = tables["layout"]
        return L.P * L.MLOC * L.NLOC

    def exchange(self, a_local, tables, axis_name: str):
        L: Layout = tables["layout"]
        h = L.h
        lanes = a_local[h:-h, h:-h].reshape(-1)
        gathered = jax.lax.all_gather(lanes, axis_name, axis=0, tiled=False)  # (P, MLOC*NLOC)
        r = jax.lax.axis_index(axis_name)
        src_dev = jnp.asarray(tables["src_dev"])[r]
        src_lane = jnp.asarray(tables["src_lane"])[r]
        return gathered[src_dev, src_lane]


register(AllGatherTransport())
