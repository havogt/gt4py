# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""All-gather broadcast transport: every device receives the whole field.

Volume is O(P) times the halo, but ``all_gather`` and the index gather both have exact
transposes, so the adjoint is exact by composition: this is the oracle the other
transports are checked against.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

from halo_transports import Layout, register


class AllGatherTransport:
    name = "allgather"

    def prepare(self, layout: Layout):
        dev, lane = layout.owner_lane()
        return {
            "h": layout.h,
            "src_dev": dev,
            "src_lane": lane,
            "recv_cells": layout.P * layout.MLOC * layout.NLOC,
        }

    def wire_cells(self, tables) -> int:
        """The receive side, ``P*MLOC*NLOC``: every device takes delivery of every block."""
        return tables["recv_cells"]

    def exchange(self, a_local, tables, axis_name: str):
        h = tables["h"]
        lanes = a_local[h:-h, h:-h].reshape(-1)
        gathered = lax.all_gather(lanes, axis_name, axis=0, tiled=False)  # (P, MLOC*NLOC)
        r = lax.axis_index(axis_name)
        src_dev = jnp.asarray(tables["src_dev"])[r]
        src_lane = jnp.asarray(tables["src_lane"])[r]
        return gathered[src_dev, src_lane]


register(AllGatherTransport())
