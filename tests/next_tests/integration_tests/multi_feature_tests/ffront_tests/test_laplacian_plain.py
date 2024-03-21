# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots
from vscodedebugvisualizer import (
    globalVisualizationFactory,
    visualize as default_visualize,
    visualizer,
)

import gt4py.next as gtx
from gt4py.next import common, int32, where
from gt4py.next.ffront.experimental import concat_where


class FieldVisualizer:
    def checkType(self, obj):
        return common.is_field(obj)

    def visualize(self, field: gtx.Field):
        metaData = {str(d): str(rng) for d, rng in field.domain}
        fig = make_subplots(rows=2, cols=1)
        # fig.add_trace(
        #     go.Table(
        #         header=dict(
        #             values=[[name] for name in metaData],
        #         ),
        #         cells=dict(
        #             values=[[metaData[name]] for name in metaData],
        #         ),
        #     ),
        # )
        fig.add_trace(
            go.Heatmap(
                z=field.ndarray,
                colorscale="Viridis",
                showscale=False,
            ),
        )
        return visualizer.PlotlyVisualizer.PlotlyVisualizer().visualize(fig)


globalVisualizationFactory.addVisualizer(FieldVisualizer())


pytestmark = pytest.mark.uses_cartesian_shift

_debug_data = {}
_cur_debug_mark = None
_cur_debug_expr = None
_debug_mark_state = {}


def debug_verify():
    if (
        _cur_debug_mark in _debug_data
        and _debug_mark_state[_cur_debug_mark] in _debug_data[_cur_debug_mark]
    ):
        return _debug_data[_cur_debug_mark][_debug_mark_state[_cur_debug_mark]] - _cur_debug_expr
    else:
        return "no entry"


def debug_mark(name: str, expr):
    global _cur_debug_mark, _cur_debug_expr
    _cur_debug_mark = name
    _debug_mark_state[name] = _debug_mark_state.get(name, -1) + 1
    _cur_debug_expr = expr
    breakpoint()
    return expr


IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")

Ioff = gtx.FieldOffset("Ioff", source=IDim, target=(IDim,))
Joff = gtx.FieldOffset("Joff", source=JDim, target=(JDim,))


@gtx.field_operator
def boundary(
    in_field: gtx.Field[[IDim, JDim], "float"],
    i: gtx.Field[[IDim], int32],
    j: gtx.Field[[JDim], int32],
) -> gtx.Field[[IDim, JDim], "float"]:
    return concat_where(
        (i == 0) | (i == 9), 42.0, concat_where((j == 0) | (j == 9), 43.0, in_field)
    )


@gtx.field_operator
def lap(
    in_field: gtx.Field[[IDim, JDim], "float"],
    i: gtx.Field[[IDim], int32],
    j: gtx.Field[[JDim], int32],
) -> gtx.Field[[IDim, JDim], "float"]:
    res = (
        -4.0 * in_field
        + in_field(Ioff[1])
        + in_field(Joff[1])
        + in_field(Ioff[-1])
        + in_field(Joff[-1])
    )
    res = boundary(res, i, j)
    return debug_mark("lap", res)


@gtx.field_operator
def laplap(
    in_field: gtx.Field[[IDim, JDim], "float"],
    i: gtx.Field[[IDim], int32],
    j: gtx.Field[[JDim], int32],
) -> gtx.Field[[IDim, JDim], "float"]:
    return debug_mark("laplap", lap(lap(in_field, i, j), i, j))


def lap_ref(inp):
    """Compute the laplacian using numpy"""
    return -4.0 * inp[1:-1, 1:-1] + inp[:-2, 1:-1] + inp[2:, 1:-1] + inp[1:-1, :-2] + inp[1:-1, 2:]


def test_ffront_lap():
    in_field = gtx.as_field((IDim, JDim), np.fromfunction(lambda i, j: i * i + j * j, (10, 10)))
    i = gtx.as_field((IDim,), np.arange(10, dtype=np.int32))
    j = gtx.as_field((JDim,), np.arange(10, dtype=np.int32))
    out_field = gtx.zeros({IDim: 10, JDim: 10})

    ref0 = gtx.as_field((IDim, JDim), np.ones_like(in_field.ndarray))
    ref0.ndarray[1:-1, 1:-1] = lap_ref(in_field.ndarray)
    ref1 = gtx.as_field((IDim, JDim), np.ones_like(in_field.ndarray))
    ref1.ndarray[2:-2, 2:-2] = lap_ref(lap_ref(in_field.ndarray))
    _debug_data["lap"] = {0: ref0}
    _debug_data["laplap"] = {0: ref1}

    laplap(in_field, i, j, out=out_field, offset_provider={"Ioff": IDim, "Joff": JDim})


test_ffront_lap()
