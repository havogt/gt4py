# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Test utility functions of the dace backend module."""

import dace
import pytest

from gt4py.next.iterator.ir_utils import ir_makers as im
from gt4py.next.program_processors.runners.dace.lowering import gtir_to_sdfg_utils


def test_safe_replace_symbolic():
    assert gtir_to_sdfg_utils.safe_replace_symbolic(
        dace.symbolic.pystr_to_symbolic("x*x + y"), symbol_mapping={"x": "y", "y": "x"}
    ) == dace.symbolic.pystr_to_symbolic("y*y + x")


@pytest.mark.parametrize("x, expected", [(0, 0), (-1, 4)])
def test_get_symbolic_if_as_operand(x, expected):
    expr = gtir_to_sdfg_utils.get_symbolic(im.minus(im.if_(im.greater_equal("x", 0), 1, 5), 1))
    assert expr.subs({"x": x}) == expected
