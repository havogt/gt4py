# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2022, ETH Zurich
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

from gt4py import eve
from gt4py.next.iterator import ir


class MergeTuple(eve.NodeTranslator):
    """Transform `make_tuple(tuple_get(0, t), tuple_get(1, t), ..., tuple_get(N-1,t))` -> t."""

    # TODO if we don't check if the inner tuple `t` has same size N, i.e. outer and inner have same size
    # we will have issues, in case the result is assigned into an external buffer which expects tuple of smaller size
    def visit_FunCall(self, node: ir.FunCall, **kwargs):
        if node.fun == ir.SymRef(id="make_tuple") and all(
            isinstance(arg, ir.FunCall) and arg.fun == ir.SymRef(id="tuple_get")
            for arg in node.args
        ):
            assert isinstance(node.args[0], ir.FunCall)
            first_expr = node.args[0].args[1]
            for i, v in enumerate(node.args):
                assert isinstance(v, ir.FunCall)
                assert isinstance(v.args[0], ir.Literal)
                if not (int(v.args[0].value) == i and v.args[1] == first_expr):
                    return self.generic_visit(node)

            return first_expr
        return self.generic_visit(node)