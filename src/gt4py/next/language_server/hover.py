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

from gt4py.eve import SourceLocation
from gt4py.eve.visitors import NodeVisitor
from lsprotocol.types import Hover, Position, Range

from gt4py.next.ffront.common_types import FieldType
from gt4py.next.ffront.field_operator_ast import Call


class _FindNodeByPosition(NodeVisitor):
    def __init__(self, line, column):
        self.nodes = []
        self.line = line
        self.column = column

    def _check_and_append(self, node):
        loc: SourceLocation = node.location
        if loc.line < self.line or (loc.line == self.line and loc.column <= self.column):
            if loc.end_line > self.line or (
                loc.end_line == self.line and loc.end_column >= self.column
            ):
                self.nodes.append(node)

    def visit_Call(self, node: Call):
        self._check_and_append(node)
        self.visit(node.args)  # skip node.func

    def visit_Node(self, node, **kwargs):
        self._check_and_append(node)
        self.generic_visit(node)

    @classmethod
    def apply(cls, node, line, column):
        x = cls(line, column)
        x.visit(node)
        return x.nodes


def _line_length(loc: SourceLocation):
    return loc.end_line - loc.line


def _find_smallest(nodes):
    if len(nodes) == 0:
        return None
    smallest = nodes[0]
    for n in nodes:
        if _line_length(n.location) < _line_length(smallest.location):
            smallest = n
        elif _line_length(n.location) == _line_length(smallest.location):
            if (
                n.location.column > smallest.location.column
                or n.location.end_column < smallest.location.end_column
            ):
                smallest = n
    return smallest


def _find_node_at_position(field_ops, line: int, character: int):
    res = _FindNodeByPosition.apply(field_ops, line, character)
    return _find_smallest(res)


def hover_info(field_ops, line: int, character: int):
    node = _find_node_at_position(field_ops, line + 1, character)
    if node:
        if hasattr(node, "type"):
            if isinstance(node.type, FieldType):
                t = node.type
                return Hover(
                    contents=f"Field[[{', '.join(d.value for d in t.dims)}], {t.dtype}]",
                    range=Range(
                        start=Position(
                            line=node.location.line - 1, character=node.location.column - 1
                        ),
                        end=Position(
                            line=node.location.end_line - 1,
                            character=node.location.end_column - 1,
                        ),
                    ),
                )
        return Hover(
            contents="Not a FieldType or not deduced!",
            range=Range(
                start=Position(line=node.location.line - 1, character=node.location.column - 1),
                end=Position(
                    line=node.location.end_line - 1,
                    character=node.location.end_column - 1,
                ),
            ),
        )
