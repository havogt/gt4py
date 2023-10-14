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

import ast
import linecache


class _ExtractDecorated(ast.NodeVisitor):
    def __init__(self, decorator_name):
        self.decorator_name = decorator_name
        self.field_ops = []

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.decorator_list:
            for d in node.decorator_list:
                if isinstance(d, ast.Name) and d.id == self.decorator_name:
                    self.field_ops.append(node)

    @classmethod
    def apply(cls, decorator_name, node):
        x = cls(decorator_name)
        x.visit(node)
        return x.field_ops


def parse_ffront(filename: str, source: str):
    program = ast.parse(source)

    field_ops = _ExtractDecorated.apply("field_operator", program)  # TODO fully qualified
    programs = _ExtractDecorated.apply("program", program)

    if len(field_ops) == 0 and len(programs) == 0:
        return None  # don't execute a file that doesn't contain ffront code

    linecache_tuple = (len(source), None, source.splitlines(True), filename)
    linecache.cache[filename] = linecache_tuple

    c = compile(source, filename, "exec")
    namespace = {}
    exec(c, namespace)

    foast_ops = []
    for f in field_ops:
        foast_ops.append(namespace[f.name].foast_node)

    past_programs = []
    for p in programs:
        past_programs.append(namespace[p.name].past_node)

    return foast_ops + past_programs
