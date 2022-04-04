# GT4Py Project - GridTools Framework
#
# Copyright (c) 2014-2021, ETH Zurich
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

from functional.ffront import decorator, func_to_foast, func_to_past
from functional.ffront.source_utils import CapturedVars, SourceDefinition


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


def parse_ffront(source: str):
    program = ast.parse(source)

    field_ops = _ExtractDecorated.apply("field_operator", program)  # TODO fully qualified
    programs = _ExtractDecorated.apply("program", program)

    if len(field_ops) == 0 and len(programs) == 0:
        return None  # don't execute a file that doesn't contain ffront code

    source_split = source.splitlines()

    decorator.LSP_MODE = (
        True  # TODO hack to be able to execute the decorator from a string (without file)
    )

    c = compile(source, "<string>", "exec")
    namespace = {}
    exec(c, namespace)

    foast_ops = []
    for f in field_ops:
        s = "\n".join(source_split[f.lineno - 1 : f.end_lineno])

        src_def = SourceDefinition(s, "<string>", f.lineno - 1)
        fun = namespace[f.name]

        foast_ops.append(
            func_to_foast.FieldOperatorParser.apply(src_def, CapturedVars.from_function(fun))
        )

    past_programs = []
    for p in programs:
        s = "\n".join(source_split[p.lineno - 1 : p.end_lineno])

        src_def = SourceDefinition(s, "<string>", p.lineno - 1)
        prog = namespace[p.name]

        past_programs.append(
            func_to_past.ProgramParser.apply(src_def, CapturedVars.from_function(prog))
        )

    return foast_ops + past_programs
