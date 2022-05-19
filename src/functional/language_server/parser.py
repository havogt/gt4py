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
import linecache

import beniget
import gast

from eve import extended_typing
from functional.ffront import decorator, func_to_foast, func_to_past
from functional.ffront.source_utils import (
    CapturedVars,
    SourceDefinition,
    make_captured_vars_from_function,
)


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


def capture(code: str):
    class _ExtractDecorated(gast.NodeVisitor):
        def __init__(self, decorator_name):
            self.decorator_name = decorator_name
            self.field_ops = []

        def visit_FunctionDef(self, node):
            if node.decorator_list:
                for d in node.decorator_list:
                    if isinstance(d, gast.Name) and d.id == self.decorator_name:
                        self.field_ops.append(node)

        @classmethod
        def apply(cls, decorator_name, node):
            x = cls(decorator_name)
            x.visit(node)
            return x.field_ops

    class Capture(gast.NodeVisitor):
        def __init__(self, module_node):
            # initialize def-use chains
            self.chains = beniget.DefUseChains()
            self.chains.visit(module_node)
            self.users = set()  # users of local definitions
            self.captured = set()  # identifiers that don't belong to local users

        def visit_FunctionDef(self, node):
            # initialize the set of node using a local variable
            for def_ in self.chains.locals[node]:
                self.users.update(use.node for use in def_.users())
            self.generic_visit(node)

        def visit_Name(self, node):
            # register load of identifiers not locally definied
            if isinstance(node.ctx, gast.Load):
                if node not in self.users:
                    self.captured.add(node.id)

    class CaptureX(gast.NodeVisitor):
        def __init__(self, module_node, fun):
            self.fun = fun
            # initialize use-def chains
            du = beniget.DefUseChains()
            du.visit(module_node)
            self.chains = beniget.UseDefChains(du)
            self.ancestors = beniget.Ancestors()
            self.ancestors.visit(module_node)
            self.external = list()
            self.visited_external = set()

        def visit_Name(self, node):
            # register load of identifiers not locally defined
            if isinstance(node.ctx, gast.Load):
                uses = self.chains.chains[node]
                for use in uses:
                    try:
                        parents = self.ancestors.parents(use.node)
                    except KeyError:
                        return  # a builtin
                    if self.fun not in parents:
                        parent = self.ancestors.parentStmt(use.node)
                        if parent not in self.visited_external:
                            self.visited_external.add(parent)
                            self.rec(parent)
                            self.external.append(parent)
            if node.annotation:
                self.visit(node.annotation)

        # def generic_visit(self, node):
        #     return super().generic_visit(node)

        def rec(self, node):
            "walk definitions to find their operands's def"
            if isinstance(node, gast.Assign):
                self.visit(node.value)
            # TODO: implement this for AugAssign etc

    module = gast.parse(code)

    field_ops = _ExtractDecorated.apply("field_operator", module)  # TODO fully qualified

    field_ops_with_captures = []

    for f in field_ops:
        inner_function = f
        capture = CaptureX(module, f)
        capture.visit(inner_function)
        field_ops_with_captures.append(
            (gast.gast_to_ast(f), [gast.gast_to_ast(n) for n in capture.external])
        )

    return field_ops_with_captures


def parse_ffront(filename: str, source: str):
    program = ast.parse(source)

    field_ops = _ExtractDecorated.apply("field_operator", program)  # TODO fully qualified
    programs = _ExtractDecorated.apply("program", program)

    if len(field_ops) == 0 and len(programs) == 0:
        return None  # don't execute a file that doesn't contain ffront code

    linecache_tuple = (len(source), None, source.splitlines(True), filename)
    linecache.cache[filename] = linecache_tuple

    field_ops_with_captures = capture(source)

    # tmp_captures = field_ops_with_captures[0][1]
    # tmp_captures.reverse()

    # module = ast.Module(body=tmp_captures, type_ignores=[])
    # c = compile(module, filename, "exec")
    # # c = compile(source, filename, "exec")
    # namespace = {}
    # exec(c, namespace)

    def get_with_name(name, lst):
        for x in lst:
            if x.name == name:
                return x

    foast_ops = []
    for f, captures in field_ops_with_captures:
        body = [
            *captures,
            # field_ops[f.name],
            # f
            get_with_name(f.name, field_ops),
        ]

        module = ast.Module(body=body, type_ignores=[])
        print(ast.dump(module))
        c = compile(module, filename, "exec")
        # c = compile(source, filename, "exec")
        namespace = {}
        exec(c, namespace)

        foast_ops.append(namespace[f.name].foast_node)

        # source_split = source.splitlines()
        # s = "\n".join(source_split[f.lineno - 1 : f.end_lineno])

        # src_def = SourceDefinition(s, "<string>", f.lineno - 1)
        # fun = namespace[f.name]
        # annotations = extended_typing.get_type_hints(fun.definition)

        # # captured_vars_from_fun = make_captured_vars_from_function(fun.definition)
        # cap = {k: v for k, v in namespace.items() if k not in ["reduction"]}  # TODO
        # captured_vars = CapturedVars({}, cap, annotations, set(), set())
        # foast_ops.append(
        #     # func_to_foast.FieldOperatorParser.apply(src_def, CapturedVars.from_function(fun))
        #     func_to_foast.FieldOperatorParser.apply(src_def, captured_vars)
        # )

    past_programs = []
    for p in programs:
        past_programs.append(namespace[p.name].past_node)

    return foast_ops + past_programs
