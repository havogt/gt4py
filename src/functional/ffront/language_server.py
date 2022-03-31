import argparse
import ast
import logging

from pygls.lsp.methods import (
    HOVER,
    TEXT_DOCUMENT_DID_CHANGE,
    TEXT_DOCUMENT_DID_CLOSE,
    TEXT_DOCUMENT_DID_OPEN,
)
from pygls.lsp.types import (
    Diagnostic,
    DidChangeTextDocumentParams,
    DidCloseTextDocumentParams,
    DidOpenTextDocumentParams,
    Hover,
    HoverParams,
    Position,
    Range,
)
from pygls.server import LanguageServer

from eve.type_definitions import SourceLocation
from eve.visitors import NodeVisitor
from functional.ffront import decorator, func_to_foast, func_to_past
from functional.ffront.common_types import FieldType
from functional.ffront.field_operator_ast import Call
from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeductionError
from functional.ffront.source_utils import CapturedVars, SourceDefinition


gt4py_server = LanguageServer()

logging.basicConfig(filename="pygls.log", level=logging.DEBUG, filemode="w")

parsed_nodes = {}


def _parse_foast(server: LanguageServer, params):
    text_doc = server.workspace.get_document(params.text_document.uri)
    source = text_doc.source
    diagnostics = []
    try:
        program = ast.parse(source)

        class ExtractDecorated(ast.NodeVisitor):
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

        field_ops = ExtractDecorated.apply("field_operator", program)  # TODO fully qualified
        programs = ExtractDecorated.apply("program", program)

        if len(field_ops) == 0 and len(programs) == 0:
            gt4py_server.show_message("excluding file form parsing (no field operator found)")
        else:
            source_split = source.splitlines()

            decorator.LSP_MODE = True

            c = compile(source, "<string>", "exec")
            namespace = {}
            exec(c, namespace)

            foast_field_ops = []
            for f in field_ops:
                s = "\n".join(source_split[f.lineno - 1 : f.end_lineno])

                src_def = SourceDefinition(s, "<string>", f.lineno - 1)
                fun = namespace[f.name]

                foast_field_ops.append(
                    func_to_foast.FieldOperatorParser.apply(
                        src_def, CapturedVars.from_function(fun)
                    )
                )

            past_programs = []
            for p in programs:
                s = "\n".join(source_split[p.lineno - 1 : p.end_lineno])

                src_def = SourceDefinition(s, "<string>", p.lineno - 1)
                prog = namespace[p.name]

                past_programs.append(
                    func_to_past.ProgramParser.apply(src_def, CapturedVars.from_function(prog))
                )

            parsed_nodes[params.text_document.uri] = foast_field_ops + past_programs
    except Exception as e:
        if isinstance(e, FieldOperatorTypeDeductionError):
            d = Diagnostic(
                range=Range(
                    start=Position(line=e.args[1][1], character=e.args[1][2] - 1),
                    end=Position(line=e.args[1][4], character=e.args[1][5] - 1),
                ),
                # TODO should be the following but somehow we mess up line numbers
                # range=Range(
                #     start=Position(line=e.lineno, character=e.offset),
                #     end=Position(line=e.end_lineno, character=e.end_offset),
                # ),
                message=e.msg,
            )
            diagnostics.append(d)
    server.publish_diagnostics(params.text_document.uri, diagnostics)


def _find_node(field_ops, position: Position):
    class FindNodeByPosition(NodeVisitor):
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

    def line_length(loc: SourceLocation):
        return loc.end_line - loc.line

    def find_smallest(nodes):
        if len(nodes) == 0:
            return None
        smallest = nodes[0]
        for n in nodes:
            if line_length(n.location) < line_length(smallest.location):
                smallest = n
            elif line_length(n.location) == line_length(smallest.location):
                if (
                    n.location.column > smallest.location.column
                    or n.location.end_column < smallest.location.end_column
                ):
                    smallest = n
        return smallest

    res = FindNodeByPosition.apply(field_ops, position.line, position.character + 1)
    return find_smallest(res)


@gt4py_server.feature(TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls, params: DidChangeTextDocumentParams):
    """Text document did change notification."""
    ls.show_message("Text document did change")
    _parse_foast(ls, params)


@gt4py_server.feature(TEXT_DOCUMENT_DID_CLOSE)
def did_close(server: LanguageServer, params: DidCloseTextDocumentParams):
    """Text document did close notification."""
    server.show_message("Text Document Did Close")


@gt4py_server.feature(TEXT_DOCUMENT_DID_OPEN)
async def did_open(ls: LanguageServer, params: DidOpenTextDocumentParams):
    """Text document did open notification."""
    ls.show_message("Text Document Did Open")
    _parse_foast(ls, params)


@gt4py_server.feature(HOVER)
def hover(params: HoverParams) -> Hover:
    field_ops = (
        parsed_nodes[params.text_document.uri] if params.text_document.uri in parsed_nodes else []
    )

    node = _find_node(field_ops, params.position)
    if node:
        if hasattr(node, "type"):
            if isinstance(node.type, FieldType):
                t = node.type
                return Hover(
                    contents=f"Field[[{','.join(d.value for d in t.dims)}], {t.dtype}]",
                    range=Range(
                        start=Position(line=node.location.line, character=node.location.column - 1),
                        end=Position(
                            line=node.location.end_line,
                            character=node.location.end_column - 1,
                        ),
                    ),
                )
        return Hover(
            contents="Not a FieldType or not deduced!",
            range=Range(
                start=Position(line=node.location.line, character=node.location.column - 1),
                end=Position(
                    line=node.location.end_line,
                    character=node.location.end_column - 1,
                ),
            ),
        )


def add_arguments(parser):
    parser.description = "simple json server example"

    parser.add_argument("--tcp", action="store_true", help="Use TCP server")
    parser.add_argument("--ws", action="store_true", help="Use WebSocket server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind to this address")
    parser.add_argument("--port", type=int, default=2087, help="Bind to this port")


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    if args.tcp:
        gt4py_server.start_tcp(args.host, args.port)
    elif args.ws:
        gt4py_server.start_ws(args.host, args.port)
    else:
        gt4py_server.start_io()


if __name__ == "__main__":
    main()
