import logging

from pygls.lsp.methods import (
    COMPLETION,
    HOVER,
    TEXT_DOCUMENT_DID_CHANGE,
    TEXT_DOCUMENT_DID_CLOSE,
    TEXT_DOCUMENT_DID_OPEN,
)
from pygls.lsp.types import (
    CompletionItem,
    CompletionList,
    CompletionOptions,
    CompletionParams,
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

from functional.common import FieldOperator
from functional.ffront.common_types import FieldType
from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeductionError


gt4py_server = LanguageServer()

logging.basicConfig(filename="pygls.log", level=logging.DEBUG, filemode="w")


@gt4py_server.feature(COMPLETION, CompletionOptions(trigger_characters=[","]))
def completions(params: CompletionParams):
    """Returns completion items."""
    print("bal")
    return CompletionList(
        is_incomplete=False,
        items=[
            CompletionItem(label='"'),
            CompletionItem(label="["),
            CompletionItem(label="]"),
            CompletionItem(label="{"),
            CompletionItem(label="}"),
        ],
    )


import ast

from devtools import debug

from eve.type_definitions import SourceLocation
from eve.visitors import NodeVisitor
from functional.ffront import decorator, func_to_foast
from functional.ffront.source_utils import ClosureRefs, SourceDefinition


def _parse_foast(server: LanguageServer, params):
    # server.show_message("parsing foast")

    text_doc = server.workspace.get_document(params.text_document.uri)
    source = text_doc.source
    # print(text_doc.source)
    # module = importCode(text_doc.source, "my_module")
    # print("loaded")
    # debug(module.lap)
    # print(module)
    diagnostics = []
    res = []
    try:
        program = ast.parse(source)

        class ExtractFieldOps(ast.NodeVisitor):
            def __init__(self):
                self.field_ops = []

            def visit_FunctionDef(self, node: ast.FunctionDef):
                if node.decorator_list:
                    for d in node.decorator_list:
                        if isinstance(d, ast.Name) and d.id == "field_operator":
                            self.field_ops.append(node)

                            print(node.name)

            @classmethod
            def apply(cls, node):
                x = cls()
                x.visit(node)
                return x.field_ops

        field_ops = ExtractFieldOps.apply(program)

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
                func_to_foast.FieldOperatorParser.apply(src_def, ClosureRefs.from_function(fun))
            )
        res = foast_field_ops
    except Exception as e:
        if isinstance(e, FieldOperatorTypeDeductionError):
            d = Diagnostic(
                range=Range(
                    start=Position(line=e.args[1][1], character=e.args[1][2] - 1),
                    end=Position(line=e.args[1][4], character=e.args[1][5] - 1),
                ),
                # range=Range(
                #     start=Position(line=e.lineno, character=e.offset),
                #     end=Position(line=e.end_lineno, character=e.end_offset),
                # ),
                message=e.msg,
            )
            diagnostics.append(d)
    print(f"diags {diagnostics}")
    server.publish_diagnostics(params.text_document.uri, diagnostics)
    return res


def _find_node(field_ops, position: Position):
    class FindNodeByPosition(NodeVisitor):
        def __init__(self, line, column):
            self.nodes = []
            self.line = line
            self.column = column

        def visit_Node(self, node):
            loc: SourceLocation = node.location
            if loc.line < self.line or (loc.line == self.line and loc.column <= self.column):
                if loc.end_line > self.line or (
                    loc.end_line == self.line and loc.end_column >= self.column
                ):
                    self.nodes.append(node)
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
    field_ops = _parse_foast(gt4py_server, params)
    node = _find_node(field_ops, params.position)
    if node:
        print("found node")
        if hasattr(node, "type") and isinstance(node.type, FieldType):
            print("found field op")
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
            contents="Type was not deduced!",
            range=Range(
                start=Position(line=node.location.line, character=node.location.column - 1),
                end=Position(
                    line=node.location.end_line,
                    character=node.location.end_column - 1,
                ),
            ),
        )
    # else:
    #     return Hover(contents="", range=Range(start=params.position, end=params.position))


def main():
    gt4py_server.start_tcp("127.0.0.1", 2087)


if __name__ == "__main__":
    main()
