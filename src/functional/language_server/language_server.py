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

import argparse
import logging

from pygls.lsp.methods import HOVER, TEXT_DOCUMENT_DID_CHANGE, TEXT_DOCUMENT_DID_OPEN
from pygls.lsp.types import (
    DidChangeTextDocumentParams,
    DidOpenTextDocumentParams,
    Hover,
    HoverParams,
)
from pygls.server import LanguageServer

from functional.language_server import diagnostics, hover, parser


gt4py_server = LanguageServer()


parsed_nodes = {}  # cache of parsed nodes TODO encapsulate


def _parse(server: LanguageServer, params):
    text_doc = server.workspace.get_document(params.text_document.uri)
    source = text_doc.source
    filename = f"<lsp:{params.text_document.uri}>"

    diags = []
    try:
        parsed_nodes[params.text_document.uri] = parser.parse_ffront(filename, source)
    except Exception as e:
        new_diags = diagnostics.from_exception(e)
        if new_diags:
            diags.extend(new_diags)
        else:
            # raise e
            server.show_message(
                str(e)
            )  # for debugging:t inform the client that something went wrong

    server.publish_diagnostics(
        params.text_document.uri, diags
    )  # resets diagnostics if len(diags) == 0


@gt4py_server.feature(TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls, params: DidChangeTextDocumentParams):
    """Text document did change notification."""
    # ls.show_message("Text document did change")
    _parse(ls, params)


@gt4py_server.feature(TEXT_DOCUMENT_DID_OPEN)
async def did_open(ls: LanguageServer, params: DidOpenTextDocumentParams):
    """Text document did open notification."""
    # ls.show_message("Text Document Did Open")
    _parse(ls, params)


@gt4py_server.feature(HOVER)
def hover_action(params: HoverParams) -> Hover:
    field_ops = (
        parsed_nodes[params.text_document.uri] if params.text_document.uri in parsed_nodes else []
    )

    return hover.hover_info(
        field_ops, params.position.line, params.position.character + 1
    )  # double-check 1/0 convention


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
        logging.basicConfig(filename="pygls.log", level=logging.DEBUG, filemode="w")
    elif args.ws:
        gt4py_server.start_ws(args.host, args.port)
    else:
        gt4py_server.start_io()


if __name__ == "__main__":
    main()
