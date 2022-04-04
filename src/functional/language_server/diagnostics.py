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

from pygls.lsp import Diagnostic, Position, Range

from functional.ffront.foast_passes.type_deduction import FieldOperatorTypeDeductionError


def from_exception(e: Exception) -> list[Diagnostic]:
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
        return [d]
