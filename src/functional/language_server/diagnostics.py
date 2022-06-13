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
from functional.ffront.func_to_foast import FieldOperatorSyntaxError


def from_exception(e: Exception) -> list[Diagnostic]:
    if isinstance(e, (FieldOperatorTypeDeductionError, FieldOperatorSyntaxError)):
        d = Diagnostic(
            range=Range(
                start=Position(line=e.lineno - 1, character=e.offset - 1),
                end=Position(line=e.end_lineno - 1, character=e.end_offset - 1),
            ),
            message=e.msg,
        )
        return [d]