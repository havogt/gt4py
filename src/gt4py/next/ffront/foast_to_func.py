# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import hashlib
import linecache
import textwrap
from typing import Any

from gt4py import eve
from gt4py.next import backend as next_backend, common
from gt4py.next.ffront import (
    decorator,
    fbuiltins,
    field_operator_ast as foast,
    foast_pretty_printer,
    stages as ffront_stages,
)
from gt4py.next.type_system import type_specifications as ts, type_translation


def _python_type_hint(type_: ts.TypeSpec) -> Any:
    match type_:
        case ts.ScalarType():
            return type_translation.as_dtype(type_).scalar_type
        case ts.FieldType():
            return common.Field[list(type_.dims), _python_type_hint(type_.dtype)]  # type: ignore[misc]
        case ts.TupleType():
            return tuple[tuple(_python_type_hint(t) for t in type_.types)]  # type: ignore[misc]
    raise NotImplementedError(f"Type '{type_}' can not be converted to a type hint.")


def field_operator_from_foast(
    foast_def: ffront_stages.FOASTOperatorDef,
    *,
    backend: next_backend.Backend | eve.NothingType | None = eve.NOTHING,
) -> decorator.FieldOperator:
    """
    Create a field or scan operator from a programmatically built and typed FOAST definition.

    The FOAST is printed to Python source, which is then executed and decorated like a
    hand-written definition. The result is a regular `FieldOperator`, usable on every backend
    (including embedded execution) and callable from other field operators.
    """
    node = foast_def.foast_node
    definition = node.definition
    assert isinstance(definition.type, ts.FunctionType)

    type_hints = {
        f"__gt_param_type_{i}": _python_type_hint(param.type)
        for i, param in enumerate(definition.params)
    }
    type_hints["__gt_return_type"] = _python_type_hint(definition.type.returns)
    params = ", ".join(
        f"{param.id}: __gt_param_type_{i}" for i, param in enumerate(definition.params)
    )
    body = textwrap.indent(foast_pretty_printer.pretty_format(definition.body), "    ")
    source = f"def {definition.id}({params}) -> __gt_return_type:\n{body}\n"

    # `inspect.getsource`, used by the frontend, finds the source of exec'ed code
    # only through `linecache`.
    filename = f"<gt4py-foast-{definition.id}-{hashlib.sha256(source.encode()).hexdigest()[:16]}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
    namespace = {**fbuiltins.BUILTINS, **foast_def.closure_vars, **type_hints}
    exec(compile(source, filename, "exec"), namespace)
    func = namespace[str(definition.id)]

    if isinstance(node, foast.ScanOperator):
        return decorator.scan_operator(
            func,
            axis=node.axis.value,
            forward=node.forward.value,
            init=node.init.value,
            backend=backend,
            grid_type=foast_def.grid_type,
        )
    return decorator.field_operator(func, backend=backend, grid_type=foast_def.grid_type)
