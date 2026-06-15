# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Monomorphize calls to dtype-generic field operators inside a PAST program.

For every call to a generic operator the type variables are bound from the concrete
call-site argument types, the callee is name-mangled per binding, and a specialized
(monomorphic) callable is put into the closure variables under the mangled name. The
program lowering then produces one ``itir.FunctionDefinition`` per binding.
"""

from typing import Any

from gt4py.eve import NodeTranslator, datamodels
from gt4py.next.ffront import (
    program_ast as past,
    type_info as ffront_type_info,
    type_specifications as ts_ffront,
)
from gt4py.next.ffront.gtcallable import GTCallable
from gt4py.next.type_system import type_info, type_specifications as ts


def _mangle_name(name: str, binding: dict[str, ts.ScalarType]) -> str:
    suffix = "_".join(f"{var}_{dtype}" for var, dtype in sorted(binding.items()))
    return f"{name}__{suffix}"


class _MonomorphizeGenericCalls(NodeTranslator):
    def __init__(self, closure_vars: dict[str, Any]):
        super().__init__()
        self.closure_vars = closure_vars
        self.specialized: dict[str, GTCallable] = {}

    def visit_Call(self, node: past.Call, **kwargs: Any) -> past.Call:
        node = self.generic_visit(node, **kwargs)
        func_type = node.func.type
        # generic scan operators are rejected at decoration time, so only field operators
        # can be generic here
        if not (
            isinstance(func_type, ts_ffront.FieldOperatorType) and type_info.is_generic(func_type)
        ):
            return node
        callee = self.closure_vars.get(node.func.id)
        if not isinstance(callee, GTCallable):
            return node

        # argument types are concrete after program type deduction
        arg_types: list[ts.TypeSpec] = []
        for arg in node.args:
            assert arg.type is not None
            arg_types.append(arg.type)
        kwarg_types: dict[str, ts.TypeSpec] = {}
        for name, expr in node.kwargs.items():
            if name not in ("out", "domain"):
                assert expr.type is not None
                kwarg_types[name] = expr.type
        binding = ffront_type_info.bind_fieldop_type_vars(func_type, arg_types, kwarg_types)
        if not binding:
            return node

        mangled = _mangle_name(node.func.id, binding)
        if mangled not in self.specialized:
            self.specialized[mangled] = callee.__gt_specialize__(binding, mangled)

        new_func = datamodels.evolve(
            node.func, id=mangled, type=self.specialized[mangled].__gt_type__()
        )
        return datamodels.evolve(node, func=new_func)


def monomorphize_generic_calls(
    past_node: past.Program, closure_vars: dict[str, Any]
) -> tuple[past.Program, dict[str, Any]]:
    """Rewrite generic operator calls to mangled, monomorphic callees.

    Returns the (possibly rewritten) program node together with closure variables in which
    every called generic operator has been replaced by its per-binding specializations.
    """
    pass_ = _MonomorphizeGenericCalls(closure_vars)
    new_node = pass_.visit(past_node)
    if not pass_.specialized:
        return past_node, closure_vars

    # drop the generic callables (they can not be lowered) and add the specialized variants
    new_closure_vars = {
        name: value
        for name, value in closure_vars.items()
        if not (isinstance(value, GTCallable) and type_info.is_generic(value.__gt_type__()))
    }
    new_closure_vars.update(pass_.specialized)
    return new_node, new_closure_vars
