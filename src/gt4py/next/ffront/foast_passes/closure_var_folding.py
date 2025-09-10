# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import types
from typing import Any

import gt4py.next.ffront.field_operator_ast as foast
from gt4py import eve
from gt4py._core import definitions as core_defs
from gt4py.eve import NodeTranslator, traits
from gt4py.eve.utils import FrozenNamespace
from gt4py.next import common, errors
from gt4py.next.ffront import fbuiltins, decorator
from gt4py.next.type_system import type_specifications as ts, type_translation
import inspect


@dataclasses.dataclass
class Foldable:
    value: Any


def error_if_not_foldable(value: Any, location: eve.SourceLocation) -> None:
    if core_defs.is_scalar_type(value):
        # TODO: allow if marked final
        raise errors.DSLError(location, "Cannot close over a non-final scalar variable.")
    if isinstance(value, common.Field):
        raise errors.DSLError(location, "Cannot close over a 'Field'.")


class NotFoldableError(Exception):
    pass


@dataclasses.dataclass
class ClosureVarFolding(NodeTranslator, traits.VisitorWithSymbolTableTrait):
    """
    Replace references to closure variables or their attributes with constants.

    `Name` nodes that refer to closure variables are replaced by `Constant`
     nodes. `Attribute` nodes that refer to attributes of closure variables
     are recursively replaced by `Constant` nodes.
    """

    closure_vars: dict[str, Any]
    new_symbols: dict[str, foast.Symbol] = dataclasses.field(default_factory=dict)

    def inline_closure_var(self, value: Any, name: str, location: eve.SourceLocation) -> foast.Node:
        if isinstance(value, (types.ModuleType, FrozenNamespace)):
            return Foldable(value)
        if value in fbuiltins.BUILTINS.values():
            return foast.Name(id=name, location=location)
        if isinstance(value, common.Dimension):
            dimension_name = value.value
            type_ = ts.DimensionType(dim=value)
            self.new_symbols[name] = foast.Symbol(
                id=dimension_name, location=location, type=type_
            )  # TODO location of the actual definition (the looked-up closure var, but doesn't matter in this case as we throw away the definition in lowering)
            return foast.Name(id=dimension_name, location=location, type=type_)
        if core_defs.is_scalar_type(value):
            annotations = inspect.get_annotations(value)
            if "final" in annotations.values():
                return foast.Constant(value=value, location=location)
        if isinstance(value, decorator.FieldOperator):
            return None

        raise NotFoldableError()

    @classmethod
    def apply(
        cls, node: foast.FunctionDefinition | foast.FieldOperator, closure_vars: dict[str, Any]
    ) -> foast.FunctionDefinition:
        return cls(closure_vars=closure_vars).visit(node)

    def visit_Name(
        self,
        node: foast.Name,
        current_closure_vars: dict[str, Any],
        symtable: dict[str, foast.Symbol],
        **kwargs: Any,
    ) -> foast.Name | foast.Constant:
        if node.id in symtable:
            definition = symtable[node.id]
            if definition in current_closure_vars:
                value = self.closure_vars[node.id]
                error_if_not_foldable(value, node.location)
                try:
                    return self.inline_closure_var(value, node.id, node.location) or node
                except NotFoldableError:
                    raise errors.DSLError(  # noqa: B904
                        node.location,
                        f"Unexpected closure variable {node.id} of type {type(value)}",
                    )
        return node

    def visit_Attribute(
        self, node: foast.Attribute, **kwargs: Any
    ) -> foast.Constant | foast.Attribute:
        value = self.visit(node.value, **kwargs)
        if isinstance(value, Foldable):
            folded = getattr(value.value, node.attr)

            error_if_not_foldable(folded, node.location)

            try:
                return self.inline_closure_var(folded, node.attr, node.location)
            except NotFoldableError:
                raise errors.DSLError(  # noqa: B904
                    node.location,
                    f"Unexpected closure variable attribute {node.attr} of type {type(folded)}",
                )
            # if folded in fbuiltins.BUILTINS.values():
            #     return foast.Name(id=node.attr, location=node.location)
            # if isinstance(folded, types.ModuleType):
            #     return Foldable(value=folded)
            # if isinstance(value.value, FrozenNamespace):
            #     return foast.Constant(value=folded, location=node.location)

        return node
        # TODO errors

    def visit_FunctionDefinition(
        self, node: foast.FunctionDefinition, **kwargs: Any
    ) -> foast.FunctionDefinition:
        result = self.generic_visit(node, current_closure_vars=node.closure_vars, **kwargs)
        return foast.FunctionDefinition(
            id=result.id,
            params=result.params,
            body=result.body,
            closure_vars=result.closure_vars,
            type=result.type,
            location=result.location,
            known_symbols=list(self.new_symbols.values()),
        )
