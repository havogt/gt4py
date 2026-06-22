# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Frontend-only utilities for `ts.PartialTypeSpec` (Approach 3 prototype).

Genericity is expressed by wrapping a target `TypeSpec` class together with its constructor
fields, where some fields are `TypeVarType` placeholders. None of this is visible downstream of
specialization, so the IR/backend never has to narrow `FieldType.dtype`.
"""

from __future__ import annotations

from typing import Any, Sequence

from gt4py.eve import extended_typing as xtyping
from gt4py.next.type_system import type_specifications as ts


def _walk_typevars(value: Any) -> xtyping.Iterator[ts.TypeVarType]:
    """Yield every `TypeVarType` reachable inside a (possibly partial) field value."""
    if isinstance(value, ts.TypeVarType):
        yield value
    elif isinstance(value, ts.PartialTypeSpec):
        for _, v in value.fields:
            yield from _walk_typevars(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk_typevars(v)
    elif isinstance(value, ts.TupleType):
        for v in value.types:
            yield from _walk_typevars(v)
    elif isinstance(value, ts.FieldType):
        yield from _walk_typevars(value.dtype)
    elif isinstance(value, ts.ListType):
        yield from _walk_typevars(value.element_type)


def is_generic(type_: Any) -> bool:
    """A `PartialTypeSpec` (or a structure containing one / a bare `TypeVarType`) is generic."""
    return next(_walk_typevars(type_), None) is not None


def type_vars(type_: Any) -> dict[str, ts.TypeVarType]:
    """Collect the type variables occurring in ``type_``, keyed by name."""
    return {var.name: var for var in _walk_typevars(type_)}


def _substitute_value(value: Any, binding: xtyping.Mapping[str, ts.ScalarType]) -> Any:
    """Replace bound type variables inside a single field value, recursing into nested partials."""
    if isinstance(value, ts.TypeVarType):
        return binding.get(value.name, value)
    if isinstance(value, ts.PartialTypeSpec):
        return substitute(value, binding)
    if isinstance(value, list):
        return [_substitute_value(v, binding) for v in value]
    if isinstance(value, tuple):
        return tuple(_substitute_value(v, binding) for v in value)
    return value


def substitute(
    partial: ts.PartialTypeSpec, binding: xtyping.Mapping[str, ts.ScalarType]
) -> ts.PartialTypeSpec:
    """Return a `PartialTypeSpec` with the bound type variables substituted (still possibly partial)."""
    return ts.PartialTypeSpec(
        target=partial.target,
        fields=tuple((name, _substitute_value(value, binding)) for name, value in partial.fields),
    )


def _materialize(value: Any) -> Any:
    """Turn any nested `PartialTypeSpec` into its concrete `target`, recursively.

    Precondition: ``value`` contains no unbound type variables.
    """
    if isinstance(value, ts.PartialTypeSpec):
        return value.target(**{name: _materialize(v) for name, v in value.fields})
    if isinstance(value, list):
        return [_materialize(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_materialize(v) for v in value)
    return value


def specialize(partial: ts.PartialTypeSpec, binding: xtyping.Mapping[str, ts.ScalarType]) -> ts.TypeSpec:
    """Substitute all type variables and instantiate the concrete ``target`` `TypeSpec`.

    Raises if a type variable remains unbound after substitution (the partial is still generic).
    Nested `PartialTypeSpec`s are materialized too, so the result is fully concrete.
    """
    substituted = substitute(partial, binding)
    if is_generic(substituted):
        unbound = ", ".join(sorted(type_vars(substituted)))
        raise ValueError(f"Cannot specialize '{partial}': unbound type variable(s) {unbound}.")
    result = _materialize(substituted)
    assert isinstance(result, ts.TypeSpec)
    return result


def _bind_var(var: ts.TypeVarType, dtype: ts.TypeSpec) -> dict[str, ts.ScalarType]:
    if not isinstance(dtype, ts.ScalarType):
        return {}
    if dtype not in var.constraints:
        raise ValueError(f"'{dtype}' does not satisfy the constraints of type variable '{var}'.")
    return {var.name: dtype}


def _merge(parts: xtyping.Iterable[dict[str, ts.ScalarType]]) -> dict[str, ts.ScalarType]:
    binding: dict[str, ts.ScalarType] = {}
    for part in parts:
        for name, dtype in part.items():
            if (prev := binding.get(name)) is not None and prev != dtype:
                raise ValueError(
                    f"Type variable '{name}' is bound inconsistently: '{prev}' and '{dtype}'."
                )
            binding[name] = dtype
    return binding


def _bind_value(param_value: Any, arg: ts.TypeSpec) -> dict[str, ts.ScalarType]:
    """Bind type variables in one partial field value against a concrete argument type."""
    if isinstance(param_value, ts.TypeVarType):
        return _bind_var(param_value, arg)
    if isinstance(param_value, ts.PartialTypeSpec):
        return bind(param_value, arg)
    return {}


def bind(partial: ts.PartialTypeSpec, arg: ts.TypeSpec) -> dict[str, ts.ScalarType]:
    """Bind the type variables in ``partial`` by structurally matching the concrete ``arg``.

    Only the dtype field participates today (dtype-scoped, per ADR 0024); generalizing to other
    fields (e.g. dims) is a matter of widening the structural match below.
    """
    if partial.target is ts.FieldType:
        dtype_value = dict(partial.fields).get("dtype")
        # scalar arguments are promoted to zero-dimensional fields
        arg_dtype = arg.dtype if isinstance(arg, ts.FieldType) else arg
        return _bind_value(dtype_value, arg_dtype)
    return {}


def bind_many(
    params: Sequence[ts.TypeSpec], args: Sequence[ts.TypeSpec]
) -> dict[str, ts.ScalarType]:
    """Compute the binding for a full parameter/argument list (mixed partial and concrete)."""
    return _merge(
        bind(param, arg)
        for param, arg in zip(params, args)
        if isinstance(param, ts.PartialTypeSpec)
    )
