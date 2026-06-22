# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""PROTOTYPE-ONLY: the FOAST type-deduction impact of Approach 3 (`PartialTypeSpec`).

This is *not* wired into the toolchain. It is a faithful, compilable sketch of how the three
representative deduction operations in `ffront/foast_passes/type_deduction.py` would have to grow
a `PartialTypeSpec` branch, so the frontend invasiveness can be judged from real code instead of
prose.

The contrast with Stage-0 (TypeVar-inside-FieldType): there, a generic field IS a `FieldType`
(with `dtype=TypeVarType`), so `case ts.FieldType(dims, dtype)` and `ts.FieldType(...)`
construction "just work" and the *predicates* (`promote`, `extract_dtype`, ...) were taught about
`TypeVarType` once, centrally. With `PartialTypeSpec` a generic field is a different object, so
every site that *matches* or *constructs* a `FieldType` in a path a generic type can reach needs a
parallel branch.
"""

from __future__ import annotations

from typing import Optional

from gt4py.next.type_system import partial_type_info as pti, type_specifications as ts


# --- representative site 1: `with_altered_scalar_kind` (ffront/foast_passes/type_deduction.py) ---
# Stage-0 today (works unchanged on a generic field, because dtype just happens to be a TypeVarType):
#
#     if isinstance(type_spec, ts.FieldType):
#         return ts.FieldType(dims=type_spec.dims,
#                             dtype=with_altered_scalar_kind(type_spec.dtype, new_scalar_kind))
#
# Approach 3: the predicate flips the dtype to bool, which is concrete, so the result is a concrete
# field even though the input was generic. But the *input* is now a PartialTypeSpec, so we must
# match it explicitly and reach into `fields`:
def with_altered_scalar_kind(
    type_spec: ts.TypeSpec, new_scalar_kind: ts.ScalarKind
) -> ts.ScalarType | ts.FieldType:
    if isinstance(type_spec, ts.PartialTypeSpec) and type_spec.target is ts.FieldType:
        fields = dict(type_spec.fields)
        # a generic dtype becomes a concrete `bool` scalar here -> a concrete FieldType
        return ts.FieldType(dims=fields["dims"], dtype=ts.ScalarType(kind=new_scalar_kind))
    if isinstance(type_spec, ts.FieldType):
        return ts.FieldType(dims=type_spec.dims, dtype=ts.ScalarType(kind=new_scalar_kind))
    if isinstance(type_spec, ts.ScalarType):
        return ts.ScalarType(kind=new_scalar_kind, shape=type_spec.shape)
    raise ValueError(f"Expected field or scalar type, got '{type_spec}'.")


# --- representative site 2: binary-op promotion (`_deduce_binop_type` -> type_info.promote) ---
# Stage-0 today: `type_info.promote(left.type, right.type)` — promote() was taught that
# `promote(T, T) = T` and that mixing T with a concrete dtype is the D3 strict-no-promotion error.
#
# Approach 3: `type_info.promote` is concrete-only again. Generic promotion is a *separate* code
# path keyed on "either operand is a PartialTypeSpec". The D3 rule lives here now, not in promote.
def promote_for_binop(left: ts.TypeSpec, right: ts.TypeSpec) -> ts.TypeSpec:
    l_gen, r_gen = isinstance(left, ts.PartialTypeSpec), isinstance(right, ts.PartialTypeSpec)
    if l_gen or r_gen:
        if not (l_gen and r_gen):
            # D3: a generic dtype combined with a concrete one is a (decoration-time) error.
            raise ValueError(
                f"Could not promote '{left}' and '{right}': a generic dtype can only be"
                " combined with the same type variable."
            )
        l_tvars, r_tvars = pti.type_vars(left), pti.type_vars(right)
        if set(l_tvars) != set(r_tvars):
            raise ValueError(f"Could not promote distinct type variables '{left}' and '{right}'.")
        # `promote(T, T) = T`: dims still promote like concrete fields; here we just return one.
        # (A real impl would promote the `dims` field and keep the shared TypeVar dtype.)
        return left
    # both concrete -> delegate to the untouched concrete promote
    from gt4py.next.type_system import type_info

    return type_info.promote(left, right)  # type: ignore[arg-type]  # concrete branch


# --- representative site 3: subscript / dimension drop (`visit_Subscript`) ---
# Stage-0 today:
#     case ts.FieldType(dims=dims, dtype=dtype):
#         new_type = ts.FieldType(dims=[d for d in dims if d != idx.dim], dtype=dtype)
#
# Approach 3: add a parallel `PartialTypeSpec` arm that rebuilds the partial with the same dtype
# (still a TypeVar) but reduced dims:
def subscript_drop_dim(value_type: ts.TypeSpec, dropped: object) -> Optional[ts.TypeSpec]:
    match value_type:
        case ts.PartialTypeSpec(target=target, fields=fields) if target is ts.FieldType:
            f = dict(fields)
            # `PartialField = object`, so the field value is untyped and must be re-narrowed.
            dims = f["dims"]
            assert isinstance(dims, list)
            return ts.PartialTypeSpec(
                target=ts.FieldType,
                fields=(("dims", [d for d in dims if d != dropped]), ("dtype", f["dtype"])),
            )
        case ts.FieldType(dims=dims, dtype=dtype):
            return ts.FieldType(dims=[d for d in dims if d != dropped], dtype=dtype)
    return None
