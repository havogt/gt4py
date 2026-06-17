---
tags: []
---

# [Unify `DeferredType` and `TypeVarType`]

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2026-06-17
- **Updated**: 2026-06-17

`ts.DeferredType` and `ts.TypeVarType` are merged into a single class,
`ts.TypeVarType`. A deferred type is now simply a type variable *without
identity* and with *more generalized constraints* (down to no constraint at
all). `ts.DeferredType(...)` survives as a thin factory.

## Context

ADR [0023](0023-Dtype-Generic-Operators.md) introduced `ts.TypeVarType` for
dtype-generic operators and explicitly left `ts.DeferredType` in place: *"The
two mechanisms coexist: `DeferredType` means 'not yet inferred' …;
`TypeVarType` means 'universally quantified over the constraint set'."*

Looking at the two classes side by side, they differ on only two axes:

- **Identity** — `TypeVarType` carries a `name` (two same-named occurrences in
  one signature denote the same type); `DeferredType` is anonymous.
- **Constraint generality** — `TypeVarType` is *value-constrained* (a finite
  tuple of concrete `ScalarType`s, resolving to exactly one); `DeferredType`
  is *category-bounded* (`type[TypeSpec]`, any subclass) or unconstrained
  (`None`).

So a `DeferredType` is exactly a `TypeVarType` with `name is None` and a
generalized/empty constraint. Keeping them as two classes duplicated the
"this position holds a still-open type" concept and the predicates that
recognize it.

## Decision

A single `ts.TypeVarType(TypeSpec)` with three fields:

- `name: Optional[str] = None` — set for a named generic parameter, `None` for
  a deferred (not-yet-inferred) placeholder. This is the discriminator:
  `name is None` ⇔ deferred.
- `bound: Optional[type[TypeSpec] | tuple[type[TypeSpec], ...]] = None` — the
  category constraint inherited from `DeferredType.constraint` (any subclass of
  the given `TypeSpec` class(es)).
- `constraints: tuple[ScalarType, ...] = ()` — the value-constraint set from
  the old `TypeVarType` (non-empty and required only for *named* variables).

Two constraint fields rather than one overloaded field: value-constraints
("resolve to exactly one of these instances") and category-bounds ("any
instance of this class") are semantically different, and the split mirrors
Python's own `TypeVar` (`bound=` vs value constraints) — the natural home for
the `bound=` extension deferred in ADR 0023.

### Base class: `DataType` → `TypeSpec`

`TypeVarType` no longer subclasses `DataType`. A deferred type must be able to
stand in for non-data types (`OffsetType`, `FunctionType`, `ProgramType`, …),
which is why the old `DeferredType` was a `TypeSpec`. It still fits into
`FieldType.dtype`, tuple/named-collection members and `foast.Symbol` because
those unions list `TypeVarType` explicitly. Consequently `isinstance(named_var,
ts.DataType)` is now `False` (it was `True` under 0023); the few sites that
relied on it were updated to also accept `ts.TypeVarType`.

### Compatibility surface

- `ts.DeferredType(constraint=...)` is now a **factory function** returning
  `TypeVarType(name=None, bound=constraint)`. All ~25 construction sites are
  unchanged. As a side effect, the no-argument `ts.DeferredType()` (previously
  a latent `TypeError`) now works.
- `ts.is_deferred(type_) -> TypeGuard[TypeVarType]` replaces
  `isinstance(x, ts.DeferredType)` everywhere — it is `True` iff `x` is a
  `TypeVarType` with `name is None`, exactly preserving prior behavior (a named
  variable is never "deferred").
- Type annotations that referenced `ts.DeferredType` now use `ts.TypeVarType`.

This is a **behavior-preserving structural merge**: `is_concrete`,
`is_generic`, `type_class`, `is_concretizable` and the binding/substitution
utilities keep their semantics. The scan-operator `DeferredType`/fabricated-
`Dimension` hack and dimension genericity remain out of scope (see ADR 0023).

## Consequences

- One concept, one class, one predicate (`is_deferred`) — less duplication.
- A clean place for the future `bound=`-constrained generic (set `name` and
  `bound` together) without another type.
- The `name is None` discriminator must be respected: code that wants "a real
  generic parameter" should check `name is not None`, not merely
  `isinstance(x, TypeVarType)`.
