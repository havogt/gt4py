---
tags: []
---

# Non-Consecutive Domain Representation

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2026-03-12
- **Updated**: 2026-03-12

Non-consecutive domains (e.g. from `Dimension.__ne__`) are represented as tuples of `Domain` objects. This representation is intentionally limited: domain union (`Domain.__or__`) only supports 1D domains, and the tuple representation handles at most 2 disjoint domains.

## Context

`concat_where` requires expressing conditions like `K != i`, which produces two disjoint 1D domains (everything before index `i` and everything after). We need a way to represent these non-consecutive domains.

A complete implementation would require designing how to handle fields on non-hypercubic domains. Currently, `Domain` is a Cartesian product of per-dimension `UnitRange`s, which inherently describes hypercubic (rectangular) regions. Supporting arbitrary non-consecutive domains in multiple dimensions would mean fields could live on non-rectangular regions, requiring fundamental changes to field storage, slicing, and iteration.

## Decision

We use a simple tuple-of-`Domain` representation for non-consecutive domains, restricted to:

- **1D only**: `Domain.__or__` raises `NotImplementedError` for multidimensional domains.
- **At most 2 domains**: `Dimension.__ne__` produces exactly 2 disjoint domains. No attempt is made to support arbitrary numbers of disjoint regions.

This is sufficient for the `concat_where` use case (`K != i` splits a dimension into two parts) without requiring a general solution for non-hypercubic fields.

## Consequences

- `concat_where` works for 1D domain conditions, which covers the primary use case of vertical level exclusion.
- Combining multiple exclusions (e.g. `(K != 2) & (K != 5)`) is not supported because it would require intersecting a `Domain` with a tuple, producing more than 2 disjoint regions.
- A future extension to support non-hypercubic domains would need to address field storage layout, slicing semantics, and how domain operations compose in multiple dimensions.

## Alternatives considered

### Dedicated non-consecutive domain type

Introduce a `DomainUnion` class that stores an arbitrary collection of `Domain` objects.

- Good, because it would be a proper type rather than an untyped tuple convention.
- Bad, because it would suggest general support for non-hypercubic domains without addressing the fundamental question of how fields are stored and accessed on such domains.

## References

- `Domain.__or__` in `src/gt4py/next/common.py`
- `Dimension.__ne__` in `src/gt4py/next/common.py`
- `_concat_where` in `src/gt4py/next/embedded/nd_array_field.py`
