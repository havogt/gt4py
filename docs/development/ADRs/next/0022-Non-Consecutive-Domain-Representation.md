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

### General `concat_where` with multi-dimensional domain conditions

A previous implementation supported multi-dimensional domain conditions (e.g. `(I != 2) | (K != 5)`) in `concat_where`. This required:

1. **A `_DomainTuple` class** with full algebra: a `tuple` subclass carrying `__and__`, `__or__`, `__rand__`, `__ror__` operators so that expressions like `tuple & Domain`, `Domain & tuple`, and `tuple | tuple` all work. `Domain.__and__` and `Domain.__or__` had to handle tuple operands (distributing intersection over tuple elements, filtering empty results).

2. **Normalization of domain tuples**: to ensure correctness, `_DomainTuple` construction required three invariants:
   - All domains promoted to the same rank (missing dimensions filled with infinite ranges).
   - Overlapping domains made non-overlapping via box subtraction (`_subtract_domain`), preserving priority order.
   - Adjacent domains differing in exactly one dimension merged greedily (`_try_merge`, `_merge_domains`).

   This alone introduced ~100 lines of geometric algorithms (box subtraction producing slab decompositions, greedy merge with O(n^2) passes).

3. **Cut-point decomposition in `concat_where`**: for multi-dimensional masks, `_concat_where` could no longer simply concatenate along a single dimension. It required collecting all cut points along the first mask dimension, slicing the field into intervals, determining which mask domains cover each interval, stripping the first dimension, and recursing. This replaced the straightforward 1D concatenation with a recursive decomposition that was harder to reason about and test.

- Bad, because the domain algebra is complex (box subtraction, merge, normalization) and fragile — each operation must maintain the same-rank, non-overlapping, merged invariants.
- Bad, because `concat_where` becomes a recursive decomposition algorithm instead of a simple 1D concatenation, making correctness harder to verify.
- Bad, because supporting multi-dimensional non-consecutive domains raises the fundamental question of how to represent fields on non-hypercubic domains — the `_DomainTuple` only describes the *condition* but does not address how the resulting field is stored or indexed.
- Bad, because the use case is narrow: the primary need is vertical level exclusion (`K != i`), which is 1D.

### Dedicated non-consecutive domain type

Introduce a `DomainUnion` class that stores an arbitrary collection of `Domain` objects.

- Good, because it would be a proper type rather than an untyped tuple convention.
- Bad, because it would suggest general support for non-hypercubic domains without addressing the fundamental question of how fields are stored and accessed on such domains.

## References

- `Domain.__or__` in `src/gt4py/next/common.py`
- `Dimension.__ne__` in `src/gt4py/next/common.py`
- `_concat_where` in `src/gt4py/next/embedded/nd_array_field.py`
