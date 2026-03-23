---
tags: []
---

# Limitations of embedded concat_where

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2026-03-12
- **Updated**: 2026-03-17

In embedded execution, `concat_where` is, for now, limited to simple but common cases.

We do not support `concat_where` in cases

- where the domain would be infinite and therefore can't be represented as an ndarray, e.g. `concat_where(I < 0, 0.0, somefield)` where the scalar 0.0 would be broadcasted to a field reaching to -infinity;
- with multi-dimensional domains, e.g. `concat_where(I > 0 | J > 0, a, b)`. These cases need to be represented by a nested `concat_where(I > 0, a, concat_where(J > 0, a, b))`;
- with non-contiguous (disjoint) domain conditions, e.g. `concat_where(I != 0, a, b)`. These cases need to be expressed using nested `concat_where`, e.g. `concat_where(I < 0, a, concat_where(I > 0, a, b))`.

## Context

`concat_where` requires expressing conditions like `I != i`, which would produce two disjoint 1D domains (everything before index `i` and everything after). We need a way to represent these non-contiguous domains.

A complete implementation would require designing how to handle fields on non-hypercubic domains. Currently, `Domain` is a Cartesian product of per-dimension `UnitRange`s, which inherently describes hypercubic (rectangular) regions. Supporting arbitrary non-contiguous domains in multiple dimensions would mean fields could live on non-rectangular regions, requiring fundamental changes to field storage, slicing, and iteration.

## Decision

Non-contiguous (disjoint) domains are **not supported** in the domain expression API:

- `Dimension.__ne__(value)` raises `NotImplementedError` when called with an integer value, since it would produce two disjoint domains.
- `Domain.__or__` raises `NotImplementedError` for both multidimensional domains and for 1D domains that are disjoint (non-overlapping and non-adjacent).

The domain expression API only supports operations that result in a single contiguous `Domain`.

## Consequences

- `concat_where` with `I != i` must be rewritten as `concat_where(I < i, ..., concat_where(I > i, ..., ...))`.
- This keeps the domain expression API simple: all supported operations return a single `Domain`.

## Alternatives considered

### General `concat_where` with multi-dimensional domain conditions

A previous implementation attempted full support for multi-dimensional domain conditions (e.g. `(I != 2) | (K != 5)`) in `concat_where`. This required three layers of complexity:

1. **A `DomainTuple` class** with full algebra: a `tuple` subclass carrying `__and__`, `__or__`, `__rand__`, `__ror__` operators so that expressions like `tuple & Domain`, `Domain & tuple`, and `tuple | tuple` all work. `Domain.__and__` and `Domain.__or__` had to handle tuple operands (distributing intersection over tuple elements, filtering empty results).

2. **Normalization of domain tuples**: to ensure correctness, `DomainTuple` construction required three invariants:
   - All domains promoted to the same rank (missing dimensions filled with infinite ranges).
   - Overlapping domains made non-overlapping via box subtraction (`_subtract_domain`), preserving priority order.
   - Adjacent domains differing in exactly one dimension merged greedily.

   This alone introduced ~100 lines of geometric algorithms (box subtraction producing slab decompositions, greedy merge with O(n²) passes).

3. **Cut-point decomposition in `concat_where`**: for multi-dimensional masks, `_concat_where` could no longer simply concatenate along a single dimension. It required collecting all cut points along the first mask dimension, slicing the field into intervals, determining which mask domains cover each interval, stripping the first dimension, and recursing. This replaced the straightforward 1D concatenation with a recursive decomposition that was harder to reason about and test.

- Bad, because the domain algebra is complex (box subtraction, merge, normalization) and fragile — each operation must maintain the same-rank, non-overlapping, merged invariants.
- Bad, because `concat_where` becomes a recursive decomposition algorithm instead of a simple 1D concatenation, making correctness harder to verify.
- Bad, because supporting multi-dimensional non-consecutive domains raises the fundamental question of how to represent fields on non-hypercubic domains — the `DomainTuple` only describes the *condition* but does not address how the resulting field is stored or indexed.
- Bad, because the use case is narrow: the primary need is vertical level exclusion (`K != i`), which is 1D.

Before implementing a complex `DomainTuple`, we should conclude on (if we want) a concept of non-contiguous fields.
