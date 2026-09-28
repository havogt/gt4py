---
tags: [frontend, gtir, backend]
---

# Vertical Reduce

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2026-09-28
- **Updated**: 2026-09-28

A reduction over (part of) a vertical column that returns a field without the vertical
dimension, e.g. a column sum or maximum.

## Context

Before this change the only way to reduce over a vertical range was a scan followed by reading
its last level, and GTIR has no way to read a field at a fixed level while dropping the dimension.
Users write the same reduction patterns (sums, maxima over a column) repeatedly.

## Decision

### Frontend

```python
reduce(op, range=(KDim, start, stop))(field)  # field without KDim
```

- `range` has the spelling and meaning of `scan`'s: a half-open `[start, stop)` over a vertical
  dimension; `start` and `stop` are integer literals or scalar arguments. It is required.
- `op` is a field operator `(a: T, b: T) -> T` on scalars or tuples of scalars. It must be
  associative; the order in which it is applied is unspecified, which leaves room for a tree
  reduction.
- There is no initial value: the reduction is seeded with the first level. A seed is
  `op(seed, reduce(op, range=...)(field))`.
- The argument is a field, or a tuple of fields for a tuple-valued `op`, and must have the reduced
  dimension.
- An empty range is undefined. A range that is empty in its literal bounds is a compile-time
  error; embedded execution raises for an empty range known at run time.

### GTIR

`reduce` lowers to a field-level builtin, like `concat_where`:

```
column_reduce(λ(a, b) → op(a, b), cartesian_domain(K: [start, stop)), field)
```

Its type is the type of `field` without `K`. Domain inference accesses `field` on the demanded
domain extended by `K: [start, stop)`. The reduce is thus recognisable in GTIR, and analyses can
treat it as a read of the column at the same horizontal point.

### Backends

Backends express `column_reduce` as a forward scan over `[start, stop)` whose carry is
`(is_initialized, value)`, and take its last level:

- gtfn writes the scan into the output without `K`. That output has no vertical stride, so each
  level overwrites the same element and the last level remains. No temporary over `K` is needed.
- dace lowers the scan and copies its last level into the output.
- The embedded iterator (roundtrip) evaluates the fold per point.
- Embedded field execution calls `op` once per level on whole horizontal slices, which traces
  well under JAX.

### Scans over different vertical dimensions

A reduce over `K` next to scans over the staggered `KHalf` must fit in one program. Programs
therefore no longer have a single column axis: each scan (and each reduce) takes its axis from
the vertical dimension of its own domain. This also lifts the restriction for plain scans, which
may now run over different vertical dimensions in one program.

## Consequences

- The GTIR builtin needs support in type inference, domain inference, temporary extraction
  (a nested `column_reduce` is materialised like an `as_fieldop`) and each backend.
- A tree reduction or a backend-native fold (e.g. gridtools `column_stage` returning its final
  carry to a `K`-less output) can replace the scan without changing the frontend or GTIR.

## Alternatives considered

- **Scan plus a projection builtin** (`project(field, K, level)`): an extra GTIR node and a
  temporary over `K`; the reduction is not recognisable as such.
- **Reading a fixed level from a stencil without `K`**: requires an implicit rule that a
  dimension missing from the domain is at index 0, which a plain `deref` would silently misuse.
- **A stencil-level loop builtin**: needs a loop construct in gtfn and closed stencils cannot see
  the range bounds for domain inference.
