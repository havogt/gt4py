# concat_where: Technical Definition & Implementation Plan

## 1. Technical Definition

### Signature

```python
concat_where(domain: common.Domain, true_field: Field, false_field: Field) -> Field
```

### Semantics

`concat_where` constructs a result field by selecting values from `true_field` where
positions fall inside `domain`, and from `false_field` where positions fall outside
`domain`. Unlike `where` (which is element-wise and requires a boolean mask field),
`concat_where` works on **domain regions** — it *concatenates* slices of the two fields
along the dimensions mentioned in the domain argument.

### Domain Construction (DSL level)

Domains are constructed via comparison operators on `Dimension` objects:

| Expression | Result Domain |
|---|---|
| `I > k` | `I ∈ [k+1, +∞)` |
| `I >= k` | `I ∈ [k, +∞)` |
| `I < k` | `I ∈ (-∞, k)` |
| `I <= k` | `I ∈ (-∞, k+1)` |
| `I == k` | `I ∈ [k, k+1)` |
| `I != k` | `I ∈ (-∞, k) ∪ [k+1, +∞)` — returned as **tuple of two Domains** |

Compound domains via logical operators:

| Expression | Semantics |
|---|---|
| `d1 & d2` | Intersection of domains (via `Domain.__and__`) |
| `d1 \| d2` | Union of domains (via `Domain.__or__`) |

### Key Invariants

1. **Result contiguity**: The output must be a contiguous hyper-rectangular domain (a
   single `Domain`). If the selection pattern would produce holes, that is an error
   (`NonContiguousDomain`).

2. **Complement partitioning**: The domain argument partitions the space into "true
   region" (intersection of mask domain with `true_field.domain`) and "false region"
   (intersection of mask complement with `false_field.domain`). These slices are
   concatenated along the mask dimension(s).

3. **Broadcasting**: Fields may have fewer dimensions than the result. Missing
   dimensions are broadcast (infinite extent). Fields are first intersected in
   dimensions orthogonal to the mask dimension to ensure compatible shapes.

4. **Trimming**: Empty regions at the boundaries of the concatenation are dropped. Only
   interior gaps are errors.

### Corner Cases & Open Questions

#### A. Multi-dimensional domains (the main gap)

Currently the embedded implementation restricts masks to 1D:
```python
if any(m.ndim for m in masks) != 1:
    raise NotImplementedError(...)
```

A multi-dimensional domain like `(I == 0) & (J == 0)` defines a single point in the
I×J plane. The "true region" is that point; the "false region" is the complement, which
is **not a hyper-rectangle** — it's the entire plane minus a point.

**Problem**: `concat_where` semantics require the output to be a single contiguous
hyper-rectangular field. For a multi-dimensional mask, the true/false partition may not
decompose into hyper-rectangles that can be reassembled.

**Current workaround in the backend** (canonicalize_domain_argument.py): The IR transform
decomposes multi-dimensional `&` into nested 1D concat_where expressions:
```
concat_where(d1 & d2, a, b)  →  concat_where(d1, concat_where(d2, a, b), b)
```
Similarly for `|`:
```
concat_where(d1 | d2, a, b)  →  concat_where(d1, a, concat_where(d2, a, b))
```

This decomposition is correct and always yields hyper-rectangular intermediates. The
embedded implementation should follow the same strategy rather than trying to handle
multi-dimensional masks natively.

**Decision**: For multi-dimensional domain arguments in the embedded backend, recursively
decompose into nested 1D `concat_where` calls (mirroring the IR canonicalization), OR
simply call the dimension-wise decomposition upfront before dispatching. This avoids the
hyper-cube assembly problem entirely.

#### B. `Domain.__or__` returning a tuple

`Domain.__or__` currently returns `Domain | tuple[Domain, Domain]` when the two domains
are disjoint. This is a problematic API — callers must always check the return type. The
embedded `_concat_where` already accepts `tuple[Domain, ...]` as its first argument to
handle this, but it makes the code fragile.

#### C. `Dimension.__ne__` returning a tuple

`I != 0` returns `(Domain(..., [-∞, 0)), Domain(..., [1, +∞)))` — a tuple of two
domains, not a single domain. This is passed directly to `_concat_where` which handles
it. However, this is inconsistent with the other operators which return a single `Domain`.

#### D. Broadcasting with infinite domains

When `false_field` is a scalar (0-dimensional), its domain is empty `Domain()`. After
broadcasting, its extent is infinite in all dimensions. When the mask's complement
includes an infinite range (e.g., mask is `I == 0`, complement is `(-∞, 0) ∪ [1, +∞)`),
we cannot materialize the field since it would require infinite memory.

**Current handling**: The `_intersect_fields` function intersects the fields first,
restricting to the finite domain of the other field. This works when at least one field
has a finite domain. If both fields have infinite extent (e.g., both scalars), the result
is only meaningful when the output domain is later restricted by the caller (program
output domain).

#### E. Bug in `_invert_domain`

Line 936 uses `domains[0]` (unsorted) instead of `sorted_domains[0]`, and line 955
uses `domains[-1]` instead of `sorted_domains[-1]`. When multiple domains are passed
in non-sorted order, this produces incorrect results.

#### F. Bug in `_concat_where` mask validation

Line 994: `if any(m.ndim for m in masks) != 1:` — this is semantically wrong.
`any(m.ndim for m in masks)` returns a **bool** (`True`/`False`), not an int. For 1D
masks (ndim=1, truthy), `any(...)` returns `True`, and `True != 1` is `False` in Python
(since `True == 1`), so the check passes. For 2D masks (ndim=2, also truthy),
`any(...)` also returns `True`, and `True != 1` is still `False` — so **2D masks
silently pass the validation**. Only 0D masks (ndim=0, falsy) would be rejected.
Should be: `if not all(m.ndim == 1 for m in masks):`

#### G. Bug in `Dimension.__ne__` — broken Domain construction

`Dimension.__ne__` (common.py:139-141) constructs domains as:
```python
Domain(self, UnitRange(Infinity.NEGATIVE, value))
```
But the `Domain` constructor expects `NamedRange` instances (or keyword `dims`/`ranges`).
Passing `(Dimension, UnitRange)` positionally raises `ValueError`. This means `I != 0`
is completely broken at runtime. Should use:
```python
Domain(dims=(self,), ranges=(UnitRange(Infinity.NEGATIVE, value),))
```

---

## 2. Implementation Plan

### Phase 1: Fix existing bugs

1. **Fix `_invert_domain` sorting bug** (`nd_array_field.py:936,955`):
   Replace `domains[0]` → `sorted_domains[0]` and `domains[-1]` → `sorted_domains[-1]`.

2. **Fix mask validation** (`nd_array_field.py:994`):
   Change `any(m.ndim for m in masks) != 1` to `not all(m.ndim == 1 for m in masks)`.

3. **Fix `Dimension.__ne__`** (`common.py:139-141`):
   Use `Domain(dims=(self,), ranges=(...))` instead of `Domain(self, UnitRange(...))`.

### Phase 2: Support multi-dimensional domains in embedded

3. **Implement recursive decomposition for multi-dim masks**:
   When the mask domain has >1 dimension, decompose it dimension-by-dimension into
   nested `_concat_where` calls. For a mask `Domain(I: r_I, J: r_J)`:
   - Extract the first dimension's range as `mask_1d = Domain(I: r_I)`
   - Create the remaining mask `mask_rest = Domain(J: r_J)`
   - Rewrite as: `_concat_where(mask_1d, _concat_where(mask_rest, true_f, false_f), false_f)`

   This mirrors the IR canonicalization strategy and guarantees that each level
   operates on a 1D mask.

4. **Handle `Domain.__or__` returning tuples in multi-dim context**:
   When `d1 | d2` produces a tuple (disjoint union), the embedded `_concat_where`
   already handles tuple-of-domains as the mask argument. We need to handle the `|`
   decomposition similarly to `&`:
   - `_concat_where((d1, d2), true_f, false_f)` decomposes to
     `_concat_where(d1, true_f, _concat_where(d2, true_f, false_f))`

   This is already partially supported since the function accepts `tuple[Domain, ...]`.

### Phase 3: Handle `__ne__` properly

5. **Support `!=` domain construction**:
   `I != k` produces a tuple of two 1D domains. This is already handled by `_concat_where`
   accepting `tuple[Domain, ...]`. Ensure test coverage for this case.

### Phase 4: Improve Domain operations

6. **Add `Domain.__or__` for multi-dimensional domains** (optional/incremental):
   Currently raises `NotImplementedError` for ndim > 1. For the embedded case, this
   isn't strictly needed if we decompose at the `_concat_where` level. But for a
   cleaner API, consider supporting it.

7. **Type annotation cleanup**: `Domain.__or__` returns `Domain | tuple[Domain, Domain]`
   which is confusing. Consider always returning a tuple or introducing a `DomainUnion`
   type.

### Phase 5: Enable commented-out tests

8. **Enable the 2D test in unit tests** (`test_nd_array_field.py:1049-1054`):
   The test `(D0 == 0) & (D1 == 0)` with 2D fields is already present but may need
   the multi-dim decomposition from Phase 2.

9. **Enable scalar broadcasting test** (`test_nd_array_field.py:1034-1040`):
   `concat_where(D0 == 0, field, scalar)` — requires handling infinite domains from
   scalar broadcast. Mark with `embedded_concat_where_infinite_domain` marker as noted
   in the commented-out code.

10. **Enable remaining commented-out tests** and understand which are still relevant
    (the old mask-based tests are from the legacy implementation and should be removed).

### Phase 6: Tests & documentation

11. **Add tests for `Dimension.__le__` and `Dimension.__ne__`** (marked as TODO in common.py).

12. **Add tests for `Domain.__or__`** (marked as TODO in common.py).

13. **Update docstring** of `concat_where` in `experimental.py` — it still references
    the old mask-field interface.

14. **Remove the type-ignore comment** on the registration line
    (`nd_array_field.py:1017`) which says "this is still the old concat_where".

### Summary of files to modify

| File | Changes |
|---|---|
| `src/gt4py/next/embedded/nd_array_field.py` | Fix bugs, add multi-dim decomposition |
| `src/gt4py/next/common.py` | (Optional) improve `Domain.__or__` |
| `src/gt4py/next/ffront/experimental.py` | Update docstring |
| `tests/.../test_nd_array_field.py` | Enable/add unit tests |
| `tests/.../test_concat_where.py` | Verify integration tests pass |

### Estimated complexity

- Phase 1 (bug fixes): Small, straightforward
- Phase 2 (multi-dim): Medium — the recursive decomposition is well-understood from
  the IR transform; main challenge is handling the intersection/broadcast correctly
  at each recursion level
- Phase 3-6: Small to medium, mostly test additions
