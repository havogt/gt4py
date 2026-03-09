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

The result domain is the **largest meaningful rectangular domain** that can be computed:
- In the mask dimension: the union of (mask ∩ true_field.domain) and
  (complement ∩ false_field.domain), after trimming empty edges
- In non-mask dimensions: the intersection of both fields' domains

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

1. **Result contiguity**: The output must be a contiguous hyper-rectangular domain.
   If the selection pattern would produce holes, that is an error
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

### Corner Cases

#### A. Multi-dimensional domains — recursive 1D decomposition

Multi-dimensional domains like `(I < 2) & (J < 2)` are decomposed into nested 1D
`concat_where` calls:
```
concat_where(d_I & d_J, a, b) → concat_where(d_I, concat_where(d_J, a, b), b)
```

**Verified correct** for all tested cases:
- 2D/3D masks with same-domain fields ✓
- 2D masks with partially overlapping fields ✓ (produces correct largest rectangle)
- 2D masks with I-only true + I×J false ✓ (boundary broadcasts in non-mask dim)

The _intersect_fields step in the inner call restricts both fields to their intersection
in non-mask dims. This is correct behavior: the output must be rectangular, so if
one field doesn't cover a region, the output can't include it. The recursive decomposition
preserves this semantic.

#### B. Broadcasting in the mask dimension

When a field broadcasts (infinite extent) in the mask dimension, the individual field's
domain has infinite shape. However, `_concat` already computes the stacked domain from
all slices via `_stack_domains`. The key insight: for non-mask dimensions, `_intersect_fields`
ensures finite extent; for the mask dimension, each slice's extent is determined by the
intersection with the mask (or its complement) and the other field's domain.

The current `_concat` calls `broadcast_to(f.ndarray, f.domain.shape)` which fails for
infinite domains. The fix: use the **stacked domain** to derive target shapes instead of
each field's individual domain. For the concat dimension, use each field's own finite
slice extent; for non-concat dimensions, use the stacked domain's (finite) extent.

This enables scalar/broadcast boundary values like `concat_where(I == 0, 0.0, interior)`:
- The scalar broadcasts to infinite extent in all dims
- After `_intersect_fields`, non-mask dims become finite
- The mask `I == 0` ∩ infinite → `[0, 1)` (finite)
- The complement `I != 0` ∩ `interior.domain` → finite
- `_concat` uses stacked domain shape → works

**Remaining limitation**: If BOTH the mask and the complement produce infinite slices
(e.g., two scalars with a semi-infinite mask), the stacked domain itself is infinite
and materialization is impossible. This is a genuine embedded-mode limitation requiring
a future lazy field implementation.

#### C. Practical boundary condition patterns

The typical BC use case works correctly:
```python
# interior: I:[1, N), boundary: I:[0, 1) — both finite in mask dim I
concat_where(I < 1, boundary, interior)  → I:[0, N) ✓

# With broadcasting: boundary is J-only (broadcasts in I, the NON-mask dim)
# Works because _intersect_fields restricts I to intersection
concat_where(J < 1, boundary_J_field, interior_IJ)  → depends on I intersection

# For 2D BCs, use sequential 1D concat_where:
step1 = concat_where(I < 1, left_bc, interior)
step2 = concat_where(J < 1, bottom_bc, step1)
```

#### D. Existing bugs

1. **`_invert_domain` sorting bug** (nd_array_field.py:936,955): Uses unsorted
   `domains[0]`/`domains[-1]` instead of `sorted_domains[0]`/`sorted_domains[-1]`.

2. **Mask validation** (nd_array_field.py:994): `any(m.ndim for m in masks) != 1`
   evaluates to `True != 1` = `False` for ANY non-zero ndim, so 2D+ masks silently
   pass. Should be `not all(m.ndim == 1 for m in masks)`.

3. **`Dimension.__ne__`** (common.py:139-141): Constructs `Domain(self, UnitRange(...))`
   but Domain expects NamedRange. `I != 0` crashes. Should use
   `Domain(dims=(self,), ranges=(...))`.

4. **`Domain.__or__` with different dimensions** (common.py:537): Does not validate
   that both domains have the same dimension. For `(I>0) | (J>0)`, it silently returns
   `I>0`, completely dropping the J domain. This is a **silent data loss bug**.

#### E. Missing operations — `&` and `|` with tuples

`Dimension.__ne__` and `Domain.__or__` (disjoint case) return `tuple[Domain, Domain]`.
Python tuples have no `__and__`/`__or__` operators, and Domain has no `__rand__`/`__ror__`.
This means all compound expressions involving `!=` or disjoint `|` fail:

| Expression | LHS | RHS | Status | Needed |
|---|---|---|---|---|
| `D(I) & D(J)` | Domain | Domain | **OK** | — |
| `D(I) \| D(I)` same dim | Domain | Domain | **OK** | — |
| `D(I) \| D(I)` disjoint | Domain | Domain | **OK** → tuple | — |
| `D(I) \| D(J)` diff dim | Domain | Domain | **BUG** (wrong result) | Fix `__or__` |
| `T & D` | tuple | Domain | **TypeError** | Add `__rand__` |
| `D & T` | Domain | tuple | **AttributeError** | Extend `__and__` |
| `T \| D` | tuple | Domain | **TypeError** | Add `__ror__` |
| `D \| T` | Domain | tuple | **AttributeError** | Extend `__or__` |
| `T & T` | tuple | tuple | **TypeError** | Introduce wrapper or handle in `_concat_where` |
| `T \| T` | tuple | tuple | **TypeError** | Introduce wrapper or handle in `_concat_where` |

Where `T` = `tuple[Domain, ...]` (from `!=` or disjoint `|`).

**Practical expressions that are currently broken:**
- `(I != 3) & (J == 0)` — tuple & Domain → TypeError
- `(I != 3) | (J == 0)` — tuple | Domain → TypeError
- `(I > 0) | (J > 0)` — different dims → wrong result (silent bug!)

#### F. Scalars as boundary values

A practical use case: `concat_where(I == 0, 0.0, interior_field)`. The scalar `0.0`
becomes a 0-dimensional field that broadcasts to infinite extent in ALL dimensions.
With a finite mask like `I == 0`, the mask∩infinite produces `[0, 1)` — finite. The
fix in `_concat` (deriving shapes from stacked domain, see B) makes this work.

For semi-infinite masks like `concat_where(I < 1, 0.0, interior_field)`, the scalar's
contribution is `(-∞, 1)` which is infinite. However, `interior_field` provides the
other bound: the stacked domain is `[interior.start, 1) ∪ [1, interior.stop)` which
is finite if `interior_field` has finite extent. So this also works with the fix.

---

## 2. Implementation Plan

### Phase 1: Fix existing bugs

1. **Fix `_invert_domain` sorting** — use `sorted_domains` consistently.
2. **Fix mask validation** — `not all(m.ndim == 1 for m in masks)`.
3. **Fix `Dimension.__ne__`** — use proper Domain constructor.
4. **Fix `Domain.__or__` for different dims** — return tuple (not wrong Domain).

### Phase 2: Domain operator support for tuples

The cleanest approach: keep using plain tuples (no new wrapper type) but add
`__rand__`/`__ror__` to Domain and extend `__and__`/`__or__` to handle tuple arguments.

5. **`Domain.__and__`** — if `other` is a tuple, distribute: return
   `tuple(self & d for d in other)`.
6. **`Domain.__rand__`** — same as `__and__` with reversed args (& is commutative
   for domain intersection): `tuple(d & self for d in other)`.
7. **`Domain.__or__`** — if `other` is a tuple, prepend self to tuple (or merge if
   overlapping same-dim).
8. **`Domain.__ror__`** — append self to tuple.
9. **`tuple & tuple`** — cannot be implemented without a wrapper type. Document
   that `(I != k) & (J != m)` is not directly supported. Users should write
   `(I != k) & (J > m)` then `& (J < m)` separately, or we introduce a
   `DomainUnion` wrapper.

### Phase 3: Multi-dimensional mask support in `_concat_where`

10. **Recursive decomposition for `&` (multi-dim Domain)**: When a mask has ndim > 1,
    decompose dimension-by-dimension:
    ```python
    # concat_where(d_I & d_J, a, b) → concat_where(d_I, concat_where(d_J, a, b), b)
    mask_1d = Domain(m[0])
    rest = Domain(*[m[i] for i in range(1, m.ndim)])
    inner = _concat_where(rest, true_field, false_field)
    return _concat_where(mask_1d, inner, false_field)
    ```

11. **Decomposition for `|` (tuple of different-dim Domains)**: When masks is a tuple
    containing domains on different dimensions:
    ```python
    # concat_where(d_I | d_J, a, b) → concat_where(d_I, a, concat_where(d_J, a, b))
    ```
    Group tuple elements by dimension. Same-dim domains are processed together
    (existing code). Different-dim groups are decomposed via nesting.

12. **Handle mixed tuples**: A tuple may contain both 1D and multi-dim Domains
    (e.g., from `(I != 3) & (J > 0)`). Decompose multi-dim entries first, then
    group 1D entries by dimension.

### Phase 4: Fix `_concat` for infinite/broadcast domains

13. **Derive shapes from stacked domain in `_concat`**: Instead of
    `broadcast_to(f.ndarray, f.domain.shape)`, compute the target shape for non-concat
    dimensions from the stacked domain (which is finite after `_intersect_fields`).
    For the concat dimension, use each field's own extent. This enables scalars and
    broadcast fields as boundary values.

    ```python
    # Instead of: broadcast_to(f.ndarray, f.domain.shape)
    # Use: broadcast_to(f.ndarray, target_shape(f, new_domain, dim))
    # where target_shape uses new_domain's ranges for non-concat dims
    # and f's own range for the concat dim.
    ```

### Phase 5: Tests

14. **Domain construction tests**: All `&`/`|` operator combinations from the table
    in section E, including reverse operations with tuples.

15. **Unit tests for multi-dim masks**: 2D/3D with same-domain fields, partially
    overlapping, subset relationships.

16. **Unit tests for `|` decomposition**: `concat_where(I==0 | J==0, ...)`,
    `concat_where(I!=3, ...)` (tuple of same-dim domains).

17. **Unit tests for broadcast cases**: Non-mask dim broadcast (works), mask dim
    broadcast with finite mask (works), mask dim broadcast with semi-infinite mask
    (error with clear message).

18. **Remove legacy commented-out tests** (boolean-mask-based, lines 1055-1118).

### Phase 6: Cleanup

19. **Update docstring** in `experimental.py`.
20. **Remove type-ignore** on registration line.

### Files to modify

| File | Changes |
|---|---|
| `src/gt4py/next/embedded/nd_array_field.py` | Fix bugs, add multi-dim decomposition, error msgs |
| `src/gt4py/next/common.py` | Fix `Dimension.__ne__` |
| `src/gt4py/next/ffront/experimental.py` | Update docstring |
| `tests/.../test_nd_array_field.py` | Add/update unit tests |
