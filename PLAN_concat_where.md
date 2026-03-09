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

When a field broadcasts (infinite extent) in the mask dimension:

| Mask type | Field broadcasts in mask dim | Result |
|---|---|---|
| Finite (`I==k`) | true_field broadcasts | **OK** — mask∩infinite is finite |
| Finite (`I==k`) | false_field broadcasts | **FAIL** — complement∩infinite is infinite |
| Semi-infinite (`I<k`) | either field broadcasts | **FAIL** — infinite intersection |

This is a **fundamental limitation of embedded mode**: without backward domain inference,
we cannot determine the finite extent of a broadcast field's contribution. In compiled
mode, the program output domain bounds the computation via backward inference.

**Recommendation**: Fields should have explicit finite extent in the mask dimension.
Broadcasting in non-mask dimensions is always fine (restricted by `_intersect_fields`).

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

---

## 2. Implementation Plan

### Phase 1: Fix existing bugs

1. **Fix `_invert_domain` sorting** — use `sorted_domains` consistently.
2. **Fix mask validation** — `not all(m.ndim == 1 for m in masks)`.
3. **Fix `Dimension.__ne__`** — use proper Domain constructor.

### Phase 2: Multi-dimensional mask support

4. **Add recursive decomposition to `_concat_where`**: When a mask has ndim > 1,
   extract first dimension as 1D mask, recurse with remaining dimensions:
   ```python
   mask_1d = Domain(m[0])
   rest = Domain(*[m[i] for i in range(1, m.ndim)])
   inner = _concat_where(rest, true_field, false_field)
   return _concat_where(mask_1d, inner, false_field)
   ```

5. **Handle tuple-of-multi-dim-domains**: Decompose each before processing.

### Phase 3: Improve error messages

6. **Better error for infinite domains**: When `_concat` encounters infinite domain
   shapes, raise a clear error explaining that the field broadcasts in the mask
   dimension and suggesting to use a field with explicit finite extent.

### Phase 4: Tests

7. **Unit tests for multi-dim masks**: 2D/3D with same-domain fields, partially
   overlapping, subset relationships.

8. **Unit tests for broadcast cases**: Non-mask dim broadcast (works), mask dim
   broadcast with finite mask (works), mask dim broadcast with semi-infinite mask
   (error).

9. **Unit tests for `Dimension.__ne__`** and `Domain.__or__`.

10. **Remove legacy commented-out tests** (boolean-mask-based, lines 1055-1118).

### Phase 5: Cleanup

11. **Update docstring** in `experimental.py`.
12. **Remove type-ignore** on registration line.

### Files to modify

| File | Changes |
|---|---|
| `src/gt4py/next/embedded/nd_array_field.py` | Fix bugs, add multi-dim decomposition, error msgs |
| `src/gt4py/next/common.py` | Fix `Dimension.__ne__` |
| `src/gt4py/next/ffront/experimental.py` | Update docstring |
| `tests/.../test_nd_array_field.py` | Add/update unit tests |
