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

#### A. Multi-dimensional domains — recursive 1D decomposition

Multi-dimensional domains like `(I == 0) & (J == 0)` cannot be handled natively because
their complement is not a hyper-rectangle. The solution is to decompose into nested 1D
calls:
```
concat_where(d_I & d_J, a, b) → concat_where(d_I, concat_where(d_J, a, b), b)
```

**Verified**: This decomposition is correct for all tested cases including 2D and 3D
masks, different field dimensionalities, and mixed finite/semi-infinite masks. The IR
transform (canonicalize_domain_argument.py) already uses this strategy.

#### B. Broadcasting with infinite domains — the central implementation challenge

When a field is broadcast (missing a dimension that gets added via broadcasting), the
broadcast dimension has infinite extent `(-∞, +∞)`. This creates problems when the
mask or its complement intersects with the infinite dimension, producing infinite-domain
slices that cannot be materialized as numpy arrays.

**Systematic analysis of which cases work/fail (1D masks):**

| Mask type | True broadcasts in mask dim | False broadcasts in mask dim |
|---|---|---|
| Finite (`I==k`) | OK (mask∩infinite = finite) | FAIL (complement∩infinite = infinite) |
| Semi-infinite (`I<k`) | FAIL (mask∩infinite = infinite) | FAIL (complement∩infinite = infinite) |

**Root cause**: `_concat` calls `broadcast_to(f.ndarray, f.domain.shape)` which fails
when the domain has infinite extent.

**Solution — "clip infinite slices"**: After computing all true/false slices, determine
the *effective finite range* in the mask dimension from the original fields' domains.
Then clip any infinite-extent slices to this range. This is correct because:

1. A broadcast field has identical values for all positions in the broadcast dimension —
   clipping doesn't lose unique data.
2. The effective range is bounded by the finite field's extent — positions outside this
   range have no data from the finite field, so the output domain can't extend beyond it.
3. When **both** fields are broadcast in the mask dim (both missing it), neither provides
   a finite bound → the output would be infinite → error.

**Algorithm**:
```python
def _effective_mask_range(t_domain, f_domain, mask_dim):
    """Compute finite bounds from the two fields' mask-dim ranges."""
    t_range = t_domain[mask_dim].unit_range
    f_range = f_domain[mask_dim].unit_range
    starts = [r.start for r in [t_range, f_range] if r.start is not Infinity.NEGATIVE]
    stops = [r.stop for r in [t_range, f_range] if r.stop is not Infinity.POSITIVE]
    if not starts or not stops:
        return None  # Both infinite → error
    return UnitRange(min(starts), max(stops))
```

Then clip each infinite-extent slice by replacing its domain (the ndarray stays as-is
since it's shape-1 in the broadcast dim):
```python
clipped_domain = s.domain.replace(mask_dim, NamedRange(mask_dim, clipped_range))
NdArrayField.from_array(s.ndarray, domain=clipped_domain)  # shape-1 dim is still valid
```

**Important**: We must NOT use `_intersect_fields` without `ignore_dims` for the mask
dim. That would intersect both fields' mask-dim ranges, losing data when they have
different but overlapping finite ranges (e.g., true: I:[0,4), false: I:[2,6) → we'd
lose I:[0,2) and I:[4,6)). The `ignore_dims` approach preserves each field's full
mask-dim extent; we only clip the broadcast (infinite) ones.

**Verified working with this approach:**
- Finite mask + false broadcasts: `concat_where(J==1, IJ_field, I_field)` ✓
- Semi-infinite mask + true broadcasts: `concat_where(J<2, I_field, IJ_field)` ✓
- Semi-infinite mask + false broadcasts: `concat_where(J<2, IJ_field, I_field)` ✓
- 2D mask + I-only true + I×J false ✓
- 2D mask + I-only true + J-only false (cross-dimensional) ✓
- Both broadcast in mask dim → correctly needs error ✓

#### C. `Domain.__or__` returning a tuple

`Domain.__or__` returns `Domain | tuple[Domain, Domain]` when domains are disjoint.
The `_concat_where` function already handles `tuple[Domain, ...]` as the mask argument —
multiple 1D masks along the same dimension are processed together.

#### D. `Dimension.__ne__` returning a tuple

`I != 0` returns a tuple of two 1D domains. This is handled by `_concat_where`
accepting `tuple[Domain, ...]`.

#### E. Bug in `_invert_domain`

Line 936 uses `domains[0]` (unsorted) instead of `sorted_domains[0]`, and line 955
uses `domains[-1]` instead of `sorted_domains[-1]`. When multiple domains are passed
in non-sorted order, this produces incorrect results.

#### F. Bug in `_concat_where` mask validation

Line 994: `if any(m.ndim for m in masks) != 1:` — `any()` returns bool, and since
`True == 1` in Python, `True != 1` is `False`. So 2D masks silently pass validation.
Should be: `if not all(m.ndim == 1 for m in masks):`

#### G. Bug in `Dimension.__ne__` — broken Domain construction

`Dimension.__ne__` (common.py:139-141) constructs `Domain(self, UnitRange(...))` but
the Domain constructor expects NamedRange instances. `I != 0` raises ValueError at
runtime. Should use `Domain(dims=(self,), ranges=(...))`.

---

## 2. Implementation Plan

### Phase 1: Fix existing bugs

1. **Fix `_invert_domain` sorting bug** (`nd_array_field.py:936,955`):
   Replace `domains[0]` → `sorted_domains[0]` and `domains[-1]` → `sorted_domains[-1]`.

2. **Fix mask validation** (`nd_array_field.py:994`):
   Change `any(m.ndim for m in masks) != 1` to `not all(m.ndim == 1 for m in masks)`.

3. **Fix `Dimension.__ne__`** (`common.py:139-141`):
   Use `Domain(dims=(self,), ranges=(...))` instead of `Domain(self, UnitRange(...))`.

### Phase 2: Handle broadcast infinite domains in `_concat_where`

4. **Add `_effective_mask_range` helper**: Computes the finite bounding range in the
   mask dimension from the two fields' domains. Returns `None` if both are infinite
   (error case).

5. **Add `_clip_infinite_slices` helper**: After computing true/false slices, clips
   any infinite-extent slice domains to the effective range. The ndarray is unchanged
   (broadcast dim stays size-1); only the domain metadata is updated.

6. **Integrate into `_concat_where`**: After computing slices, before calling `_concat`,
   call `_clip_infinite_slices`. Raise an error if `_effective_mask_range` returns
   `None` (both fields broadcast in mask dim — output would be infinite).

### Phase 3: Support multi-dimensional domains via recursive decomposition

7. **Add multi-dim decomposition to `_concat_where`**: When a mask has ndim > 1,
   extract the first dimension as a 1D mask, recursively call `_concat_where` with the
   remaining dimensions, then call `_concat_where` with the 1D mask on the result.
   ```python
   if m.ndim > 1:
       mask_1d = Domain(m[0])
       rest = Domain(*[m[i] for i in range(1, m.ndim)])
       inner = _concat_where(rest, true_field, false_field)
       return _concat_where(mask_1d, inner, false_field)
   ```

8. **Handle tuple-of-multi-dim-domains**: When `_concat_where` receives a tuple of
   domains and some are multi-dimensional, decompose each multi-dim domain before
   processing.

### Phase 4: Enable and add tests

9. **Add unit tests for broadcast cases** (`test_nd_array_field.py`):
   - Finite mask + false broadcasts in mask dim
   - Semi-infinite mask + true broadcasts in mask dim
   - Semi-infinite mask + false broadcasts in mask dim
   - Both broadcast in mask dim → expect error
   - Cross-dimensional: I-only true + J-only false with I×J mask

10. **Add unit tests for multi-dim masks**:
    - 2D mask with same-dim fields
    - 2D mask with I-only true + I×J false
    - 3D mask with same-dim fields
    - Enable existing 2D test (line 1049-1054)

11. **Add unit tests for `Dimension.__ne__`**: Verify `I != 0` creates correct domains.

12. **Remove commented-out legacy tests**: The old boolean-mask-based tests
    (lines 1055-1118) are from the legacy `concat_where` and should be removed.

### Phase 5: Cleanup

13. **Update docstring** of `concat_where` in `experimental.py`.

14. **Remove type-ignore comment** on registration line (`nd_array_field.py:1017`).

### Summary of files to modify

| File | Changes |
|---|---|
| `src/gt4py/next/embedded/nd_array_field.py` | Fix bugs, add infinite-clip + multi-dim decomposition |
| `src/gt4py/next/common.py` | Fix `Dimension.__ne__` |
| `src/gt4py/next/ffront/experimental.py` | Update docstring |
| `tests/.../test_nd_array_field.py` | Enable/add unit tests, remove legacy tests |
| `tests/.../test_concat_where.py` | Verify integration tests pass |

### Estimated complexity

- Phase 1 (bug fixes): Small — 3 targeted fixes
- Phase 2 (infinite domain clipping): Medium — new helpers + integration, well-understood
- Phase 3 (multi-dim decomposition): Small — straightforward recursion, verified working
- Phase 4-5: Medium — test additions and cleanup
