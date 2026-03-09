# concat_where: Technical Definition & Implementation Plan

## 1. Technical Definition

### Signature

```python
concat_where(mask: common.Domain | tuple[common.Domain, ...], true_field: Field, false_field: Field) -> Field
```

### Semantics — concat_where is NOT where

`concat_where` **concatenates** slices of two fields based on domain regions. It is
fundamentally different from `where`:

- **`where(cond, a, b)`**: Element-wise selection using a boolean mask field. The output
  domain equals the intersection of all three inputs. Both `a` and `b` must be defined
  everywhere on the output domain.
- **`concat_where(mask, a, b)`**: Region-based assembly. The mask (a Domain, not a Field)
  partitions the space. `a` only needs to be defined where the mask selects it, and `b`
  only where the complement selects it. The output domain is the **union** of the
  contributed regions (concatenation along the mask dimension), not the intersection.

This is what makes `concat_where` suitable for boundary conditions: the interior field
and the boundary field can be non-overlapping, and `concat_where` stitches them together.

Example:
```python
# interior: I:[1, N), boundary: I:[0, 1) — non-overlapping!
concat_where(I < 1, boundary, interior)  → field with domain I:[0, N)
```

A `where` call would produce an empty result here (no overlap), but `concat_where`
produces the concatenation.

### Naming discussion

`concat_where` is the current name. The "where" suffix risks confusion with element-wise
`where`. Possible alternatives:

| Name | Pro | Con |
|---|---|---|
| `concat_where` | Established, "concat" hints at assembly | "where" misleading |
| `piecewise` | Mathematical term for exactly this | numpy uses it with lists of conditions |
| `select_concat` | Descriptive | Verbose |
| `domain_select` | Clear it's domain-based | Doesn't convey concatenation |
| `concat_fields` | Simple | Doesn't convey conditional aspect |

### Domain Construction (DSL level)

Domains are constructed via comparison operators on `Dimension` objects:

| Expression | Result Domain |
|---|---|
| `I > k` | `I in [k+1, +inf)` |
| `I >= k` | `I in [k, +inf)` |
| `I < k` | `I in (-inf, k)` |
| `I <= k` | `I in (-inf, k+1)` |
| `I == k` | `I in [k, k+1)` |
| `I != k` | `I in (-inf, k) U [k+1, +inf)` — returned as **tuple of two Domains** |

Compound domains via logical operators:

| Expression | Semantics |
|---|---|
| `d1 & d2` | Intersection of domains (via `Domain.__and__`) |
| `d1 \| d2` | Union of domains (via `Domain.__or__`) |

### Key Invariants

1. **Result contiguity**: The output must be a contiguous hyper-rectangular domain.
   If the selection pattern would produce holes, that is an error
   (`NonContiguousDomain`).

2. **Complement partitioning**: The mask partitions the space into "true region"
   (mask intersected with `true_field.domain`) and "false region" (complement of mask
   intersected with `false_field.domain`). These slices are concatenated along the mask
   dimension. Each field only needs to cover its own region.

3. **Broadcasting**: Fields may have fewer dimensions than the result. Missing
   dimensions are broadcast (infinite extent). Fields are first intersected in
   dimensions orthogonal to the mask dimension to ensure compatible shapes.

4. **Trimming**: Empty regions at the boundaries of the concatenation are dropped. Only
   interior gaps are errors.

### Corner Cases

#### A. Multi-dimensional domains — recursive 1D decomposition

Multi-dimensional domains like `(I < 2) & (J < 2)` (which is `Domain(I: (-inf,2), J: (-inf,2))`)
are decomposed into nested 1D `concat_where` calls:
```
concat_where(d_I & d_J, a, b) -> concat_where(d_I, concat_where(d_J, a, b), b)
```

The decomposition works by peeling off one dimension at a time:
```python
first_dim_mask = Domain(mask[mask.dims[0]])  # 1D Domain for first dim
rest_mask = mask[1:]                          # Domain for remaining dims
inner = _concat_where(rest_mask, true_field, false_field)
return _concat_where(first_dim_mask, inner, false_field)
```

#### B. Tuple masks — union decomposition

When `masks` is a tuple of Domains on different dimensions (e.g., from `(I > 0) | (J > 0)`
after fixing `__or__`):
```
concat_where(d_I | d_J, a, b) -> concat_where(d_I, a, concat_where(d_J, a, b))
```

Note the asymmetry: `true_field` is used for the outer call (where d_I holds), and
for the inner call (where d_J holds but d_I doesn't).

Same-dim tuple entries (from `!=` or same-dim `|`) stay grouped and are handled
by the existing 1D logic which already supports multiple mask domains.

#### C. Scalar / broadcast fields as boundary values

A scalar like `0.0` becomes a field with infinite extent in all dimensions after
broadcasting. This works correctly when the mask produces **finite** slices:
```python
# I == 0 intersected with scalar's infinite domain → I:[0,1) — finite
concat_where(I == 0, 0.0, interior_field)  # works
```

**Limitation**: If the combination of mask, complement, and field domains produces
infinite slices (e.g., `concat_where(I < 1, 0.0, other_scalar)` where the complement
`I >= 1` intersected with another scalar gives `I:[1, +inf)`), this cannot be
materialized as a concrete array. This requires a future lazy/constant field. For now,
this is an error.

#### D. Existing bugs

1. **Mask validation** (nd_array_field.py:994): `any(m.ndim for m in masks) != 1`
   is wrong — `any()` returns a bool, so this checks `bool != 1` which is always
   `False` (since `True == 1` in Python). Multi-dim masks silently pass without
   decomposition. Fix: `not all(m.ndim == 1 for m in masks)`.

2. **`Dimension.__ne__`** (common.py): Constructs `Domain(self, UnitRange(...))`
   but Domain expects NamedRange. `I != 0` crashes. Fix: use
   `Domain(dims=(self,), ranges=(...))`.

3. **`Domain.__or__` with different dimensions** (common.py): Does not validate
   that both domains have the same dimension. For `(I>0) | (J>0)`, it silently returns
   `I>0`, completely dropping the J domain. Fix: return tuple for different dims.

#### E. Missing operations — `&` and `|` with tuples

`Dimension.__ne__` and `Domain.__or__` (disjoint/different-dim cases) return
`tuple[Domain, ...]`. Python tuples have no `__and__`/`__or__` operators, and Domain
has no `__rand__`/`__ror__`. This means compound expressions involving tuples fail:

| Expression | LHS | RHS | Status | Needed |
|---|---|---|---|---|
| `D(I) & D(J)` | Domain | Domain | **OK** | — |
| `D(I) \| D(I)` same dim | Domain | Domain | **OK** | — |
| `D(I) \| D(I)` disjoint | Domain | Domain | **OK** -> tuple | — |
| `D(I) \| D(J)` diff dim | Domain | Domain | **BUG** (wrong result) | Fix `__or__` |
| `T & D` | tuple | Domain | **TypeError** | Add `__rand__` |
| `D & T` | Domain | tuple | **AttributeError** | Extend `__and__` |
| `T \| D` | tuple | Domain | **TypeError** | Add `__ror__` |
| `D \| T` | Domain | tuple | **AttributeError** | Extend `__or__` |
| `T & T` | tuple | tuple | **TypeError** | Not supported (document) |
| `T \| T` | tuple | tuple | **TypeError** | Not supported (document) |

Where `T` = `tuple[Domain, ...]`.

---

## 2. Implementation Plan

### Approach: build upon existing implementation

The existing helper structure is clean and well-factored:
- `_invert_domain` — computes complement of 1D domain(s). Already correct.
- `_intersect_multiple` — clips domains to field extent. Clean.
- `_intersect_fields` — broadcasts + intersects non-mask dims. Correct.
- `_concat` — assembles sorted contiguous slices. Works for finite domains.
- `_stack_domains` — validates contiguity. Good.

No rewrite needed. Changes are additive (multi-dim decomposition at the top of
`_concat_where`) plus localized bug fixes.

`_trim_empty_domains` is unused and can be removed.

### Phase 1: Fix existing bugs

1. **Fix mask validation** in `_concat_where` — change to `not all(m.ndim == 1 for m in masks)`.
2. **Fix `Dimension.__ne__`** — use `Domain(dims=(self,), ranges=(...))`.
3. **Fix `Domain.__or__` for different dims** — return tuple instead of wrong Domain.

### Phase 2: Domain operator support for tuples

Keep using plain tuples (no wrapper type), add reverse operators to Domain.

4. **`Domain.__and__`** — if `other` is a tuple, distribute:
   `tuple(self & d for d in other)`.
5. **`Domain.__rand__`** — `tuple(d & self for d in other)`.
6. **`Domain.__or__`** — if `other` is a tuple, append self to tuple
   (or merge if overlapping same-dim).
7. **`Domain.__ror__`** — prepend self to tuple.
8. **`tuple & tuple` / `tuple | tuple`** — not supported. Document limitation.

### Phase 3: Multi-dimensional mask support in `_concat_where`

9. **Recursive decomposition for multi-dim Domain**: When a mask has ndim > 1,
    peel off first dimension and recurse:
    ```python
    first_dim_mask = Domain(mask[mask.dims[0]])
    rest_mask = mask[1:]
    inner = _concat_where(rest_mask, true_field, false_field)
    return _concat_where(first_dim_mask, inner, false_field)
    ```

10. **Decomposition for tuple with different-dim entries**: Group by dimension,
    decompose across dimension groups:
    ```python
    # concat_where(d_I | d_J, a, b) -> concat_where(d_I, a, concat_where(d_J, a, b))
    ```

11. **Validation**: After decomposition, all masks reaching the 1D path must be
    same-dimension, 1D Domains. Error otherwise.

### Phase 4: Tests

12. **Domain construction tests**: All `&`/`|` operator combinations from table E,
    including reverse operations with tuples.

13. **Unit tests for multi-dim masks**: 2D/3D with same-domain fields, partially
    overlapping, subset relationships.

14. **Unit tests for `|` decomposition**: `concat_where(I==0 | J==0, ...)`,
    `concat_where(I!=3, ...)` (tuple of same-dim domains).

15. **Unit tests for broadcast cases**: Non-mask dim broadcast, mask dim
    broadcast with finite mask, scalar boundary values.

16. **Remove legacy commented-out tests** (boolean-mask-based, lines 1055-1118).

### Phase 5: Cleanup

17. **Remove `_trim_empty_domains`** — unused.
18. **Update docstring** in `experimental.py`.
19. **Remove type-ignore** on registration line.

### Files to modify

| File | Changes |
|---|---|
| `src/gt4py/next/embedded/nd_array_field.py` | Fix bugs, add multi-dim decomposition, remove dead code |
| `src/gt4py/next/common.py` | Fix `Dimension.__ne__`, `Domain.__or__`, add `__rand__`/`__ror__` |
| `src/gt4py/next/ffront/experimental.py` | Update docstring |
| `tests/.../test_nd_array_field.py` | Add/update unit tests |
