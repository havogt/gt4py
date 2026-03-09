# Analysis of the Premap Workaround (Issue #1583)

## 1. The Bug

The issue manifests with connectivities where **source dimension == target dimension** and there are
**multiple neighbors** (a LOCAL dimension). The canonical example is `C2E2CO`: a Cell-to-Cell
connectivity via edge neighbors.

```
C2E2CO connectivity:
  domain:   (CellDim, C2E2CODim)   # C2E2CODim is LOCAL
  codomain: CellDim
```

When premapping a `Field[[CellDim, KDim], float]` with this connectivity, the result should be
`Field[[CellDim, C2E2CODim, KDim], float]` — the LOCAL dimension `C2E2CODim` is introduced.

The **original** `kind` heuristic on `NdArrayConnectivityField` was:

```python
ALTER_DIMS if self.domain.dim_index(self.codomain) is None
```

This checks whether the codomain dimension appears in the connectivity's own domain. For C2E2CO,
`CellDim` IS in the domain `(CellDim, C2E2CODim)`, so `dim_index` returns 0 (not None), and
`ALTER_DIMS` is NOT set. The connectivity is classified as **reshuffling** instead of **remapping**.

`_reshuffling_premap` then tries to match the connectivity's 2D domain `(CellDim, C2E2CODim)`
against the field's domain `(CellDim, KDim)` — the LOCAL dim doesn't exist in the field, causing
dimension mismatches during transpose (`ValueError: axes don't match array`).


## 2. The Workaround (commit 48a460f)

Two changes were made:

### Change A — `kind` classification (nd_array_field.py:499-500)

```python
# BEFORE (original):
ConnectivityKind.ALTER_DIMS if self.domain.dim_index(self.codomain) is None

# AFTER (workaround):
ConnectivityKind.ALTER_DIMS if any(dim.kind == DimensionKind.LOCAL for dim in self.domain.dims)
```

### Change B — 0-based indexing fix (later properly merged as #1845)

```python
# BEFORE:
take_indices = tuple(conn_map[dim].ndarray for dim in data.domain.dims)

# AFTER:
take_indices = tuple(
    conn_map[dim].ndarray - data.domain[dim].unit_range.start
    for dim in data.domain.dims
)
```

Change B is a straightforward, correct bug fix — `_reshuffling_premap` was missing the
offset-to-zero-based correction that `_remapping_premap` already had. This is independent of the
classification issue.

Change A is the conceptual workaround. It "works" for known cases but papers over a deeper design
problem.


## 3. Conceptual Problems

### Problem 3.1: `ConnectivityKind` conflates connectivity properties with field-context-dependent behavior

The `ConnectivityKind` 2×2 table is:

| Dims \ Struct | No                       | Yes                        |
|---------------|--------------------------|----------------------------|
| **No**        | Translation (`I → I`)    | Reshuffling (`I × K → K`)  |
| **Yes**       | Relocation (`I → I_half`)| Remapping (`V × V2E → E`)  |

`ALTER_DIMS` means "the dimensions of the **result field** differ from the **input field**." But
this depends on **which field** the connectivity is applied to — it is a property of the
`(connectivity, field)` pair, not the connectivity alone.

Consider `c: (I, K) → K` (the reshuffling example from the table). If applied to:
- `f: I × K → ℝ` → result is `I × K → ℝ` — dims unchanged (reshuffling)
- `f: K → ℝ` → result is `I × K → ℝ` — dim `I` added! (more like remapping)

Yet `ConnectivityKind` is computed as a property of the connectivity alone in
`NdArrayConnectivityField.kind`. The dispatch in `premap()` (lines 293-316) relies entirely on
`connectivity.kind`, never considering the field's domain.

### Problem 3.2: The original heuristic was wrong, and the workaround is a different (also imperfect) heuristic

**Original heuristic**: `ALTER_DIMS ↔ codomain NOT in connectivity domain`
- Works for E2V: domain `(E, E2V_local)`, codomain `V` → V not in domain → remapping ✓
- Fails for C2E2CO: domain `(C, C2E2CO_local)`, codomain `C` → C in domain → reshuffling ✗

**Workaround heuristic**: `ALTER_DIMS ↔ any domain dim is LOCAL`
- Works for E2V: has LOCAL → remapping ✓
- Works for C2E2CO: has LOCAL → remapping ✓
- Works for V2V (single-neighbor, domain `(V,)`): no LOCAL → reshuffling ✓

The workaround works because LOCAL dimensions are never part of a data field's original domain, so
they always introduce new dimensions. But it uses `DimensionKind` as a proxy for structural
analysis. A more direct criterion would be: **ALTER_DIMS whenever the connectivity domain contains
dimensions beyond just the codomain**, i.e., `set(domain.dims) - {codomain}` is non-empty. Let's
verify:

- V2V single: `{V} - {V} = {}` → no ALTER_DIMS → reshuffling ✓
- E2V: `{E, E2V_local} - {V} = {E, E2V_local}` → ALTER_DIMS → remapping ✓
- C2E2CO: `{C, C2E2CO_local} - {C} = {C2E2CO_local}` → ALTER_DIMS → remapping ✓
- `I × K → K` (reshuffling from table): `{I, K} - {K} = {I}` → ALTER_DIMS → **remapping** ✗

That last case shows this criterion also doesn't fully work — the table's reshuffling example
`I × K → K` has extra dims beyond codomain, but it's classified as reshuffling. The correct
classification for that case depends on whether `I` is already in the field domain.

This reinforces that **no connectivity-only heuristic can correctly classify all cases**.

### Problem 3.3: The `__post_init__` assertion is inconsistent with both the old and new `kind` logic

```python
def __post_init__(self) -> None:
    assert self._kind is None or bool(self._kind & ConnectivityKind.ALTER_DIMS) == (
        self.domain.dim_index(self.codomain) is not None
    )
```

This asserts: **`ALTER_DIMS ↔ codomain IS in domain`**.

But the **original** `kind` property computed: **`ALTER_DIMS ↔ codomain NOT in domain`** — the
exact opposite!

And the **workaround** computes: **`ALTER_DIMS ↔ has LOCAL dims`** — a third semantics.

The assertion only doesn't fire because `_kind` is `None` at construction time (short-circuited by
the `or`). But it encodes a contradictory understanding of what `ALTER_DIMS` means, which makes the
code confusing and fragile.

### Problem 3.4: `_reshuffling_premap` vs `_remapping_premap` have overlapping but incompatible capabilities

Both functions perform advanced indexing to rearrange data, but they handle multi-dimensional
connectivity differently:

- `_remapping_premap` uses `xp.take(data, indices, axis=dim_idx)` — clean, handles one
  connectivity, correctly introduces new dimensions
- `_reshuffling_premap` uses `data.__getitem__(tuple_of_index_arrays)` — more general NumPy
  advanced indexing, handles multiple connectivities, but assumes all dims stay the same

When C2E2CO gets misclassified as reshuffling, `_reshuffling_premap` fails because it tries to
match the connectivity's 2D domain `(C, C2E2CO_local)` against the field's domain `(C, K)` — the
LOCAL dim doesn't exist in the field, causing dimension mismatches during transpose.


## 4. Suggested Clean Fix Direction

Rather than trying to make `ConnectivityKind` work as a connectivity-only property, **move the
classification logic into `premap()`** where both connectivity and field are available:

```python
# In premap(), after resolving connectivities:
new_dims = set(connectivity.domain.dims) - {connectivity.codomain}
has_new_dims = bool(new_dims - set(self.domain.dims))

if not alters_struct:
    return _domain_premap(self, *conn_fields)
elif has_new_dims:
    return _remapping_premap(self, conn_fields[0])
else:
    return _reshuffling_premap(self, *conn_fields)
```

This checks whether the connectivity introduces dimensions that are genuinely new to the field,
which is the actual semantic question being asked. The `ConnectivityKind` enum can remain for
documentation and type-level classification, but the runtime dispatch should not rely solely on
`connectivity.kind`.

The `kind` property on `NdArrayConnectivityField` could be simplified or deprecated — or redefined
to only classify `ALTER_STRUCT` (which IS a connectivity-only property: does it rearrange data or
just shift the domain?), leaving the `ALTER_DIMS` determination to the premap dispatch.


## 5. Summary of Issues

| # | Issue                                                            | Severity                       |
|---|------------------------------------------------------------------|--------------------------------|
| 1 | `ALTER_DIMS` is context-dependent but computed context-free      | **Design flaw** — root cause   |
| 2 | `__post_init__` assertion contradicts `kind` property            | Latent bug (masked by default) |
| 3 | `_reshuffling_premap` missing 0-based index correction           | Bug — **fixed** in #1845       |
| 4 | Workaround uses `DimensionKind.LOCAL` as proxy for "new dims"    | Works but conceptually fragile |
| 5 | No test coverage for mixed reshuffling/remapping cases           | Testing gap                    |
