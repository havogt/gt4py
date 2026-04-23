# Design: make ffront DSL stages self-contained

Status: design / ready to implement
Base: `gt4py/main` (commit `c1b9c80f8`, April 2026)
Scope: `gt4py.next.ffront` (ffront stages + parser entry points)

## Summary

Today `DSLFieldOperatorDef` and `DSLProgramDef` (in
`src/gt4py/next/ffront/stages.py`) wrap a raw `types.FunctionType`. Every
frontend parse step (`func_to_foast`, `func_to_past`) then re-extracts
source code, closure variables, and type annotations from that function
object on the compile hot path:

```python
source_def = source_utils.SourceDefinition.from_function(inp.definition)
closure_vars = source_utils.get_closure_vars_from_function(inp.definition)
annotations = typing.get_type_hints(inp.definition)
```

This proposal hoists that extraction to decoration time, stores the results
as first-class fields on the DSL stage, and makes parser steps read them
directly. The live function is retained as a field so embedded execution
can still call it, but the compile pipeline never touches it.

## Motivation

Three reasons to move the source / closure / annotations up:

1. **Extraction cost lives at the wrong layer.** `inspect.getsource`,
   `inspect.getclosurevars`, and `typing.get_type_hints` are called *every*
   time a program is parsed — including in the `func_to_foast` /
   `func_to_past` `CachedStep`s, where the cache key is computed by
   `fingerprint_stage`, which itself re-runs much of the same extraction.
   A single program compile invokes these three APIs multiple times.
   Done once at decoration and stored, the repeated work goes away.

2. **DSL stage as an authoritative unit of information.** Right now the DSL
   stage only *references* the program (via a function handle); the actual
   parsable payload (source + closure + annotations) is a derived artifact
   that has to be re-derived on demand. Making the payload part of the DSL
   stage turns it into a self-contained snapshot: everything a later step
   needs to parse this program into FOAST / PAST is present on the object
   itself, without re-entering `inspect`.

3. **No raw-function dependency on the compile hot path.** Downstream
   features — persistent ahead-of-time caches keyed on stable source
   strings, cross-process compilation that needs a picklable handoff,
   distributed build systems — all have to work around the fact that a
   `types.FunctionType` is not a portable artifact. With the extracted
   triple as the authoritative representation, those features can treat
   the DSL stage as ordinary data.

## Current state on main

`src/gt4py/next/ffront/stages.py`:

```python
@dataclasses.dataclass(frozen=True)
class DSLFieldOperatorDef:
    definition: types.FunctionType
    node_class: type[foast.OperatorNode] = foast.FieldOperator
    attributes: dict[str, Any] = dataclasses.field(default_factory=dict)
    grid_type: Optional[common.GridType] = None
    debug: bool = False


@dataclasses.dataclass(frozen=True)
class DSLProgramDef:
    definition: types.FunctionType
    grid_type: Optional[common.GridType] = None
    debug: bool = False
```

`src/gt4py/next/ffront/func_to_foast.py` (inside `func_to_foast`):

```python
source_def = source_utils.SourceDefinition.from_function(inp.definition)
closure_vars = source_utils.get_closure_vars_from_function(inp.definition)
annotations = typing.get_type_hints(inp.definition)
foast_definition_node = FieldOperatorParser.apply(source_def, closure_vars, annotations)
```

`src/gt4py/next/ffront/func_to_past.py` (inside `func_to_past`): identical
three-line extraction, then `ProgramParser.apply(source_def, closure_vars, annotations)`.

The machinery already exists in `src/gt4py/next/ffront/source_utils.py`:
`SourceDefinition.from_function`, `get_closure_vars_from_function`, and a
`SourceDefinition` frozen dataclass that holds `(source, filename,
line_offset, column_offset)`.

## Target design

### DSL stage fields

```python
@dataclasses.dataclass(frozen=True)
class DSLFieldOperatorDef:
    """DSL-stage field operator definition.

    The authoritative representation for the compile pipeline is the triple
    ``(source_definition, closure_vars, annotations)``, extracted once at decoration
    time. ``definition`` — the live Python function — is retained so embedded
    execution can still call it directly, but frontend / codegen passes read from
    the extracted fields.
    """

    definition: types.FunctionType
    node_class: type[foast.OperatorNode] = foast.FieldOperator
    attributes: dict[str, Any] = dataclasses.field(default_factory=dict)
    grid_type: Optional[common.GridType] = None
    debug: bool = False
    # Populated from ``definition`` in ``__post_init__`` when not supplied.
    source_definition: Optional[source_utils.SourceDefinition] = None
    closure_vars: Optional[dict[str, Any]] = None
    annotations: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.source_definition is None:
            object.__setattr__(
                self,
                "source_definition",
                source_utils.SourceDefinition.from_function(self.definition),
            )
        if self.closure_vars is None:
            object.__setattr__(
                self,
                "closure_vars",
                source_utils.get_closure_vars_from_function(self.definition),
            )
        if self.annotations is None:
            object.__setattr__(self, "annotations", typing.get_type_hints(self.definition))
```

`DSLProgramDef` gets the same three fields and the same `__post_init__`
body. The DSL stages remain `frozen=True`; `object.__setattr__` is the
standard pattern for `__post_init__`-computed fields on frozen dataclasses.

The new fields are declared `Optional[...] = None` purely so that existing
callers of the form `DSLProgramDef(definition=f, grid_type=g)` keep
working; post-construction they are always non-`None`.

`dataclasses.replace(dsl_stage, grid_type=new)` still works as before: the
existing `source_definition` / `closure_vars` / `annotations` are passed
through to `__init__`, `__post_init__` sees them non-`None`, and skips
re-extraction. Good for the grid-type rebinding call site in
`ffront/decorator.py`.

### Parser entry points

`func_to_foast`:

```python
def func_to_foast(inp: DSLFieldOperatorDef) -> FOASTOperatorDef:
    # Read the source / closure / annotations the DSL stage already extracted
    # at decoration time. The live ``inp.definition`` function is no longer
    # touched by the compile pipeline.
    source_def = inp.source_definition
    closure_vars = inp.closure_vars
    annotations = inp.annotations
    assert source_def is not None and closure_vars is not None and annotations is not None
    foast_definition_node = FieldOperatorParser.apply(source_def, closure_vars, annotations)
    ...
```

`func_to_past`: identical change — replace the three extraction calls with
three field reads.

### Nothing else changes

- `FOASTOperatorDef` / `PASTProgramDef` already carry their own
  `closure_vars` field (populated by the parser). Unchanged.
- `fingerprint_stage` walks all dataclass fields via
  `add_stage_to_fingerprint`. It now also incorporates the new fields into
  the hash; this is correct but slightly redundant with the hash of
  `definition` itself (which still goes through
  `add_func_to_fingerprint` → `SourceDefinition.from_function`). Optional
  tightening: specialise `add_stage_to_fingerprint` for `DSLFieldOperatorDef`
  / `DSLProgramDef` to skip the raw `definition` field (correctness is
  unaffected either way).
- `ffront/decorator.py` still uses `self.definition_stage.definition.__name__`
  and `self.definition_stage.definition(*args, **kwargs)` for naming and
  embedded execution. Unchanged — these rely on the retained `definition`
  field.
- Every `DSLProgramDef(definition=...)` / `DSLFieldOperatorDef(definition=...)`
  construction site (two in `ffront/decorator.py`, plus docstrings /
  tests) keeps working because the new fields default to `None` and are
  populated automatically.

## API implications

`DSLFieldOperatorDef` and `DSLProgramDef` gain three public fields:
`source_definition`, `closure_vars`, `annotations`. Post-construction they
are always populated (typed `Optional[...]` only so the three-arg
constructor signature stays stable).

`Compiler`-side callers and downstream tools that previously re-extracted
from `inp.definition` now prefer the stored fields. Nothing forces them to —
calling `source_utils.SourceDefinition.from_function(inp.definition)` still
works — but it's redundant.

## Test updates

None expected. The two parser tests for `func_to_foast` / `func_to_past`
construct DSL stages via the three-arg constructor and then invoke the
workflow step — both forms still work.

Verification this refactor was validated against on a prototype branch:

- `tests/next_tests/unit_tests/ffront_tests/` — 195 passed, 1 skipped,
  3 xfailed.
- `tests/next_tests/unit_tests/otf_tests/` +
  `unit_tests/program_processor_tests/runners_tests/test_gtfn.py` +
  `unit_tests/type_system_tests/` — 194 passed.
- End-to-end compile of gtfn / dace CPU and GPU programs — unchanged
  behaviour.

## Non-goals

- **Removing `definition: types.FunctionType`.** Embedded execution
  (`self.definition_stage.definition(*args, **kwargs)` in
  `decorator.py:408`) calls the live function; several display paths use
  `definition.__name__`. Keeping the field costs nothing and preserves that
  behaviour. The refactor's point is that the *compile pipeline* no longer
  requires it, not that it disappears.
- **Making DSL stages picklable with stdlib pickle.** `closure_vars` can
  hold arbitrary Python objects (modules, other FieldOperators, ...) that
  stdlib pickle doesn't handle. That's orthogonal — tools that need
  cross-process DSL stages continue to use `cloudpickle` or similar.
- **Pruning closure vars to gt4py-relevant entries.** Would be a separate
  clean-up; today `FOASTOperatorDef.closure_vars` already carries the full
  unpruned dict, so the DSL stage matches that behaviour by design.
- **Removing the `CachedStep` wrapper around `func_to_foast` /
  `func_to_past`.** With extraction at decoration time the cache is less
  load-bearing, but still useful for the AST-build cost; leave it in place.

## File-by-file change list

```
src/gt4py/next/ffront/stages.py         * DSLFieldOperatorDef / DSLProgramDef: add source_definition, closure_vars, annotations (+ __post_init__)
src/gt4py/next/ffront/func_to_foast.py  * read inp.source_definition / closure_vars / annotations instead of re-extracting
src/gt4py/next/ffront/func_to_past.py   * same
```

Three files, roughly ~40 lines added and ~6 replaced. The edits are
mechanical and independent of any downstream refactor (`OTFCompileWorkflow`
build/finalize split, async compile pools, etc.).
