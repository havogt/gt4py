# Design: split `OTFCompileWorkflow` into `build` + `finalize` phases

Status: design / ready to implement
Base: `gt4py/main` (commit `c1b9c80f8`, April 2026)
Scope: `gt4py.next` only (`gt4py.cartesian` not affected)

## Summary

The on-the-fly compilation pipeline in `gt4py.next.otf` is a `NamedStepSequence`
of four steps: `translation → bindings → compilation → decoration`. The
`compilation` step today conflates two operations with quite different
characteristics — running the native build system (cmake / ninja / nvcc) and
dynamically importing the freshly built module. This proposal splits them and
groups the pipeline into two named sub-workflows, `build` and `finalize`,
connected by an explicit picklable artifact type.

The resulting pipeline:

```
CompilableProgramDef
  ├─ build : OTFBuildWorkflow
  │   ├─ translation  : CompilableProgramDef -> ProgramSource
  │   ├─ bindings     : ProgramSource        -> CompilableProject
  │   └─ compilation  : CompilableProject    -> BuildArtifact         # <- returns an artifact
  └─ finalize : OTFFinalizeWorkflow
      ├─ load        : BuildArtifact        -> ExecutableProgram      # <- new step (dynamic import)
      └─ decoration  : ExecutableProgram    -> ExecutableProgram
```

The composition model (`NamedStepSequence` + `Workflow.__call__`) is unchanged;
`NamedStepSequence` nests naturally inside `NamedStepSequence`, and
`OTFCompileWorkflow.__call__` still runs the full pipeline.

## Motivation

Three things the current shape makes awkward and the new shape makes obvious:

1. **Step responsibility.** The `compilation` step today does two things: it
   runs the build system (filesystem I/O, expensive, its output — the
   compiled `.so` — is a picklable descriptor) and it imports the module
   dynamically (cheap, binds into `sys.modules`, its output is a live
   Python function tied to this interpreter). Splitting them makes each
   step's input and output types align with its actual responsibility.

2. **An explicit artifact boundary.** Several downstream features want to
   stop the pipeline at the `.so`-on-disk point rather than the
   live-Python-callable point:

   - Async compilation (process pools, distributed schedulers) where the
     live callable can't cross the process boundary but a path can.
   - AOT compilation: "compile this program, give me the artifact; I'll
     run it later from a different process."
   - Build caches that key `compilable -> artifact` (already the shape of
     the on-disk `FileCache` used by the `cached_translation` trait).
   - Cross-language packaging / bindings.

   Today each of these either has to duplicate the `Compiler` body or
   thread a side-channel through the pipeline.

3. **Correct placement of the executor-level `CachedStep`.** The
   `cached=True` factory trait currently wraps the whole executor in a
   `CachedStep`, which caches `compilable -> ExecutableProgram`. The
   payload of that cache is a live Python module — not portable across
   processes, not meaningfully reusable beyond a single in-memory run,
   and the wrong granularity (the cheap part, `decoration`, is cached
   alongside the expensive part). After the split, wrapping just the
   `build` sub-workflow caches the right thing (a picklable artifact
   descriptor) at the right granularity.

## Current state on main

Key files, with the shapes that change:

**`src/gt4py/next/otf/recipes.py`** — the workflow:

```python
@dataclasses.dataclass(frozen=True)
class OTFCompileWorkflow(workflow.NamedStepSequence):
    translation: definitions.TranslationStep
    bindings:    workflow.Workflow[stages.ProgramSource, stages.CompilableProject]
    compilation: workflow.Workflow[stages.CompilableProject, stages.ExecutableProgram]
    decoration:  workflow.Workflow[stages.ExecutableProgram, stages.ExecutableProgram]
```

**`src/gt4py/next/otf/definitions.py`** — the protocol that `compilation`
implements:

```python
class CompilationStep(
    workflow.Workflow[
        stages.CompilableProject[CodeSpecT, TargetCodeSpecT], stages.ExecutableProgram
    ],
    Protocol[CodeSpecT, TargetCodeSpecT],
):
    def __call__(
        self, source: stages.CompilableProject[CodeSpecT, TargetCodeSpecT]
    ) -> stages.ExecutableProgram: ...
```

**`src/gt4py/next/otf/compilation/compiler.py`** — the GTFN compiler: runs
the build system **and** dynamically imports the result in one `__call__`.

**`src/gt4py/next/program_processors/runners/dace/workflow/compilation.py`** —
`DaCeCompiler`: runs `sdfg.compile(...)` **and** wraps the returned
`dace.CompiledSDFG` in a `CompiledDaceProgram` in one `__call__`.

**`src/gt4py/next/program_processors/runners/gtfn.py`** — factories:

- `GTFNCompileWorkflowFactory` — flat (translation, bindings, compilation,
  decoration fields, with a `cached_translation` trait on `translation`).
- `GTFNBackendFactory` — has a `cached=True` trait that wraps the whole
  `otf_workflow` in `workflow.CachedStep`.

**`src/gt4py/next/program_processors/runners/dace/workflow/factory.py`** and
**`.../dace/workflow/backend.py`** — same shape as GTFN, plus
`make_dace_backend(...)` which passes kwargs down via
`otf_workflow__bare_translation__...` paths.

## Target design

### `stages.py` — artifact type

```python
@dataclasses.dataclass(frozen=True)
class BuildArtifact:
    """On-disk result of a compilation: everything a later step needs to import it."""
    src_dir: pathlib.Path
    module: pathlib.Path
    entry_point_name: str
```

No marker base class; `BuildArtifact` is just an intermediate type in the
pipeline. Backends with a different on-disk shape (DaCe) define their own
dataclass — it does not need to inherit from `BuildArtifact`.

### `definitions.py` — `CompilationStep` return type

The protocol for the compilation step:

```python
class CompilationStep(
    workflow.Workflow[
        stages.CompilableProject[CodeSpecT, TargetCodeSpecT], stages.BuildArtifact
    ],
    Protocol[CodeSpecT, TargetCodeSpecT],
):
    def __call__(
        self, source: stages.CompilableProject[CodeSpecT, TargetCodeSpecT]
    ) -> stages.BuildArtifact: ...
```

### `compilation/compiler.py` — `Compiler` returns an artifact, `load_artifact` is a separate step

```python
@dataclasses.dataclass(frozen=True)
class Compiler(
    workflow.ChainableWorkflowMixin[..., stages.BuildArtifact],
    workflow.ReplaceEnabledWorkflowMixin[..., stages.BuildArtifact],
    definitions.CompilationStep[CPPLikeCodeSpecT, code_specs.PythonCodeSpec],
):
    cache_lifetime: config.BuildCacheLifetime
    builder_factory: BuildSystemProjectGenerator[...]
    force_recompile: bool = False

    def __call__(self, inp) -> stages.BuildArtifact:
        # existing body, minus the final import_from_path block; returns a BuildArtifact
        ...


def load_artifact(artifact: stages.BuildArtifact) -> stages.ExecutableProgram:
    """Dynamically import a previously-built module and return its entry point.

    Used as the ``load`` step of :class:`OTFFinalizeWorkflow` for C++-based
    backends (GTFN). Must run in the process that will ultimately call the
    returned program, since the module is registered in that process's
    ``sys.modules`` under the ``gt4py.__compiled_programs__.`` prefix.
    """
    m = importer.import_from_path(
        artifact.src_dir / artifact.module,
        sys_modules_prefix="gt4py.__compiled_programs__.",
    )
    return getattr(m, artifact.entry_point_name)
```

The existing `Compiler.__call__` body splits cleanly around the
`importer.import_from_path(...)` block — everything before it stays in
`Compiler.__call__`, the import call moves to `load_artifact`.

### `dace/workflow/compilation.py` — `DaCeCompiler` + `DaCeLoader`

DaCe's on-disk artifact has a different shape (SDFG build folder +
binding source), and its live program (`CompiledDaceProgram`) wraps a
`dace.CompiledSDFG` ctypes handle that cannot cross processes, so the
split is even more natural here.

```python
@dataclasses.dataclass(frozen=True)
class DaCeBuildArtifact:
    build_folder: pathlib.Path
    binding_source_code: str
    bind_func_name: str


@dataclasses.dataclass(frozen=True)
class DaCeCompiler(ChainableWorkflowMixin[..., DaCeBuildArtifact], ...):
    """Run DaCe's build system and produce an on-disk artifact."""
    bind_func_name: str
    cache_lifetime: config.BuildCacheLifetime
    device_type: core_defs.DeviceType
    cmake_build_type: config.CMakeBuildType = config.CMakeBuildType.DEBUG

    def __call__(self, inp) -> DaCeBuildArtifact:
        with gtx_wfdcommon.dace_context(device_type=..., cmake_build_type=...):
            sdfg_build_folder = gtx_cache.get_cache_folder(inp, self.cache_lifetime)
            sdfg_build_folder.mkdir(parents=True, exist_ok=True)
            sdfg = dace.SDFG.from_json(inp.program_source.source_code)
            sdfg.build_folder = sdfg_build_folder
            with locking.lock(sdfg_build_folder):
                sdfg.compile(validate=False, return_program_handle=False)
        assert inp.binding_source is not None
        return DaCeBuildArtifact(
            build_folder=pathlib.Path(sdfg_build_folder),
            binding_source_code=inp.binding_source.source_code,
            bind_func_name=self.bind_func_name,
        )


@dataclasses.dataclass(frozen=True)
class DaCeLoader(ChainableWorkflowMixin[DaCeBuildArtifact, CompiledDaceProgram], ...):
    """Rehydrate a previously-built DaCe artifact into a live CompiledDaceProgram."""
    device_type: core_defs.DeviceType
    cmake_build_type: config.CMakeBuildType = config.CMakeBuildType.DEBUG

    def __call__(self, artifact) -> CompiledDaceProgram:
        # locate the SDFG dump that dace wrote into the build folder:
        for dump_name in ("program.sdfgz", "program.sdfg"):
            sdfg_dump = artifact.build_folder / dump_name
            if sdfg_dump.exists():
                break
        else:
            raise RuntimeError(...)

        sdfg = dace.SDFG.from_file(str(sdfg_dump))
        sdfg.build_folder = str(artifact.build_folder)

        with gtx_wfdcommon.dace_context(device_type=..., cmake_build_type=...):
            # use_cache=True forces dace to load the existing .so without re-codegen.
            with dace.config.set_temporary("compiler", "use_cache", value=True):
                sdfg_program = sdfg.compile(validate=False)

        import types as _types
        binding_source_shim = _types.SimpleNamespace(source_code=artifact.binding_source_code)
        return CompiledDaceProgram(sdfg_program, artifact.bind_func_name, binding_source_shim)
```

Why **not** use `dace.codegen.compiler.load_precompiled_sdfg` here: it was
added in a recent dace release. `SDFG.from_file` + `sdfg.compile` under
`compiler.use_cache = True` does the same thing and is present in every dace
version gt4py currently targets (verified on the icon4py pinned version, where
`load_precompiled_sdfg` is missing).

### `recipes.py` — the two-phase workflow

```python
@dataclasses.dataclass(frozen=True)
class OTFBuildWorkflow(
    workflow.NamedStepSequence[definitions.CompilableProgramDef, stages.BuildArtifact]
):
    translation: definitions.TranslationStep
    bindings:    workflow.Workflow[stages.ProgramSource, stages.CompilableProject]
    compilation: workflow.Workflow[stages.CompilableProject, stages.BuildArtifact]


@dataclasses.dataclass(frozen=True)
class OTFFinalizeWorkflow(
    workflow.NamedStepSequence[stages.BuildArtifact, stages.ExecutableProgram]
):
    load:       workflow.Workflow[stages.BuildArtifact, stages.ExecutableProgram]
    decoration: workflow.Workflow[stages.ExecutableProgram, stages.ExecutableProgram]


@dataclasses.dataclass(frozen=True)
class OTFCompileWorkflow(
    workflow.NamedStepSequence[definitions.CompilableProgramDef, stages.ExecutableProgram]
):
    build:    workflow.Workflow[definitions.CompilableProgramDef, stages.BuildArtifact]
    finalize: workflow.Workflow[stages.BuildArtifact, stages.ExecutableProgram]
```

`NamedStepSequence.__call__` is inherited unchanged; it runs `build` then
`finalize`. No new composition primitive, no slicing helper — the two phases
are directly reachable as attributes.

### `runners/gtfn.py` — factory split

Split the flat `GTFNCompileWorkflowFactory` into:

- `GTFNBuildWorkflowFactory` (Meta: `OTFBuildWorkflow`) — owns
  `translation`, `bindings`, `compilation`. Keeps the `cached_translation`
  Trait (wraps `translation` in a `CachedStep` with the `FileCache`) and the
  `bare_translation` / `translation` LazyAttribute chain for the
  `cached_translation=True` case.
- `GTFNFinalizeWorkflowFactory` (Meta: `OTFFinalizeWorkflow`) — owns
  `load = factory.LazyFunction(lambda: compiler.load_artifact)` and
  `decoration = LazyAttribute(lambda o: functools.partial(convert_args, device=o.device_type))`.
- `GTFNCompileWorkflowFactory` (Meta: `OTFCompileWorkflow`) — two
  `SubFactory` fields, `build` and `finalize`. Forwards
  `cached_translation` to `build` via a `factory.Trait(build__cached_translation=True)`.

`GTFNBackendFactory`'s `cached` trait changes from:

```python
cached = factory.Trait(
    executor=factory.LazyAttribute(
        lambda o: workflow.CachedStep(o.otf_workflow, hash_function=o.hash_function)
    ),
    name_cached="_cached",
)
```

to (moves the cache inward to wrap only `build`, preserving the
`OTFCompileWorkflow` shape of `executor`):

```python
cached = factory.Trait(
    executor=factory.LazyAttribute(
        lambda o: dataclasses.replace(
            o.otf_workflow,
            build=workflow.CachedStep(o.otf_workflow.build, hash_function=o.hash_function),
        )
    ),
    name_cached="_cached",
)
```

Existing `otf_workflow__translation__...` / `otf_workflow__bare_translation__...` /
`otf_workflow__cached_translation=...` kwargs at the `GTFNBackendFactory`
call sites must be updated to route through `build`:

| Before                                                | After                                                       |
|-------------------------------------------------------|-------------------------------------------------------------|
| `otf_workflow__translation__use_imperative_backend`   | `otf_workflow__build__translation__use_imperative_backend`  |
| `otf_workflow__cached_translation=True`               | `otf_workflow__build__cached_translation=True`              |
| `otf_workflow__bare_translation__enable_itir_transforms` | `otf_workflow__build__bare_translation__enable_itir_transforms` |

Affected backend declarations in the file: `run_gtfn_imperative`,
`run_gtfn_cached`, `run_gtfn_gpu_cached`, `run_gtfn_no_transforms`.

### `runners/dace/workflow/factory.py` + `runners/dace/workflow/backend.py` — same split for DaCe

Analogous: `DaCeBuildWorkflowFactory` + `DaCeFinalizeWorkflowFactory` +
`DaCeWorkflowFactory`. The finalize factory's `load` field is a
`DaCeLoader(device_type, cmake_build_type)` built via `factory.LazyAttribute`.

`DaCeBackendFactory.cached` trait gets the same inward-move treatment:

```python
cached = factory.Trait(
    executor=factory.LazyAttribute(
        lambda o: dataclasses.replace(
            o.otf_workflow,
            build=workflow.CachedStep(o.otf_workflow.build, hash_function=o.hash_function),
        )
    ),
    name_cached="_cached",
)
```

`make_dace_backend(...)` updates its kwarg paths analogously: every
`otf_workflow__bare_translation__X` becomes
`otf_workflow__build__bare_translation__X`.

## API implications

The field layout of `OTFCompileWorkflow` changes. Anyone reaching into
`backend.executor.translation` / `.compilation` / `.bindings` /
`.decoration` needs to route through `build` or `finalize`:

| Before                              | After                                     |
|-------------------------------------|-------------------------------------------|
| `backend.executor.translation`      | `backend.executor.build.translation`      |
| `backend.executor.bindings`         | `backend.executor.build.bindings`         |
| `backend.executor.compilation`      | `backend.executor.build.compilation`      |
| `backend.executor.decoration`       | `backend.executor.finalize.decoration`    |

`backend.executor(compilable)` still returns an `ExecutableProgram`; callers
that only invoke the executor see no change.

`Compiler.__call__` returns `BuildArtifact` instead of `ExecutableProgram`.
Anyone constructing a `Compiler` directly and using its result expects the
import to be done separately (via `load_artifact` or the `OTFFinalizeWorkflow`).
`CompilationStep`'s protocol return type changes correspondingly.

`DaCeCompiler.__call__` returns `DaCeBuildArtifact` instead of
`CompiledDaceProgram`. The live `CompiledDaceProgram` comes from
`DaCeLoader(artifact)`.

The `cached=True` trait cache payload changes: it now caches `compilable ->
BuildArtifact` (picklable, persistable) instead of `compilable ->
ExecutableProgram` (a live Python module, not portable). Users of the
in-memory hit rate in repeated in-session calls see a slightly smaller
speedup (the finalize step still runs), but the cached value is now the
right shape and the existing on-disk `FileCache` under `cached_translation`
already handled the cross-run case.

## Test updates

Three existing tests reach into the old executor structure and need the
attribute-path update:

- `tests/next_tests/unit_tests/program_processor_tests/runners_tests/test_gtfn.py::test_backend_factory_trait_device`,
  `test_backend_factory_trait_cached`, `test_backend_factory_build_cache_config`,
  `test_backend_factory_build_type_config`.
- `tests/next_tests/integration_tests/feature_tests/iterator_tests/test_builtins.py`
  (reaches into `run_gtfn.executor.translation.replace(...)`).
- `tests/next_tests/integration_tests/feature_tests/ffront_tests/test_temporaries_with_sizes.py`
  (reaches into `run_gtfn.executor.translation.replace(...)`).

The `test_gtfn.test_backend_factory_trait_cached` check that
`isinstance(cached_version.executor, CachedStep)` becomes
`isinstance(cached_version.executor.build, CachedStep)`.

One non-test source file reaches into the old structure:

- `src/gt4py/next/program_processors/formatters/gtfn.py` —
  `gtfn.GTFNBackendFactory().executor.translation` becomes
  `....executor.build.translation`.

## Validation plan

- `uv run pytest tests/next_tests/unit_tests/otf_tests/
  tests/next_tests/unit_tests/program_processor_tests/runners_tests/
  tests/next_tests/unit_tests/type_system_tests/` should pass without
  changes beyond the attribute-path updates above.
- Smoke-check construction: `gtfn.run_gtfn.executor.build` exists, is an
  `OTFBuildWorkflow`; `.finalize.load` is `compiler.load_artifact`;
  `gtfn.run_gtfn_cached.executor.build` is a `CachedStep`; the same for DaCe.
- Run a small field-operator / program end-to-end on each backend
  (`run_gtfn`, `run_gtfn_cached`, `run_gtfn_gpu`, `run_gtfn_gpu_cached`,
  `run_dace_cpu`, `run_dace_cpu_cached`, `run_dace_gpu`, `run_dace_gpu_cached`)
  and verify results match.

## Non-goals

Intentionally **not** part of this refactor (they are separate, independent
changes that stand on their own and are easier to land one at a time):

- The `CompileTimeArgs.offset_provider → OffsetProviderType` migration
  flagged by the pre-existing TODO in `arguments.py`.
- Any frontend-stage change in `ffront_stages.py` (moving away from raw
  `types.FunctionType` in `DSLDefinitionT.definition`).
- Replacing `factory-boy` with plain constructors.
- Name-uniqueness enforcement on `Backend.name`.
- Any async-compile / process-pool machinery. The new artifact boundary
  makes those cleaner to add later; it doesn't require them.

## File-by-file change list

Edited:

```
src/gt4py/next/otf/stages.py                                            + BuildArtifact dataclass
src/gt4py/next/otf/definitions.py                                       * CompilationStep protocol return type
src/gt4py/next/otf/compilation/compiler.py                              * Compiler.__call__ returns BuildArtifact; + load_artifact free function
src/gt4py/next/otf/recipes.py                                           * split into OTFBuildWorkflow / OTFFinalizeWorkflow / OTFCompileWorkflow
src/gt4py/next/program_processors/runners/gtfn.py                       * GTFN{Build,Finalize,Compile}WorkflowFactory; cached trait moves inward; kwarg paths updated
src/gt4py/next/program_processors/runners/dace/workflow/compilation.py  + DaCeBuildArtifact, DaCeLoader; * DaCeCompiler.__call__ returns DaCeBuildArtifact
src/gt4py/next/program_processors/runners/dace/workflow/factory.py      * DaCe{Build,Finalize,Compile}WorkflowFactory
src/gt4py/next/program_processors/runners/dace/workflow/backend.py      * DaCeBackendFactory.cached trait moves inward; make_dace_backend kwarg paths
src/gt4py/next/program_processors/formatters/gtfn.py                    * attribute path update
tests/next_tests/unit_tests/program_processor_tests/runners_tests/test_gtfn.py                      * attribute paths in four tests
tests/next_tests/integration_tests/feature_tests/iterator_tests/test_builtins.py                     * attribute path
tests/next_tests/integration_tests/feature_tests/ffront_tests/test_temporaries_with_sizes.py        * attribute path
```

All edits together are on the order of ~150 lines changed plus ~50 lines
added (the nested factories + `DaCeLoader`). The change is mechanical after
the first file.
