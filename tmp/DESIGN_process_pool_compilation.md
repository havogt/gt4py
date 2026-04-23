# Design: ProcessPool-based compilation for `gt4py.next`

Status: prototype (opt-in behind `GT4PY_BUILD_JOBS_MODE=process`)
Author: prototype investigation, 2026-04-22; revised 2026-04-23 (cloudpickle
handoff, cached-trait transparency, portable DaCe load path, NamespaceProxy
unpickle fix)
Scope: `gt4py.next` only (`gt4py.cartesian` not affected)
Validated backends: `run_gtfn`, `run_gtfn_gpu`, `run_gtfn_cached`,
`run_gtfn_gpu_cached`, `run_dace_cpu`, `run_dace_gpu`, `run_dace_cpu_cached`,
`run_dace_gpu_cached`

## Motivation

`CompiledProgramsPool._compile_variant` in `src/gt4py/next/otf/compiled_program.py`
submits compile jobs to a module-level `concurrent.futures.Executor`. Before
this prototype it was always a `ThreadPoolExecutor`; a long-standing
`TODO(havogt)` comment asked for `ProcessPoolExecutor`.

Threads help because C++ compilation happens in a subprocess and releases the GIL, but:

- the actual frontend / codegen / binding / ITIR-transform work is pure-Python and
  GIL-bound, so thread pools do not parallelize it;
- large icon4py-style workloads with dozens of stencils per program and many
  static-arg variants spend non-trivial time in that Python-side work;
- `ProcessPoolExecutor` sidesteps the GIL for all of it.

## Constraints identified during prototyping

Going across a process boundary forced a concrete picklability / sharing contract on
the pool's work unit. The constraints, in the order they bit us:

1. **Raw user `types.FunctionType` does not round-trip through pickle.** After
   `@gtx.program(backend=...)` / `@gtx.field_operator` decorate a stencil, the module
   attribute points to the `Program` / `FieldOperator` wrapper, not the raw function.
   Pickle serializes functions by `(module, qualname)` reference and then asserts
   "is the resolved attribute the same object?" — it isn't, and pickling errors out
   with `PicklingError: it's not the same object as ...`. This breaks even when the
   stencil lives in a real module (not `__main__`).
2. **The returned `ExecutableProgram` is a function pulled from a dynamically
   imported module** (`gt4py.next.otf.compilation.compiler.Compiler.__call__` →
   `importer.import_from_path(..., sys_modules_prefix="gt4py.__compiled_programs__.")`).
   That module only exists in `sys.modules` of the process that imported it, so
   pickling the function back to the main process fails.
3. **`spawn` is the only safe start method** for a process pool that runs alongside
   user code using threads, CUDA, or MPI. On Linux the multiprocessing default is
   still `fork`, which is unsafe in those contexts (3.14 flips the default away from
   `fork`). `spawn` means every worker pays the `gt4py`+`gridtools_cpp` import cost
   from scratch; that cost is significant and must be paid *once per worker*, not
   once per job.
4. **Session-lifetime build cache is per-process.**
   `gt4py.next.otf.compilation.cache._session_cache_dir` is a `tempfile.Temporary-
   Directory` created at module import. Each process (main + each worker) makes its
   own, and each worker's gets scrubbed at worker exit (via `TemporaryDirectory`'s
   finalizer). If a worker writes a `.so` under its own session dir and returns the
   path, the main process will find the file exists *while the worker is still alive*
   but fail to `dlopen` it after pool shutdown (or simply hit a stale path the next
   run). Workers must share the main process's session dir.
5. **Pickle-recursion in `type_translation.NamespaceProxy`.** The class defines
   `__getattr__` to delegate to `self._object`. When cloudpickle unpickles a
   ``NamespaceProxy``, it probes for dunders (``__setstate__``, ``__reduce_ex__``)
   on the freshly-constructed instance *before* ``_object`` has been restored —
   ``self._object`` then itself falls through to ``__getattr__`` and recurses
   unboundedly. A latent bug, not specific to the process pool, but unreachable
   before cloudpickle-of-backend exercised the path.
6. **`cached=True` trait wraps the whole executor in `CachedStep`.**
   ``run_gtfn_cached`` / ``run_dace_gpu_cached`` have
   ``backend.executor = CachedStep(OTFCompileWorkflow(...))``, not the
   ``OTFCompileWorkflow`` itself. The process-pool path needs ``build_artifact`` /
   ``finalize_artifact`` on whatever ``backend.executor`` is, so ``CachedStep`` has
   to delegate them transparently to its wrapped step.
7. **Latent name collisions in backends.** ``run_gtfn_no_transforms`` shared
   ``.name == "run_gtfn_cpu"`` with ``run_gtfn``; ``roundtrip.no_transforms``
   shared ``"roundtrip"`` with ``roundtrip.default``. This was surfaced by an
   earlier iteration of the prototype (a name-based registry), but the clashing
   names were also misleading to metrics and logs regardless. Fixed by giving
   each a distinct ``name_postfix``; the fix is kept even though the final
   cloudpickle-based design no longer depends on unique names.

Things that are *not* blockers despite being worried about on the first pass:

- `CompileTimeArgs.offset_provider` pickling. Frontend lowering is cheap; even if
  we ship concrete offset providers, the cost is amortized against the downstream
  C++ compile. (Also: offset provider is read-only during compilation.) Verified
  to round-trip correctly for a concrete `V2E` NeighborTable (smoke_conn.py).
- In-memory caches in `workflow.CachedStep`. Workers see an empty cache, and the
  executor-level `CachedStep` (`cached=True` trait) is explicitly bypassed by
  its own `build_artifact` / `finalize_artifact` pass-through — the on-disk
  `FileCache` at the translation layer (`FileCache` under
  `BUILD_CACHE_DIR/gtfn_cache`) does the cross-run / cross-process caching
  (file-locked via `gt4py._core.locking`).
- `id()`-keyed `hash_offset_provider_items_by_id`. Only used main-side to build
  the cache key before submit; never crosses the process boundary.
- Metrics setup (`compile_variant_hook`, `_metrics_source_key_cache`,
  `_pools_per_root`). All runs before submit, main-side; unchanged.
- Pickling `Backend` itself. Stdlib pickle fails (nested name-mangled classes,
  lambdas, DaCe module refs), but cloudpickle handles it in ~8 KB per backend,
  so the extra dep is the only cost.

## Design

### Work unit (picklable contract)

The pool submits only:

```
compile_job(backend_blob: bytes, compilable: CompilableProgramDef) -> CompilationArtifact
```

where:

- `backend_blob` is the `Backend` serialized with :mod:`cloudpickle`, not
  :mod:`pickle`. Stdlib pickle refuses a `Backend`: the executor transitively
  holds lambdas, nested name-mangled classes (`StepSequence.__Steps`), factory-
  boy closures, and DaCe `module` references. cloudpickle serializes those by
  value. Payload is ~8 KB per backend — negligible vs. compile time.
  The alternative of "pass a backend-name / import-path string and
  `pkgutil.resolve_name` in the worker" was considered but rejected: users
  legitimately construct backends on-the-fly via the existing
  `GTFNBackendFactory` / `DaCeBackendFactory`, and those instances have no
  stable module-global home to resolve against.
- `compilable: CompilableProgramDef` is the *post-frontend* artifact:
  `data: itir.Program`, `args: CompileTimeArgs(types, offset_provider, ...)`. All
  fields are dataclasses / Eve nodes / numpy arrays, i.e. trivially pickle-safe.
- The return value is a backend-specific `stages.CompilationArtifact` subclass
  (GTFN: `BuildArtifact` — src_dir + module + entry_point_name; DaCe:
  `DaCeBuildArtifact` — build_folder + binding_source_code + bind_func_name).
  Pickle-safe by construction.

The frontend transform (`DSL types.FunctionType` → `itir.Program`) runs **in the
main process**. This works around obstacle 1 above (raw functions don't round-trip);
it costs main-thread time proportional to frontend passes but is cheap relative to
C++ compile.

New (light) dependency: :mod:`cloudpickle`, imported lazily only when process
mode is active. Thread-mode users pay nothing.

### Split points in the existing workflow

Changes kept deliberately small and localized:

- `gt4py/next/otf/stages.py`: new `CompilationArtifact` (marker base) +
  `BuildArtifact` (GTFN-shape: src_dir + module + entry point) dataclass.
- `gt4py/next/otf/compilation/compiler.py`: split `Compiler.__call__` into
  `Compiler.build(inp) -> BuildArtifact` and `Compiler.load(artifact) ->
  ExecutableProgram`. Existing `Compiler.__call__` is now
  `self.load(self.build(inp))`, unchanged behavior for every caller that
  doesn't explicitly use `.build`.
- `gt4py/next/program_processors/runners/dace/workflow/compilation.py`: same
  split for DaCe: new `DaCeBuildArtifact` (build_folder + binding_source_code
  + bind_func_name), `DaCeCompiler.build` / `DaCeCompiler.load`.
  `DaCeCompiler.load` rehydrates a live `dace.CompiledSDFG` from the build
  folder alone using only long-stable DaCe API: ``dace.SDFG.from_file`` on the
  ``program.sdfgz`` / ``program.sdfg`` dump written by ``build``, followed by
  ``sdfg.compile(validate=False)`` inside a ``compiler.use_cache = True``
  context (which short-circuits to a library load, no re-codegen). The
  newer ``dace.codegen.compiler.load_precompiled_sdfg`` helper would have
  been slightly tidier but doesn't exist on older pinned DaCe versions
  (e.g. the one icon4py ships with). The existing ``CompiledDaceProgram``
  constructor is reused with a minimal ``types.SimpleNamespace`` shim
  standing in for the ``BindingSource``.
- `gt4py/next/otf/workflow.py`: `CachedStep` gains `build_artifact` and
  `finalize_artifact` methods that delegate transparently to the wrapped
  step, so the ``cached=True`` factory trait (which puts
  ``CachedStep(OTFCompileWorkflow(...))`` in ``backend.executor``) doesn't
  block the process-pool path. The in-memory ``compilable ->
  ExecutableProgram`` cache is deliberately bypassed: its payload is the
  final decorated program, which is produced main-side after the worker's
  artifact is finalized, and the on-disk FileCache at the ``compilation``
  layer already caches across runs cross-process.
- `gt4py/next/type_system/type_translation.py`: ``NamespaceProxy.__getattr__``
  now short-circuits dunder probes and uses ``object.__getattribute__`` to
  reach ``_object``, so cloudpickle's dunder probes during unpickle don't
  recurse before ``_object`` is bound.
- `gt4py/next/otf/recipes.py`: `OTFCompileWorkflow` gains `build_artifact(inp)`
  and `finalize_artifact(artifact)` methods. Both dispatch through
  `self.compilation.build` / `self.compilation.load` — i.e. the workflow
  doesn't need to know whether it's GTFN or DaCe, only that its `compilation`
  step implements the `build` / `load` contract.
- `gt4py/next/backend.py`:
  - Same split at the `Backend` level (`compile_to_artifact`, `finalize_artifact`).
  - `serialize_backend_for_worker` / `deserialize_backend_from_worker` helpers
    that wrap `cloudpickle` with a clear error message if cloudpickle is
    missing. No registry, no `__post_init__` side effects.
- `gt4py/next/otf/compiled_program.py`:
  - New `BUILD_JOBS_MODE` config knob (`thread` | `process`); pool selection is
    lazy and guarded (`multiprocessing.parent_process() is not None` means "I am
    a worker, don't create a pool").
  - `_process_pool_compile_job(backend_blob: bytes, compilable)` top-level
    function — must be top-level for pickle. Deserializes the backend via
    cloudpickle and calls ``backend.executor.build_artifact(compilable)``.
  - Submit site branches on mode: thread submits full `backend.compile`; process
    cloudpickles the backend, runs `backend.transforms(...)` main-side, and
    submits ``_process_pool_compile_job(backend_blob, compilable)``.
  - `_finish_compilation_job` detects any `CompilationArtifact` result
    (GTFN's `BuildArtifact` or DaCe's `DaCeBuildArtifact`) and runs
    `backend.finalize_artifact(artifact)` main-side (module import /
    SDFG reload + decoration).

### Worker initializer

In `spawn` mode each worker starts by overriding
`compilation.cache._session_cache_dir_path` with the main process's path —
obstacle 4. The worker's own `TemporaryDirectory` still exists but is unused;
its cleanup at exit touches only an empty dir.

Backend modules are **not** eagerly imported in the initializer: the
cloudpickle blob sent with each job carries everything the worker needs to
materialise the backend, and any modules the backend references get imported
as a side effect of unpickling (Python module import is cached, so this cost
is paid once per worker regardless).

### User contract for the process-pool mode

Opt-in:

```
GT4PY_BUILD_JOBS_MODE=process GT4PY_BUILD_JOBS=<n> python user_script.py
```

Requirements on user code:

- Stencils must be defined in an importable module (not `__main__` or a notebook).
  This is almost always true in real ICON4Py/dawn4py workflows; it does rule out
  one-file smoke scripts unless the stencil module is a sibling file. Smoke script
  example: see `tmp/smoke.py` + `tmp/stencil_mod.py`.
- The entry-point script must use the `if __name__ == "__main__":` guard, because
  `spawn` workers re-execute the main module as a side effect of bootstrap.
- The backend must be cloudpickle-serializable. Standard backends and factory-
  constructed variants are (~8 KB each); backends that hold references to
  `__main__`-defined classes or live file handles will not round-trip.
  `cloudpickle` must be importable (it's imported lazily, only when process
  mode is active — thread-mode users pay nothing).

## What's not addressed in this prototype

- **Generic programs / scan operators.** `CompiledProgramsPool._is_generic`
  branch not exercised.
- **Notebook support.** `__main__`-defined stencils cannot round-trip.
  Running the frontend transform in the main process (as this prototype does)
  covers the `definition_stage` side, but a backend constructed in a notebook
  cell still won't deserialise in a spawned worker because its executor
  transitively references things living in `__main__`. Mitigation would be
  to run the factory-boy construction inside a `cloudpickle.register_pickle
  _by_value(module)` context, or to move backend construction into an
  importable helper module. Out of scope for this prototype.
- **Pool shutdown semantics.** `wait_for_compilation()` currently shuts the
  pool down and re-inits it. This is fine in thread mode (workers are threads
  in the same process, import state sticks). In process mode, shut-down kills
  workers; re-init pays the `spawn` import cost again. Consider making
  shutdown on-exit only, and having `wait_for_compilation()` just drain
  outstanding futures.
- **CompileTimeArgs still holds concrete OffsetProvider.** There's a pre-
  existing TODO on that in `arguments.py:148`. Moving it to `OffsetProviderType`
  would cut pickle cost for GPU backends. Orthogonal to this prototype.

## Findings / decisions worth preserving

- Going through **cloudpickle for the `Backend` handoff** is the pivotal
  design decision. An earlier iteration used a name-based registry
  (`_backend_registry` populated from `Backend.__post_init__`) and a second
  iteration used `pkgutil.resolve_name` on a `"module:attr"` locator scanned
  from `sys.modules`. Both break for factory-constructed backends that live in
  a local variable — a legitimate pattern with the existing
  `GTFNBackendFactory` / `make_dace_backend` factories. cloudpickle handles
  those uniformly and costs ~8 KB per submit. The earlier "registry" design
  surfaced two latent name collisions
  (`run_gtfn_no_transforms` / `roundtrip.no_transforms` shared names with
  their non-"no_transforms" siblings) — those fixes are kept for metrics /
  logging hygiene, even though the final cloudpickle-based design has no
  dependence on unique names.
- The right split inside `Backend.compile` is **after the C++ compile, before
  `importer.import_from_path`** — not after `decoration`. `decoration` must run
  main-side because it wraps the Python function that only exists in the main
  process's `sys.modules`.
- The right split inside the pipeline as a whole is **after `transforms`, before
  `executor`**. Transforms need the Python function (which can be introspected
  in-process freely); everything downstream of transforms is AST/IR material
  that pickles trivially.
- Shared session cache dir is a **worker-initializer concern**, not a
  build-system or cache-lifetime concern. Forcing `BUILD_CACHE_LIFETIME =
  PERSISTENT` would also work but would leak build artifacts to disk that
  users expect to be ephemeral.

## Prototype files

Source changes, all small:

```
src/gt4py/next/backend.py                                              + compile_to_artifact, finalize_artifact, serialize_backend_for_worker, deserialize_backend_from_worker
src/gt4py/next/config.py                                               + BUILD_JOBS_MODE
src/gt4py/next/otf/stages.py                                           + CompilationArtifact marker + BuildArtifact
src/gt4py/next/otf/compilation/compiler.py                             + Compiler.build / Compiler.load
src/gt4py/next/otf/recipes.py                                          + OTFCompileWorkflow.build_artifact / finalize_artifact (dispatch via compilation.build/load)
src/gt4py/next/otf/workflow.py                                         + CachedStep.build_artifact / finalize_artifact (transparent pass-through)
src/gt4py/next/otf/compiled_program.py                                 + process-pool branch, worker initializer, artifact finalization
src/gt4py/next/type_system/type_translation.py                         * NamespaceProxy.__getattr__: guard against dunder probes + unbound `_object` (unpickle-safe)
src/gt4py/next/program_processors/runners/dace/workflow/compilation.py + DaCeBuildArtifact, DaCeCompiler.build / DaCeCompiler.load (portable dace API: SDFG.from_file + compile(use_cache=True))
src/gt4py/next/program_processors/runners/gtfn.py                      * run_gtfn_no_transforms gets name_postfix="_no_transforms"
src/gt4py/next/program_processors/runners/roundtrip.py                 * no_transforms renamed "roundtrip_no_transforms"
pyproject.toml                                                         + cloudpickle (new dep; currently under `test` group, should move to a proper optional feature group)
```

Smoke tests under `tmp/`:

- `tmp/smoke.py` + `tmp/stencil_mod.py` — single-variant gtfn CPU, both modes.
- `tmp/smoke_multi.py` + `tmp/stencil_multi_mod.py` — 4 static-arg variants of
  the same program, exercised with `.compile(...)` + `wait_for_compilation()`.
- `tmp/smoke_dace.py` + `tmp/stencil_dace_mod.py` — DaCe CPU.
- `tmp/smoke_gpu.py` + `tmp/stencil_gpu_mod.py` — gtfn + dace on CUDA GPU.
- `tmp/smoke_cached.py` + `tmp/stencil_cached_mod.py` — `run_gtfn_cached` /
  `run_dace_cpu_cached` (``cached=True`` trait → ``CachedStep`` wraps the
  executor).
- `tmp/smoke_gpu_cached.py` + `tmp/stencil_gpu_cached_mod.py` — cached + GPU
  (matches the icon4py failure path for ``run_dace_gpu_cached_opt``-shaped
  backends).
- `tmp/smoke_conn.py` + `tmp/stencil_conn_mod.py` — unstructured mesh
  `neighbor_sum` with a concrete `V2E` connectivity (NeighborTable) in the
  offset provider, to verify concrete connectivity arrays round-trip under
  `spawn`.
- `tmp/smoke_factory_local.py` — backend built via `GTFNBackendFactory(...)`
  in the smoke script itself, kept in a local variable (never assigned to a
  module global). Exercises the case the earlier registry / `pkgutil
  .resolve_name` designs couldn't handle.

Observed wall-times (4 variants, `GT4PY_BUILD_JOBS=4`, release-unoptimised
user-site, small stencil):

- synchronous (`BUILD_JOBS=0`): 12.18 s
- thread pool: 9.43 s
- process pool: 9.73 s (+ ~0.3 s spawn / import overhead)

On workloads with more and larger stencils (ICON4Py scale) the process-pool
advantage widens — that is the use case to measure next.

Regression: `tests/next_tests/unit_tests/program_processor_tests/runners_tests/`
runs clean (366 passed, 4 xfailed, ~11 min) modulo one pre-existing failure
(`dace_tests/test_dace.py::test_dace_fastcall[exec_alloc_descriptor1]`, which
also fails on untouched `main` and is therefore deselected). `otf_tests/` +
`type_system_tests/` + `runners_tests/test_gtfn.py` together: 194 passed.

## Refactoring opportunities

These are things the prototype would have been **cleaner** with. None is a
prerequisite for shipping — this section is a punch-list of follow-up work
ordered roughly by "how much of the prototype's awkwardness goes away."

### 1. Make "the split" an OTF-native pipeline boundary

The OTF composition model in `otf/workflow.py` is built on a single primitive:
a `Workflow[StartT, EndT]` Protocol whose only verb is `__call__(inp) -> out`.
Every composition mechanism (`NamedStepSequence`, `StepSequence`,
`MultiWorkflow`, `CachedStep`, `ChainableWorkflowMixin.chain`) builds on that
one arrow. To stay inside that model, the boundary between "runs in a worker
process" and "runs in the caller's process" must be a *step boundary in the
pipeline*, not a method pair living on individual steps.

The `compilation` step hides two operations with very different characteristics:

1. Codegen + native build — expensive, filesystem-bound, its output
   (``.so`` path + entry-point name) is picklable.
2. Dynamically import the freshly built module — cheap, ``sys.modules``-bound,
   its output (a live Python function) is not picklable.

Pulling those apart into two named steps makes the boundary explicit:

```
translation: CompilableProgramDef -> ProgramSource
bindings:    ProgramSource        -> CompilableProject
compilation: CompilableProject    -> BuildArtifact
load:        BuildArtifact        -> ExecutableProgram
decoration:  ExecutableProgram    -> ExecutableProgram
```

`BuildArtifact` is just an intermediate type in the named sequence. It has no
special supertype and no marker interface — it simply happens to be picklable,
which is the whole point.

**One new capability on `NamedStepSequence`**: slicing by step name.

```python
class NamedStepSequence:
    def __call__(self, inp):                       # unchanged
        for name in self.step_order:
            inp = getattr(self, name)(inp)
        return inp

    def subsequence(self, start: str | None = None,
                    stop:  str | None = None) -> Workflow:
        """Slice the named sequence by step name (half-open, like slicing)."""
        ...
```

The process pool is a pair of slices:

```python
# worker
artifact = backend.executor.subsequence(stop="load")(compilable)
# main (after cloudpickle round-trip of `artifact`)
executable = backend.executor.subsequence(start="load")(artifact)
```

**Consequences:**

- Every compile step exposes its behaviour via `__call__` only. No
  backend-specific artifact classes, no marker bases, no runtime `hasattr`
  dispatch.
- `Backend.compile` is the sole public compile entry point. Async strategies
  slice `self.executor` directly and have no need for separate
  `compile_to_artifact` / `finalize_artifact` API.
- The pipeline gains a type-checkable artifact boundary useful well beyond
  the process pool (distributed build caches, AOT compilation to disk,
  cross-language bindings all want to stop at the same point).

**Wrinkles worth flagging:**

- `MultiWorkflow.step_order(inp)` is dynamic; `subsequence` only makes sense
  on `NamedStepSequence`. `Transforms` uses `MultiWorkflow`, but the frontend
  runs main-side anyway — no conflict.
- `StepSequence` (the result of `.chain()`) holds an anonymous tuple, so it
  has no named split points. Backends that want to be process-pool-friendly
  should use `NamedStepSequence`. All built-in runners already do.
- `subsequence(start, stop)` can't in general statically type-check its output
  vs. input — it's a low-level slicing tool. A few named helpers on
  `OTFCompileWorkflow` (e.g. `build_phase`, `load_phase`) with concrete
  signatures recover the type story where it matters.

### 2. Frontend-as-AST end-to-end

`DSLFieldOperatorDef.definition` / `DSLProgramDef.definition` holds a raw
`types.FunctionType`. The prototype is forced to run the
DSL→FOAST/PAST→itir transform **in the main process** for two reasons:

- Raw Python functions generally don't round-trip through pickle (the
  `__main__.run_add_one` / wrapper-vs-raw-function issue we hit early).
- Closure-captured variables are lost on pickle-by-reference.

A cleaner design: the `@gtx.program` / `@gtx.field_operator` decorator
captures the function as `(SourceDefinition, closure_vars)` at decoration
time — infrastructure that already exists in `ffront/source_utils.py` and
is used by `fingerprint_stage`. Downstream stages never see `types
.FunctionType`. Then the *entire* pipeline, including frontend passes, can
run in a worker. Removes the "frontend stays main-side" carve-out and
lets process-pool parallelism scale to the frontend too (worth doing for
ICON4Py where `past_to_itir` is non-trivial on large programs).

### 3. Cache at the `compilation` step, not around the whole workflow

The `cached=True` trait (`run_gtfn_cached`, `run_dace_gpu_cached`) wraps the
whole executor in `workflow.CachedStep(OTFCompileWorkflow(...))`, caching
`compilable -> ExecutableProgram` in memory. That's the wrong granularity:
the payload is an already-imported Python module, which is neither picklable
nor safe to share across processes, and it duplicates what the on-disk
`FileCache` (attached to the `translation` step via the `cached_translation`
trait) already does cross-process.

Under refactor #1 the natural place to cache is the `compilation` step
itself: wrap it in a `CachedStep[CompilableProject, BuildArtifact]`. The
payload is picklable, the cache can be file-backed and cross-process, and
its granularity sits cleanly next to the existing `cached_translation`
cache.

### 4. Decouple async strategy from `CompiledProgramsPool`

`CompiledProgramsPool._compile_variant` should not know how compilation is
dispatched; it should ask an `AsyncCompileStrategy` to do it. One
implementation per pool mode, uniform contract:

```python
class AsyncCompileStrategy(Protocol):
    def submit(self, executor: NamedStepSequence, inp) -> Future[ExecutableProgram]: ...
    # process impl: run  executor.subsequence(stop="load")  remotely,
    #               run  executor.subsequence(start="load") in the resolve callback
    # thread impl:  run  executor(inp)                      in a worker thread
    # sync impl:    run  executor(inp)                      inline
```

`CompiledProgramsPool` holds an `AsyncCompileStrategy` and submits through
it. Finalization becomes a future callback inside the strategy, not a
separate bookkeeping path visible to the pool.

### 5. Session cache as a service, not module-level mutable state

`gt4py.next.otf.compilation.cache._session_cache_dir_path` is a module
global that each worker mutates in `_pool_worker_initializer` to point at
the main process's dir. That works but is fragile — anything that imports
`cache` before the initializer runs sees the wrong value, and the
mutation is invisible to readers of the module.

A cleaner interface: `cache.get_cache_folder(...)` takes an explicit
`SessionCache` handle (contextvar-bound in the main process, replaced
with the main's handle on the worker via `initargs`). Same behaviour,
but the "who owns the session dir" question has a single answer visible
in the code.

### 6. Robust proxy-class pattern

`type_translation.NamespaceProxy.__getattr__` recursed on unpickle because
it didn't guard against dunder probes before `_object` was bound. This is
a generic failure mode for any class that defines `__getattr__` to
delegate to an attribute-that-might-not-yet-exist. A small helper in Eve
(or even a recipe in `CODING_GUIDELINES.md`) — something like:

```python
def delegating_getattr(backing_attr: str, key: str) -> Any:
    if key.startswith("_"):
        raise AttributeError(key)
    return getattr(object.__getattribute__(self, backing_attr), key)
```

— would catch the class of bug once, rather than discovering it per-proxy.

### 7. Unique backend names as an invariant

Two backends shipped with the same `.name` before this prototype surfaced
it (`run_gtfn_no_transforms` shared `"run_gtfn_cpu"` with `run_gtfn`;
`roundtrip.no_transforms` shared `"roundtrip"` with `roundtrip.default`).
The `.name` field is user-visible (metrics source keys, logs,
`metrics_source_key`'s `_pools_per_root` counter) — collisions make those
observables misleading.

An assertion in `Backend.__post_init__` enforcing `existing_name != self
.name` would have caught this at import time. The prototype avoids that
check because some tests build Backends to probe factory behaviour and
intentionally reuse names, but a test-only escape hatch
(`_skip_name_check=True`) is cheap.

### 8. Push the `OffsetProvider → OffsetProviderType` TODO through

`CompileTimeArgs.offset_provider` is `OffsetProvider` (concrete
connectivity ndarrays), not `OffsetProviderType`. The pre-existing TODO at
`arguments.py:148` is to move the concrete side elsewhere. For the
process pool this is not blocking — pickling a small concrete connectivity
is fast — but for GPU with large unstructured meshes it becomes
wasteful (cupy arrays round-trip via host memory). Once the TODO is
resolved, the worker only ever sees types, and the offset-provider
pickle cost disappears entirely.

### 9. Replace factory-boy with plain constructors

`GTFNBackendFactory` / `DaCeBackendFactory` use `factory.Factory` with
`LazyAttribute` / `SubFactory` / `Trait`. Factory-boy is designed for test
fixtures with random data; using it as a production composition framework
means every "what does this attribute hold?" question requires reading
the factory DSL and running it mentally. A plain function
`make_gtfn_backend(*, gpu=False, cached=False, ...) -> Backend` would be
easier to read, easier to type-check (factory-boy returns `Any`), and
would make cloudpickle's job even simpler (no factory metaclasses in the
pickle graph). The `cached` trait could then be a call site choice, not a
trait:

```python
def make_gtfn_backend(...): ...                   # raw backend
def cached(b: Backend) -> Backend: ...            # wraps in cached layer
```

Orthogonal to this prototype but would simplify every one of the
refactorings above.

### 10. Collapse `Backend.compile` / `compile_to_artifact` / `finalize_artifact`

Once (1) and (4) are in place, the `Backend` class doesn't need three
separate compile methods. `Backend.compile(...)` becomes the sole entry
point, delegating to the configured `AsyncCompileStrategy`. Sync users
see no difference; process-pool users get the build/load split
transparently. The trade-off is that `Backend.compile` becomes async
(returns a `Future` or `Awaitable`), which is a public-API change — so
this is the last refactor to do, not the first.

---

Together refactors 1, 2, 3 reduce the process-pool integration to a handful
of plumbing lines: the pipeline has an explicit artifact boundary (a `load`
step and a `subsequence` slicer), the frontend runs alongside the rest of
the pipeline in workers, and caching sits at the right layer by construction.
Refactors 4–7 are cleanup with no new features. 8–10 are larger and mostly
independent of the process-pool motivation.

## Open questions for discussion

- Do we want `BUILD_JOBS_MODE` as an env var or a call-time switch on
  `CompiledProgramsPool`? (Env feels right for a first version; it matches
  `BUILD_JOBS`.)
- Should the pool default to `process` once DaCe/GPU are validated, or stay
  opt-in indefinitely? (My lean: opt-in for a release, then flip the default.)
- The `backend.compile_to_artifact` / `backend.finalize_artifact` API is
  currently marked experimental. Stabilising it is low-risk; the question is
  whether it ends up being the "canonical" way to cross a process boundary
  for more downstream users (e.g. distributed build caches).
