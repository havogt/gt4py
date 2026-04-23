# Design: ProcessPool-based compilation for `gt4py.next`

Status: prototype (opt-in behind `GT4PY_BUILD_JOBS_MODE=process`)
Author: prototype investigation, 2026-04-22 (updated same day with DaCe + GPU validation)
Scope: `gt4py.next` only (`gt4py.cartesian` not affected)
Validated backends: `run_gtfn`, `run_gtfn_gpu`, `run_dace_cpu`, `run_dace_gpu`

## Motivation

`CompiledProgramsPool._compile_variant` in `src/gt4py/next/otf/compiled_program.py`
submits compile jobs to a module-level `concurrent.futures.Executor`. It is currently a
`ThreadPoolExecutor`; the `TODO` at line 157 asks for `ProcessPoolExecutor`.

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
5. **Latent name collisions in backends.** `run_gtfn_no_transforms` and
   `roundtrip.no_transforms` shared `.name` with their non-"no_transforms" siblings
   (`run_gtfn` / `roundtrip.default`). The registry we need for name-based backend
   lookup in workers surfaces this as a real bug: the last-registered backend wins,
   so a worker asked to compile "run_gtfn_cpu" would use the no-transforms variant,
   skip `apply_common_transforms`, and fail `visit_SetAt`'s `is_applied_as_fieldop`
   assertion.

Things that are *not* blockers despite being worried about on the first pass:

- `CompileTimeArgs.offset_provider` pickling. Frontend lowering is cheap; even if
  we ship concrete offset providers, the cost is amortized against the downstream
  C++ compile. (Also: offset provider is read-only during compilation.)
- In-memory caches in `workflow.CachedStep`. Workers see an empty cache, but the
  second run hits the on-disk `FileCache` (`filecache.FileCache` under
  `BUILD_CACHE_DIR/gtfn_cache`), which is already cross-process-safe
  (file-locked via `gt4py._core.locking`).
- `id()`-keyed `hash_offset_provider_items_by_id`. Only used main-side to build
  the cache key before submit; never crosses the process boundary.
- Metrics setup (`compile_variant_hook`, `_metrics_source_key_cache`,
  `_pools_per_root`). All runs before submit, main-side; unchanged.

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
  `DaCeCompiler.load` uses `dace.codegen.compiler.load_precompiled_sdfg` to
  rehydrate a live `dace.CompiledSDFG` from the build folder alone — no
  recompile. The existing `CompiledDaceProgram` constructor is reused with
  a minimal `SimpleNamespace` shim standing in for the `BindingSource`.
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
  - `_process_pool_compile_job(backend_name, compilable)` top-level function —
    must be top-level for pickle.
  - Submit site branches on mode: thread submits full `backend.compile`; process
    submits `_process_pool_compile_job` with a main-side-transformed
    `CompilableProgramDef`.
  - `_finish_compilation_job` detects any `CompilationArtifact` result
    (GTFN's `BuildArtifact` or DaCe's `DaCeBuildArtifact`) and runs
    `backend.finalize_artifact(artifact)` main-side (import / precompiled-SDFG
    reload + decoration).

### Worker initializer

In `spawn` mode each worker starts with:

1. Override `compilation.cache._session_cache_dir_path` with the main process's
   path — obstacle 4. The worker's own `TemporaryDirectory` still exists but is
   unused; its cleanup at exit touches only an empty dir.
(Backend modules do NOT need to be eagerly imported: the cloudpickle blob
carries everything the worker needs to materialise the backend, and any modules
the backend references get imported as a side effect of unpickling.)

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
  workers and their loaded compiled modules; re-init costs ~1s per worker for
  `spawn` + eager import. Consider making shutdown on-exit only, and having
  `wait_for_compilation()` just drain outstanding futures.
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
  `GTFNBackendFactory(cmake_build_type=..., ...)`. cloudpickle handles those
  uniformly and costs ~8 KB per submit. The earlier "registry" design also
  surfaced two latent name collisions in the runner modules
  (`run_gtfn_no_transforms` / `roundtrip.no_transforms` shared names with
  their non-"no_transforms" siblings) — those fixes are kept, they're still
  good hygiene for metrics and logs.
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
src/gt4py/next/backend.py                                        + compile_to_artifact, finalize_artifact, serialize_backend_for_worker, deserialize_backend_from_worker
src/gt4py/next/config.py                                         + BUILD_JOBS_MODE
src/gt4py/next/otf/stages.py                                     + CompilationArtifact marker + BuildArtifact
src/gt4py/next/otf/compilation/compiler.py                       + Compiler.build / Compiler.load
src/gt4py/next/otf/recipes.py                                    + OTFCompileWorkflow.build_artifact / finalize_artifact (dispatch via compilation.build/load)
src/gt4py/next/otf/compiled_program.py                           + process-pool branch, worker initializer, artifact finalization
src/gt4py/next/program_processors/runners/dace/workflow/compilation.py   + DaCeBuildArtifact, DaCeCompiler.build / DaCeCompiler.load
src/gt4py/next/program_processors/runners/gtfn.py                * run_gtfn_no_transforms gets name_postfix="_no_transforms"
src/gt4py/next/program_processors/runners/roundtrip.py           * no_transforms renamed "roundtrip_no_transforms"
```

Smoke tests under `tmp/`:

- `tmp/smoke.py` + `tmp/stencil_mod.py` — single-variant gtfn CPU, both modes.
- `tmp/smoke_multi.py` + `tmp/stencil_multi_mod.py` — 4 static-arg variants of
  the same program, exercised with `.compile(...)` + `wait_for_compilation()`.
- `tmp/smoke_dace.py` + `tmp/stencil_dace_mod.py` — DaCe CPU.
- `tmp/smoke_gpu.py` + `tmp/stencil_gpu_mod.py` — gtfn + dace on CUDA GPU.
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
runs clean (366 passed) modulo one pre-existing failure
(`dace_tests/test_dace.py::test_dace_fastcall[exec_alloc_descriptor1]`) that
also fails on untouched `main` — unrelated to this prototype.

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
