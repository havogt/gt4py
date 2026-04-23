"""Invoke the process-pool worker function synchronously in-process to isolate whether
the bug is in (a) the transforms-in-main / executor-in-worker split or (b) the pickle
round-trip."""
import os
import sys
import pathlib
import pickle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ.setdefault("GT4PY_BUILD_JOBS", "0")  # synchronous baseline

import numpy as np

import gt4py.next as gtx
from gt4py.next.otf import compiled_program as cp
from gt4py.next.otf import definitions as otf_defs

from stencil_mod import IDim, run_add_one  # noqa: F401


def main() -> None:
    # Prime the program so the CompiledProgramsPool exists
    a = gtx.as_field([IDim], np.arange(8, dtype=np.float64))
    out = gtx.as_field([IDim], np.zeros(8, dtype=np.float64))

    # Monkey-patch the pool's _compile_variant to intercept `compilable` and exercise the
    # worker entry point synchronously, with and without a pickle round-trip.
    from gt4py.next.otf.compiled_program import CompiledProgramsPool, _process_pool_compile_job

    original = CompiledProgramsPool._compile_variant

    seen = {}
    def spy(self, *, argument_descriptors, offset_provider, arg_specialization_info=None,
            call_key=None):
        # Replicate the main-side transforms step
        # Need argument_descriptor_contexts; recompute minimally
        from gt4py.next.otf import arguments as otf_args
        compile_time_args = otf_args.CompileTimeArgs(
            offset_provider=offset_provider,
            column_axis=None,
            args=(
                *self.program_type.definition.pos_only_args,
                *self.program_type.definition.pos_or_kw_args.values(),
            ),
            kwargs=self.program_type.definition.kw_only_args,
            argument_descriptor_contexts={},
        )
        concrete = otf_defs.ConcreteProgramDef(data=self.definition_stage, args=compile_time_args)
        compilable = self.backend.transforms(concrete)
        seen["compilable"] = compilable
        # restore original
        CompiledProgramsPool._compile_variant = original
        return original(self, argument_descriptors=argument_descriptors,
                        offset_provider=offset_provider,
                        arg_specialization_info=arg_specialization_info, call_key=call_key)

    CompiledProgramsPool._compile_variant = spy  # type: ignore[method-assign]

    # Trigger a compile via synchronous path to capture `compilable` without any pickling.
    run_add_one(a, out, offset_provider={})
    print("[main] sync compile succeeded, out =", out.asnumpy().tolist())

    compilable = seen["compilable"]
    print("\n-- captured compilable.data --")
    print(compilable.data)

    # 1) Call the worker function directly (no pickle)
    try:
        artifact = _process_pool_compile_job("run_gtfn_cpu", compilable)
        print(f"\n[A] inline worker call: OK -> {artifact}")
    except Exception as e:
        print(f"\n[A] inline worker call FAILED: {e!r}")

    # 2) Pickle roundtrip then worker call
    try:
        restored = pickle.loads(pickle.dumps(compilable))
        artifact = _process_pool_compile_job("run_gtfn_cpu", restored)
        print(f"[B] pickle roundtrip + worker: OK -> {artifact}")
    except Exception as e:
        print(f"[B] pickle roundtrip + worker FAILED: {e!r}")


if __name__ == "__main__":
    main()
