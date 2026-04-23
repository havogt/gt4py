"""Compare executor.__call__() vs executor.build_artifact() directly."""
import os, sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ.setdefault("GT4PY_BUILD_JOBS", "0")

import numpy as np
import gt4py.next as gtx
from gt4py.next.otf import definitions as otf_defs
from gt4py.next.otf import arguments as otf_args
from stencil_mod import IDim, run_add_one  # noqa: F401


def main() -> None:
    pool = run_add_one._compiled_programs  # type: ignore[attr-defined]
    backend = pool.backend
    program_type = pool.program_type

    arg_types = (
        *program_type.definition.pos_only_args,
        *program_type.definition.pos_or_kw_args.values(),
    )
    compile_time_args = otf_args.CompileTimeArgs(
        offset_provider={}, column_axis=None, args=arg_types,
        kwargs=program_type.definition.kw_only_args, argument_descriptor_contexts={},
    )
    concrete = otf_defs.ConcreteProgramDef(data=pool.definition_stage, args=compile_time_args)
    compilable = backend.transforms(concrete)
    print(f"executor type: {type(backend.executor).__name__}")

    # (A) executor(compilable)
    try:
        fn = backend.executor(compilable)
        print("[A] executor(compilable): OK")
    except Exception as e:
        print(f"[A] executor(compilable) FAILED: {e!r}")

    # (B) executor.build_artifact(compilable)
    try:
        art = backend.executor.build_artifact(compilable)
        print(f"[B] executor.build_artifact: OK -> {art}")
    except Exception as e:
        print(f"[B] executor.build_artifact FAILED: {e!r}")


if __name__ == "__main__":
    main()
