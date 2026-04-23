"""Call _process_pool_compile_job synchronously in-process."""
import os, sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ.setdefault("GT4PY_BUILD_JOBS", "0")

import pickle
import numpy as np
import gt4py.next as gtx
from gt4py.next.otf import compiled_program as cp
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

    from gt4py.next import backend as gtx_backend_mod
    reg_backend = gtx_backend_mod.get_registered_backend(backend.name)
    print(f"id(pool.backend)={id(backend)}")
    print(f"id(registered)  ={id(reg_backend)}")
    print(f"same instance? {backend is reg_backend}")
    print(f"id(pool.backend.executor)={id(backend.executor)}")
    print(f"id(registered.executor)  ={id(reg_backend.executor)}")
    print(f"pool.backend.executor type: {type(backend.executor).__name__}")
    print(f"registered.executor type:  {type(reg_backend.executor).__name__}")

    # Try registered.executor.build_artifact directly
    try:
        art = reg_backend.executor.build_artifact(compilable)
        print(f"[0] registered.executor.build_artifact: OK -> {art}")
    except Exception as e:
        print(f"[0] registered.executor.build_artifact FAILED: {e!r}")

    # A: direct
    try:
        art = cp._process_pool_compile_job(backend.name, compilable)
        print(f"[A] direct: OK -> {art}")
    except Exception as e:
        print(f"[A] direct FAILED: {e!r}")

    # B: pickle roundtrip
    try:
        restored = pickle.loads(pickle.dumps(compilable))
        art = cp._process_pool_compile_job(backend.name, restored)
        print(f"[B] pickle roundtrip: OK -> {art}")
    except Exception as e:
        print(f"[B] pickle roundtrip FAILED: {e!r}")


if __name__ == "__main__":
    main()
