"""Direct equivalence test: calling backend.compile(...) vs explicitly
backend.executor(backend.transforms(...))"""
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ.setdefault("GT4PY_BUILD_JOBS", "0")

import numpy as np
import gt4py.next as gtx
from gt4py.next.otf import definitions as otf_defs
from gt4py.next.otf import arguments as otf_args

from stencil_mod import IDim, run_add_one  # noqa: F401


def main() -> None:
    a = gtx.as_field([IDim], np.arange(8, dtype=np.float64))
    out = gtx.as_field([IDim], np.zeros(8, dtype=np.float64))

    pool = run_add_one._compiled_programs  # type: ignore[attr-defined]
    backend = pool.backend
    definition_stage = pool.definition_stage
    program_type = pool.program_type

    arg_types = (
        *program_type.definition.pos_only_args,
        *program_type.definition.pos_or_kw_args.values(),
    )
    compile_time_args = otf_args.CompileTimeArgs(
        offset_provider={},
        column_axis=None,
        args=arg_types,
        kwargs=program_type.definition.kw_only_args,
        argument_descriptor_contexts={},
    )

    # 1) What backend.compile does:
    try:
        fn = backend.compile(definition_stage, compile_time_args=compile_time_args)
        print("[1] backend.compile: OK", type(fn).__name__)
    except Exception as e:
        print(f"[1] backend.compile FAILED: {e!r}")

    # 2) Explicit split:
    try:
        concrete = otf_defs.ConcreteProgramDef(data=definition_stage, args=compile_time_args)
        compilable = backend.transforms(concrete)
        print("[2a] transforms OK -> compilable.data:")
        print("     ", str(compilable.data).replace("\n", "\n      "))
        fn = backend.executor(compilable)
        print("[2b] executor OK", type(fn).__name__)
    except Exception as e:
        print(f"[2] split FAILED: {e!r}")


if __name__ == "__main__":
    main()
