"""Run transforms + executor in the main process separately (no pool) to check whether
the split itself is the problem vs something pickling-related."""
import os
import pickle

os.environ.setdefault("GT4PY_BUILD_JOBS", "0")  # synchronous

import numpy as np

import gt4py.next as gtx
from gt4py.next.otf import definitions as otf_defs

from stencil_mod import IDim, run_add_one, add_one  # noqa: F401


def main() -> None:
    from gt4py.next.program_processors.runners import gtfn
    backend = gtfn.run_gtfn

    # Same path as in CompiledProgramsPool._compile_variant, but for the raw field operator.
    op = add_one
    definition_stage = op.definition_stage
    op_future_compiled_programs = op._compiled_programs  # type: ignore[attr-defined]
    program_type = op_future_compiled_programs.program_type
    print(f"definition_stage type: {type(definition_stage).__name__}")
    print(f"program_type: {program_type}")

    from gt4py.next.otf import arguments as otf_args
    compile_time_args = otf_args.CompileTimeArgs(
        args=(program_type.definition.pos_or_kw_args["a"],),
        kwargs={},
        offset_provider={},
        column_axis=None,
        argument_descriptor_contexts={},
    )

    concrete = otf_defs.ConcreteProgramDef(data=definition_stage, args=compile_time_args)
    compilable = backend.transforms(concrete)
    print(f"type(compilable): {type(compilable).__name__}")
    print(f"type(compilable.data): {type(compilable.data).__name__}")
    print("compilable.data:")
    print(compilable.data)

    # Can we pickle the compilable?
    pkl = pickle.dumps(compilable)
    print(f"pickled size: {len(pkl)} bytes")
    restored = pickle.loads(pkl)
    print(f"restored type: {type(restored.data).__name__}")

    # Try running executor on the (unpickled) compilable
    try:
        artifact = backend.executor.build_artifact(compilable)
        print(f"build_artifact (no pickle): {artifact}")
    except Exception as e:
        print(f"build_artifact (no pickle) FAILED: {e!r}")

    # And on the pickled/unpickled one
    try:
        artifact = backend.executor.build_artifact(restored)
        print(f"build_artifact (after pickle roundtrip): {artifact}")
    except Exception as e:
        print(f"build_artifact (after pickle roundtrip) FAILED: {e!r}")


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    main()
