"""Precompile multiple static-arg variants to exercise parallel compilation and timing."""
import os
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

import gt4py.next as gtx
from gt4py.next.otf import arguments as otf_args
from gt4py.next.otf import compiled_program

from stencil_multi_mod import IDim, run_scaled_add  # noqa: F401


def main() -> None:
    pool_type = type(compiled_program._async_compilation_pool).__name__
    n_variants = 4
    print(
        f"[main] pid={os.getpid()} "
        f"mode={os.environ.get('GT4PY_BUILD_JOBS_MODE', '<default>')} "
        f"jobs={os.environ.get('GT4PY_BUILD_JOBS', '<default>')} "
        f"pool={pool_type} variants={n_variants}"
    )

    run_scaled_add._compiled_programs.argument_descriptor_mapping = {  # type: ignore[attr-defined]
        otf_args.StaticArg: ["scale"],
    }

    t0 = time.perf_counter()
    run_scaled_add.compile(
        offset_provider={},
        scale=[float(i + 1) for i in range(n_variants)],
    )
    compiled_program.wait_for_compilation()
    t1 = time.perf_counter()
    print(f"[main] parallel compile of {n_variants} variants: {t1 - t0:.2f}s")

    a = gtx.as_field([IDim], np.arange(8, dtype=np.float64))
    out = gtx.as_field([IDim], np.zeros(8, dtype=np.float64))
    for s in [1.0, 2.0, 3.0, 4.0]:
        run_scaled_add(a, out, scale=s, offset_provider={}, enable_jit=False)
        expected = a.asnumpy() * s
        assert np.allclose(out.asnumpy(), expected), (s, out.asnumpy(), expected)
    print("[main] all variants correct")


if __name__ == "__main__":
    main()
