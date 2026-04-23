"""GPU + cached process-pool smoke (matches user's run_dace_gpu_cached_opt failure)."""

import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cupy
import numpy as np

import gt4py.next as gtx
from gt4py.next.otf import compiled_program

from stencil_gpu_cached_mod import (
    IDim,
    run_inc_gtfn_gpu_cached,
    run_inc_dace_gpu_cached_opt,
)


def main() -> None:
    which = os.environ.get("BACKEND", "dace")
    program = {
        "gtfn": run_inc_gtfn_gpu_cached,
        "dace": run_inc_dace_gpu_cached_opt,
    }[which]
    print(
        f"[main] backend={which} pid={os.getpid()} "
        f"mode={os.environ.get('GT4PY_BUILD_JOBS_MODE', '<default>')} "
        f"pool={type(compiled_program._async_compilation_pool).__name__}"
    )

    host = np.arange(8, dtype=np.float64)
    a = gtx.as_field([IDim], cupy.asarray(host), allocator=program.backend)
    out = gtx.as_field([IDim], cupy.zeros(8, dtype=cupy.float64), allocator=program.backend)

    program(a, out, offset_provider={})
    cupy.cuda.runtime.deviceSynchronize()
    got = cupy.asnumpy(out.ndarray)
    assert np.allclose(got, host + 1.0), got
    print("[main] OK")


if __name__ == "__main__":
    main()
