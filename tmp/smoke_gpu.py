"""End-to-end smoke test for the process-pool compilation path on GPU.

Usage: BACKEND=gtfn|dace GT4PY_BUILD_JOBS_MODE=thread|process uv run python tmp/smoke_gpu.py
"""

import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import cupy
import numpy as np

import gt4py.next as gtx
from gt4py.next import common
from gt4py.next.otf import compiled_program

from stencil_gpu_mod import IDim, run_add_one_gtfn_gpu, run_add_one_dace_gpu


def main() -> None:
    which = os.environ.get("BACKEND", "gtfn")
    program = {"gtfn": run_add_one_gtfn_gpu, "dace": run_add_one_dace_gpu}[which]
    print(
        f"[main] backend={which} pid={os.getpid()} "
        f"mode={os.environ.get('GT4PY_BUILD_JOBS_MODE', '<default>')} "
        f"pool={type(compiled_program._async_compilation_pool).__name__}"
    )

    # Allocate directly on GPU.
    host = np.arange(8, dtype=np.float64)
    a = gtx.as_field([IDim], cupy.asarray(host), allocator=program.backend)
    out = gtx.as_field([IDim], cupy.zeros(8, dtype=cupy.float64), allocator=program.backend)

    program(a, out, offset_provider={})
    cupy.cuda.runtime.deviceSynchronize()

    got = cupy.asnumpy(out.ndarray)
    print(f"[main] out = {got.tolist()}")
    assert np.allclose(got, host + 1.0)
    print("[main] OK")


if __name__ == "__main__":
    main()
