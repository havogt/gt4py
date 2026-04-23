"""End-to-end smoke test for the process-pool compilation path, DaCe CPU backend."""

import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

import gt4py.next as gtx
from gt4py.next.otf import compiled_program

from stencil_dace_mod import IDim, run_add_one_dace


def main() -> None:
    print(
        f"[main] pid={os.getpid()} "
        f"mode={os.environ.get('GT4PY_BUILD_JOBS_MODE', '<default>')} "
        f"pool={type(compiled_program._async_compilation_pool).__name__}"
    )
    a = gtx.as_field([IDim], np.arange(8, dtype=np.float64))
    out = gtx.as_field([IDim], np.zeros(8, dtype=np.float64))

    run_add_one_dace(a, out, offset_provider={})
    print(f"[main] out = {out.asnumpy().tolist()}")
    assert np.allclose(out.asnumpy(), a.asnumpy() + 1.0)
    print("[main] OK")


if __name__ == "__main__":
    main()
