"""Smoke test for the `cached=True` backend trait (CachedStep wraps the executor).

Usage: BACKEND=gtfn|dace GT4PY_BUILD_JOBS_MODE=thread|process uv run python tmp/smoke_cached.py
"""

import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

import gt4py.next as gtx
from gt4py.next.otf import compiled_program

from stencil_cached_mod import IDim, run_inc_gtfn_cached, run_inc_dace_cached


def main() -> None:
    which = os.environ.get("BACKEND", "gtfn")
    program = {"gtfn": run_inc_gtfn_cached, "dace": run_inc_dace_cached}[which]
    print(
        f"[main] backend={which} pid={os.getpid()} "
        f"mode={os.environ.get('GT4PY_BUILD_JOBS_MODE', '<default>')} "
        f"pool={type(compiled_program._async_compilation_pool).__name__}"
    )

    a = gtx.as_field([IDim], np.arange(8, dtype=np.float64))
    out = gtx.as_field([IDim], np.zeros(8, dtype=np.float64))
    program(a, out, offset_provider={})
    # run a second time: in-memory CachedStep should short-circuit
    out2 = gtx.as_field([IDim], np.zeros(8, dtype=np.float64))
    program(a, out2, offset_provider={})
    assert np.allclose(out.asnumpy(), a.asnumpy() + 1.0)
    assert np.allclose(out2.asnumpy(), a.asnumpy() + 1.0)
    print("[main] OK")


if __name__ == "__main__":
    main()
