"""Build a Backend on the fly via the factory and use it with the process pool.

The backend is NEVER assigned to a module global anywhere — this is the case the
`pkgutil.resolve_name` approach would have failed at. cloudpickle-based handoff
handles it.
"""

import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

import gt4py.next as gtx
from gt4py.next import config
from gt4py.next.otf import compiled_program
from gt4py.next.program_processors.runners import gtfn as gtfn_mod


# Build a non-default backend via the factory and keep it local (not a module global).
def main() -> None:
    print(
        f"[main] pid={os.getpid()} "
        f"mode={os.environ.get('GT4PY_BUILD_JOBS_MODE', '<default>')} "
        f"pool={type(compiled_program._async_compilation_pool).__name__}"
    )

    custom_backend = gtfn_mod.GTFNBackendFactory(
        name_postfix="_factory_local",
    )
    print(f"[main] backend name = {custom_backend.name}")

    IDim = gtx.Dimension("I")

    @gtx.field_operator
    def inc(a: gtx.Field[[IDim], gtx.float64]) -> gtx.Field[[IDim], gtx.float64]:
        return a + 1.0

    @gtx.program(backend=custom_backend)
    def run_inc(a: gtx.Field[[IDim], gtx.float64], out: gtx.Field[[IDim], gtx.float64]):
        inc(a, out=out)

    a = gtx.as_field([IDim], np.arange(8, dtype=np.float64))
    out = gtx.as_field([IDim], np.zeros(8, dtype=np.float64))
    run_inc(a, out, offset_provider={})
    assert np.allclose(out.asnumpy(), a.asnumpy() + 1.0)
    print("[main] OK")


if __name__ == "__main__":
    main()
