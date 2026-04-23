"""Stencil for the DaCe CPU process-pool smoke test."""

import gt4py.next as gtx
from gt4py.next.program_processors.runners.dace.workflow import backend as dace_backend


IDim = gtx.Dimension("I")


@gtx.field_operator
def add_one_dace(a: gtx.Field[[IDim], gtx.float64]) -> gtx.Field[[IDim], gtx.float64]:
    return a + 1.0


@gtx.program(backend=dace_backend.run_dace_cpu)
def run_add_one_dace(a: gtx.Field[[IDim], gtx.float64], out: gtx.Field[[IDim], gtx.float64]):
    add_one_dace(a, out=out)
