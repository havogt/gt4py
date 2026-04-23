"""Stencil for the GPU process-pool smoke test (gtfn + dace)."""

import gt4py.next as gtx
from gt4py.next.program_processors.runners import gtfn
from gt4py.next.program_processors.runners.dace.workflow import backend as dace_backend


IDim = gtx.Dimension("I")


@gtx.field_operator
def add_one_gpu(a: gtx.Field[[IDim], gtx.float64]) -> gtx.Field[[IDim], gtx.float64]:
    return a + 1.0


@gtx.program(backend=gtfn.run_gtfn_gpu)
def run_add_one_gtfn_gpu(
    a: gtx.Field[[IDim], gtx.float64], out: gtx.Field[[IDim], gtx.float64]
):
    add_one_gpu(a, out=out)


@gtx.program(backend=dace_backend.run_dace_gpu)
def run_add_one_dace_gpu(
    a: gtx.Field[[IDim], gtx.float64], out: gtx.Field[[IDim], gtx.float64]
):
    add_one_gpu(a, out=out)
