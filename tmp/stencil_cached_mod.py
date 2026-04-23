"""Stencil exercising cached backends (CachedStep wraps the executor)."""

import gt4py.next as gtx
from gt4py.next.program_processors.runners import gtfn
from gt4py.next.program_processors.runners.dace.workflow import backend as dace_backend


IDim = gtx.Dimension("I")


@gtx.field_operator
def inc_cached(a: gtx.Field[[IDim], gtx.float64]) -> gtx.Field[[IDim], gtx.float64]:
    return a + 1.0


@gtx.program(backend=gtfn.run_gtfn_cached)
def run_inc_gtfn_cached(
    a: gtx.Field[[IDim], gtx.float64], out: gtx.Field[[IDim], gtx.float64]
):
    inc_cached(a, out=out)


@gtx.program(backend=dace_backend.run_dace_cpu_cached)
def run_inc_dace_cached(
    a: gtx.Field[[IDim], gtx.float64], out: gtx.Field[[IDim], gtx.float64]
):
    inc_cached(a, out=out)
