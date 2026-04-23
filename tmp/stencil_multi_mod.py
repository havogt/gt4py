"""Multi-variant stencil for the parallel-compilation smoke test."""

import gt4py.next as gtx
from gt4py.next.program_processors.runners import gtfn


IDim = gtx.Dimension("I")


@gtx.field_operator
def scaled_add(
    a: gtx.Field[[IDim], gtx.float64], scale: gtx.float64
) -> gtx.Field[[IDim], gtx.float64]:
    return a * scale


@gtx.program(backend=gtfn.run_gtfn)
def run_scaled_add(
    a: gtx.Field[[IDim], gtx.float64],
    out: gtx.Field[[IDim], gtx.float64],
    scale: gtx.float64,
):
    scaled_add(a, scale, out=out)
