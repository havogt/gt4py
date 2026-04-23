"""Stencils used by the process-pool smoke test, kept in a real module so their raw
definition functions are picklable-by-qualname (the main script's ``__main__`` is re-executed
under a different ``__name__`` in spawned workers, which breaks pickling of functions defined
there).
"""

import gt4py.next as gtx
from gt4py.next.program_processors.runners import gtfn


IDim = gtx.Dimension("I")


@gtx.field_operator
def add_one(a: gtx.Field[[IDim], gtx.float64]) -> gtx.Field[[IDim], gtx.float64]:
    return a + 1.0


@gtx.program(backend=gtfn.run_gtfn)
def run_add_one(a: gtx.Field[[IDim], gtx.float64], out: gtx.Field[[IDim], gtx.float64]):
    add_one(a, out=out)
