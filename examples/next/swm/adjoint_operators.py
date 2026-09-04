# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Halo exchange and its adjoint, written in the GT4Py field view.

Support module for `nb01_halo_exchange_adjoint.ipynb`. The two deliberately
broken adjoints are teaching material, not dead code -- the notebook runs the
dot-product test against them to show what each mistake costs.
"""

from gt4py import next as gtx
from gt4py.next.experimental import concat_where

from operators import I, J, IJField, dtype, make_periodic


def interior_domain(M: int, N: int) -> gtx.Domain:
    return gtx.domain({I: (0, M), J: (0, N)})


def halo_domain(M: int, N: int) -> gtx.Domain:
    return gtx.domain({I: (-1, M + 1), J: (-1, N + 1)})


def halo_exchange(f: IJField, M: int, N: int) -> IJField:
    """Overwrite the halo of `f` from its interior, periodically.

    Maps halo domain to halo domain, which is what a distributed halo update
    does: whatever was in the halo before is discarded.
    """
    return make_periodic(f[interior_domain(M, N)], M, N)


@gtx.field_operator
def halo_exchange_adjoint(g: IJField, M: gtx.int32, N: gtx.int32) -> IJField:
    """Transpose of `halo_exchange`.

    Stages run in reverse order, each copy becomes an accumulation into the
    owner, and each halo line is zeroed once its contribution has been
    collected.
    """
    zero = 0.0 * g
    # reverse of  concat_where(J == N, f(J - N), f)
    g = concat_where(J == 0, g + g(J + N), g)
    g = concat_where(J == N, zero, g)
    # reverse of  concat_where(J == -1, f(J + N), f)
    g = concat_where(J == N - 1, g + g(J - N), g)
    g = concat_where(J == -1, zero, g)
    # reverse of  concat_where(I == M, f(I - M), f)
    g = concat_where(I == 0, g + g(I + M), g)
    g = concat_where(I == M, zero, g)
    # reverse of  concat_where(I == -1, f(I + M), f)
    g = concat_where(I == M - 1, g + g(I - M), g)
    g = concat_where(I == -1, zero, g)
    return g


@gtx.field_operator
def halo_exchange_adjoint_no_accumulate(g: IJField, M: gtx.int32, N: gtx.int32) -> IJField:
    """Broken: assigns the halo cotangent onto the owner instead of adding it."""
    zero = 0.0 * g
    g = concat_where(J == 0, g(J + N), g)
    g = concat_where(J == N, zero, g)
    g = concat_where(J == N - 1, g(J - N), g)
    g = concat_where(J == -1, zero, g)
    g = concat_where(I == 0, g(I + M), g)
    g = concat_where(I == M, zero, g)
    g = concat_where(I == M - 1, g(I - M), g)
    g = concat_where(I == -1, zero, g)
    return g


@gtx.field_operator
def halo_exchange_adjoint_no_zeroing(g: IJField, M: gtx.int32, N: gtx.int32) -> IJField:
    """Broken: accumulates correctly but leaves the halo cotangent in place."""
    g = concat_where(J == 0, g + g(J + N), g)
    g = concat_where(J == N - 1, g + g(J - N), g)
    g = concat_where(I == 0, g + g(I + M), g)
    g = concat_where(I == M - 1, g + g(I - M), g)
    return g


@gtx.field_operator
def periodic_1d(
    f: gtx.Field[gtx.Dims[I], dtype], M: gtx.int32
) -> gtx.Field[gtx.Dims[I], dtype]:
    """One-dimensional periodic halo fill, interior domain to halo domain."""
    f = concat_where(I == -1, f(I + M), f)
    f = concat_where(I == M, f(I - M), f)
    return f
