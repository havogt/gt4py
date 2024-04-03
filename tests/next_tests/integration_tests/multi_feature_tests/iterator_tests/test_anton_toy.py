# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

import gt4py.next as gtx
from gt4py.next import common
from gt4py.next.iterator import embedded
from gt4py.next.iterator.builtins import cartesian_domain, deref, lift, named_range, shift
from gt4py.next.iterator.runtime import closure, fendef, fundef, offset
from gt4py.next.program_processors.runners import gtfn

from next_tests.integration_tests import cases
from next_tests.integration_tests.cases import cartesian_case
from next_tests.integration_tests.feature_tests.ffront_tests.ffront_test_utils import (
    exec_alloc_descriptor,
)
from next_tests.unit_tests.conftest import lift_mode, program_processor, run_processor


@fundef
def ldif(d):
    return lambda inp: deref(shift(d, -1)(inp)) - deref(inp)


@fundef
def rdif(d):
    return lambda inp: ldif(d)(shift(d, 1)(inp))


@fundef
def dif2(d):
    return lambda inp: ldif(d)(lift(rdif(d))(inp))


i = offset("i")
j = offset("j")


@fundef
def lap(inp):
    return dif2(i)(inp) + dif2(j)(inp)


IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")
KDim = gtx.Dimension("KDim")


@fendef(offset_provider={"i": IDim, "j": JDim})
def fencil(x, y, z, out, inp):
    closure(
        cartesian_domain(named_range(IDim, 0, x), named_range(JDim, 0, y), named_range(KDim, 0, z)),
        lap,
        out,
        [inp],
    )


def _apply_stencil(fun, domain):
    def impl(*args, **kwargs):
        new_domain = None
        if common.is_domain_like(domain):
            new_domain = domain
        else:
            assert callable(domain)
            new_domain = domain(*args, **kwargs)
        out = gtx.empty(gtx.domain(new_domain))  # TODO dtype, tuple structure etc
        closure(
            cartesian_domain(
                *(named_range(d, start, stop) for d, (start, stop) in new_domain.items())
            ),
            fun,
            out,
            list(args),  # TODO kwargs
        )
        return out

    return impl


def stencil(fun=None, *, domain=None):
    if fun is None:

        def impl(inner_fun):
            return _apply_stencil(inner_fun, domain)

        return impl
    else:
        return _apply_stencil(fun, domain)


@stencil(domain={IDim: (1, 9), JDim: (1, 9), KDim: (1, 9)})
def lap_mapped(inp):
    return dif2(i)(inp) + dif2(j)(inp)


@gtx.field_operator
def lap_fop(inp: gtx.Field[[IDim, JDim, KDim], float]):
    return lap_mapped(inp)
    # return stencil(lap, domain={IDim: (1, 9), JDim: (1, 9), KDim: (1, 9)})(
    #     inp
    # )  # TODO hard coded sizes


@gtx.program
def prog(
    x: int,
    y: int,
    z: int,
    out: gtx.Field[[IDim, JDim, KDim], float],
    inp: gtx.Field[[IDim, JDim, KDim], float],
):
    lap_fop(inp, out=out, domain={IDim: (1, x - 1), JDim: (1, y - 1), KDim: (1, z - 1)})


def naive_lap(inp):
    shape = [inp.shape[0] - 2, inp.shape[1] - 2, inp.shape[2]]
    out = np.zeros(shape)
    inp_data = inp.asnumpy()
    for i in range(1, shape[0] + 1):
        for j in range(1, shape[1] + 1):
            for k in range(0, shape[2]):
                out[i - 1, j - 1, k] = -4 * inp_data[i, j, k] + (
                    inp_data[i + 1, j, k]
                    + inp_data[i - 1, j, k]
                    + inp_data[i, j + 1, k]
                    + inp_data[i, j - 1, k]
                )
    return out


def test_lap():
    in_field = gtx.as_field((IDim, JDim, KDim), np.resize(np.arange(1000), (10, 10, 10)))
    out_field = gtx.empty(gtx.domain({IDim: 10, JDim: 10, KDim: 10}))

    prog(10, 10, 10, out_field, in_field, offset_provider={"i": IDim, "j": JDim})

    # cases.verify(
    #     cartesian_case,
    #     prog,
    #     10,
    #     10,
    #     10,
    #     out_field,
    #     in_field,
    #     inout=out_field[1:-1, 1:-1],
    #     ref=naive_lap(in_field),
    # )


# @pytest.mark.uses_origin
# def test_anton_toy(program_processor, lift_mode):
#     program_processor, validate = program_processor

#     if program_processor in [
#         gtfn.run_gtfn.executor,
#         gtfn.run_gtfn_imperative.executor,
#         gtfn.run_gtfn_with_temporaries.executor,
#     ]:
#         from gt4py.next.iterator import transforms

#         if lift_mode != transforms.LiftMode.FORCE_INLINE:
#             pytest.xfail("TODO: issue with temporaries that crashes the application")

#     shape = [5, 7, 9]
#     rng = np.random.default_rng()
#     inp = gtx.as_field(
#         [IDim, JDim, KDim],
#         rng.normal(size=(shape[0] + 2, shape[1] + 2, shape[2])),
#         origin={IDim: 1, JDim: 1, KDim: 0},
#     )
#     out = gtx.as_field([IDim, JDim, KDim], np.zeros(shape))
#     ref = naive_lap(inp)

#     run_processor(
#         fencil, program_processor, shape[0], shape[1], shape[2], out, inp, lift_mode=lift_mode
#     )

#     if validate:
#         assert np.allclose(out.asnumpy(), ref)
