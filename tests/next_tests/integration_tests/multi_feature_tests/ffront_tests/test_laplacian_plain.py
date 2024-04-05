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

import dataclasses
from typing import Any

import numpy as np
import xarray

import gt4py.next as gtx
from gt4py.next import backend
from gt4py.next.embedded import nd_array_field
from gt4py.next.ffront import stages as ffront_stages


IDim = gtx.Dimension("IDim")
JDim = gtx.Dimension("JDim")

Ioff = gtx.FieldOffset("Ioff", source=IDim, target=(IDim,))
Joff = gtx.FieldOffset("Joff", source=JDim, target=(JDim,))


@gtx.field_operator
def lap(
    in_field: gtx.Field[[IDim, JDim], "float"],
) -> gtx.Field[[IDim, JDim], "float"]:
    res = (
        -4.0 * in_field
        + in_field(Ioff[1])
        + in_field(Joff[1])
        + in_field(Ioff[-1])
        + in_field(Joff[-1])
    )
    return res


@gtx.field_operator
def laplap(
    in_field: gtx.Field[[IDim, JDim], "float"],
) -> gtx.Field[[IDim, JDim], "float"]:
    return lap(lap(in_field))


@gtx.program
def laplap_program(
    in_field: gtx.Field[[IDim, JDim], "float"], out_field: gtx.Field[[IDim, JDim], "float"]
):
    laplap(in_field, out=out_field[2:-2, 2:-2])


def lap_ref(inp):
    """Compute the laplacian using numpy"""
    return -4.0 * inp[1:-1, 1:-1] + inp[:-2, 1:-1] + inp[2:, 1:-1] + inp[1:-1, :-2] + inp[1:-1, 2:]


def dataarray_to_field(da):
    return nd_array_field.NumPyArrayField.from_array(
        da.values, domain=gtx.domain({gtx.Dimension(d): da.sizes[d] for d in da.dims})
    )


@dataclasses.dataclass
class DataArrayBackend:
    backend: backend.Backend

    def __call__(
        self, program: ffront_stages.ProgramDefinition, *args: tuple[Any], **kwargs: dict[str, Any]
    ) -> None:
        transformed_args = [
            dataarray_to_field(arg) if isinstance(arg, xarray.DataArray) else arg for arg in args
        ]
        transformed_kwargs = {
            k: (dataarray_to_field(v) if isinstance(v, xarray.DataArray) else v)
            for k, v in kwargs.items()
        }
        self.backend(program, *transformed_args, **transformed_kwargs)


def test_ffront_lap():
    in_field = xarray.DataArray(
        np.fromfunction(lambda i, j: i * i + j * j, (10, 10)),
        dims=["IDim", "JDim"],
    )
    out_field = xarray.DataArray(np.zeros((10, 10)), dims=["IDim", "JDim"])

    laplap_program.with_backend(DataArrayBackend(gtx.itir_python))(
        in_field, out_field, offset_provider={"Ioff": IDim, "Joff": JDim}
    )

    np.testing.assert_allclose(out_field[2:-2, 2:-2], lap_ref(lap_ref(in_field)))


test_ffront_lap()

# - what to do with dimensions?
# = how to map coords?
