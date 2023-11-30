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

import jax
import numpy as np
from jax import _src as jaxsrc, make_jaxpr, numpy as jnp

from gt4py import next as gtx
from gt4py.next import common
from gt4py.next.embedded import nd_array_field


# jaxsrc.core.ShapedArray.__slots__.append("custom_data")
# print(jaxsrc.core.ShapedArray.__slots__)

# class CustomShapedArray(jaxsrc.core.ShapedArray):
#     def __init__(self, )


def field_abstractify(x: nd_array_field.JaxArrayField):
    dtype = x.dtype
    metadata = {}
    metadata["domain"] = x.domain
    if isinstance(x, nd_array_field.JaxArrayConnectivityField):
        metadata["codomain"] = x.codomain
    res = jaxsrc.core.ShapedArray(x.shape, dtype, named_shape=metadata)
    return res


jaxsrc.api_util._shaped_abstractify_handlers[nd_array_field.JaxArrayField] = field_abstractify
jaxsrc.api_util._shaped_abstractify_handlers[
    nd_array_field.JaxArrayConnectivityField
] = field_abstractify

a = jnp.asarray([0, 1])
b = jnp.asarray([2, 3])


def fop(fun):
    def impl(*args):
        assert all(
            isinstance(arg, jaxsrc.interpreters.partial_eval.DynamicJaxprTracer) for arg in args
        )
        new_args = []
        for arg in args:
            metadata = arg.named_shape
            domain = metadata["domain"]
            codomain = metadata.get("codomain", None)
            # domain = field.domain
            # domain = arg.named_shape["domain"]
            arg.aval.named_shape = {}
            if codomain is not None:
                new_args.append(nd_array_field.JaxArrayConnectivityField(domain, arg, codomain))
            else:
                new_args.append(nd_array_field.JaxArrayField(domain, arg))
        res = fun(*new_args)
        return res.ndarray

    return impl


@fop
def foo(a, b):
    return a + b


# print(make_jaxpr(foo)(a,b))


I = gtx.Dimension("I")


fa = common.field(jnp.asarray([1, 2, 3]), domain=common.domain({I: 3}))
fb = common.field(jnp.asarray([1, 2, 3, 4]), domain=common.domain({I: 4}))
print(make_jaxpr(foo)(fa, fb))


jitted_foo = jax.jit(foo)
print(jitted_foo(fa, fb))

# @fop
# def bar(a, conn):
#     return a(conn)

conn = common.connectivity(jnp.asarray([2, 1, 0]), codomain=I, domain=common.domain({I: 3}))
# jitted_bar = jax.jit(bar)
# print(jitted_bar(fa,conn))

conn_ndarray = jnp.asarray(conn.ndarray)

# res = make_jaxpr(lambda fa: fa*np.asarray([2,1,0]))(fa.ndarray)
res = make_jaxpr(lambda fa, fb: fb - fa * jnp.add(conn_ndarray, 1))(fa.ndarray, fb.ndarray[:-1])
print(res)
print(type(res))
