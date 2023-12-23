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
from jax import _src as jaxsrc, numpy as jnp
from jax.tree_util import tree_structure

from gt4py import next as gtx
from gt4py.next import common
from gt4py.next.embedded import nd_array_field


def show_example(structured):
    flat, tree = jax.tree_flatten(structured)
    unflattened = jax.tree_unflatten(tree, flat)
    print(f"{structured=}\n  {flat=}\n  {tree=}\n  {unflattened=}")


a = jnp.asarray([0, 1])
b = jnp.asarray([2, 3])


show_example([a, b])


I = gtx.Dimension("I")


fa = common.field(jnp.asarray([1.0, 2.0, 3.0]), domain=common.domain({I: 3}))
fb = common.field(jnp.asarray([1.0, 2.0, 3.0, 4.0]), domain=common.domain({I: 4}))


def _flatten(v: nd_array_field.JaxArrayField):
    return (v.ndarray,), v.domain


def _unflatten(aux_data, children):
    return nd_array_field.JaxArrayField(aux_data, children[0])


jax.tree_util.register_pytree_node(nd_array_field.JaxArrayField, _flatten, _unflatten)

show_example(fa)


# @gtx.field_operator
def foo(a: gtx.Field[[I], float], b: gtx.Field[[I], float]) -> gtx.Field[[I], float]:
    return a + b(common.CartesianConnectivity(I, -1))


print(jax.make_jaxpr(foo)(fa, fb))


res = jax.jit(foo)(fa, fb)

print(res)
print(res.ndarray)


# def bar(a):
#     return 2*a

# res = jax.grad(bar)(fa)
# print(res)
# print(res.ndarray)
