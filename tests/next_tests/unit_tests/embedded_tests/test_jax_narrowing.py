# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest


pytestmark = pytest.mark.requires_jax


def _expensive(x):
    import jax.numpy as jnp

    y = x
    for i in range(8):
        y = jnp.sin(y) * jnp.cos(x + i) + jnp.exp(-y * y)
    return y


def _consumers(x, z):
    import jax.numpy as jnp

    y = _expensive(x)
    a = y[2:102] + z
    b = jnp.concatenate([y[4:54], y[54:104]]) * 2.0
    c = jnp.where(y[3:103] > 0.5, y[3:103], jnp.broadcast_to(z[0], (100, 80)))
    return a, b, c, y[3:103].sum(axis=0)


def _inputs():
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    return jnp.asarray(rng.random((108, 80))), jnp.asarray(rng.random((100, 80)))


def test_narrowed_function_matches_bitwise():
    import jax

    from gt4py.next.embedded.jax_narrowing import narrow

    x, z = _inputs()
    reference = jax.jit(_consumers)(x, z)
    narrowed = jax.jit(narrow(_consumers))(x, z)

    for r, n in zip(reference, narrowed):
        assert r.shape == n.shape
        np.testing.assert_array_equal(np.asarray(r), np.asarray(n))


def test_producer_is_computed_on_the_union_of_the_read_windows():
    import jax

    from gt4py.next.embedded.jax_narrowing import narrow

    def producer_with_two_readers(x):
        y = _expensive(x)
        return y[2:102] + 1.0, y[4:104] * 2.0

    x, _ = _inputs()
    closed = jax.make_jaxpr(narrow(producer_with_two_readers))(x)

    shapes = {tuple(v.aval.shape) for eqn in closed.jaxpr.eqns for v in eqn.outvars}
    assert (108, 80) not in shapes
    assert (102, 80) in shapes


def test_pytrees_and_python_scalars():
    import jax

    from gt4py.next.embedded.jax_narrowing import narrow

    def fun(fields, factor):
        return {"out": (fields["a"][1:-1] * factor + fields["b"][:-2])}

    fields = {"a": jax.numpy.arange(10.0), "b": jax.numpy.ones(10)}
    np.testing.assert_array_equal(
        np.asarray(jax.jit(narrow(fun))(fields, 3.0)["out"]),
        np.asarray(fun(fields, 3.0)["out"]),
    )


def test_operand_that_is_not_read_is_not_computed():
    import jax

    from gt4py.next.embedded.jax_narrowing import narrow

    def fun(x):
        stacked = jax.numpy.concatenate([_expensive(x[:4]), x[4:] * 2.0])
        return stacked[5:]

    x, _ = _inputs()
    closed = jax.make_jaxpr(narrow(fun))(x)

    assert not any(eqn.primitive.name == "sin" for eqn in closed.jaxpr.eqns)
    np.testing.assert_array_equal(np.asarray(jax.jit(narrow(fun))(x)), np.asarray(jax.jit(fun)(x)))
