import numpy as np

from functional.common import Dimension
from functional.iterator.builtins import *
from functional.iterator.embedded import np_as_located_field
from functional.iterator.runtime import fundef


IDim = Dimension("IDim")
JDim = Dimension("JDim")
KDim = Dimension("KDim")


def vadd2(a, b):
    return make_tuple(tuple_get(0, a) + tuple_get(0, b), tuple_get(1, a) + tuple_get(1, b))


def vadd(n):
    def impl(a, b):
        res = []
        for i in range(n):
            res.append(tuple_get(i, a) + tuple_get(i, b))
        return make_tuple(*res)

    return impl


@fundef
def vadd_stencil(a, b):
    a_deref = deref(a)
    b_deref = deref(b)
    # return vadd2(a_deref, b_deref)
    return vadd(2)(a_deref, b_deref)


def test_vector_op(backend):
    backend, validate = backend

    shape = [5, 7, 9]
    rng = np.random.default_rng()

    inp = rng.normal(size=(shape[0], shape[1], shape[2], 2))
    expected = 2.0 * inp

    inp = np_as_located_field(IDim, JDim, KDim, None)(inp)
    out = np_as_located_field(IDim, JDim, KDim, None)(np.zeros_like(inp))

    dom = {
        IDim: range(0, shape[0]),
        JDim: range(0, shape[1]),
        KDim: range(0, shape[2]),
    }
    vadd_stencil[dom](inp, inp, out=out, offset_provider={}, backend=backend)
    if validate:
        assert np.allclose(out, expected)
