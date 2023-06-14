from __future__ import annotations

import itertools
import operator
from dataclasses import dataclass

import numpy as np


# simplifications:
# - no named dimension (implicit n-th dimension key from domain maps to n-th dimension of buffer)
# - domain cannot contain negative values, buffers are allocated from 0


def _domain2slice(domain):
    return tuple(slice(None, v[1]) for v in domain.values())


class Field:
    def __init__(self, domain, init=None):
        self.domain = domain
        _shape = tuple(b for a, b in domain.values())
        if init is None:
            self._buffer = np.zeros(shape=_shape, dtype=float)
        elif isinstance(init, np.ndarray):
            self._buffer = init
        elif isinstance(init, float):
            self._buffer = np.ones(shape=_shape, dtype=float) * init
        elif callable(init):
            self._buffer = np.fromfunction(init, shape=_shape, dtype=float)
        else:
            assert False

    def __str__(self):
        slce = tuple(slice(v[0], v[1]) for v in self.domain.values())
        return f"<{self.domain},\n{self._buffer[slce]}>"


@dataclass(frozen=True)
class Iterator:
    _field: Field
    _pos: dict[str, tuple[int, int]]


def _field_binary(op):
    def impl(a: Field, b: Field) -> Field:
        assert a.domain.keys() == b.domain.keys()
        domain = {
            k: (max(a[0], b[0]), min(a[1], b[1]))
            for k, (a, b) in dict(
                zip(a.domain.keys(), zip(a.domain.values(), b.domain.values()))
            ).items()
        }
        slce = tuple(slice(None, v[1]) for v in domain.values())
        return Field(domain, op(a._buffer[slce], b._buffer[slce]))

    return impl


def _local_binary(op):
    def impl(a, b):
        assert isinstance(a, float)
        assert isinstance(b, float)
        return op(a, b)

    return impl


def deref(it: Iterator):
    return it._field._buffer[tuple(it._pos.values())]


def shift(tag: str, value: int):
    def impl(it: Iterator):
        pos = it._pos.copy()
        pos[tag] += value
        return Iterator(it._field, pos)

    return impl


def fencil_map(stencil, *, out, domain):
    def impl(*args):
        for pos in itertools.product(*[range(b, e) for b, e in domain.values()]):
            print(pos)
            out._buffer[pos] = stencil(
                *[Iterator(arg, dict(zip(domain.keys(), pos))) for arg in args]
            )

    return impl


add = _local_binary(operator.add)
mul = _local_binary(operator.mul)

fadd = _field_binary(np.add)
fmul = _field_binary(np.multiply)


# =============

I = "I"
J = "J"

f = Field({I: (1, 3), J: (0, 3)}, lambda x, y: x + y)
print(f)
g = Field({I: (1, 4), J: (1, 3)}, lambda x, y: x**2 * y**2)
print(g)
print(fadd(f, g))

inp = Field({I: (0, 4), J: (0, 5)}, lambda x, y: x**2 + y**2)
out = Field({I: (1, 3), J: (1, 4)})


def lap_local(inp: Iterator[[I, J], float]) -> float:
    return add(
        add(
            add(add(mul(-4.0, deref(inp)), deref(shift(I, -1)(inp))), deref(shift(I, 1)(inp))),
            deref(shift(J, -1)(inp)),
        ),
        deref(shift(J, 1)(inp)),
    )
    # return (
    #     -4.0 * inp() + inp(I - 1) + inp(I + 1) + inp(J - 1) + inp(J + 1)
    # )  # beautified


fencil_map(lap_local, out=out, domain=out.domain)(inp)
print(inp)
print(out)


def fencil(inp: Field[[I, J], float], out: Field[[I, J]]) -> None:
    fencil_map(lap_local, out=out, domain=out.domain)(
        inp
    )  # not sure if `fmap` should be the same as in the next example


def lap_local_as_field_operator(inp: Field[[I, J], float]) -> Field[[I, J], float]:
    return fmap(lap_local)(inp)  # of beautified:
    # return lap_local[...](inp)
