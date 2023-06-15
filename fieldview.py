from __future__ import annotations

import itertools
import operator
from dataclasses import dataclass
import functools

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

    def __array__(self):
        slce = tuple(slice(v[0], v[1]) for v in self.domain.values())
        return self._buffer[slce]

    def __str__(self):
        return f"<{self.domain},\n{np.asarray(self)}>"


class Iterator:
    ...


@dataclass(frozen=True)
class FieldIterator(Iterator):
    _field: Field
    _pos: dict[str, tuple[int, int]]

    def _deref(self):
        return self._field._buffer[tuple(self._pos.values())]

    def _shift(self, tag, value):
        pos = self._pos.copy()
        pos[tag] += value
        return FieldIterator(self._field, pos)


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


@dataclass(frozen=True)
class LocalOperatorObject:
    extent: dict[str, Tuple[int, int]]
    _fun: Callable

    def __call__(self, *args):
        return self._fun(*args)


def local_operator(*, extent):  # aka stencil
    def impl(fun):
        return LocalOperatorObject(extent, fun)

    return impl


def _local_binary(op):
    def impl(a, b):
        assert isinstance(a, float)
        assert isinstance(b, float)
        return op(a, b)

    return impl


def deref(it: Iterator):
    return it._deref()


def shift(tag: str, value: int):
    def impl(it: Iterator):
        return it._shift(tag, value)

    return impl


def fshift(tag: str, value: int):
    def impl(f: Field):
        domain = f.domain.copy()
        domain[tag] = tuple(v + value for v in domain[tag])
        buffer = f._buffer
        if value < 0:
            buffer = buffer[tuple(slice(1 if dim == tag else None, None) for dim in domain.keys())]
        if value > 0:
            buffer = np.pad(buffer, [(1 if dim == tag else 0, 0) for dim in domain.keys()])

        return Field(domain, buffer)

    return impl


def fencil_map(stencil, *, out, domain):
    def impl(*args):
        for pos in itertools.product(*[range(b, e) for b, e in domain.values()]):
            # print(pos)
            out._buffer[pos] = stencil(
                *[FieldIterator(arg, dict(zip(domain.keys(), pos))) for arg in args]
            )

    return impl


def _shrink_domain_with_extent(domain, stencil_extent):  # -> domain
    assert domain.keys() == stencil_extent.keys()
    return dict(
        zip(
            domain.keys(),
            ((d[0] - e[0], d[1] - e[1]) for d, e in zip(domain.values(), stencil_extent.values())),
        )
    )


def _intersect_domains(*domains):
    assert all(domains[0].keys() == d.keys() for d in domains)
    return dict(
        zip(
            domains[0].keys(),
            (
                functools.reduce(lambda x, y: (max(x[0], y[0]), min(x[1], y[1])), elem)
                for elem in zip(*(d.values() for d in domains))
            ),
        )
    )


def test_intersect_domains():
    result = _intersect_domains({"I": (0, 5), "J": (4, 5)}, {"I": (1, 6), "J": (3, 6)})
    expected = {"I": (1, 5), "J": (4, 5)}
    assert result == expected


def fmap(stencil):
    def impl(*args):
        domain = _intersect_domains(
            *(_shrink_domain_with_extent(f.domain, e) for f, e in zip(args, stencil.extent))
        )
        out = Field(domain)
        for pos in itertools.product(*[range(b, e) for b, e in domain.values()]):
            out._buffer[pos] = stencil(
                *[FieldIterator(arg, dict(zip(domain.keys(), pos))) for arg in args]
            )
        return out

    return impl


def apply(field: Field, *, out: Field, domain: dict[str, tuple[int, int]]):
    slce = tuple(slice(d[0], d[1]) for d in domain.values())
    out._buffer[slce] = field._buffer[slce]  # doesn't change the domain of out


def literal(value):
    return value


add = _local_binary(operator.add)
mul = _local_binary(operator.mul)

fadd = _field_binary(np.add)
fmul = _field_binary(np.multiply)


def field_all_close(a: Field, b: Field):
    return a.domain == b.domain and np.allclose(np.asarray(a), np.asarray(b))


def test_fshift():
    inp = Field({"I": (1, 3), "J": (0, 3)}, lambda x, y: x + y)

    # I - 1
    result = fshift("I", -1)(inp)
    expected = Field({"I": (0, 2), "J": (0, 3)}, np.asarray(inp))
    assert field_all_close(result, expected)

    # I + 1
    result = fshift("I", 1)(inp)
    expected = Field({"I": (2, 4), "J": (0, 3)}, np.pad(np.asarray(inp), ((2, 0), (0, 0))))
    assert field_all_close(result, expected)


def test_fadd():
    f = Field({I: (1, 3), J: (0, 3)}, lambda x, y: x + y)
    g = Field({I: (1, 4), J: (1, 3)}, lambda x, y: x**2 * y**2)
    result = fadd(f, g)
    expected = Field(
        {"I": (1, 3), "J": (1, 3)},
        np.fromfunction(lambda x, y: x + y + x**2 * y**2, shape=(3, 3), dtype=float),
    )

    assert field_all_close(result, expected)


# =============

I = "I"
J = "J"


inp = Field({I: (0, 5), J: (0, 6)}, lambda x, y: x**4 + y**4)
print(inp)
out_local = Field({I: (1, 4), J: (1, 5)})


@local_operator(
    extent=[{I: (-1, 1), J: (-1, 1)}]
)  # decorator only required to be able to provide extent (could be computed)
def lap_local(inp: Iterator[[I, J], float]) -> float:
    return add(
        add(
            add(
                add(mul(literal(-4.0), deref(inp)), deref(shift(I, -1)(inp))),
                deref(shift(I, 1)(inp)),
            ),
            deref(shift(J, -1)(inp)),
        ),
        deref(shift(J, 1)(inp)),
    )
    # return (
    #     -4.0 * inp() + inp(I - 1) + inp(I + 1) + inp(J - 1) + inp(J + 1)
    # )  # beautified


def lap_field(inp: Field[[I, J], float]) -> Field[[I, J], float]:
    minus_four = Field(inp.domain, init=-4.0)
    return fadd(
        fadd(
            fadd(fadd(fmul(minus_four, inp), fshift(I, -1)(inp)), fshift(I, 1)(inp)),
            fshift(J, -1)(inp),
        ),
        fshift(J, 1)(inp),
    )


fencil_map(lap_local, out=out_local, domain=out_local.domain)(inp)
print(out_local)
res_field = lap_field(inp)

assert np.allclose(np.asarray(out_local), np.asarray(res_field))


def fencil(inp: Field[[I, J], float], out: Field[[I, J]]) -> None:
    fencil_map(lap_local, out=out, domain=out.domain)(
        inp
    )  # not sure if `fmap` should be the same as in the next example


def lap_local_as_field_operator(inp: Field[[I, J], float]) -> Field[[I, J], float]:
    return fmap(lap_local)(inp)  # of beautified:
    # return lap_local[...](inp)


res_lap_local_as_field_op = fmap(lap_local)(inp)  # domain should not be here
assert np.allclose(out_local, res_lap_local_as_field_op)


def lap_as_fieldview_program(inp, out):
    apply(lap_field(inp), out=out, domain=out.domain)


out_field = Field({I: (1, 4), J: (1, 5)})
lap_as_fieldview_program(inp, out_field)
print(out_field)
assert np.allclose(out_local, out_field)


@dataclass(frozen=True)
class LiftedStencilIterator:
    _fun: Callable
    _its: Iterable[Iterator]

    def _deref(self):
        return self._fun(*self._its)

    def _shift(self, tag, value):
        return LiftedStencilIterator(self._fun, (shift(tag, value)(arg) for arg in self._its))


def lift(fun):
    def impl(*args):
        return LiftedStencilIterator(fun, args)

    return impl


@local_operator(extent=[{I: (-2, 2), J: (-2, 2)}])
def lap_lap_local(inp: Iterator) -> float:
    return lap_local(lift(lap_local)(inp))


lap_lap_local_res = fmap(lap_lap_local)(inp)
lap_lap_field_res = lap_field(lap_field(inp))

assert field_all_close(lap_lap_local_res, lap_lap_field_res)

# TODO scan wip


def partial_sum_fun(state: float, inp: Iterator[[I], float]):
    return state + deref(inp)


K = "K"


def scan(fun, init):
    def impl(*args):
        ...


def partial_sum(inp: Iterator[[I, K], float]) -> Field[[K], float]:
    return scan(partial_sum_fun, 0.0)(inp)
