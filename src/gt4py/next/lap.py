from typing import Callable


class Iterator:
    ...


class Field:
    ...


def deref(it: Iterator) -> float:
    return 0.0


def minus(a: float, b: float) -> float:
    return 0.0


def fminus(a: Field, b: Field) -> Field:
    return Field()


def shift(tag: str, offset: int) -> Callable[[Iterator], Iterator]:
    def impl(it: Iterator) -> Iterator:
        return Iterator()

    return impl


def fshift(tag: str, offset: int) -> Callable[[Field], Field]:
    def impl(it: Field) -> Field:
        return Field()

    return impl


def lift(fun: Callable[[Iterator], float]) -> Callable[[Iterator], Iterator]:
    def impl(it: Iterator) -> Iterator:
        return Iterator()

    return impl


def fmap(fun: Callable[[Iterator], float]) -> Callable[[Field], Field]:
    def impl(inp: Field) -> Field:
        return Field()

    return impl


def as_fieldoperator(fun: Callable[[Iterator], float]) -> Callable[[Field], Field]:
    return fmap(fun)


I = "I"
J = "J"

# ====


def diff_local(inp: Iterator) -> float:
    return minus(deref(shift(I, 1)(inp)), deref(shift(I, -1)(inp)))


def diff_local_diff_local(inp: Iterator) -> float:
    return diff_local(lift(diff_local)(inp))


def diff_field(inp: Field) -> Field:
    return fminus(fshift(I, 1)(inp), fshift(I, -1)(inp))


def diff_field_diff_field(inp: Field) -> Field:
    return diff_field(diff_field(inp))


def diff_field_diff_local(inp: Field) -> Field:
    return diff_field(fmap(diff_local)(inp))


# ====


@as_fieldoperator
def fmapped_diff_local(inp: Iterator) -> float:
    return minus(deref(shift(I, 1)(inp)), deref(shift(I, -1)(inp)))
