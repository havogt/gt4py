from iterator.dispatcher import Dispatcher


__all__ = [
    "compose",
    "deref",
    "div",
    "domain",
    "greater",
    "if_",
    "is_none",
    "lift",
    "make_tuple",
    "minus",
    "mul",
    "named_range",
    "nth",
    "plus",
    "reduce",
    "scan",
    "shift",
]

builtin_dispatch = Dispatcher()


class BackendNotSelectedError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("Backend not selected")


@builtin_dispatch
def deref(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def shift(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def lift(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def reduce(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def scan(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def is_none(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def domain(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def named_range(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def compose(sten):
    raise BackendNotSelectedError()


@builtin_dispatch
def if_(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def minus(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def plus(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def mul(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def div(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def greater(*args):
    raise BackendNotSelectedError()


@builtin_dispatch
def make_tuple(*args):
    raise BackendNotSelectedError


@builtin_dispatch
def nth(*args):
    raise BackendNotSelectedError
