from dataclasses import dataclass
from typing import Optional, Tuple, Union

from functional.iterator.builtins import BackendNotSelectedError, builtin_dispatch, lift


__all__ = ["offset", "fundef", "fendef", "closure", "CartesianAxis"]


@dataclass
class Offset:
    value: Optional[Union[int, str]] = None
    length: Optional[int] = None

    def __len__(self):
        return self.length

    def __hash__(self) -> int:
        return hash(self.value)

    def __add__(self, offset: int) -> Tuple["Offset", int]:
        return (self, offset)

    def __sub__(self, offset: int) -> Tuple["Offset", int]:
        return (self, -offset)

    def __getitem__(self, offset: int) -> Tuple["Offset", int]:
        if self.length is not None and offset >= self.length:
            raise IndexError()
        return (self, offset)


def offset(value, length=None):
    return Offset(value, length=length)


@dataclass
class CartesianAxis:
    value: str

    def __hash__(self) -> int:
        return hash(self.value)


# dependency inversion, register fendef for embedded execution or for tracing/parsing here
fendef_embedded = None
fendef_codegen = None


def fendef(*dec_args, **dec_kwargs):
    """
    Dispatches to embedded execution or execution with code generation.

    If `backend` keyword argument is not set or None `fendef_embedded` will be called,
    else `fendef_codegen` will be called.
    """

    def wrapper(fun):
        def impl(*args, **kwargs):
            kwargs = {**kwargs, **dec_kwargs}

            if "backend" in kwargs and kwargs["backend"] is not None:
                if fendef_codegen is None:
                    raise RuntimeError("Backend execution is not registered")
                fendef_codegen(fun, *args, **kwargs)
            else:
                if fendef_embedded is None:
                    raise RuntimeError("Embedded execution is not registered")
                fendef_embedded(fun, *args, **kwargs)

        return impl

    if len(dec_args) == 1 and len(dec_kwargs) == 0 and callable(dec_args[0]):
        return wrapper(dec_args[0])
    else:
        assert len(dec_args) == 0
        return wrapper


class FundefDispatcher:
    _hook = None
    # hook is an object that
    # - evaluates to true if it should be used,
    # - is callable with an instance of FundefDispatcher
    # - returns callable that takes the function arguments

    def __init__(self, fun) -> None:
        self.fun = fun
        self.__name__ = fun.__name__

    def __getitem__(self, arg):
        if arg is Ellipsis:

            def fun(*args):
                return lift(self)(*args)

            return fun
        # elif _is_domain(arg):
        else:

            def implicit_fencil(*args, out, **kwargs):
                if len(args) == 1:

                    @fendef
                    def impl(out, inp0):
                        closure(arg(), self, [out], [inp0])

                elif len(args) == 2:

                    @fendef
                    def impl(out, inp0, inp1):
                        closure(arg(), self, [out], [inp0, inp1])

                elif len(args) == 3:

                    @fendef
                    def impl(out, inp0, inp1, inp2):
                        closure(arg(), self, [out], [inp0, inp1, inp2])

                elif len(args) == 7:

                    @fendef
                    def impl(out, inp0, inp1, inp2, inp3, inp4, inp5, inp6):
                        closure(arg(), self, [out], [inp0, inp1, inp2, inp3, inp4, inp5, inp6])

                else:
                    assert False

                impl(out, *args, **kwargs)

            return implicit_fencil

        # else:
        #     raise AssertionError("Invalid argument in stencil []")

    def __call__(self, *args):
        if type(self)._hook:
            return type(self)._hook(self)(*args)
        else:
            return self.fun(*args)

    @classmethod
    def register_hook(cls, hook):
        cls._hook = hook


def fundef(fun):
    return FundefDispatcher(fun)


@builtin_dispatch
def closure(*args):
    return BackendNotSelectedError()
