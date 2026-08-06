# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Prototype: ambient values, bound at program-execution time.

An ambient value is declared once and reached from any program without appearing
in a signature, so it does not have to be threaded through nested operators.
Binding happens at execution time (JIT time for compiled backends), which is what
lets the same programs run against a second mesh in one process.

A declaration is an *annotation* in a container, and the thing it binds to is a
plain `contextvars.ContextVar`::

    class Grid(Container):
        dx: Static[float]
        nu: Extern[float]


    grid = Grid()


    @gtx.field_operator
    def delta_x(f: IJField) -> IJField:
        return (1.0 / grid.dx) * (f(I + 1) - f)


    prog(f, out, bind={Grid.dx: 0.5, Grid.nu: 1e-3})

`Grid.dx` (class access) *is* the `ContextVar`, which is what `bind=` takes as a
key; `grid.dx` (instance access) is its current value, so embedded execution --
which runs the operator body as plain Python -- sees an ordinary float. A
`FieldOffset` binds the same way, carrying its own `ContextVar`.

`Static[T]` and `Extern[T]` are `Annotated` aliases, so a type checker sees plain
`T` and only the binding machinery reads the marker.

A declaration becomes a **synthesised program parameter** when the program is
defined (`func_to_past`), so from there it travels the ordinary path: type
checking, lowering, `static_params` and the compiled-program key all treat it
like any other argument, and only the *value* is supplied per call. The two forms
differ in one place only -- whether that parameter is listed as static:

- `Extern[T]` is an ordinary runtime argument: one compiled program serves every
  value.
- `Static[T]` is a static argument, so the existing static-argument machinery
  folds it into the generated code and keys the compiled variant on it: one
  compiled program per distinct value.

Nothing here is global: what a program can see is decided by *its own* closure
variables, never by a registry of everything that happens to be bound.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import typing
from collections.abc import Generator, Mapping
from typing import Annotated, Any, ClassVar

import numpy as np

from gt4py.eve import utils as eve_utils
from gt4py.next import common
from gt4py.next.ffront import fbuiltins


_UNSET: Any = object()


class _StaticMarker:
    """Annotation marker: fold the value into the generated code."""


class _ExternMarker:
    """Annotation marker: pass the value as a runtime argument."""


#: `dx: Static[float]` -- folded in; one compiled variant per distinct value.
type Static[T] = Annotated[T, _StaticMarker]
#: `nu: Extern[float]` -- a runtime argument; one compiled program for all values.
type Extern[T] = Annotated[T, _ExternMarker]


@dataclasses.dataclass(frozen=True)
class Declaration:
    """What a container annotation declares: a name, a type, a kind, a variable."""

    #: container-qualified, so two modules declaring a `dx` cannot collide
    name: str
    type_hint: Any
    static: bool
    var: contextvars.ContextVar

    def __gt_type__(self) -> Any:
        from gt4py.next.type_system import type_translation

        return type_translation.from_type_hint(self.type_hint)

    @property
    def value(self) -> Any:
        value = self.var.get(_UNSET)
        if value is _UNSET:
            raise ValueError(
                f"Ambient value '{self.name}' is not bound."
                " Pass 'bind={<declaration>: <value>}' at the call, or use 'gtx.bind'."
            )
        return value


def _declared(hint: Any) -> tuple[Any, bool] | None:
    """Split a `Static[T]` / `Extern[T]` annotation into `(T, is_static)`."""
    alias = getattr(hint, "__origin__", None)
    markers = getattr(getattr(alias, "__value__", None), "__metadata__", ())
    if _StaticMarker in markers:
        static = True
    elif _ExternMarker in markers:
        static = False
    else:
        return None
    (base,) = typing.get_args(hint)
    return base, static


class _ContainerMeta(type):
    def __getattr__(cls, name: str) -> contextvars.ContextVar:
        # class access yields the variable, which is the `bind=` key
        declarations: dict[str, Declaration] = cls.__dict__.get("_declarations", {})
        if name in declarations:
            return declarations[name].var
        raise AttributeError(name)


class Container(metaclass=_ContainerMeta):
    """
    Base for a container of ambient declarations.

    Declarations are annotations, so they create no class attribute and instance
    access reaches `__getattr__` -- which is where the `ContextVar` is read.
    """

    _declarations: ClassVar[dict[str, Declaration]] = {}
    #: class whose attributes carry the declared *types*, for the frontend
    _type_view: ClassVar[type]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._declarations = {}
        for attr, hint in typing.get_type_hints(cls, include_extras=True).items():
            if (declared := _declared(hint)) is None:
                continue
            type_hint, static = declared
            cls._declarations[attr] = Declaration(
                name=f"{cls.__name__}_{attr}",
                type_hint=type_hint,
                static=static,
                var=contextvars.ContextVar(f"{cls.__name__}.{attr}"),
            )
        cls._type_view = type(f"{cls.__name__}_types", (), dict(cls._declarations))

    def __getattr__(self, name: str) -> Any:
        try:
            declaration = type(self)._declarations[name]
        except KeyError:
            raise AttributeError(name) from None
        return declaration.value

    def __gt_type__(self) -> Any:
        from gt4py.next.type_system import type_translation

        # a container types itself as a namespace over its declared types, so
        # `grid.dx` resolves at definition time with nothing bound
        return type_translation.NamespaceProxy(type(self)._type_view)


def variable_for(declaration: Any) -> contextvars.ContextVar:
    """The `ContextVar` a declaration binds to."""
    if isinstance(declaration, contextvars.ContextVar):
        return declaration
    if isinstance(declaration, fbuiltins.FieldOffset):
        return declaration._ambient_var
    raise TypeError(f"'{declaration!r}' is not an ambient declaration.")


def freeze(elem: Any, *, readonly: bool = False) -> Any:
    """
    Give an offset provider element a content hash, so it identifies by value.

    An ambient value is static for the jitted programs that see it, so it may
    identify itself by *content* rather than by `id`. The hash is computed once,
    here -- the O(size) cost is paid at freeze time, never per call.

    `readonly` additionally marks the buffer immutable, which is what makes the
    cached hash trustworthy. It is **off by default because it breaks the gtfn
    bindings**: they are generated with mutable `ndarray` parameters and reject a
    non-writeable array outright. Until that is fixed, the hash is only as stable
    as the caller's discipline.
    """
    if common.frozen_content_hash(elem) is not None:
        return elem
    if not hasattr(elem, "ndarray"):
        return elem
    if readonly:
        # numpy-only; device buffers have no writeable flag
        elem.ndarray.flags.writeable = False
    # asnumpy, not np.asarray: the latter refuses device arrays outright, so
    # hashing through it fails on every GPU backend.
    host = elem.asnumpy() if hasattr(elem, "asnumpy") else np.asarray(elem.ndarray)
    object.__setattr__(elem, common.FROZEN_HASH_ATTR, int(eve_utils.content_hash(host), 16))
    return elem


def as_bindings(spec: Any) -> dict[Any, Any]:
    """Normalise what `bind=` accepts into a declaration -> value mapping."""
    if isinstance(spec, Mapping):
        return dict(spec)
    raise TypeError(f"'{spec!r}' is not a mapping of declarations to values.")


@contextlib.contextmanager
def bindings(mapping: Mapping[Any, Any]) -> Generator[None, None, None]:
    """Bind declarations to values for the duration of the context."""
    tokens: list[tuple[contextvars.ContextVar, contextvars.Token]] = []
    for declaration, value in mapping.items():
        if isinstance(value, common.Connectivity):
            freeze(value)
        var = variable_for(declaration)
        tokens.append((var, var.set(value)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)


@contextlib.contextmanager
def bind(declaration: Any, value: Any) -> Generator[None, None, None]:
    """Bind `value` to `declaration` for the duration of the context."""
    with bindings({declaration: value}):
        yield


def offset_provider_for(closure_vars: Mapping[str, Any]) -> common.OffsetProvider:
    """
    Assemble the offset provider from the offsets *this* program references.

    Scoped to the given closure variables rather than to everything currently
    bound: an unrelated mesh must not leak into a program's offset provider,
    where it would also perturb the compiled-program key.
    """
    return {
        str(offset.value): value
        for offset in closure_vars.values()
        if isinstance(offset, fbuiltins.FieldOffset)
        if (value := offset._ambient_var.get(_UNSET)) is not _UNSET
    }


def resolve(
    explicit: common.OffsetProvider | None, closure_vars: Mapping[str, Any]
) -> common.OffsetProvider:
    """The caller's offset provider if given, otherwise the ambient one."""
    return offset_provider_for(closure_vars) if explicit is None else explicit


def declarations_in(closure_vars: Mapping[str, Any]) -> dict[str, Declaration]:
    """Ambient declarations reachable from a set of closure variables, by parameter name."""
    return {
        decl.name: decl
        for value in closure_vars.values()
        if isinstance(value, Container)
        for decl in type(value)._declarations.values()
    }


def attribute_declarations(
    closure_vars: Mapping[str, Any],
) -> dict[tuple[str | None, str], Declaration]:
    """Map `(container closure-var name, attribute)` to the declaration it reads."""
    return {
        (var_name, attr): decl
        for var_name, value in closure_vars.items()
        if isinstance(value, Container)
        for attr, decl in type(value)._declarations.items()
    }


def attribute_parameter_names(
    closure_vars: Mapping[str, Any],
) -> dict[tuple[str | None, str], str]:
    """Map `(container closure-var name, attribute)` to the synthesised parameter name."""
    return {key: decl.name for key, decl in attribute_declarations(closure_vars).items()}
