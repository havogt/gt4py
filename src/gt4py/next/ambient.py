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
import hashlib
import typing
import weakref
from collections.abc import Generator, Mapping
from typing import Annotated, Any, ClassVar

import numpy as np

from gt4py.eve import utils as eve_utils
from gt4py.next import common
from gt4py.next.ffront import fbuiltins


_UNSET: Any = object()

#: live ambient containers by stable key, so indistinguishable ones are rejected
#: at definition. Weak, write-once, and never consulted on the execution path.
_containers: weakref.WeakValueDictionary[str, type] = weakref.WeakValueDictionary()


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

    #: the synthesised parameter name. Readable, but disambiguated by a digest of
    #: the fully qualified name: a container class name is not unique, and two
    #: modules each declaring a `Grid.dx` would otherwise share one parameter and
    #: silently take one another's value.
    name: str
    #: fully qualified, for diagnostics
    qualname: str
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
                f"Ambient value '{self.qualname}' is not bound."
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
    #: stable identity of this container, unique among live containers
    _key: ClassVar[str] = ""
    #: class whose attributes carry the declared *types*, for the frontend
    _type_view: ClassVar[type]

    def __init_subclass__(cls, *, name: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Identity has to be *stable across interpreter restarts*: it ends up in
        # the synthesised parameter name, which changes the stage fingerprint and
        # therefore the build-cache key. `id()` would be unique but would miss the
        # cache on every run. Module and qualified name are stable, but not always
        # unique -- two containers built by the same factory are indistinguishable
        # -- so that case is rejected rather than silently sharing a parameter.
        key = name or f"{cls.__module__}.{cls.__qualname__}"
        if (previous := _containers.get(key)) is not None and previous is not cls:
            raise TypeError(
                f"Ambient container '{key}' is already defined."
                " Two containers that share a module and qualified name cannot be"
                " told apart across runs; pass a distinct"
                " 'class MyGrid(Container, name=...)' to separate them."
            )
        _containers[key] = cls
        cls._key = key
        cls._declarations = {}
        for attr, hint in typing.get_type_hints(cls, include_extras=True).items():
            if (declared := _declared(hint)) is None:
                continue
            type_hint, static = declared
            qualname = f"{key}.{attr}"
            # a stable digest, not a counter: the name lands in the compiled
            # signature, so it must not shift with import order
            digest = hashlib.sha1(qualname.encode()).hexdigest()[:6]
            cls._declarations[attr] = Declaration(
                name=f"{cls.__name__}_{attr}_{digest}",
                qualname=qualname,
                type_hint=type_hint,
                static=static,
                var=contextvars.ContextVar(f"{cls.__name__}.{attr}"),
            )
        cls._type_view = type(f"{cls.__name__}_types", (), dict(cls._declarations))

    def __init__(self, **values: Any) -> None:
        """
        Carry values for this container's declarations: `Grid(dx=0.5, nu=1e-3)`.

        A container instance is used two ways, and the two do not collide: one
        constructed *with* values carries them in its instance dict and is what
        `bind=` takes; one constructed empty carries nothing, so attribute access
        falls through to `__getattr__` and reads the bound value. That is the
        instance an operator reads through.
        """
        unknown = set(values) - set(type(self)._declarations)
        if unknown:
            raise TypeError(
                f"'{type(self).__name__}' does not declare {sorted(unknown)};"
                f" it declares {sorted(type(self)._declarations)}."
            )
        self.__dict__.update(values)

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
    """
    Normalise what `bind=` accepts into a declaration -> value mapping.

    A filled container binds everything it carries, which is usually what you
    want: a grid or a mesh is one thing semantically, and each program picks the
    parts it needs rather than the caller tracking which those are.
    """
    if isinstance(spec, Container):
        declarations = type(spec)._declarations
        return {declarations[attr].var: value for attr, value in vars(spec).items()}
    if isinstance(spec, Mapping):
        return dict(spec)
    if isinstance(spec, (list, tuple)):
        merged: dict[Any, Any] = {}
        for element in spec:
            merged.update(as_bindings(element))
        return merged
    raise TypeError(
        f"'{spec!r}' is not a container, a mapping of declarations to values,"
        " or a sequence of those."
    )


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
def bind(*specs: Any, **values: Any) -> Generator[None, None, None]:
    """
    Bind for the duration of the context.

        with gtx.bind(Grid(dx=0.5, nu=1e-3)):  # a filled container
        with gtx.bind(Grid.dx, 0.5):           # one declaration
    """
    if len(specs) == 2 and not isinstance(specs[0], (Container, Mapping, list, tuple)):
        mapping = {specs[0]: specs[1]}
    else:
        mapping = as_bindings(list(specs) + list(values.values()))
    with bindings(mapping):
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


def referenced_declarations(closure_vars: Mapping[str, Any]) -> dict[str, Declaration]:
    """
    Declarations the operators reachable from `closure_vars` actually read.

    Walks each operator's *own* closure variables rather than a merged mapping:
    merging is keyed by name, so two modules that both call their container
    `grid` would shadow one another. Only read declarations are returned — an
    operator that never reads `grid.dx` must not acquire it as a parameter, or a
    `Static[T]` would specialise the compiled program on a value it does not use.
    """
    from gt4py.next.ffront import field_operator_ast as foast

    referenced: dict[str, Declaration] = {}
    for value in closure_vars.values():
        foast_stage = getattr(value, "foast_stage", None)
        if foast_stage is None:
            continue
        by_attribute = attribute_declarations(foast_stage.closure_vars)
        for node in foast_stage.foast_node.walk_values().if_isinstance(foast.Attribute):
            decl = by_attribute.get((getattr(node.value, "id", None), node.attr), None)
            if decl is not None:
                referenced[decl.name] = decl
    return referenced


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
