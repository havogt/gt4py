# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Prototype: ambient values, bound at program-execution time.

An ambient value is declared once and reached from any program without
appearing in a signature. Binding happens at execution time (JIT time for
compiled backends), so the same programs can run against a second mesh in one
process.

Everything ambient is bound the same way: **the declaration is the key**.

    prog(a, out, bind={V2E: connectivity, dx: 0.5})

A `FieldOffset` is already a declaration — it names the offset and fixes its
source and target — so it binds exactly like a `Static[T]` / `Extern[T]` value,
and a program called without an ``offset_provider`` assembles one from the bound
offsets. A container may declare what it supplies, in which case the class
attribute is the declaration and the instance attribute the value::

    class Mesh:
        V2E = V2E  # this mesh supplies the V2E connectivity
        dx = physics.dx

Binding by declaration rather than by attribute *name* is what lets a container
supply the very offset an operator refers to, instead of something that merely
shares its name.

A value declared this way is referenced by bare name inside an operator and
never appears in a signature, so it does not have to be threaded through nested
operators.

A declaration becomes a **synthesised program parameter** when the program is
defined (`func_to_past`), so from there on it travels the ordinary path: type
checking, lowering, `static_params` and the compiled-program key all treat it
like any other argument, and only the *value* is supplied per call. The caller
never names it.

The two forms differ in one place only — whether that parameter is listed as a
static one:

- ``Extern[T]`` is an ordinary runtime argument: one compiled program serves
  every value.
- ``Static[T]`` is a static argument, so the existing static-argument machinery
  folds it into the generated code and keys the compiled variant on it: one
  compiled program per distinct value.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Generator, Mapping
from typing import Any

import numpy as np

from gt4py.eve import utils as eve_utils
from gt4py.next import common
from gt4py.next.ffront import fbuiltins


_UNBOUND: Any = object()

#: keyed by declaration object: a `Namespace` or an `AmbientValue`
_bindings: contextvars.ContextVar[Mapping[Any, Any]] = contextvars.ContextVar("_ambient_bindings")


class AmbientValue:
    """
    A value declared here and bound later: `dx = Extern[float]`.

    The declaration carries the *type*, which is all the frontend needs when the
    operator is defined; the *value* arrives at bind time. Use the declaration
    object itself as the binding key: `bind={dx: 0.5}`.

    `Extern[T]` is supplied to the compiled program as a runtime argument.
    `Static[T]` is folded into it as a literal, so each distinct value gets its
    own compiled variant.
    """

    def __init__(self, type_hint: Any, *, static: bool, name: str = "?") -> None:
        self.type_hint = type_hint
        self.static = static
        self.name = name

    def __repr__(self) -> str:
        return f"{'Static' if self.static else 'Extern'}[{getattr(self.type_hint, '__name__', self.type_hint)}]"

    def __gt_type__(self) -> Any:
        from gt4py.next.type_system import type_translation

        return type_translation.from_type_hint(self.type_hint)

    def __set_name__(self, owner: type, name: str) -> None:
        # a declaration inside a container knows where it lives, which gives the
        # synthesised parameter a name that cannot collide across modules
        self.name = f"{owner.__name__}_{name}"

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        """Class access yields the declaration, instance access the bound value.

        That is what lets a container be both the thing you *declare* with
        (`Grid.dx` as a `bind=` key) and the thing you *read* inside an operator
        (`grid.dx`), without the declaration having to impersonate a scalar.
        """
        return self if obj is None else self.value

    @property
    def value(self) -> Any:
        binding = _bindings.get({}).get(self, _UNBOUND)
        if binding is _UNBOUND:
            raise ValueError(
                f"Ambient value '{self!r}' is not bound."
                f" Pass 'bind={{<declaration>: <value>}}' at the call, or use 'gtx.bind'."
            )
        return binding


class Container:
    """
    Base for a container of ambient declarations: `class Grid(Container): dx = Static[float]`.

    Reading `grid.dx` inside an operator goes through the declaration's
    descriptor, so in embedded execution it is simply the bound value — the
    declaration never has to impersonate a scalar. `Grid.dx` (class access)
    stays the declaration, which is what `bind=` takes as its key.

    The container types itself as a *namespace* over its own class, so the
    frontend resolves `grid.dx` to the declared type without needing a value.
    """

    def __gt_type__(self) -> Any:
        from gt4py.next.type_system import type_translation

        return type_translation.NamespaceProxy(type(self))


class _Declarator:
    def __init__(self, static: bool) -> None:
        self._static = static

    def __getitem__(self, type_hint: Any) -> AmbientValue:
        return AmbientValue(type_hint, static=self._static)


#: `dx = Extern[float]` — supplied as a runtime argument.
Extern = _Declarator(static=False)
#: `dx = Static[float]` — folded in as a literal; one compiled variant per value.
Static = _Declarator(static=True)


def declarations_in(closure_vars: Mapping[str, Any]) -> dict[str, AmbientValue]:
    """
    Ambient declarations reachable from a set of closure variables, by parameter name.

    Declarations live in containers, so the key is the declaration's own
    container-qualified name (`Grid_dx`) rather than whatever the referring
    scope happens to call the container. That is what keeps two modules from
    colliding when both declare a `dx`.
    """
    found: dict[str, AmbientValue] = {}
    for value in closure_vars.values():
        if isinstance(value, Container):
            for decl in vars(type(value)).values():
                if isinstance(decl, AmbientValue):
                    found[decl.name] = decl
    return found


def freeze(elem: Any, *, readonly: bool = False) -> Any:
    """
    Give an offset provider element a content hash, so it identifies by value.

    An ambient value is static for the jitted programs that see it, so it may
    identify itself by *content* rather than by `id`. The hash is computed once,
    here — the O(size) cost is paid at freeze time, never per call.

    `readonly` additionally marks the buffer immutable, which is what makes the
    cached hash trustworthy. It is **off by default because it breaks the gtfn
    bindings**: they are generated with mutable `ndarray` parameters and reject
    a non-writeable array outright. Until that is fixed, the hash is only as
    stable as the caller's discipline.
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

    A mapping is taken as-is. Anything else is treated as a *container*: its
    class attributes name the declarations it supplies, and the instance
    attribute of the same name carries the value. The declaration object itself
    is the key, so a container binds the very `Extern` an operator refers to
    rather than something that merely shares its name.

        class Grid:
            dx = physics.dx        # the declaration this grid supplies

        grid = Grid(); grid.dx = 0.5
        prog(f, out, bind=grid)
    """
    if isinstance(spec, Mapping):
        return dict(spec)
    resolved: dict[Any, Any] = {}
    for name, decl in vars(type(spec)).items():
        if not isinstance(decl, (AmbientValue, fbuiltins.FieldOffset)):
            continue
        if name not in vars(spec):
            raise ValueError(
                f"'{type(spec).__name__}' declares '{name}' but the instance does not set it."
            )
        resolved[decl] = vars(spec)[name]
    return resolved


@contextlib.contextmanager
def bindings(mapping: Mapping[Any, Any]) -> Generator[None, None, None]:
    """Bind several namespaces at once, for the duration of the context."""
    if not mapping:
        yield
        return
    for value in mapping.values():
        if isinstance(value, common.Connectivity):
            freeze(value)
    token = _bindings.set({**_bindings.get({}), **mapping})
    try:
        yield
    finally:
        _bindings.reset(token)


@contextlib.contextmanager
def bind(declaration: Any, value: Any) -> Generator[None, None, None]:
    """Bind `value` to `declaration` for the duration of the context."""
    with bindings({declaration: value}):
        yield


def resolve(explicit: common.OffsetProvider | None) -> common.OffsetProvider:
    """The caller's offset provider if given, otherwise the ambient one."""
    return offset_provider() if explicit is None else explicit


def offset_provider() -> common.OffsetProvider:
    """
    Assemble the offset provider from the bound offset declarations.

    A `FieldOffset` is itself the declaration — it already names the offset and
    fixes its source and target — so binding one is the same act as binding an
    ambient value, and no attribute of the bound object has to be inspected.
    """
    return {
        str(decl.value): value
        for decl, value in _bindings.get({}).items()
        if isinstance(decl, fbuiltins.FieldOffset)
    }


def attribute_declarations(
    closure_vars: Mapping[str, Any],
) -> dict[tuple[str | None, str], AmbientValue]:
    """Map `(container closure-var name, attribute)` to the declaration it reads."""
    return {
        (var_name, attr): decl
        for var_name, value in closure_vars.items()
        if isinstance(value, Container)
        for attr, decl in vars(type(value)).items()
        if isinstance(decl, AmbientValue)
    }


def attribute_parameter_names(
    closure_vars: Mapping[str, Any],
) -> dict[tuple[str | None, str], str]:
    """Map `(container closure-var name, attribute)` to the synthesised parameter name."""
    return {key: decl.name for key, decl in attribute_declarations(closure_vars).items()}
