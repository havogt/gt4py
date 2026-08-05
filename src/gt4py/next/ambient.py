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

Two things are ambient:

- **Connectivities**, via a `Namespace`: a program called without an
  ``offset_provider`` takes one from whatever is bound.
- **Values**, via a declaration ``dx = Static[float]`` referenced by bare name
  inside an operator. It never appears in a signature, so it does not have to be
  threaded through nested operators.

``Static[T]`` is folded into the generated code, so each distinct value gets its
own compiled variant. ``Extern[T]`` is meant to be supplied as a runtime
argument instead — **not yet implemented**: it currently behaves like
``Static[T]``, because making it a runtime argument requires synthesising a
program parameter (the reference cannot stay a free symbol: eve validates symbol
refs when ``itir.Program`` is constructed). Ambient *fields* (``mesh.edge_length``)
need that same parameter machinery.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
from collections.abc import Generator, Mapping
from typing import Any

import numpy as np

from gt4py.eve import utils as eve_utils
from gt4py.next import common


_UNBOUND: Any = object()

#: keyed by declaration object: a `Namespace` or an `AmbientValue`
_bindings: contextvars.ContextVar[Mapping[Any, Any]] = contextvars.ContextVar("_ambient_bindings")


class Namespace:
    """
    A named collection of ambient values, resolved by attribute access.

    The declaration is the namespace object itself; ``mesh.e2v`` is a reference
    that only becomes a value once something is bound to ``mesh``. Attribute
    names are not declared up front in this prototype.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return f"Namespace('{self._name}')"

    def __getattr__(self, name: str) -> AmbientRef:
        # only reached when normal lookup fails; dunder/private probes (pickle,
        # copy, the DSL frontend) must not be turned into references
        if name.startswith("_"):
            raise AttributeError(name)
        return AmbientRef(self, name)

    @property
    def bound(self) -> Any:
        """The object currently bound to this namespace."""
        binding = _bindings.get({}).get(self, None)
        if binding is None:
            raise ValueError(
                f"Nothing is bound to ambient namespace '{self._name}'."
                " Use 'gtx.bind(<namespace>, <value>)' around the call."
            )
        return binding


@dataclasses.dataclass(frozen=True)
class AmbientRef:
    """
    A deferred reference to `namespace.name`, resolved when something is bound.

    Attribute access on a `Namespace` yields one of these instead of a value,
    because at the point an operator is *defined* nothing is bound yet.
    """

    namespace: Namespace
    name: str

    def __repr__(self) -> str:
        return f"{self.namespace._name}.{self.name}"

    @property
    def value(self) -> Any:
        return getattr(self.namespace.bound, self.name)


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

    @property
    def value(self) -> Any:
        binding = _bindings.get({}).get(self, _UNBOUND)
        if binding is _UNBOUND:
            raise ValueError(
                f"Ambient value '{self!r}' is not bound."
                f" Pass 'bind={{<declaration>: <value>}}' at the call, or use 'gtx.bind'."
            )
        return binding

    # Embedded execution runs the operator body as plain Python, so a bound
    # declaration has to behave like the scalar it stands for.
    def _v(self) -> Any:
        return self.value

    def __float__(self) -> float:
        return float(self.value)

    def __int__(self) -> int:
        return int(self.value)

    def __bool__(self) -> bool:
        return bool(self.value)

    def __neg__(self) -> Any:
        return -self.value

    def __add__(self, other: Any) -> Any:
        return self.value + other

    def __radd__(self, other: Any) -> Any:
        return other + self.value

    def __sub__(self, other: Any) -> Any:
        return self.value - other

    def __rsub__(self, other: Any) -> Any:
        return other - self.value

    def __mul__(self, other: Any) -> Any:
        return self.value * other

    def __rmul__(self, other: Any) -> Any:
        return other * self.value

    def __truediv__(self, other: Any) -> Any:
        return self.value / other

    def __rtruediv__(self, other: Any) -> Any:
        return other / self.value

    def __pow__(self, other: Any) -> Any:
        return self.value**other

    def __rpow__(self, other: Any) -> Any:
        return other**self.value


class _Declarator:
    def __init__(self, static: bool) -> None:
        self._static = static

    def __getitem__(self, type_hint: Any) -> AmbientValue:
        return AmbientValue(type_hint, static=self._static)


#: `dx = Extern[float]` — supplied as a runtime argument.
Extern = _Declarator(static=False)
#: `dx = Static[float]` — folded in as a literal; one compiled variant per value.
Static = _Declarator(static=True)


def ambient_values_in(closure_vars: Mapping[str, Any]) -> dict[str, AmbientValue]:
    """The ambient declarations referenced by a set of closure variables."""
    return {k: v for k, v in closure_vars.items() if isinstance(v, AmbientValue)}


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


@contextlib.contextmanager
def bindings(mapping: Mapping[Any, Any]) -> Generator[None, None, None]:
    """Bind several namespaces at once, for the duration of the context."""
    if not mapping:
        yield
        return
    for value in mapping.values():
        for elem in offset_provider_of(value).values():
            freeze(elem)
    token = _bindings.set({**_bindings.get({}), **mapping})
    try:
        yield
    finally:
        _bindings.reset(token)


@contextlib.contextmanager
def bind(namespace: Namespace, value: Any) -> Generator[None, None, None]:
    """Bind `value` to `namespace` for the duration of the context."""
    with bindings({namespace: value}):
        yield


def resolve(explicit: common.OffsetProvider | None) -> common.OffsetProvider:
    """The caller's offset provider if given, otherwise the ambient one."""
    return offset_provider() if explicit is None else explicit


def offset_provider_of(value: Any) -> dict[str, Any]:
    """
    The connectivities and dimensions reachable as public attributes of `value`.

    Attribute *order* is preserved from the bound object's own `__dict__`, not
    taken from `dir()`: `dir()` sorts alphabetically, and gt4py's offset
    provider is order-sensitive (see `hash_offset_provider_items_by_id`), so
    reordering silently hands a compiled program the wrong tables.
    """
    names = vars(value).keys() if hasattr(value, "__dict__") else dir(value)
    return {
        key: elem
        for key in names
        if not key.startswith("_")
        and isinstance(elem := getattr(value, key), (common.Connectivity, common.Dimension))
    }


def offset_provider() -> common.OffsetProvider:
    """
    Collect the offset provider from all bound namespaces.

    Every `common.Connectivity` reachable as an attribute of a bound object
    contributes under its attribute name. Names must not collide across
    namespaces — an ambiguous offset would silently pick one mesh's table.
    """
    collected: dict[str, Any] = {}
    for namespace, value in _bindings.get({}).items():
        for key, elem in offset_provider_of(value).items():
            if key in collected and collected[key] is not elem:
                raise ValueError(
                    f"Ambient offset '{key}' is provided by more than one namespace;"
                    f" '{namespace}' conflicts with an earlier binding."
                )
            collected[key] = elem
    return collected


def bound_values_in(closure_vars: Mapping[str, Any]) -> dict[str, Any]:
    """
    Resolved values of the ambient declarations referenced by `closure_vars`.

    Unbound declarations are skipped rather than raising: a program may close
    over declarations it does not use on this path, and the ones it does use
    surface later as a missing symbol.
    """
    current = _bindings.get({})
    return {
        name: current[decl]
        for name, decl in ambient_values_in(closure_vars).items()
        if decl in current
    }


def current_static_key() -> tuple[Any, ...]:
    """A hashable summary of the bound `Static[T]` values, for compiled-program keys."""
    return tuple(
        sorted(
            (id(decl), eve_utils.content_hash(value))
            for decl, value in _bindings.get({}).items()
            if isinstance(decl, AmbientValue) and decl.static
        )
    )


def fingerprint_declaration(decl: AmbientValue) -> Any:
    """
    What an ambient declaration contributes to a stage fingerprint.

    A `Static[T]` value is baked into the lowered code, so the *current binding*
    has to be part of the fingerprint — otherwise the lowering cache serves one
    value's code for another. Reads the binding rather than mirroring it onto
    the declaration, so it stays context-local.
    """
    current = _bindings.get({})
    bound = current.get(decl, _UNBOUND)
    return (
        decl.static,
        decl.type_hint,
        None if bound is _UNBOUND else eve_utils.content_hash(bound),
    )
