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

This module implements the *binding* half only: the connectivities a program
needs are taken from the ambient context when the caller passes no
``offset_provider``. Referring to ambient *fields* by name inside an operator
(``mesh.edge_length``) is not implemented here.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Generator, Mapping
from typing import Any

import numpy as np

from gt4py.eve import utils as eve_utils
from gt4py.next import common


_bindings: contextvars.ContextVar[Mapping[Namespace, Any]] = contextvars.ContextVar(
    "_ambient_bindings"
)


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
    buffer = getattr(elem, "ndarray", None)
    if buffer is None:
        return elem
    if readonly:
        buffer.flags.writeable = False
    digest = int(eve_utils.content_hash(np.asarray(buffer)), 16)
    object.__setattr__(elem, common.FROZEN_HASH_ATTR, digest)
    return elem


@contextlib.contextmanager
def bindings(mapping: Mapping[Namespace, Any]) -> Generator[None, None, None]:
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
    """The connectivities and dimensions reachable as public attributes of `value`."""
    return {
        key: elem
        for key in dir(value)
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
