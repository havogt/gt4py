# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay of repeated program calls with the same field arguments.

A call plan maps one call signature (identity of every non-scalar argument, values of static
scalar arguments) to a ready argument vector of the compiled program. On a hit the program is
called through that vector directly; only the non-static scalars are re-converted.
"""

from __future__ import annotations

import functools
import inspect
import os
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Protocol

import numpy as np

from gt4py.next.otf import arguments
from gt4py.next.type_system import type_specifications as ts


ENABLED: bool = os.environ.get("GT4PY_CALL_PLANS", "1") not in ("0", "false", "False")
#: Lower-bound experiment only: replay the first call's argument vector whatever is passed.
FREEZE: bool = os.environ.get("GT4PY_CALL_PLANS", "1") == "freeze"

MAX_PLANS_PER_PROGRAM = int(os.environ.get("GT4PY_CALL_PLANS_MAX", "8"))

SCALAR_TYPES: frozenset[type] = frozenset(
    {
        bool,
        int,
        float,
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float32,
        np.float64,
    }
)

#: An argument re-converted on every replay: (is_kwarg, position or name, param name, kind),
#: where kind is "scalar" or "zero_dim_field".
Patch = tuple[bool, int | str, str, str]

stats = {"hits": 0, "misses": 0, "plans": 0, "rejected": 0}
rejections: dict[str, str] = {}


class ReplayableProgram(Protocol):
    def make_replay(
        self, patches: Sequence[Patch]
    ) -> Callable[[tuple[Any, ...], dict[str, Any]], None] | None: ...


class ProgramCallPlans:
    """Call plans of one program, keyed by `key()`."""

    __slots__ = ("_param_names", "_param_types", "_static", "disabled", "misses", "plans")

    def __init__(
        self,
        param_names: Sequence[str],
        param_types: dict[str, ts.TypeSpec],
        static_names: frozenset[str],
    ):
        self._param_names = tuple(param_names)
        self._param_types = param_types
        self._static = static_names
        # value: (replay, args, kwargs, offset_provider); the call's objects are kept alive so that
        # the ids in the key cannot be reused by other objects
        self.plans: dict[Hashable, tuple[Callable, Any, Any, Any]] = {}
        #: misses since the last hit; a program whose arguments never repeat stops building plans
        self.misses = 0
        self.disabled = False

    def key(self, args: tuple[Any, ...], kwargs: dict[str, Any], offset_provider: Any) -> Hashable:
        if FREEZE:
            return None
        static = self._static
        names = self._param_names
        return (
            id(offset_provider),
            tuple(kwargs),
            *[
                (a if names[i] in static else None) if a.__class__ in SCALAR_TYPES else id(a)
                for i, a in enumerate(args)
            ],
            *[
                (v if k in static else None) if v.__class__ in SCALAR_TYPES else id(v)
                for k, v in kwargs.items()
            ],
        )

    def _patch(self, is_kw: bool, src: int | str, name: str, value: Any) -> Patch | bool | None:
        """Return the patch for one argument, None if it needs none, False if plans can't handle it."""
        type_ = self._param_types.get(name)
        if type_ is None:
            return False
        # enum members and other scalar subclasses are keyed by identity, which is fine for
        # singletons and only costs a miss otherwise
        if value.__class__ in SCALAR_TYPES or isinstance(value, (int, float, np.generic)):
            if not isinstance(type_, ts.ScalarType):
                return False
            return None if name in self._static else (is_kw, src, name, "scalar")
        if isinstance(type_, ts.FieldType):
            if not _has_fixed_buffer(type(value)):
                return False
            return (is_kw, src, name, "zero_dim_field") if not type_.dims else None
        if isinstance(type_, ts.TupleType):
            # tuples are immutable, but scalars or 0-d fields inside would be baked into the vector
            return None if _only_nd_fields(type_, value) else False
        return False

    def add(
        self,
        key: Hashable,
        program: ReplayableProgram,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        offset_provider: Any,
    ) -> None:
        self.misses += 1
        if self.misses > 4 * MAX_PLANS_PER_PROGRAM:
            self.disabled = True
            return
        if len(args) > len(self._param_names):
            stats["rejected"] += 1
            return
        patches: list[Patch] = []
        for is_kw, items in (
            ()
            if FREEZE
            else (
                (False, ((i, self._param_names[i], a) for i, a in enumerate(args))),
                (True, ((k, k, v) for k, v in kwargs.items())),
            )
        ):
            for src, name, value in items:
                patch = self._patch(is_kw, src, name, value)
                if patch is False:
                    stats["rejected"] += 1
                    rejections[name] = f"{type(value).__name__}: {self._param_types.get(name)}"
                    return
                if patch is not None:
                    patches.append(patch)  # type: ignore[arg-type]
        replay = program.make_replay([] if FREEZE else patches)
        if replay is None:
            stats["rejected"] += 1
            rejections["<make_replay>"] = type(program).__name__
            return
        if len(self.plans) >= MAX_PLANS_PER_PROGRAM:
            del self.plans[next(iter(self.plans))]
        self.plans[key] = (replay, args, kwargs, offset_provider)
        stats["plans"] += 1


_fixed_buffer: dict[type, bool] = {}


def _has_fixed_buffer(cls: type) -> bool:
    """Whether the buffer of an instance can never change, so that its identity identifies it."""
    if (result := _fixed_buffer.get(cls)) is None:
        result = _fixed_buffer[cls] = isinstance(
            inspect.getattr_static(cls, "__gt_buffer_info__", None), functools.cached_property
        )
    return result


def _only_nd_fields(type_: ts.TypeSpec, value: Any) -> bool:
    if isinstance(type_, ts.TupleType):
        return isinstance(value, tuple) and all(
            _only_nd_fields(t, v) for t, v in zip(type_.types, value, strict=True)
        )
    return isinstance(type_, ts.FieldType) and bool(type_.dims) and _has_fixed_buffer(type(value))


def make_program_call_plans(
    param_names: Sequence[str],
    param_types: dict[str, ts.TypeSpec],
    argument_descriptor_mapping: dict[type[arguments.ArgStaticDescriptor], Sequence[str]] | None,
) -> ProgramCallPlans | None:
    descriptors = argument_descriptor_mapping or {}
    # field domains are fixed by the field identity, which is part of the key
    if set(descriptors) - {arguments.StaticArg, arguments.FieldDomainDescriptor}:
        return None
    static = frozenset(descriptors.get(arguments.StaticArg, ()))
    if not static <= set(param_names):
        return None
    return ProgramCallPlans(param_names, param_types, static)
