# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Ambient values: values declared once and read from any operator without appearing in a signature.

A declaration is an annotation in a `Container` subclass::

    class Grid(gtx.Container):
        dx: gtx.Static[float]
        nu: gtx.Extern[float]


    grid = Grid()

Class access (`Grid.dx`) yields the `contextvars.ContextVar` a value is bound to, instance
access (`grid.dx`) its current value. Values are bound for the dynamic extent of a program
call, either with the `bind` context manager or with the `bind=` argument of a call.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import hashlib
import typing
import weakref
from typing import Annotated, Any, ClassVar, Iterator, Mapping, TypeAlias

from gt4py.next.ffront import (
    fbuiltins,
    field_operator_ast as foast,
    stages as ffront_stages,
    type_specifications as ts_ffront,
)
from gt4py.next.type_system import type_specifications as ts, type_translation


class _StaticMarker: ...


class _ExternMarker: ...


type Static[T] = Annotated[T, _StaticMarker]
type Extern[T] = Annotated[T, _ExternMarker]


def parameter_name(container_id: str, attr: str) -> str:
    """
    Name of the program parameter synthesised for a declaration.

    The name reaches the build-cache key, therefore it is derived from the qualified name of
    the declaration instead of anything that changes between interpreter runs.

    Examples:
        >>> parameter_name("some.module.Grid", "dx")
        'Grid_dx_1235ba'
    """
    digest = hashlib.sha256(f"{container_id}.{attr}".encode()).hexdigest()[:6]
    return f"{container_id.rsplit('.', 1)[-1]}_{attr}_{digest}"


@dataclasses.dataclass(frozen=True)
class Declaration:
    container_id: str
    attr: str
    type_: ts.TypeSpec
    static: bool
    var: contextvars.ContextVar[Any]

    @property
    def qualified_name(self) -> str:
        return f"{self.container_id}.{self.attr}"

    @property
    def param_name(self) -> str:
        return parameter_name(self.container_id, self.attr)

    def value(self) -> Any:
        try:
            return self.var.get()
        except LookupError:
            raise ValueError(
                f"Ambient declaration '{self.qualified_name}' is not bound. Bind it with "
                f"'gtx.bind(...)' or with the 'bind=' argument of the call."
            ) from None


def _declaration_type(hint: Any) -> tuple[ts.TypeSpec, bool] | None:
    """Deduce type and staticness of a declaration, or `None` if `hint` is not a declaration."""
    origin = typing.get_origin(hint)
    if origin is not Static and origin is not Extern:
        return None
    (value_hint,) = typing.get_args(hint)
    return type_translation.from_type_hint(value_hint), origin is Static


#: Containers by qualified name, to reject ambiguous declarations at class definition.
_CONTAINERS: weakref.WeakValueDictionary[str, type] = weakref.WeakValueDictionary()


class _ContainerMeta(type):
    __declarations__: dict[str, Declaration]

    def __getattr__(cls, attr: str) -> contextvars.ContextVar[Any]:
        if attr.startswith("__"):
            raise AttributeError(attr)
        try:
            return cls.__declarations__[attr].var
        except KeyError:
            raise AttributeError(attr) from None


class Container(metaclass=_ContainerMeta):
    """
    Base class of ambient value declarations.

    An instance without values reads the currently bound values, an instance with values binds
    them (`bind(Grid(dx=0.5))`). Values given at construction live in the instance dictionary,
    so they never reach `__getattr__` and the two uses do not interfere.
    """

    __declarations__: ClassVar[dict[str, Declaration]] = {}
    __container_id__: ClassVar[str] = ""

    def __init_subclass__(cls, /, *, name: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)

        if derived_from := [
            base
            for base in cls.__mro__[1:]
            if isinstance(base, _ContainerMeta) and base is not Container
        ]:
            raise TypeError(
                f"Container '{cls.__name__}' must not derive from container "
                f"'{derived_from[0].__name__}', use composition instead."
            )

        container_id = f"{cls.__module__}.{name or cls.__qualname__}"
        if container_id in _CONTAINERS:
            raise TypeError(
                f"Container '{container_id}' is already declared. Give one of them an explicit "
                f"name, e.g. 'class {cls.__name__}(gtx.Container, name=\"...\")'."
            )
        _CONTAINERS[container_id] = cls

        declarations = {}
        for attr, hint in typing.get_type_hints(cls, include_extras=True).items():
            if typing.get_origin(hint) is ClassVar:
                continue
            if (declaration_type := _declaration_type(hint)) is None:
                raise TypeError(
                    f"Invalid declaration '{cls.__name__}.{attr}', "
                    f"expected 'Static[...]' or 'Extern[...]'."
                )
            type_, static = declaration_type
            declarations[attr] = Declaration(
                container_id=container_id,
                attr=attr,
                type_=type_,
                static=static,
                var=contextvars.ContextVar(f"{container_id}.{attr}"),
            )
        cls.__declarations__ = declarations
        cls.__container_id__ = container_id

    def __init__(self, **values: Any) -> None:
        for attr in values:
            if attr not in self.__declarations__:
                raise TypeError(f"'{type(self).__name__}' has no declaration '{attr}'.")
        self.__dict__.update(values)

    def __getattr__(self, attr: str) -> Any:
        try:
            declaration = self.__declarations__[attr]
        except KeyError:
            raise AttributeError(attr) from None
        return declaration.value()

    def __gt_type__(self) -> ts.NamespaceType:
        return ts.NamespaceType(
            qualified_name=self.__container_id__,
            element_types=tuple(
                (attr, declaration.type_) for attr, declaration in self.__declarations__.items()
            ),
        )


Binding: TypeAlias = Container | Mapping[Any, Any]


def _context_var(key: Any) -> contextvars.ContextVar[Any]:
    if isinstance(key, contextvars.ContextVar):
        return key
    if isinstance(key, fbuiltins.FieldOffset):
        return key.ambient_var
    raise TypeError(f"Cannot bind '{key}', expected a container declaration or a 'FieldOffset'.")


def _pairs(binding: Binding) -> list[tuple[contextvars.ContextVar[Any], Any]]:
    if isinstance(binding, Container):
        declarations = type(binding).__declarations__
        return [(declarations[attr].var, value) for attr, value in vars(binding).items()]
    if isinstance(binding, Mapping):
        return [(_context_var(key), value) for key, value in binding.items()]
    raise TypeError(f"Cannot bind '{binding}', expected a container or a mapping.")


@contextlib.contextmanager
def bind(binding: Binding) -> Iterator[None]:
    """
    Bind ambient values for the duration of the context.

    Args:
        binding: A container carrying values, e.g. `Grid(dx=0.5)`, or a mapping from
            declarations or offsets to values, e.g. `{Grid.dx: 0.5, V2E: connectivity}`.

    Examples:
        >>> class Grid(Container):
        ...     dx: Static[float]
        >>> with bind(Grid(dx=0.5)):
        ...     Grid().dx
        0.5
    """
    tokens = [var.set(value) for var, value in _pairs(binding)]
    try:
        yield
    finally:
        for token in reversed(tokens):
            token.var.reset(token)


def _operator_stage(value: Any) -> ffront_stages.FOASTOperatorDef | None:
    """Get the FOAST stage of `value` if it is a field operator, in any of its wrappings."""
    stage = getattr(value, "foast_stage", None)
    if stage is None and (definition := getattr(value, "definition", None)) is not None:
        stage = getattr(definition, "data", None)
    return stage if isinstance(stage, ffront_stages.FOASTOperatorDef) else None


def declarations(closure_vars: Mapping[str, Any]) -> dict[str, Declaration]:
    """Declarations read by the operators reachable from `closure_vars`, by parameter name."""
    result: dict[str, Declaration] = {}
    for value in closure_vars.values():
        if (stage := _operator_stage(value)) is not None:
            result |= operator_declarations(stage)
    return dict(sorted(result.items()))


def operator_declarations(stage: ffront_stages.FOASTOperatorDef) -> dict[str, Declaration]:
    """Declarations read by an operator itself and by the operators it calls, by parameter name."""
    result: dict[str, Declaration] = {}
    # Attributes are resolved against this operator's own closure variables: the same name may
    # refer to different containers in different operators.
    for node in stage.foast_node.pre_walk_values().if_isinstance(foast.Attribute):
        if isinstance(node.value, foast.Name) and isinstance(
            container := stage.closure_vars.get(str(node.value.id)), Container
        ):
            if (declaration := type(container).__declarations__.get(node.attr)) is not None:
                result[declaration.param_name] = declaration
    return dict(sorted((result | declarations(stage.closure_vars)).items()))


def values(declarations: Mapping[str, Declaration]) -> dict[str, Any]:
    """Currently bound value of each declaration, by parameter name."""
    return {name: declaration.value() for name, declaration in declarations.items()}


def with_parameters(
    program_type: ts_ffront.ProgramType, declarations: Mapping[str, Declaration]
) -> ts_ffront.ProgramType:
    """Add the synthesised parameters of `declarations` to a program type."""
    definition = program_type.definition
    if not (
        new_params := {
            name: declaration.type_
            for name, declaration in declarations.items()
            if name not in definition.pos_or_kw_args
        }
    ):
        return program_type
    return ts_ffront.ProgramType(
        definition=ts.FunctionType(
            pos_only_args=definition.pos_only_args,
            pos_or_kw_args={**definition.pos_or_kw_args, **new_params},
            kw_only_args=definition.kw_only_args,
            returns=definition.returns,
        )
    )


def _offsets(closure_vars: Mapping[str, Any]) -> dict[str, fbuiltins.FieldOffset]:
    result: dict[str, fbuiltins.FieldOffset] = {}
    for value in closure_vars.values():
        if isinstance(value, fbuiltins.FieldOffset):
            result[str(value.value)] = value
        elif (stage := _operator_stage(value)) is not None:
            result |= _offsets(stage.closure_vars)
    return result


def offset_provider(closure_vars: Mapping[str, Any]) -> dict[str, Any]:
    """
    Offset provider assembled from the bound offsets referenced by `closure_vars`.

    Only the offsets reachable from `closure_vars` are considered, so that an unrelated bound
    offset does not leak into the offset provider (and hence into the compiled program key).
    """
    return {
        name: value
        for name, offset in sorted(_offsets(closure_vars).items())
        if (value := offset.ambient_var.get(None)) is not None
    }
