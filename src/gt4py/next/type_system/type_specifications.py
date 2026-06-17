# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Final, Iterator, Optional, Sequence, TypeVar

from gt4py.eve import (
    datamodels as eve_datamodels,
    extended_typing as xtyping,
    type_definitions as eve_types,
)
from gt4py.next import common


class TypeSpec(eve_datamodels.DataModel, kw_only=False, frozen=True): ...  # type: ignore[call-arg]


class DataType(TypeSpec):
    """
    A base type for all types that represent data storage.

    Derive floating point, integral or field types from this class.
    """


class CallableType(TypeSpec):
    """
    A base type for all types are callable.

    Derive other callable types, such as FunctionType or FieldOperatorType from
    this class.
    """


class VoidType(TypeSpec):
    """
    Return type of a function without return values.

    Note: only useful for stateful dialects.
    """


class DimensionType(TypeSpec):
    dim: common.Dimension

    def __str__(self) -> str:
        return str(self.dim)


class IndexType(TypeSpec):
    """
    Represents the type of an index into a dimension.
    """

    dim: common.Dimension

    def __str__(self) -> str:
        return f"Index[{self.dim}]"


class OffsetType(TypeSpec):
    # TODO(havogt): replace by ConnectivityType
    source: common.Dimension
    target: tuple[common.Dimension] | tuple[common.Dimension, common.Dimension]

    def __str__(self) -> str:
        return f"Offset[{self.source}, {self.target}]"


class ScalarKind(eve_types.IntEnum):
    BOOL = 1
    INT8 = 2
    UINT8 = 3
    INT16 = 4
    UINT16 = 5
    INT32 = 6
    UINT32 = 7
    INT64 = 8
    UINT64 = 9
    FLOAT32 = 10
    FLOAT64 = 11
    STRING = 12


class ScalarType(DataType):
    kind: ScalarKind
    shape: Optional[list[int]] = None

    def __str__(self) -> str:
        kind_str = self.kind.name.lower()
        if self.shape is None:
            return kind_str
        return f"{kind_str}{self.shape}"


def _canonicalize_constraints(constraints: Sequence[ScalarType]) -> tuple[ScalarType, ...]:
    # A value-constrained type variable resolves to exactly one of its constraints, so their
    # order carries no meaning; canonicalize it to make `TypeVarType` identity order-insensitive.
    return tuple(sorted(constraints, key=lambda c: c.kind))


class TypeVarType(TypeSpec):
    """
    A type variable, spanning two roles unified into one representation.

    - **Named** (``name`` is set): a universally quantified, value-constrained generic
      parameter. Represents the type of a value-constrained Python ``typing.TypeVar``
      (e.g. ``TypeVar("T", float32, float64)``, or the PEP 695 ``def op[T: (float32,
      float64)]``) used in the signature of a generic operator. Identity is the ``name``,
      scoped to one operator signature: two occurrences with the same ``name`` denote the
      same type, and each use resolves to exactly one of ``constraints``.
    - **Deferred / anonymous** (``name`` is ``None``): a placeholder for a type that is
      not yet inferred. It carries no identity, and ``bound`` optionally constrains the
      *category* of the eventual type (any subclass of the given ``TypeSpec`` class(es)),
      or is ``None`` for no constraint at all. Build these via the :func:`DeferredType`
      factory and test for them via :func:`is_deferred`.

    Subclassing ``TypeSpec`` (rather than ``DataType``) lets a deferred type stand in for
    non-data types as well (e.g. ``OffsetType``, ``FunctionType``, ``ProgramType``), while
    still fitting into ``FieldType.dtype``, tuple members and ``foast.Symbol`` via the
    explicit unions that list ``TypeVarType``.
    """

    name: Optional[str] = None
    bound: Optional[type[TypeSpec] | tuple[type[TypeSpec], ...]] = None
    constraints: tuple[ScalarType, ...] = eve_datamodels.field(
        default=(), converter=_canonicalize_constraints
    )

    def __str__(self) -> str:
        if self.name is not None:
            return f"{self.name}: ({' | '.join(map(str, self.constraints))})"
        if self.bound is None:
            return "<deferred>"
        bound = self.bound if isinstance(self.bound, tuple) else (self.bound,)
        return f"<deferred: {' | '.join(b.__name__ for b in bound)}>"

    @eve_datamodels.validator("constraints")
    def _constraints_validator(
        self, attribute: eve_datamodels.Attribute, constraints: tuple[ScalarType, ...]
    ) -> None:
        if self.name is not None and not constraints:
            raise ValueError(
                f"Type variable '{self.name}' must be value-constrained, i.e. have at"
                " least one constraint."
            )


def DeferredType(
    constraint: Optional[type[TypeSpec] | tuple[type[TypeSpec], ...]] = None,
) -> TypeVarType:
    """Construct an anonymous, not-yet-inferred type placeholder.

    Thin factory over :class:`TypeVarType`: a deferred type is just a type variable
    without identity (``name is None``) whose optional ``constraint`` bounds the category
    of the eventual type. Kept as a named factory for readability and backwards
    compatibility of the many construction sites.
    """
    return TypeVarType(name=None, bound=constraint)


def is_deferred(type_: TypeSpec) -> xtyping.TypeGuard[TypeVarType]:
    """Whether ``type_`` is an anonymous, not-yet-inferred placeholder (see :func:`DeferredType`)."""
    return isinstance(type_, TypeVarType) and type_.name is None


def is_type_var(type_: TypeSpec) -> xtyping.TypeGuard[TypeVarType]:
    """Whether ``type_`` is a *named*, value-constrained generic type variable.

    The complement of :func:`is_deferred` among ``TypeVarType`` instances: ``True`` only
    for type variables with identity (``name is not None``), whose ``constraints`` carry
    the value set the variable ranges over. The constraint-evaluating predicates and the
    binding/promotion utilities apply only to these.
    """
    return isinstance(type_, TypeVarType) and type_.name is not None


class ListType(DataType):
    """Represents a neighbor list in the ITIR representation.

    Note:
      - not used in the frontend. The concept is represented as Field with local Dimension.
      - `None` is used to describe lists originating from `make_const_list`.
    """

    element_type: DataType
    offset_type: common.Dimension | None


class FieldType(DataType, CallableType):
    dims: list[common.Dimension]
    dtype: ScalarType | ListType | TypeVarType

    def __str__(self) -> str:
        dims = "..." if self.dims is Ellipsis else f"[{', '.join(dim.value for dim in self.dims)}]"
        return f"Field[{dims}, {self.dtype}]"

    @eve_datamodels.validator("dims")
    def _dims_validator(
        self, attribute: eve_datamodels.Attribute, dims: list[common.Dimension]
    ) -> None:
        common.check_dims(dims)


class TupleType(DataType):
    # TODO(tehrengruber): Remove the deferred `TypeVarType` again. This was erroneously
    #  introduced before we checked the annotations at runtime. All attributes of
    #  a type that are types themselves must be concrete.
    types: list[DataType | DimensionType | TypeVarType]

    def __str__(self) -> str:
        return f"tuple[{', '.join(map(str, self.types))}]"

    def __iter__(self) -> Iterator[DataType | DimensionType | TypeVarType]:
        yield from self.types

    def __len__(self) -> int:
        return len(self.types)


class AnyPythonType:
    """Marker type representing any Python type which cannot be used for instantiation.

    This is used as a workaround for missing generic support in the case of passing
    a named collection of fields to a scan where it becomes a named collection of scalars.
    """

    def __init__(self) -> None:
        raise AssertionError("Internal Error: The 'AnyPythonType' should never be instantiated.")


#: The 'ANY_PYTHON_TYPE_NAME' can be used in 'NamedCollectionType.original_python_type' to indicate
#: that any original python type is acceptable that is structurally compatible.
#: This is used as a workaround for missing generic support in the case of passing
#: a named collection of fields to a scan where it becomes a named collection of scalars.
#: Note: 'Any' cannot be instantiated and therefore should only be used for type-checking,
#: but not in places where the original python type is actually needed,
#: e.g. `make_named_collection_constructor_from_type_spec`.
ANY_PYTHON_TYPE_NAME: Final[str] = "typing:Any"


class NamedCollectionType(DataType):
    types: list[DataType | DimensionType | TypeVarType]
    keys: list[str]
    #: The original python type. It should be only used in the boundaries between
    #: Python and GT4Py DSL, that is, `type translation` and in constructor/extractor
    #: steps for custom containers.
    #: It uses the "entry-point"-like format required by `pkgutil.resolve_name()`:
    #:   '__module__:__qualname__'
    original_python_type: (
        str  # Format: '__module__:__qualname__' (as required by `pkgutil.resolve_name()`)
    )

    def __getattr__(self, name: str) -> DataType | DimensionType | TypeVarType:
        keys = object.__getattribute__(self, "keys")
        if name in keys:
            return self.types[keys.index(name)]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __str__(self) -> str:
        return f"NamedTuple{{{', '.join(f'{k}: {v}' for k, v in zip(self.keys, self.types))}}}"

    def __iter__(self) -> Iterator[DataType | DimensionType | TypeVarType]:
        # Note: Unlike `Mapping`s, we iterate the values (not the keys) by default.
        yield from self.types

    def __len__(self) -> int:
        return len(self.types)


CollectionTypeSpecT = TypeVar("CollectionTypeSpecT", TupleType, NamedCollectionType)
CollectionTypeSpec = TupleType | NamedCollectionType
COLLECTION_TYPE_SPECS: Final[tuple[type[CollectionTypeSpec], ...]] = xtyping.get_args(
    CollectionTypeSpec
)


class FunctionType(CallableType):
    pos_only_args: Sequence[TypeSpec]
    pos_or_kw_args: dict[str, TypeSpec]
    kw_only_args: dict[str, TypeSpec]
    returns: TypeSpec

    def __str__(self) -> str:
        arg_strs = [str(arg) for arg in self.pos_only_args]
        kwarg_strs = [f"{key}: {value}" for key, value in self.pos_or_kw_args.items()]
        args_str = ", ".join((*arg_strs, *kwarg_strs))
        return f"({args_str}) -> {self.returns}"


class ConstructorType(CallableType):
    definition: FunctionType

    @property
    def constructed_type(self) -> TypeSpec:
        return self.definition.returns


class DomainType(DataType):
    dims: list[common.Dimension]
