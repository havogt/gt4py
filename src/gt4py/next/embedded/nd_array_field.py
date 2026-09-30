# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import collections
import dataclasses
import functools
import itertools
import math
import weakref
from collections.abc import Callable, Sequence
from types import ModuleType

import numpy as np
from numpy import typing as npt

from gt4py._core import definitions as core_defs
from gt4py.eve.extended_typing import (
    Any,
    ClassVar,
    Never,
    Optional,
    ParamSpec,
    TypeAlias,
    TypeVar,
    cast,
)
from gt4py.next import common, config
from gt4py.next.embedded import (
    common as embedded_common,
    context as embedded_context,
    exceptions as embedded_exceptions,
    structured_connectivity as _structured_conn,
)
from gt4py.next.ffront import experimental, fbuiltins


try:
    import cupy as cp
except ImportError:
    cp: Optional[ModuleType] = None  # type: ignore[no-redef]

try:
    import jax
    from jax import numpy as jnp
except ImportError:
    jax: Optional[ModuleType] = None  # type: ignore[no-redef]
    jnp: Optional[ModuleType] = None  # type: ignore[no-redef]

try:
    import torch
except ImportError:
    torch: Optional[ModuleType] = None  # type: ignore[no-redef]

try:
    import dace
except ImportError:
    dace: Optional[ModuleType] = None  # type: ignore[no-redef]


def _get_nd_array_class(*fields: common.Field | core_defs.Scalar) -> type[NdArrayField]:
    for f in fields:
        if isinstance(f, NdArrayField):
            return f.__class__
    raise AssertionError("No 'NdArrayField' found in the arguments.")


def _make_builtin(
    builtin_name: str, array_builtin_name: str, reverse: bool = False
) -> Callable[..., NdArrayField]:
    def _builtin_op(*fields: common.Field | core_defs.Scalar) -> NdArrayField:
        cls_ = _get_nd_array_class(*fields)
        xp = cls_.array_ns
        op = _get_builtin(xp, array_builtin_name)

        domain_intersection = embedded_common.domain_intersection(
            *[f.domain for f in fields if isinstance(f, common.Field)]
        )

        transformed: list[core_defs.NDArrayObject | core_defs.Scalar] = []
        for f in fields:
            if isinstance(f, common.Field):
                if f.domain == domain_intersection:
                    transformed.append(xp.asarray(f.ndarray))
                else:
                    f_broadcasted = _broadcast(f, domain_intersection.dims)
                    f_slices = _get_slices_from_domain_slice(
                        f_broadcasted.domain, domain_intersection
                    )
                    transformed.append(xp.asarray(f_broadcasted.ndarray[f_slices]))
            else:
                assert core_defs.is_scalar_type(f)
                transformed.append(f)
        if reverse:
            transformed.reverse()
        new_data = op(*transformed)
        return cls_.from_array(new_data, domain=domain_intersection)

    _builtin_op.__name__ = builtin_name
    return _builtin_op


try:
    from scipy.special import gamma as _np_gamma
except ImportError:

    def _np_gamma(a: core_defs.NDArrayObject) -> core_defs.NDArrayObject:
        return np.vectorize(math.gamma, otypes=[a.dtype])(a)


def _get_builtin(xp: ModuleType, name: str) -> Callable:
    match name:
        case "gamma":
            if xp is np:
                return _np_gamma
            if xp is cp:
                import cupyx.scipy.special

                return cupyx.scipy.special.gamma
            if xp is jnp:
                import jax.scipy.special

                return jax.scipy.special.gamma
            if (gamma := getattr(xp, "gamma", None)) is not None:
                return gamma
            raise NotImplementedError(
                f"'gamma' is not implemented for array namespace '{xp.__name__}'."
            )
        case _:
            return getattr(xp, name)


_Value: TypeAlias = common.Field | core_defs.ScalarT
_P = ParamSpec("_P")
_R = TypeVar("_R", _Value, tuple[_Value, ...])


@dataclasses.dataclass(frozen=True)
class NdArrayField(
    common.MutableField[common.DimsT, core_defs.ScalarT],
    common.FieldBuiltinFuncRegistry,
):
    """
    Shared field implementation for NumPy-like fields.

    Builtin function implementations are registered in a dictionary.
    Note: Currently, all concrete NdArray-implementations share
    the same implementation, dispatching is handled inside of the registered
    function via its namespace.
    """

    _domain: common.Domain
    _ndarray: core_defs.NDArrayObject

    array_ns: ClassVar[ModuleType]  # TODO(havogt): introduce a NDArrayNamespace protocol

    def __getstate__(self) -> dict[str, Any]:
        # Serialize only the dataclass fields, excluding cached properties
        # stored in `__dict__` (which may not be picklable).
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}

    @classmethod
    def from_array(
        cls,
        data: (
            npt.ArrayLike | core_defs.NDArrayObject
        ),  # TODO: NDArrayObject should be part of ArrayLike
        /,
        *,
        domain: common.DomainLike,
        dtype: Optional[core_defs.DTypeLike] = None,
    ) -> NdArrayField:
        domain = common.domain(domain)
        xp = cls.array_ns

        xp_dtype = None if dtype is None else xp.dtype(core_defs.dtype(dtype).scalar_type)
        array = xp.asarray(data, dtype=xp_dtype)

        if dtype is not None:
            assert cls._scalar_type_of(array) == core_defs.dtype(dtype).scalar_type

        assert issubclass(cls._scalar_type_of(array), core_defs.SCALAR_TYPES)

        assert all(isinstance(d, common.Dimension) for d in domain.dims), domain
        assert len(domain) == array.ndim
        assert all(s == 1 or len(r) == s for r, s in zip(domain.ranges, array.shape))

        return cls(domain, array)

    @staticmethod
    def _scalar_type_of(array: core_defs.NDArrayObject) -> type[core_defs.Scalar]:
        return array.dtype.type

    @staticmethod
    def _astype(array: core_defs.NDArrayObject, type_: type) -> core_defs.NDArrayObject:
        return array.astype(type_)

    @functools.cached_property
    def __gt_origin__(self) -> tuple[int, ...]:
        assert common.Domain.is_finite(self.domain)
        return tuple(-r.start for r in self.domain.ranges)

    @functools.cached_property
    def __gt_buffer_info__(self) -> common.BufferInfo:
        """
        Interface to retrieve the low-level description of a Field buffer.

        Since by default NdArrayFields are implemented as frozen dataclasses,
        and therefore the backing ndarray cannot be replaced after creation,
        this is implemented as a cached property for performance reasons.

        NDArrayField subclasses where the backing ndarray can be replaced
        should override this and make it a regular property.
        """
        return common.BufferInfo.from_ndarray(self.ndarray)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._ndarray.shape

    @property
    def domain(self) -> common.Domain:
        return self._domain

    @property
    def codomain(self) -> type[core_defs.ScalarT]:
        return self.dtype.scalar_type

    @functools.cached_property
    def dtype(self) -> core_defs.DType[core_defs.ScalarT]:
        return core_defs.dtype(self._scalar_type_of(self._ndarray))

    @property
    def ndarray(self) -> core_defs.NDArrayObject:
        return self._ndarray

    def asnumpy(self) -> np.ndarray:
        if self.array_ns == cp:
            return cp.asnumpy(self._ndarray)
        else:
            return np.asarray(self._ndarray)

    def as_scalar(self) -> core_defs.ScalarT:
        if self.domain.ndim != 0:
            raise ValueError(
                f"'as_scalar' is only valid on 0-dimensional 'Field's, got a {self.domain.ndim}-dimensional 'Field'."
            )
        # note: `.item()` will return a Python type, therefore we use indexing with an empty tuple
        return self.asnumpy()[()]  # type: ignore[return-value] # should be ensured by the 0-d check

    def premap(
        self: NdArrayField,
        *connectivities: common.Connectivity | fbuiltins.FieldOffset,
    ) -> NdArrayField:
        """
        Rearrange the field content using the provided connectivities (index mappings).

        This operation is conceptually equivalent to a regular composition of mappings
        `f∘c`, being `c` the `connectivity` argument and `f` the `self` data field.
        Note that the connectivity field appears at the right of the composition
        operator and the data field at the left.

        The composition operation is only well-defined when the codomain of `c: A → B`
        matches the domain of `f: B → ℝ` and it would then result in a new mapping
        `f∘c: A → ℝ` defined as `(f∘c)(x) = f(c(x))`. When remaping a field whose
        domain has multiple dimensions `f: A × B → ℝ`, the domain of the connectivity
        argument used in the right hand side of the operator should therefore have the
        same product of dimensions `c: S × T → A × B`. Such a mapping can also be
        expressed as a pair of mappings `c1: S × T → A` and `c2: S × T → B`, and this
        is actually the only supported form in GT4Py because `Connectivity` instances
        can only deal with a single dimension in its codomain. This approach makes
        connectivities reusable for any combination of dimensions in a field domain
        and matches the NumPy advanced indexing API, which basically is a
        composition of mappings of natural numbers representing tensor indices.

        In general, the `premap()` function is able to deal with data fields with multiple
        dimensions even if only one connectivity is passed. Connectivity arguments are then
        expanded to fully defined connectivities for each dimension in the domain of the
        field according to some rules covering the most common use cases.

        Assuming a field `f: Field[Dims[A, B], DT]` the following cases are supported:

        - If the connectivity domain only contains dimensions which are NOT part of the
          field domain (new dimensions), this function will use the same rules of
          advanced-indexing and replace the connectivity codomain dimension by its domain
          dimensions. A way to think about this is that the data field is transformed into
          a curried mapping whose domain only contains the connectivity codomain dimension,
          then composed as usual with the connectivity, and finally uncurried again:

            `f: A × B → ℝ` => `f': A → (B → ℝ)`
            `c: X × Y → A`
            `(f'∘c): X × Y → (B → ℝ)` => `(f∘c): X × Y × B → ℝ`

        - If the connectivity domain only contains dimensions which are ALREADY part of the
          data field domain, the connectivity field would be interpreted as an homomorphic
          function which preserves the domain dimensions. A way to think about this is that
          the connectivity defines how the current field data gets translated and rearranged
          into new domain ranges, and the mappings for the missing domain dimensions
          are assumed to be identities:

            `f: A × B × C → ℝ`
            `c: A × B → A` => `c0: A × B × C → A`, `c1: A × B × C → B`, `c2: A × B × C → C`
            `(f∘c): A × B × C → ℝ` => `(f∘(c0 × c1 × c2)): A × B × C → ℝ)`

        Note that cartesian shifts (e.g. `I → I_half`, `(I+1): I → I`) are just simpler
        versions of these cases where the internal structure of the data (codomain) is
        preserved and therefore the `premap` operation can be implemented as a compact
        domain translation (i.e. only transform the domain without altering the data). Such affine
        connectivities only relabel the domain; data-rearranging cases are handled as
        advanced-indexing gathers (:class:`common.GatherConnectivity`).

        Args:
            *connectivities: connectivities to be used for the `premap` operation. If only one
                connectivity is passed, it will be expanded to fully defined connectivities for
                each dimension in the domain of the field according to the rules described above.
                If more than one connectivity is passed, they all must satisfy:
                - they are either all affine (domain-only) or all gather connectivities
                - their codomains are pairwise distinct
                - no connectivity reads a dimension that another one replaces (i.e. removes
                  from the field domain by introducing new dimensions in its place)

        """  # noqa: RUF002  # TODO(egparedes): move docstring to the `premap` builtin function when it exists

        # StructuredConnectivity short-circuit: takes a different path than
        # integer-table connectivities and does not go through the
        # ConnectivityKind dispatch below. Only relevant in the full-offset
        # form (``inp(C2E)``); the indexed form ``inp(C2E[k])`` resolves to a
        # `_StructuredConnectivityK` already before reaching `premap`.
        # TODO(havogt): merge into the ConnectivityKind-based dispatch once
        # `StructuredConnectivity` implements the `common.Connectivity` protocol.
        if len(connectivities) == 1:
            conn0 = connectivities[0]
            if isinstance(conn0, fbuiltins.FieldOffset):
                conn0 = conn0.as_connectivity_field()
            if isinstance(
                conn0,
                (
                    _structured_conn.StructuredConnectivity,
                    _structured_conn._StructuredConnectivityK,
                ),
            ):
                return _structured_premap(self, conn0)

        conn_fields: list[common.Connectivity] = []
        codomains_counter: collections.Counter[common.Dimension] = collections.Counter()

        for connectivity in connectivities:
            # For neighbor reductions, a FieldOffset is passed instead of an actual Connectivity
            if not isinstance(connectivity, common.Connectivity):
                assert isinstance(connectivity, fbuiltins.FieldOffset)
                connectivity = connectivity.as_connectivity_field()
            assert isinstance(connectivity, common.Connectivity)

            conn_fields.append(connectivity)
            codomains_counter[connectivity.codomain] += 1

        if unknown_dims := [dim for dim in codomains_counter.keys() if dim not in self.domain.dims]:
            raise ValueError(
                f"Incompatible dimensions in the connectivity codomain(s) {unknown_dims}"
                f"while pre-mapping a field with domain {self.domain}."
            )

        if repeated_codomain_dims := [dim for dim, count in codomains_counter.items() if count > 1]:
            raise ValueError(
                "All connectivities must have different codomains but some are repeated:"
                f" {repeated_codomain_dims}."
            )

        is_gather = [isinstance(c, common.GatherConnectivity) for c in conn_fields]
        if any(is_gather) and not all(is_gather):
            raise ValueError(
                "Mixing connectivities that rearrange the data (gather) with connectivities that "
                "only relabel the domain (affine) is not allowed."
            )

        if not any(is_gather):
            return _domain_premap(self, *conn_fields)

        # Reject only order-dependent chains: a connectivity reading a dimension that another
        # one removes (one whose codomain is not in its own domain, so it gets replaced in the
        # output). Connectivities that keep their codomain do not remove any dimension and can
        # always be combined.
        removed = {c.codomain for c in conn_fields if c.codomain not in c.domain.dims}
        for c in conn_fields:
            if reads := removed & ({*c.domain.dims} - {c.codomain}):
                raise ValueError(
                    f"Cannot 'premap' with connectivities where one reads dimension(s) {reads} that "
                    "another replaces; apply such chained remaps in separate 'premap' calls."
                )

        return _gather_premap(self, *cast(list[common.GatherConnectivity], conn_fields))

    def __call__(
        self,
        index_field: common.Connectivity | fbuiltins.FieldOffset,
        *args: common.Connectivity | fbuiltins.FieldOffset,
    ) -> common.Field:
        return functools.reduce(
            lambda field, current_index_field: field.premap(current_index_field),
            [index_field, *args],
            self,
        )

    def restrict(self, index: common.AnyIndexSpec) -> NdArrayField:
        new_domain, buffer_slice = self._slice(index)
        new_buffer = self.ndarray[buffer_slice]
        new_buffer = self.__class__.array_ns.asarray(new_buffer)
        return self.__class__.from_array(new_buffer, domain=new_domain)

    __getitem__ = restrict

    def __setitem__(
        self: NdArrayField[common.DimsT, core_defs.ScalarT],
        index: common.AnyIndexSpec,
        value: common.Field | core_defs.NDArrayObject | core_defs.ScalarT,
    ) -> None:
        target_domain, target_slice = self._slice(index)

        if isinstance(value, common.Field):
            if not value.domain == target_domain:
                raise ValueError(
                    f"Incompatible 'Domain' in assignment. Source domain = '{value.domain}', target domain = '{target_domain}'."
                )
            value = value.ndarray

        assert hasattr(self.ndarray, "__setitem__")
        self._ndarray[target_slice] = self._as_assignable(value)  # type: ignore[index] # np and cp allow index assignment, jax overrides

    def _as_assignable(
        self, value: core_defs.NDArrayObject | core_defs.ScalarT
    ) -> core_defs.NDArrayObject | core_defs.ScalarT:
        return value

    __abs__ = _make_builtin("abs", "abs")

    __neg__ = _make_builtin("neg", "negative")

    __add__ = __radd__ = _make_builtin("add", "add")

    __pos__ = _make_builtin("pos", "positive")

    __sub__ = _make_builtin("sub", "subtract")
    __rsub__ = _make_builtin("sub", "subtract", reverse=True)

    __mul__ = __rmul__ = _make_builtin("mul", "multiply")

    __truediv__ = _make_builtin("div", "divide")
    __rtruediv__ = _make_builtin("div", "divide", reverse=True)

    __floordiv__ = _make_builtin("floordiv", "floor_divide")
    __rfloordiv__ = _make_builtin("floordiv", "floor_divide", reverse=True)

    __pow__ = _make_builtin("pow", "power")

    __mod__ = _make_builtin("mod", "mod")
    __rmod__ = _make_builtin("mod", "mod", reverse=True)

    __ne__ = _make_builtin("not_equal", "not_equal")  # type: ignore # mypy wants return `bool`

    __eq__ = _make_builtin("equal", "equal")  # type: ignore # mypy wants return `bool`

    __gt__ = _make_builtin("greater", "greater")

    __ge__ = _make_builtin("greater_equal", "greater_equal")

    __lt__ = _make_builtin("less", "less")

    __le__ = _make_builtin("less_equal", "less_equal")

    def __and__(self, other: common.Field | core_defs.ScalarT) -> NdArrayField:
        if self.dtype == core_defs.BoolDType():
            return _make_builtin("logical_and", "logical_and")(self, other)
        raise NotImplementedError("'__and__' not implemented for non-'bool' fields.")

    __rand__ = __and__

    def __or__(self, other: common.Field | core_defs.ScalarT) -> NdArrayField:
        if self.dtype == core_defs.BoolDType():
            return _make_builtin("logical_or", "logical_or")(self, other)
        raise NotImplementedError("'__or__' not implemented for non-'bool' fields.")

    __ror__ = __or__

    def __xor__(self, other: common.Field | core_defs.ScalarT) -> NdArrayField:
        if self.dtype == core_defs.BoolDType():
            return _make_builtin("logical_xor", "logical_xor")(self, other)
        raise NotImplementedError("'__xor__' not implemented for non-'bool' fields.")

    __rxor__ = __xor__

    def __invert__(self) -> NdArrayField:
        if self.dtype == core_defs.BoolDType():
            return _make_builtin("invert", "invert")(self)
        raise NotImplementedError("'__invert__' not implemented for non-'bool' fields.")

    def _slice(
        self, index: common.AnyIndexSpec
    ) -> tuple[common.Domain, common.RelativeIndexSequence]:
        index = embedded_common.canonicalize_any_index_sequence(index)
        new_domain = embedded_common.sub_domain(self.domain, index)

        index_sequence = common.as_any_index_sequence(index)
        slice_ = (
            _get_slices_from_domain_slice(self.domain, index_sequence)
            if common.is_absolute_index_sequence(index_sequence)
            else index_sequence
        )
        assert common.is_relative_index_sequence(slice_)
        return new_domain, slice_

    if dace:

        def _dace_data_ptr(self) -> int:
            return self.__gt_buffer_info__.data_ptr

        def _dace_descriptor(self) -> dace.data.Data:
            return dace.data.create_datadescriptor(self.ndarray)

    else:

        def _dace_data_ptr(self) -> int:
            raise NotImplementedError(
                "data_ptr is only supported when the 'dace' module is available."
            )

        def _dace_descriptor(self) -> dace.data.Data:
            raise NotImplementedError(
                "__descriptor__ is only supported when the 'dace' module is available."
            )

    data_ptr = _dace_data_ptr
    """
    Returns the pointer of the underlying data buffer.

    Fully equivalent to `self.__gt_buffer_info__.data_ptr`. It is only defined to emulate the
    PyTorch API for DaCe interoperability.

    Note:
        This method is experimental and will be likely removed in future versions.
    """

    __descriptor__ = _dace_descriptor
    """Extension of NdArrayField adding SDFGConvertible support in GT4Py Programs."""


@dataclasses.dataclass(frozen=True)
class NdArrayConnectivityField(
    common.GatherConnectivity[common.DimsT, common.DimT],
    NdArrayField[common.DimsT, core_defs.IntegralScalar],
):
    _codomain: common.DimT
    _skip_value: Optional[core_defs.IntegralScalar]

    @classmethod
    def from_array(  # type: ignore[override]
        cls,
        data: npt.ArrayLike | core_defs.NDArrayObject,
        /,
        codomain: common.DimT,
        *,
        domain: common.DomainLike,
        dtype: Optional[core_defs.DTypeLike] = None,
        skip_value: Optional[core_defs.IntegralScalar] = None,
    ) -> NdArrayConnectivityField:
        domain = common.domain(domain)
        xp = cls.array_ns

        xp_dtype = None if dtype is None else xp.dtype(core_defs.dtype(dtype).scalar_type)
        array = xp.asarray(data, dtype=xp_dtype)

        if dtype is not None:
            assert cls._scalar_type_of(array) == core_defs.dtype(dtype).scalar_type

        assert issubclass(cls._scalar_type_of(array), core_defs.INTEGRAL_TYPES)

        assert all(isinstance(d, common.Dimension) for d in domain.dims), domain
        assert len(domain) == array.ndim
        assert all(len(r) == s or s == 1 for r, s in zip(domain.ranges, array.shape))

        assert isinstance(codomain, common.Dimension)

        return cls(domain, array, codomain, _skip_value=skip_value)

    @classmethod
    def __gt_builtin_func__(cls, _: fbuiltins.BuiltInFunction) -> Never:  # type: ignore[override]
        raise NotImplementedError()

    @property
    def codomain(self) -> common.DimT:  # type: ignore[override] # TODO(havogt): instead of inheriting from NdArrayField, steal implementation or common base
        return self._codomain

    @property
    def skip_value(self) -> Optional[core_defs.IntegralScalar]:
        return self._skip_value

    # This embedded run-time cache is only used to speed up repeated calls to
    # `inverse_image` and `restrict`, and it should not be considered part of
    # the connectivity field definition, and therefore it should not be serialized.
    @functools.cached_property
    def _cache(self) -> dict:
        return {}

    def inverse_image(self, image_range: common.UnitRange | common.NamedRange) -> common.Domain:
        cache_key = hash((id(self.ndarray), self.domain, image_range))

        if (new_domain := self._cache.get(cache_key, None)) is None:
            if not isinstance(
                image_range, common.UnitRange
            ):  # TODO(havogt): cleanup duplication with CartesianConnectivity
                if image_range.dim != self.codomain:
                    raise ValueError(
                        f"Dimension '{image_range.dim}' does not match the codomain dimension '{self.codomain}'."
                    )

                image_range = image_range.unit_range

            assert isinstance(image_range, common.UnitRange)
            assert common.UnitRange.is_finite(image_range)

            slices = self._image_slices(image_range)
            if slices is None:
                raise ValueError("Restriction generates non-contiguous or empty dimensions.")

            new_domain = self.domain.slice_at[slices]
            self._cache[cache_key] = new_domain

        return new_domain

    @property
    def _index_table(self) -> core_defs.NDArrayObject:
        # domain inference needs concrete contents, which a traced `_ndarray` does not have;
        # the jax subclass answers with the concrete table behind the tracer
        return self._ndarray

    @property
    def _image_unknown(self) -> bool:
        """Whether the table's values are not available for domain inference."""
        return False

    def __setitem__(
        self,
        index: common.AnyIndexSpec,
        value: common.Field | core_defs.NDArrayObject | core_defs.IntegralScalar,
    ) -> Never:
        raise TypeError("'Connectivity' does not support item assignment.")

    @functools.cached_property
    def _image_bounds(self) -> Optional[tuple[int, int, tuple[slice, ...]]]:
        """Smallest value, largest value and bounding hyperslice of the non-skip entries."""
        xp = self.array_ns
        table = self._index_table
        if math.prod(table.shape) == 0:
            return None
        if self.skip_value is None:
            return int(xp.min(table)), int(xp.max(table)), tuple(slice(0, n) for n in table.shape)
        valid = table != self.skip_value
        if not xp.any(valid):
            return None
        support = []
        for axis in range(table.ndim):
            other_axes = tuple(a for a in range(table.ndim) if a != axis)
            (nonzero,) = xp.nonzero(xp.any(valid, axis=other_axes))
            support.append(slice(int(nonzero[0]), int(nonzero[-1]) + 1))
        info = xp.iinfo(table.dtype)
        value_min = int(xp.min(xp.where(valid, table, info.max)))
        value_max = int(xp.max(xp.where(valid, table, info.min)))
        return value_min, value_max, tuple(support)

    def _image_slices(self, image_range: common.UnitRange) -> Optional[tuple[slice, ...]]:
        if (bounds := self._image_bounds) is not None:
            value_min, value_max, support = bounds
            # with the skip value inside the range, skip-only rows would be selected as well
            skip_selected = self.skip_value is not None and self.skip_value in image_range
            if (
                image_range.start <= value_min
                and value_max < image_range.stop
                and not skip_selected
            ):
                return support
        return _hyperslice(self._index_table, image_range, self.array_ns, self.skip_value)

    def restrict(self, index: common.AnyIndexSpec) -> NdArrayConnectivityField:
        cache_key = (id(self.ndarray), self.domain, index)

        if (restricted_connectivity := self._cache.get(cache_key, None)) is None:
            new_domain, buffer_slice = self._slice(index)
            restricted_connectivity = self._restrict_buffer(new_domain, buffer_slice)
            self._cache[cache_key] = restricted_connectivity

        return restricted_connectivity

    def _restrict_buffer(
        self, new_domain: common.Domain, buffer_slice: common.RelativeIndexSequence
    ) -> NdArrayConnectivityField:
        cls = self.__class__
        new_buffer = cls.array_ns.asarray(self.ndarray[buffer_slice])
        return cls(new_domain, new_buffer, self.codomain, self.skip_value)

    __getitem__ = restrict


def _domain_premap(data: NdArrayField, *connectivities: common.Connectivity) -> NdArrayField:
    """`premap` implementation transforming only the field domain not the data (i.e. translation and relocation)."""
    new_domain = data.domain
    for connectivity in connectivities:
        dim = connectivity.codomain
        dim_idx = data.domain.dim_index(dim)
        if dim_idx is None:
            raise ValueError(
                f"Incompatible index field expects a data field with dimension '{dim}'"
                f"but got '{data.domain}'."
            )

        current_range: common.UnitRange = data.domain[dim_idx].unit_range
        new_ranges = connectivity.inverse_image(current_range)
        new_domain = new_domain.replace(dim_idx, *new_ranges)

    return data.__class__.from_array(data._ndarray, domain=new_domain, dtype=data.dtype)


def _gather_output_domain(
    field_domain: common.Domain, connectivities: Sequence[common.GatherConnectivity]
) -> common.Domain:
    """Output domain of a simultaneous gather: each codomain is replaced by the dimensions of its
    connectivity's domain; dimensions shared with the field domain are intersected in place."""
    domain = field_domain
    for conn in connectivities:
        cod = conn.codomain
        # the connectivity's domain, restricted to where it maps into the codomain's range
        narrowed = {
            nr.dim: nr.unit_range for nr in conn.inverse_image(field_domain[cod].unit_range)
        }
        # dimensions the connectivity adds that are not in the field yet
        introduced = [
            common.NamedRange(dim, rng) for dim, rng in narrowed.items() if dim not in domain.dims
        ]
        result: list[common.NamedRange] = []
        for nr in domain:
            if nr.dim == cod:  # the codomain expands into the connectivity's domain dimensions
                if cod in narrowed:  # keep the codomain itself when it maps to itself
                    result.append(common.NamedRange(cod, nr.unit_range & narrowed[cod]))
                result.extend(introduced)
            elif nr.dim in narrowed:  # a dimension shared with the connectivity: narrow it
                result.append(common.NamedRange(nr.dim, nr.unit_range & narrowed[nr.dim]))
            else:
                result.append(nr)
        domain = common.Domain(*result)
    return domain


def _structured_premap(
    data: NdArrayField,
    connectivity: (
        _structured_conn.StructuredConnectivity | _structured_conn._StructuredConnectivityK
    ),
) -> NdArrayField:
    """Dispatch for `StructuredConnectivity` in the embedded path.

    - ``inp(C2E[k])`` → `_StructuredConnectivityK` → k-th neighbor gather.
    - ``inp(C2E)``    → `StructuredConnectivity`  → per-k gather stacked along
      the connectivity's ``local_dim`` so ``neighbor_sum(..., axis=local_dim)``
      can reduce it.
    """
    if isinstance(connectivity, _structured_conn._StructuredConnectivityK):
        return cast(
            NdArrayField,
            _structured_conn.expand_k(data, connectivity.connectivity, connectivity.k),
        )

    return _structured_conn.expand_stacked(data, connectivity)


def _gather_premap(data: NdArrayField, *connectivities: common.GatherConnectivity) -> NdArrayField:
    """`premap` via a single advanced-index gather (dimension-preserving and -introducing cases)."""
    xp = data.array_ns
    new_domain = _gather_output_domain(data.domain, connectivities)
    if len(connectivities) == 1 and (
        set(connectivities[0].domain.dims) & set(data.domain.dims) <= {connectivities[0].codomain}
    ):
        return _row_gather_premap(data, connectivities[0], new_domain)
    conn_by_codomain = {conn.codomain: conn for conn in connectivities}

    # one index array per original field dimension (the connectivity's, or identity), broadcast over
    # the output domain and shifted to 0-based buffer indices, then a single advanced-index gather
    def take_index(dim: common.Dimension) -> core_defs.NDArrayObject:
        start = data.domain[dim].unit_range.start
        if (conn := conn_by_codomain.get(dim)) is None:
            return _identity_index_array(new_domain, dim, xp) - start
        # skip entries read the domain start instead of wrapping around to an arbitrary element;
        # what is gathered there, and its cotangent, is left to the reduction mask to discard
        index = (
            _connectivity_index_array(conn, new_domain, xp, skip_value_replacement=start) - start
        )
        if getattr(conn, "_image_unknown", False):
            # the output domain assumed all targets inside the field; out-of-range ones read its boundary
            index = xp.clip(index, 0, len(data.domain[dim].unit_range) - 1)
        return index

    new_buffer = data._ndarray[tuple(take_index(dim) for dim in data.domain.dims)]
    return data.__class__.from_array(new_buffer, domain=new_domain, dtype=data.dtype)


def _row_gather_premap(
    data: NdArrayField, connectivity: common.GatherConnectivity, new_domain: common.Domain
) -> NdArrayField:
    """Gather whole rows along the connectivity's codomain, the other dimensions sliced unchanged."""
    xp = data.array_ns
    codomain = connectivity.codomain
    codomain_axis = data.domain.dim_index(codomain, allow_missing=False)
    start = data.domain[codomain].unit_range.start

    index = (
        _connectivity_index_array(
            connectivity,
            common.Domain(*(new_domain[d] for d in connectivity.domain.dims)),
            xp,
            skip_value_replacement=start,
        )
        - start
    )
    if getattr(connectivity, "_image_unknown", False):
        index = xp.clip(index, 0, len(data.domain[codomain].unit_range) - 1)

    other_slices = tuple(
        slice(None)
        if dim == codomain
        else slice(
            new_domain[dim].unit_range.start - data.domain[dim].unit_range.start,
            new_domain[dim].unit_range.stop - data.domain[dim].unit_range.start,
        )
        for dim in data.domain.dims
    )
    # a single index array keeps its dimensions in place of the codomain axis
    gathered = data._ndarray[other_slices][(slice(None),) * codomain_axis + (index,)]
    gathered_dims = [
        *data.domain.dims[:codomain_axis],
        *connectivity.domain.dims,
        *data.domain.dims[codomain_axis + 1 :],
    ]
    new_buffer = xp.transpose(gathered, [gathered_dims.index(d) for d in new_domain.dims])
    return data.__class__.from_array(new_buffer, domain=new_domain, dtype=data.dtype)


def _connectivity_index_array(
    connectivity: common.GatherConnectivity,
    domain: common.Domain,
    xp: ModuleType,
    *,
    skip_value_replacement: Optional[core_defs.IntegralScalar] = None,
) -> core_defs.NDArrayObject:
    """`connectivity`'s index table laid out over `domain` (not yet shifted to 0-based)."""
    # restrict the table to the output ranges of the connectivity's own dimensions
    sub_domain = common.Domain(*(domain[d] for d in connectivity.domain.dims))
    conn = connectivity if sub_domain == connectivity.domain else connectivity.restrict(sub_domain)
    arr = xp.asarray(conn.ndarray)
    if skip_value_replacement is not None and conn.skip_value is not None:
        arr = xp.where(arr == conn.skip_value, skip_value_replacement, arr)
    # the axis of `arr` for each output dimension: the connectivity's own axis, or a fresh appended one
    ndim = conn.domain.ndim
    fresh_axis = {
        dim: ndim + i for i, dim in enumerate(d for d in domain.dims if d not in conn.domain.dims)
    }
    transposed_axes = tuple(
        fresh_axis[dim] if dim in fresh_axis else conn.domain.dim_index(dim, allow_missing=False)
        for dim in domain.dims
    )
    if fresh_axis:  # add size-1 axes for output dimensions the connectivity does not span
        arr = xp.expand_dims(arr, axis=tuple(fresh_axis.values()))
    arr = xp.transpose(arr, transposed_axes)  # reorder to the output dimension order
    if arr.shape != domain.shape:
        arr = xp.broadcast_to(arr, domain.shape)  # broadcast the size-1 axes to the full shape
    return arr


def _identity_index_array(
    domain: common.Domain,
    dim: common.Dimension,
    xp: ModuleType,
    dtype: Optional[npt.DTypeLike] = None,
) -> core_defs.NDArrayObject:
    """Index array selecting `dim` unchanged over `domain`, in the field's index space."""
    d_idx = domain.dim_index(dim, allow_missing=False)
    unit_range = domain[d_idx].unit_range
    indices = xp.arange(unit_range.start, unit_range.stop, dtype=dtype)
    shape = tuple(len(indices) if i == d_idx else 1 for i in range(len(domain)))
    return xp.broadcast_to(xp.reshape(indices, shape), domain.shape)


def _hyperslice(
    index_array: core_defs.NDArrayObject,
    image_range: common.UnitRange,
    xp: ModuleType,
    skip_value: Optional[core_defs.IntegralScalar] = None,
) -> Optional[tuple[slice, ...]]:
    """
    Return the hypercube slice that contains all indices in `index_array` that are within `image_range`, or `None` if no such hypercube exists.

    If `skip_value` is given, the selected values are ignored. It returns the smallest hypercube.
    A bigger hypercube could be constructed by adding lines that contain only `skip_value`s.

    Example:
        index_array =  0  1 -1
                       3  4 -1
                      -1 -1 -1
        skip_value = -1

        would currently select the 2x2 range [0,2], [0,2], but could also select the 3x3 range [0,3], [0,3].
    """
    select_mask = (index_array >= image_range.start) & (index_array < image_range.stop)

    if not xp.any(select_mask):
        return None

    nnz: tuple[core_defs.NDArrayObject, ...] = xp.nonzero(select_mask)

    slices = tuple(
        slice(xp.min(dim_nnz_indices).item(), xp.max(dim_nnz_indices).item() + 1)
        for dim_nnz_indices in nnz
    )
    hcube = select_mask[tuple(slices)]
    if skip_value is not None:
        ignore_mask = index_array == skip_value
        hcube |= ignore_mask[tuple(slices)]
    if not xp.all(hcube):
        return None

    return slices


# -- Specialized implementations for builtin operations on array fields --

NdArrayField.register_builtin_func(
    fbuiltins.abs,
    NdArrayField.__abs__,
)
NdArrayField.register_builtin_func(
    fbuiltins.power,
    NdArrayField.__pow__,
)

for name in (
    fbuiltins.UNARY_MATH_FP_BUILTIN_NAMES
    + fbuiltins.UNARY_MATH_FP_PREDICATE_BUILTIN_NAMES
    + fbuiltins.UNARY_MATH_NUMBER_BUILTIN_NAMES
):
    if name in ["abs", "power"]:
        continue
    NdArrayField.register_builtin_func(getattr(fbuiltins, name), _make_builtin(name, name))

NdArrayField.register_builtin_func(
    fbuiltins.minimum,
    _make_builtin("minimum", "minimum"),
)
NdArrayField.register_builtin_func(
    fbuiltins.maximum,
    _make_builtin("maximum", "maximum"),
)
NdArrayField.register_builtin_func(
    fbuiltins.fmod,
    _make_builtin("fmod", "fmod"),
)
NdArrayField.register_builtin_func(fbuiltins.where, _make_builtin("where", "where"))


def _to_field(
    value: common.Field | core_defs.Scalar, nd_array_field_type: type[NdArrayField]
) -> common.Field:
    # TODO(havogt): this function is only to workaround broadcasting of scalars, once we have a ConstantField, we can broadcast to that directly
    return (
        value
        if isinstance(value, common.Field)
        else nd_array_field_type.from_array(
            nd_array_field_type.array_ns.asarray(value), domain=common.Domain()
        )
    )


def _intersect_fields(
    *fields: common.Field | core_defs.Scalar,
    ignore_dims: Optional[common.Dimension | tuple[common.Dimension, ...]] = None,
) -> tuple[common.Field, ...]:
    # TODO(havogt): this function could be moved to common, but then requires a broadcast implementation for all field implementations;
    # currently blocked, because requiring the `_to_field` function, see comment there.
    nd_array_class = _get_nd_array_class(*fields)
    promoted_dims = common.promote_dims(
        *(f.domain.dims for f in fields if isinstance(f, common.Field))
    )
    broadcasted_fields = [_broadcast(_to_field(f, nd_array_class), promoted_dims) for f in fields]

    intersected_domains = embedded_common.restrict_to_intersection(
        *[f.domain for f in broadcasted_fields], ignore_dims=ignore_dims
    )

    return tuple(
        nd_array_class.from_array(
            f.ndarray[_get_slices_from_domain_slice(f.domain, intersected_domain)],
            domain=intersected_domain,
        )
        for f, intersected_domain in zip(broadcasted_fields, intersected_domains, strict=True)
    )


def _stack_domains(*domains: common.Domain, dim: common.Dimension) -> common.Domain:
    if not domains:
        return common.Domain()
    dim_start = domains[0][dim].unit_range.start
    dim_stop = domains[-1][dim].unit_range.stop
    return domains[0].replace(dim, common.NamedRange(dim, common.UnitRange(dim_start, dim_stop)))


def _concat(*fields: common.Field, dim: common.Dimension) -> common.Field:
    # TODO(havogt): this function could be extended to a general concat
    # currently only concatenate along the given dimension
    sorted_fields = sorted(fields, key=lambda f: f.domain[dim].unit_range.start)

    for prev, curr in itertools.pairwise(sorted_fields):
        left = prev.domain[dim].unit_range.stop
        right = curr.domain[dim].unit_range.start
        if left > right:
            raise ValueError("Fields to concatenate must not overlap.")
        if left < right:
            raise embedded_exceptions.NonContiguousDomain(f"Cannot concatenate fields along {dim}.")
    new_domain = _stack_domains(*[f.domain for f in sorted_fields], dim=dim)
    nd_array_class = _get_nd_array_class(*sorted_fields)
    return nd_array_class.from_array(
        nd_array_class.array_ns.concatenate(
            [
                nd_array_class.array_ns.broadcast_to(f.ndarray, f.domain.shape)
                for f in sorted_fields
            ],
            axis=new_domain.dim_index(dim, allow_missing=False),
        ),
        domain=new_domain,
    )


def _invert_domain(domain: common.Domain) -> tuple[common.Domain, ...]:
    assert domain.ndim == 1
    dim = domain.dims[0]
    rng = domain.ranges[0]

    if rng.is_empty():
        return (common.Domain(dims=(dim,), ranges=(common.UnitRange.infinite(),)),)

    result = []
    if rng.start is not common.Infinity.NEGATIVE:
        result.append(
            common.Domain(
                dims=(dim,), ranges=(common.UnitRange(common.Infinity.NEGATIVE, rng.start),)
            )
        )
    if rng.stop is not common.Infinity.POSITIVE:
        result.append(
            common.Domain(
                dims=(dim,), ranges=(common.UnitRange(rng.stop, common.Infinity.POSITIVE),)
            )
        )
    return tuple(result)


def _size0_field(
    nd_array_class: type[NdArrayField], dims: tuple[common.Dimension, ...], dtype: core_defs.DType
) -> NdArrayField:
    return nd_array_class.from_array(
        nd_array_class.array_ns.empty(
            (0,) * len(dims), dtype=nd_array_class.array_ns.dtype(dtype.scalar_type)
        ),
        domain=common.Domain(dims=dims, ranges=(common.UnitRange(0, 0),) * len(dims)),
    )


def _bound_dim(field: common.Field, named_range: common.NamedRange) -> common.Field:
    nd_array_class = _get_nd_array_class(field)
    new_domain = field.domain.replace(named_range.dim, named_range)
    return nd_array_class.from_array(
        nd_array_class.array_ns.broadcast_to(field.ndarray, new_domain.shape), domain=new_domain
    )


def _concat_where_slices(
    domain: common.Domain, true_field: common.Field, false_field: common.Field
) -> tuple[tuple[common.Field, ...], tuple[common.Field, ...]]:
    true_domain = embedded_common.domain_intersection(true_field.domain, domain)
    t_slices = () if true_domain.is_empty() else (true_field[true_domain],)
    false_domains = tuple(
        intersection
        for d in _invert_domain(domain)
        if not (
            intersection := embedded_common.domain_intersection(false_field.domain, d)
        ).is_empty()
    )
    return t_slices, tuple(false_field[d] for d in false_domains)


def _concat_where(
    domain: common.Domain,
    true_field: common.Field,
    false_field: common.Field,
) -> common.Field:
    if domain.ndim != 1:
        raise NotImplementedError(
            "'concat_where': Can only concatenate fields with a 1-dimensional domain."
        )
    domain_dim = domain.dims[0]

    # intersect the field in dimensions orthogonal to the domain, then all slices in the domain field have same domain
    t_broadcasted, f_broadcasted = _intersect_fields(
        true_field, false_field, ignore_dims=domain_dim
    )

    t_slices, f_slices = _concat_where_slices(domain, t_broadcasted, f_broadcasted)
    if not all(
        common.UnitRange.is_finite(s.domain[domain_dim].unit_range) for s in (*t_slices, *f_slices)
    ):
        # a branch without extent along `domain_dim` (e.g. a scalar) takes the extent of the other
        t_range, f_range = (f.domain[domain_dim].unit_range for f in (t_broadcasted, f_broadcasted))
        if common.UnitRange.is_finite(f_range):
            t_broadcasted = _bound_dim(t_broadcasted, common.NamedRange(domain_dim, f_range))
        elif common.UnitRange.is_finite(t_range):
            f_broadcasted = _bound_dim(f_broadcasted, common.NamedRange(domain_dim, t_range))
        t_slices, f_slices = _concat_where_slices(domain, t_broadcasted, f_broadcasted)

    if len(t_slices) + len(f_slices) == 0:
        # no data to concatenate, return an empty field
        nd_array_class = _get_nd_array_class(true_field, false_field)
        return _size0_field(nd_array_class, dims=t_broadcasted.domain.dims, dtype=true_field.dtype)
    if (
        config.EMBEDDED_CONCAT_WHERE_WITHOUT_CONCATENATE
        and t_slices
        and f_slices
        and all(
            common.Domain.is_finite(f.domain) and f.ndarray.shape == f.domain.shape
            for f in (t_broadcasted, f_broadcasted)
        )
    ):
        pieces = (*t_slices, *f_slices)
        result_range = common.UnitRange(
            min(p.domain[domain_dim].unit_range.start for p in pieces),
            max(p.domain[domain_dim].unit_range.stop for p in pieces),
        )
        if all(
            f.domain[domain_dim].unit_range.start <= result_range.start
            and result_range.stop <= f.domain[domain_dim].unit_range.stop
            for f in (t_broadcasted, f_broadcasted)
        ):
            return _where_on_range(
                cast(NdArrayField, t_broadcasted),
                cast(NdArrayField, f_broadcasted),
                domain,
                result_range,
            )
        tiled = sum(len(p.domain[domain_dim].unit_range) for p in pieces) == len(result_range)
        for covering, partial_pieces in ((f_broadcasted, t_slices), (t_broadcasted, f_slices)):
            covering_range = covering.domain[domain_dim].unit_range
            if (
                tiled
                and covering_range.start <= result_range.start
                and (result_range.stop <= covering_range.stop)
            ):
                return _update_on_range(
                    cast(NdArrayField, covering), partial_pieces, domain_dim, result_range
                )
    return _concat(*f_slices, *t_slices, dim=domain_dim)


def _update_on_range(
    covering: NdArrayField,
    pieces: Sequence[common.Field],
    dim: common.Dimension,
    result_range: common.UnitRange,
) -> NdArrayField:
    """`concat_where` where one branch covers `result_range`: write the other branch's pieces into it."""
    result_domain = covering.domain.replace(dim, common.NamedRange(dim, result_range))
    xp = covering.array_ns
    result = covering.__class__.from_array(
        xp.array(covering[result_domain].ndarray, copy=True), domain=result_domain
    )
    for piece in pieces:
        result[piece.domain] = piece
    return result


def _where_on_range(
    true_field: NdArrayField,
    false_field: NdArrayField,
    condition: common.Domain,
    result_range: common.UnitRange,
) -> NdArrayField:
    """`concat_where` for branches that both cover `result_range`: a select instead of a concatenation."""
    (dim,) = condition.dims
    result_domain = true_field.domain.replace(dim, common.NamedRange(dim, result_range))
    xp = true_field.array_ns
    axis = result_domain.dim_index(dim, allow_missing=False)
    coords = xp.arange(result_range.start, result_range.stop)
    mask = (coords >= max(condition[dim].unit_range.start, result_range.start)) & (
        coords < min(condition[dim].unit_range.stop, result_range.stop)
    )
    mask = xp.reshape(mask, tuple(-1 if i == axis else 1 for i in range(result_domain.ndim)))
    new_buffer = xp.where(
        mask, true_field[result_domain].ndarray, false_field[result_domain].ndarray
    )
    return true_field.__class__.from_array(new_buffer, domain=result_domain)


NdArrayField.register_builtin_func(experimental.concat_where, _concat_where)  # type: ignore[arg-type]


def _as_offset(offset: fbuiltins.FieldOffset, offset_field: NdArrayField) -> common.Connectivity:
    if not fbuiltins.is_cartesian_offset(offset):
        target_dims = ", ".join(d.value for d in offset.target)
        raise ValueError(
            f"'as_offset' is only supported for Cartesian offsets "
            f"(single target dimension equal to source dimension); "
            f"got source '{offset.source.value}' and target ({target_dims})."
        )
    source_dim = offset.source
    coords = _identity_index_array(
        offset_field.domain, source_dim, offset_field.array_ns, dtype=fbuiltins.IndexType
    )
    return common._connectivity(
        offset_field.ndarray + coords, codomain=source_dim, domain=offset_field.domain
    )


NdArrayField.register_builtin_func(experimental.as_offset, _as_offset)  # type: ignore[arg-type]


def _make_reduction(
    builtin_name: str, array_builtin_name: str, initial_value_op: Callable
) -> Callable[..., NdArrayField[common.DimsT, core_defs.ScalarT]]:
    def _builtin_op(
        field: NdArrayField[common.DimsT, core_defs.ScalarT], axis: common.Dimension
    ) -> NdArrayField[common.DimsT, core_defs.ScalarT]:
        xp = field.array_ns

        if not axis.kind == common.DimensionKind.LOCAL:
            raise ValueError("Can only reduce local dimensions.")
        if axis not in field.domain.dims:
            raise ValueError(f"Field can not be reduced as it doesn't have dimension '{axis}'.")
        if len([d for d in field.domain.dims if d.kind is common.DimensionKind.LOCAL]) > 1:
            raise NotImplementedError(
                "Reducing a field with more than one local dimension is not supported."
            )
        reduce_dim_index = field.domain.dims.index(axis)
        current_offset_provider = embedded_context.get_offset_provider(None)
        assert current_offset_provider is not None
        offset_definition = common.get_offset(
            current_offset_provider, axis.value
        )  # assumes offset and local dimension have same name

        # StructuredConnectivity: no skip_value masking — all colored-Cartesian
        # neighbors are always valid by construction.
        # TODO(havogt): unify reduction dispatch once StructuredConnectivity
        # conforms to the common.Connectivity protocol.
        if isinstance(offset_definition, _structured_conn.StructuredConnectivity):
            new_domain = common.Domain(*[nr for nr in field.domain if nr.dim != axis])
            return field.__class__.from_array(
                getattr(xp, array_builtin_name)(field.ndarray, axis=reduce_dim_index),
                domain=new_domain,
            )

        assert common.is_neighbor_table(offset_definition)
        new_domain = common.Domain(*[nr for nr in field.domain if nr.dim != axis])

        if offset_definition.skip_value is None:
            values = field.ndarray
        else:
            assert isinstance(offset_definition, common.GatherConnectivity)
            table = _connectivity_index_array(offset_definition, field.domain, xp)
            values = xp.where(
                table != offset_definition.skip_value, field.ndarray, initial_value_op(field)
            )

        return field.__class__.from_array(
            getattr(xp, array_builtin_name)(values, axis=reduce_dim_index), domain=new_domain
        )

    _builtin_op.__name__ = builtin_name
    return _builtin_op


NdArrayField.register_builtin_func(
    fbuiltins.neighbor_sum, _make_reduction("neighbor_sum", "sum", lambda x: x.dtype.scalar_type(0))
)
NdArrayField.register_builtin_func(
    fbuiltins.max_over, _make_reduction("max_over", "max", lambda x: x.array_ns.min(x._ndarray))
)
NdArrayField.register_builtin_func(
    fbuiltins.min_over, _make_reduction("min_over", "min", lambda x: x.array_ns.max(x._ndarray))
)


# -- torch.compile support --
_torch_compile_entry_points: weakref.WeakValueDictionary[int, Callable[..., Any]] = (
    weakref.WeakValueDictionary()
)
_within_torch_nonstrict_call = False


def register_torch_compile_entry_point(entry_point: Callable[..., Any]) -> None:
    # Dynamo applies mutations of Python state only after tracing, so the entry point must be
    # registered before 'torch.compile' traces a call to it.
    if torch is not None:
        _torch_compile_entry_points[id(entry_point)] = entry_point


def is_torch_compiling() -> bool:
    return torch is not None and not _within_torch_nonstrict_call and torch.compiler.is_compiling()


_ENCODED_DIMENSION = "__gt4py_dimension__"


def _encode_dimensions(value: Any) -> Any:
    # 'Dimension's (also as dict keys, e.g. in 'domain') are no constants for Dynamo
    if isinstance(value, common.Dimension):
        return (_ENCODED_DIMENSION, value.value, value.kind)
    if isinstance(value, dict):
        return {_encode_dimensions(k): _encode_dimensions(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)) and not isinstance(value, common.Field):
        return type(value)(_encode_dimensions(v) for v in value)
    return value


def _decode_dimensions(value: Any) -> Any:
    if (
        isinstance(value, tuple)
        and len(value) == 3
        and isinstance(value[0], str)
        and value[0] == _ENCODED_DIMENSION
    ):
        return common.Dimension(value[1], value[2])
    if isinstance(value, dict):
        return {_decode_dimensions(k): _decode_dimensions(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)) and not isinstance(value, common.Field):
        return type(value)(_decode_dimensions(v) for v in value)
    return value


def torch_compile_call(entry_point: Callable[..., Any], args: tuple, kwargs: dict[str, Any]) -> Any:
    """Call an embedded entry point from a 'torch.compile' region, tracing it non-strictly."""
    result = _torch_nonstrict_call(
        id(entry_point), _encode_dimensions(args), _encode_dimensions(kwargs)
    )
    return result[0] if result else None


# -- Concrete array implementations --
# NumPy
_nd_array_implementations = [np]


@dataclasses.dataclass(frozen=True, eq=False)
class NumPyArrayField(NdArrayField):
    array_ns: ClassVar[ModuleType] = np


common._field.register(np.ndarray, NumPyArrayField.from_array)


@dataclasses.dataclass(frozen=True, eq=False)
class NumPyArrayConnectivityField(NdArrayConnectivityField):
    array_ns: ClassVar[ModuleType] = np


common._connectivity.register(np.ndarray, NumPyArrayConnectivityField.from_array)

# CuPy
if cp:
    _nd_array_implementations.append(cp)

    @dataclasses.dataclass(frozen=True, eq=False)
    class CuPyArrayField(NdArrayField):
        array_ns: ClassVar[ModuleType] = cp

        def _as_assignable(self, value: Any) -> cp.ndarray:
            return cp.asarray(value)

    common._field.register(cp.ndarray, CuPyArrayField.from_array)

    @dataclasses.dataclass(frozen=True, eq=False)
    class CuPyArrayConnectivityField(NdArrayConnectivityField):
        array_ns: ClassVar[ModuleType] = cp

    common._connectivity.register(cp.ndarray, CuPyArrayConnectivityField.from_array)

# JAX
if jnp:
    assert jax is not None

    _nd_array_implementations.append(jnp)
    # TODO(havogt): we currently enable 64-bit support by default, but we might want to make this configurable via the GT4Py config
    jax.config.update("jax_enable_x64", True)

    @dataclasses.dataclass(frozen=True, eq=False)
    class JaxArrayField(NdArrayField):
        array_ns: ClassVar[ModuleType] = jnp

        @property
        def __gt_buffer_info__(self) -> common.BufferInfo:
            raise NotImplementedError("'__gt_buffer_info__' for JaxArrayField not yet implemented.")

        def __setitem__(
            self,
            index: common.AnyIndexSpec,
            value: common.Field | core_defs.NDArrayObject | core_defs.ScalarT,
        ) -> None:
            target_domain, target_slice = self._slice(index)

            if isinstance(value, common.Field):
                if not value.domain == target_domain:
                    raise ValueError(
                        f"Incompatible 'Domain' in assignment. Source domain = '{value.domain}', target domain = '{target_domain}'."
                    )
                value = value.ndarray

            object.__setattr__(self, "_ndarray", self._ndarray.at[target_slice].set(value))  # type: ignore[attr-defined] # `NDArrayObject` typing is not complete

    class _TableHandle:
        """Trace-time reference to a connectivity table, compared by buffer identity."""

        __slots__ = ("table",)

        def __init__(self, table: core_defs.NDArrayObject) -> None:
            self.table = table

        def __eq__(self, other: object) -> bool:
            return isinstance(other, _TableHandle) and self.table is other.table

        def __hash__(self) -> int:
            return id(self.table)

    _JaxConnectivityAuxData: TypeAlias = tuple[
        common.Domain, common.Dimension, Optional[core_defs.IntegralScalar], Optional[_TableHandle]
    ]

    @dataclasses.dataclass(frozen=True, eq=False)
    class JaxArrayConnectivityField(NdArrayConnectivityField):
        array_ns: ClassVar[ModuleType] = jnp
        _table_handle: Optional[_TableHandle] = None

        @property
        def _index_table(self) -> core_defs.NDArrayObject:
            if self._table_handle is not None and isinstance(self._ndarray, jax.core.Tracer):
                return self._table_handle.table
            return self._ndarray

        @property
        def _image_unknown(self) -> bool:
            return self._table_handle is None and isinstance(self._ndarray, jax.core.Tracer)

        def _image_slices(self, image_range: common.UnitRange) -> Optional[tuple[slice, ...]]:
            if self._image_unknown:
                # values computed inside a trace: assume every target lies in `image_range`
                return tuple(slice(0, n) for n in self._ndarray.shape)
            with jax.ensure_compile_time_eval():
                return super()._image_slices(image_range)

        def _restrict_buffer(
            self, new_domain: common.Domain, buffer_slice: common.RelativeIndexSequence
        ) -> JaxArrayConnectivityField:
            handle = None
            if isinstance(self._ndarray, jax.core.Tracer):
                if self._table_handle is not None:
                    with jax.ensure_compile_time_eval():
                        handle = _TableHandle(self._table_handle.table[buffer_slice])
                new_buffer = jnp.asarray(self._ndarray[buffer_slice])
            else:
                # a concrete table must stay concrete under a trace: domain inference reads it
                with jax.ensure_compile_time_eval():
                    new_buffer = jnp.asarray(self._ndarray[buffer_slice])
            return JaxArrayConnectivityField(
                new_domain,
                new_buffer,
                self.codomain,
                self.skip_value,
                handle,
            )

    common._field.register(jnp.ndarray, JaxArrayField.from_array)
    common._connectivity.register(jnp.ndarray, JaxArrayConnectivityField.from_array)
    # jax >= 0.11: 'Tracer' is no longer a subclass of 'jax.Array' (only 'isinstance' says so, via
    # 'ArrayMeta.__instancecheck__'), and 'singledispatch' resolves on the class hierarchy.
    common._field.register(jax.core.Tracer, JaxArrayField.from_array)
    common._connectivity.register(jax.core.Tracer, JaxArrayConnectivityField.from_array)

    def _flatten_jax_field(
        field: JaxArrayField,
    ) -> tuple[tuple[core_defs.NDArrayObject], common.Domain]:
        return (field.ndarray,), field.domain

    def _unflatten_jax_field(
        domain: common.Domain, children: tuple[core_defs.NDArrayObject]
    ) -> JaxArrayField:
        return JaxArrayField(domain, children[0])  # type: ignore[abstract] # mypy does not see '__gt_builtin_func__' as implemented by 'FieldBuiltinFuncRegistry'

    jax.tree_util.register_pytree_node(
        JaxArrayField,  # type: ignore[type-abstract, unused-ignore] # only reported when 'jax' is installed, see '_unflatten_jax_field'
        _flatten_jax_field,
        _unflatten_jax_field,
    )

    def _flatten_jax_connectivity(
        connectivity: JaxArrayConnectivityField,
    ) -> tuple[tuple[core_defs.NDArrayObject], _JaxConnectivityAuxData]:
        table = connectivity._ndarray
        handle = (
            connectivity._table_handle
            if isinstance(table, jax.core.Tracer)
            else _TableHandle(table)
        )
        return (table,), (
            connectivity.domain,
            connectivity.codomain,
            connectivity.skip_value,
            handle,
        )

    def _unflatten_jax_connectivity(
        aux_data: _JaxConnectivityAuxData, children: tuple[core_defs.NDArrayObject]
    ) -> JaxArrayConnectivityField:
        domain, codomain, skip_value, handle = aux_data
        return JaxArrayConnectivityField(domain, children[0], codomain, skip_value, handle)

    jax.tree_util.register_pytree_node(
        JaxArrayConnectivityField,  # type: ignore[type-abstract, unused-ignore] # see '_unflatten_jax_field'
        _flatten_jax_connectivity,
        _unflatten_jax_connectivity,
    )

# PyTorch
if torch:
    import array_api_compat.torch as _torch_compat

    _NUMPY_TO_TORCH_DTYPE: dict[type, torch.dtype] = {
        np.bool_: torch.bool,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.uint8: torch.uint8,
        np.uint16: torch.uint16,
        np.uint32: torch.uint32,
        np.uint64: torch.uint64,
        np.float32: torch.float32,
        np.float64: torch.float64,
    }
    _TORCH_TO_NUMPY_DTYPE: dict[torch.dtype, type] = {
        v: k for k, v in _NUMPY_TO_TORCH_DTYPE.items()
    }

    def _to_torch_dtype(dtype: Any) -> Optional[torch.dtype]:
        if dtype is None or isinstance(dtype, torch.dtype):
            return dtype
        return _NUMPY_TO_TORCH_DTYPE[np.dtype(dtype).type]

    def _torch_cbrt(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.abs(x) ** (1.0 / 3.0)

    def _torch_gamma(x: torch.Tensor) -> torch.Tensor:
        # 'torch' only provides 'lgamma' (log of the absolute value): gamma is negative
        # for negative arguments with an odd floor
        sign = torch.where((x < 0) & (torch.floor(x) % 2 == 1), -1.0, 1.0).to(x.dtype)
        return sign * torch.exp(torch.lgamma(x))

    def _with_tensor_operands(func: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        # these 'torch' functions reject Python and NumPy scalar operands
        def wrapper(*args: Any) -> torch.Tensor:
            device = next((a.device for a in args if isinstance(a, torch.Tensor)), None)
            return func(
                *(
                    a if isinstance(a, torch.Tensor) else torch.as_tensor(a, device=device)
                    for a in args
                )
            )

        return wrapper

    class _TorchNamespace(ModuleType):
        """
        NumPy-like namespace for 'torch' tensors on one device kind.

        Based on 'array_api_compat.torch', with the NumPy names used by 'NdArrayField'.
        Array creation functions default to the namespace device (the current device for CUDA),
        and accept NumPy dtypes.
        """

        def __init__(self, device_type: str) -> None:
            super().__init__(f"torch_{device_type}")
            self.device_type = device_type

        def __getattr__(self, name: str) -> Any:
            return getattr(_torch_compat, name)

        @property
        def default_device(self) -> torch.device:
            if self.device_type == "cuda":
                return torch.device("cuda", torch.cuda.current_device())
            return torch.device(self.device_type)

        def _device(self, obj: Any, device: Any) -> Any:
            if device is not None:
                return device
            if isinstance(obj, torch.Tensor) and obj.device.type == self.device_type:
                return obj.device
            return self.default_device

        newaxis = None

        @staticmethod
        def dtype(dtype: Any) -> torch.dtype:
            result = _to_torch_dtype(dtype)
            assert result is not None
            return result

        def asarray(
            self, obj: Any, /, *, dtype: Any = None, device: Any = None, copy: Optional[bool] = None
        ) -> torch.Tensor:
            return _torch_compat.asarray(
                obj, dtype=_to_torch_dtype(dtype), device=self._device(obj, device), copy=copy
            )

        def array(
            self, obj: Any, /, *, dtype: Any = None, device: Any = None, copy: bool = True
        ) -> torch.Tensor:
            return self.asarray(obj, dtype=dtype, device=device, copy=copy)

        def arange(
            self,
            start: int,
            /,
            stop: Optional[int] = None,
            step: int = 1,
            *,
            dtype: Any = None,
            device: Any = None,
        ) -> torch.Tensor:
            return _torch_compat.arange(
                start,
                stop,
                step,
                dtype=_to_torch_dtype(dtype),
                device=self._device(None, device),
            )

        def empty(
            self, shape: tuple[int, ...], *, dtype: Any = None, device: Any = None
        ) -> torch.Tensor:
            return _torch_compat.empty(
                shape, dtype=_to_torch_dtype(dtype), device=self._device(None, device)
            )

        equal = staticmethod(_with_tensor_operands(_torch_compat.equal))
        not_equal = staticmethod(_with_tensor_operands(_torch_compat.not_equal))
        greater = staticmethod(_with_tensor_operands(_torch_compat.greater))
        greater_equal = staticmethod(_with_tensor_operands(_torch_compat.greater_equal))
        less = staticmethod(_with_tensor_operands(_torch_compat.less))
        less_equal = staticmethod(_with_tensor_operands(_torch_compat.less_equal))
        minimum = staticmethod(_with_tensor_operands(_torch_compat.minimum))
        maximum = staticmethod(_with_tensor_operands(_torch_compat.maximum))
        fmod = staticmethod(_with_tensor_operands(torch.fmod))
        logical_and = staticmethod(_with_tensor_operands(torch.logical_and))
        logical_or = staticmethod(_with_tensor_operands(torch.logical_or))
        logical_xor = staticmethod(_with_tensor_operands(torch.logical_xor))
        power = staticmethod(_torch_compat.pow)
        mod = staticmethod(_torch_compat.remainder)
        invert = staticmethod(_torch_compat.bitwise_invert)
        transpose = staticmethod(_torch_compat.permute_dims)
        cbrt = staticmethod(_torch_cbrt)
        gamma = staticmethod(_torch_gamma)

    _torch_cpu_ns = _TorchNamespace("cpu")
    _torch_cuda_ns = _TorchNamespace("cuda")
    _nd_array_implementations.append(_torch_cpu_ns)
    if torch.cuda.is_available():
        _nd_array_implementations.append(_torch_cuda_ns)

    class _TorchArrayFieldMixin:
        @staticmethod
        def _scalar_type_of(array: core_defs.NDArrayObject) -> type[core_defs.Scalar]:
            return _TORCH_TO_NUMPY_DTYPE[array.dtype]

        @staticmethod
        def _astype(array: core_defs.NDArrayObject, type_: type) -> core_defs.NDArrayObject:
            return array.to(_to_torch_dtype(type_))  # type: ignore[attr-defined] # `NDArrayObject` typing is not complete

        @property
        def __gt_buffer_info__(self) -> common.BufferInfo:
            raise NotImplementedError("'__gt_buffer_info__' for torch fields not yet implemented.")

        def _as_assignable(self, value: Any) -> torch.Tensor:
            return self.array_ns.asarray(value)  # type: ignore[attr-defined] # mixin of 'NdArrayField'

        def asnumpy(self) -> np.ndarray:
            return self._ndarray.detach().cpu().numpy()  # type: ignore[attr-defined] # `NDArrayObject` typing is not complete

    @dataclasses.dataclass(frozen=True, eq=False)
    class TorchArrayField(_TorchArrayFieldMixin, NdArrayField):
        array_ns: ClassVar[ModuleType] = _torch_cpu_ns

    @dataclasses.dataclass(frozen=True, eq=False)
    class TorchCUDAArrayField(_TorchArrayFieldMixin, NdArrayField):
        array_ns: ClassVar[ModuleType] = _torch_cuda_ns

    # Connectivities created outside of 'torch.compile', to read their concrete tables while
    # 'torch.compile' traces with fake tensors.
    _torch_concrete_connectivities: weakref.WeakValueDictionary[int, NdArrayConnectivityField] = (
        weakref.WeakValueDictionary()
    )

    class _TorchConnectivityMixin:
        _concrete_id: Optional[int] = None

        def __post_init__(self) -> None:
            # not the fake and functional tensors of a trace
            if type(self._ndarray) is torch.Tensor:  # type: ignore[attr-defined] # mixin of 'NdArrayConnectivityField'
                _torch_concrete_connectivities[id(self)] = self  # type: ignore[assignment] # mixin of 'NdArrayConnectivityField'

        def _concrete(self) -> Optional[NdArrayConnectivityField]:
            if self._concrete_id is None:
                return None
            return _torch_concrete_connectivities.get(self._concrete_id)

        def inverse_image(self, image_range: common.UnitRange | common.NamedRange) -> common.Domain:
            if (concrete := self._concrete()) is None:
                return super().inverse_image(image_range)  # type: ignore[misc] # mixin of 'NdArrayConnectivityField'
            with torch.utils._python_dispatch._disable_current_modes():
                return concrete.inverse_image(image_range)

        def restrict(self, index: common.AnyIndexSpec) -> NdArrayConnectivityField:
            restricted = super().restrict(index)  # type: ignore[misc] # mixin of 'NdArrayConnectivityField'
            if (concrete := self._concrete()) is not None and restricted._concrete_id is None:
                with torch.utils._python_dispatch._disable_current_modes():
                    object.__setattr__(restricted, "_concrete_id", id(concrete.restrict(index)))
            return restricted

        __getitem__ = restrict

    @dataclasses.dataclass(frozen=True, eq=False)
    class TorchArrayConnectivityField(  # type: ignore[misc] # 'restrict' of the mixin returns a connectivity
        _TorchConnectivityMixin, _TorchArrayFieldMixin, NdArrayConnectivityField
    ):
        array_ns: ClassVar[ModuleType] = _torch_cpu_ns

    @dataclasses.dataclass(frozen=True, eq=False)
    class TorchCUDAArrayConnectivityField(  # type: ignore[misc] # 'restrict' of the mixin returns a connectivity
        _TorchConnectivityMixin, _TorchArrayFieldMixin, NdArrayConnectivityField
    ):
        array_ns: ClassVar[ModuleType] = _torch_cuda_ns

    def _torch_field(data: torch.Tensor, /, **kwargs: Any) -> NdArrayField:
        cls = TorchCUDAArrayField if data.device.type == "cuda" else TorchArrayField
        return cls.from_array(data, **kwargs)

    def _torch_connectivity(
        data: torch.Tensor, /, *args: Any, **kwargs: Any
    ) -> NdArrayConnectivityField:
        cls = (
            TorchCUDAArrayConnectivityField
            if data.device.type == "cuda"
            else TorchArrayConnectivityField
        )
        return cls.from_array(data, *args, **kwargs)

    common._field.register(torch.Tensor, _torch_field)
    common._connectivity.register(torch.Tensor, _torch_connectivity)

    import torch.utils._pytree as _torch_pytree

    # pytree contexts must be plain constants for Dynamo, frozen dataclasses are not accepted
    _DomainContext: TypeAlias = tuple[tuple[str, common.DimensionKind, int, int], ...]

    def _domain_to_context(domain: common.Domain) -> _DomainContext:
        return tuple(
            (dim.value, dim.kind, rng.start, rng.stop)
            for dim, rng in zip(domain.dims, domain.ranges, strict=True)
        )

    def _context_to_domain(context: _DomainContext) -> common.Domain:
        return common.Domain(
            dims=tuple(common.Dimension(value, kind) for value, kind, _, _ in context),
            ranges=tuple(common.UnitRange(start, stop) for _, _, start, stop in context),
        )

    for _field_cls in (TorchArrayField, TorchCUDAArrayField):
        _torch_pytree.register_pytree_node(
            _field_cls,
            lambda field: ([field.ndarray], _domain_to_context(field.domain)),
            lambda children, context, cls=_field_cls: cls(_context_to_domain(context), children[0]),
        )

    def _flatten_torch_connectivity(
        conn: _TorchConnectivityMixin,
    ) -> tuple[list[torch.Tensor], tuple[Any, ...]]:
        assert isinstance(conn, NdArrayConnectivityField)
        context = (
            _domain_to_context(conn.domain),
            conn.codomain.value,
            conn.codomain.kind,
            conn.skip_value,
            # later tracing passes flatten the connectivities unflattened by the first one
            id(conn) if conn._concrete_id is None else conn._concrete_id,
        )
        return [conn.ndarray], context

    def _unflatten_torch_connectivity(
        cls: type[NdArrayConnectivityField], children: list[torch.Tensor], context: tuple[Any, ...]
    ) -> NdArrayConnectivityField:
        domain, codomain_value, codomain_kind, skip_value, concrete_id = context
        conn = cls(
            _context_to_domain(domain),
            children[0],
            common.Dimension(codomain_value, codomain_kind),
            skip_value,
        )
        object.__setattr__(conn, "_concrete_id", concrete_id)
        return conn

    for _connectivity_cls in (TorchArrayConnectivityField, TorchCUDAArrayConnectivityField):
        _torch_pytree.register_pytree_node(
            _connectivity_cls,
            _flatten_torch_connectivity,
            functools.partial(_unflatten_torch_connectivity, _connectivity_cls),
        )

    @torch.compiler.nonstrict_trace
    def _torch_nonstrict_call(
        entry_point_id: int, args: tuple, kwargs: dict[str, Any]
    ) -> tuple[Any, ...]:
        global _within_torch_nonstrict_call
        _within_torch_nonstrict_call = True
        try:
            result = _torch_compile_entry_points[entry_point_id](
                *_decode_dimensions(args), **_decode_dimensions(kwargs)
            )
        finally:
            _within_torch_nonstrict_call = False
        # a 'None' output is not supported by 'nonstrict_trace'
        return () if result is None else (result,)


def _broadcast(field: common.Field, new_dimensions: Sequence[common.Dimension]) -> common.Field:
    if field.domain.dims == new_dimensions:
        return field
    domain_slice: list[slice | None] = []
    named_ranges = []
    for dim in new_dimensions:
        if (pos := embedded_common._find_index_of_dim(dim, field.domain)) is not None:
            domain_slice.append(slice(None))
            named_ranges.append(common.NamedRange(dim, field.domain[pos].unit_range))
        else:
            domain_slice.append(None)  # np.newaxis
            named_ranges.append(common.NamedRange(dim, common.UnitRange.infinite()))
    return common._field(field.ndarray[tuple(domain_slice)], domain=common.Domain(*named_ranges))


def _builtins_broadcast(
    field: common.Field | core_defs.Scalar, new_dimensions: tuple[common.Dimension, ...]
) -> common.Field:  # separated for typing reasons
    if isinstance(field, common.Field):
        return _broadcast(field, new_dimensions)
    raise AssertionError("Scalar case not reachable from 'fbuiltins.broadcast'.")


NdArrayField.register_builtin_func(fbuiltins.broadcast, _builtins_broadcast)


def _astype(field: common.Field | core_defs.ScalarT | tuple, type_: type) -> NdArrayField:
    if isinstance(field, NdArrayField):
        return field.__class__.from_array(field._astype(field.ndarray, type_), domain=field.domain)
    raise AssertionError("This is the NdArrayField implementation of 'fbuiltins.astype'.")


NdArrayField.register_builtin_func(fbuiltins.astype, _astype)  # type: ignore[arg-type]  # because fbuiltins.astype is overloaded


def _get_slices_from_domain_slice(
    domain: common.Domain,
    domain_slice: common.Domain | Sequence[common.NamedRange | common.NamedIndex],
) -> common.RelativeIndexSequence:
    """Generate slices for sub-array extraction based on named ranges or named indices within a Domain.

    This function generates a tuple of slices that can be used to extract sub-arrays from a field. The provided
    named ranges or indices specify the dimensions and ranges of the sub-arrays to be extracted.

    Args:
        domain (common.Domain): The Domain object representing the original field.
        domain_slice (DomainSlice): A sequence of dimension names and associated ranges.

    Returns:
        tuple[slice | int | None, ...]: A tuple of slices representing the sub-array extraction along each dimension
                                       specified in the Domain. If a dimension is not included in the named indices
                                       or ranges, a None is used to indicate expansion along that axis.
    """
    slice_indices: list[slice | common.IntIndex] = []

    for pos_old, (dim, _) in enumerate(domain):
        if (pos := embedded_common._find_index_of_dim(dim, domain_slice)) is not None:
            _, index_or_range = domain_slice[pos]
            slice_indices.append(_compute_slice(index_or_range, domain, pos_old))
        else:
            slice_indices.append(slice(None))
    return tuple(slice_indices)


def _compute_slice(
    rng: common.UnitRange | common.IntIndex, domain: common.Domain, pos: int
) -> slice | common.IntIndex:
    """Compute a slice or integer based on the provided range, domain, and position.

    Args:
        rng (DomainRange): The range to be computed as a slice or integer.
        domain (common.Domain): The domain containing dimension information.
        pos (int): The position of the dimension in the domain.

    Returns:
        slice | int: Slice if `new_rng` is a UnitRange, otherwise an integer.

    Raises:
        ValueError: If `new_rng` is not an integer or a UnitRange.
    """
    if isinstance(rng, common.UnitRange):
        start = (
            rng.start - domain.ranges[pos].start
            if common.UnitRange.is_left_finite(domain.ranges[pos])
            else None
        )
        stop = (
            rng.stop - domain.ranges[pos].start
            if common.UnitRange.is_right_finite(domain.ranges[pos])
            else None
        )
        return slice(start, stop)
    elif common.is_int_index(rng):
        assert common.Domain.is_finite(domain)
        return rng - domain.ranges[pos].start
    else:
        raise ValueError(f"Can only use integer or UnitRange ranges, provided type: '{type(rng)}'.")
