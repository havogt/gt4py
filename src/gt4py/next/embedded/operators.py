# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
from typing import Any, Callable, Generic, Iterator, Optional, ParamSpec, Sequence, TypeVar, cast

from gt4py._core import definitions as core_defs
from gt4py.eve import extended_typing as xtyping
from gt4py.next import common, errors, field_utils, named_collections, utils
from gt4py.next.embedded import common as embedded_common, context as embedded_context
from gt4py.next.field_utils import get_array_ns
from gt4py.next.otf import arguments
from gt4py.next.type_system import type_info, type_specifications as ts, type_translation


_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclasses.dataclass(frozen=True)
class EmbeddedOperator(Generic[_R, _P]):
    fun: Callable[_P, _R]

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        return self.fun(*args, **kwargs)


@dataclasses.dataclass(frozen=True)
class ScanOperator(EmbeddedOperator[xtyping.MaybeNestedInTuple[core_defs.ScalarT], _P]):
    """
    Embedded execution of a scan.

    The scan pass is called once per level on the slices of the arguments orthogonal to the
    scan dimension, with JAX arrays inside `jax.lax.scan`. A scan pass that branches on values
    (`per_column`) can only be called on scalars, so it is called per column and level.
    """

    forward: bool
    init: xtyping.MaybeNestedInTuple[core_defs.ScalarT | common.Field]
    range: common.NamedRange
    per_column: bool = False

    def __call__(  # type: ignore[override]
        self,
        *args: common.Field | core_defs.Scalar,
        **kwargs: common.Field | core_defs.Scalar,
    ) -> (
        common.Field[Any, core_defs.ScalarT]
        | tuple[common.Field[Any, core_defs.ScalarT] | tuple, ...]
    ):
        scan_range = self.range
        scan_axis = scan_range.dim
        all_args = [*args, *kwargs.values()]
        domain_intersection = _intersect_scan_args(*all_args, self.init)
        non_scan_domain = common.Domain(*[nr for nr in domain_intersection if nr.dim != scan_axis])

        out_domain = common.Domain(
            *[scan_range if nr.dim == scan_axis else nr for nr in domain_intersection]
        )
        if scan_axis not in out_domain.dims:
            # even if the scan dimension is not in the input, we can scan over it
            out_domain = common.Domain(*out_domain, (scan_range))

        xp = get_array_ns(*(arguments.extract(arg) for arg in [*all_args, self.init]))
        init_type = type_info.tree_map_type(
            lambda t: t.dtype if isinstance(t, ts.FieldType) else t
        )(type_translation.from_value(self.init))
        assert isinstance(init_type, ts.TupleType | ts.ScalarType | ts.NamedCollectionType)

        if xp.__name__.startswith("jax"):
            if self.per_column:
                raise ValueError(
                    f"Scan pass '{getattr(self.fun, '__name__', self.fun)}' branches on values "
                    "('if' or a conditional expression), which cannot be traced by JAX. "
                    "Use 'where' instead."
                )
            return self._jax_scan(args, kwargs, init_type, non_scan_domain, out_domain)

        res = field_utils.field_from_typespec(init_type, out_domain, xp)

        if not self.per_column:
            acc = self.init
            for k in scan_range.unit_range if self.forward else reversed(scan_range.unit_range):
                level = common.NamedIndex(scan_axis, k)
                acc = self.fun(
                    acc,  # type: ignore[arg-type] # need to express that the first argument is the same type as the return
                    *(_slice_at(level, arg) for arg in args),
                    **{name: _slice_at(level, arg) for name, arg in kwargs.items()},
                )
                level_domain = common.Domain(
                    *non_scan_domain, common.NamedRange(scan_axis, common.UnitRange(k, k + 1))
                )
                _tuple_assign_field(
                    arguments.extract(res),  # type: ignore[arg-type] # `res` is a (tuple of) mutable field(s)
                    utils.tree_map(
                        lambda x: (
                            _broadcast_to(x, level_domain)  # noqa: B023 # used within the iteration
                            if isinstance(x, common.Field)
                            else x
                        )
                    )(arguments.extract(acc)),
                    domain=level_domain,
                )
            return res

        def scan_loop(hpos: Sequence[common.NamedIndex]) -> None:
            acc = cast(xtyping.MaybeNestedInTuple[core_defs.ScalarT], _tuple_at(hpos, self.init))
            for k in scan_range.unit_range if self.forward else reversed(scan_range.unit_range):
                pos = (*hpos, common.NamedIndex(scan_axis, k))
                new_args = [_tuple_at(pos, arg) for arg in args]
                new_kwargs = {k: _tuple_at(pos, v) for k, v in kwargs.items()}
                acc = self.fun(acc, *new_args, **new_kwargs)  # type: ignore[arg-type] # need to express that the first argument is the same type as the return
                # convert custom NamedCollections to plain tuples for assignment
                acc_extracted = arguments.extract(acc)
                res_extracted = arguments.extract(res)
                assert xtyping.is_maybe_nested_in_tuple_of(acc_extracted, core_defs.Scalar)  # type: ignore[arg-type]  # Scalar is a Union
                assert xtyping.is_maybe_nested_in_tuple_of(res_extracted, common.MutableField)  # type: ignore[type-abstract]  # MutableField is abstract/generic
                _tuple_assign_value(pos, res_extracted, acc_extracted)

        if len(non_scan_domain) == 0:
            # if we don't have any dimension orthogonal to scan_axis, we need to do one scan_loop
            scan_loop(())
        else:
            for hpos in embedded_common.iterate_domain(non_scan_domain):
                scan_loop(hpos)

        return res

    def _jax_scan(
        self,
        args: Sequence[Any],
        kwargs: dict[str, Any],
        init_type: ts.TypeSpec,
        non_scan_domain: common.Domain,
        out_domain: common.Domain,
    ) -> Any:
        from jax import lax, numpy as jnp

        scan_axis = self.range.dim
        values = [*args, *kwargs.values()]

        # `lax.scan` iterates over the leading axis of arrays; the leaves of the (possibly named)
        # collections are passed as flat lists and put back in place in the same order.
        def is_scanned(x: Any) -> bool:
            return isinstance(x, common.Field) and scan_axis in x.domain.dims

        def stack(field: common.Field) -> Any:
            field = field[
                common.Domain(*(self.range if nr.dim == scan_axis else nr for nr in field.domain))
            ]
            return jnp.moveaxis(field.ndarray, field.domain.dims.index(scan_axis), 0)

        def level_field(field: common.Field, array: Any) -> common.Field:
            return common._field(
                array, domain=common.Domain(*(nr for nr in field.domain if nr.dim != scan_axis))
            )

        dtypes = [
            type_translation.as_dtype(cast(ts.ScalarType, t)).scalar_type
            for t in type_info.primitive_constituents(init_type)
        ]

        def carry_arrays(value: Any) -> list[Any]:
            return [
                _broadcast_to(x, non_scan_domain).ndarray.astype(dtype)
                if isinstance(x, common.Field)
                else jnp.full(non_scan_domain.shape, x, dtype=dtype)
                for x, dtype in zip(_leaves(value, lambda _: True), dtypes, strict=True)
            ]

        def body(carry: list[Any], level_arrays: list[Any]) -> tuple[list[Any], list[Any]]:
            level_iter = iter(level_arrays)
            level_values = [
                _replace_leaves(
                    value,
                    is_scanned,
                    (level_field(x, next(level_iter)) for x in _leaves(value, is_scanned)),
                )
                for value in values
            ]
            acc = _replace_leaves(
                self.init,
                lambda _: True,
                iter(common._field(array, domain=non_scan_domain) for array in carry),
            )
            new_carry = carry_arrays(
                self.fun(
                    acc,
                    *level_values[: len(args)],
                    **dict(zip(kwargs.keys(), level_values[len(args) :])),
                )
            )
            return new_carry, new_carry

        _, ys = lax.scan(
            body,
            carry_arrays(self.init),
            [stack(x) for value in values for x in _leaves(value, is_scanned)],
            length=len(self.range.unit_range),
            reverse=not self.forward,
        )
        scan_axis_index = out_domain.dims.index(scan_axis)
        return _replace_leaves(
            self.init,
            lambda _: True,
            iter(common._field(jnp.moveaxis(y, 0, scan_axis_index), domain=out_domain) for y in ys),
        )


def _leaves(value: Any, predicate: Callable[[Any], bool]) -> list[Any]:
    found: list[Any] = []
    named_collections.tree_map_named_collection(
        lambda x: found.append(x) if predicate(x) else None
    )(value)
    return found


def _replace_leaves(value: Any, predicate: Callable[[Any], bool], new: Iterator[Any]) -> Any:
    return named_collections.tree_map_named_collection(lambda x: next(new) if predicate(x) else x)(
        value
    )


def _slice_at(level: common.NamedIndex, arg: Any) -> Any:
    @named_collections.tree_map_named_collection
    def impl(x: common.Field | core_defs.Scalar) -> common.Field | core_defs.Scalar:
        return x[level] if isinstance(x, common.Field) and level.dim in x.domain.dims else x

    return impl(arg)


def _broadcast_to(field: common.Field, domain: common.Domain) -> common.Field:
    from gt4py.next.embedded import nd_array_field

    array = nd_array_field._broadcast(field, domain.dims)[domain].ndarray
    return common._field(get_array_ns(field).broadcast_to(array, domain.shape), domain=domain)


def field_operator_call(op: EmbeddedOperator[_R, _P], args: Any, kwargs: Any) -> Optional[_R]:
    if "out" in kwargs:
        # called from program or direct field_operator as program
        new_context_kwargs = {}
        if embedded_context.within_valid_context():
            # called from program
            assert "offset_provider" not in kwargs
        else:
            # field_operator as program
            if "offset_provider" not in kwargs:
                raise errors.MissingArgumentError(None, "offset_provider", True)
            offset_provider = kwargs.pop("offset_provider", None)

            new_context_kwargs["offset_provider"] = offset_provider

        out = kwargs.pop("out")

        domain = kwargs.pop("domain", None)

        # TODO(havogt): To do the assignment of the resulting fields we extract containers and act on plain tuples.
        # We currently apply the extract on both the rhs (`res`) computed by the operator and the lhs (`out`, provided by the user)
        # without checking if the types are consistent. However, these errors are caught in linting if enabled.
        container_extracted_out = arguments.extract(out)
        assert xtyping.is_maybe_nested_in_tuple_of(container_extracted_out, common.MutableField)  # type: ignore[type-abstract]  # MutableField is abstract/generic
        out_domain = (
            utils.tree_map(common.domain)(domain)
            if domain is not None
            else utils.tree_map(lambda f: f.domain)(container_extracted_out)
        )

        with embedded_context.update(**new_context_kwargs):
            res = op(*args, **kwargs)
        container_extracted_res = arguments.extract(res)  # type: ignore[arg-type] # TODO(havogt): see notes above
        _tuple_assign_field(container_extracted_out, container_extracted_res, domain=out_domain)  # type: ignore[arg-type]
        return None
    elif "domain" in kwargs and not embedded_context.within_valid_context():
        # direct field_operator call returning the result on `domain`
        if "offset_provider" not in kwargs:
            raise errors.MissingArgumentError(None, "offset_provider", True)
        offset_provider = kwargs.pop("offset_provider")
        domain = utils.tree_map(common.domain)(kwargs.pop("domain"))

        with embedded_context.update(offset_provider=offset_provider):
            full_res: Any = op(*args, **kwargs)

        xp = get_array_ns(*(arguments.extract(arg) for arg in [*args, *kwargs.values()]))
        container_extracted_res = arguments.extract(full_res)
        if not isinstance(domain, tuple):
            domain = utils.tree_map(lambda _: domain)(container_extracted_res)
        result: Any = utils.tree_map(lambda source, dom: _field_on_domain(source, dom, xp))(
            container_extracted_res, domain
        )
        if named_collections.is_named_collection_type(type(full_res)):
            return named_collections.make_named_collection_constructor(type(full_res))(result)  # type: ignore[arg-type]
        return result
    else:
        # called from other field_operator or missing `out` argument
        if "offset_provider" in kwargs:
            # assuming we wanted to call the field_operator as program, otherwise `offset_provider` would not be there
            raise errors.MissingArgumentError(None, "out", True)
        return op(*args, **kwargs)


def _field_on_domain(
    source: common.Field | core_defs.Scalar, domain: common.Domain, xp: Any
) -> common.Field:
    if isinstance(source, common.Field):
        xp = source.array_ns  # type: ignore[attr-defined]
        values = source[domain].ndarray
    else:
        values = xp.asarray(source)
    return common._field(xp.array(xp.broadcast_to(values, domain.shape)), domain=domain)


def _tuple_assign_field(
    target: xtyping.MaybeNestedInTuple[common.MutableField],
    source: xtyping.MaybeNestedInTuple[common.Field],
    domain: xtyping.MaybeNestedInTuple[common.Domain],
) -> None:
    @named_collections.tree_map_named_collection
    def impl(target: common.MutableField, source: common.Field, domain: common.Domain) -> None:
        if isinstance(source, common.Field):
            target[domain] = source[domain]
        else:
            assert core_defs.is_scalar_type(source)
            target[domain] = source

    if not isinstance(domain, tuple):
        domain = named_collections.tree_map_named_collection(lambda _: domain)(target)  # type: ignore[assignment] # typing not precise enough
    impl(target, source, domain)


def _intersect_scan_args(
    *args: xtyping.MaybeNestedInTuple[core_defs.Scalar | common.Field],
) -> common.Domain:
    return embedded_common.domain_intersection(
        *[arg.domain for arg in utils.flatten_nested_tuple(args) if isinstance(arg, common.Field)]
    )


def _tuple_assign_value(
    pos: Sequence[common.NamedIndex],
    target: xtyping.MaybeNestedInTuple[common.MutableField],
    source: xtyping.MaybeNestedInTuple[core_defs.Scalar],
) -> None:
    @utils.tree_map
    def impl(target: common.MutableField, source: core_defs.Scalar) -> None:
        target[pos] = source

    impl(target, source)


def _tuple_at(
    pos: Sequence[common.NamedIndex],
    field: xtyping.MaybeNestedInTuple[common.Field | core_defs.Scalar],
) -> core_defs.Scalar | tuple[core_defs.ScalarT | tuple, ...]:
    @named_collections.tree_map_named_collection
    def impl(field: common.Field | core_defs.Scalar) -> core_defs.Scalar:
        res = (
            field[tuple(p for p in pos if p.dim in field.domain.dims)].as_scalar()
            if isinstance(field, common.Field)
            else field
        )
        assert core_defs.is_scalar_type(res)
        return res

    return impl(field)  # type: ignore[return-value]
