# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# This file is part of the GT4Py project and the GridTools framework.
# GT4Py is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
from types import ModuleType
from typing import Any, Callable, Generic, Optional, ParamSpec, Sequence, TypeVar

import numpy as np

from gt4py import eve
from gt4py._core import definitions as core_defs
from gt4py.next import common, errors, utils
from gt4py.next.embedded import common as embedded_common, context as embedded_context
from gt4py.next.ffront import fbuiltins


_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclasses.dataclass(frozen=True)
class EmbeddedOperator(Generic[_R, _P]):
    fun: Callable[_P, _R]

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        return self.fun(*args, **kwargs)


@dataclasses.dataclass(frozen=True)
class ScanOperator(EmbeddedOperator[core_defs.ScalarT | tuple[core_defs.ScalarT | tuple, ...], _P]):
    forward: bool
    init: core_defs.ScalarT | tuple[core_defs.ScalarT | tuple, ...]
    axis: common.Dimension

    def __call__(  # type: ignore[override]
        self,
        *args: common.Field | core_defs.Scalar,
        **kwargs: common.Field | core_defs.Scalar,  # type: ignore[override]
    ) -> (
        common.Field[Any, core_defs.ScalarT]
        | tuple[common.Field[Any, core_defs.ScalarT] | tuple, ...]
    ):
        scan_range = embedded_context.closure_column_range.get()
        assert self.axis == scan_range.dim
        scan_axis = scan_range.dim
        all_args = [*args, *kwargs.values()]
        domain_intersection = _intersect_scan_args(*all_args)
        non_scan_domain = common.Domain(*[nr for nr in domain_intersection if nr.dim != scan_axis])

        out_domain = common.Domain(
            *[scan_range if nr.dim == scan_axis else nr for nr in domain_intersection]
        )
        if scan_axis not in out_domain.dims:
            # even if the scan dimension is not in the input, we can scan over it
            out_domain = common.Domain(*out_domain, (scan_range))

        xp = _get_array_ns(*all_args)
        res = _construct_scan_array(out_domain, xp)(self.init)

        def scan_loop(hpos: Sequence[common.NamedIndex]) -> None:
            acc: core_defs.ScalarT | tuple[core_defs.ScalarT | tuple, ...] = self.init
            for k in scan_range.unit_range if self.forward else reversed(scan_range.unit_range):
                pos = (*hpos, common.NamedIndex(scan_axis, k))
                new_args = [_tuple_at(pos, arg) for arg in args]
                new_kwargs = {k: _tuple_at(pos, v) for k, v in kwargs.items()}
                acc = self.fun(acc, *new_args, **new_kwargs)  # type: ignore[arg-type] # need to express that the first argument is the same type as the return
                _tuple_assign_value(pos, res, acc)

        if len(non_scan_domain) == 0:
            # if we don't have any dimension orthogonal to scan_axis, we need to do one scan_loop
            scan_loop(())
        else:
            for hpos in embedded_common.iterate_domain(non_scan_domain):
                scan_loop(hpos)

        return res


@dataclasses.dataclass(frozen=True)
class ScanOperatorVectorized(
    EmbeddedOperator[core_defs.ScalarT | tuple[core_defs.ScalarT | tuple, ...], _P]
):
    forward: bool
    init: core_defs.ScalarT | tuple[core_defs.ScalarT | tuple, ...]
    axis: common.Dimension

    def __call__(  # type: ignore[override]
        self,
        *args: common.Field | core_defs.Scalar,
        **kwargs: common.Field | core_defs.Scalar,  # type: ignore[override]
    ) -> (
        common.Field[Any, core_defs.ScalarT]
        | tuple[common.Field[Any, core_defs.ScalarT] | tuple, ...]
    ):
        scan_range = embedded_context.closure_column_range.get()
        assert self.axis == scan_range.dim
        scan_axis = scan_range.dim
        all_args = [*args, *kwargs.values()]
        domain_intersection = _intersect_scan_args(*all_args)
        non_scan_domain = common.Domain(*[nr for nr in domain_intersection if nr.dim != scan_axis])

        out_domain = common.Domain(
            *[scan_range if nr.dim == scan_axis else nr for nr in domain_intersection]
        )
        if scan_axis not in out_domain.dims:
            # even if the scan dimension is not in the input, we can scan over it
            out_domain = common.Domain(*out_domain, (scan_range))

        xp = _get_array_ns(*all_args)
        res = _construct_scan_array(out_domain, xp)(self.init)

        def scan_loop() -> None:
            acc: common.MutableField | tuple[common.MutableField | tuple, ...] = (
                _construct_scan_array(non_scan_domain, xp)(self.init)
            )
            _tuple_assign_field(target=acc, source=self.init, domain=non_scan_domain)
            for k in scan_range.unit_range if self.forward else reversed(scan_range.unit_range):
                new_args = [
                    arg if core_defs.is_scalar_type(arg) else arg[common.NamedIndex(scan_axis, k)]
                    for arg in args
                ]
                new_kwargs = {k: v[common.NamedIndex(scan_axis, k)] for k, v in kwargs.items()}
                acc = self.fun(acc, *new_args, **new_kwargs)  # type: ignore[arg-type] # need to express that the first argument is the same type as the return

                k_slice = common.Domain(
                    *non_scan_domain,
                    common.named_range((scan_axis, (k, k + 1))),
                )
                broadcasted_acc = _tuple_broadcast_field(acc, (*non_scan_domain.dims, scan_axis))
                _tuple_assign_field(res, broadcasted_acc, k_slice)

        scan_loop()

        return res


def to_jax_field(field):
    from gt4py.next.embedded import nd_array_field
    from jax import numpy as jnp

    if not isinstance(field, common.Field) or isinstance(field, nd_array_field.JaxArrayField):
        return field
    else:
        assert isinstance(field, nd_array_field.NumPyArrayField)
        return common._field(jnp.asarray(field.ndarray), domain=field.domain)


def to_numpy_field(field):
    if not isinstance(field, common.Field):
        return field
    return common._field(np.asarray(field.ndarray), domain=field.domain)


def transpose(f, dims):
    @utils.tree_map
    def impl(f):
        xp = _get_array_ns(f)
        arr = xp.transpose(
            f.ndarray,
            axes=[f.domain.dim_index(dim) for dim in dims],
        )
        domain = common.Domain(*(f.domain[dim] for dim in dims))
        return common._field(arr, domain=domain)

    return impl(f)


def broadcast_to(f, domain):
    xp = _get_array_ns(f)
    return common._field(xp.broadcast_to(f.ndarray, domain.shape), domain=domain)


@dataclasses.dataclass(frozen=True)
class ScanOperatorJax(
    EmbeddedOperator[core_defs.ScalarT | tuple[core_defs.ScalarT | tuple, ...], _P]
):
    forward: bool
    init: core_defs.ScalarT | tuple[core_defs.ScalarT | tuple, ...]
    axis: common.Dimension

    def __call__(  # type: ignore[override]
        self,
        *args: common.Field | core_defs.Scalar,
        **kwargs: common.Field | core_defs.Scalar,  # type: ignore[override]
    ) -> (
        common.Field[Any, core_defs.ScalarT]
        | tuple[common.Field[Any, core_defs.ScalarT] | tuple, ...]
    ):
        scan_range = embedded_context.closure_column_range.get()
        assert self.axis == scan_range.dim
        scan_axis = scan_range.dim

        # args = [to_jax_field(arg) for arg in args]
        # kwargs = {k: to_jax_field(v) for k, v in kwargs.items()}

        all_args = [*args, *kwargs.values()]
        domain_intersection = _intersect_scan_args(*all_args)
        non_scan_domain = common.Domain(*[nr for nr in domain_intersection if nr.dim != scan_axis])

        out_domain = common.Domain(
            *[scan_range if nr.dim == scan_axis else nr for nr in domain_intersection]
        )
        if scan_axis not in out_domain.dims:
            # even if the scan dimension is not in the input, we can scan over it
            out_domain = common.Domain(*out_domain, (scan_range))

        from jax import numpy as jnp, lax

        # res = _construct_scan_array(out_domain, jnp)(self.init)

        def jax_fun(carry, x):
            x = utils.tree_map(lambda f: common._field(f.ndarray, domain=f.domain[1:]))(x)
            res = self.fun(carry, *x)
            return (res, res)

        def make_field(f):
            return common._field(jnp.full(out_domain.shape, f), domain=out_domain)

        def scan_loop() -> None:
            new_dims = (scan_axis, *non_scan_domain.dims)
            new_args = tuple(
                make_field(arg) if not isinstance(arg, common.Field) else arg for arg in args
            )
            new_args = tuple(
                transpose(
                    broadcast_to(
                        fbuiltins.broadcast(arg[scan_range], (*non_scan_domain.dims, scan_axis)),
                        out_domain,
                    ),
                    new_dims,
                )
                for arg in new_args
            )
            assert len(kwargs) == 0

            init: common.MutableField | tuple[common.MutableField | tuple, ...] = (
                _construct_scan_array(non_scan_domain, jnp)(self.init)
            )
            _tuple_assign_field(target=init, source=self.init, domain=non_scan_domain)
            res = lax.scan(jax_fun, init, new_args, reverse=not self.forward)
            res = res[1]
            res = utils.tree_map(
                lambda f: common._field(f.ndarray, domain=common.Domain(scan_range, *f.domain))
            )(res)
            res = transpose(res, out_domain.dims)
            return res

        res = scan_loop()
        return res

        # return utils.tree_map(lambda a: to_numpy_field(a))(res)


def _get_out_domain(
    out: common.MutableField | tuple[common.MutableField | tuple, ...],
) -> common.Domain:
    return embedded_common.domain_intersection(
        *[f.domain for f in utils.flatten_nested_tuple((out,))]
    )


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

        out_domain = common.domain(domain) if domain is not None else _get_out_domain(out)

        new_context_kwargs["closure_column_range"] = _get_vertical_range(out_domain)

        with embedded_context.new_context(**new_context_kwargs) as ctx:
            res = ctx.run(op, *args, **kwargs)
            _tuple_assign_field(
                out,
                res,  # type: ignore[arg-type] # maybe can't be inferred properly because decorator.py is not properly typed yet
                domain=out_domain,
            )
        return None
    else:
        # called from other field_operator or missing `out` argument
        new_context_kwargs = {}
        if "offset_provider" in kwargs:
            # this enables calling a field operator entry point which returns (if `out` is omitted)
            offset_provider = kwargs.pop("offset_provider", None)
            new_context_kwargs["offset_provider"] = offset_provider
            domain = kwargs.pop("domain", None)
            assert domain
            out_domain = common.domain(domain)
            new_context_kwargs["closure_column_range"] = _get_vertical_range(out_domain)
            use_jax = kwargs.pop("use_jax", False)
            with embedded_context.new_context(**new_context_kwargs) as ctx:
                if not use_jax:
                    return ctx.run(op, *args, **kwargs)
                else:
                    import jax

                    print(ctx.run(jax.make_jaxpr(op), *args, **kwargs))

                    return ctx.run(jax.jit(op), *args, **kwargs)

        return op(*args, **kwargs)


def _get_vertical_range(domain: common.Domain) -> common.NamedRange | eve.NothingType:
    vertical_dim_filtered = [nr for nr in domain if nr.dim.kind == common.DimensionKind.VERTICAL]
    assert len(vertical_dim_filtered) <= 1
    return vertical_dim_filtered[0] if vertical_dim_filtered else eve.NOTHING


def _tuple_assign_field(
    target: tuple[common.MutableField | tuple, ...] | common.MutableField,
    source: tuple[common.Field | tuple, ...] | common.Field,
    domain: common.Domain,
) -> None:
    @utils.tree_map
    def impl(target: common.MutableField, source: common.Field) -> None:
        if isinstance(source, common.Field):
            target[domain] = source[domain]
        else:
            assert core_defs.is_scalar_type(source)
            target[domain] = source

    impl(target, source)


def _tuple_broadcast_field(f, domain):
    @utils.tree_map
    def impl(f):
        return fbuiltins.broadcast(f, domain)

    return impl(f)


def _intersect_scan_args(
    *args: core_defs.Scalar | common.Field | tuple[core_defs.Scalar | common.Field | tuple, ...],
) -> common.Domain:
    return embedded_common.domain_intersection(
        *[arg.domain for arg in utils.flatten_nested_tuple(args) if isinstance(arg, common.Field)]
    )


def _get_array_ns(
    *args: core_defs.Scalar | common.Field | tuple[core_defs.Scalar | common.Field | tuple, ...],
) -> ModuleType:
    for arg in utils.flatten_nested_tuple(args):
        if hasattr(arg, "array_ns"):
            return arg.array_ns
    return np


def _construct_scan_array(
    domain: common.Domain,
    xp: ModuleType,  # TODO(havogt) introduce a NDArrayNamespace protocol
) -> Callable[
    [core_defs.Scalar | tuple[core_defs.Scalar | tuple, ...]],
    common.MutableField | tuple[common.MutableField | tuple, ...],
]:
    @utils.tree_map
    def impl(init: core_defs.Scalar) -> common.MutableField:
        res = common._field(xp.empty(domain.shape, dtype=type(init)), domain=domain)
        assert isinstance(res, common.MutableField)
        return res

    return impl


def _tuple_assign_value(
    pos: Sequence[common.NamedIndex],
    target: common.MutableField | tuple[common.MutableField | tuple, ...],
    source: core_defs.Scalar | tuple[core_defs.Scalar | tuple, ...],
) -> None:
    @utils.tree_map
    def impl(target: common.MutableField, source: core_defs.Scalar) -> None:
        target[pos] = source

    impl(target, source)


def _tuple_at(
    pos: Sequence[common.NamedIndex],
    field: common.Field | core_defs.Scalar | tuple[common.Field | core_defs.Scalar | tuple, ...],
) -> core_defs.Scalar | tuple[core_defs.ScalarT | tuple, ...]:
    @utils.tree_map
    def impl(field: common.Field | core_defs.Scalar) -> core_defs.Scalar:
        res = field[pos].as_scalar() if isinstance(field, common.Field) else field
        assert core_defs.is_scalar_type(res)
        return res

    return impl(field)  # type: ignore[return-value]
