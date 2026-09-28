# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import functools
from typing import Iterator

from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
)
from gt4py.next.type_system import type_info, type_specifications as ts, type_translation


def _leaf_paths(type_: ts.TypeSpec, prefix: tuple[int, ...] = ()) -> Iterator[tuple[int, ...]]:
    if isinstance(type_, ts.TupleType):
        for i, el_type in enumerate(type_.types):
            yield from _leaf_paths(el_type, (*prefix, i))
    else:
        yield prefix


def _rebuild(
    type_: ts.TypeSpec, prefix: tuple[int, ...], leaves: dict[tuple[int, ...], itir.Expr]
) -> itir.Expr:
    if isinstance(type_, ts.TupleType):
        return im.make_tuple(
            *(_rebuild(el_type, (*prefix, i), leaves) for i, el_type in enumerate(type_.types))
        )
    return leaves[prefix]


def as_scan(node: itir.FunCall, output_domain: domain_utils.SymbolicDomain) -> itir.Expr:
    """
    Express `column_reduce(op, reduce_domain, field)` as a forward scan over the reduced range.

    The scan is posed on `output_domain` extended by the range of `reduce_domain`. Its value on
    the last level of that range is the reduction. The carry is `(is_initialized, value)`,
    seeded with the first level.
    """
    assert cpm.is_call_to(node, "column_reduce")
    assert isinstance(node.type, ts.TypeSpec)
    op, reduce_domain, field = node.args
    domain = domain_utils.SymbolicDomain(
        output_domain.grid_type,
        {**output_domain.ranges, **domain_utils.SymbolicDomain.from_expr(reduce_domain).ranges},
    ).as_expr()

    def zero(type_: ts.TypeSpec) -> itir.Literal:
        dtype = type_info.extract_dtype(type_)
        assert isinstance(dtype, ts.ScalarType)
        return im.literal_from_value(type_translation.as_dtype(dtype).scalar_type(0))

    placeholder = type_info.tree_map_type(
        zero, result_collection_constructor=lambda _, elts: im.make_tuple(*elts)
    )(node.type)
    assert isinstance(field.type, ts.TypeSpec)
    paths = list(_leaf_paths(field.type))
    params = [f"__cr_it{i}" for i in range(len(paths))]
    args = [functools.reduce(lambda expr, i: im.tuple_get(i, expr), path, field) for path in paths]
    value = _rebuild(field.type, (), dict(zip(paths, map(im.deref, params), strict=True)))

    definition = im.lambda_("__cr_carry", *params)(
        im.let("__cr_value", value)(
            im.make_tuple(
                im.literal_from_value(True),
                im.if_(
                    im.tuple_get(0, "__cr_carry"),
                    im.call(op)(im.tuple_get(1, "__cr_carry"), "__cr_value"),
                    "__cr_value",
                ),
            )
        )
    )
    seed = im.make_tuple(im.literal_from_value(False), placeholder)
    return im.tuple_get(1, im.as_fieldop(im.scan(definition, True, seed), domain)(*args))
