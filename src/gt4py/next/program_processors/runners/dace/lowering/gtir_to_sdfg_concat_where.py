# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Implements the lowering of concat_where operator.

This builtin translator implements the `PrimitiveTranslator` protocol as other
translators in `gtir_to_sdfg_primitives` module.
"""

from __future__ import annotations

from typing import Sequence

import dace
from dace import nodes as dace_nodes, subsets as dace_subsets

from gt4py.next import common as gtx_common
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    domain_utils,
    ir_makers as im,
    misc as gtir_misc,
)
from gt4py.next.iterator.type_system import inference as gtir_type_inference
from gt4py.next.program_processors.runners.dace import sdfg_args as gtx_dace_args
from gt4py.next.program_processors.runners.dace.lowering import (
    gtir_domain,
    gtir_to_sdfg,
    gtir_to_sdfg_types,
    gtir_to_sdfg_utils,
)
from gt4py.next.type_system import type_specifications as ts


def _translate_concat_where_branch(
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
    source_expr: gtir.Expr,
    is_lower: bool,
    concat_dim: gtx_common.Dimension,
    concat_dim_bound_expr: gtir.Expr,
    output_domain: domain_utils.SymbolicDomain,
    output_type: ts.FieldType,
    output_desc: dace.data.Array,
    output_node: dace_nodes.AccessNode,
    output_origin: Sequence[dace.symbolic.SymbolicType],
) -> None:
    """
    Translate one of the two branches of a 'concat_where' expression.

    The 'concat_where' expression requires two input fields, one is written on the
    lower part of the result domain, with respect to the domain boundary extracted
    from the first argument; the other input field is written on the upper domain.
    The handling of the two branches is similar, there is only one small difference
    in case the input field does not contain the 'concat_where' dimension, which
    is the case of scalar values or horizontal slice fields. In this case, we use
    the 'is_lower' argument to specify which of the two branches this function
    should target.

    The _regular_ case is `concat_where` applied to two fields with same domain
    as the result field, for example two 'IJK'-fields:
    ```python
    @gtx.field_operator
    def testee(a: cases.IJKField, b: cases.IJKField) -> cases.IJKField:
        return concat_where(KDim < 10, a, b)
    ```

    In case of the arguments is a scalar value, we broadcast it on all dimensions
    of the result field, for example:
    ```python
    @gtx.field_operator
    def testee(a: np.int32, b: IJKField) -> IJKField:
        return concat_where(KDim == 0, a, b)
    ```
    Similarly, we broadcast a field defined as a slice of the output domain, i.e.
    a field with a smaller number of dimensions than the result field.
    Consider for example the following field operator, where the 'boundary' IJ-field
    is used for the vertical boundary (`KDim == 0`):
    ```python
    @gtx.field_operator
    def testee(interior: cases.IJKField, boundary: cases.IJField) -> cases.IJKField:
        return concat_where(KDim == 0, boundary, interior)
    ```
    A field of constant lists, as produced by `make_const_list`, is broadcast in the
    same way to the neighbor-list type of the result field.

    Args:
        ctx: The SDFG context in which to lower the `concat_where` branch expression.
        sdfg_builder: The visitor object to build the SDFG.
        source_expr: The expression producing the input field for this branch.
        is_lower: Flag to specify whether the input expression is for the lower or upper domain.
        concat_dim: The dimension along which to apply `concat_where` on the input fields.
        concat_dim_bound_expr: The expression of the `concat_where` domain condition.
        output_domain: The domain of the result field.
        output_type: The GT4Py type descriptor of the result field.
        output_desc: The DaCe array descriptor of the result field.
        output_node: The SDFG access node for writing the result field.
        output_origin: The origin of the result field, derived from `output_domain`,
            to be used in the write memlet to `output_node`.
    """
    assert isinstance(source_expr.type, (ts.FieldType, ts.ScalarType))

    source_domain = source_expr.annex.domain
    if source_expr.type != output_type:
        # We promote the input expression to a field defined on the output domain,
        # refer to the function documentation for examples of such field operators.
        if concat_dim not in source_domain.ranges:
            source_domain.ranges[concat_dim] = (
                domain_utils.SymbolicRange(
                    start=output_domain.ranges[concat_dim].start,
                    stop=concat_dim_bound_expr,
                )
                if is_lower
                else domain_utils.SymbolicRange(
                    start=concat_dim_bound_expr,
                    stop=output_domain.ranges[concat_dim].stop,
                )
            )
        source_domain = domain_utils.promote_domain(source_domain, output_type.dims)
        source_domain = domain_utils.domain_intersection(source_domain, output_domain)

        # Use a 'deref' field operator to broadcast the input expression on the target domain.
        bcast_expr = im.as_fieldop("deref", source_domain.as_expr())(source_expr)
        bcast_expr.type = output_type

        source = sdfg_builder.visit(bcast_expr, ctx=ctx)
    else:
        # The input field is defined on all dimensions of the result field.
        source = sdfg_builder.visit(source_expr, ctx=ctx)

    assert source.gt_type == output_type
    source_domain_range = source_domain.ranges[concat_dim]
    if cpm.is_applied_scan(gtir_misc.extract_projector(source_expr)[1]):
        # A scan is computed on its own range, which can exceed the region of this branch.
        output_range = output_domain.ranges[concat_dim]
        branch_start, branch_stop = (
            (output_range.start, concat_dim_bound_expr)
            if is_lower
            else (concat_dim_bound_expr, output_range.stop)
        )
        source_domain_range = domain_utils.SymbolicRange(
            start=im.maximum(source_domain_range.start, branch_start),
            stop=im.minimum(source_domain_range.stop, branch_stop),
        )
    source_range_0 = gtir_to_sdfg_utils.get_symbolic(source_domain_range.start)
    source_range_1 = gtir_to_sdfg_utils.get_symbolic(
        im.maximum(source_domain_range.start, source_domain_range.stop)
    )
    source_range_size = source_range_1 - source_range_0

    src_origins = list(source.origin)
    dst_origins = list(output_origin)
    if isinstance(output_type.dtype, ts.ScalarType):
        all_dims = gtx_common.order_dimensions(output_type.dims)
    else:
        assert output_type.dtype.offset_type
        all_dims = gtx_common.order_dimensions([*output_type.dims, output_type.dtype.offset_type])
        local_idx = all_dims.index(output_type.dtype.offset_type)
        src_origins.insert(local_idx, 0)
        dst_origins.insert(local_idx, 0)

    source_subset = []
    output_subset = []
    for dim, size, src_origin, dst_origin in zip(
        all_dims,
        output_desc.shape,
        src_origins,
        dst_origins,
        strict=True,
    ):
        if dim == concat_dim:
            # Write only the subset corresponding to the range of lower or upper branch.
            source_subset.append(
                (
                    source_range_0 - src_origin,
                    source_range_1 - src_origin - 1,
                    1,
                )
            )
            output_subset.append(
                (
                    source_range_0 - dst_origin,
                    source_range_0 - dst_origin + source_range_size - 1,
                    1,
                )
            )
        else:
            # Write the full subset which covers the array size in this dimension.
            source_subset.append(
                (
                    dst_origin - src_origin,
                    dst_origin - src_origin + size - 1,
                    1,
                )
            )
            output_subset.append((0, size - 1, 1))

    ctx.state.add_nedge(
        source.dc_node,
        output_node,
        dace.Memlet(
            data=source.dc_node.data,
            subset=dace_subsets.Range(source_subset),
            other_subset=dace_subsets.Range(output_subset),
        ),
    )


_INFINITY_LITERALS = (gtir.InfinityLiteral.POSITIVE, gtir.InfinityLiteral.NEGATIVE)


def _color_slab_stencil_body(
    expr: gtir.Expr,
    color_dim: gtx_common.Dimension,
    pos: str,
    domain: domain_utils.SymbolicDomain,
    args: dict[str, gtir.Expr],
) -> gtir.Expr | None:
    """
    Stencil body selecting, by the value of the `pos` index iterator, the stencil of
    the colour slab that `expr` computes there, or `None` if `expr` is not a chain of
    `concat_where` along `color_dim` whose leaves are `as_fieldop` of a lambda on the
    slab of `domain`. The leaf arguments are collected in `args`, by parameter name.
    """
    if cpm.is_call_to(expr, "concat_where"):
        mask = domain_utils.SymbolicDomain.from_expr(expr.args[0])
        if list(mask.ranges.keys()) != [color_dim]:
            return None
        mask_range = mask.ranges[color_dim]
        if mask_range.start in _INFINITY_LITERALS:
            cond = im.less(im.deref(pos), mask_range.stop)
        elif mask_range.stop in _INFINITY_LITERALS:
            cond = im.greater_equal(im.deref(pos), mask_range.start)
        else:
            return None
        true_body = _color_slab_stencil_body(expr.args[1], color_dim, pos, domain, args)
        false_body = _color_slab_stencil_body(expr.args[2], color_dim, pos, domain, args)
        if true_body is None or false_body is None:
            return None
        return im.if_(cond, true_body, false_body)

    if not (cpm.is_applied_as_fieldop(expr) and isinstance(expr.fun.args[0], gtir.Lambda)):
        return None
    stencil = expr.fun.args[0]
    leaf_domain = domain_utils.SymbolicDomain.from_expr(expr.fun.args[1])
    if any(
        leaf_domain.ranges.get(dim) != dim_range
        for dim, dim_range in domain.ranges.items()
        if dim != color_dim
    ):
        return None
    # The leaf parameters are bound by `let`, not substituted: a `let` keeps the full
    # field as iterator inside the `if_` branch, so no read is declared on the slabs
    # of the other colours.
    bindings = []
    for param, arg in zip(stencil.params, expr.args, strict=True):
        name = f"__cs_arg{len(args)}"
        args[name] = arg
        bindings.append((param, im.ref(name)))
    return im.let(*bindings)(stencil.expr)


def _merge_color_slabs(
    node: gtir.FunCall, sdfg_builder: gtir_to_sdfg.SDFGBuilder
) -> gtir.FunCall | None:
    """
    Rewrite a `concat_where` chain along the colour dimension of a structured layout,
    whose leaves are `as_fieldop`s on one colour slab each, into one `as_fieldop` on
    the whole domain, which selects the stencil of the colour by an `if_` on the index.
    Returns `None` if `node` is not such a chain.
    """
    mask_dims = list(domain_utils.SymbolicDomain.from_expr(node.args[0]).ranges.keys())
    if len(mask_dims) != 1 or mask_dims[0] not in sdfg_builder.get_color_dims():
        return None
    color_dim = mask_dims[0]
    domain = node.annex.domain
    pos = "__cs_pos"
    args: dict[str, gtir.Expr] = {}
    body = _color_slab_stencil_body(node, color_dim, pos, domain, args)
    if body is None:
        return None
    merged = im.as_fieldop(im.lambda_(pos, *args.keys())(body), domain.as_expr())(
        im.index(color_dim), *args.values()
    )
    gtir_type_inference.reinfer(merged)
    assert merged.type == node.type
    merged.annex.domain = domain
    return merged


def translate_concat_where(
    node: gtir.Node,
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
) -> gtir_to_sdfg_types.FieldopResult:
    """
    Lowers a `concat_where` expression to a dataflow where two memlets write
    disjoint subsets, for the lower and upper domain, on one data access node.

    Implements the `PrimitiveTranslator` protocol.
    """
    assert cpm.is_call_to(node, "concat_where")
    assert len(node.args) == 3
    assert isinstance(node.type, (ts.FieldType, ts.TupleType))

    if isinstance(node.type, ts.TupleType):
        raise NotImplementedError("Unexpected 'concat_where' with tuple output in SDFG lowering.")

    # First argument is a domain expression that defines the mask of the true branch:
    # we extract the dimension along which we need to concatenate the field arguments,
    # and determine whether the true branch argument should be on the lower or upper
    # range with respect to the boundary value.
    mask_domain = domain_utils.SymbolicDomain.from_expr(node.args[0])
    if len(mask_domain.ranges) != 1:
        raise NotImplementedError("Expected `concat_where` along single axis.")

    concat_dim = next(iter(mask_domain.ranges.keys()))

    # Expect unbound range in the concat domain expression on range start or end:
    #  - If the domain expression is unbound on range start (negative infinite),
    #    the expression on the true branch is to be considered the input for the
    #    lower domain.
    #  - Vice versa, if the domain expression is unbound on range stop (positive
    #    infinite), the true expression represents the input for the upper domain.
    if (merged := _merge_color_slabs(node, sdfg_builder)) is not None:
        return sdfg_builder.visit(merged, ctx=ctx)

    if mask_domain.ranges[concat_dim].start in _INFINITY_LITERALS:
        bound_expr = mask_domain.ranges[concat_dim].stop
        lower_expr, upper_expr = node.args[1:]
    elif mask_domain.ranges[concat_dim].stop in _INFINITY_LITERALS:
        bound_expr = mask_domain.ranges[concat_dim].start
        upper_expr, lower_expr = node.args[1:]
    else:
        raise ValueError(f"Unexpected concat mask {mask_domain} with finite domain.")

    # We use the concat domain, stored in the annex, as the domain of output field.
    output_domain = gtir_domain.get_field_domain(node.annex.domain)
    output_dims, output_origin, output_shape = gtir_domain.get_field_layout(output_domain)
    assert output_dims == node.type.dims

    if isinstance(node.type.dtype, ts.ScalarType):
        dtype = gtx_dace_args.as_dace_type(node.type.dtype)
    else:
        assert isinstance(node.type.dtype.element_type, ts.ScalarType)
        local_dim = node.type.dtype.offset_type
        assert local_dim is not None
        dtype = gtx_dace_args.as_dace_type(node.type.dtype.element_type)
        offset_provider_type = sdfg_builder.get_offset_provider_type(local_dim.value)
        assert isinstance(offset_provider_type, gtx_common.NeighborConnectivityType)
        local_idx = gtx_common.order_dimensions([*output_dims, local_dim]).index(local_dim)
        output_shape.insert(local_idx, offset_provider_type.max_neighbors)

    output, output_desc = sdfg_builder.add_temp_array(ctx.sdfg, output_shape, dtype)
    output_node = ctx.state.add_access(output)

    # Translate the input expression on the lower domain.
    _translate_concat_where_branch(
        ctx=ctx,
        sdfg_builder=sdfg_builder,
        source_expr=lower_expr,
        is_lower=True,
        concat_dim=concat_dim,
        concat_dim_bound_expr=bound_expr,
        output_domain=node.annex.domain,
        output_type=node.type,
        output_desc=output_desc,
        output_node=output_node,
        output_origin=output_origin,
    )

    # Translate the input expression on the upper domain.
    _translate_concat_where_branch(
        ctx=ctx,
        sdfg_builder=sdfg_builder,
        source_expr=upper_expr,
        is_lower=False,
        concat_dim=concat_dim,
        concat_dim_bound_expr=bound_expr,
        output_domain=node.annex.domain,
        output_type=node.type,
        output_desc=output_desc,
        output_node=output_node,
        output_origin=output_origin,
    )

    return gtir_to_sdfg_types.FieldopData(output_node, node.type, origin=tuple(output_origin))
