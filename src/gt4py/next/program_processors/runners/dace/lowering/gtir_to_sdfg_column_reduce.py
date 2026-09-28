# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Implements the lowering of the column_reduce operator.

This builtin translator implements the `PrimitiveTranslator` protocol as other
translators in `gtir_to_sdfg_primitives` module.
"""

from __future__ import annotations

import dace
from dace import subsets as dace_subsets

from gt4py.next import utils as gtx_utils
from gt4py.next.iterator import ir as gtir
from gt4py.next.iterator.ir_utils import (
    column_reduce as ir_column_reduce,
    common_pattern_matcher as cpm,
    domain_utils,
    misc as gtir_misc,
)
from gt4py.next.iterator.type_system import inference as gtir_type_inference
from gt4py.next.program_processors.runners.dace.lowering import (
    gtir_to_sdfg,
    gtir_to_sdfg_types,
    gtir_to_sdfg_utils,
)
from gt4py.next.type_system import type_info, type_specifications as ts


def translate_column_reduce(
    node: gtir.Node,
    ctx: gtir_to_sdfg.SubgraphContext,
    sdfg_builder: gtir_to_sdfg.SDFGBuilder,
) -> gtir_to_sdfg_types.FieldopResult:
    """
    Lowers a `column_reduce` expression to a scan over the reduced range, followed by a copy
    of the last level of the scan result into a field without the reduced dimension.

    Implements the `PrimitiveTranslator` protocol.
    """
    assert cpm.is_call_to(node, "column_reduce")
    reduce_domain = domain_utils.SymbolicDomain.from_expr(node.args[1])
    ((axis, reduce_range),) = reduce_domain.ranges.items()

    output_domain = domain_utils.domain_union(*gtx_utils.flatten_nested_tuple((node.annex.domain,)))
    scan_expr = gtir_type_inference.reinfer(
        ir_column_reduce.as_scan(node, output_domain), offset_provider_type={}
    )
    _, scan = gtir_misc.extract_projector(scan_expr)
    assert cpm.is_applied_as_fieldop(scan)
    scan_domain = domain_utils.SymbolicDomain.from_expr(scan.fun.args[1])
    scan.annex.domain = type_info.tree_map_type(
        lambda _: scan_domain, result_collection_constructor=lambda _, elts: tuple(elts)
    )(scan.type)
    scan_result = sdfg_builder.visit(scan_expr, ctx=ctx)
    last_level = gtir_to_sdfg_utils.get_symbolic(reduce_range.stop) - 1

    def extract_last_level(
        data: gtir_to_sdfg_types.FieldopData,
    ) -> gtir_to_sdfg_types.FieldopData:
        assert isinstance(data.gt_type, ts.FieldType)
        axis_index = data.gt_type.dims.index(axis)
        desc = data.dc_node.desc(ctx.sdfg)
        shape = [size for i, size in enumerate(desc.shape) if i != axis_index]
        output, _ = sdfg_builder.add_temp_array(ctx.sdfg, shape, desc.dtype)
        output_node = ctx.state.add_access(output)
        level = last_level - data.origin[axis_index]
        ctx.state.add_nedge(
            data.dc_node,
            output_node,
            dace.Memlet(
                data=data.dc_node.data,
                subset=dace_subsets.Range(
                    [
                        (level, level, 1) if i == axis_index else (0, size - 1, 1)
                        for i, size in enumerate(desc.shape)
                    ]
                ),
                other_subset=dace_subsets.Range([(0, size - 1, 1) for size in shape]),
            ),
        )
        return gtir_to_sdfg_types.FieldopData(
            output_node,
            ts.FieldType(
                dims=[dim for dim in data.gt_type.dims if dim != axis], dtype=data.gt_type.dtype
            ),
            origin=tuple(o for i, o in enumerate(data.origin) if i != axis_index),
        )

    return gtx_utils.tree_map(extract_last_level)(scan_result)
