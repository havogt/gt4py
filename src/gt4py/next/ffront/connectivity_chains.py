# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from gt4py.next import common
from gt4py.next.ffront import (
    decorator,
    field_operator_ast as foast,
    type_specifications as ts_ffront,
)
from gt4py.next.iterator.transforms import connectivity_chains as chains_analysis
from gt4py.next.iterator.transforms.connectivity_chains import (
    Access,
    Accesses,
    AccessesBySymbol,
    Unbounded,
)
from gt4py.next.otf import arguments, workflow
from gt4py.next.type_system import type_info, type_specifications as ts


__all__ = ["Access", "Accesses", "Unbounded", "connectivity_chains", "required_indices"]


def _signature(
    operator: decorator.FieldOperator,
) -> tuple[dict[str, ts.TypeSpec], ts.TypeSpec]:
    node = operator.foast_stage.foast_node
    params = {str(param.id): param.type for param in node.definition.params}
    assert isinstance(node.definition.type, ts.FunctionType)
    return_type = node.definition.type.returns
    if isinstance(node, foast.ScanOperator):
        assert isinstance(node.type, ts_ffront.ScanOperatorType)
        axis = node.type.axis

        def as_column_type(type_: ts.TypeSpec) -> ts.TypeSpec:
            return type_info.tree_map_type(lambda t: ts.FieldType(dims=[axis], dtype=t))(type_)

        state_param = next(iter(params))
        params = {name: as_column_type(t) for name, t in params.items() if name != state_param}
        return_type = as_column_type(return_type)
    return params, return_type


def connectivity_chains(
    operator: decorator.FieldOperator,
    offset_provider: common.OffsetProvider | common.OffsetProviderType,
) -> dict[tuple[int, ...], AccessesBySymbol]:
    """
    Trace how each output point of a field (or scan) operator reaches its inputs.

    Nested field operators are followed. Every read of a parameter is reported as an `Access`:
    the sequence of neighbor connectivities applied from the output point outwards, and the total
    displacement along cartesian (on unstructured grids: vertical) dimensions. Displacements
    computed from field values are reported as `Unbounded.DYNAMIC`, reads of a whole column by a
    scan as `Unbounded.COLUMN`.

    Args:
        operator: The field or scan operator.
        offset_provider: The offset provider (or its type) the operator is called with.

    Returns:
        A mapping from the path of each output (tuple indices, `()` for a non-tuple return) to a
        mapping from parameter name to its accesses: a set of `Access`, or a tuple of those for a
        tuple parameter. Parameters that are not read for an output are omitted.

    Examples:
        >>> import gt4py.next as gtx
        >>> from gt4py.next import neighbor_sum
        >>> Cell = gtx.Dimension("Cell")
        >>> Edge = gtx.Dimension("Edge")
        >>> E2CDim = gtx.Dimension("E2C", kind=gtx.DimensionKind.LOCAL)
        >>> E2C = gtx.FieldOffset("E2C", source=Cell, target=(Edge, E2CDim))
        >>> @gtx.field_operator
        ... def cell_to_edge(c: gtx.Field[[Cell], float]) -> gtx.Field[[Edge], float]:
        ...     return neighbor_sum(c(E2C), axis=E2CDim)
        >>> e2c_type = common.NeighborConnectivityType(
        ...     domain=(Edge, E2CDim),
        ...     codomain=Cell,
        ...     skip_value=None,
        ...     dtype=np.dtype(np.int32),
        ...     max_neighbors=2,
        ... )
        >>> connectivity_chains(cell_to_edge, {"E2C": e2c_type})
        {(): {'c': frozenset({Access(chain=('E2C',), displacement=())})}}
    """
    offset_provider_type = common.offset_provider_to_type(offset_provider)
    params, return_type = _signature(operator)
    compile_time_args = arguments.CompileTimeArgs(
        args=(*params.values(), return_type),
        kwargs={},
        offset_provider=offset_provider_type,  # type: ignore[arg-type]  # only the types are needed
        column_axis=None,
        argument_descriptor_contexts={},
    )
    program = operator._frontend_transforms(
        workflow.ConcreteArtifact(operator.foast_stage, compile_time_args)
    ).data
    return chains_analysis.trace_program(program, offset_provider_type)


def required_indices(
    operator: decorator.FieldOperator,
    offset_provider: common.OffsetProvider,
    owned: Mapping[common.Dimension, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Compute the horizontal indices of each input field needed to compute the outputs on `owned`.

    Args:
        operator: The field or scan operator.
        offset_provider: The offset provider the operator is called with; its neighbor tables
            are followed, skipping missing neighbors.
        owned: The indices at which the outputs are computed, per horizontal dimension.

    Returns:
        A mapping from each field parameter with a horizontal dimension to the sorted, unique
        indices of that dimension that are read.
    """
    params, return_type = _signature(operator)
    horizontal_dim_of = {name: _horizontal_dim(type_) for name, type_ in params.items()}
    output_types = dict(_leaf_types(return_type))

    cache: dict[tuple[common.Dimension, tuple[str, ...]], np.ndarray] = {}

    def follow(dim: common.Dimension, chain: tuple[str, ...]) -> np.ndarray:
        if (dim, chain) not in cache:
            if not chain:
                cache[dim, chain] = np.unique(np.asarray(owned[dim]))
            else:
                start = follow(dim, chain[:-1])
                connectivity = offset_provider[chain[-1]]
                assert isinstance(connectivity, common.Connectivity)
                neighbors = connectivity.asnumpy()[start].ravel()
                if connectivity.skip_value is not None:
                    neighbors = neighbors[neighbors != connectivity.skip_value]
                cache[dim, chain] = np.unique(neighbors)
        return cache[dim, chain]

    needed: dict[str, list[np.ndarray]] = {}
    for path, accesses_by_param in connectivity_chains(operator, offset_provider).items():
        output_dim = _horizontal_dim(output_types[path])
        if output_dim is None:
            continue
        for name, accesses in accesses_by_param.items():
            if horizontal_dim_of.get(name) is None:
                continue
            for access in chains_analysis.flatten(accesses):
                needed.setdefault(name, []).append(follow(output_dim, access.chain))
    return {name: np.unique(np.concatenate(indices)) for name, indices in needed.items()}


def _horizontal_dim(type_: ts.TypeSpec) -> common.Dimension | None:
    dims = {
        dim
        for el_type in type_info.primitive_constituents(type_)
        for dim in type_info.extract_dims(el_type)
        if dim.kind == common.DimensionKind.HORIZONTAL
    }
    if len(dims) > 1:
        raise ValueError(f"Expected at most one horizontal dimension, got {sorted(dims, key=str)}.")
    return next(iter(dims), None)


def _leaf_types(
    type_: ts.TypeSpec, path: tuple[int, ...] = ()
) -> list[tuple[tuple[int, ...], ts.TypeSpec]]:
    if isinstance(type_, ts.COLLECTION_TYPE_SPECS):
        return [leaf for i, t in enumerate(type_.types) for leaf in _leaf_types(t, (*path, i))]
    return [(path, type_)]
