# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Trace which neighbor chains and displacements connect each output point to the inputs it reads."""

from __future__ import annotations

import dataclasses
from typing import TypeAlias

from gt4py import eve
from gt4py.next import common, utils
from gt4py.next.iterator import builtins, ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm
from gt4py.next.iterator.transforms import (
    concat_where,
    dead_code_elimination,
    expand_tuple_maps,
    inline_fundefs,
    trace_shifts,
)
from gt4py.next.iterator.transforms.merge_let import MergeLet
from gt4py.next.iterator.transforms.normalize_shifts import NormalizeShifts
from gt4py.next.iterator.type_system import inference as itir_type_inference
from gt4py.next.type_system import type_info, type_specifications as ts


class Unbounded(eve.StrEnum):
    """Displacement whose extent is not a compile-time constant."""

    #: The offset is computed from field values (e.g. `as_offset`).
    DYNAMIC = "dynamic"
    #: A scan reads the whole column along its axis.
    COLUMN = "column"


Displacement: TypeAlias = int | float | Unbounded


@dataclasses.dataclass(frozen=True)
class Access:
    """
    One way an output point reaches a read of an input.

    Attributes:
        chain: Names of the neighbor connectivities applied, from the output point outwards.
        displacement: Total displacement per (non-staggered) cartesian dimension, sorted by
            dimension name. On unstructured grids these are the vertical offsets. Half-integer
            values come from staggered dimensions.
    """

    chain: tuple[str, ...] = ()
    displacement: tuple[tuple[str, Displacement], ...] = ()

    def shifted(
        self,
        shift: tuple[itir.OffsetLiteral | itir.CartesianOffset | trace_shifts.Sentinel, ...],
        offset_provider_type: common.OffsetProviderType,
    ) -> Access:
        chain = list(self.chain)
        displacement = dict(self.displacement)
        for offset, index in zip(shift[::2], shift[1::2], strict=True):
            if isinstance(offset, itir.CartesianOffset):
                source = common.Dimension(offset.domain.value, offset.domain.kind)
                target = common.Dimension(offset.codomain.value, offset.codomain.kind)
            else:
                assert isinstance(offset, itir.OffsetLiteral) and isinstance(offset.value, str)
                offset_type = common.get_offset_type(offset_provider_type, offset.value)
                if isinstance(offset_type, common.NeighborConnectivityType):
                    chain.append(offset.value)
                    continue
                assert isinstance(offset_type, common.Dimension)
                source = target = offset_type
            step: Displacement
            if index is trace_shifts.Sentinel.VALUE:
                step = Unbounded.DYNAMIC
            else:
                assert isinstance(index, itir.OffsetLiteral) and isinstance(index.value, int)
                step = index.value + 0.5 * (
                    common.is_staggered(source) - common.is_staggered(target)
                )
            name = common.as_non_staggered(target).value
            displacement[name] = _add(displacement.get(name, 0), step)
        return Access(tuple(chain), _sorted_displacement(displacement))

    def with_column(self, dims: list[common.Dimension]) -> Access:
        displacement = dict(self.displacement)
        for dim in dims:
            name = common.as_non_staggered(dim).value
            displacement[name] = _add(displacement.get(name, 0), Unbounded.COLUMN)
        return Access(self.chain, _sorted_displacement(displacement))


def _add(a: Displacement, b: Displacement) -> Displacement:
    if Unbounded.DYNAMIC in (a, b):
        return Unbounded.DYNAMIC
    if Unbounded.COLUMN in (a, b):
        return Unbounded.COLUMN
    assert not isinstance(a, Unbounded) and not isinstance(b, Unbounded)
    total = a + b
    return int(total) if float(total).is_integer() else total


def _sorted_displacement(
    displacement: dict[str, Displacement],
) -> tuple[tuple[str, Displacement], ...]:
    return tuple(sorted((k, v) for k, v in displacement.items() if v != 0))


#: The accesses of a value, nested like the value if it is a tuple. An empty set means never read.
Accesses: TypeAlias = frozenset[Access] | tuple["Accesses", ...]
AccessesBySymbol: TypeAlias = dict[str, Accesses]


def _union(*accesses: Accesses) -> Accesses:
    if not any(isinstance(a, tuple) for a in accesses):
        return frozenset().union(*accesses)  # type: ignore[arg-type]
    length = max(len(a) for a in accesses if isinstance(a, tuple))
    promoted = [
        (*a, *([frozenset()] * (length - len(a)))) if isinstance(a, tuple) else (a,) * length
        for a in accesses
    ]
    return tuple(_union(*elements) for elements in zip(*promoted))


def flatten(accesses: Accesses) -> frozenset[Access]:
    return frozenset().union(*utils.flatten_nested_tuple((accesses,)))


def _merge(a: AccessesBySymbol, b: AccessesBySymbol) -> AccessesBySymbol:
    result = dict(a)
    for key, value in b.items():
        result[key] = _union(result[key], value) if key in result else value
    return result


def _trace_as_fieldop(
    expr: itir.FunCall, accesses: Accesses, offset_provider_type: common.OffsetProviderType
) -> AccessesBySymbol:
    stencil, inputs = expr.fun.args[0], expr.args  # type: ignore[attr-defined]  # ensured by caller
    outer = flatten(accesses)
    if cpm.is_call_to(stencil, "scan"):
        assert isinstance(expr.type, ts.TypeSpec)
        column_dims = [
            dim
            for el_type in type_info.primitive_constituents(expr.type)
            for dim in type_info.extract_dims(el_type)
            if dim.kind == common.DimensionKind.VERTICAL
        ]
        outer = frozenset(access.with_column(column_dims) for access in outer)
    shifts_per_input = trace_shifts.trace_stencil(stencil, num_args=len(inputs))

    result: AccessesBySymbol = {}
    for input_, shifts in zip(inputs, shifts_per_input, strict=True):
        input_accesses = frozenset(
            access.shifted(shift, offset_provider_type) for access in outer for shift in shifts
        )
        result = _merge(result, _trace_expr(input_, input_accesses, offset_provider_type))
    return result


def _trace_expr(
    expr: itir.Expr, accesses: Accesses, offset_provider_type: common.OffsetProviderType
) -> AccessesBySymbol:
    if isinstance(expr, itir.SymRef):
        return {str(expr.id): accesses}
    if isinstance(expr, (itir.Literal, itir.AxisLiteral, itir.OffsetLiteral)):
        return {}
    if cpm.is_applied_as_fieldop(expr):
        return _trace_as_fieldop(expr, accesses, offset_provider_type)
    if cpm.is_let(expr):
        assert isinstance(expr.fun, itir.Lambda)
        body_accesses = _trace_expr(expr.fun.expr, accesses, offset_provider_type)
        params = [str(param.id) for param in expr.fun.params]
        result = {k: v for k, v in body_accesses.items() if k not in params}
        for param, arg in zip(params, expr.args, strict=True):
            if param in body_accesses and flatten(body_accesses[param]):
                result = _merge(
                    result, _trace_expr(arg, body_accesses[param], offset_provider_type)
                )
        return result
    if cpm.is_call_to(expr, "make_tuple"):
        elements = (
            accesses
            if isinstance(accesses, tuple)
            else tuple(accesses for _ in range(len(expr.args)))
        )
        result = {}
        for i, arg in enumerate(expr.args):
            if i < len(elements):
                result = _merge(result, _trace_expr(arg, elements[i], offset_provider_type))
        return result
    if cpm.is_call_to(expr, "tuple_get"):
        index_literal, tuple_expr = expr.args
        assert isinstance(index_literal, itir.Literal)
        index = int(index_literal.value)
        tuple_accesses = tuple(accesses if i == index else frozenset() for i in range(index + 1))
        return _trace_expr(tuple_expr, tuple_accesses, offset_provider_type)
    if cpm.is_call_to(expr, "if_"):
        cond, true_branch, false_branch = expr.args
        result = _trace_expr(cond, flatten(accesses), offset_provider_type)
        for branch in (true_branch, false_branch):
            result = _merge(result, _trace_expr(branch, accesses, offset_provider_type))
        return result
    if cpm.is_call_to(expr, "concat_where"):
        _, true_branch, false_branch = expr.args
        return _merge(
            _trace_expr(true_branch, accesses, offset_provider_type),
            _trace_expr(false_branch, accesses, offset_provider_type),
        )
    if cpm.is_call_to(expr, "broadcast"):
        return _trace_expr(expr.args[0], accesses, offset_provider_type)
    if cpm.is_call_to(expr, ("index", "unstructured_domain", "cartesian_domain")):
        return {}
    if cpm.is_call_to(expr, builtins.ARITHMETIC_BUILTINS) or cpm.is_call_to(
        expr, (*builtins.TYPE_BUILTINS, "cast_")
    ):
        result = {}
        for arg in expr.args:
            result = _merge(result, _trace_expr(arg, flatten(accesses), offset_provider_type))
        return result
    raise ValueError(f"Unsupported expression: '{expr}'.")


def _output_paths(type_: ts.TypeSpec, path: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    if isinstance(type_, ts.TupleType):
        return [p for i, t in enumerate(type_.types) for p in _output_paths(t, (*path, i))]
    return [path]


def _accesses_at(type_: ts.TypeSpec, path: tuple[int, ...]) -> Accesses:
    if not path:
        return frozenset({Access()})
    assert isinstance(type_, ts.TupleType)
    return tuple(
        _accesses_at(t, path[1:]) if i == path[0] else frozenset()
        for i, t in enumerate(type_.types)
    )


def trace_program(
    program: itir.Program, offset_provider_type: common.OffsetProviderType
) -> dict[tuple[int, ...], AccessesBySymbol]:
    """
    Trace, for every output of `program`, how each parameter is reached from an output point.

    The program must consist of a single `SetAt` statement, as generated for a field operator.

    Returns:
        A mapping from the path of each output (tuple indices, `()` for a non-tuple output) to a
        mapping from parameter name to its `Accesses`. Parameters not read for an output are
        omitted.
    """
    uids = utils.IDGeneratorPool()
    program = MergeLet().visit(program)
    program = inline_fundefs.InlineFundefs().visit(program)
    program = inline_fundefs.prune_unreferenced_fundefs(program)
    program = NormalizeShifts().visit(program)
    program = concat_where.expand_tuple_args(program, offset_provider_type=offset_provider_type)  # type: ignore[assignment]  # always an itir.Program
    program = expand_tuple_maps.ExpandTupleMaps.apply(
        program, uids=uids, offset_provider_type=offset_provider_type
    )
    program = dead_code_elimination.dead_code_elimination(
        program, uids=uids, offset_provider_type=offset_provider_type
    )
    program = itir_type_inference.infer(program, offset_provider_type=offset_provider_type)

    if len(program.body) != 1 or not isinstance(program.body[0], itir.SetAt):
        raise ValueError("Expected a program consisting of a single 'SetAt' statement.")
    expr = program.body[0].expr
    assert isinstance(expr.type, ts.TypeSpec)
    params = {str(param.id) for param in program.params}

    result = {}
    for path in _output_paths(expr.type):
        accessed = _trace_expr(expr, _accesses_at(expr.type, path), offset_provider_type)
        result[path] = {k: v for k, v in accessed.items() if k in params and flatten(v)}
    return result
