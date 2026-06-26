# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Callable, ClassVar, Optional, Union

from gt4py.eve import Coerced, SymbolName, datamodels
from gt4py.eve.traits import SymbolTableTrait, ValidatedSymbolTableTrait
from gt4py.next import common
from gt4py.next.iterator import builtins
from gt4py.next.program_processors.codegens.gtfn.gtfn_im_ir import ImperativeFunctionDefinition
from gt4py.next.program_processors.codegens.gtfn.gtfn_ir_common import Expr, Node, Sym, SymRef


class UnaryExpr(Expr):
    op: str
    expr: Expr


class BinaryExpr(Expr):
    op: str
    lhs: Expr
    rhs: Expr


class TernaryExpr(Expr):
    cond: Expr
    true_expr: Expr
    false_expr: Expr


class CastExpr(Expr):
    obj_expr: Expr
    new_dtype: SymRef


class Literal(Expr):
    value: str
    type: str


class IntegralConstant(Expr):
    value: int  # generalize to other types if needed


class OffsetLiteral(Expr):
    value: Union[int, str]


class Lambda(Expr, SymbolTableTrait):
    params: list[Sym]
    expr: Expr


class FunCall(Expr):
    fun: Expr  # VType[Callable]
    args: list[Expr]


class FunctionDefinition(Node, SymbolTableTrait):
    id: Coerced[SymbolName]
    params: list[Sym]
    expr: Expr


class ScanPassDefinition(Node, SymbolTableTrait):
    id: Coerced[SymbolName]
    params: list[Sym]
    expr: Expr
    forward: bool
    projector: Optional[Lambda]


class TaggedValues(Node):
    tags: list[Expr]
    values: list[Expr]


class CartesianDomain(Node):
    tagged_sizes: TaggedValues
    tagged_offsets: TaggedValues


class UnstructuredDomain(Node):
    tagged_sizes: TaggedValues
    tagged_offsets: TaggedValues
    connectivities: list[SymRef]  # SymRef to offset declaration


class Backend(Node):
    domain: Union[SymRef, CartesianDomain, UnstructuredDomain]
    #: per-executor K-coarsening: True → use the loop-blocked backend, False → the plain one.
    loop_blocked: bool = False


def _is_tuple_expr_of(pred: Callable[[Expr], bool], expr: Expr) -> bool:
    if (
        isinstance(expr, FunCall)
        and isinstance(expr.fun, SymRef)
        and expr.fun.id == "tuple_get"
        and len(expr.args) == 2
        and _is_tuple_expr_of(pred, expr.args[1])
    ):
        return True
    if (
        isinstance(expr, FunCall)
        and isinstance(expr.fun, SymRef)
        and expr.fun.id == "make_tuple"
        and all(_is_tuple_expr_of(pred, arg) for arg in expr.args)
    ):
        return True
    return pred(expr)


class SidComposite(Expr):
    values: list[Expr]

    @datamodels.validator("values")
    def _values_validator(
        self: datamodels.DataModelTP, attribute: datamodels.Attribute, value: list[Expr]
    ) -> None:
        if not all(
            isinstance(el, (SidFromScalar, SidComposite))
            or _is_tuple_expr_of(
                lambda expr: (
                    isinstance(expr, (SymRef, Literal))
                    or (isinstance(expr, FunCall) and expr.fun == SymRef(id="index"))
                ),
                el,
            )
            for el in value
        ):
            raise ValueError(
                "Only 'SymRef', 'Literal', tuple expr thereof, 'SidFromScalar', or 'SidComposite' allowed."
            )


def _might_be_scalar_expr(expr: Expr) -> bool:
    if isinstance(expr, BinaryExpr):
        return all(_is_tuple_expr_of(_might_be_scalar_expr, arg) for arg in (expr.lhs, expr.rhs))
    if isinstance(expr, UnaryExpr):
        return _is_tuple_expr_of(_might_be_scalar_expr, expr.expr)
    if (
        isinstance(expr, FunCall)
        and isinstance(expr.fun, SymRef)
        and expr.fun.id in ARITHMETIC_BUILTINS
    ):
        return all(_might_be_scalar_expr(arg) for arg in expr.args)
    if isinstance(expr, CastExpr):
        return _might_be_scalar_expr(expr.obj_expr)
    if _is_tuple_expr_of(lambda e: isinstance(e, (SymRef, Literal)), expr):
        return True
    return False


class SidFromScalar(Expr):
    arg: Expr

    @datamodels.validator("arg")
    def _arg_validator(
        self: datamodels.DataModelTP, attribute: datamodels.Attribute, value: Expr
    ) -> None:
        if not _might_be_scalar_expr(value):
            raise ValueError(
                "Only 'SymRef', 'Literal', arithmetic op or tuple expr thereof allowed."
            )


class Stmt(Node):
    pass


class StencilExecution(Stmt):
    backend: Backend
    stencil: SymRef
    output: Union[SymRef, SidComposite]
    inputs: list[Union[SymRef, SidComposite, SidFromScalar, FunCall]]

    @datamodels.validator("inputs")
    def _arg_validator(
        self: datamodels.DataModelTP, attribute: datamodels.Attribute, inputs: list[Expr]
    ) -> None:
        for inp in inputs:
            if not _is_tuple_expr_of(
                lambda expr: (
                    isinstance(expr, (SymRef, SidComposite, SidFromScalar))
                    or (
                        isinstance(expr, FunCall)
                        and isinstance(expr.fun, SymRef)
                        and expr.fun.id == "index"
                    )
                ),
                inp,
            ):
                raise ValueError(
                    "Only 'SymRef', 'SidComposite', 'SidFromScalar', 'index' call or tuple expr thereof allowed."
                )


class ScanTailDefinition(Node, SymbolTableTrait):
    #: Top-level struct (template on the backend arg-offset) for a folded post-scan consumer.
    #: `res`/`acc`/`surface` are the scan's per-level register values (result at K, result at
    #: K+1, seed); `input_params` are bound to iterators built from the other SID args; each
    #: `outputs` entry writes one SID arg with the (rewritten) consumer expression.
    id: Coerced[SymbolName]
    res: Sym
    acc: Sym
    surface: Sym
    #: (param bound to an iterator at the current level, raw arg index it reads)
    input_params: list[tuple[Sym, int]]
    outputs: list[tuple[int, Expr]]  # (raw output arg index, expr over res/acc/surface/inputs)


class ScanTail(Node):
    #: Wiring for a folded post-scan consumer. `body_*_trim` bound the levels where the scan
    #: recurrence runs (its original K-range); `tail_*_trim` bound where the consumer writes (its
    #: K-range, which may be wider — it can handle the column-top level with a scan-free fallback).
    definition: SymRef
    inputs: list[int]  # raw arg indices for input_params (passed as the scan_with_tail Ins)
    body_top_trim: int = 0
    body_bot_trim: int = 0
    tail_top_trim: int = 0
    tail_bot_trim: int = 0


class MapColumnTailDefinition(Node, SymbolTableTrait):
    #: Top-level struct (template on the backend arg-offset) for the consumer of a fused map-chain.
    #: `cur`/`prev` are the producer's per-level register values (T at K, T at K-1); `input_params`
    #: are bound to iterators built from the consumer's other SID args; each `outputs` entry writes
    #: one SID arg with the (rewritten) consumer expression.
    id: Coerced[SymbolName]
    cur: Sym
    prev: Sym
    #: (param bound to an iterator at the current level, raw arg index it reads)
    input_params: list[tuple[Sym, int]]
    outputs: list[tuple[int, Expr]]  # (raw output arg index, expr over cur/prev/inputs)


class MapColumnExecution(Stmt):
    #: One fused column kernel for a same-domain cell-local producer->consumer map-chain with a
    #: single Koff[-1] edge. The producer stencil writes T (held in a register `cur`); the consumer
    #: reads T at center (`cur`) and Koff[-1] (`prev`, a one-deep history register) and writes its
    #: own outputs — no DRAM round-trip for the chain edge. Lowers to a map_column_stage.
    backend: Backend
    producer: SymRef  # FunctionDefinition id
    producer_output: int  # raw arg index T is written to
    producer_inputs: list[int]  # raw arg indices of the producer's inputs
    consumer: SymRef  # MapColumnTailDefinition id
    args: list[Expr]
    axis: SymRef


class KoffWindowTailDefinition(Node, SymbolTableTrait):
    #: Top-level struct (template on the backend arg-offset) for a column-ified single stencil
    #: (increment-2b). `windows` are the (cur, prev) register pairs holding an ALREADY-materialized
    #: input field at level K and K-1; the body reads them from registers instead of two DRAM loads
    #: (center + Koff[-1]). `input_params` bind the remaining SID args at the current level; each
    #: `outputs` entry writes one SID arg with the rewritten body expression.
    id: Coerced[SymbolName]
    #: (cur Sym, prev Sym) per windowed input, in window order
    windows: list[tuple[Sym, Sym]]
    #: (param bound to an iterator at the current level, raw arg index it reads)
    input_params: list[tuple[Sym, int]]
    outputs: list[tuple[int, Expr]]  # (raw output arg index, expr over windows/inputs)


class KoffWindowExecution(Stmt):
    #: One column kernel for a single multi-output cell-local stencil whose inputs WinIns are read
    #: at center + Koff[-1] only (history). The K-loop runs in-thread; each windowed field is loaded
    #: once per level (`cur`) and its K-1 value carried in a one-deep history register (`prev`), so
    #: the Koff[-1] re-read never round-trips DRAM. The fields stay materialized (their producer is
    #: untouched). Lowers to a koff_window_column_stage.
    backend: Backend
    consumer: SymRef  # KoffWindowTailDefinition id
    window_inputs: list[int]  # raw arg indices of the windowed (center+Koff[-1]) input fields
    args: list[Expr]
    axis: SymRef


class MapWindowTailDefinition(Node, SymbolTableTrait):
    #: Consumer of a fused map-chain that ALSO windows N already-materialized inputs (increment-2b
    #: "keep-and-write + extra windows"). `cur`/`prev` are the producer's per-level register values
    #: (T at K, T at K-1); `windows` are (cur, prev) register pairs for each windowed input (its
    #: value at K and K-1, served from registers instead of a second DRAM load). `input_params` bind
    #: the remaining SID args at the current level; each `outputs` entry writes one SID arg.
    id: Coerced[SymbolName]
    cur: Sym
    prev: Sym
    #: (cur Sym, prev Sym) per windowed input, in window order
    windows: list[tuple[Sym, Sym]]
    input_params: list[tuple[Sym, int]]
    outputs: list[tuple[int, Expr]]


class MapWindowExecution(Stmt):
    #: One fused column kernel for a cell-local producer->consumer map-chain whose consumer ALSO
    #: reads N already-materialized inputs at center + Koff[-1]. The producer value is held in a
    #: register (`cur`, K-1 in `prev`); each windowed field is loaded once per level and its K-1
    #: value carried in a register too — so neither the chain edge nor the windowed Koff[-1] re-reads
    #: round-trip DRAM. The producer output is (maybe) stored to SID (producer_output<0 drops it).
    #: Lowers to a map_window_column_stage.
    backend: Backend
    producer: SymRef  # FunctionDefinition id
    producer_output: int  # raw arg index T is written to (<0 = drop)
    producer_inputs: list[int]  # raw arg indices of the producer's inputs
    consumer: SymRef  # MapWindowTailDefinition id
    window_inputs: list[int]  # raw arg indices of the windowed (center+Koff[-1]) consumer inputs
    args: list[Expr]
    axis: SymRef


class Scan(Node):
    function: SymRef
    output: int
    inputs: list[int]
    init: Expr
    #: For single-kernel scan merges only: vertical levels to skip at the top/bottom of the
    #: merged (union) column so this scan still runs over exactly its original K-range.
    top_trim: int = 0
    bot_trim: int = 0
    #: When set, the scan output is not written to a SID; instead this tail consumes it per level.
    tail: Optional[ScanTail] = None


class ScanExecution(Stmt):
    backend: Backend
    scans: list[Scan]
    args: list[Expr]
    axis: SymRef
    #: When set, the scans are emitted as one fused kernel (merged_seeded_column_stage) via
    #: assign_merged_scans, instead of one column-stage launch per scan.
    merged_kernel: bool = False


class IfStmt(Stmt):
    cond: Expr
    true_branch: list[Stmt]
    false_branch: list[Stmt]


class TemporaryAllocation(Node):
    id: SymbolName
    dtype: str
    domain: Union[SymRef, CartesianDomain, UnstructuredDomain]


GTFN_BUILTINS = [
    "deref",
    "shift",
    "make_tuple",
    "tuple_get",
    "can_deref",
    "cartesian_domain",
    "unstructured_domain",
    "get_domain_range",
    "named_range",
    "reduce",
    "index",
    # combined branchless skip-reduce + neighbor-row hoist (hoist_neighbor_row.py)
    "neighbor_row",
    "horizontal_shift_to",
]
ARITHMETIC_BUILTINS = builtins.ARITHMETIC_BUILTINS
TYPEBUILTINS = builtins.TYPE_BUILTINS

BUILTINS = {*GTFN_BUILTINS, *ARITHMETIC_BUILTINS, *TYPEBUILTINS}


class TagDefinition(Node):
    name: Sym
    alias: Optional[Union[str, SymRef]] = None


class Program(Node, ValidatedSymbolTableTrait):
    id: SymbolName
    params: list[Sym]
    function_definitions: list[
        Union[
            FunctionDefinition,
            ScanPassDefinition,
            ImperativeFunctionDefinition,
            ScanTailDefinition,
            MapColumnTailDefinition,
            KoffWindowTailDefinition,
            MapWindowTailDefinition,
        ]
    ]
    executions: list[Stmt]
    offset_definitions: list[TagDefinition]
    grid_type: common.GridType
    temporaries: list[TemporaryAllocation]

    _NODE_SYMBOLS_: ClassVar[list[Sym]] = [Sym(id=name) for name in BUILTINS]
