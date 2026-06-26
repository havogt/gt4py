# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import dataclasses
import functools
import os
from typing import Any, Callable, ClassVar, Iterable, Optional, Type, TypeGuard, Union

import gt4py.eve as eve
from gt4py.eve.concepts import SymbolName
from gt4py.next import common, utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    ir_makers as im,
    misc as ir_utils_misc,
)
from gt4py.next.iterator.type_system import inference as itir_type_inference
from gt4py.next.otf import cpp_utils
from gt4py.next.program_processors.codegens.gtfn.gtfn_ir import (
    Backend,
    BinaryExpr,
    CartesianDomain,
    CastExpr,
    FunCall,
    FunctionDefinition,
    IfStmt,
    IntegralConstant,
    KoffWindowExecution,
    KoffWindowTailDefinition,
    Lambda,
    Literal,
    MapColumnExecution,
    MapColumnTailDefinition,
    MapWindowExecution,
    MapWindowTailDefinition,
    OffsetLiteral,
    Program,
    Scan,
    ScanExecution,
    ScanPassDefinition,
    ScanTail,
    ScanTailDefinition,
    SidComposite,
    SidFromScalar,
    StencilExecution,
    Stmt,
    TagDefinition,
    TaggedValues,
    TemporaryAllocation,
    TernaryExpr,
    UnaryExpr,
    UnstructuredDomain,
)
from gt4py.next.program_processors.codegens.gtfn.gtfn_ir_common import Expr, Node, Sym, SymRef
from gt4py.next.type_system import type_info, type_specifications as ts


_vertical_dimension = "gtfn::unstructured::dim::vertical"
_horizontal_dimension = "gtfn::unstructured::dim::horizontal"


def _is_tuple_of_ref_or_literal(expr: itir.Expr) -> bool:
    if (
        isinstance(expr, itir.FunCall)
        and isinstance(expr.fun, itir.SymRef)
        and expr.fun.id == "tuple_get"
        and len(expr.args) == 2
        and _is_tuple_of_ref_or_literal(expr.args[1])
    ):
        return True
    if (
        isinstance(expr, itir.FunCall)
        and isinstance(expr.fun, itir.SymRef)
        and expr.fun.id == "make_tuple"
        and all(_is_tuple_of_ref_or_literal(arg) for arg in expr.args)
    ):
        return True
    if isinstance(expr, (itir.SymRef, itir.Literal)):
        return True
    return False


def _get_domains(nodes: Iterable[itir.Stmt]) -> Iterable[itir.FunCall]:
    result = set()
    for node in nodes:
        result |= node.walk_values().if_isinstance(itir.SetAt).getattr("domain").to_set()
    return result


def _name_from_named_range(named_range_call: itir.FunCall) -> str:
    assert isinstance(named_range_call, itir.FunCall) and named_range_call.fun == itir.SymRef(
        id="named_range"
    )
    assert isinstance(named_range_call.args[0], itir.AxisLiteral)
    return named_range_call.args[0].value


def _collect_dimensions_from_domain(
    body: Iterable[itir.Stmt],
) -> dict[str, TagDefinition]:
    domains = _get_domains(body)
    offset_definitions = {}
    for domain in domains:
        if domain.fun == itir.SymRef(id="cartesian_domain"):
            for nr in domain.args:
                assert isinstance(nr, itir.FunCall)
                dim_name = _name_from_named_range(nr)
                offset_definitions[dim_name] = TagDefinition(name=Sym(id=dim_name))
        elif domain.fun == itir.SymRef(id="unstructured_domain"):
            if len(domain.args) > 2:
                raise ValueError("Unstructured_domain must not have more than 2 arguments.")
            if len(domain.args) > 0:
                horizontal_range = domain.args[0]
                assert isinstance(horizontal_range, itir.FunCall)
                horizontal_name = _name_from_named_range(horizontal_range)
                offset_definitions[horizontal_name] = TagDefinition(
                    name=Sym(id=horizontal_name), alias=_horizontal_dimension
                )
            if len(domain.args) > 1:
                vertical_range = domain.args[1]
                assert isinstance(vertical_range, itir.FunCall)
                vertical_name = _name_from_named_range(vertical_range)
                offset_definitions[vertical_name] = TagDefinition(
                    name=Sym(id=vertical_name), alias=_vertical_dimension
                )
        else:
            raise AssertionError(
                "Expected either a call to 'cartesian_domain' or to 'unstructured_domain'."
            )
    return offset_definitions


def _collect_offset_definitions(
    node: itir.Node,
    grid_type: common.GridType,
    offset_provider_type: common.OffsetProviderType,
) -> dict[str, TagDefinition]:
    used_offset_tags: set[str] = (
        node.walk_values()
        .if_isinstance(itir.OffsetLiteral)
        .filter(lambda offset_literal: isinstance(offset_literal.value, str))
        .getattr("value")
    ).to_set()
    # implicit offsets don't occur in the `offset_provider_type`, get them from the used offset tags
    offset_provider_type = {
        offset_name: common.get_offset_type(offset_provider_type, offset_name)
        for offset_name in used_offset_tags
    } | {**offset_provider_type}
    offset_definitions = {}

    for offset_name, dim_or_connectivity_type in offset_provider_type.items():
        if isinstance(dim_or_connectivity_type, common.Dimension):
            dim: common.Dimension = dim_or_connectivity_type
            if grid_type == common.GridType.CARTESIAN:
                # create alias from offset to dimension
                offset_definitions[dim.value] = TagDefinition(name=Sym(id=dim.value))
                offset_definitions[offset_name] = TagDefinition(
                    name=Sym(id=offset_name), alias=SymRef(id=dim.value)
                )
            else:
                assert grid_type == common.GridType.UNSTRUCTURED
                if not dim.kind == common.DimensionKind.VERTICAL:
                    raise ValueError(
                        "Mapping an offset to a horizontal dimension in unstructured is not allowed."
                    )
                # create alias from vertical offset to vertical dimension
                offset_definitions[dim.value] = TagDefinition(
                    name=Sym(id=dim.value), alias=_vertical_dimension
                )
                offset_definitions[offset_name] = TagDefinition(
                    name=Sym(id=offset_name), alias=SymRef(id=dim.value)
                )
        elif isinstance(
            connectivity_type := dim_or_connectivity_type, common.NeighborConnectivityType
        ):
            assert grid_type == common.GridType.UNSTRUCTURED
            offset_definitions[offset_name] = TagDefinition(name=Sym(id=offset_name))
            if offset_name != connectivity_type.neighbor_dim.value:
                offset_definitions[connectivity_type.neighbor_dim.value] = TagDefinition(
                    name=Sym(id=connectivity_type.neighbor_dim.value)
                )

            for dim in [connectivity_type.source_dim, connectivity_type.codomain]:
                if dim.kind != common.DimensionKind.HORIZONTAL:
                    raise NotImplementedError()
                offset_definitions[dim.value] = TagDefinition(
                    name=Sym(id=dim.value), alias=_horizontal_dimension
                )
        else:
            raise AssertionError(
                "Elements of offset provider need to be either 'Dimension' or 'Connectivity'."
            )
    return offset_definitions


def _literal_as_integral_constant(node: itir.Literal) -> IntegralConstant:
    assert type_info.is_integer(node.type)
    return IntegralConstant(value=int(node.value))


def _is_scan(node: itir.Node) -> TypeGuard[itir.FunCall]:
    return isinstance(node, itir.FunCall) and node.fun == itir.SymRef(id="scan")


def _static_int(expr: Expr) -> Optional[int]:
    if isinstance(expr, IntegralConstant):
        return expr.value
    if isinstance(expr, Literal):
        try:
            return int(expr.value)
        except ValueError:
            return None
    if isinstance(expr, BinaryExpr):
        lhs, rhs = _static_int(expr.lhs), _static_int(expr.rhs)
        if lhs is None or rhs is None:
            return None
        return {"+": lhs + rhs, "-": lhs - rhs, "*": lhs * rhs}.get(expr.op)
    return None


def _static_k_range(domain: Any, axis: SymRef) -> Optional[tuple[int, int]]:
    """(start, stop) of the `axis` dimension if both are static integers, else None."""
    if not isinstance(domain, UnstructuredDomain):
        return None
    tags = domain.tagged_offsets.tags
    for i, tag in enumerate(tags):
        if getattr(tag, "id", getattr(tag, "value", None)) != axis.id:
            continue
        start = _static_int(domain.tagged_offsets.values[i])
        size = domain.tagged_sizes.values[i]
        # size is built as BinaryExpr('-', stop, start); recover stop directly when possible
        stop = (
            _static_int(size.lhs)
            if isinstance(size, BinaryExpr) and size.op == "-"
            else (None if (s := _static_int(size)) is None or start is None else start + s)
        )
        if start is None or stop is None:
            return None
        return start, stop
    return None


def _symref_ids(expr: Expr) -> set[str]:
    return set(expr.pre_walk_values().if_isinstance(SymRef).getattr("id").to_set())


def _linear_terms(expr: Expr, sign: int, pos: list, neg: list, const: list) -> bool:
    """Flatten a +/- expression into multisets of positive/negative non-constant terms plus a
    running constant. Returns False if it hits a node it can't treat as a sum (e.g. a product)."""
    c = _static_int(expr)
    if c is not None:
        const[0] += sign * c
        return True
    if isinstance(expr, BinaryExpr) and expr.op in ("+", "-"):
        if not _linear_terms(expr.lhs, sign, pos, neg, const):
            return False
        return _linear_terms(expr.rhs, sign if expr.op == "+" else -sign, pos, neg, const)
    (pos if sign > 0 else neg).append(expr)
    return True


def _static_diff(a: Expr, b: Expr) -> Optional[int]:
    """Value of `a - b` when the non-constant terms cancel, else None."""
    pos: list = []
    neg: list = []
    const = [0]
    if not _linear_terms(a, 1, pos, neg, const):
        return None
    if not _linear_terms(b, -1, pos, neg, const):
        return None

    def cancel(p: list, n: list) -> None:
        for term in list(p):
            if term in n:
                p.remove(term)
                n.remove(term)

    cancel(pos, neg)
    if pos or neg:
        return None
    return const[0]


def _domain_k_bounds(domain: Any, axis: SymRef) -> Optional[tuple[Expr, Expr]]:
    """(start_expr, stop_expr = start + size) of the `axis` dimension, or None if absent."""
    if not isinstance(domain, (UnstructuredDomain, CartesianDomain)):
        return None
    tags = domain.tagged_offsets.tags
    for i, tag in enumerate(tags):
        if getattr(tag, "id", getattr(tag, "value", None)) != axis.id:
            continue
        start = domain.tagged_offsets.values[i]
        size = domain.tagged_sizes.values[i]
        return start, BinaryExpr(op="+", lhs=start, rhs=size)
    return None


def _strip_k_dim(domain: Any, axis: SymRef) -> Any:
    """Copy of `domain` with the `axis` dimension removed (for comparing the non-K extents)."""
    if not isinstance(domain, (UnstructuredDomain, CartesianDomain)):
        return domain
    keep = [
        i
        for i, tag in enumerate(domain.tagged_offsets.tags)
        if getattr(tag, "id", getattr(tag, "value", None)) != axis.id
    ]
    off = TaggedValues(
        tags=[domain.tagged_offsets.tags[i] for i in keep],
        values=[domain.tagged_offsets.values[i] for i in keep],
    )
    siz = TaggedValues(
        tags=[domain.tagged_sizes.tags[i] for i in keep],
        values=[domain.tagged_sizes.values[i] for i in keep],
    )
    if isinstance(domain, UnstructuredDomain):
        return UnstructuredDomain(
            tagged_sizes=siz, tagged_offsets=off, connectivities=domain.connectivities
        )
    return CartesianDomain(tagged_sizes=siz, tagged_offsets=off)


def _is_deref_of(expr: Expr, sym: str) -> bool:
    return (
        isinstance(expr, FunCall)
        and isinstance(expr.fun, SymRef)
        and expr.fun.id == "deref"
        and len(expr.args) == 1
        and isinstance(expr.args[0], SymRef)
        and expr.args[0].id == sym
    )


def _is_deref_koff1_of(expr: Expr, sym: str) -> bool:
    # deref(shift(<sym>, Koff, 1))  -> the K+1 (acc) read for a backward scan
    if not (isinstance(expr, FunCall) and isinstance(expr.fun, SymRef) and expr.fun.id == "deref"):
        return False
    inner = expr.args[0]
    return (
        isinstance(inner, FunCall)
        and isinstance(inner.fun, SymRef)
        and inner.fun.id == "shift"
        and len(inner.args) == 3
        and isinstance(inner.args[0], SymRef)
        and inner.args[0].id == sym
        and isinstance(inner.args[2], OffsetLiteral)
        and inner.args[2].value == 1
    )


@dataclasses.dataclass
class _ScanOutputRewriter(eve.NodeTranslator):
    """Replace the scan-output param's level reads in a consumer body:
    deref(<sym>) -> res, deref(shift(<sym>, Koff, 1)) -> acc. Sets `ok=False` if the param
    appears in any other position (a shift we can't fold: Koff[-1], a connectivity, etc.)."""

    sym: str
    res: str
    acc: str
    ok: bool = True

    def visit_FunCall(self, node: FunCall) -> Expr:
        if _is_deref_of(node, self.sym):
            return SymRef(id=self.res)
        if _is_deref_koff1_of(node, self.sym):
            return SymRef(id=self.acc)
        return self.generic_visit(node)

    def visit_SymRef(self, node: SymRef) -> SymRef:
        # Any bare reference to the scan-output param that escaped the deref patterns above means
        # an access we can't fold (e.g. deref(shift(sym, ...)) with a different offset).
        if node.id == self.sym:
            self.ok = False
        return node


def _is_deref_koff_of(expr: Expr, sym: str, off: int) -> bool:
    # deref(shift(<sym>, Koff, off))
    if not (isinstance(expr, FunCall) and isinstance(expr.fun, SymRef) and expr.fun.id == "deref"):
        return False
    inner = expr.args[0]
    return (
        isinstance(inner, FunCall)
        and isinstance(inner.fun, SymRef)
        and inner.fun.id == "shift"
        and len(inner.args) == 3
        and isinstance(inner.args[0], SymRef)
        and inner.args[0].id == sym
        and isinstance(inner.args[2], OffsetLiteral)
        and inner.args[2].value == off
    )


@dataclasses.dataclass
class _RenameSymRef(eve.NodeTranslator):
    mapping: dict[str, str]

    def visit_SymRef(self, node: SymRef) -> SymRef:
        return SymRef(id=self.mapping.get(node.id, node.id))


@dataclasses.dataclass
class _MapChainRewriter(eve.NodeTranslator):
    """Replace a fused map-chain producer-temp's level reads in the consumer body:
    deref(<sym>) -> cur, deref(shift(<sym>, Koff, -1)) -> prev. Sets `ok=False` if the param
    appears in any other position (center / Koff[-1] are the only foldable accesses here)."""

    sym: str
    cur: str
    prev: str
    ok: bool = True

    def visit_FunCall(self, node: FunCall) -> Expr:
        if _is_deref_of(node, self.sym):
            return SymRef(id=self.cur)
        if _is_deref_koff_of(node, self.sym, -1):
            return SymRef(id=self.prev)
        return self.generic_visit(node)

    def visit_SymRef(self, node: SymRef) -> SymRef:
        if node.id == self.sym:
            self.ok = False
        return node


@dataclasses.dataclass
class _KoffWindowRewriter(eve.NodeTranslator):
    """Rewrite, for each windowed input param `sym`, deref(sym) -> cur and
    deref(shift(sym, Koff, -1)) -> prev (increment-2b column-ification of a single stencil). Sets
    `ok=False` for any windowed param accessed in another way (so it cannot be column-windowed)."""

    #: sym -> (cur_name, prev_name)
    mapping: dict[str, tuple[str, str]]
    ok: bool = True

    def visit_FunCall(self, node: FunCall) -> Expr:
        for sym, (cur, prev) in self.mapping.items():
            if _is_deref_of(node, sym):
                return SymRef(id=cur)
            if _is_deref_koff_of(node, sym, -1):
                return SymRef(id=prev)
        return self.generic_visit(node)

    def visit_SymRef(self, node: SymRef) -> SymRef:
        if node.id in self.mapping:
            self.ok = False
        return node


def _koff_window_offsets(expr: Expr, sym: str, koff: str) -> Optional[set[int]]:
    """The set of vertical offsets `sym` is read at in `expr` (0 = center, -1 = Koff[-1]), or None
    if it is accessed in any other way (a different offset, a connectivity, a bare reference).
    Only used to decide whether `sym` is a center+Koff[-1] history window."""
    offsets: set[int] = set()
    ok = [True]

    def walk(n: Any) -> None:
        if not ok[0]:
            return
        if isinstance(n, FunCall):
            if _is_deref_of(n, sym):
                offsets.add(0)
                return
            # deref(shift(sym, <koff>, off))
            if (
                isinstance(n.fun, SymRef)
                and n.fun.id == "deref"
                and isinstance(n.args[0], FunCall)
                and isinstance(n.args[0].fun, SymRef)
                and n.args[0].fun.id == "shift"
            ):
                inner = n.args[0]
                if (
                    isinstance(inner.args[0], SymRef)
                    and inner.args[0].id == sym
                    and len(inner.args) == 3
                    and isinstance(inner.args[1], OffsetLiteral)
                    and inner.args[1].value == koff
                    and isinstance(inner.args[2], OffsetLiteral)
                    and isinstance(inner.args[2].value, int)
                ):
                    offsets.add(inner.args[2].value)
                    return
            for a in n.args:
                walk(a)
            walk(n.fun)
            return
        if isinstance(n, SymRef):
            if n.id == sym:
                ok[0] = False
            return
        if not isinstance(n, Node):
            return
        for child in n.iter_children_values():
            if isinstance(child, (list, tuple)):
                for c in child:
                    walk(c)
            else:
                walk(child)

    walk(expr)
    if not ok[0]:
        return None
    return offsets


def _bool_from_literal(node: itir.Node) -> bool:
    assert isinstance(node, itir.Literal)
    assert type_info.is_logical(node.type) and node.value in ("True", "False")
    return node.value == "True"


class _CannonicalizeUnstructuredDomain(eve.NodeTranslator):
    def visit_FunCall(self, node: itir.FunCall) -> itir.FunCall:
        if node.fun == itir.SymRef(id="unstructured_domain"):
            # for no good reason, the domain arguments for unstructured need to be in order (horizontal, vertical)
            assert isinstance(node.args[0], itir.FunCall)
            first_axis_literal = node.args[0].args[0]
            assert isinstance(first_axis_literal, itir.AxisLiteral)
            if first_axis_literal.kind == itir.DimensionKind.VERTICAL:
                assert len(node.args) == 2
                assert isinstance(node.args[1], itir.FunCall)
                assert isinstance(node.args[1].args[0], itir.AxisLiteral)
                assert node.args[1].args[0].kind == itir.DimensionKind.HORIZONTAL
                return itir.FunCall(fun=node.fun, args=[node.args[1], node.args[0]])
        return node

    @classmethod
    def apply(
        cls,
        node: itir.Program,
    ) -> itir.Program:
        if not isinstance(node, itir.Program):
            raise TypeError(f"Expected a 'Program', got '{type(node).__name__}'.")

        return cls().visit(node)


def _process_elements(
    process_func: Callable[..., Expr],
    obj: Expr,
    type_: ts.TypeSpec,
    *,
    tuple_constructor: Callable[..., Expr] = lambda _, *elements: FunCall(
        fun=SymRef(id="make_tuple"), args=list(elements)
    ),
) -> Expr:
    """
    Recursively applies a processing function to all primitive constituents of a tuple.

    Be aware that this function duplicates the `obj` expression and should hence be used with care.

    Arguments:
        process_func: A callable that takes a gtfn_ir.Expr representing a leaf-element of `obj`.
        obj: The object whose elements are to be transformed.
        type_: A type with the same structure as the elements of `obj`.
        tuple_constructor: By default all transformed tuple elements are just put in a tuple again.
            This can be customized by passing a different Callable.
    """
    assert isinstance(type_, ts.TypeSpec)

    def _gen_constituent_expr(el_type: ts.ScalarType | ts.FieldType, path: tuple[int, ...]) -> Expr:
        # construct expression for the currently processed element
        el = functools.reduce(
            lambda cur_expr, i: FunCall(
                fun=SymRef(id="tuple_get"), args=[IntegralConstant(value=i), cur_expr]
            ),
            path,
            obj,
        )
        return process_func(el, el_type)

    result = type_info.apply_to_primitive_constituents(
        _gen_constituent_expr,
        type_,
        with_path_arg=True,
        tuple_constructor=tuple_constructor,
    )
    return result


@dataclasses.dataclass(frozen=True)
class GTFN_lowering(eve.NodeTranslator, eve.VisitorWithSymbolTableTrait):
    _binary_op_map: ClassVar[dict[str, str]] = {
        "plus": "+",
        "minus": "-",
        "multiplies": "*",
        "divides": "/",
        "eq": "==",
        "not_eq": "!=",
        "less": "<",
        "less_equal": "<=",
        "greater": ">",
        "greater_equal": ">=",
        "and_": "&&",
        "or_": "||",
        "xor_": "^",
        "mod": "%",
    }
    _unary_op_map: ClassVar[dict[str, str]] = {"not_": "!"}

    offset_provider_type: common.OffsetProviderType
    column_axis: Optional[common.Dimension]
    grid_type: common.GridType

    # we use one UID generator per instance such that the generated ids are
    # stable across multiple runs (required for caching to properly work)
    uids: utils.IDGeneratorPool = dataclasses.field(
        init=False, repr=False, default_factory=utils.IDGeneratorPool
    )

    @classmethod
    def apply(
        cls,
        node: itir.Program,
        *,
        offset_provider_type: common.OffsetProviderType,
        column_axis: Optional[common.Dimension],
    ) -> Program:
        if not isinstance(node, itir.Program):
            raise TypeError(f"Expected a 'Program', got '{type(node).__name__}'.")

        node = itir_type_inference.infer(node, offset_provider_type=offset_provider_type)
        grid_type = ir_utils_misc.grid_type_from_program(node)
        if grid_type == common.GridType.UNSTRUCTURED:
            node = _CannonicalizeUnstructuredDomain.apply(node)
        return cls(
            offset_provider_type=offset_provider_type, column_axis=column_axis, grid_type=grid_type
        ).visit(node)

    def visit_Sym(self, node: itir.Sym, **kwargs: Any) -> Sym:
        return Sym(id=node.id)

    def visit_SymRef(
        self,
        node: itir.SymRef,
        force_function_extraction: bool = False,
        extracted_functions: Optional[list] = None,
        **kwargs: Any,
    ) -> SymRef:
        if force_function_extraction and node.id == "deref":
            assert extracted_functions is not None
            fun_id = next(self.uids["_fun"])
            fun_def = FunctionDefinition(
                id=fun_id,
                params=[Sym(id="x")],
                expr=FunCall(fun=SymRef(id="deref"), args=[SymRef(id="x")]),
            )
            extracted_functions.append(fun_def)
            return SymRef(id=fun_id)
        return SymRef(id=node.id)

    def visit_Lambda(
        self,
        node: itir.Lambda,
        *,
        force_function_extraction: bool = False,
        extracted_functions: Optional[list] = None,
        **kwargs: Any,
    ) -> Union[SymRef, Lambda]:
        if force_function_extraction:
            assert extracted_functions is not None
            fun_id = next(self.uids["_fun"])
            fun_def = FunctionDefinition(
                id=fun_id,
                params=self.visit(node.params, **kwargs),
                expr=self.visit(node.expr, **kwargs),
            )
            extracted_functions.append(fun_def)
            return SymRef(id=fun_id)
        return Lambda(
            params=self.visit(node.params, **kwargs), expr=self.visit(node.expr, **kwargs)
        )

    def visit_Literal(self, node: itir.Literal, **kwargs: Any) -> Literal:
        return Literal(value=node.value, type=node.type.kind.name.lower())

    def visit_OffsetLiteral(self, node: itir.OffsetLiteral, **kwargs: Any) -> OffsetLiteral:
        return OffsetLiteral(value=node.value)

    def visit_AxisLiteral(self, node: itir.AxisLiteral, **kwargs: Any) -> Literal:
        return Literal(value=node.value, type="axis_literal")

    def _make_domain(self, node: itir.FunCall) -> tuple[TaggedValues, TaggedValues]:
        tags = []
        sizes = []
        offsets = []
        for named_range in node.args:
            if not (
                isinstance(named_range, itir.FunCall)
                and named_range.fun == itir.SymRef(id="named_range")
            ):
                raise ValueError("Arguments to 'domain' need to be calls to 'named_range'.")
            tags.append(self.visit(named_range.args[0]))
            sizes.append(
                BinaryExpr(
                    op="-", lhs=self.visit(named_range.args[2]), rhs=self.visit(named_range.args[1])
                )
            )
            offsets.append(self.visit(named_range.args[1]))
        return TaggedValues(tags=tags, values=sizes), TaggedValues(tags=tags, values=offsets)

    @staticmethod
    def _collect_offset_or_axis_node(
        node_type: Type, tree: eve.Node | Iterable[eve.Node]
    ) -> set[str]:
        if not isinstance(tree, Iterable):
            tree = [tree]
        result = set()
        for n in tree:
            result.update(
                n.pre_walk_values()
                .if_isinstance(node_type)
                .getattr("value")
                .if_isinstance(str)
                .to_set()
            )
        return result

    def _visit_if_(self, node: itir.FunCall, **kwargs: Any) -> Node:
        assert len(node.args) == 3
        return TernaryExpr(
            cond=self.visit(node.args[0], **kwargs),
            true_expr=self.visit(node.args[1], **kwargs),
            false_expr=self.visit(node.args[2], **kwargs),
        )

    def _visit_cast_(self, node: itir.FunCall, **kwargs: Any) -> Node:
        assert len(node.args) == 2
        return CastExpr(
            obj_expr=self.visit(node.args[0], **kwargs),
            new_dtype=self.visit(node.args[1], **kwargs),
        )

    def _visit_tuple_get(self, node: itir.FunCall, **kwargs: Any) -> Node:
        assert isinstance(node.args[0], itir.Literal)
        return FunCall(
            fun=SymRef(id="tuple_get"),
            args=[_literal_as_integral_constant(node.args[0]), self.visit(node.args[1])],
        )

    def _visit_list_get(self, node: itir.FunCall, **kwargs: Any) -> Node:
        # should only reach this for the case of an external sparse field
        tuple_idx = (
            _literal_as_integral_constant(node.args[0])
            if isinstance(node.args[0], itir.Literal)
            else self.visit(
                node.args[0]
            )  # from unroll_reduce we get a `SymRef` which is refering to an `OffsetLiteral` which is lowered to integral_constant
        )
        return FunCall(fun=SymRef(id="tuple_get"), args=[tuple_idx, self.visit(node.args[1])])

    def _visit_cartesian_domain(self, node: itir.FunCall, **kwargs: Any) -> Node:
        sizes, domain_offsets = self._make_domain(node)
        return CartesianDomain(tagged_sizes=sizes, tagged_offsets=domain_offsets)

    def _visit_unstructured_domain(self, node: itir.FunCall, **kwargs: Any) -> Node:
        sizes, domain_offsets = self._make_domain(node)
        connectivities = []
        if "stencil" in kwargs:
            shift_offsets = self._collect_offset_or_axis_node(itir.OffsetLiteral, kwargs["stencil"])
            for o in shift_offsets:
                if o in self.offset_provider_type and isinstance(
                    common.get_offset_type(self.offset_provider_type, o),
                    common.NeighborConnectivityType,
                ):
                    connectivities.append(SymRef(id=o))
        return UnstructuredDomain(
            tagged_sizes=sizes, tagged_offsets=domain_offsets, connectivities=connectivities
        )

    def _visit_get_domain_range(self, node: itir.FunCall, **kwargs: Any) -> Node:
        field, dim = node.args

        return FunCall(
            fun=SymRef(id="get_domain_range"),
            args=[self.visit(field, **kwargs), self.visit(dim, **kwargs)],
        )

    def visit_FunCall(self, node: itir.FunCall, **kwargs: Any) -> Node:
        if isinstance(node.fun, itir.SymRef):
            if node.fun.id in self._unary_op_map:
                assert len(node.args) == 1
                return UnaryExpr(
                    op=self._unary_op_map[node.fun.id], expr=self.visit(node.args[0], **kwargs)
                )
            elif node.fun.id in self._binary_op_map:
                assert len(node.args) == 2
                return BinaryExpr(
                    op=self._binary_op_map[node.fun.id],
                    lhs=self.visit(node.args[0], **kwargs),
                    rhs=self.visit(node.args[1], **kwargs),
                )
            elif hasattr(self, visit_method := f"_visit_{node.fun.id}"):
                # special handling of applied builtins is handled in `_visit_<builtin>`
                return getattr(self, visit_method)(node, **kwargs)
            elif node.fun.id == "shift":
                raise ValueError("Unapplied shift call not supported: '{node}'.")
            elif node.fun.id == "scan":
                raise ValueError("Scans are only supported at the top level of a stencil closure.")
        if isinstance(node.fun, itir.FunCall):
            if node.fun.fun == itir.SymRef(id="shift"):
                assert len(node.args) == 1
                return FunCall(
                    fun=self.visit(node.fun.fun, **kwargs),
                    args=self.visit(node.args, **kwargs) + self.visit(node.fun.args, **kwargs),
                )
        return FunCall(fun=self.visit(node.fun, **kwargs), args=self.visit(node.args, **kwargs))

    def visit_FunctionDefinition(
        self, node: itir.FunctionDefinition, **kwargs: Any
    ) -> FunctionDefinition:
        return FunctionDefinition(
            id=node.id,
            params=self.visit(node.params, **kwargs),
            expr=self.visit(node.expr, **kwargs),
        )

    def _visit_output_argument(self, node: itir.Expr) -> SidComposite | SymRef:
        lowered_output = self.visit(node)

        # just a sanity check, identity function otherwise
        def check_el_type(el_expr: Expr, el_type: ts.ScalarType | ts.FieldType) -> Expr:
            assert isinstance(el_type, ts.FieldType)
            return el_expr

        assert isinstance(node.type, ts.TypeSpec)
        lowered_output_as_sid = _process_elements(
            check_el_type,
            lowered_output,
            node.type,
            tuple_constructor=lambda *elements: SidComposite(values=list(elements)),
        )

        assert isinstance(lowered_output_as_sid, (SidComposite, SymRef))
        return lowered_output_as_sid

    @staticmethod
    def _merge_scans(
        executions: list[Union[StencilExecution, ScanExecution]],
        scan_forward: Optional[dict[str, bool]] = None,
    ) -> list[Union[StencilExecution, ScanExecution]]:
        scan_forward = scan_forward or {}

        def _dedup_b_into_a(a: ScanExecution, b: ScanExecution) -> tuple[dict[int, int], list[Expr]]:
            index_map = dict[int, int]()
            compacted_b_args = list[Expr]()
            for b_idx, b_arg in enumerate(b.args):
                try:
                    a_idx = a.args.index(b_arg)
                    index_map[b_idx] = a_idx
                except ValueError:
                    index_map[b_idx] = len(a.args) + len(compacted_b_args)
                    compacted_b_args.append(b_arg)
            return index_map, compacted_b_args

        def merge(a: ScanExecution, b: ScanExecution) -> ScanExecution:
            assert a.backend == b.backend
            assert a.axis == b.axis
            index_map, compacted_b_args = _dedup_b_into_a(a, b)

            def remap_args(s: Scan) -> Scan:
                return Scan(
                    function=s.function,
                    output=index_map[s.output],
                    inputs=[index_map[i] for i in s.inputs],
                    init=s.init,
                )

            return ScanExecution(
                backend=a.backend,
                scans=a.scans + [remap_args(s) for s in b.scans],
                args=a.args + compacted_b_args,
                axis=a.axis,
            )

        def merge_fwd_bwd(a: ScanExecution, b: ScanExecution) -> Optional[ScanExecution]:
            # Fuse a forward sweep (a) and the back-substitution (b) that consumes its output
            # into ONE kernel. They run over different K-extents (forward [1, N) avoids the
            # Koff[-1] read at the top, back-substitution [0, N)); we launch over the union K
            # range and give each scan vertical trims so it covers exactly its original range.
            if os.environ.get("GT4PY_DISABLE_FWDBWD_MERGE"):
                return None
            if not (len(a.scans) == 1 and len(b.scans) == 1):
                return None
            if not (
                scan_forward.get(a.scans[0].function.id) is True
                and scan_forward.get(b.scans[0].function.id) is False
            ):
                return None
            a_range = _static_k_range(a.backend.domain, a.axis)
            b_range = _static_k_range(b.backend.domain, b.axis)
            if a_range is None or b_range is None:
                return None
            # producer -> consumer: b reads a's output field(s)
            a_out_syms = _symref_ids(a.args[a.scans[0].output])
            b_in_syms: set[str] = set()
            for i in b.scans[0].inputs:
                b_in_syms |= _symref_ids(b.args[i])
            if not (a_out_syms & b_in_syms):
                return None
            u_start = min(a_range[0], b_range[0])
            u_stop = max(a_range[1], b_range[1])
            # union must coincide with one of the two domains (one contains the other)
            if a_range == (u_start, u_stop):
                union_domain = a.backend.domain
            elif b_range == (u_start, u_stop):
                union_domain = b.backend.domain
            else:
                return None
            index_map, compacted_b_args = _dedup_b_into_a(a, b)
            a_s, b_s = a.scans[0], b.scans[0]
            a_scan = Scan(
                function=a_s.function,
                output=a_s.output,
                inputs=a_s.inputs,
                init=a_s.init,
                top_trim=a_range[0] - u_start,
                bot_trim=u_stop - a_range[1],
            )
            b_scan = Scan(
                function=b_s.function,
                output=index_map[b_s.output],
                inputs=[index_map[i] for i in b_s.inputs],
                init=b_s.init,
                top_trim=b_range[0] - u_start,
                bot_trim=u_stop - b_range[1],
            )
            return ScanExecution(
                backend=Backend(domain=union_domain),
                scans=[a_scan, b_scan],
                args=a.args + compacted_b_args,
                axis=a.axis,
                merged_kernel=True,
            )

        res = executions[:1]
        for execution in executions[1:]:
            prev = res[-1]
            if (
                isinstance(execution, ScanExecution)
                and isinstance(prev, ScanExecution)
                and execution.axis == prev.axis
            ):
                fused = merge_fwd_bwd(prev, execution)
                if fused is not None:
                    res[-1] = fused
                    continue
                if execution.backend == prev.backend:
                    res[-1] = merge(prev, execution)
                    continue
            res.append(execution)
        return res

    def _expr_offset_ids(self, expr: Expr) -> set[str]:
        return set(
            expr.pre_walk_values().if_isinstance(OffsetLiteral).getattr("value").if_isinstance(str).to_set()
        )

    def _has_connectivity_offset(self, expr: Expr) -> bool:
        for o in self._expr_offset_ids(expr):
            if o in self.offset_provider_type and not isinstance(
                common.get_offset_type(self.offset_provider_type, o), common.Dimension
            ):
                return True
        return False

    def _fuse_map_chains(
        self,
        executions: list,
        extracted_functions: list,
        temp_names: set[str],
    ) -> tuple[list, set[str]]:
        """PROTOTYPE (GT4PY_FN_MAP_COLUMN_FUSION, default OFF): fuse a same-domain cell-local
        producer->consumer map-chain joined by a single Koff[-1] edge into ONE column kernel
        (MapColumnExecution -> map_column_stage). The producer's per-level value stays register-
        resident and the consumer reads its K-1 value from a one-deep history register, so the
        chain edge never round-trips DRAM. The gtfn analog of a dace shared transient, generalizing
        the scan_with_tail fold from scans to plain maps.

        First increment is intentionally narrow: producer is a single-output cell-local field-op,
        consumer is a single-output cell-local field-op reading the producer at center and/or
        Koff[-1] only (downward/history edge), same static domain. Anything else is left unfused."""
        if not os.environ.get("GT4PY_FN_MAP_COLUMN_FUSION"):
            return executions, set()
        # GT4PY_FN_KOFF_WINDOW_ONLY suppresses the producer->consumer map-chain fold so the
        # single-stencil koff-window column-ification (increment-2b) can target the same consumer.
        if os.environ.get("GT4PY_FN_KOFF_WINDOW_ONLY"):
            return executions, set()

        fun_by_id = {f.id: f for f in extracted_functions if isinstance(f, FunctionDefinition)}
        _dbg = os.environ.get("GT4PY_MAP_COLUMN_DEBUG")

        def reject(why: str) -> None:
            if _dbg:
                print(f"[map_column] reject: {why}")

        if os.environ.get("GT4PY_MAP_COLUMN_DUMP"):
            for _i, _e in enumerate(executions):
                if isinstance(_e, StencilExecution):
                    _o = (
                        _e.output.id
                        if isinstance(_e.output, SymRef)
                        else f"Composite[{len(_e.output.values)}]"
                        if isinstance(_e.output, SidComposite)
                        else type(_e.output).__name__
                    )
                    _ins = ",".join(sorted(s for inp in _e.inputs for s in _symref_ids(inp)))
                    print(f"[map_column DUMP] {_i} Stencil {_e.stencil.id} OUT={_o} IN=[{_ins}]")
                else:
                    print(f"[map_column DUMP] {_i} {type(_e).__name__}")

        def out_syms(e: Stmt) -> set[str]:
            if isinstance(e, StencilExecution):
                return _symref_ids(e.output) if isinstance(e.output, (SymRef, SidComposite)) else set()
            if isinstance(e, ScanExecution):
                return {e.args[s.output].id for s in e.scans if isinstance(e.args[s.output], SymRef)}
            return set()

        def in_syms(e: Stmt) -> set[str]:
            if isinstance(e, StencilExecution):
                return {sym for inp in e.inputs for sym in _symref_ids(inp)}
            if isinstance(e, ScanExecution):
                s = set[str]()
                for sc in e.scans:
                    for i in sc.inputs:
                        s |= _symref_ids(e.args[i])
                return s
            return set()

        def is_cell_local_fieldop(sten: StencilExecution) -> Optional[FunctionDefinition]:
            if not isinstance(sten.stencil, SymRef) or sten.stencil.id not in fun_by_id:
                return None
            fun = fun_by_id[sten.stencil.id]
            if self._has_connectivity_offset(fun.expr):
                return None
            if len(fun.params) != len(sten.inputs):
                return None
            return fun

        def try_fuse(
            prod: StencilExecution, cons: StencilExecution
        ) -> Optional[tuple[Stmt, Optional[str]]]:
            # Producer: single SymRef output T, cell-local field-op.
            if not isinstance(prod.output, SymRef):
                return reject("producer is multi-output")
            T = prod.output.id
            prod_fun = is_cell_local_fieldop(prod)
            if prod_fun is None:
                return reject("producer is not a cell-local field-op")
            if any(
                o in self.offset_provider_type
                and isinstance(common.get_offset_type(self.offset_provider_type, o), common.Dimension)
                for o in self._collect_offset_or_axis_node(OffsetLiteral, prod_fun.expr)
            ):
                return reject("producer has a vertical shift")
            if not all(isinstance(i, SymRef) for i in prod.inputs):
                return reject("producer inputs are not plain SymRefs")
            # Consumer: cell-local field-op reading T. Output is either a single SymRef field or a
            # SidComposite (multi-output, increment-2a): the consumer computes several outputs, none
            # of which is T (T is producer-only), and reads T at center+Koff[-1]. The composite is
            # kept as ONE arg and the tuple-valued body is written to it exactly like a plain
            # multi-output stencil; T's reads still resolve to the cur/prev registers.
            cons_fun = is_cell_local_fieldop(cons)
            if cons_fun is None:
                return reject("consumer is not a cell-local field-op")
            if not isinstance(cons.output, (SymRef, SidComposite)):
                return reject("consumer output is neither SymRef nor SidComposite")
            if isinstance(cons.output, SidComposite) and T in _symref_ids(cons.output):
                return reject(f"{T} is among the multi-output consumer's outputs")
            t_positions = [
                j for j, inp in enumerate(cons.inputs) if isinstance(inp, SymRef) and inp.id == T
            ]
            if len(t_positions) != 1:
                return reject(f"{T} appears {len(t_positions)} times among consumer inputs")
            t_pos = t_positions[0]
            t_param = cons_fun.params[t_pos].id
            # Vertical axis name: the (unique) Koff offset in the consumer whose offset-type is a
            # Dimension (the vertical/K dim). column_axis may be None for scan-free programs.
            vaxis = None
            for o in self._collect_offset_or_axis_node(OffsetLiteral, cons_fun.expr):
                if o in self.offset_provider_type and isinstance(
                    common.get_offset_type(self.offset_provider_type, o), common.Dimension
                ):
                    vaxis = common.get_offset_type(self.offset_provider_type, o).value
                    break
            if vaxis is None and self.column_axis is not None:
                vaxis = self.column_axis.value
            if vaxis is None:
                return reject("could not determine vertical axis")
            axisref = SymRef(id=vaxis)
            # Domain match. The producer/consumer domains query bounds via their OWN output SID
            # (tmp vs out), so after renaming the producer's output SID to the consumer's they are
            # structurally comparable. Two cases:
            #  (a) producer is a program OUTPUT field (not a temp): its domain equals the consumer's
            #      exactly -> require structural equality (increment-0).
            #  (b) producer writes a real TEMPORARY fed by a Koff[-1] consumer read: the temporaries
            #      pass extends the producer's K-range by 1 level at the TOP (a halo level for the
            #      K-1 read), so producer-K-start == consumer-K-start - 1, producer-K-size ==
            #      consumer-K-size + 1, horizontal extent identical. In the COLUMN-kernel form that
            #      halo level is subsumed by the one-deep history register, so we launch over the
            #      CONSUMER's K-range and may drop the producer write entirely (increment-1).
            # Normalize the producer's output-SID name out of its domain so it compares against the
            # consumer's domain. For a single-SymRef consumer output rename T -> that name (the only
            # textual difference). For a multi-output (SidComposite) consumer there is no single name
            # to rename to: in compile_time_domain mode the domains are literal tuples and already
            # equal, so renaming T to a neutral sentinel suffices (it just removes the producer-only
            # symbol, which cannot appear in the consumer domain anyway).
            rename_to = cons.output.id if isinstance(cons.output, SymRef) else f"__mc_dom_{T}"
            prod_dom_norm = _RenameSymRef(mapping={T: rename_to}).visit(prod.backend.domain)
            is_temp = T in temp_names
            drop_producer_output = False
            if prod_dom_norm == cons.backend.domain:
                # case (a): exact same domain. A producer-only temp can still drop its SID write —
                # its values flow solely through the cur/prev registers (the K=0 Koff[-1] read is
                # guard-protected by the consumer and `prev` is seeded with `cur`).
                drop_producer_output = is_temp
            elif is_temp:
                # case (b): producer extended by exactly 1 level at the top (Koff[-1] halo).
                pk = _domain_k_bounds(prod_dom_norm, axisref)
                ck = _domain_k_bounds(cons.backend.domain, axisref)
                if pk is None or ck is None:
                    return reject("could not read K-bounds for temp-producer domain check")
                top_ext = _static_diff(ck[0], pk[0])  # cons_start - prod_start
                bot_ext = _static_diff(pk[1], ck[1])  # prod_stop  - cons_stop
                if top_ext != 1 or bot_ext != 0:
                    return reject(
                        f"temp producer not extended by exactly 1 top level "
                        f"(top_ext={top_ext}, bot_ext={bot_ext})"
                    )
                # also require horizontal extents identical (everything but K matches after rename)
                if _strip_k_dim(prod_dom_norm, axisref) != _strip_k_dim(
                    cons.backend.domain, axisref
                ):
                    return reject("producer/consumer non-K domain extents differ")
                drop_producer_output = True
            else:
                return reject("producer/consumer domains differ after output-SID normalization")
            # T must be producer-only: written by no other execution, read by no execution but cons.
            if any(T in out_syms(e) for e in executions if e is not prod):
                return reject(f"{T} written elsewhere")
            if any(T in in_syms(e) for e in executions if e is not cons):
                return reject(f"{T} read by an execution other than the consumer")
            # Dropping the producer SID write is only sound for a true producer-only temporary.
            if drop_producer_output and not is_temp:
                drop_producer_output = False

            cur_name, prev_name = f"{t_param}__cur", f"{t_param}__prev"
            rewriter = _MapChainRewriter(sym=t_param, cur=cur_name, prev=prev_name)
            new_expr = rewriter.visit(cons_fun.expr)
            if not rewriter.ok:
                return reject(f"{t_param} accessed in an unfoldable way (not center/Koff[-1])")

            # Increment-2b "extra windows": besides the fused producer temp T, the consumer may read
            # OTHER already-materialized inputs at center + Koff[-1] (e.g. pert's _fun_3 reading
            # perturbed_exner_mc + temporal_extrapolation, which _fun_0 materializes for OTHER
            # consumers so they cannot be fused/dropped). Window each such input: load it once per
            # level and serve its K-1 value from a register, instead of a second DRAM load. Only when
            # the launch K-range starts at the field origin is the seeded K==0 `prev` safe (the
            # consumer's where(k>0,...) guards discard it) — same gate as _koffwindow_columnify.
            koff = None
            for o in self.offset_provider_type:
                ot = common.get_offset_type(self.offset_provider_type, o)
                if isinstance(ot, common.Dimension) and ot.value == vaxis:
                    koff = o
                    break
            win_positions: list[int] = []
            if (
                koff is not None
                and os.environ.get("GT4PY_FN_MAP_COLUMN_WINDOWS") is not None
                and not os.environ.get("GT4PY_FN_MAP_COLUMN_WINDOWS_OFF")
            ):
                launch_dom = cons.backend.domain if drop_producer_output else prod.backend.domain
                kb = _domain_k_bounds(launch_dom, axisref)
                k_start = _static_int(kb[0]) if kb is not None else None
                if k_start == 0:
                    for j, (param, inp) in enumerate(zip(cons_fun.params, cons.inputs)):
                        if j == t_pos or not isinstance(inp, SymRef):
                            continue
                        offs = _koff_window_offsets(cons_fun.expr, param.id, koff)
                        if offs is None:
                            continue
                        if -1 in offs and offs <= {0, -1}:
                            win_positions.append(j)

            window_mapping: dict[str, tuple[str, str]] = {}
            windows: list[tuple[Sym, Sym]] = []
            if win_positions:
                for j in win_positions:
                    pid = cons_fun.params[j].id
                    wc, wp = f"{pid}__wcur", f"{pid}__wprev"
                    window_mapping[pid] = (wc, wp)
                    windows.append((Sym(id=wc), Sym(id=wp)))
                wrew = _KoffWindowRewriter(mapping=window_mapping)
                new_expr = wrew.visit(new_expr)
                if not wrew.ok:
                    return reject("a windowed consumer input escaped the cur/prev rewrite")

            # Build the new combined arg list. When the producer writes a producer-only temp we
            # DROP its SID write (increment-1): the producer output is NOT added as an arg and the
            # temp allocation is removed; the chain value lives only in the `cur`/`prev` registers.
            # Otherwise (producer is a real output field) keep its output arg and still materialize
            # it (increment-0). Followed by deduped producer inputs, the consumer's output, and the
            # consumer's non-T inputs.
            new_args: list[Expr] = []

            def arg_index(a: Expr) -> int:
                for k, existing in enumerate(new_args):
                    if existing == a:
                        return k
                new_args.append(a)
                return len(new_args) - 1

            if drop_producer_output:
                producer_output = -1  # sentinel: map_column_stage skips the producer SID write
            else:
                producer_output = arg_index(prod.output)
            producer_inputs = [arg_index(i) for i in prod.inputs]
            out_idx = arg_index(cons.output)
            window_inputs = [arg_index(cons.inputs[j]) for j in win_positions]
            lifted: list[tuple[Sym, int]] = []
            for j, inp in enumerate(cons.inputs):
                if j == t_pos or j in win_positions:
                    continue
                lifted.append((Sym(id=cons_fun.params[j].id), arg_index(inp)))

            # Launch over the consumer's domain: in the column form the producer's extra top halo
            # level (when the producer is a temp extended for the Koff[-1] read) is subsumed by the
            # one-deep history register, so the producer never needs that level.
            launch_domain = cons.backend.domain if drop_producer_output else prod.backend.domain
            if windows:
                tail_id = next(self.uids["_mwtail"])
                extracted_functions.append(
                    MapWindowTailDefinition(
                        id=tail_id,
                        cur=Sym(id=cur_name),
                        prev=Sym(id=prev_name),
                        windows=windows,
                        input_params=lifted,
                        outputs=[(out_idx, new_expr)],
                    )
                )
                exe: Stmt = MapWindowExecution(
                    backend=Backend(domain=launch_domain),
                    producer=prod.stencil,
                    producer_output=producer_output,
                    producer_inputs=producer_inputs,
                    consumer=SymRef(id=tail_id),
                    window_inputs=window_inputs,
                    args=new_args,
                    axis=SymRef(id=vaxis),
                )
            else:
                tail_id = next(self.uids["_mctail"])
                extracted_functions.append(
                    MapColumnTailDefinition(
                        id=tail_id,
                        cur=Sym(id=cur_name),
                        prev=Sym(id=prev_name),
                        input_params=lifted,
                        outputs=[(out_idx, new_expr)],
                    )
                )
                exe = MapColumnExecution(
                    backend=Backend(domain=launch_domain),
                    producer=prod.stencil,
                    producer_output=producer_output,
                    producer_inputs=producer_inputs,
                    consumer=SymRef(id=tail_id),
                    args=new_args,
                    axis=SymRef(id=vaxis),
                )
            return exe, (T if drop_producer_output else None)

        res: list = []
        dropped: set[str] = set()
        i = 0
        while i < len(executions):
            cur = executions[i]
            nxt = executions[i + 1] if i + 1 < len(executions) else None
            if isinstance(cur, StencilExecution) and isinstance(nxt, StencilExecution):
                fused = try_fuse(cur, nxt)
                if fused is not None:
                    exe, dropped_temp = fused
                    res.append(exe)
                    if dropped_temp is not None:
                        dropped.add(dropped_temp)
                    i += 2
                    continue
            res.append(cur)
            i += 1
        return res, dropped

    def _koffwindow_columnify(
        self,
        executions: list,
        extracted_functions: list,
    ) -> list:
        """PROTOTYPE (GT4PY_FN_MAP_COLUMN_FUSION + GT4PY_FN_KOFF_WINDOW, increment-2b): column-ify a
        SINGLE multi-output cell-local stencil whose ALREADY-materialized inputs are read at center
        + Koff[-1] (history) only. No producer is fused in — the windowed fields stay in DRAM (their
        producer is untouched); the win is that the consumer loads each windowed field ONCE per level
        and reads its K-1 value from a one-deep history register instead of a second DRAM load. This
        is the gtfn way to get dace's materialize-once-read-at-Koff WITHOUT recompute (vsf) and
        WITHOUT a global temp, for the keep-and-write case where the producer feeds other consumers
        too (e.g. pert's _fun_3 reading temporal_extrapolation + perturbed_exner_mc at center+Koff[-1]).

        The transform is a per-execution rewrite (not a pair fold): it never reorders, so the
        non-adjacency / data-dependency of the producer is irrelevant — the producer kernel stays
        exactly where it is and still materializes the fields for the other consumers."""
        if not (
            os.environ.get("GT4PY_FN_MAP_COLUMN_FUSION") and os.environ.get("GT4PY_FN_KOFF_WINDOW")
        ):
            return executions

        fun_by_id = {f.id: f for f in extracted_functions if isinstance(f, FunctionDefinition)}
        _dbg = os.environ.get("GT4PY_MAP_COLUMN_DEBUG")

        def reject(why: str) -> None:
            if _dbg:
                print(f"[koff_window] reject: {why}")

        # vertical (Koff) offset tag.
        koff = None
        for o in self.offset_provider_type:
            ot = common.get_offset_type(self.offset_provider_type, o)
            if isinstance(ot, common.Dimension):
                koff = o
                break

        min_windows = int(os.environ.get("GT4PY_FN_KOFF_WINDOW_MIN", "1"))

        def columnify(sten: StencilExecution) -> Optional[KoffWindowExecution]:
            if not isinstance(sten.stencil, SymRef) or sten.stencil.id not in fun_by_id:
                return reject("stencil is not an extracted field-op")
            fun = fun_by_id[sten.stencil.id]
            if len(fun.params) != len(sten.inputs):
                return reject("param/input arity mismatch")
            if koff is None:
                return reject("no vertical offset in the program")
            vaxis = common.get_offset_type(self.offset_provider_type, koff).value
            axisref = SymRef(id=vaxis)
            # Soundness gate: the column kernel strips K from the launch grid and walks K in-thread
            # from the launch-K-start. The K==0 `prev` is seeded with `cur` (the launch-top value),
            # which is only correct when the launch K-range starts at the field origin (offset 0) so
            # the column top IS the field top (the consumer's where(k>0,...) guard then discards the
            # seed). If the K-range starts mid-column an unguarded Koff[-1] read at the top would get
            # the wrong (seeded) value, so refuse to columnify.
            kb = _domain_k_bounds(sten.backend.domain, axisref)
            if kb is None:
                return reject(f"{sten.stencil.id}: no K-range in domain")
            k_start = _static_int(kb[0])
            if k_start is None or k_start != 0:
                return reject(f"{sten.stencil.id}: K-range does not start at the field origin")
            # Find inputs read at center + Koff[-1] only (a 1-deep history window). Each must be a
            # plain SID arg (SymRef input). A windowed input may also be read at center only; we
            # window any input that has at least one Koff[-1] read and NO other access shape.
            win_positions: list[int] = []
            for j, (param, inp) in enumerate(zip(fun.params, sten.inputs)):
                if not isinstance(inp, SymRef):
                    continue
                offs = _koff_window_offsets(fun.expr, param.id, koff)
                if offs is None:
                    continue  # accessed in a non-windowable way -> leave as a normal SID read
                if -1 in offs and offs <= {0, -1}:
                    win_positions.append(j)
            if len(win_positions) < min_windows:
                return reject(
                    f"{sten.stencil.id}: {len(win_positions)} center+Koff[-1] window input(s) "
                    f"(min {min_windows})"
                )

            # Build the combined arg list: outputs (composite or single) first, then deduped inputs.
            new_args: list[Expr] = []

            def arg_index(a: Expr) -> int:
                for k, existing in enumerate(new_args):
                    if existing == a:
                        return k
                new_args.append(a)
                return len(new_args) - 1

            out_idx = arg_index(sten.output)
            # rewrite the windowed params to cur/prev registers.
            mapping: dict[str, tuple[str, str]] = {}
            windows: list[tuple[Sym, Sym]] = []
            window_inputs: list[int] = []
            for j in win_positions:
                pid = fun.params[j].id
                cur_name, prev_name = f"{pid}__cur", f"{pid}__prev"
                mapping[pid] = (cur_name, prev_name)
                windows.append((Sym(id=cur_name), Sym(id=prev_name)))
                window_inputs.append(arg_index(sten.inputs[j]))
            rewriter = _KoffWindowRewriter(mapping=mapping)
            new_expr = rewriter.visit(fun.expr)
            if not rewriter.ok:
                return reject(f"{sten.stencil.id}: a windowed param escaped the cur/prev rewrite")
            # remaining (non-windowed) inputs become make_iterator-bound params at the current level.
            lifted: list[tuple[Sym, int]] = []
            for j, inp in enumerate(sten.inputs):
                if j in win_positions:
                    continue
                lifted.append((Sym(id=fun.params[j].id), arg_index(inp)))

            tail_id = next(self.uids["_kwtail"])
            extracted_functions.append(
                KoffWindowTailDefinition(
                    id=tail_id,
                    windows=windows,
                    input_params=lifted,
                    outputs=[(out_idx, new_expr)],
                )
            )
            if _dbg:
                print(
                    f"[koff_window] columnify {sten.stencil.id}: "
                    f"{len(windows)} window(s) {[w[0].id for w in windows]}"
                )
            return KoffWindowExecution(
                backend=sten.backend,
                consumer=SymRef(id=tail_id),
                window_inputs=window_inputs,
                args=new_args,
                axis=SymRef(id=vaxis),
            )

        res: list = []
        for e in executions:
            if isinstance(e, StencilExecution) and isinstance(e.output, (SymRef, SidComposite)):
                col = columnify(e)
                if col is not None:
                    res.append(col)
                    continue
            res.append(e)
        return res

    def _fuse_scan_tails(
        self,
        executions: list[Union[StencilExecution, ScanExecution]],
        scan_forward: dict[str, bool],
        extracted_functions: list,
    ) -> tuple[list[Union[StencilExecution, ScanExecution]], set[str]]:
        if os.environ.get("GT4PY_DISABLE_POSTSCAN_FOLD"):
            return executions, set()

        fun_by_id = {f.id: f for f in extracted_functions if isinstance(f, FunctionDefinition)}
        # A temp may be dropped only if the scan that writes it is the sole writer and the folded
        # consumer is its sole reader. Count reads/writes across all executions.
        dropped: set[str] = set()

        def out_syms(e: Stmt) -> set[str]:
            if isinstance(e, ScanExecution):
                return {e.args[s.output].id for s in e.scans if isinstance(e.args[s.output], SymRef)}
            if isinstance(e, StencilExecution):
                return _symref_ids(e.output) if isinstance(e.output, (SymRef, SidComposite)) else set()
            if isinstance(e, IfStmt):
                s: set[str] = set()
                for b in (e.true_branch, e.false_branch):
                    for inner in b:
                        s |= out_syms(inner)
                return s
            return set()

        def in_syms(e: Stmt) -> set[str]:
            if isinstance(e, ScanExecution):
                s = set[str]()
                for sc in e.scans:
                    for i in sc.inputs:
                        s |= _symref_ids(e.args[i])
                return s
            if isinstance(e, StencilExecution):
                return {sym for inp in e.inputs for sym in _symref_ids(inp)}
            if isinstance(e, IfStmt):
                s = set[str]()
                for b in (e.true_branch, e.false_branch):
                    for inner in b:
                        s |= in_syms(inner)
                return s
            return set()

        _dbg = os.environ.get("GT4PY_POSTSCAN_FOLD_DEBUG")

        def reject(why: str) -> None:
            if _dbg:
                print(f"[postscan_fold] reject: {why}")

        def try_fold(scan_exec: ScanExecution, sten: StencilExecution, idx: int) -> Optional[ScanExecution]:
            # Find the single backward scan in this (possibly fwd+bwd merged) kernel whose output
            # temp is consumed only by `sten`. Other substages (e.g. the forward sweep) are kept.
            bwd_candidates = [
                k
                for k, sc in enumerate(scan_exec.scans)
                if sc.tail is None and scan_forward.get(sc.function.id) is False
            ]
            target = None
            for k in bwd_candidates:
                out_arg = scan_exec.args[scan_exec.scans[k].output]
                if isinstance(out_arg, SymRef):
                    target = k
                    break
            if target is None:
                return reject("no backward substage with a SymRef output temp")
            scan = scan_exec.scans[target]
            T = scan_exec.args[scan.output].id
            # Temp must be a producer-only temp: written by no other execution, read by no execution
            # other than this consumer (and not read by another substage of this kernel).
            if any(T in out_syms(e) for e in executions if e is not scan_exec):
                return reject(f"{T} written elsewhere")
            if any(T in in_syms(e) for e in executions if e is not sten):
                return reject(f"{T} read by an execution other than the consumer")
            for k, sc in enumerate(scan_exec.scans):
                if k == target:
                    continue
                if any(T in _symref_ids(scan_exec.args[i]) for i in sc.inputs):
                    return reject(f"{T} read by sibling substage {sc.function.id}")
            # Consumer must be a plain field-op referencing an extracted function, cell-local.
            if not isinstance(sten.stencil, SymRef) or sten.stencil.id not in fun_by_id:
                return reject("consumer is not an extracted field-op")
            fun = fun_by_id[sten.stencil.id]
            if self._has_connectivity_offset(fun.expr):
                return reject("consumer has a connectivity (horizontal) shift")
            if len(fun.params) != len(sten.inputs):
                return reject("consumer param/input count mismatch")
            t_positions = [
                j for j, inp in enumerate(sten.inputs) if isinstance(inp, SymRef) and inp.id == T
            ]
            if len(t_positions) != 1:
                return reject(f"{T} appears {len(t_positions)} times among consumer inputs")
            t_pos = t_positions[0]
            t_param = fun.params[t_pos].id
            # Single-output consumer only (milestone 1).
            if not isinstance(sten.output, SymRef):
                return reject("consumer is multi-output")

            res_name, acc_name, surf_name = (
                f"{t_param}__res",
                f"{t_param}__acc",
                f"{t_param}__surface",
            )
            rewriter = _ScanOutputRewriter(sym=t_param, res=res_name, acc=acc_name)
            new_expr = rewriter.visit(fun.expr)
            if not rewriter.ok:
                return reject(f"{t_param} accessed in an unfoldable way (not deref/Koff[1])")

            kern_b = _domain_k_bounds(scan_exec.backend.domain, scan_exec.axis)
            cons_b = _domain_k_bounds(sten.backend.domain, scan_exec.axis)
            if kern_b is None or cons_b is None:
                return reject("non-static-difference K domains")
            # The launch column must cover both the existing kernel and the consumer; require one to
            # contain the other (their start/stop differ by static amounts).
            d_top = _static_diff(cons_b[1], kern_b[1])  # cons_stop - kern_stop
            d_bot = _static_diff(kern_b[0], cons_b[0])  # kern_start - cons_start
            if d_top is None or d_bot is None:
                return reject(f"K-trim not a static int (d_top={d_top}, d_bot={d_bot})")
            # New kernel domain = the wider of the two on each end. Each existing substage's trims
            # are rebased onto the new column so its absolute K-range is preserved; the consumer's
            # tail spans the new column where the consumer is defined.
            grow_top = max(d_top, 0)  # consumer extends above the kernel
            grow_bot = max(d_bot, 0)
            # The new launch column is the wider of the two; require nesting (the narrower is fully
            # inside the wider on both ends).
            cons_wider = d_top >= 0 and d_bot >= 0
            kern_wider = d_top <= 0 and d_bot <= 0
            if not (cons_wider or kern_wider):
                return reject("consumer K-range not nested with kernel K-range")
            new_domain = sten.backend.domain if cons_wider else scan_exec.backend.domain

            def rebase(tt: int, bt: int) -> tuple[int, int]:
                # keep the same absolute range after growing the column by grow_top/grow_bot
                return tt + grow_bot, bt + grow_top

            new_args: list[Expr] = []

            def arg_index(a: Expr) -> int:
                for k, existing in enumerate(new_args):
                    if existing == a:
                        return k
                new_args.append(a)
                return len(new_args) - 1

            new_scans: list[Scan] = []
            for k, sc in enumerate(scan_exec.scans):
                if k == target:
                    new_scans.append(sc)  # placeholder, replaced below
                    continue
                tt, bt = rebase(sc.top_trim, sc.bot_trim)
                new_scans.append(
                    Scan(
                        function=sc.function,
                        output=arg_index(scan_exec.args[sc.output]),
                        inputs=[arg_index(scan_exec.args[i]) for i in sc.inputs],
                        init=sc.init,
                        top_trim=tt,
                        bot_trim=bt,
                    )
                )

            scan_in_idx = [arg_index(scan_exec.args[i]) for i in scan.inputs]
            lifted: list[tuple[Sym, int]] = []
            for j, inp in enumerate(sten.inputs):
                if j == t_pos:
                    continue
                lifted.append((Sym(id=fun.params[j].id), arg_index(inp)))
            out_idx = arg_index(sten.output)

            tail_id = next(self.uids["_tail"])
            extracted_functions.append(
                ScanTailDefinition(
                    id=tail_id,
                    res=Sym(id=res_name),
                    acc=Sym(id=acc_name),
                    surface=Sym(id=surf_name),
                    input_params=lifted,
                    outputs=[(out_idx, new_expr)],
                )
            )

            # body window = the bwd scan's original absolute K-range (rebased onto the new column);
            # tail window = the consumer's K-range (its domain == new column when consumer is wider).
            body_tt, body_bt = rebase(scan.top_trim, scan.bot_trim)
            new_b = _domain_k_bounds(new_domain, scan_exec.axis)
            tail_tt = _static_diff(cons_b[0], new_b[0])
            tail_bt = _static_diff(new_b[1], cons_b[1])
            if tail_tt is None or tail_bt is None or tail_tt < 0 or tail_bt < 0:
                return reject(f"tail trim not static non-negative (tt={tail_tt}, bt={tail_bt})")

            new_scans[target] = Scan(
                function=scan.function,
                output=scan.output,  # unused by scan_with_tail
                inputs=scan_in_idx,
                init=scan.init,
                top_trim=body_tt,
                bot_trim=body_bt,
                tail=ScanTail(
                    definition=SymRef(id=tail_id),
                    inputs=[i for _, i in lifted],
                    body_top_trim=body_tt,
                    body_bot_trim=body_bt,
                    tail_top_trim=tail_tt,
                    tail_bot_trim=tail_bt,
                ),
            )
            dropped.add(T)
            return ScanExecution(
                backend=Backend(domain=new_domain),
                scans=new_scans,
                args=new_args,
                axis=scan_exec.axis,
                merged_kernel=scan_exec.merged_kernel,
            )

        res: list[Union[StencilExecution, ScanExecution]] = []
        i = 0
        while i < len(executions):
            cur = executions[i]
            nxt = executions[i + 1] if i + 1 < len(executions) else None
            if isinstance(cur, ScanExecution) and isinstance(nxt, StencilExecution):
                folded = try_fold(cur, nxt, i)
                if folded is not None:
                    res.append(folded)
                    i += 2
                    continue
            res.append(cur)
            i += 1
        return res, dropped

    def visit_Stmt(self, node: itir.Stmt, **kwargs: Any) -> None:
        raise AssertionError("All Stmts need to be handled explicitly.")

    def visit_IfStmt(self, node: itir.IfStmt, **kwargs: Any) -> IfStmt:
        return IfStmt(
            cond=self.visit(node.cond, **kwargs),
            true_branch=self.visit(node.true_branch, **kwargs),
            false_branch=self.visit(node.false_branch, **kwargs),
        )

    def _stencil_has_vertical_shift(self, stencil: itir.Node) -> bool:
        for o in self._collect_offset_or_axis_node(itir.OffsetLiteral, stencil):
            if o in self.offset_provider_type and isinstance(
                common.get_offset_type(self.offset_provider_type, o), common.Dimension
            ):
                # in unstructured a Dimension-typed offset is the vertical (Koff) dimension
                return True
        return False

    def visit_SetAt(
        self, node: itir.SetAt, *, extracted_functions: list, **kwargs: Any
    ) -> Union[StencilExecution, ScanExecution]:
        if _is_tuple_of_ref_or_literal(node.expr):
            node.expr = im.as_fieldop("deref", node.domain)(node.expr)

        itir_projector, extracted_expr = ir_utils_misc.extract_projector(node.expr)
        projector = self.visit(itir_projector, **kwargs) if itir_projector is not None else None
        node.expr = extracted_expr

        assert cpm.is_applied_as_fieldop(node.expr), node.expr
        stencil = node.expr.fun.args[0]
        domain = node.domain
        inputs = node.expr.args
        lowered_inputs = []
        for input_ in inputs:
            lowered_input = self.visit(input_, **kwargs)

            # convert scalar elements into SIDs, leave rest as is
            def convert_el_to_sid(el_expr: Expr, el_type: ts.ScalarType | ts.FieldType) -> Expr:
                if isinstance(el_type, ts.ScalarType):
                    return SidFromScalar(arg=el_expr)
                else:
                    assert isinstance(el_type, ts.FieldType)
                    return el_expr

            assert isinstance(input_.type, ts.TypeSpec)
            lowered_input_as_sid = _process_elements(
                convert_el_to_sid,
                lowered_input,
                input_.type,
                tuple_constructor=lambda *elements: SidComposite(values=list(elements)),
            )

            lowered_inputs.append(lowered_input_as_sid)

        backend_domain = self.visit(domain, stencil=stencil, **kwargs)
        # K-coarsening (loop-block) only pays off for kernels with a vertical (Koff) shift —
        # cell-local kernels gain nothing and only lose occupancy to the extra registers, and
        # scans are themselves the K-loop. So mark only vertical-shift stencils as loop-blocked.
        loop_blocked = (not _is_scan(stencil)) and self._stencil_has_vertical_shift(stencil)
        backend = Backend(domain=backend_domain, loop_blocked=loop_blocked)
        if _is_scan(stencil):
            scan_id = next(self.uids["_scan"])
            scan_lambda = self.visit(stencil.args[0], **kwargs)
            forward = _bool_from_literal(stencil.args[1])
            scan_def = ScanPassDefinition(
                id=scan_id,
                params=scan_lambda.params,
                expr=scan_lambda.expr,
                forward=forward,
                projector=projector,
            )
            extracted_functions.append(scan_def)
            scan = Scan(
                function=SymRef(id=scan_id),
                output=0,
                inputs=[i + 1 for i, _ in enumerate(inputs)],
                init=self.visit(stencil.args[2], **kwargs),
            )
            column_axis = self.column_axis
            assert isinstance(column_axis, common.Dimension)
            return ScanExecution(
                backend=backend,
                scans=[scan],
                args=[self._visit_output_argument(node.target), *lowered_inputs],
                axis=SymRef(id=column_axis.value),
            )
        assert projector is None  # only scans have projectors
        return StencilExecution(
            stencil=self.visit(
                stencil,
                force_function_extraction=True,
                extracted_functions=extracted_functions,
                **kwargs,
            ),
            output=self._visit_output_argument(node.target),
            inputs=lowered_inputs,
            backend=backend,
        )

    def visit_Program(self, node: itir.Program, **kwargs: Any) -> Program:
        extracted_functions: list[Union[FunctionDefinition, ScanPassDefinition]] = []
        executions = self.visit(node.body, extracted_functions=extracted_functions)
        scan_forward = {
            str(f.id): f.forward
            for f in extracted_functions
            if isinstance(f, ScanPassDefinition)
        }
        executions = self._merge_scans(executions, scan_forward)
        executions, dropped_temps = self._fuse_scan_tails(
            executions, scan_forward, extracted_functions
        )
        temp_names = {str(t.id) for t in node.declarations}
        executions, dropped_map_temps = self._fuse_map_chains(
            executions, extracted_functions, temp_names
        )
        dropped_temps |= dropped_map_temps
        executions = self._koffwindow_columnify(executions, extracted_functions)
        function_definitions = self.visit(node.function_definitions) + extracted_functions
        offset_definitions = {
            **_collect_dimensions_from_domain(node.body),
            **_collect_offset_definitions(node, self.grid_type, self.offset_provider_type),
        }
        temporaries = self.visit(node.declarations, params=[p.id for p in node.params])
        # A folded scan no longer materializes its output temp; drop its allocation.
        temporaries = [t for t in temporaries if t.id not in dropped_temps]
        return Program(
            id=SymbolName(node.id),
            params=self.visit(node.params),
            executions=executions,
            grid_type=self.grid_type,
            offset_definitions=list(offset_definitions.values()),
            function_definitions=function_definitions,
            temporaries=temporaries,
        )

    def visit_Temporary(
        self, node: itir.Temporary, *, params: list, **kwargs: Any
    ) -> TemporaryAllocation:
        def dtype_to_cpp(x: ts.DataType) -> str:
            if isinstance(x, ts.TupleType):
                assert all(isinstance(i, ts.ScalarType) for i in x.types)
                return "::gridtools::tuple<" + ", ".join(dtype_to_cpp(i) for i in x.types) + ">"  # type: ignore[arg-type] # ensured by assert
            assert isinstance(x, ts.ScalarType)
            res = cpp_utils.pytype_to_cpptype(x)
            assert isinstance(res, str)
            return res

        assert node.dtype
        return TemporaryAllocation(
            id=node.id, dtype=dtype_to_cpp(node.dtype), domain=self.visit(node.domain, **kwargs)
        )
