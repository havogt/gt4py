# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses

from gt4py import eve
from gt4py.next import common, utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import (
    common_pattern_matcher as cpm,
    ir_makers as im,
    misc as ir_misc,
)
from gt4py.next.iterator.transforms import fuse_as_fieldop, inline_lambdas, inline_lifts
from gt4py.next.iterator.transforms.collapse_list_get import CollapseListGet
from gt4py.next.iterator.transforms.fuse_maps import FuseMaps
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas
from gt4py.next.iterator.transforms.normalize_shifts import NormalizeShifts
from gt4py.next.iterator.transforms.unroll_reduce import UnrollReduce
from gt4py.next.iterator.type_system import inference as type_inference
from gt4py.next.type_system import type_specifications as ts


#: Bound on the fixpoint loops below; each is expected to converge in at most a few iterations.
_MAX_ITERATIONS = 10


def _structured_connectivities(
    offset_provider_type: common.OffsetProviderType,
) -> dict[str, common.StructuredConnectivityType]:
    return {
        tag: conn
        for tag, conn in offset_provider_type.items()
        if isinstance(conn, common.StructuredConnectivityType)
    }


def _is_list_field(expr: itir.Expr) -> bool:
    return isinstance(expr.type, ts.FieldType) and isinstance(expr.type.dtype, ts.ListType)


def _structured_tags(node: itir.Node, structured: dict[str, common.StructuredConnectivityType]):
    return (
        node.pre_walk_values()
        .if_isinstance(itir.OffsetLiteral)
        .filter(lambda lit: isinstance(lit.value, str) and lit.value in structured)
    )


@dataclasses.dataclass(frozen=True)
class _RewriteDomains(eve.PreserveLocationVisitor, eve.NodeTranslator):
    """`unstructured_domain(...)` -> `cartesian_domain(...)`, entity dims of `broadcast` -> lattice."""

    entity_dims: frozenset[common.Dimension]
    lattice_dims: tuple[common.Dimension, ...]

    def visit_FunCall(self, node: itir.FunCall, **kwargs) -> itir.FunCall:
        node = self.generic_visit(node, **kwargs)
        if cpm.is_call_to(node, "unstructured_domain"):
            return im.call("cartesian_domain")(*node.args)
        if (
            cpm.is_call_to(node, "broadcast")
            and cpm.is_call_to(node.args[1], "make_tuple")
            and all(isinstance(axis, itir.AxisLiteral) for axis in node.args[1].args)
        ):
            dims = [ir_misc.dim_from_axis_literal(axis) for axis in node.args[1].args]  # type: ignore[arg-type]  # checked above
            if any(dim in self.entity_dims for dim in dims):
                new_dims = [dim for dim in dims if dim not in self.entity_dims]
                new_dims = common.order_dimensions({*new_dims, *self.lattice_dims})
                return im.call("broadcast")(
                    node.args[0], im.make_tuple(*(im.axis_literal(dim) for dim in new_dims))
                )
        return node


@dataclasses.dataclass(frozen=True)
class _FuseListFields(eve.PreserveLocationVisitor, eve.NodeTranslator):
    """
    Inline list-valued lets and fuse list-valued `as_fieldop` arguments into their consumers.

    `let(v, as_fieldop(λ(a, b) → map_list(f)(·a, ·b))(as_fieldop(λ(e) → neighbors(C2E, e))(e), w))`
    `(as_fieldop(λ(it) → list_get(1, ·it))(v))` becomes one `as_fieldop` whose stencil holds
    `list_get(1, map_list(f)(neighbors(C2E, e), ·w))` (modulo the lets `fuse_as_fieldop` leaves).
    A let used `r` times is duplicated `r` times; fusion is repeated per consumer until none of its
    arguments is a list-valued `as_fieldop`, which terminates as every step removes one
    `as_fieldop` from the finite argument tree.
    """

    offset_provider_type: common.OffsetProviderType
    uids: utils.IDGeneratorPool

    def visit_FunCall(self, node: itir.FunCall, **kwargs) -> itir.Expr:
        # pre-order: the let arguments still carry the types of the whole-program inference
        if cpm.is_let(node):
            eligible = [_is_list_field(arg) for arg in node.args]
            if any(eligible):
                return self.visit(
                    inline_lambdas.inline_lambda(node, eligible_params=eligible), **kwargs
                )

        node = self.generic_visit(node, **kwargs)
        if cpm.is_applied_as_fieldop(node):
            return self._fuse_list_args(ir_misc.canonicalize_as_fieldop(node))
        return node

    def _fuse_list_args(self, node: itir.FunCall) -> itir.FunCall:
        for _ in range(_MAX_ITERATIONS):
            eligible = [cpm.is_applied_as_fieldop(arg) and _is_list_field(arg) for arg in node.args]
            if not any(eligible):
                return node
            fused = fuse_as_fieldop.fuse_as_fieldop(
                node,
                eligible,
                offset_provider_type=self.offset_provider_type,
                enable_cse=False,
                uids=self.uids,
            )
            assert isinstance(fused, itir.FunCall)
            type_inference.copy_type(from_=node, to=fused, allow_untyped=True)
            node = fused
        raise RuntimeError("Fusing list-valued 'as_fieldop' arguments did not converge.")


class _InlineIteratorLets(eve.PreserveLocationVisitor, eve.NodeTranslator):
    """`let(it, shift(...)(a))(... it ... it ...)` -> `... shift(...)(a) ... shift(...)(a) ...`."""

    def visit_FunCall(self, node: itir.FunCall, **kwargs) -> itir.Expr:
        node = self.generic_visit(node, **kwargs)
        if cpm.is_let(node):
            eligible = [cpm.is_applied_shift(arg) for arg in node.args]
            if any(eligible):
                return inline_lambdas.inline_lambda(node, eligible_params=eligible)
        return node


@dataclasses.dataclass(frozen=True)
class _ResolveChains(eve.PreserveLocationVisitor, eve.NodeTranslator):
    """Replace every structured shift chain by its constant Cartesian shift at output colour `color`."""

    structured: dict[str, common.StructuredConnectivityType]
    params: frozenset[str]
    color: int

    def _resolve(self, pairs: list[itir.Expr]) -> list[itir.Expr]:
        color_dims = {conn.color_dim for conn in self.structured.values()}
        if len(color_dims) != 1:
            raise NotImplementedError("Structured connectivities with different colour dimensions.")
        (color_dim,) = color_dims

        color = self.color
        offsets: dict[common.Dimension, int] = {}
        kept: list[itir.Expr] = []
        for tag, index in zip(pairs[::2], pairs[1::2], strict=True):
            if isinstance(tag, itir.OffsetLiteral) and tag.value in self.structured:
                if not (isinstance(index, itir.OffsetLiteral) and isinstance(index.value, int)):
                    raise ValueError(f"Structured shift '{tag.value}' has a non-constant index.")
                conn = self.structured[tag.value]
                if color not in conn.colors:
                    raise ValueError(f"Colour {color} is not a source colour of '{tag.value}'.")
                for dim, offset in conn.neighbor_offset(color, index.value).items():
                    if dim == color_dim:
                        color += offset
                    else:
                        offsets[dim] = offsets.get(dim, 0) + offset
            elif isinstance(tag, itir.CartesianOffset):
                if ir_misc.dim_from_axis_literal(tag.domain) == color_dim:
                    raise NotImplementedError("Cartesian shift in the colour dimension.")
                kept.extend((tag, index))
            else:
                raise ValueError(f"Shift '{tag}' mixed with structured shifts.")
        offsets[color_dim] = color - self.color

        resolved: list[itir.Expr] = []
        for dim in sorted(offsets, key=lambda d: d.value):
            if offsets[dim] != 0:
                resolved.extend((im.cartesian_offset(dim), itir.OffsetLiteral(value=offsets[dim])))
        return [*resolved, *kept]

    def visit_FunCall(self, node: itir.FunCall, **kwargs) -> itir.Expr:
        node = self.generic_visit(node, **kwargs)
        if not cpm.is_applied_shift(node):
            return node
        pairs = node.fun.args
        if not any(
            isinstance(tag, itir.OffsetLiteral) and tag.value in self.structured
            for tag in pairs[::2]
        ):
            return node
        (base,) = node.args
        if not (isinstance(base, itir.SymRef) and base.id in self.params):
            raise ValueError(f"Structured shift of '{base}', which is not a stencil parameter.")
        resolved = self._resolve(pairs)
        if not resolved:
            return im.ref(base.id, base.type)
        return im.call(im.call("shift")(*resolved))(base)


def _normalize_stencil(
    stencil: itir.Lambda,
    arg_types: list[ts.TypeSpec | None],
    *,
    offset_provider_type: common.OffsetProviderType,
    uids: utils.IDGeneratorPool,
) -> itir.Lambda:
    """Unroll reductions and flatten shift chains so that every chain is `shift(...)(param)`."""

    def typed(stencil: itir.Lambda) -> itir.Lambda:
        applied = im.as_fieldop(stencil)(
            *(im.ref(f"__arg{i}", type_) for i, type_ in enumerate(arg_types))
        )
        applied = type_inference.infer(
            applied, offset_provider_type=offset_provider_type, allow_undeclared_symbols=True
        )
        return applied.fun.args[0]

    for _ in range(_MAX_ITERATIONS):
        new_stencil = FuseMaps(uids=uids).visit(typed(stencil))
        new_stencil = UnrollReduce.apply(
            typed(new_stencil), offset_provider_type=offset_provider_type, uids=uids
        )
        new_stencil = CollapseListGet().visit(new_stencil)
        new_stencil = NormalizeShifts().visit(new_stencil)
        new_stencil = inline_lifts.InlineLifts().visit(new_stencil)
        # `force_inline_lambda_args` beta-reduces the `_step` functions of `UnrollReduce`, so that
        # shift indices become literals, and the thunks of `InlineCenterDerefLiftVars`, so that
        # `list_get` meets the `neighbors` it indexes.
        new_stencil = InlineLambdas.apply(
            new_stencil,
            opcount_preserving=True,
            force_inline_lift_args=True,
            force_inline_lambda_args=True,
        )
        new_stencil = _InlineIteratorLets().visit(new_stencil)
        new_stencil = CollapseListGet().visit(new_stencil)
        new_stencil = NormalizeShifts().visit(new_stencil)
        if new_stencil == stencil:
            return stencil
        stencil = new_stencil
    raise RuntimeError("Normalising a structured stencil did not converge.")


def _check_postcondition(
    program: itir.Program, structured: dict[str, common.StructuredConnectivityType]
) -> None:
    if leftover := _structured_tags(program, structured).to_list():
        raise ValueError(f"Structured offsets {[lit.value for lit in leftover]} remain.")
    for call in program.pre_walk_values().if_isinstance(itir.FunCall):
        if cpm.is_call_to(call, ("neighbors", "unstructured_domain")) or cpm.is_applied_reduce(
            call
        ):
            raise ValueError(f"'{call}' remains after 'StructuredToCartesian'.")
        if cpm.is_let(call) and any(_is_list_field(arg) for arg in call.args):
            raise ValueError(f"List-valued let '{call}' remains after 'StructuredToCartesian'.")


@dataclasses.dataclass(frozen=True)
class StructuredToCartesian(eve.PreserveLocationVisitor, eve.NodeTranslator):
    """
    Rewrite a program on a structured (colour-indexed lattice) layout into a Cartesian program.

    Fields of an unstructured entity (Cell, Edge, ...) live on a lattice `(I, J, X)` whose colour
    dimension `X` enumerates the entities of one lattice cell; the offset provider maps each
    unstructured offset to a `common.StructuredConnectivityType` giving, per source colour and
    neighbor, a constant `(dI, dJ, dX)`. With C2E colour 0: `[{}, {X: 1}, {X: 2}]` and
    colour 1: `[{X: -1}, {I: 1}, {X: 1}]`::

        as_fieldop(λ(e, w) → reduce(plus, 0)(map_list(multiplies)(neighbors(C2E, e), ·w)))(e, w)

    becomes::

        concat_where(X < 1,
          as_fieldop(λ(e, w) → ((0 + ·e × list_get(0, ·w))
                                   + ·⟪X, 1⟫(e) × list_get(1, ·w)) + ·⟪X, 2⟫(e) × list_get(2, ·w))(e, w),
          as_fieldop(λ(e, w) → ((0 + ·⟪X, -1⟫(e) × list_get(0, ·w))
                                   + ·⟪I, 1⟫(e) × list_get(1, ·w)) + ·⟪X, 1⟫(e) × list_get(2, ·w))(e, w))

    Steps: domains become `cartesian_domain`s; list-valued lets are inlined and list-valued
    arguments fused into their consumers; per `as_fieldop` whose stencil uses a structured offset
    or a `reduce`, reductions are unrolled and shift chains flattened, then every chain is resolved
    to a constant Cartesian shift per output colour (the source colours of the first structured
    offset of the chains, which must share their source dimension) and the colour stencils are
    joined by `concat_where` on half-infinite colour ranges. Cartesian shifts in a chain, static or
    dynamic, are kept after the resolved ones. `list_get` on a sparse field is left as is.

    Growth: a list-valued let is copied once per use, a reduction `max_neighbors` times and a
    structured stencil once per output colour; non-trivial arguments are let-bound once outside
    the `concat_where`.

    Preconditions: runs before domain inference; program parameters are typed on the lattice;
    tuple arguments are expanded; the provider mixes no neighbor tables with structured entries.
    A program without structured entries in the offset provider is returned unchanged.
    """  # noqa: RUF002  # ambiguous multiplication character in printed IR

    PRESERVED_ANNEX_ATTRS = ("domain",)

    structured: dict[str, common.StructuredConnectivityType]
    offset_provider_type: common.OffsetProviderType
    uids: utils.IDGeneratorPool

    @classmethod
    def apply(
        cls,
        program: itir.Program,
        *,
        offset_provider_type: common.OffsetProviderType,
        uids: utils.IDGeneratorPool,
    ) -> itir.Program:
        structured = _structured_connectivities(offset_provider_type)
        if not structured:
            return program
        if any(
            isinstance(conn, common.NeighborConnectivityType)
            for conn in offset_provider_type.values()
        ):
            raise ValueError(
                "An offset provider must not mix neighbor tables with structured connectivities."
            )

        entity_dims = frozenset(
            dim for conn in structured.values() for dim in (conn.source_dim, conn.codomain)
        )
        lattice_dims = tuple(
            common.order_dimensions(
                {conn.color_dim for conn in structured.values()}
                | {
                    dim
                    for conn in structured.values()
                    for _, per_neighbor in conn.offsets
                    for offset in per_neighbor
                    for dim, _ in offset
                }
            )
        )
        program = _RewriteDomains(entity_dims=entity_dims, lattice_dims=lattice_dims).visit(program)
        program = type_inference.infer(program, offset_provider_type=offset_provider_type)
        program = _FuseListFields(offset_provider_type=offset_provider_type, uids=uids).visit(
            program
        )
        program = type_inference.infer(program, offset_provider_type=offset_provider_type)
        program = cls(
            structured=structured, offset_provider_type=offset_provider_type, uids=uids
        ).visit(program)
        program = type_inference.infer(program, offset_provider_type=offset_provider_type)
        _check_postcondition(program, structured)
        return program

    def _is_structured_stencil(self, stencil: itir.Expr) -> bool:
        return bool(_structured_tags(stencil, self.structured).to_list()) or any(
            cpm.is_applied_reduce(call)
            for call in stencil.pre_walk_values().if_isinstance(itir.FunCall)
        )

    def _output_colors(self, stencil: itir.Lambda) -> tuple[common.StructuredConnectivityType, ...]:
        first_tags = []
        for call in stencil.pre_walk_values().if_isinstance(itir.FunCall):
            if cpm.is_applied_shift(call):
                tags = [
                    tag.value
                    for tag in call.fun.args[::2]
                    if isinstance(tag, itir.OffsetLiteral) and tag.value in self.structured
                ]
                if tags:
                    first_tags.append(self.structured[tags[0]])
        return tuple(dict.fromkeys(first_tags))  # ordered set for reproducibility

    def visit_FunCall(self, node: itir.FunCall, **kwargs) -> itir.Expr:
        # pre-order: the arguments still carry the types of the whole-program inference
        if not cpm.is_applied_as_fieldop(node) or not self._is_structured_stencil(node.fun.args[0]):
            return self.generic_visit(node, **kwargs)

        node = ir_misc.canonicalize_as_fieldop(node)
        assert cpm.is_applied_as_fieldop(node)
        stencil, *domain = node.fun.args
        if cpm.is_call_to(stencil, "scan"):
            raise NotImplementedError("Structured shifts in a 'scan' are not supported.")
        assert isinstance(stencil, itir.Lambda)
        result_type = node.type
        arg_types = [arg.type for arg in node.args]
        args = self.visit(node.args, **kwargs)

        stencil = _normalize_stencil(
            stencil, arg_types, offset_provider_type=self.offset_provider_type, uids=self.uids
        )
        params = frozenset(param.id for param in stencil.params)
        first_conns = self._output_colors(stencil)
        if len({(conn.source_dim, conn.colors) for conn in first_conns}) > 1:
            raise ValueError(
                "Structured shift chains of one stencil start from different source dimensions."
            )
        if not first_conns:
            return im.as_fieldop(stencil, *domain)(*args)
        conn = first_conns[0]
        branches = [
            _ResolveChains(structured=self.structured, params=params, color=color).visit(stencil)
            for color in conn.colors
        ]
        if len(branches) == 1:
            return im.as_fieldop(branches[0], *domain)(*args)

        if isinstance(result_type, ts.TupleType):
            raise NotImplementedError("Tuple-valued structured 'as_fieldop's are not supported.")
        bindings: dict[str, itir.Expr] = {}
        arg_names: list[tuple[str, ts.TypeSpec | None]] = []
        for arg in args:
            if isinstance(arg, itir.SymRef):
                arg_names.append((str(arg.id), arg.type))
            else:
                name = next(self.uids["__sc_arg"])
                bindings[name] = arg
                arg_names.append((name, arg.type))

        def branch(stencil: itir.Lambda) -> itir.Expr:
            return im.as_fieldop(stencil, *domain)(*(im.ref(name, t) for name, t in arg_names))

        expr = branch(branches[-1])
        for color, stencil in reversed(list(zip(conn.colors[:-1], branches[:-1]))):
            expr = im.concat_where(
                im.less(im.axis_literal(conn.color_dim), color + 1), branch(stencil), expr
            )
        return im.let(*bindings.items())(expr) if bindings else expr


structured_to_cartesian = StructuredToCartesian.apply
