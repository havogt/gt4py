# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import warnings
from typing import Optional, Protocol

from gt4py.next import common, utils
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms import (
    concat_where,
    dead_code_elimination,
    fuse_as_fieldop,
    global_tmps,
    infer_domain,
    infer_domain_ops,
    inline_dynamic_shifts,
    inline_fundefs,
    inline_lifts,
    prune_casts,
    prune_empty_concat_where,
    remove_broadcast,
    symbol_ref_utils,
)
from gt4py.next.iterator.transforms.collapse_list_get import CollapseListGet
from gt4py.next.iterator.transforms.collapse_tuple import CollapseTuple
from gt4py.next.iterator.transforms.constant_folding import ConstantFolding
from gt4py.next.iterator.transforms.cse import CommonSubexpressionElimination
from gt4py.next.iterator.transforms.fuse_maps import FuseMaps
from gt4py.next.iterator.transforms.inline_lambdas import InlineLambdas, rename_symbols
from gt4py.next.iterator.transforms.inline_scalar import InlineScalar
from gt4py.next.iterator.transforms.merge_let import MergeLet
from gt4py.next.iterator.transforms.normalize_shifts import NormalizeShifts
from gt4py.next.iterator.transforms.unroll_reduce import UnrollReduce
from gt4py.next.iterator.type_system.inference import infer


class GTIRTransform(Protocol):
    def __call__(
        self, _: itir.Program, *, offset_provider: common.OffsetProvider
    ) -> itir.Program: ...


def _max_domain_range_sizes(offset_provider: common.OffsetProvider) -> dict[str, itir.Literal]:
    """
    Extract horizontal domain sizes from an `offset_provider`.

    Considers the shape of the neighbor table to get the size of each `source_dim` and the maximum
    value inside the neighbor table to get the size of each `codomain`.
    """
    sizes: dict[str, int] = {}
    for provider in offset_provider.values():
        if common.is_neighbor_connectivity(provider):
            src_dim = provider.__gt_type__().source_dim.value
            codomain_dim = provider.__gt_type__().codomain.value
            sizes[src_dim] = max(sizes.get(src_dim, 0), provider.ndarray.shape[0])
            sizes[codomain_dim] = max(
                sizes.get(codomain_dim, 0),
                int(provider.ndarray.max()) + 1,  # type: ignore[attr-defined] # TODO(havogt): improve typing for NDArrayObject
            )

    sizes_exprs = {k: im.literal_from_value(v) for k, v in sizes.items()}
    return sizes_exprs


def _has_dynamic_domains(ir: itir.Program) -> bool:
    # note: this function does not respect symbol collisions with builtins. As it is a temporary
    # workaround we don't care about this corner case.
    domains = set()
    domains |= ir.walk_values().if_isinstance(itir.SetAt).getattr("domain").to_set()
    for as_fop in (
        ir.walk_values()
        .if_isinstance(itir.FunCall)
        .filter(lambda node: cpm.is_call_to(node, "as_fieldop") and len(node.args) == 2)
    ):
        domains.add(as_fop.args[1])
    return len(symbol_ref_utils.collect_symbol_refs(domains)) > 0


def _process_symbolic_domains_option(
    ir: itir.Program,
    offset_provider: common.OffsetProvider,
    symbolic_domain_sizes: Optional[dict[str, itir.Expr]],
    use_max_domain_range_on_unstructured_shift: Optional[bool],
) -> Optional[dict[str, itir.Expr]]:
    """
    Given a program, offset_provider and some configuration options determine how domains are
    inferred.

    The output of this function is used as `symbolic_domain_sizes` argument of domain inference, i.e.
    :func:`infer_domain.infer_program`.

    Right now domains of `as_fieldop` expressions can be inferred either a) using static information
    from the offset provider, or b) they are set to an expression controlled by
    the user and configured in the backend, or c) they are set to the maximum possible domain /
    everywhere (see :func:`_max_domain_range_sizes`)

    Option a) applies when the program is decorated with `static_domains = True` (unless option c)
    is explicitly requested). Then all dynamic domains were replaced with static ones
    which we recognize here. The domain inference then uses this static information which we
    communicate by returning `None`, i.e. no symbolic domain sizes.
    Option b) applies when the user explicitly configured `symbolic_domain_sizes` in the backend.
    In that case we just forward the value.
    Option c) applies when `static_domains = False` or when explicitly configured in the backend
    with `use_max_domain_range_on_unstructured_shift = True`. In that case we determine the
    maximum sizes using :func:`_max_domain_range_sizes` and return them.
    """
    if symbolic_domain_sizes:
        assert not use_max_domain_range_on_unstructured_shift, "Options are mutually exclusive."
        return symbolic_domain_sizes

    if use_max_domain_range_on_unstructured_shift is None:
        use_max_domain_range_on_unstructured_shift = _has_dynamic_domains(ir)
    elif use_max_domain_range_on_unstructured_shift:
        if not _has_dynamic_domains(ir):
            warnings.warn(
                "You are using static domains together with "
                "'use_max_domain_range_on_unstructured_shift'. This is "
                "likely not what you wanted.",
                stacklevel=2,
            )
    if use_max_domain_range_on_unstructured_shift:
        assert not symbolic_domain_sizes, "Options are mutually exclusive."
        symbolic_domain_sizes = _max_domain_range_sizes(offset_provider)  # type: ignore[assignment]
    return symbolic_domain_sizes


def _merge_same_domain_temporaries(
    program: itir.Program,
    *,
    offset_provider_type: common.OffsetProviderType,
    uids: utils.IDGeneratorPool,
) -> itir.Program:
    """Merge runs of same-domain, mutually-independent temporary `SetAt`s (whose RHS is an
    applied `as_fieldop`) into a single tuple-valued `SetAt` whose RHS is ONE tuple-returning
    `as_fieldop`, sharing common loads via CSE — so they lower to a single kernel instead of one
    kernel per temporary (e.g. the 4 Green-Gauss gradient reductions → 1, sharing C2E2CO gathers).

    The merged `as_fieldop` is constructed DIRECTLY (not via `make_tuple` + `FuseAsFieldOp`): each
    member's sub-stencil body is remapped onto a structurally-deduplicated union of the members'
    arguments, then combined into `λ(merged_params) → make_tuple(bodies...)`. The result is always
    a single applied `as_fieldop` (gtfn-lowerable) regardless of (dynamic) domain. As a safety net,
    a group that cannot be constructed this way is left unmerged."""

    def build_merged(group: list[itir.Stmt]) -> Optional[itir.Expr]:
        merged_args: list[itir.Expr] = []
        merged_params: list[str] = []
        bodies: list[itir.Expr] = []
        key_to_param: dict[str, str] = {}
        for g in group:
            stencil = g.expr.fun.args[0]  # type: ignore[attr-defined]
            args = g.expr.args  # type: ignore[attr-defined]
            if not isinstance(stencil, itir.Lambda) or len(stencil.params) != len(args):
                return None  # only handle lambda stencils with matching arity
            rename_map: dict[str, str | itir.SymRef] = {}
            for param, arg in zip(stencil.params, args):
                key = arg.id if isinstance(arg, itir.SymRef) else str(arg)
                if key not in key_to_param:
                    fresh = next(uids["__merged_arg"])
                    key_to_param[key] = fresh
                    merged_params.append(fresh)
                    merged_args.append(arg)
                rename_map[param.id] = key_to_param[key]
            bodies.append(rename_symbols(stencil.expr, rename_map))
        merged_stencil = im.lambda_(*merged_params)(im.make_tuple(*bodies))
        return im.as_fieldop(merged_stencil, group[0].domain)(*merged_args)  # type: ignore[attr-defined]

    def is_mergeable(stmt: itir.Stmt) -> bool:
        return (
            isinstance(stmt, itir.SetAt)
            and isinstance(stmt.target, itir.SymRef)
            and cpm.is_applied_as_fieldop(stmt.expr)
        )

    stmts = program.body
    new_body: list[itir.Stmt] = []
    merged_any = False
    i = 0
    while i < len(stmts):
        stmt = stmts[i]
        if is_mergeable(stmt):
            dom = stmt.domain
            group = [stmt]
            group_targets = {stmt.target.id}  # type: ignore[attr-defined]
            j = i + 1
            while (
                j < len(stmts)
                and is_mergeable(stmts[j])
                and stmts[j].domain == dom
                and not (
                    set(symbol_ref_utils.collect_symbol_refs(stmts[j].expr)) & group_targets
                )
            ):
                group.append(stmts[j])
                group_targets.add(stmts[j].target.id)  # type: ignore[attr-defined]
                j += 1
            if len(group) >= 2:
                merged_expr = build_merged(group)
                # safety net: only commit a single applied `as_fieldop` (gtfn-lowerable)
                if merged_expr is not None and cpm.is_applied_as_fieldop(merged_expr):
                    new_body.append(
                        itir.SetAt(
                            target=im.make_tuple(*(g.target for g in group)),  # type: ignore[attr-defined]
                            domain=dom,
                            expr=merged_expr,
                        )
                    )
                    merged_any = True
                    i = j
                    continue
        new_body.append(stmt)
        i += 1

    if not merged_any:
        return program
    merged = itir.Program(
        id=program.id,
        function_definitions=program.function_definitions,
        params=program.params,
        declarations=program.declarations,
        body=new_body,
    )
    # type, then CSE to share the common gathers across the merged tuple stencil
    merged = infer(merged, inplace=True, offset_provider_type=offset_provider_type)
    merged = CommonSubexpressionElimination.apply(
        merged, offset_provider_type=offset_provider_type, uids=uids
    )
    return merged


# TODO(tehrengruber): Revisit interface to configure temporary extraction. We currently forward
#  `extract_temporaries` and `temporary_extraction_heuristics` which is inconvenient.
def apply_common_transforms(
    ir: itir.Program,
    *,
    offset_provider: common.OffsetProvider | common.OffsetProviderType,
    extract_temporaries=False,
    unroll_reduce=False,
    common_subexpression_elimination=True,
    force_inline_lambda_args=False,
    #: A dictionary mapping axes names to their length. See :func:`infer_domain.infer_expr` for
    #: more details.
    symbolic_domain_sizes: Optional[dict[str, itir.Expr]] = None,
    # TODO(tehrengruber): Remove this option again as soon as we have the necessary builtins
    #  to work with / translate domains.
    use_max_domain_range_on_unstructured_shift: Optional[bool] = None,
    #: Merge same-domain, independent, statically-shaped temporaries into a single kernel
    #: (e.g. the Green-Gauss gradient reductions). Not yet robust for dynamic-domain programs,
    #: hence opt-in.
    merge_tmps: bool = False,
    #: Inline (recompute) a reduction-temp accessed at a vertical (Koff) shift instead of
    #: materializing it as a temp, dropping a kernel and its DRAM round-trip. Opt-in.
    vertical_shift_fusion: bool = False,
) -> itir.Program:
    assert isinstance(ir, itir.Program)
    # TODO(tehrengruber): Allow `common.OffsetProviderType`, but domain inference currently
    #  relies on static information or `symbolic_domain_sizes`.
    assert common.is_offset_provider(offset_provider)

    offset_provider_type = common.offset_provider_to_type(offset_provider)

    symbolic_domain_sizes = _process_symbolic_domains_option(
        ir, offset_provider, symbolic_domain_sizes, use_max_domain_range_on_unstructured_shift
    )

    uids = utils.IDGeneratorPool()

    ir = MergeLet().visit(ir)
    ir = inline_fundefs.InlineFundefs().visit(ir)

    ir = inline_fundefs.prune_unreferenced_fundefs(ir)
    ir = NormalizeShifts().visit(ir)

    # TODO(tehrengruber): Many iterator test contain lifts that need to be inlined, e.g.
    #  test_can_deref. We didn't notice previously as FieldOpFusion did this implicitly everywhere.
    ir = inline_lifts.InlineLifts().visit(ir)

    ir = concat_where.expand_tuple_args(ir, offset_provider_type=offset_provider_type)  # type: ignore[assignment]  # always an itir.Program
    ir = dead_code_elimination.dead_code_elimination(
        ir, uids=uids, offset_provider_type=offset_provider_type
    )  # domain inference does not support dead-code
    ir = inline_dynamic_shifts.InlineDynamicShifts.apply(
        ir, offset_provider_type=offset_provider_type, uids=uids
    )  # domain inference does not support dynamic offsets yet
    ir = infer_domain_ops.InferDomainOps.apply(ir)
    ir = concat_where.canonicalize_domain_argument(ir)

    ir = infer_domain.infer_program(
        ir,
        offset_provider=offset_provider,
        symbolic_domain_sizes=symbolic_domain_sizes,
    )
    ir = prune_empty_concat_where.prune_empty_concat_where(ir)
    ir = remove_broadcast.RemoveBroadcast.apply(ir)

    ir = concat_where.transform_to_as_fieldop(ir)

    for _ in range(10):
        inlined = ir

        inlined = InlineLambdas.apply(inlined, opcount_preserving=True)
        inlined = ConstantFolding.apply(inlined)  # type: ignore[assignment]  # always an itir.Program
        # This pass is required to be in the loop such that when an `if_` call with tuple arguments
        # is constant-folded the surrounding tuple_get calls can be removed.
        inlined = CollapseTuple.apply(
            inlined,
            enabled_transformations=~CollapseTuple.Transformation.PROPAGATE_TO_IF_ON_TUPLES,
            uids=uids,
            offset_provider_type=offset_provider_type,
        )  # type: ignore[assignment]  # always an itir.Program
        inlined = InlineScalar.apply(inlined, offset_provider_type=offset_provider_type)

        # This pass is required to run after CollapseTuple as otherwise we can not inline
        # expressions like `tuple_get(make_tuple(as_fieldop(stencil)(...)))` where stencil returns
        # a list. Such expressions must be inlined however because no backend supports such
        # field operators right now.
        inlined = fuse_as_fieldop.FuseAsFieldOp.apply(
            inlined,
            uids=uids,
            offset_provider_type=offset_provider_type,
            vertical_shift_fusion=vertical_shift_fusion,
        )

        if inlined == ir:
            break
        ir = inlined
    else:
        raise RuntimeError("Inlining 'lift' and 'lambdas' did not converge.")

    # breaks in test_zero_dim_tuple_arg as trivial tuple_get is not inlined
    if common_subexpression_elimination:
        ir = CommonSubexpressionElimination.apply(
            ir, offset_provider_type=offset_provider_type, uids=uids
        )
        ir = MergeLet().visit(ir)
        ir = InlineLambdas.apply(ir, opcount_preserving=True)

    if extract_temporaries:
        ir = infer(ir, inplace=True, offset_provider_type=offset_provider_type)
        # Fusion can create identity `as_fieldop(λx→·x)(field)` copies (e.g. from pruned
        # vp==wp casts) as lambda args; global_tmps would materialize each as a standalone
        # copy kernel + temporary. Eliminate them (PruneCasts removes the identity fieldop)
        # and inline the resulting trivial (SymRef) lets before temp extraction.
        ir = prune_casts.PruneCasts.apply(ir)
        ir = InlineLambdas.apply(ir, opcount_preserving=True)
        ir = global_tmps.create_global_tmps(
            ir,
            offset_provider=offset_provider,
            symbolic_domain_sizes=symbolic_domain_sizes,
            uids=uids,
        )
        # EXPERIMENT(h7): merge same-domain independent temporaries into one kernel (the 4
        # Green-Gauss gradient reductions → 1, sharing C2E2CO gathers). The merge constructs a
        # single tuple-returning `as_fieldop` directly and only commits gtfn-lowerable results,
        # so it is safe for any program; best-effort try/except is a final safety net.
        if merge_tmps:
            try:
                ir = _merge_same_domain_temporaries(
                    ir, offset_provider_type=offset_provider_type, uids=uids
                )
            except Exception:
                pass  # merge is best-effort; on any failure keep the un-merged program

    ir = NormalizeShifts().visit(ir)

    ir = FuseMaps(uids=uids).visit(ir)
    ir = CollapseListGet().visit(ir)

    if unroll_reduce:
        for _ in range(10):
            unrolled = UnrollReduce.apply(ir, offset_provider_type=offset_provider_type, uids=uids)
            unrolled = CollapseListGet().visit(unrolled)
            unrolled = NormalizeShifts().visit(unrolled)
            # this is required as nested neighbor reductions can contain lifts, e.g.,
            # `neighbors(V2Eₒ, ↑f(...))`
            unrolled = inline_lifts.InlineLifts().visit(unrolled)
            unrolled = NormalizeShifts().visit(unrolled)
            if unrolled == ir:
                break
            ir = unrolled
        else:
            raise RuntimeError("Reduction unrolling failed.")

    ir = InlineLambdas.apply(
        ir, opcount_preserving=True, force_inline_lambda_args=force_inline_lambda_args
    )

    assert isinstance(ir, itir.Program)
    return ir


def apply_fieldview_transforms(
    ir: itir.Program,
    *,
    offset_provider: common.OffsetProvider,
    # TODO(tehrengruber): Remove this option again as soon as we have the necessary builtins
    #  to work with / translate domains.
    use_max_domain_range_on_unstructured_shift: Optional[bool] = None,
) -> itir.Program:
    offset_provider_type = common.offset_provider_to_type(offset_provider)

    uids = utils.IDGeneratorPool()

    symbolic_domain_sizes = _process_symbolic_domains_option(
        ir, offset_provider, None, use_max_domain_range_on_unstructured_shift
    )

    ir = inline_fundefs.InlineFundefs().visit(ir)
    ir = inline_fundefs.prune_unreferenced_fundefs(ir)
    # required for dead-code-elimination and `prune_empty_concat_where` pass
    ir = concat_where.expand_tuple_args(ir, offset_provider_type=offset_provider_type)  # type: ignore[assignment]  # always an itir.Program
    ir = dead_code_elimination.dead_code_elimination(
        ir, offset_provider_type=offset_provider_type, uids=uids
    )
    ir = inline_dynamic_shifts.InlineDynamicShifts.apply(
        ir, offset_provider_type=offset_provider_type, uids=uids
    )  # domain inference does not support dynamic offsets yet

    ir = infer_domain_ops.InferDomainOps.apply(ir)
    ir = concat_where.canonicalize_domain_argument(ir)
    ir = ConstantFolding.apply(ir)  # type: ignore[assignment]  # always an itir.Program

    ir = infer_domain.infer_program(
        ir,
        symbolic_domain_sizes=symbolic_domain_sizes,
        offset_provider=offset_provider,
    )
    ir = prune_empty_concat_where.prune_empty_concat_where(ir)
    ir = remove_broadcast.RemoveBroadcast.apply(ir)
    return ir
