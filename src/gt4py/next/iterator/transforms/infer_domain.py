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

import copy

from gt4py.eve import utils as eve_utils
from gt4py.eve.extended_typing import Dict, Tuple
from gt4py.next.common import Dimension
from gt4py.next.iterator import ir as itir
from gt4py.next.iterator.ir_utils import common_pattern_matcher as cpm, ir_makers as im
from gt4py.next.iterator.transforms.global_tmps import AUTO_DOMAIN, SymbolicDomain, domain_union
from gt4py.next.iterator.transforms.trace_shifts import TraceShifts


def _merge_domains(
    original_domains: Dict[str, SymbolicDomain], additional_domains: Dict[str, SymbolicDomain]
) -> Dict[str, SymbolicDomain]:
    new_domains = {**original_domains}
    for key, value in additional_domains.items():
        if key in original_domains:
            new_domains[key] = domain_union([original_domains[key], value])
        else:
            new_domains[key] = value

    return new_domains


# TODO: Revisit. Until TraceShifts directly supports stencils we just wrap our expression into a dummy closure in this
#  helper function.
def trace_shifts(stencil: itir.Expr, input_ids: list[str], domain: itir.Expr):
    node = itir.StencilClosure(
        stencil=stencil,
        inputs=[im.ref(id_) for id_ in input_ids],
        output=im.ref("__dummy"),
        domain=domain,
    )
    return TraceShifts.apply(node)


def extract_shifts_and_translate_domains(
    stencil: itir.Expr,
    input_ids: list[str],
    input_domain: SymbolicDomain,
    offset_provider: Dict[str, Dimension],
    accessed_domains: Dict[str, SymbolicDomain],
):
    shifts_results = trace_shifts(stencil, input_ids, SymbolicDomain.as_expr(input_domain))

    for in_field_id in input_ids:
        shifts_list = shifts_results[in_field_id]

        new_domains = [
            SymbolicDomain.translate(input_domain, shift, offset_provider) for shift in shifts_list
        ]
        accessed_domains[in_field_id] = domain_union(new_domains)


def process_expr(
    expr: itir.FunCall, domain: SymbolicDomain | itir.FunCall, offset_provider
) -> Tuple[itir.FunCall, Dict[str, SymbolicDomain]]:
    if isinstance(expr, itir.SymRef):
        return expr, {expr.id: domain}
    elif isinstance(expr, itir.Literal):
        return expr, {}
    elif isinstance(expr.fun, itir.Lambda):
        return infer_let(expr, domain, offset_provider)
    elif cpm.is_call_to(expr, "cond"):
        return infer_cond(expr, domain, offset_provider)
    elif cpm.is_call_to(expr.fun, "as_fieldop"):
        return infer_as_fieldop(expr, domain, offset_provider)
    elif cpm.is_call_to(expr, "make_tuple"):
        return infer_make_tuple(expr, domain, offset_provider)
    elif cpm.is_call_to(expr, "tuple_get"):
        return infer_tuple_get(expr, domain, offset_provider)
    else:
        raise ValueError(f"Unsupported function call: {expr.fun}")


def infer_as_fieldop(
    applied_fieldop: itir.FunCall,
    input_domain: SymbolicDomain | itir.FunCall,
    offset_provider: Dict[str, Dimension],
) -> Tuple[itir.FunCall, Dict[str, SymbolicDomain]]:
    assert isinstance(applied_fieldop, itir.FunCall)
    assert cpm.is_call_to(applied_fieldop.fun, "as_fieldop")

    stencil, inputs = applied_fieldop.fun.args[0], applied_fieldop.args

    input_ids: list[str] = []
    accessed_domains: Dict[str, SymbolicDomain] = {}

    # Assign ids for all inputs to `as_fieldop`. `SymRef`s stay as is, nested `as_fieldop` get a
    # temporary id.
    tmp_uid_gen = eve_utils.UIDGenerator(prefix="__dom_inf")
    for in_field in inputs:
        if isinstance(in_field, itir.FunCall) or isinstance(in_field, itir.Literal):
            id_ = tmp_uid_gen.sequential_id()
        elif isinstance(in_field, itir.SymRef):
            id_ = in_field.id
        else:
            raise ValueError(f"Unsupported type {type(in_field)}")
        input_ids.append(id_)

    if isinstance(input_domain, itir.FunCall):
        input_domain = SymbolicDomain.from_expr(input_domain)

    extract_shifts_and_translate_domains(
        stencil, input_ids, input_domain, offset_provider, accessed_domains
    )

    # Recursively infer domain of inputs and update domain arg of nested `as_fieldops`
    transformed_inputs: list[itir.Expr] = []
    for in_field_id, in_field in zip(input_ids, inputs):
        if isinstance(in_field, itir.FunCall):
            transformed_input, accessed_domains_tmp = process_expr(
                in_field, accessed_domains[in_field_id], offset_provider
            )
            # transformed_input, accessed_domains_tmp = infer_as_fieldop(
            #     in_field, accessed_domains[in_field_id], offset_provider
            # )
            transformed_inputs.append(transformed_input)

            # Merge accessed_domains and accessed_domains_tmp
            accessed_domains = _merge_domains(accessed_domains, accessed_domains_tmp)
        elif isinstance(in_field, itir.SymRef) or isinstance(in_field, itir.Literal):
            transformed_inputs.append(in_field)
        else:
            raise ValueError(f"Unsupported type {type(in_field)}")

    transformed_call = im.as_fieldop(stencil, SymbolicDomain.as_expr(input_domain))(
        *transformed_inputs
    )

    accessed_domains_without_tmp = {
        k: v
        for k, v in accessed_domains.items()
        if not k.startswith(tmp_uid_gen.prefix)  # type: ignore[arg-type] # prefix is always str
    }

    return transformed_call, accessed_domains_without_tmp


def infer_let(
    applied_let: itir.FunCall,
    input_domain: SymbolicDomain | itir.FunCall,
    offset_provider: Dict[str, Dimension],
) -> Tuple[itir.FunCall, Dict[str, SymbolicDomain]]:
    assert isinstance(applied_let, itir.FunCall) and isinstance(applied_let.fun, itir.Lambda)
    assert isinstance(applied_let.fun.expr, itir.FunCall)

    accessed_domains: Dict[str, SymbolicDomain] = {}

    transformed_calls_expr, accessed_domains_expr = process_expr(
        applied_let.fun.expr, input_domain, offset_provider
    )

    transformed_calls_args: list[itir.FunCall] = []
    for arg in applied_let.args:
        assert isinstance(arg, itir.FunCall)
        param_id = applied_let.fun.params[0].id
        transformed_calls_arg, accessed_domains_arg = process_expr(
            arg, accessed_domains_expr[param_id], offset_provider
        )
        transformed_calls_args.append(transformed_calls_arg)
        accessed_domains = _merge_domains(accessed_domains, accessed_domains_arg)

    transformed_call = im.let(
        *(
            (str(param.id), call)
            for param, call in zip(applied_let.fun.params, transformed_calls_args)
        )
    )(transformed_calls_expr)

    return transformed_call, accessed_domains


def infer_make_tuple(make_tuple_expr: itir.FunCall, input_domain, offset_provider):
    accessed_domain = {}

    transformed_calls_args = []
    for arg in make_tuple_expr.args:
        transformed_calls_arg, accessed_domains_arg = process_expr(
            arg, input_domain, offset_provider
        )
        transformed_calls_args.append(transformed_calls_arg)
        accessed_domain = _merge_domains(accessed_domain, accessed_domains_arg)

    transformed_call = im.make_tuple(*transformed_calls_args)
    return transformed_call, accessed_domain


def infer_tuple_get(tuple_get_expr, input_domain, offset_provider):
    transformed_elem, accessed_domain = process_expr(
        tuple_get_expr.args[1], input_domain, offset_provider
    )
    return im.tuple_get(tuple_get_expr.args[0], transformed_elem), accessed_domain


def infer_cond(
    applied_cond: itir.FunCall,
    input_domain: SymbolicDomain | itir.FunCall,
    offset_provider: Dict[str, Dimension],
) -> Tuple[itir.FunCall, Dict[str, SymbolicDomain]]:
    assert isinstance(applied_cond, itir.FunCall) and cpm.is_call_to(applied_cond, "cond")
    assert (
        isinstance(applied_cond.args[0], bool)
        and isinstance(applied_cond.args[1], itir.FunCall)
        and isinstance(applied_cond.args[2], itir.FunCall)
    )
    call_true, domains_true = infer_as_fieldop(applied_cond.args[1], input_domain, offset_provider)
    call_false, domains_false = infer_as_fieldop(
        applied_cond.args[2], input_domain, offset_provider
    )
    return im.cond(applied_cond.args[0], call_true, call_false), _merge_domains(
        domains_true, domains_false
    )


def _validate_temporary_usage(body: list[itir.Stmt], temporaries: list[str]):
    assigned_targets = set()
    for stmt in body:
        assert isinstance(stmt, itir.SetAt)  # TODO: extend for if-statements when they land
        assert isinstance(
            stmt.target, itir.SymRef
        )  # TODO: stmt.target can be an expr, e.g. make_tuple
        if stmt.target.id in assigned_targets:
            raise ValueError("Temporaries can only be used once within a program.")
        if stmt.target.id in temporaries:
            assigned_targets.add(stmt.target.id)


def infer_program(
    program: itir.Program,
    offset_provider: Dict[str, Dimension],
) -> itir.Program:
    accessed_domains: dict[str, SymbolicDomain] = {}
    transformed_set_ats: list[itir.SetAt] = []

    temporaries: list[str] = [tmp.id for tmp in program.declarations]

    _validate_temporary_usage(program.body, temporaries)

    for set_at in reversed(program.body):
        assert isinstance(set_at, itir.SetAt)
        if isinstance(set_at.expr, itir.SymRef):
            transformed_set_ats.insert(0, set_at)
            continue
        assert isinstance(set_at.expr, itir.FunCall)
        assert isinstance(
            set_at.target, itir.SymRef
        )  # TODO: stmt.target can be an expr, e.g. make_tuple
        if set_at.target.id in temporaries:
            # ignore temporaries as their domain is the `AUTO_DOMAIN` placeholder
            assert set_at.domain == AUTO_DOMAIN
        else:
            accessed_domains[set_at.target.id] = SymbolicDomain.from_expr(set_at.domain)
        transformed_call, current_accessed_domains = process_expr(
            set_at.expr, accessed_domains[set_at.target.id], offset_provider
        )
        # if cpm.is_call_to(set_at.expr.fun, "as_fieldop"):
        #     transformed_call, current_accessed_domains = infer_as_fieldop(
        #         set_at.expr, accessed_domains[set_at.target.id], offset_provider
        #     )
        # elif isinstance(set_at.expr.fun, itir.Lambda):
        #     transformed_call, current_accessed_domains = infer_let(
        #         set_at.expr, accessed_domains[set_at.target.id], offset_provider
        #     )
        # else:
        #     raise NotImplementedError(f"{set_at.expr.fun}")
        transformed_set_ats.insert(
            0,
            itir.SetAt(
                expr=transformed_call,
                domain=SymbolicDomain.as_expr(accessed_domains[set_at.target.id]),
                target=set_at.target,
            ),
        )

        for field in current_accessed_domains:
            if field in accessed_domains:
                # multiple accesses to the same field -> compute union of accessed domains
                if field in temporaries:
                    accessed_domains[field] = domain_union(
                        [accessed_domains[field], current_accessed_domains[field]]
                    )
                else:
                    # TODO(tehrengruber): if domain_ref is an external field the domain must
                    #  already be larger. This should be checked, but would require additions
                    #  to the IR.
                    pass
            else:
                accessed_domains[field] = current_accessed_domains[field]

    new_declarations = copy.deepcopy(program.declarations)
    for temporary in new_declarations:
        temporary.domain = SymbolicDomain.as_expr(accessed_domains[temporary.id])

    return itir.Program(
        id=program.id,
        function_definitions=program.function_definitions,
        params=program.params,
        declarations=new_declarations,
        body=transformed_set_ats,
    )
