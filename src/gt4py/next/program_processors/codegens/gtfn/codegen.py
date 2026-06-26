# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import Any, Collection, Final, Union

from gt4py.eve import codegen
from gt4py.eve.codegen import FormatTemplate as as_fmt, MakoTemplate as as_mako
from gt4py.next import common
from gt4py.next.otf import cpp_utils
from gt4py.next.program_processors.codegens.gtfn import gtfn_im_ir, gtfn_ir, gtfn_ir_common


class GTFNCodegen(codegen.TemplatedGenerator):
    _grid_type_str: Final = {
        common.GridType.CARTESIAN: "cartesian",
        common.GridType.UNSTRUCTURED: "unstructured",
    }

    _builtins_mapping: Final = {
        "abs": "std::abs",
        "neg": "std::negate<>{}",
        "sin": "std::sin",
        "cos": "std::cos",
        "tan": "std::tan",
        "arcsin": "std::asin",
        "arccos": "std::acos",
        "arctan": "std::atan",
        "sinh": "std::sinh",
        "cosh": "std::cosh",
        "tanh": "std::tanh",
        "arcsinh": "std::asinh",
        "arccosh": "std::acosh",
        "arctanh": "std::atanh",
        "sqrt": "std::sqrt",
        "exp": "std::exp",
        "log": "std::log",
        "gamma": "std::tgamma",
        "cbrt": "std::cbrt",
        "isfinite": "std::isfinite",
        "isinf": "std::isinf",
        "isnan": "std::isnan",
        "floor": "std::floor",
        "ceil": "std::ceil",
        "trunc": "std::trunc",
        "minimum": "std::min",
        "maximum": "std::max",
        "fmod": "std::fmod",
        "power": "std::pow",
        "float32": "float",
        "float64": "double",
        "int8": "std::int8_t",
        "uint8": "std::uint8_t",
        "int16": "std::int16_t",
        "uint16": "std::uint16_t",
        "int32": "std::int32_t",
        "uint32": "std::uint32_t",
        "int64": "std::int64_t",
        "uint64": "std::uint64_t",
        "bool": "bool",
        "plus": "std::plus<>{}",
        "minus": "std::minus<>{}",
        "multiplies": "std::multiplies<>{}",
        "divides": "std::divides<>{}",
        "eq": "std::equal_to<>{}",
        "not_eq": "std::not_equal_to<>{}",
        "less": "std::less<>{}",
        "less_equal": "std::less_equal<>{}",
        "greater": "std::greater<>{}",
        "greater_equal": "std::greater_equal<>{}",
        "and_": "std::logical_and<>{}",
        "or_": "std::logical_or<>{}",
        "xor_": "std::bit_xor<>{}",
        "mod": "std::modulus<>{}",
        "not_": "std::logical_not<>{}",
    }

    Sym = as_fmt("{id}")

    def visit_SymRef(self, node: gtfn_ir_common.SymRef, **kwargs: Any) -> str:
        if node.id == "get":
            return "::gridtools::tuple_util::get"
        if node.id in self._builtins_mapping:
            return self._builtins_mapping[node.id]
        if node.id in gtfn_ir.GTFN_BUILTINS:
            qualified_fun_name = f"gtfn::{node.id}"
            return qualified_fun_name

        return node.id

    def visit_Literal(self, node: gtfn_ir.Literal, **kwargs: Any) -> str:
        # TODO(tehrengruber): isn't this wrong and int32 should be casted to an actual int32?
        match node.type:
            case "float32" | "float64":
                cpp_value = node.value

                if node.value in ["inf", "-inf"]:
                    sign = "-" if node.value == "-inf" else ""
                    cpp_value = (
                        f"std::numeric_limits<{cpp_utils.pytype_to_cpptype(node.type)}>::infinity()"
                    )
                    return f"{sign}{cpp_value}"
                elif node.value == "nan":
                    cpp_value = f"std::numeric_limits<{cpp_utils.pytype_to_cpptype(node.type)}>::signaling_NaN()"
                    return cpp_value

                if not any(c in cpp_value for c in ".eE"):
                    cpp_value = f"{cpp_value}."

                if node.type == "float32":
                    cpp_value = f"{cpp_value}f"
                return cpp_value
            case "bool":
                return node.value.lower()
            case "axis_literal":
                return node.value
            case _:
                # TODO(tehrengruber): we should probably shouldn't just allow anything here. Revisit.
                return node.value

    IntegralConstant = as_fmt("{value}_c")

    UnaryExpr = as_fmt("{op}({expr})")
    # add an extra space between the operators is needed such that `minus(1, -1)` does not get
    # translated into `1--1`, but `1 - -1`
    BinaryExpr = as_fmt("({lhs} {op} {rhs})")
    TernaryExpr = as_fmt("({cond}?{true_expr}:{false_expr})")
    CastExpr = as_fmt("static_cast<{new_dtype}>({obj_expr})")

    def visit_TaggedValues(self, node: gtfn_ir.TaggedValues, **kwargs: Any) -> str:
        tags = self.visit(node.tags)
        values = self.visit(node.values)
        if self.is_cartesian:
            return f"::gridtools::hymap::keys<{','.join(t + '_t' for t in tags)}>::make_values({','.join(values)})"
        else:
            return f"::gridtools::tuple({','.join(values)})"

    CartesianDomain = as_fmt("gtfn::cartesian_domain({tagged_sizes}, {tagged_offsets})")
    UnstructuredDomain = as_mako(
        "gtfn::unstructured_domain(${tagged_sizes}, ${tagged_offsets}, connectivities__...)"
    )

    def visit_OffsetLiteral(self, node: gtfn_ir.OffsetLiteral, **kwargs: Any) -> str:
        return node.value if isinstance(node.value, str) else f"{node.value}_c"

    SidComposite = as_mako(
        "::gridtools::sid::composite::keys<${','.join(f'::gridtools::integral_constant<int,{i}>' for i in range(len(values)))}>::make_values(${','.join(values)})"
    )

    SidFromScalar = as_fmt("gridtools::stencil::global_parameter({arg})")

    def is_functor_call(self, node: gtfn_ir.FunCall) -> bool:
        return (
            isinstance(node.fun, gtfn_ir_common.SymRef)
            and node.fun.id in self.user_defined_function_ids
        )

    def visit_FunCall(self, node: gtfn_ir.FunCall, **kwargs: Any) -> str:
        # functions are represented as function objects that need to be instantiated
        instantiate = "{}()" if self.is_functor_call(node) else ""
        return self.generic_visit(node, instantiate=instantiate)

    FunCall = as_fmt("{fun}{instantiate}({','.join(args)})")

    Lambda = as_mako(
        "[=](${','.join('auto ' + p for p in params)}){return ${expr};}"
    )  # TODO capture

    Backend = as_mako(
        "make_backend(${'backend' if _this_node.loop_blocked else 'backend_nlb'}, ${domain})"
    )

    StencilExecution = as_mako(
        """
        ${backend}.stencil_executor()().arg(${output})${''.join('.arg(' + i + ')' for i in inputs)}.assign(0_c, ${stencil}() ${',' if inputs else ''} ${','.join(str(i) + '_c' for i in range(1, len(inputs) + 1))}).execute();
        """
    )

    Scan = as_fmt(
        "assign({output}_c, {function}(), {', '.join([init] + [input + '_c' for input in inputs])})"
    )

    def visit_ScanTailDefinition(self, node: gtfn_ir.ScanTailDefinition, **kwargs: Any) -> str:
        res = self.visit(node.res, **kwargs)
        acc = self.visit(node.acc, **kwargs)
        surface = self.visit(node.surface, **kwargs)
        # Bind each lifted SID input param to an iterator at the current level (its raw arg index
        # + the backend offset AO); the rewritten output exprs deref these as usual.
        input_binds = "".join(
            f"auto const& {self.visit(p, **kwargs)} = make_iterator("
            f"::gridtools::integral_constant<int, {idx} + AO>{{}}, ptr, strides);"
            for p, idx in node.input_params
        )
        writes = "".join(
            f"*::gridtools::host_device::at_key<::gridtools::integral_constant<int, {out} + AO>>(ptr) = "
            f"{self.visit(expr, **kwargs)};"
            for out, expr in node.outputs
        )
        return (
            f"template <int AO> struct {node.id} {{\n"
            f"  template <class Res, class Acc, class Surface, class Mk, class Ptr, class Strides>\n"
            f"  GT_FUNCTION void operator()(Res const& {res}, Acc const& {acc}, Surface const& {surface},\n"
            f"      Mk&& make_iterator, Ptr const& ptr, Strides const& strides) const {{\n"
            f"    {input_binds}\n"
            f"    {writes}\n"
            f"  }}\n"
            f"}};"
        )

    def visit_MapColumnTailDefinition(
        self, node: gtfn_ir.MapColumnTailDefinition, **kwargs: Any
    ) -> str:
        cur = self.visit(node.cur, **kwargs)
        prev = self.visit(node.prev, **kwargs)
        input_binds = "".join(
            f"auto const& {self.visit(p, **kwargs)} = make_iterator("
            f"::gridtools::integral_constant<int, {idx} + AO>{{}}, ptr, strides);"
            for p, idx in node.input_params
        )
        writes = "".join(
            f"*::gridtools::host_device::at_key<::gridtools::integral_constant<int, {out} + AO>>(ptr) = "
            f"{self.visit(expr, **kwargs)};"
            for out, expr in node.outputs
        )
        return (
            f"template <int AO> struct {node.id} {{\n"
            f"  template <class Cur, class Prev, class Mk, class Ptr, class Strides>\n"
            f"  GT_FUNCTION void operator()(Cur const& {cur}, Prev const& {prev},\n"
            f"      Mk&& make_iterator, Ptr const& ptr, Strides const& strides) const {{\n"
            f"    {input_binds}\n"
            f"    {writes}\n"
            f"  }}\n"
            f"}};"
        )

    def visit_MapColumnExecution(self, node: gtfn_ir.MapColumnExecution, **kwargs: Any) -> str:
        backend = self.visit(node.backend, **kwargs)
        axis = self.visit(node.axis, **kwargs)
        args = "".join(f".arg({self.visit(a, **kwargs)})" for a in node.args)
        producer = self.visit(node.producer, **kwargs)
        consumer = self.visit(node.consumer, **kwargs)
        pins = "".join(f", {i}" for i in node.producer_inputs)
        raw = (
            f"gtfn::map_column_stage_raw<{producer}, {node.producer_output}, {consumer}{pins}>{{}}"
        )
        return (
            f"{backend}.vertical_executor({axis})(){args}"
            f".assign_map_column({raw}).execute();"
        )

    def visit_KoffWindowTailDefinition(
        self, node: gtfn_ir.KoffWindowTailDefinition, **kwargs: Any
    ) -> str:
        # The consumer receives make_iterator + ptr + strides, then all cur registers, then all
        # prev registers (matching koff_window_column_stage::invoke's argument order).
        cur_params = "".join(
            f", Cur{i} const& {self.visit(cur, **kwargs)}" for i, (cur, _prev) in enumerate(node.windows)
        )
        prev_params = "".join(
            f", Prev{i} const& {self.visit(prev, **kwargs)}" for i, (_cur, prev) in enumerate(node.windows)
        )
        win_params = cur_params + prev_params
        win_templates = "".join(f", class Cur{i}" for i in range(len(node.windows))) + "".join(
            f", class Prev{i}" for i in range(len(node.windows))
        )
        input_binds = "".join(
            f"auto const& {self.visit(p, **kwargs)} = make_iterator("
            f"::gridtools::integral_constant<int, {idx} + AO>{{}}, ptr, strides);"
            for p, idx in node.input_params
        )
        writes = "".join(
            f"*::gridtools::host_device::at_key<::gridtools::integral_constant<int, {out} + AO>>(ptr) = "
            f"{self.visit(expr, **kwargs)};"
            for out, expr in node.outputs
        )
        return (
            f"template <int AO> struct {node.id} {{\n"
            f"  template <class Mk, class Ptr, class Strides{win_templates}>\n"
            f"  GT_FUNCTION void operator()(Mk&& make_iterator, Ptr const& ptr,\n"
            f"      Strides const& strides{win_params}) const {{\n"
            f"    {input_binds}\n"
            f"    {writes}\n"
            f"  }}\n"
            f"}};"
        )

    def visit_KoffWindowExecution(self, node: gtfn_ir.KoffWindowExecution, **kwargs: Any) -> str:
        backend = self.visit(node.backend, **kwargs)
        axis = self.visit(node.axis, **kwargs)
        args = "".join(f".arg({self.visit(a, **kwargs)})" for a in node.args)
        consumer = self.visit(node.consumer, **kwargs)
        wins = "".join(f", {i}" for i in node.window_inputs)
        raw = f"gtfn::koff_window_column_stage_raw<{consumer}{wins}>{{}}"
        return (
            f"{backend}.vertical_executor({axis})(){args}"
            f".assign_koff_window({raw}).execute();"
        )

    def visit_MapWindowTailDefinition(
        self, node: gtfn_ir.MapWindowTailDefinition, **kwargs: Any
    ) -> str:
        # operator()(prod_cur, prod_prev, win0_cur..winN_cur, win0_prev..winN_prev,
        #            make_iterator, ptr, strides) — matching map_window_column_stage::invoke.
        cur = self.visit(node.cur, **kwargs)
        prev = self.visit(node.prev, **kwargs)
        win_cur = "".join(
            f", WCur{i} const& {self.visit(c, **kwargs)}" for i, (c, _p) in enumerate(node.windows)
        )
        win_prev = "".join(
            f", WPrev{i} const& {self.visit(p, **kwargs)}" for i, (_c, p) in enumerate(node.windows)
        )
        win_templates = "".join(f", class WCur{i}" for i in range(len(node.windows))) + "".join(
            f", class WPrev{i}" for i in range(len(node.windows))
        )
        input_binds = "".join(
            f"auto const& {self.visit(p, **kwargs)} = make_iterator("
            f"::gridtools::integral_constant<int, {idx} + AO>{{}}, ptr, strides);"
            for p, idx in node.input_params
        )
        writes = "".join(
            f"*::gridtools::host_device::at_key<::gridtools::integral_constant<int, {out} + AO>>(ptr) = "
            f"{self.visit(expr, **kwargs)};"
            for out, expr in node.outputs
        )
        return (
            f"template <int AO> struct {node.id} {{\n"
            f"  template <class Cur, class Prev, class Mk, class Ptr, class Strides{win_templates}>\n"
            f"  GT_FUNCTION void operator()(Cur const& {cur}, Prev const& {prev}{win_cur}{win_prev},\n"
            f"      Mk&& make_iterator, Ptr const& ptr, Strides const& strides) const {{\n"
            f"    {input_binds}\n"
            f"    {writes}\n"
            f"  }}\n"
            f"}};"
        )

    def visit_MapWindowExecution(self, node: gtfn_ir.MapWindowExecution, **kwargs: Any) -> str:
        backend = self.visit(node.backend, **kwargs)
        axis = self.visit(node.axis, **kwargs)
        args = "".join(f".arg({self.visit(a, **kwargs)})" for a in node.args)
        producer = self.visit(node.producer, **kwargs)
        consumer = self.visit(node.consumer, **kwargs)
        nwin = len(node.window_inputs)
        win_then_pins = "".join(f", {i}" for i in node.window_inputs) + "".join(
            f", {i}" for i in node.producer_inputs
        )
        raw = (
            f"gtfn::map_window_column_stage_raw<{producer}, {node.producer_output}, "
            f"{consumer}, {nwin}{win_then_pins}>{{}}"
        )
        return (
            f"{backend}.vertical_executor({axis})(){args}"
            f".assign_map_window({raw}).execute();"
        )

    def visit_ScanExecution(self, node: gtfn_ir.ScanExecution, **kwargs: Any) -> str:
        backend = self.visit(node.backend, **kwargs)
        axis = self.visit(node.axis, **kwargs)
        args = "".join(f".arg({self.visit(a, **kwargs)})" for a in node.args)
        def scan_with_tail_raw(s: gtfn_ir.Scan) -> str:
            # scan_with_tail_raw carries the scan struct, the Tail struct (template on AO), the body
            # and tail vertical trims, and the scan's *body* input indices (the Ins). The tail's own
            # inputs/outputs are baked into the Tail struct.
            t = s.tail
            ins = "".join(f", {i}" for i in s.inputs)
            return (
                f"gtfn::scan_with_tail_raw<{self.visit(s.function, **kwargs)}, "
                f"{self.visit(t.definition, **kwargs)}, "
                f"{t.body_top_trim}, {t.body_bot_trim}, {t.tail_top_trim}, {t.tail_bot_trim}{ins}>{{}}"
            )

        if not node.merged_kernel and len(node.scans) == 1 and node.scans[0].tail is not None:
            # Folded post-scan consumer of a standalone backward scan.
            s = node.scans[0]
            return (
                f"{backend}.vertical_executor({axis})(){args}"
                f".assign_scan_with_tail({scan_with_tail_raw(s)}, {self.visit(s.init, **kwargs)}).execute();"
            )
        if node.merged_kernel:
            # one fused kernel: each scan struct (e.g. _scan_0, carrying its fwd/bwd base and body)
            # is the ScanOrFold of a scan_substage_raw; inits become the per-stage seeds. A substage
            # with a folded post-scan consumer is emitted as a scan_with_tail_raw instead.
            seeds = ", ".join(self.visit(s.init, **kwargs) for s in node.scans)
            substages = ", ".join(
                scan_with_tail_raw(s)
                if s.tail is not None
                else "gtfn::scan_substage_raw<{fn}, {tt}, {bt}, {out}{ins}>{{}}".format(
                    fn=self.visit(s.function, **kwargs),
                    tt=s.top_trim,
                    bt=s.bot_trim,
                    out=s.output,
                    ins="".join(f", {i}" for i in s.inputs),
                )
                for s in node.scans
            )
            return (
                f"{backend}.vertical_executor({axis})(){args}"
                f".assign_merged_scans(gtfn::make_tuple({seeds}), {substages}).execute();"
            )
        scans = ".".join(self.visit(s, **kwargs) for s in node.scans)
        return f"{backend}.vertical_executor({axis})(){args}.{scans}.execute();"

    IfStmt = as_mako(
        """
          if (${cond}) {
            ${'\\n'.join(true_branch)}
          } else {
            ${'\\n'.join(false_branch)}
          }
        """
    )

    ScanPassDefinition = as_mako(
        """
        struct ${id} : ${'gtfn::fwd' if _this_node.forward else 'gtfn::bwd'} {
            static constexpr GT_FUNCTION auto body() {
                return gtfn::scan_pass([](${','.join('auto const& ' + p for p in params)}) {
                    return ${expr};
                }, ${projector if _this_node.projector is not None else '::gridtools::host_device::identity()'});
            }
        };
        """
    )

    FunctionDefinition = as_mako(
        """
        struct ${id} {
            constexpr auto operator()() const {
                return [](${','.join('auto const& ' + p for p in params)}){
                    return ${expr};
                };
            }
        };
    """
    )

    TagDefinition = as_mako(
        """
        %if _this_node.alias:
            %if isinstance(_this_node.alias, str):
                using ${name}_t = ${alias};
            %else:
                using ${name}_t = ${alias}_t;
            %endif
        %else:
            struct ${name}_t{};
        %endif
        constexpr inline ${name}_t ${name}{};
        """
    )

    def visit_TemporaryAllocation(self, node: gtfn_ir.TemporaryAllocation, **kwargs: Any) -> str:
        assert isinstance(node.domain, (gtfn_ir.CartesianDomain, gtfn_ir.UnstructuredDomain))
        assert node.domain.tagged_offsets.tags == node.domain.tagged_sizes.tags
        tags = node.domain.tagged_offsets.tags

        origins = [
            gtfn_ir.UnaryExpr(op="-", expr=offset) for offset in node.domain.tagged_offsets.values
        ]

        return self.generic_visit(
            node,
            tmp_sizes=self.visit(node.domain.tagged_sizes, **kwargs),
            shifts=self.visit(gtfn_ir.TaggedValues(tags=tags, values=origins), **kwargs),
            **kwargs,
        )

    TemporaryAllocation = as_fmt(
        "auto {id} = gridtools::sid::shift_sid_origin(gtfn::allocate_global_tmp<{dtype}>(tmp_alloc__, {tmp_sizes}), {shifts});"
    )

    def _shmem_staged_override(self, node: gtfn_ir.Program, **kwargs: Any) -> Union[dict, None]:
        # P2 shmem-staging: when GT4PY_FN_SHMEM_STAGING is set and the program is exactly the
        # single-temp two-unstructured-stage "C2E-reduce producer -> E2C-read consumer" pattern,
        # emit ONE fused kernel that stages the cell temp in __shared__ (gtfn::shmem_staged_
        # unstructured) instead of allocate_global_tmp + two executes. Default OFF => path unchanged.
        if not os.environ.get("GT4PY_FN_SHMEM_STAGING"):
            return None
        if os.environ.get("SHMEM_DIAG") and node.grid_type == common.GridType.UNSTRUCTURED:
            import pathlib
            lines=[f"=== {node.id}: {len(node.temporaries)} temps, {len(node.executions)} execs ==="]
            for t in node.temporaries:
                lines.append(f"  TEMP {t.id} dtype={t.dtype}")
            for i,e in enumerate(node.executions):
                if isinstance(e, gtfn_ir.StencilExecution):
                    outs=str(e.output.id) if isinstance(e.output, gtfn_ir_common.SymRef) else type(e.output).__name__
                    ins=",".join(str(x.id) if isinstance(x, gtfn_ir_common.SymRef) else type(x).__name__ for x in e.inputs)
                    lines.append(f"  EXEC{i} stencil={e.stencil.id} OUT={outs} IN=[{ins}]")
                else:
                    lines.append(f"  EXEC{i} {type(e).__name__}")
            pathlib.Path(os.environ["SHMEM_DIAG"]).write_text(chr(10).join(lines))
            # SUBPAIR_FIND: generalized stageable-pair finder (the part replacing the whole-program match)
            execs=[e for e in node.executions if isinstance(e, gtfn_ir.StencilExecution)]
            cand=[]
            for tmp in node.temporaries:
                tid=str(tmp.id)
                prods=[e for e in execs if isinstance(e.output, gtfn_ir_common.SymRef) and str(e.output.id)==tid]
                cons=[e for e in execs if any(isinstance(i, gtfn_ir_common.SymRef) and str(i.id)==tid for i in e.inputs)]
                if len(prods)==1 and len(cons)==1 and prods[0] is not cons[0]:
                    pin=[str(i.id) if isinstance(i,gtfn_ir_common.SymRef) else type(i).__name__ for i in prods[0].inputs]
                    cand.append(f"  STAGEABLE {tid}: producer={prods[0].stencil.id}(in={pin}) consumer={cons[0].stencil.id} (consumer total inputs={len(cons[0].inputs)})")
            pathlib.Path(os.environ["SHMEM_DIAG"]).write_text(chr(10).join(lines+["--- sub-pair finder candidates ---"]+cand))
        if node.grid_type != common.GridType.UNSTRUCTURED:
            return None
        if len(node.temporaries) != 1 or len(node.executions) != 1 + 1:
            return None
        prod, cons = node.executions
        if not (
            isinstance(prod, gtfn_ir.StencilExecution)
            and isinstance(cons, gtfn_ir.StencilExecution)
        ):
            return None
        tmp = node.temporaries[0]
        tmp_id = str(tmp.id)
        # producer must write the temp from non-temp inputs; consumer must read ONLY the temp.
        if not (isinstance(prod.output, gtfn_ir_common.SymRef) and str(prod.output.id) == tmp_id):
            return None
        if not all(isinstance(i, gtfn_ir_common.SymRef) for i in prod.inputs):
            return None
        if any(str(i.id) == tmp_id for i in prod.inputs):
            return None
        if not (
            len(cons.inputs) == 1
            and isinstance(cons.inputs[0], gtfn_ir_common.SymRef)
            and str(cons.inputs[0].id) == tmp_id
        ):
            return None
        if not isinstance(cons.output, gtfn_ir_common.SymRef):
            return None

        cell_domain = self.visit(tmp.domain, **kwargs)
        edge_domain = self.visit(cons.backend.domain, **kwargs)
        out = self.visit(cons.output, **kwargs)
        prod_ins = ", ".join(self.visit(i, **kwargs) for i in prod.inputs)
        call = (
            f"gtfn::shmem_staged_unstructured<{prod.stencil.id}, {cons.stencil.id}, {tmp.dtype}>("
            f"backend_nlb, {cell_domain}, {edge_domain}, {out}"
            f"{', ' + prod_ins if prod_ins else ''});"
        )
        return {
            "temporaries": [],
            "executions": [call],
            "extra_includes": ["#include <gridtools/fn/shmem_staged_unstructured.hpp>"],
        }

    def visit_Program(self, node: gtfn_ir.Program, **kwargs: Any) -> Union[str, Collection[str]]:
        self.is_cartesian = node.grid_type == common.GridType.CARTESIAN
        self.user_defined_function_ids = list(
            str(fundef.id) for fundef in node.function_definitions
        )
        override = self._shmem_staged_override(node, **kwargs) or {}
        return self.generic_visit(
            node,
            grid_type_str=self._grid_type_str[node.grid_type],
            block_sizes=self._block_sizes(
                node.offset_definitions,
                thread_block_sizes=kwargs.get("thread_block_sizes"),
                loop_block_sizes=kwargs.get("loop_block_sizes"),
            ),
            extra_includes=override.get("extra_includes", []),
            **{k: v for k, v in override.items() if k != "extra_includes"},
            **kwargs,
        )

    Program = as_mako(
        """
    #include <cmath>
    #include <cstdint>
    #include <functional>
    #include <gridtools/fn/${grid_type_str}.hpp>
    #include <gridtools/fn/sid_neighbor_table.hpp>
    #include <gridtools/stencil/global_parameter.hpp>
    ${'\\n'.join(extra_includes)}

    // TODO(tehrengruber): This should disappear as soon as we introduce a proper builtin.
    namespace gridtools::fn {
        """
        # TODO(tehrengruber): The return type should be
        #  `typename gridtools::sid::lower_bounds_type<S>, typename gridtools::sid::upper_bounds_type<S>`,
        #  but fails as type used for index calculations in gtfn differs
        """
        template <class S, class D>
        GT_FUNCTION gridtools::tuple<int, int> get_domain_range(S &&sid, D) {
            return {gridtools::host_device::at_key<D>(gridtools::sid::get_lower_bounds(sid)),
                gridtools::host_device::at_key<D>(gridtools::sid::get_upper_bounds(sid))};
        }
    }
    
    namespace generated{

    namespace gtfn = ::gridtools::fn;

    namespace{
    using namespace ::gridtools::literals;

    ${'\\n'.join(offset_definitions)}
    ${'\\n'.join(function_definitions)}

    ${block_sizes}

    inline auto ${id} = [](auto... connectivities__){
        return [connectivities__...](auto backend, auto backend_nlb, ${','.join('auto&& ' + p for p in params)}){
            auto tmp_alloc__ = gtfn::backend::tmp_allocator(backend);
            ${'\\n'.join(temporaries)}
            ${'\\n'.join(executions)}
        };
    };
    }
    }
    """
    )

    def _block_sizes(
        self,
        offset_definitions: list[gtfn_ir.TagDefinition],
        thread_block_sizes: tuple[int, int] | None = None,
        loop_block_sizes: tuple[int, int] | None = None,
    ) -> str:
        if self.is_cartesian:
            block_dims = []
            block_sizes = [32, 8] + [1] * (len(offset_definitions) - 2)
            for i, tag in enumerate(offset_definitions):
                if tag.alias is None:
                    block_dims.append(
                        f"gridtools::meta::list<{tag.name.id}_t, "
                        f"gridtools::integral_constant<int, {block_sizes[i]}>>"
                    )
            sizes_str = ",\n".join(block_dims)
            return (
                f"using block_sizes_t = gridtools::meta::list<{sizes_str}>;\n"
                "using loop_block_sizes_t = gridtools::meta::list<>;\n"
                "using loop_block_sizes_none_t = gridtools::meta::list<>;"
            )
        else:
            # Unstructured GPU thread-block shape and per-thread loop-block (K-coarsening) shape.
            th, tv = thread_block_sizes if thread_block_sizes is not None else (32, 8)
            lh, lv = loop_block_sizes if loop_block_sizes is not None else (1, 1)
            return (
                "using block_sizes_t = gridtools::meta::list<"
                f"gridtools::meta::list<gtfn::unstructured::dim::horizontal, gridtools::integral_constant<int, {th}>>, "
                f"gridtools::meta::list<gtfn::unstructured::dim::vertical, gridtools::integral_constant<int, {tv}>>>;\n"
                "using loop_block_sizes_t = gridtools::meta::list<"
                f"gridtools::meta::list<gtfn::unstructured::dim::horizontal, gridtools::integral_constant<int, {lh}>>, "
                f"gridtools::meta::list<gtfn::unstructured::dim::vertical, gridtools::integral_constant<int, {lv}>>>;\n"
                "using loop_block_sizes_none_t = gridtools::meta::list<"
                "gridtools::meta::list<gtfn::unstructured::dim::horizontal, gridtools::integral_constant<int, 1>>, "
                "gridtools::meta::list<gtfn::unstructured::dim::vertical, gridtools::integral_constant<int, 1>>>;"
            )

    @classmethod
    def apply(cls, root: Any, **kwargs: Any) -> str:
        generated_code = super().apply(root, **kwargs)
        return generated_code


class GTFNIMCodegen(GTFNCodegen):
    Stmt = as_fmt("{lhs} {op} {rhs};")

    InitStmt = as_fmt("{init_type} {lhs} {op} {rhs};")

    EmptyListInitializer = as_mako("{}")

    Conditional = as_mako(
        """
          using ${cond_type} = typename std::common_type<decltype(${if_rhs_}), decltype(${else_rhs_})>::type;
          ${init_stmt}
          if (${cond}) {
            ${if_stmt}
          } else {
            ${else_stmt}
          }
    """
    )

    ImperativeFunctionDefinition = as_mako(
        """
        struct ${id} {
            constexpr auto operator()() const {
                return [](${','.join('auto const& ' + p for p in params)}){
                    ${expr_};
                };
            }
        };
    """
    )

    ReturnStmt = as_fmt("return {ret};")

    def visit_Conditional(self, node: gtfn_im_ir.Conditional, **kwargs: Any) -> str:
        if_rhs_ = self.visit(node.if_stmt.rhs)
        else_rhs_ = self.visit(node.else_stmt.rhs)
        return self.generic_visit(node, if_rhs_=if_rhs_, else_rhs_=else_rhs_)

    def visit_ImperativeFunctionDefinition(
        self, node: gtfn_im_ir.ImperativeFunctionDefinition, **kwargs: Any
    ) -> str:
        expr_ = "".join(self.visit(stmt) for stmt in node.fun)
        return self.generic_visit(node, expr_=expr_)

    @classmethod
    def apply(cls, root: Any, **kwargs: Any) -> str:
        generated_code = super().apply(root, **kwargs)
        return generated_code
