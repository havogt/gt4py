from typing import Any

from eve import codegen
from eve.codegen import FormatTemplate as as_fmt
from eve.codegen import MakoTemplate as as_mako
from functional.iterator.backends import backend
from functional.iterator.embedded import (
    NeighborTableOffsetProviderBase,  # TODO must not import embedded
)
from functional.iterator.ir import (
    FencilDefinition,
    FunCall,
    OffsetLiteral,
    Program,
    StencilClosure,
    SymRef,
)
from functional.iterator.transforms import apply_common_transforms


class GTFn(codegen.TemplatedGenerator):
    Sym = as_fmt("{id}")
    SymRef = as_fmt("{id}")
    IntLiteral = as_fmt("{value}")
    FloatLiteral = as_fmt("{value}")
    AxisLiteral = as_fmt("{value}")

    def visit_OffsetLiteral(self, node: OffsetLiteral, **kwargs):
        return node.value if isinstance(node.value, str) else str(node.value)

    StringLiteral = as_fmt("{value}")

    def visit_FunCall(self, node: FunCall, *, pargs=None, **kwargs):
        template_builtins = ["shift", "lift", "reduce"]
        kwargs["oparen"] = "("
        kwargs["cparen"] = ")"
        if isinstance(node.fun, SymRef) and node.fun.id in template_builtins:
            kwargs["oparen"] = "<"
            kwargs["cparen"] = ">"
        pargs = self.visit(node.args)
        if isinstance(node.fun, SymRef) and node.fun.id == "reduce":
            # init arg needs to be wrapped
            pargs[1] = f"[](auto&& ...){{return {pargs[1]};}}"
        return self.generic_visit(node, pargs=pargs, **kwargs)

    FunCall = as_fmt("{fun}{oparen}{','.join(pargs)}{cparen}")
    Lambda = as_mako(
        "[=](${','.join('auto ' + p for p in params)}){return ${expr};}"
    )  # TODO capture
    # StencilClosure = as_mako(
    #     "closure(${domain}, ${stencil}, out(${','.join(outputs)}), ${','.join(inputs)})"
    # )
    StencilClosure = as_mako(
        "make_stage<${stencil}, std::identity{}, ${','.join(str(i) for i in range(len(outputs)+len(inputs)))}>"
    )
    FunctionDefinition = as_mako(
        """
    inline constexpr auto ${id} = [](${','.join('auto ' + p for p in params)}){
        return ${expr};
        };
    """
    )

    def visit_FencilDefinition(self, node: FencilDefinition, *, offset_provider, **kwargs):
        neigh_offset_provider = {
            k: v
            for k, v in offset_provider.items()
            if isinstance(v, NeighborTableOffsetProviderBase)
        }
        neigh_tbls = [f"auto && {k}_tbl" for k in neigh_offset_provider.keys()]
        neigh_tbl_assigns = [f"{k}_tag::ptr() = {k}_tbl;" for k in neigh_offset_provider.keys()]
        tags = []
        for i, e in enumerate(neigh_offset_provider.items()):
            o, p = e
            tags.append(
                f"using {o}_tag = tag<gridtools::integral_constant<int, {i}>, {p.max_neighbors}>;"
            )
            tags.append(f"constexpr auto {o} = {o}_tag{{}};")

        # TODO this is hacky and we should change either the Python IR or the C++ implementation
        assert len(node.closures) == 1
        closure: StencilClosure = node.closures[0]

        def get_length(dim):
            return f"{self.visit(dim[2])}-{self.visit(dim[1])}"

        def get_offset(dim):
            return f"{self.visit(dim[1])}"

        if len(neigh_offset_provider) > 0:  # unstructured
            named_ranges_args = [named_range.args for named_range in closure.domain.args]
            assert len(named_ranges_args) == 2
            assert named_ranges_args[0][1].value == 0  # IntLiteral(value=0)
            assert named_ranges_args[1][1].value == 0  # IntLiteral(value=0)
            horizontal = self.visit(named_ranges_args[0][2])
            domain = f"unstructured(std::tuple{{{horizontal},{get_length(named_ranges_args[1])}}})"
        else:  # Cartesian
            named_ranges_args = [named_range.args for named_range in closure.domain.args]
            assert len(named_ranges_args) == 3

            domain = f"cartesian(std::tuple({','.join(get_length(arg) for arg in named_ranges_args)}), std::tuple({','.join(get_offset(arg) for arg in named_ranges_args)}))"

        params_str = [self.visit(p) for p in [*closure.outputs, *closure.inputs]]

        return self.generic_visit(
            node,
            neigh_tbls=neigh_tbls,
            neigh_tbl_assigns=neigh_tbl_assigns,
            tags=tags,
            offset_provider=offset_provider,
            domain_str=domain,
            params_str=params_str,
            **kwargs,
        )

    FencilDefinition = as_mako(
        """
        ${'\\n'.join(tags)}
        constexpr auto fn_main = [](${','.join(neigh_tbls)}) {
        ${'\\n'.join(neigh_tbl_assigns)}
        
        return [fen=fencil<naive,${','.join(closures)}>](${','.join('auto&& ' + p for p in params)}){
            auto domain = ${domain_str};
            fen(domain, ${','.join(p for p in params_str)} );
        };
    };
    """
    )

    def visit_Program(self, node: Program, **kwargs):
        if len(node.fencil_definitions) != 1:
            raise NotImplementedError("Exactly one fencil expected.")
        return self.generic_visit(node, **kwargs)

    Program = as_mako(
        """
        // TODO move to a separate header
        using namespace gridtools;
        using namespace literals;
        using namespace fn;

        namespace conn_impl {
            template <class id, int max_neighs>
            struct tag {
                static const std::array<int, max_neighs> *&ptr() {
                    static thread_local const std::array<int, max_neighs> *ptr;
                    return ptr;
                }

                template <class Offset>
                friend const std::array<int, max_neighs> &fn_get_neighbour_offsets(tag const &, Offset offset) {
                    return tag<id, max_neighs>::ptr()[offset];
                }
            };
        } // namespace conn_impl

        template <class id, int max_neighs>
        using tag = conn_impl::tag<id, max_neighs>;

        ${''.join(function_definitions)} ${''.join(fencil_definitions)}
        """
    )

    @classmethod
    def apply(cls, root, **kwargs: Any) -> str:
        transformed = apply_common_transforms(
            root,
            use_tmps=kwargs.get("use_tmps", False),
            offset_provider=kwargs.get("offset_provider", None),
        )
        generated_code = super().apply(transformed, **kwargs)
        formatted_code = codegen.format_source("cpp", generated_code, style="LLVM")
        return formatted_code


def gen(prog, *arg, **kwargs):
    assert "file" in kwargs
    with open(kwargs["file"], "w") as f:
        f.write(GTFn.apply(prog, **kwargs))


backend.register_backend("gtfn", gen)
# backend.register_backend("gtfn", lambda prog, *args, **kwargs: print(GTFn.apply(prog, **kwargs)))
