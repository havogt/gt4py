# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Stage a transient between two GPU kernels in shared memory.

Opt-in, not part of `gt_auto_optimize()`; meant for the `GT4PyAutoOptHook.AfterToGPU` hook.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import functools
from typing import Any, Callable, Optional, Sequence

import dace
from dace import subsets as dace_subsets, symbolic as dace_symbolic
from dace.sdfg import nodes as dace_nodes


_PREFIX = "__gt_smem"


@dataclasses.dataclass(frozen=True)
class _Staged:
    """One staged transient `T` and its shared-memory tile `S`, indexed `(I, J, X, buffer)`."""

    data: str
    tile: str
    origin_i: Any
    origin_j: Any


def _range_bounds(rng: tuple[Any, Any, Any]) -> tuple[Any, Any]:
    if rng[2] != 1:
        raise NotImplementedError("Strided map ranges are not supported.")
    return rng[0], rng[1]


def _split_params(params: Sequence[str]) -> tuple[str, str, str, str]:
    if len(params) != 4 or not params[0].endswith("_vertical"):
        raise NotImplementedError(f"Expected map parameters ordered (K, X, J, I), got {params}.")
    return params[0], params[1], params[2], params[3]


def _write_origin(state: dace.SDFGState, producer_entry: dace_nodes.MapEntry, data: str) -> tuple:
    """Constant offsets `(oI, oJ)` of the producer's write `T[i_I + oI, i_J + oJ, x, i_K]`."""
    k, _, j, i = _split_params(producer_entry.map.params)
    writes = [
        edge for edge in state.in_edges(state.exit_node(producer_entry)) if edge.data.data == data
    ]
    if len(writes) != 1:
        raise NotImplementedError(f"Expected one write of '{data}' in the producer.")
    subset = writes[0].data.subset
    sym = dace_symbolic.symbol
    origins = []
    for dim, param in ((0, i), (1, j)):
        begin, end, _ = subset[dim]
        if begin != end:
            raise NotImplementedError(f"Expected a point write of '{data}'.")
        origin = dace_symbolic.simplify(begin - sym(param))
        if origin.free_symbols:
            raise NotImplementedError(f"Write of '{data}' is not a constant offset of '{param}'.")
        origins.append(origin)
    if subset[3][0] != sym(k):
        raise NotImplementedError(f"Write of '{data}' is not at the vertical map index.")
    return tuple(origins)


def _scope_nodes(state: dace.SDFGState, entry: dace_nodes.MapEntry) -> set[dace_nodes.Node]:
    return set(state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes())


def _ancestors(
    state: dace.SDFGState, roots: Sequence[dace_nodes.Node], inside: set[dace_nodes.Node]
) -> set[dace_nodes.Node]:
    seen: set[dace_nodes.Node] = set()
    stack = [node for node in roots if node in inside]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(edge.src for edge in state.in_edges(node) if edge.src in inside)
    return seen


def _index_rebase(
    expr_str: str, dim: int, staged: _Staged, ti: str, tj: str, halo: int, double_buffer: bool
) -> str:
    if dim == 0:
        return f"(({expr_str}) - ({staged.origin_i}) - ({ti} - {halo}))"
    if dim == 1:
        return f"(({expr_str}) - ({staged.origin_j}) - ({tj} - {halo}))"
    if dim == 3:
        return f"(({expr_str}) % 2)" if double_buffer else "0"
    return expr_str


def _rebase_subset(
    subset: dace_subsets.Range,
    staged: _Staged,
    ti: str,
    tj: str,
    halo: int,
    double_buffer: bool,
) -> dace_subsets.Range:
    ranges = []
    for dim, (begin, end, _) in enumerate(subset):
        if begin != end:
            raise NotImplementedError("Only point accesses of a staged transient are supported.")
        index = dace_symbolic.pystr_to_symbolic(
            _index_rebase(str(begin), dim, staged, ti, tj, halo, double_buffer)
        )
        ranges.append((index, index, 1))
    return dace_subsets.Range(ranges)


def _full_tile_memlet(sdfg: dace.SDFG, staged: _Staged, like: dace.Memlet) -> dace.Memlet:
    """The whole tile, with the dimensions of `like` that are fixed kept fixed (same squeezing)."""
    desc = sdfg.arrays[staged.tile]
    ranges = []
    for dim, (begin, end, _) in enumerate(like.subset):
        if begin == end and dim == 2:
            ranges.append((begin, end, 1))
        elif begin == end:
            raise NotImplementedError("A staged transient is read through a squeezed I, J or K.")
        else:
            ranges.append((0, desc.shape[dim] - 1, 1))
    return dace.Memlet(data=staged.tile, subset=dace_subsets.Range(ranges))


class _RebaseSubscripts(ast.NodeTransformer):
    def __init__(self, conn: str, dims: list[int], rebase: Callable[[str, int], str]) -> None:
        self.conn, self.dims, self.rebase = conn, dims, rebase

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if not (isinstance(node.value, ast.Name) and node.value.id == self.conn):
            return node
        indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if len(indices) != len(self.dims):
            raise NotImplementedError(f"Unexpected subscript of '{self.conn}'.")
        new = [
            ast.parse(self.rebase(ast.unparse(idx), dim), mode="eval").body
            for idx, dim in zip(indices, self.dims, strict=True)
        ]
        node.slice = ast.Tuple(elts=new, ctx=ast.Load()) if len(new) > 1 else new[0]
        return node


def _halo(
    state: dace.SDFGState, consumer_entry: dace_nodes.MapEntry, staged: dict[str, _Staged]
) -> int:
    """Largest horizontal distance between a consumer point and the staged values it reads."""
    _, _, j, i = _split_params(consumer_entry.map.params)
    halo = 0

    def account(index_i: str, index_j: str, origin_i: Any, origin_j: Any) -> None:
        nonlocal halo
        for index, origin, param in ((index_i, origin_i, i), (index_j, origin_j, j)):
            offset = dace_symbolic.simplify(
                dace_symbolic.pystr_to_symbolic(index) - origin - dace_symbolic.symbol(param)
            )
            if offset.free_symbols:
                raise NotImplementedError("Staged reads must be at constant horizontal offsets.")
            halo = max(halo, abs(int(offset)))

    for edge in state.out_edges(consumer_entry):
        if edge.data.data not in staged:
            continue
        st = staged[edge.data.data]
        if isinstance(edge.dst, dace_nodes.Tasklet) and any(
            begin != end for begin, end, _ in edge.data.subset
        ):
            dims = [d for d, (b, e, _) in enumerate(edge.data.subset) if b != e]
            for sub in ast.walk(ast.parse(edge.dst.code.as_string)):
                if (
                    isinstance(sub, ast.Subscript)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == edge.dst_conn
                ):
                    idx = sub.slice.elts if isinstance(sub.slice, ast.Tuple) else [sub.slice]
                    by_dim = dict(zip(dims, (ast.unparse(x) for x in idx), strict=True))
                    if 3 in by_dim:
                        k_offset = dace_symbolic.simplify(
                            dace_symbolic.pystr_to_symbolic(by_dim[3])
                            - dace_symbolic.symbol(consumer_entry.map.params[0])
                        )
                        if k_offset != 0:
                            raise NotImplementedError("Staged reads must have no vertical offset.")
                    account(by_dim[0], by_dim[1], st.origin_i, st.origin_j)
        else:
            subset = edge.data.subset
            account(str(subset[0][0]), str(subset[1][0]), st.origin_i, st.origin_j)
    return halo


class _ScopeCopier:
    """Copies nodes of a top-level map body into a new scope chain, renaming transients."""

    def __init__(
        self,
        sdfg: dace.SDFG,
        state: dace.SDFGState,
        chain_entries: list[dace_nodes.MapEntry],
        chain_exits: list[dace_nodes.MapExit],
        in_routes: dict[tuple[dace_nodes.MapEntry, dace_nodes.AccessNode], str],
    ) -> None:
        self.sdfg, self.state = sdfg, state
        self.entries, self.exits = chain_entries, chain_exits
        self.renamed: dict[str, str] = {}
        self.inner: set[str] = set()
        self.in_routes = in_routes
        self.out_routes: dict[str, str] = {}

    def data_name(self, name: str) -> str:
        if name not in self.inner:
            return name
        if name not in self.renamed:
            self.renamed[name] = self.sdfg.add_datadesc(
                f"{_PREFIX}_{name}", copy.deepcopy(self.sdfg.arrays[name]), find_new_name=True
            )
        return self.renamed[name]

    def memlet(self, memlet: dace.Memlet) -> dace.Memlet:
        new = copy.deepcopy(memlet)
        if new.data is not None:
            new.data = self.data_name(new.data)
        return new

    def copy_nodes(self, nodes: set[dace_nodes.Node]) -> dict[dace_nodes.Node, dace_nodes.Node]:
        self.inner = {node.data for node in nodes if isinstance(node, dace_nodes.AccessNode)}
        mapping = {}
        for node in nodes:
            new = copy.deepcopy(node)
            if isinstance(new, dace_nodes.AccessNode):
                new.data = self.data_name(new.data)
            self.state.add_node(new)
            mapping[node] = new
        for node in nodes:
            for edge in self.state.out_edges(node):
                if edge.dst in mapping:
                    self.state.add_edge(
                        mapping[node],
                        edge.src_conn,
                        mapping[edge.dst],
                        edge.dst_conn,
                        self.memlet(edge.data),
                    )
        return mapping

    def route_in(self, outer_src: dace_nodes.AccessNode, outer_memlet: dace.Memlet) -> str:
        """Bring `outer_src` into the innermost entry of the chain; returns the out-connector."""
        src: dace_nodes.Node = outer_src
        src_conn: Optional[str] = None
        for entry in self.entries:
            key = (entry, outer_src)
            if key not in self.in_routes:
                conn = f"{outer_src.data}_{len(self.in_routes)}"
                entry.add_in_connector(f"IN_{conn}")
                entry.add_out_connector(f"OUT_{conn}")
                self.state.add_edge(src, src_conn, entry, f"IN_{conn}", copy.deepcopy(outer_memlet))
                self.in_routes[key] = f"OUT_{conn}"
            src, src_conn = entry, self.in_routes[key]
        assert src_conn is not None
        return src_conn

    def route_out(
        self, outer_dst: dace_nodes.AccessNode, outer_memlet: dace.Memlet, last: int
    ) -> str:
        """Connect the innermost exit of the chain to `outer_dst`; returns the in-connector."""
        key = f"{id(outer_dst)}:{last}"
        if key not in self.out_routes:
            conn = f"{outer_dst.data}_{id(outer_dst) % 100000}"
            exits = self.exits[last:]
            for depth, exit_ in enumerate(exits):
                exit_.add_in_connector(f"IN_{conn}")
                exit_.add_out_connector(f"OUT_{conn}")
                dst, dst_conn = (
                    (exits[depth + 1], f"IN_{conn}")
                    if depth + 1 < len(exits)
                    else (outer_dst, None)
                )
                self.state.add_edge(
                    exit_, f"OUT_{conn}", dst, dst_conn, copy.deepcopy(outer_memlet)
                )
            self.out_routes[key] = f"IN_{conn}"
        return self.out_routes[key]


def _remove_dead(state: dace.SDFGState, nodes: set) -> None:
    """Remove `nodes`, then connectors of scope nodes left without edges."""
    touched = set()
    for node in nodes:
        for edge in list(state.all_edges(node)):
            touched.add(edge.src)
            touched.add(edge.dst)
        state.remove_node(node)
    for node in touched - nodes:
        if isinstance(node, dace_nodes.EntryNode):
            for conn in list(node.out_connectors):
                if conn.startswith("OUT_") and not list(state.out_edges_by_connector(node, conn)):
                    for edge in list(state.in_edges_by_connector(node, "IN_" + conn[4:])):
                        state.remove_edge(edge)
                        if (
                            isinstance(edge.src, dace_nodes.AccessNode)
                            and state.degree(edge.src) == 0
                        ):
                            state.remove_node(edge.src)
                    node.remove_in_connector("IN_" + conn[4:])
                    node.remove_out_connector(conn)


def apply_staging(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    producer_entry: dace_nodes.MapEntry,
    consumer_entry: dace_nodes.MapEntry,
    data: Sequence[str],
    *,
    tile: tuple[int, int] = (32, 16),
    k_chunk: int = 40,
    double_buffer: bool = True,
) -> dace_nodes.MapEntry:
    """Compute the transients `data` of `producer_entry` in shared memory inside `consumer_entry`.

    The consumer becomes one kernel over horizontal tiles of `tile` threads (core plus halo)
    and chunks of `k_chunk` levels. Per level, one thread-block map computes the producer's
    dataflow of `data` on the tile into a shared array, the next one runs the consumer on the
    core. The producer keeps everything else it computes; the transients are removed.

    Both maps must be top-level `GPU_Device` maps with parameters ordered `(K, X, J, I)`, the
    producer must write each transient at `[i_I + oI, i_J + oJ, x, i_K]`, the consumer must be
    its only reader, reading it at constant horizontal offsets, without vertical offset, through
    memlets or tasklet subscripts.
    """
    tile_i, tile_j = tile
    p_params, c_params = producer_entry.map.params, consumer_entry.map.params
    k, x, j, i = _split_params(c_params)
    if list(p_params) != list(c_params):
        raise NotImplementedError("Producer and consumer must have the same map parameters.")
    for name in data:
        readers = [
            node
            for node in state.data_nodes()
            if node.data == name
            for edge in state.out_edges(node)
            if edge.dst is not consumer_entry
        ]
        if readers or any(
            n.data == name for s in sdfg.states() if s is not state for n in s.data_nodes()
        ):
            raise NotImplementedError(f"'{name}' has readers other than the consumer.")

    staged: dict[str, _Staged] = {}
    for name in data:
        desc = sdfg.arrays[name]
        origin_i, origin_j = _write_origin(state, producer_entry, name)
        tile_name = sdfg.add_array(
            f"{_PREFIX}_tile_{name}",
            shape=(tile_i, tile_j, desc.shape[2], 2 if double_buffer else 1),
            dtype=desc.dtype,
            storage=dace.StorageType.GPU_Shared,
            lifetime=dace.AllocationLifetime.Scope,
            transient=True,
            find_new_name=True,
        )[0]
        staged[name] = _Staged(name, tile_name, origin_i, origin_j)

    halo = _halo(state, consumer_entry, staged)
    p_exit, c_exit = state.exit_node(producer_entry), state.exit_node(consumer_entry)
    p_body, c_body = _scope_nodes(state, producer_entry), _scope_nodes(state, consumer_entry)
    core_i, core_j = tile_i - 2 * halo, tile_j - 2 * halo
    if core_i < 1 or core_j < 1:
        raise ValueError(f"Tile {tile} is too small for a halo of {halo}.")

    c_k, c_x, c_j, c_i = (_range_bounds(r) for r in consumer_entry.map.range)
    p_k, p_x, p_j, p_i = (_range_bounds(r) for r in producer_entry.map.range)
    if p_k != c_k:
        raise NotImplementedError("Producer and consumer must have the same vertical range.")

    kb, tj, ti = f"{_PREFIX}_kb", f"{_PREFIX}_tj", f"{_PREFIX}_ti"
    pj, pi, cj, ci = f"{_PREFIX}_pj", f"{_PREFIX}_pi", f"{_PREFIX}_cj", f"{_PREFIX}_ci"
    label = f"{_PREFIX}_{consumer_entry.map.label}"

    kernel_entry, kernel_exit = state.add_map(
        label,
        {
            kb: f"{c_k[0]}:{c_k[1]}+1:{k_chunk}",
            tj: f"{c_j[0]}:{c_j[1]}+1:{core_j}",
            ti: f"{c_i[0]}:{c_i[1]}+1:{core_i}",
        },
        schedule=dace.ScheduleType.GPU_Device,
    )
    kernel_entry.map.gpu_block_size = [tile_i, tile_j, 1]
    kernel_entry.map.gpu_launch_bounds = "0"
    k_entry, k_exit = state.add_map(
        f"{label}_k",
        {k: f"{kb}:Min({kb} + {k_chunk}, {c_k[1]} + 1)"},
        schedule=dace.ScheduleType.Sequential,
    )

    def tb_map(suffix: str, lj: str, li: str) -> tuple[dace_nodes.MapEntry, dace_nodes.MapExit]:
        return state.add_map(
            f"{label}_{suffix}",
            {lj: f"0:{tile_j}", li: f"0:{tile_i}"},
            schedule=dace.ScheduleType.GPU_ThreadBlock,
        )

    tbp_entry, tbp_exit = tb_map("tb_producer", pj, pi)
    tbc_entry, tbc_exit = tb_map("tb_consumer", cj, ci)
    pg_entry, pg_exit = state.add_map(
        f"{label}_producer",
        {
            x: f"{p_x[0]}:{p_x[1]}+1",
            j: f"Max({p_j[0]}, {tj} - {halo} + {pj}):Min({p_j[1]}, {tj} - {halo} + {pj})+1",
            i: f"Max({p_i[0]}, {ti} - {halo} + {pi}):Min({p_i[1]}, {ti} - {halo} + {pi})+1",
        },
        schedule=dace.ScheduleType.Sequential,
    )
    cg_entry, cg_exit = state.add_map(
        f"{label}_consumer",
        {
            x: f"{c_x[0]}:{c_x[1]}+1",
            j: f"Max({c_j[0]}, {tj} - {halo} + {cj}, {tj}):"
            f"Min({c_j[1]}, {tj} - {halo} + {cj}, {tj} + {core_j - 1})+1",
            i: f"Max({c_i[0]}, {ti} - {halo} + {ci}, {ti}):"
            f"Min({c_i[1]}, {ti} - {halo} + {ci}, {ti} + {core_i - 1})+1",
        },
        schedule=dace.ScheduleType.Sequential,
    )
    p_chain = (
        [kernel_entry, k_entry, tbp_entry, pg_entry],
        [pg_exit, tbp_exit, k_exit, kernel_exit],
    )
    c_chain = (
        [kernel_entry, k_entry, tbc_entry, cg_entry],
        [cg_exit, tbc_exit, k_exit, kernel_exit],
    )

    tile_nodes = {name: state.add_access(st.tile) for name, st in staged.items()}
    for name, node in tile_nodes.items():
        full = dace.Memlet.from_array(staged[name].tile, sdfg.arrays[staged[name].tile])
        state.add_edge(tbp_exit, f"OUT_{node.data}", node, None, copy.deepcopy(full))
        tbp_exit.add_out_connector(f"OUT_{node.data}")
        tbp_exit.add_in_connector(f"IN_{node.data}")
        pg_exit.add_out_connector(f"OUT_{node.data}")
        pg_exit.add_in_connector(f"IN_{node.data}")
        state.add_edge(
            pg_exit, f"OUT_{node.data}", tbp_exit, f"IN_{node.data}", copy.deepcopy(full)
        )
        for entry in (tbc_entry, cg_entry):
            entry.add_in_connector(f"IN_{node.data}")
            entry.add_out_connector(f"OUT_{node.data}")
        state.add_edge(node, None, tbc_entry, f"IN_{node.data}", copy.deepcopy(full))
        state.add_edge(
            tbc_entry, f"OUT_{node.data}", cg_entry, f"IN_{node.data}", copy.deepcopy(full)
        )

    # Producer: copy the dataflow of the staged transients.
    staged_writes = [e for e in state.in_edges(p_exit) if e.data.data in staged]
    other_writes = [e for e in state.in_edges(p_exit) if e.data.data not in staged]
    cone = _ancestors(state, [e.src for e in staged_writes], p_body)
    in_routes: dict = {}
    copier = _ScopeCopier(sdfg, state, *p_chain, in_routes)
    mapping = copier.copy_nodes(cone)
    _connect_inputs(state, copier, producer_entry, mapping)
    for edge in staged_writes:
        st = staged[edge.data.data]
        memlet = dace.Memlet(
            data=st.tile,
            subset=_rebase_subset(edge.data.subset, st, ti, tj, halo, double_buffer),
        )
        state.add_edge(mapping[edge.src], edge.src_conn, pg_exit, f"IN_{st.tile}", memlet)
    _connect_empty_entry(state, producer_entry, pg_entry, mapping)

    # Consumer: copy the whole body, reading the staged transients from the tile.
    copier = _ScopeCopier(sdfg, state, *c_chain, in_routes)
    mapping = copier.copy_nodes(c_body)
    for edge in state.out_edges(consumer_entry):
        if edge.data.data not in staged:
            continue
        st = staged[edge.data.data]
        dst = mapping[edge.dst]
        if isinstance(dst, dace_nodes.Tasklet) and any(b != e for b, e, _ in edge.data.subset):
            dims = [d for d, (b, e, _) in enumerate(edge.data.subset) if b != e]
            tree = _RebaseSubscripts(
                edge.dst_conn,
                dims,
                functools.partial(
                    _index_rebase,
                    staged=st,
                    ti=ti,
                    tj=tj,
                    halo=halo,
                    double_buffer=double_buffer,
                ),
            ).visit(ast.parse(dst.code.as_string))
            dst.code = dace.properties.CodeBlock(ast.unparse(tree), dst.code.language)
            memlet = _full_tile_memlet(sdfg, st, edge.data)
        else:
            memlet = dace.Memlet(
                data=st.tile,
                subset=_rebase_subset(edge.data.subset, st, ti, tj, halo, double_buffer),
            )
        state.add_edge(cg_entry, f"OUT_{st.tile}", dst, edge.dst_conn, memlet)
    _connect_inputs(state, copier, consumer_entry, mapping, skip=set(staged))
    _connect_empty_entry(state, consumer_entry, cg_entry, mapping)
    for edge in state.in_edges(c_exit):
        outer = next(state.out_edges_by_connector(c_exit, "OUT_" + edge.dst_conn[3:]))
        conn = copier.route_out(outer.dst, outer.data, 0)
        state.add_edge(mapping[edge.src], edge.src_conn, cg_exit, conn, copier.memlet(edge.data))

    # Remove the consumer and the producer's dataflow of the staged transients.
    staged_nodes = {edge.dst for edge in state.out_edges(p_exit) if edge.data.data in staged} | {
        edge.src for edge in state.in_edges(consumer_entry) if edge.data.data in staged
    }
    _remove_dead(state, c_body | {consumer_entry, c_exit})
    keep = _ancestors(state, [e.src for e in other_writes], p_body)
    for edge in staged_writes:
        state.remove_edge(edge)
        p_exit.remove_in_connector(edge.dst_conn)
    for edge in list(state.out_edges(p_exit)):
        if edge.data.data in staged:
            state.remove_edge(edge)
            p_exit.remove_out_connector(edge.src_conn)
    if other_writes:
        _remove_dead(state, cone - keep)
    else:
        inputs = {edge.src for edge in state.in_edges(producer_entry)}
        _remove_dead(state, p_body | {producer_entry, p_exit})
        for node in inputs:
            if state.degree(node) == 0:
                state.remove_node(node)
    for node in staged_nodes:
        if node in state.nodes() and state.degree(node) == 0:
            state.remove_node(node)
    for name in staged:
        if not any(n.data == name for s in sdfg.states() for n in s.data_nodes()):
            sdfg.remove_data(name, validate=False)
    return kernel_entry


def _connect_inputs(
    state: dace.SDFGState,
    copier: _ScopeCopier,
    old_entry: dace_nodes.MapEntry,
    mapping: dict[dace_nodes.Node, dace_nodes.Node],
    skip: Optional[set[str]] = None,
) -> None:
    for edge in state.out_edges(old_entry):
        if edge.dst not in mapping or edge.data.is_empty():
            continue
        if skip and edge.data.data in skip:
            continue
        outer = next(state.in_edges_by_connector(old_entry, "IN_" + edge.src_conn[4:]))
        conn = copier.route_in(outer.src, outer.data)
        state.add_edge(
            copier.entries[-1], conn, mapping[edge.dst], edge.dst_conn, copier.memlet(edge.data)
        )


def _connect_empty_entry(
    state: dace.SDFGState,
    old_entry: dace_nodes.MapEntry,
    new_entry: dace_nodes.MapEntry,
    mapping: dict[dace_nodes.Node, dace_nodes.Node],
) -> None:
    for edge in state.out_edges(old_entry):
        if edge.dst in mapping and edge.data.is_empty():
            state.add_edge(new_entry, None, mapping[edge.dst], edge.dst_conn, dace.Memlet())
    for node in mapping.values():
        if state.in_degree(node) == 0:
            state.add_edge(new_entry, None, node, None, dace.Memlet())


def gt_stage_in_shared_memory(
    sdfg: dace.SDFG,
    *,
    pairs: Sequence[tuple[str, str, Sequence[str]]],
    tile: tuple[int, int] = (32, 16),
    k_chunk: int = 40,
    double_buffer: bool = True,
) -> int:
    """Apply `apply_staging()` to the `(producer label, consumer label, transients)` pairs.

    Maps are matched by label prefix among the top-level maps of the SDFG's states. Returns the
    number of pairs staged; pairs whose maps are not found are skipped.
    """
    count = 0
    for producer, consumer, data in pairs:
        for state in sdfg.states():
            entries = {
                node.map.label: node
                for node in state.nodes()
                if isinstance(node, dace_nodes.MapEntry) and state.entry_node(node) is None
            }
            p = [e for label, e in entries.items() if label.startswith(producer)]
            c = [e for label, e in entries.items() if label.startswith(consumer)]
            if len(p) == 1 and len(c) == 1:
                apply_staging(
                    sdfg,
                    state,
                    p[0],
                    c[0],
                    data,
                    tile=tile,
                    k_chunk=k_chunk,
                    double_buffer=double_buffer,
                )
                count += 1
                break
    return count
