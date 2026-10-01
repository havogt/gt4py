# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Stage a transient between two GPU kernels in shared memory (opt-in)."""

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
_MAX_STATIC_SHARED_BYTES = 48 * 1024


class StagingRefusedError(NotImplementedError):
    """The producer/consumer pair does not have the shape the staging supports."""


@dataclasses.dataclass(frozen=True)
class _Staged:
    """One staged transient and its shared-memory tile, indexed `(I, J, X, buffer)`."""

    data: str
    tile: str
    origin_i: Any
    origin_j: Any


@dataclasses.dataclass(frozen=True)
class _Read:
    """A read of a staged transient by the consumer: memlet or tasklet subscripts."""

    edge: Any
    subscript_dims: Optional[list[int]]


def _refuse(message: str) -> None:
    raise StagingRefusedError(message)


def _range_bounds(rng: tuple[Any, Any, Any]) -> tuple[Any, Any]:
    if rng[2] != 1:
        _refuse("Strided map ranges are not supported.")
    return rng[0], rng[1]


def _split_params(params: Sequence[str]) -> tuple[str, str, str, str]:
    if len(params) != 4 or not params[0].endswith("_vertical"):
        _refuse(f"Expected map parameters ordered (K, X, J, I), got {params}.")
    return params[0], params[1], params[2], params[3]


def _offset(expr: Any, param: str) -> int:
    """`expr - param` as an integer, refusing anything that is not a constant offset."""
    offset = dace_symbolic.simplify(
        dace_symbolic.pystr_to_symbolic(str(expr)) - dace_symbolic.symbol(param)
    )
    if offset.free_symbols:
        _refuse(f"'{expr}' is not a constant offset of '{param}'.")
    return int(offset)


def _write_origin(
    state: dace.SDFGState, producer_exit: dace_nodes.MapExit, data: str, params: Sequence[str]
) -> tuple[int, int]:
    """Constant offsets `(oI, oJ)` of the producer's write `T[i_I + oI, i_J + oJ, x, i_K]`."""
    k, _, j, i = params
    writes = [edge for edge in state.in_edges(producer_exit) if edge.data.data == data]
    if len(writes) != 1:
        _refuse(f"Expected one write of '{data}' in the producer.")
    subset = writes[0].data.subset
    if any(begin != end for begin, end, _ in subset):
        _refuse(f"Expected a point write of '{data}'.")
    if _offset(subset[3][0], k) != 0:
        _refuse(f"Write of '{data}' is not at the vertical map index.")
    return _offset(subset[0][0], i), _offset(subset[1][0], j)


def _ordered(state: dace.SDFGState, nodes: set[dace_nodes.Node]) -> list[dace_nodes.Node]:
    return [node for node in state.nodes() if node in nodes]


def _scope_nodes(state: dace.SDFGState, entry: dace_nodes.MapEntry) -> list[dace_nodes.Node]:
    inside = set(state.scope_subgraph(entry, include_entry=False, include_exit=False).nodes())
    return _ordered(state, inside)


def _ancestors(
    state: dace.SDFGState, roots: Sequence[dace_nodes.Node], inside: Sequence[dace_nodes.Node]
) -> list[dace_nodes.Node]:
    inside_set = set(inside)
    seen: set[dace_nodes.Node] = set()
    stack = [node for node in roots if node in inside_set]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(edge.src for edge in state.in_edges(node) if edge.src in inside_set)
    return _ordered(state, seen)


def _subscripts(code: str, conn: str) -> list[ast.Subscript]:
    """The subscripts of `conn` in `code`; refuses any other use of `conn`."""
    tree = ast.parse(code)
    subscripts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == conn
    ]
    names = [node for node in ast.walk(tree) if isinstance(node, ast.Name) and node.id == conn]
    if len(names) != len(subscripts):
        _refuse(f"'{conn}' is used without a subscript.")
    return subscripts


def _subscript_indices(node: ast.Subscript) -> list[str]:
    indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    return [ast.unparse(index) for index in indices]


def _index_rebase(
    expr_str: str, dim: int, staged: _Staged, ti: str, tj: str, halo: int, double_buffer: bool
) -> Optional[str]:
    if dim == 0:
        return f"(({expr_str}) - ({staged.origin_i}) - ({ti} - {halo}))"
    if dim == 1:
        return f"(({expr_str}) - ({staged.origin_j}) - ({tj} - {halo}))"
    if dim == 3:
        return f"(({expr_str}) % 2)" if double_buffer else None
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
    for dim, (begin, _, _) in enumerate(subset):
        rebased = _index_rebase(str(begin), dim, staged, ti, tj, halo, double_buffer)
        index = dace_symbolic.pystr_to_symbolic(rebased) if rebased is not None else 0
        ranges.append((index, index, 1))
    return dace_subsets.Range(ranges)


def _full_tile_memlet(sdfg: dace.SDFG, staged: _Staged, like: dace.Memlet) -> dace.Memlet:
    """The whole tile, with the colour index of `like` kept fixed (same squeezing)."""
    desc = sdfg.arrays[staged.tile]
    ranges = [
        (begin, end, 1) if dim == 2 else (0, desc.shape[dim] - 1, 1)
        for dim, (begin, end, _) in enumerate(like.subset)
    ]
    return dace.Memlet(data=staged.tile, subset=dace_subsets.Range(ranges))


class _RebaseSubscripts(ast.NodeTransformer):
    def __init__(
        self, conn: str, dims: list[int], rebase: Callable[[str, int], Optional[str]]
    ) -> None:
        self.conn, self.dims, self.rebase = conn, dims, rebase

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        self.generic_visit(node)
        if not (isinstance(node.value, ast.Name) and node.value.id == self.conn):
            return node
        rebased = [
            self.rebase(index, dim)
            for index, dim in zip(_subscript_indices(node), self.dims, strict=True)
        ]
        # a single buffer is a squeezed dimension of the tile memlet
        new = [ast.parse(index, mode="eval").body for index in rebased if index is not None]
        node.slice = ast.Tuple(elts=new, ctx=ast.Load()) if len(new) > 1 else new[0]
        return node


def _check_reads(
    state: dace.SDFGState,
    consumer_entry: dace_nodes.MapEntry,
    staged: dict[str, _Staged],
) -> tuple[list[_Read], int]:
    """The consumer's reads of the staged transients and the largest horizontal offset."""
    k, _, j, i = _split_params(consumer_entry.map.params)
    reads: list[_Read] = []
    halo = 0

    def account(indices: dict[int, Any], name: str) -> None:
        nonlocal halo
        st = staged[name]
        if _offset(indices[3], k) != 0:
            _refuse(f"'{name}' is read with a vertical offset.")
        halo = max(
            halo,
            abs(_offset(indices[0], i) - st.origin_i),
            abs(_offset(indices[1], j) - st.origin_j),
        )

    for edge in state.out_edges(consumer_entry):
        if edge.data.data not in staged:
            continue
        if not isinstance(edge.dst, dace_nodes.Tasklet):
            _refuse(f"'{edge.data.data}' is read by a {type(edge.dst).__name__}, not a tasklet.")
        subset = edge.data.subset
        points = [begin == end for begin, end, _ in subset]
        if all(points):
            account({dim: begin for dim, (begin, _, _) in enumerate(subset)}, edge.data.data)
            reads.append(_Read(edge, None))
            continue
        if not (points[2] and not any(points[d] for d in (0, 1, 3))):
            _refuse(f"'{edge.data.data}' is read through a squeezed I, J or K dimension.")
        dims = [0, 1, 3]
        for node in _subscripts(edge.dst.code.as_string, edge.dst_conn):
            indices = _subscript_indices(node)
            if len(indices) != len(dims):
                _refuse(f"Unexpected subscript of '{edge.dst_conn}'.")
            account(dict(zip(dims, indices, strict=True)), edge.data.data)
        reads.append(_Read(edge, dims))
    return reads, halo


def _check_scope_free(nodes: Sequence[dace_nodes.Node], what: str) -> None:
    for node in nodes:
        if isinstance(node, (dace_nodes.EntryNode, dace_nodes.ExitNode, dace_nodes.NestedSDFG)):
            _refuse(f"The {what} contains a nested scope or SDFG ({node}).")


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
        self.out_routes: dict[dace_nodes.AccessNode, str] = {}

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

    def copy_nodes(
        self, nodes: Sequence[dace_nodes.Node]
    ) -> dict[dace_nodes.Node, dace_nodes.Node]:
        self.inner = {
            node.data
            for node in nodes
            if isinstance(node, dace_nodes.AccessNode) and self.sdfg.arrays[node.data].transient
        }
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

    def route_in(self, outer_src: dace_nodes.AccessNode) -> str:
        """Bring `outer_src` into the innermost entry of the chain; returns the out-connector."""
        full = dace.Memlet.from_array(outer_src.data, self.sdfg.arrays[outer_src.data])
        src: dace_nodes.Node = outer_src
        src_conn: Optional[str] = None
        for entry in self.entries:
            key = (entry, outer_src)
            if key not in self.in_routes:
                conn = f"{outer_src.data}_{len(self.in_routes)}"
                entry.add_in_connector(f"IN_{conn}")
                entry.add_out_connector(f"OUT_{conn}")
                self.state.add_edge(src, src_conn, entry, f"IN_{conn}", copy.deepcopy(full))
                self.in_routes[key] = f"OUT_{conn}"
            src, src_conn = entry, self.in_routes[key]
        assert src_conn is not None
        return src_conn

    def route_out(self, outer_dst: dace_nodes.AccessNode) -> str:
        """Connect the innermost exit of the chain to `outer_dst`; returns the in-connector."""
        if outer_dst not in self.out_routes:
            full = dace.Memlet.from_array(outer_dst.data, self.sdfg.arrays[outer_dst.data])
            conn = f"{outer_dst.data}_out_{len(self.out_routes)}"
            for depth, exit_ in enumerate(self.exits):
                exit_.add_in_connector(f"IN_{conn}")
                exit_.add_out_connector(f"OUT_{conn}")
                dst, dst_conn = (
                    (self.exits[depth + 1], f"IN_{conn}")
                    if depth + 1 < len(self.exits)
                    else (outer_dst, None)
                )
                self.state.add_edge(exit_, f"OUT_{conn}", dst, dst_conn, copy.deepcopy(full))
            self.out_routes[outer_dst] = f"IN_{conn}"
        return self.out_routes[outer_dst]


def _remove_dead(state: dace.SDFGState, nodes: Sequence[dace_nodes.Node]) -> None:
    """Remove `nodes`, then connectors of scope nodes left without edges."""
    removed = set(nodes)
    touched: list[dace_nodes.Node] = []
    for node in nodes:
        for edge in list(state.all_edges(node)):
            for end in (edge.src, edge.dst):
                if end not in removed and end not in touched:
                    touched.append(end)
        state.remove_node(node)
    for node in touched:
        if not isinstance(node, dace_nodes.EntryNode):
            continue
        for conn in sorted(node.out_connectors):
            if conn.startswith("OUT_") and not list(state.out_edges_by_connector(node, conn)):
                for edge in list(state.in_edges_by_connector(node, "IN_" + conn[4:])):
                    state.remove_edge(edge)
                    if isinstance(edge.src, dace_nodes.AccessNode) and state.degree(edge.src) == 0:
                        state.remove_node(edge.src)
                node.remove_in_connector("IN_" + conn[4:])
                node.remove_out_connector(conn)


def _sink_dangling(
    sdfg: dace.SDFG,
    state: dace.SDFGState,
    node: dace_nodes.Node,
    originals: Sequence[Any],
) -> None:
    """Connect out-connectors of `node` left without edges to scalar transients.

    `originals` are the edges that left those connectors before, for the type.
    """
    for edge in originals:
        conn = edge.src_conn
        if conn is None or list(state.out_edges_by_connector(node, conn)):
            continue
        name, _ = sdfg.add_scalar(
            f"{_PREFIX}_sink",
            sdfg.arrays[edge.data.data].dtype,
            storage=dace.StorageType.Register,
            transient=True,
            find_new_name=True,
        )
        state.add_edge(node, conn, state.add_access(name), None, dace.Memlet(f"{name}[0]"))


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
    """Compute transients of a producer kernel in shared memory inside their consumer.

    The consumer becomes one kernel over horizontal tiles of `tile` threads
    (core plus halo) and chunks of `k_chunk` levels. Per level, one thread-block
    map computes the producer's dataflow of `data` on the tile into a shared
    array, the next one runs the consumer on the core. The producer keeps its
    other outputs and is removed if it has none; the transients are removed.

    Requirements, all checked before the SDFG is modified: both maps are
    top-level `GPU_Device` maps with the same parameters ordered `(K, X, J, I)`
    and the same vertical range starting at 0 or above; the producer is the only
    writer of each transient, at `[i_I + oI, i_J + oJ, x, i_K]`, and covers the
    consumer's reads; the consumer is the only reader, through tasklets, at
    constant horizontal offsets and without vertical offset, either by point
    memlets or by subscripts of a memlet over the whole I, J and K extent;
    neither copied body contains a nested scope or SDFG.

    Args:
        sdfg: The SDFG.
        state: The state holding both maps.
        producer_entry: The map entry of the producer kernel.
        consumer_entry: The map entry of the consumer kernel.
        data: The transients to stage.
        tile: Threads per block in I and J, halo included.
        k_chunk: Vertical levels per block.
        double_buffer: Alternate between two tile buffers per level.

    Returns:
        The map entry of the new kernel.

    Raises:
        StagingRefusedError: The pair does not meet the requirements; the SDFG
            is unchanged.
        ValueError: The tile leaves no core, `k_chunk` is not positive, or the
            tiles exceed 48 KiB of static shared memory; the SDFG is unchanged.
    """
    tile_i, tile_j = tile
    if k_chunk < 1:
        raise ValueError(f"'k_chunk' must be positive, got {k_chunk}.")
    params = _split_params(consumer_entry.map.params)
    k, x, j, i = params
    if list(producer_entry.map.params) != list(params):
        _refuse("Producer and consumer must have the same map parameters.")
    for entry in (producer_entry, consumer_entry):
        if state.entry_node(entry) is not None:
            _refuse(f"'{entry.map.label}' is not a top-level map.")
    p_exit, c_exit = state.exit_node(producer_entry), state.exit_node(consumer_entry)
    c_k, c_x, c_j, c_i = (_range_bounds(r) for r in consumer_entry.map.range)
    p_k, p_x, p_j, p_i = (_range_bounds(r) for r in producer_entry.map.range)
    if p_k != c_k:
        _refuse("Producer and consumer must have the same vertical range.")
    if (c_k[0] < 0) == True:  # noqa: E712 [true-false-comparison]  # SymPy comparison
        _refuse("The vertical range must not start below 0.")

    staged: dict[str, _Staged] = {}
    tile_bytes = 0
    for name in data:
        desc = sdfg.arrays.get(name)
        if desc is None or not desc.transient:
            _refuse(f"'{name}' is not a transient.")
        for other in sdfg.states():
            for node in other.data_nodes():
                if node.data != name:
                    continue
                if other is not state:
                    _refuse(f"'{name}' is accessed in another state.")
                if any(edge.src is not p_exit for edge in state.in_edges(node)):
                    _refuse(f"'{name}' has writers other than the producer.")
                if any(edge.dst is not consumer_entry for edge in state.out_edges(node)):
                    _refuse(f"'{name}' has readers other than the consumer.")
        origin_i, origin_j = _write_origin(state, p_exit, name, params)
        staged[name] = _Staged(name, f"{_PREFIX}_tile_{name}", origin_i, origin_j)
        tile_bytes += (
            tile_i * tile_j * desc.shape[2] * (2 if double_buffer else 1) * (desc.dtype.bytes)
        )
    if tile_bytes > _MAX_STATIC_SHARED_BYTES:
        raise ValueError(f"The tiles need {tile_bytes} bytes of shared memory.")

    reads, halo = _check_reads(state, consumer_entry, staged)
    core_i, core_j = tile_i - 2 * halo, tile_j - 2 * halo
    if core_i < 1 or core_j < 1:
        raise ValueError(f"Tile {tile} is too small for a halo of {halo}.")
    for (p_lo, p_hi), (c_lo, c_hi) in ((p_i, c_i), (p_j, c_j)):
        if ((p_lo > c_lo - halo) == True) or ((p_hi < c_hi + halo) == True):  # noqa: E712 [true-false-comparison]  # SymPy comparison
            _refuse("The producer does not cover the consumer's reads.")

    p_body, c_body = _scope_nodes(state, producer_entry), _scope_nodes(state, consumer_entry)
    staged_writes = [e for e in state.in_edges(p_exit) if e.data.data in staged]
    other_writes = [e for e in state.in_edges(p_exit) if e.data.data not in staged]
    cone = _ancestors(state, [e.src for e in staged_writes], p_body)
    _check_scope_free(cone, "producer's dataflow of the staged transients")
    _check_scope_free(c_body, "consumer")
    if any(edge.data.is_empty() for edge in state.in_edges(c_exit)):
        _refuse("The consumer has an empty write.")

    # All checks done: modify the SDFG.
    for st in staged.values():
        desc = sdfg.arrays[st.data]
        n_x = desc.shape[2]
        # I is thread-x: unit stride avoids shared-memory bank conflicts
        tile_name = sdfg.add_array(
            st.tile,
            shape=(tile_i, tile_j, n_x, 2 if double_buffer else 1),
            strides=(1, tile_i, tile_i * tile_j, tile_i * tile_j * n_x),
            dtype=desc.dtype,
            storage=dace.StorageType.GPU_Shared,
            lifetime=dace.AllocationLifetime.Scope,
            transient=True,
            find_new_name=True,
        )[0]
        staged[st.data] = dataclasses.replace(st, tile=tile_name)

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

    for st in staged.values():
        node = state.add_access(st.tile)
        full = dace.Memlet.from_array(st.tile, sdfg.arrays[st.tile])
        for scope_node in (pg_exit, tbp_exit, tbc_entry, cg_entry):
            scope_node.add_in_connector(f"IN_{st.tile}")
            scope_node.add_out_connector(f"OUT_{st.tile}")
        state.add_edge(pg_exit, f"OUT_{st.tile}", tbp_exit, f"IN_{st.tile}", copy.deepcopy(full))
        state.add_edge(tbp_exit, f"OUT_{st.tile}", node, None, copy.deepcopy(full))
        state.add_edge(node, None, tbc_entry, f"IN_{st.tile}", copy.deepcopy(full))
        state.add_edge(tbc_entry, f"OUT_{st.tile}", cg_entry, f"IN_{st.tile}", copy.deepcopy(full))

    in_routes: dict[tuple[dace_nodes.MapEntry, dace_nodes.AccessNode], str] = {}
    rebase = functools.partial(_index_rebase, ti=ti, tj=tj, halo=halo, double_buffer=double_buffer)

    # Producer: copy the dataflow of the staged transients.
    copier = _ScopeCopier(
        sdfg,
        state,
        [kernel_entry, k_entry, tbp_entry, pg_entry],
        [pg_exit, tbp_exit, k_exit, kernel_exit],
        in_routes,
    )
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
    for node in cone:
        if isinstance(node, dace_nodes.Tasklet):
            _sink_dangling(sdfg, state, mapping[node], state.out_edges(node))

    # Consumer: copy the whole body, reading the staged transients from the tile.
    copier = _ScopeCopier(
        sdfg,
        state,
        [kernel_entry, k_entry, tbc_entry, cg_entry],
        [cg_exit, tbc_exit, k_exit, kernel_exit],
        in_routes,
    )
    mapping = copier.copy_nodes(c_body)
    for read in reads:
        edge = read.edge
        st = staged[edge.data.data]
        dst = mapping[edge.dst]
        if read.subscript_dims is not None:
            tree = _RebaseSubscripts(
                edge.dst_conn, read.subscript_dims, functools.partial(rebase, staged=st)
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
        conn = copier.route_out(outer.dst)
        state.add_edge(mapping[edge.src], edge.src_conn, cg_exit, conn, copier.memlet(edge.data))

    # Remove the consumer and the producer's dataflow of the staged transients.
    staged_nodes = _ordered(
        state,
        {edge.dst for edge in state.out_edges(p_exit) if edge.data.data in staged}
        | {edge.src for edge in state.in_edges(consumer_entry) if edge.data.data in staged},
    )
    _remove_dead(state, [*c_body, consumer_entry, c_exit])
    keep = set(_ancestors(state, [e.src for e in other_writes], p_body))
    for edge in staged_writes:
        state.remove_edge(edge)
        p_exit.remove_in_connector(edge.dst_conn)
        if other_writes and edge.src in keep and isinstance(edge.src, dace_nodes.Tasklet):
            _sink_dangling(sdfg, state, edge.src, [edge])
    for edge in list(state.out_edges(p_exit)):
        if edge.data.data in staged:
            state.remove_edge(edge)
            p_exit.remove_out_connector(edge.src_conn)
    if other_writes:
        _remove_dead(state, [node for node in cone if node not in keep])
    else:
        inputs = _ordered(state, {edge.src for edge in state.in_edges(producer_entry)})
        _remove_dead(state, [*p_body, producer_entry, p_exit])
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
        conn = copier.route_in(outer.src)
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
    """Apply `apply_staging()` to pairs of top-level maps given by label.

    Args:
        sdfg: The SDFG.
        pairs: `(producer label, consumer label, transients)` triples; labels
            are matched exactly against the top-level maps of each state.
        tile: See `apply_staging()`.
        k_chunk: See `apply_staging()`.
        double_buffer: See `apply_staging()`.

    Returns:
        The number of pairs staged; a pair whose maps are not found is skipped.

    Raises:
        ValueError: A label matches more than one map of a state, or see
            `apply_staging()`.
        StagingRefusedError: See `apply_staging()`.
    """
    count = 0
    for producer, consumer, data in pairs:
        for state in sdfg.states():
            entries = [
                node
                for node in state.nodes()
                if isinstance(node, dace_nodes.MapEntry) and state.entry_node(node) is None
            ]
            found = []
            for label in (producer, consumer):
                matches = [entry for entry in entries if entry.map.label == label]
                if len(matches) > 1:
                    raise ValueError(f"Label '{label}' matches {len(matches)} maps.")
                found.append(matches)
            if found[0] and found[1]:
                apply_staging(
                    sdfg,
                    state,
                    found[0][0],
                    found[1][0],
                    data,
                    tile=tile,
                    k_chunk=k_chunk,
                    double_buffer=double_buffer,
                )
                count += 1
                break
    return count
