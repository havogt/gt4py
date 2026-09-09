# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared interface for FESOM2-JAX-style halo transports on a structured 2-D torus.

A *transport* is a pure JAX function that refreshes the halo rim of one device's
local block inside a ``shard_map`` body. It is built only from ``jax.lax``
collectives and array operations, so ``jax.vjp`` transposes it automatically and
the distributed adjoint comes by composition (arXiv:2608.01546, Sect. 2.2-2.3).

Decomposition: ``P = Rx * Ry`` devices on a 1-D mesh axis; rank ``r`` owns the
block at ``(rx, ry) = divmod(r, Ry)``, i.e. global rows ``[rx*MLOC, (rx+1)*MLOC)``
and columns ``[ry*NLOC, (ry+1)*NLOC)``. This is the ordering of
``halo_lib.Decomposition`` and of ``swm_ghex_2d.py``. Local arrays are
``(MLOC+2h, NLOC+2h)`` with the interior at ``[h:-h, h:-h]``.

Directions: axis 0 is ``I`` (x, E/W), axis 1 is ``J`` (y, N/S).
``E = (+1, 0)``, ``W = (-1, 0)``, ``N = (0, +1)``, ``S = (0, -1)``, and the four
diagonals accordingly.
"""

from __future__ import annotations

import glob
import importlib
import inspect
import os
import re
import sys
from typing import Any, Protocol, runtime_checkable

import jax
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


NEIGHBOUR_OFFSETS = {
    "E": (1, 0),
    "W": (-1, 0),
    "N": (0, 1),
    "S": (0, -1),
    "NE": (1, 1),
    "NW": (-1, 1),
    "SE": (1, -1),
    "SW": (-1, -1),
}


class Layout:
    """Static 2-D torus decomposition of an ``(M, N)`` grid over ``Rx * Ry`` ranks."""

    def __init__(self, M: int, N: int, Rx: int, Ry: int, h: int = 1):
        if M % Rx or N % Ry:
            raise ValueError(f"grid {M}x{N} is not divisible by the {Rx}x{Ry} layout")
        self.M, self.N, self.Rx, self.Ry, self.h = M, N, Rx, Ry, h
        self.MLOC, self.NLOC = M // Rx, N // Ry
        if self.MLOC < h or self.NLOC < h:
            raise ValueError(
                f"halo {h} wider than the local block {self.MLOC}x{self.NLOC}: "
                "a nearest-neighbour exchange cannot fill it"
            )
        self.P = Rx * Ry
        self.local_shape = (self.MLOC + 2 * h, self.NLOC + 2 * h)
        self.ranks = np.arange(self.P, dtype=np.int32)
        rxs, rys = np.divmod(self.ranks, Ry)
        self.rx = rxs.astype(np.int32)
        self.ry = rys.astype(np.int32)
        self.neighbour_tables = {
            name: self.rank((rxs + dx) % Rx, (rys + dy) % Ry).astype(np.int32)
            for name, (dx, dy) in NEIGHBOUR_OFFSETS.items()
        }

    def coords(self, r):
        return divmod(r, self.Ry)

    def rank(self, rx, ry):
        return (np.asarray(rx) % self.Rx) * self.Ry + (np.asarray(ry) % self.Ry)

    def neighbour(self, dx: int, dy: int) -> np.ndarray:
        """Rank of the ``(dx, dy)`` torus neighbour, for every rank; shape ``(P,)``."""
        rx, ry = self.rx, self.ry
        return self.rank(rx + dx, ry + dy).astype(np.int32)

    def neighbours(self, r: int) -> dict[str, int]:
        """``{"E": rank, "W": rank, ...}`` for one rank; the 8 torus neighbours."""
        return {name: int(tab[r]) for name, tab in self.neighbour_tables.items()}

    def owner_lane(self):
        """Per-rank source maps for the local halo box.

        Returns ``(src_dev, src_lane)``, both ``(P, MLOC+2h, NLOC+2h)`` int32.
        ``src_dev[r, i, j]`` is the rank that owns local cell ``(i-h, j-h)`` of rank
        ``r``, and ``src_lane[r, i, j]`` is that cell's index in the owner's
        row-major flattened ``(MLOC, NLOC)`` interior. Interior lanes map to
        ``(r, own index)``; halo lanes to the owning neighbour.
        """
        h, M, N = self.h, self.M, self.N
        H, W = self.local_shape
        dev = np.empty((self.P, H, W), dtype=np.int32)
        lane = np.empty((self.P, H, W), dtype=np.int32)
        for r in range(self.P):
            i0, j0 = self.origin(r)
            gi = (np.arange(i0 - h, i0 + self.MLOC + h) % M).reshape(-1, 1)
            gj = (np.arange(j0 - h, j0 + self.NLOC + h) % N).reshape(1, -1)
            dev[r] = (gi // self.MLOC) * self.Ry + (gj // self.NLOC)
            lane[r] = (gi % self.MLOC) * self.NLOC + (gj % self.NLOC)
        return dev, lane

    def origin(self, r):
        rx, ry = self.coords(r)
        return rx * self.MLOC, ry * self.NLOC

    def block_slice(self, r):
        i0, j0 = self.origin(r)
        return (slice(i0, i0 + self.MLOC), slice(j0, j0 + self.NLOC))

    def __repr__(self):
        return (
            f"Layout(M={self.M}, N={self.N}, Rx={self.Rx}, Ry={self.Ry}, h={self.h}, "
            f"MLOC={self.MLOC}, NLOC={self.NLOC}, P={self.P})"
        )


def block(g, layout: Layout):
    """Global ``(M, N)`` -> rank-major stack of interior blocks, ``(P*MLOC, NLOC)``."""
    L = layout
    x = g.reshape(L.Rx, L.MLOC, L.Ry, L.NLOC)
    return x.transpose(0, 2, 1, 3).reshape(L.P * L.MLOC, L.NLOC)


def unblock(x, layout: Layout):
    """Rank-major stack of interior blocks ``(P*MLOC, NLOC)`` -> global ``(M, N)``."""
    L = layout
    y = x.reshape(L.Rx, L.Ry, L.MLOC, L.NLOC)
    return y.transpose(0, 2, 1, 3).reshape(L.M, L.N)


def block_halo(a, layout: Layout):
    """Stack of per-rank halo arrays ``(P, MLOC+2h, NLOC+2h)`` -> ``(P*(MLOC+2h), NLOC+2h)``."""
    L = layout
    return a.reshape(L.P * L.local_shape[0], L.local_shape[1])


def unblock_halo(a, layout: Layout):
    """``(P*(MLOC+2h), NLOC+2h)`` -> ``(P, MLOC+2h, NLOC+2h)``."""
    L = layout
    return a.reshape(L.P, *L.local_shape)


@runtime_checkable
class Transport(Protocol):
    """Contract every halo transport implements.

    ``prepare`` runs once, outside jit, and returns whatever static Python/NumPy
    data ``exchange`` needs (index arrays, permutations, slot maps).

    ``exchange`` runs *inside* a ``shard_map`` body on this device's
    ``(MLOC+2h, NLOC+2h)`` jnp array and returns the same shape with all halo
    cells -- four faces AND four corners -- refreshed from the owning neighbours,
    periodic in both directions. It must be pure jnp built only from ``jax.lax``
    collectives and array ops: no ``pure_callback``, no ``custom_vjp``, so that
    ``jax.vjp`` transposes it automatically.
    """

    name: str

    def prepare(self, layout: Layout) -> Any: ...

    def exchange(self, a_local: Any, tables: Any, axis_name: str) -> Any: ...


REGISTRY: dict[str, "Transport"] = {}


def register(transport: "Transport") -> "Transport":
    REGISTRY[transport.name] = transport
    return transport


def import_all_transports() -> dict[str, str]:
    """Import every ``transport_*.py`` sibling; return ``{module: "ok" | error}``."""
    seen = {}
    for path in sorted(glob.glob(os.path.join(_HERE, "transport_*.py"))):
        mod = os.path.splitext(os.path.basename(path))[0]
        try:
            importlib.import_module(mod)
            seen[mod] = "ok"
        except Exception as e:  # noqa: BLE001 - a broken sibling must not hide the others
            seen[mod] = f"{type(e).__name__}: {e}"
    return seen


def get_transport(name: str) -> "Transport":
    """Look up a transport by registered name.

    Tries ``transport_<name>.py`` first, then -- because a module may register
    several transports under names that are not its file name (``coloured8``,
    ``ragged_emul``) -- imports every ``transport_*.py`` sibling and retries.
    """
    if name in REGISTRY:
        return REGISTRY[name]
    try:
        importlib.import_module(f"transport_{name}")
    except ImportError:
        pass
    if name not in REGISTRY:
        failed = {m: e for m, e in import_all_transports().items() if e != "ok"}
        if name not in REGISTRY:
            raise KeyError(
                f"unknown transport {name!r}; registered: {sorted(REGISTRY)}"
                + (f"; modules that failed to import: {failed}" if failed else "")
            )
    return REGISTRY[name]


def available_transports(names=None):
    """Every transport that can be imported right now, in registration order."""
    if names is None:
        import_all_transports()
        return list(REGISTRY)
    out = []
    for n in names:
        try:
            get_transport(n)
        except KeyError:
            continue
        out.append(n)
    return out


# --- shard_map compatibility ---------------------------------------------------------------
# jax >= 0.11: ``jax.shard_map(f, mesh=, in_specs=, out_specs=, check_vma=)`` and
# ``jax.experimental.shard_map`` is an empty shim module. jax 0.6.2: the API lives in
# ``jax.experimental.shard_map`` and the replication flag is ``check_rep``. 0.6.2 also
# exposes a top-level ``jax.shard_map`` with the new keyword names, so prefer that and
# fall back only where it does not exist.
_NEW_SHARD_MAP = hasattr(jax, "shard_map") and (
    "check_vma" in inspect.signature(jax.shard_map).parameters
)


def shard_map(f, mesh, in_specs, out_specs, check_rep: bool = True):
    """``shard_map`` across jax versions; ``check_rep`` is ``check_vma`` on jax >= 0.11."""
    if _NEW_SHARD_MAP:
        return jax.shard_map(
            f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=check_rep
        )
    from jax.experimental.shard_map import shard_map as _legacy

    return _legacy(f, mesh, in_specs, out_specs, check_rep=check_rep)


def shard_map_api() -> str:
    return "jax.shard_map(check_vma=)" if _NEW_SHARD_MAP else "jax.experimental.shard_map(check_rep=)"


# --- wire volume from the prepared tables --------------------------------------------------
def true_halo_cells(layout: Layout) -> int:
    """Cells in one local halo rim: ``2h(MLOC + NLOC) + 4h^2``. The reference volume."""
    h = layout.h
    return 2 * h * (layout.MLOC + layout.NLOC) + 4 * h * h


def wire_cells(transport: "Transport", tables) -> int | None:
    """Cells crossing the wire *per device* per exchange, or ``None`` if not reported.

    Optional ``Transport.wire_cells(tables) -> int``, computed from the prepared tables;
    FESOM's ``bench_halo_micro`` ``lanes`` dict. It is the larger of the two directions,
    which for ``padded``, ``coloured`` and ``ragged`` are equal, and for ``allgather`` is
    the **receive** side ``P*MLOC*NLOC`` -- its send side is only ``MLOC*NLOC`` and *falls*
    with P, but every device still has to take delivery of the whole field. Table-derived
    volume is the primary metric; the HLO byte count is the cross-check.
    """
    fn = getattr(transport, "wire_cells", None)
    return None if fn is None else int(fn(tables))


# --- GT4Py dispatch on JAX tracers (jax >= 0.11) ---------------------------------------------
def _patch_gt4py_tracer_dispatch() -> str:
    """Make GT4Py's ``singledispatch`` field constructor accept JAX tracers.

    GT4Py registers ``jax.Array`` with ``functools.singledispatch``. On jax >= 0.11 a
    tracer is still ``isinstance(t, jax.Array)`` but is no longer *dispatched* through
    it, so ``gtx.as_field`` works eagerly and raises ``NotImplementedError`` under
    ``grad``/``jit``/``scan``. Registering ``jax.core.Tracer`` explicitly restores it;
    on jax 0.6.2 the tracer already resolves and this is a no-op.
    """
    try:
        from gt4py.next import common
        from gt4py.next.embedded import nd_array_field as _ndf
    except ImportError:
        return "no gt4py"
    tracer = jax.core.Tracer
    if common._field.dispatch(tracer) is not common._field.dispatch(object):
        return "not needed"
    common._field.register(tracer, _ndf.JaxArrayField.from_array)
    common._connectivity.register(tracer, _ndf.JaxArrayConnectivityField.from_array)
    return "patched"


GT4PY_TRACER_DISPATCH = _patch_gt4py_tracer_dispatch()


# --- HLO collective accounting (nb03's helper, corrected for tuple-typed results) -----------
NAMES = ("collective-permute", "all-reduce", "all-gather", "all-to-all",
         "reduce-scatter", "collective-broadcast")

# Match the *defining* instruction only: a name, '=', the result type, the opcode, then '('.
# Matching the opcode anywhere on the line instead would also match every line that
# merely *consumes* the result, because GSPMD names instructions after their opcode
# ("%collective-permute.1"), so consumer lines mention it too. That mistake
# over-counts the compiler-generated path and not the shard_map path, which names
# instructions after JAX primitives ("%ppermute.6").
#
# The result type is captured whole rather than as one shape, because XLA:CPU compiles
# `lax.all_to_all(tiled=True)` into a TUPLE-typed instruction with one element per peer,
#     %all-to-all.2 = (f64[16]{0}, f64[16]{0}, f64[16]{0}, f64[16]{0}) all-to-all(...)
# and reading only the first element undercounts the volume by a factor P. From arity 6
# XLA also prints inline `/*index=5*/` markers inside that type, so the capture must be
# permissive; it stays anchored by the ' <opcode>(' that follows, which a line merely
# *consuming* the result never has (it says '%all-to-all.2)', never 'all-to-all(').
DEFN = re.compile(r"^\s*(?:ROOT\s+)?%[\w.\-]+\s*=\s*(.*?)\s"
                  r"(" + "|".join(NAMES) + r")(-start|-done)?\(", re.M)
_ELEM = re.compile(r"\[([0-9,<=\s]*)\]")


def _shapes(shape_text):
    """The dims of every array in an HLO result type, tuple or not, as clean strings."""
    out = [",".join(d.strip() for d in m.group(1).replace("<=", "").split(",") if d.strip())
           for m in _ELEM.finditer(shape_text)]
    return out or [""]  # a scalar result prints as 'f64[]'


def _cells(dims: str) -> int:
    n = 1
    for d in dims.split(","):
        if d:
            n *= int(d)
    return n


def _format(op, shapes):
    if len(shapes) == 1:
        return f"{op}[{shapes[0]}]"
    if len(set(shapes)) == 1:
        return f"{op}[{shapes[0]}]x{len(shapes)}"
    return op + "".join(f"[{s}]" for s in shapes)


def collectives_in(hlo_text):
    """Every collective instruction, as ``opcode[dims]`` (``opcode[dims]xK`` if tuple-typed).

    A tuple-typed result contributes *every* element, which is what a dense
    ``all_to_all`` actually puts on the wire.
    """
    out = []
    for m in DEFN.finditer(hlo_text):
        if m.group(3) == "-done":
            continue
        shapes = _shapes(m.group(1))
        # async '-start' returns (operand..., result...): only the largest buffer moves.
        if m.group(3) == "-start" and len(shapes) > 1:
            shapes = [max(shapes, key=_cells)]
        out.append(_format(m.group(2), shapes))
    return out


def bytes_moved(cs, itemsize=8):
    """Bytes in the results of ``collectives_in``; ``[dims]xK`` counts K times."""
    total = 0
    for c in cs:
        for m in re.finditer(r"\[([0-9,]*)\](?:x(\d+))?", c):
            total += _cells(m.group(1)) * int(m.group(2) or 1) * itemsize
    return total
