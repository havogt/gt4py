# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Collective instructions and their byte volume, read off compiled HLO text."""

from __future__ import annotations

import re

_COLLECTIVE_NAMES = (
    "collective-permute",
    "all-reduce",
    "all-gather",
    "all-to-all",
    "ragged-all-to-all",
    "reduce-scatter",
    "collective-broadcast",
)

# Match only the defining instruction, "%name = <type> <opcode>(": lines that merely
# consume the result mention the opcode too, because GSPMD names instructions after it
# ("%collective-permute.1"). The result type is captured whole because XLA:CPU compiles a
# tiled all_to_all into a tuple-typed instruction with one element per peer,
#     %all-to-all.2 = (f64[16]{0}, f64[16]{0}, f64[16]{0}, f64[16]{0}) all-to-all(...)
_DEFN = re.compile(
    r"^\s*(?:ROOT\s+)?%[\w.\-]+\s*=\s*(.*?)\s("
    + "|".join(_COLLECTIVE_NAMES)
    + r")(-start|-done)?\(",
    re.M,
)
_ELEM = re.compile(r"\[([0-9,<=\s]*)\]")


def _shapes(shape_text):
    """The dims of every array in an HLO result type, tuple or not, as clean strings."""
    out = [
        ",".join(d.strip() for d in m.group(1).replace("<=", "").split(",") if d.strip())
        for m in _ELEM.finditer(shape_text)
    ]
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
    """Every collective instruction, as ``opcode[dims]`` (``opcode[dims]xK`` if tuple-typed)."""
    out = []
    for m in _DEFN.finditer(hlo_text):
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
