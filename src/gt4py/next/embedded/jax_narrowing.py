# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Compute every intermediate of a JAX function only on the index window its consumers read.

Embedded execution evaluates each field expression on the largest domain its inputs allow. When the
result is later read through slices (shifted accesses, cropping to the output domain), XLA narrows a
producer to the window it is read on only if it has a single consumer. `narrow` does it for all
producers: it traces the function to a jaxpr, propagates the read windows backwards from the outputs,
and evaluates the jaxpr with every value restricted to its window.

Windows are propagated through elementwise primitives, `slice`, `concatenate`, `broadcast_in_dim` and
nested jaxprs of `jit`; every other primitive reads its inputs in full.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, Sequence, cast

import jax
import jax.extend.core as jex_core
from jax import lax


Window = tuple[tuple[int, int], ...]

# Dimensions up to this extent (e.g. colors, neighbors) are never narrowed: cutting them splits
# concatenations into separately computed pieces that XLA materializes.
_MIN_NARROWED_EXTENT = 8

_ELEMENTWISE: frozenset[str] = frozenset(
    {
        "abs", "acos", "acosh", "add", "and", "asin", "asinh", "atan", "atan2", "atanh", "cbrt",
        "ceil", "clamp", "conj", "convert_element_type", "copy", "copy_p", "cos", "cosh",
        "digamma", "div", "eq", "erf", "erf_inv", "erfc", "exp", "exp2", "expm1", "floor", "ge",
        "gt", "imag", "integer_pow", "is_finite", "le", "lgamma", "log", "log1p", "logistic",
        "lt", "max", "min", "mul", "ne", "neg", "nextafter", "not", "or", "population_count",
        "pow", "real", "reduce_precision", "rem", "round", "rsqrt", "select_n",
        "shift_left", "shift_right_arithmetic", "shift_right_logical", "sign", "sin", "sinh",
        "sqrt", "square", "sub", "tan", "tanh", "xor",
    }
)  # fmt: skip
_NESTED_JAXPR: frozenset[str] = frozenset({"jit", "pjit", "closed_call"})


def narrow(fun: Callable) -> Callable:
    """Wrap `fun` so that all its intermediates are computed only where they are read."""

    @functools.wraps(fun)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        leaves, in_tree = jax.tree.flatten((args, kwargs))

        def flat_fun(*flat_args: Any) -> Any:
            a, k = jax.tree.unflatten(in_tree, flat_args)
            return fun(*a, **k)

        closed, out_shape = jax.make_jaxpr(flat_fun, return_shape=True)(*leaves)
        jaxpr = closed.jaxpr
        needed = _needed_windows(jaxpr, [_full(v) for v in jaxpr.outvars])
        outs = _evaluate(jaxpr, closed.consts, [(x, _zero(x)) for x in leaves], needed)
        return jax.tree.unflatten(jax.tree.structure(out_shape), [x for x, _ in outs])

    return wrapper


def _shape(v: Any) -> tuple[int, ...]:
    return tuple(v.aval.shape)


def _full(v: Any) -> Window:
    return tuple((0, n) for n in _shape(v))


def _zero(x: Any) -> tuple[int, ...]:
    return (0,) * len(getattr(x, "shape", ()))


def _union(a: Optional[Window], b: Optional[Window]) -> Optional[Window]:
    if a is None:
        return b
    if b is None:
        return a
    return tuple((min(lo_a, lo_b), max(hi_a, hi_b)) for (lo_a, hi_a), (lo_b, hi_b) in zip(a, b))


def _is_var(v: Any) -> bool:
    return isinstance(v, jex_core.Var) and not isinstance(v, jex_core.DropVar)


def _is_elementwise(eqn: Any) -> bool:
    if eqn.primitive.name not in _ELEMENTWISE or len(eqn.outvars) != 1:
        return False
    out_shape = _shape(eqn.outvars[0])
    return all(_shape(v) in (out_shape, ()) for v in eqn.invars)


def _needed_windows(jaxpr: Any, out_windows: Sequence[Optional[Window]]) -> dict[Any, Window]:
    needed: dict[Any, Window] = {}

    def use(v: Any, window: Optional[Window]) -> None:
        if _is_var(v) and window is not None:
            window = tuple(
                w if n > _MIN_NARROWED_EXTENT else (0, n) for w, n in zip(window, _shape(v))
            )
            needed[v] = _union(needed.get(v), window)  # type: ignore[assignment]

    for v, window in zip(jaxpr.outvars, out_windows):
        use(v, window)
    for eqn in reversed(jaxpr.eqns):
        outs = [needed.get(v) if _is_var(v) else None for v in eqn.outvars]
        if all(w is None for w in outs) and not eqn.effects:
            continue
        for v, window in zip(eqn.invars, _input_windows(eqn, outs)):
            use(v, window)
    return needed


def _input_windows(eqn: Any, outs: list[Optional[Window]]) -> list[Optional[Window]]:
    name, params = eqn.primitive.name, eqn.params
    (out,) = outs if len(outs) == 1 else (None,)
    if out is not None and _is_elementwise(eqn):
        return [out if _shape(v) == _shape(eqn.outvars[0]) else _full(v) for v in eqn.invars]
    if out is not None and name == "slice" and _unit_strides(params):
        return [tuple((lo + s, hi + s) for (lo, hi), s in zip(out, params["start_indices"]))]
    if out is not None and name == "concatenate":
        return _concatenate_input_windows(eqn, out)
    if out is not None and name == "broadcast_in_dim":
        return [_broadcast_operand_window(eqn, out), *(_full(v) for v in eqn.invars[1:])]
    if name in _NESTED_JAXPR:
        inner = params["jaxpr"].jaxpr
        inner_needed = _needed_windows(inner, outs)
        return [inner_needed.get(v) for v in inner.invars]
    return [_full(v) for v in eqn.invars]


def _unit_strides(params: dict[str, Any]) -> bool:
    return params["strides"] is None or all(s == 1 for s in params["strides"])


def _concatenate_segments(eqn: Any) -> list[tuple[int, int]]:
    axis = eqn.params["dimension"]
    segments, start = [], 0
    for v in eqn.invars:
        segments.append((start, start + _shape(v)[axis]))
        start += _shape(v)[axis]
    return segments


def _concatenate_input_windows(eqn: Any, out: Window) -> list[Optional[Window]]:
    axis = eqn.params["dimension"]
    lo, hi = out[axis]
    windows: list[Optional[Window]] = []
    for s, e in _concatenate_segments(eqn):
        a, b = max(lo, s), min(hi, e)
        windows.append(None if a >= b else (*out[:axis], (a - s, b - s), *out[axis + 1 :]))
    return windows


def _broadcast_operand_window(eqn: Any, out: Window) -> Window:
    operand = eqn.invars[0]
    out_shape = _shape(eqn.outvars[0])
    return tuple(
        (0, 1) if n == 1 and out_shape[d] != 1 else out[d]
        for n, d in zip(_shape(operand), eqn.params["broadcast_dimensions"])
    )


def _cut(value: tuple[Any, tuple[int, ...]], window: Window) -> Any:
    array, offset = value
    shape = tuple(getattr(array, "shape", ()))
    if not shape:
        return array
    start = [lo - o for (lo, _), o in zip(window, offset)]
    limit = [hi - o for (_, hi), o in zip(window, offset)]
    if all(s == 0 for s in start) and tuple(limit) == shape:
        return array
    return lax.slice(array, start, limit)


def _evaluate(
    jaxpr: Any,
    consts: Sequence[Any],
    args: Sequence[Optional[tuple[Any, tuple[int, ...]]]],
    needed: dict[Any, Window],
) -> list[tuple[Any, tuple[int, ...]]]:
    env: dict[Any, tuple[Any, tuple[int, ...]]] = {}

    def read(v: Any) -> Optional[tuple[Any, tuple[int, ...]]]:
        if isinstance(v, jex_core.Literal):
            return v.val, _zero(v.val)
        return env.get(v)  # `None` for operands that are not read, e.g. concatenated pieces

    for v, c in zip(jaxpr.constvars, consts):
        env[v] = (c, _zero(c))
    for v, a in zip(jaxpr.invars, args):
        if a is not None:
            env[v] = a

    for eqn in jaxpr.eqns:
        outs = [needed.get(v) if _is_var(v) else None for v in eqn.outvars]
        if all(w is None for w in outs) and not eqn.effects:
            continue
        for v, value in zip(eqn.outvars, _evaluate_eqn(eqn, outs, [read(v) for v in eqn.invars])):
            if _is_var(v):
                env[v] = value
    return [cast(tuple[Any, tuple[int, ...]], read(v)) for v in jaxpr.outvars]


def _evaluate_eqn(
    eqn: Any, outs: list[Optional[Window]], ins: list[Any]
) -> list[tuple[Any, tuple[int, ...]]]:
    name, params, prim = eqn.primitive.name, eqn.params, eqn.primitive
    (out,) = outs if len(outs) == 1 else (None,)
    if out is not None and _is_elementwise(eqn):
        out_shape = _shape(eqn.outvars[0])
        args = [
            _cut(value, out) if _shape(v) == out_shape else value[0]
            for v, value in zip(eqn.invars, ins)
        ]
        return [(prim.bind(*args, **params), tuple(lo for lo, _ in out))]
    if out is not None and name == "slice" and _unit_strides(params):
        window = tuple((lo + s, hi + s) for (lo, hi), s in zip(out, params["start_indices"]))
        return [(_cut(ins[0], window), tuple(lo for lo, _ in out))]
    if out is not None and name == "concatenate":
        pieces = [
            _cut(value, window)
            for value, window in zip(ins, _concatenate_input_windows(eqn, out))
            if window is not None
        ]
        result = pieces[0] if len(pieces) == 1 else lax.concatenate(pieces, params["dimension"])
        return [(result, tuple(lo for lo, _ in out))]
    if out is not None and name == "broadcast_in_dim":
        operand = _cut(ins[0], _broadcast_operand_window(eqn, out))
        shape = tuple(hi - lo for lo, hi in out)
        result = prim.bind(operand, *(v for v, _ in ins[1:]), **{**params, "shape": shape})
        return [(result, tuple(lo for lo, _ in out))]
    if name in _NESTED_JAXPR:
        closed = params["jaxpr"]
        inner_needed = _needed_windows(closed.jaxpr, outs)
        return _evaluate(closed.jaxpr, closed.consts, ins, inner_needed)
    args = [_cut(value, _full(v)) if _is_var(v) else value[0] for v, value in zip(eqn.invars, ins)]
    results = prim.bind(*args, **params)
    if not prim.multiple_results:
        results = [results]
    return [(r, _zero(r)) for r in results]
