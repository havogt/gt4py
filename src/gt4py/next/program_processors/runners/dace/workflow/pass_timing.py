# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in wall-time log of the dace translation, optimization and compilation steps.

Enabled by setting `GT4PY_DACE_PASS_TIMING_LOG` to a file path (or `-` for stderr). Every
record is flushed immediately, so a run killed by a time limit shows the step it was in.
Individual dace passes and `apply_transformations*` calls are only logged if they take at least
`GT4PY_DACE_PASS_TIMING_MIN_SECONDS` (default 1). `GT4PY_DACE_PASS_TIMING_STACK_INTERVAL`
(seconds) additionally logs the stack of the thread that enabled the log at that interval.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import os
import sys
import threading
import time
import traceback
import types
from typing import Any, Callable, Iterator, Optional, TextIO

import dace
from dace.codegen import codegen as dace_codegen, compiler as dace_compiler
from dace.sdfg import nodes as dace_nodes, propagation as dace_propagation
from dace.transformation import pass_pipeline as dace_ppl, transformation as dace_transformation
from dace.transformation.passes import pattern_matching as dace_pattern_matching


_LOG_ENV_VAR = "GT4PY_DACE_PASS_TIMING_LOG"
_STACK_INTERVAL_ENV_VAR = "GT4PY_DACE_PASS_TIMING_STACK_INTERVAL"
_MIN_SECONDS_ENV_VAR = "GT4PY_DACE_PASS_TIMING_MIN_SECONDS"

_stream: Optional[TextIO] = None
_lock = threading.RLock()
_t0 = time.perf_counter()
_min_seconds = 1.0
_depth = 0
# name -> [calls, seconds], only top-level calls are counted for recursive functions
_counters: collections.defaultdict[str, list[float]] = collections.defaultdict(lambda: [0, 0.0])
_active_counters: set[str] = set()


def enabled() -> bool:
    return _stream is not None


def log(message: str) -> None:
    if _stream is None:
        return
    with _lock:
        _stream.write(
            f"[{os.getpid()} {time.perf_counter() - _t0:10.1f}s] {'  ' * _depth}{message}\n"
        )
        _stream.flush()


def sdfg_stats(sdfg: dace.SDFG) -> str:
    n_nodes = n_states = n_nested = 0
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.SDFGState):
            n_states += 1
        else:
            n_nodes += 1
            if isinstance(node, dace_nodes.NestedSDFG):
                n_nested += 1
    top_level_maps = 0
    for state in sdfg.states():
        scope_dict = state.scope_dict()
        top_level_maps += sum(
            1
            for node in state.nodes()
            if isinstance(node, dace_nodes.MapEntry) and scope_dict[node] is None
        )
    return (
        f"nodes={n_nodes} states={n_states} nested_sdfgs={n_nested} top_level_maps={top_level_maps}"
    )


def _counter_snapshot() -> dict[str, tuple[float, float]]:
    return {k: (v[0], v[1]) for k, v in _counters.items()}


def _counter_delta(before: dict[str, tuple[float, float]]) -> str:
    parts = []
    for name, (calls, seconds) in sorted(_counters.items()):
        calls0, seconds0 = before.get(name, (0, 0.0))
        if calls != calls0:
            parts.append(f"{name}: {int(calls - calls0)}x {seconds - seconds0:.1f}s")
    return f" [{'; '.join(parts)}]" if parts else ""


@contextlib.contextmanager
def timed(
    name: str, sdfg: Optional[dace.SDFG] = None, quiet: bool = False
) -> Iterator[dict[str, Any]]:
    """
    Log `name` with its wall time; the caller may put a `result` into the yielded dict.

    A `quiet` record is logged only once it is done and only if it took at least
    `GT4PY_DACE_PASS_TIMING_MIN_SECONDS`.
    """
    global _depth
    if _stream is None:
        yield {}
        return
    info: dict[str, Any] = {}
    if not quiet:
        stats_before = f" {sdfg_stats(sdfg)}" if sdfg is not None else ""
        log(f"> {name}{stats_before}")
    counters = _counter_snapshot()
    start = time.perf_counter()
    if not quiet:
        _depth += 1
    try:
        yield info
    finally:
        if not quiet:
            _depth -= 1
        elapsed = time.perf_counter() - start
        if not quiet or elapsed >= _min_seconds:
            result = f" result={info['result']!r}" if "result" in info else ""
            stats_after = f" {sdfg_stats(sdfg)}" if sdfg is not None else ""
            log(
                f"{'=' if quiet else '<'} {name} {elapsed:.2f}s{result}{stats_after}"
                f"{_counter_delta(counters)}"
            )


def _caller() -> str:
    frame: Optional[types.FrameType] = sys._getframe(2)
    for _ in range(8):
        if frame is None:
            break
        filename = frame.f_code.co_filename
        if "gt4py" in filename and "pass_timing" not in filename and "contextlib" not in filename:
            return f"{os.path.basename(filename)}:{frame.f_lineno}"
        frame = frame.f_back
    return "?"


def _transformation_names(xforms: Any) -> str:
    if not isinstance(xforms, (list, tuple)):
        xforms = [xforms]
    return "+".join(x.__name__ if isinstance(x, type) else type(x).__name__ for x in xforms)


def _wrap_sdfg_method(method_name: str) -> None:
    original = getattr(dace.SDFG, method_name)

    @functools.wraps(original)
    def wrapper(self: dace.SDFG, xforms: Any, *args: Any, **kwargs: Any) -> Any:
        with timed(
            f"{method_name}({_transformation_names(xforms)}) @{_caller()}", self, quiet=True
        ) as info:
            info["result"] = original(self, xforms, *args, **kwargs)
        return info["result"]

    setattr(dace.SDFG, method_name, wrapper)


def _counted(name: str, original: Callable) -> Callable:
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if name in _active_counters:
            return original(*args, **kwargs)
        _active_counters.add(name)
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            _active_counters.discard(name)
            _counters[name][0] += 1
            _counters[name][1] += time.perf_counter() - start

    return wrapper


def _wrap_match_patterns() -> None:
    original = dace_pattern_matching.match_patterns

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Iterator[Any]:
        sweeps, matches = _counters["match_patterns sweeps"], _counters["match_patterns matches"]
        sweeps[0] += 1
        matching = original(*args, **kwargs)
        while True:
            start = time.perf_counter()
            try:
                match = next(matching)
            except StopIteration:
                return
            finally:
                sweeps[1] += time.perf_counter() - start
            matches[0] += 1
            yield match

    dace_pattern_matching.match_patterns = wrapper


def _wrap_apply_to() -> None:
    original = dace_transformation.PatternTransformation.apply_to.__func__

    def wrapper(cls: type, *args: Any, **kwargs: Any) -> Any:
        return _counted(f"apply_to({cls.__name__})", functools.partial(original, cls))(
            *args, **kwargs
        )

    dace_transformation.PatternTransformation.apply_to = classmethod(wrapper)


def _all_subclasses(cls: type) -> Iterator[type]:
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_subclasses(sub)


def _wrap_passes() -> None:
    for cls in {dace_ppl.Pass, *_all_subclasses(dace_ppl.Pass)}:
        original = cls.__dict__.get("apply_pass")
        if original is None or getattr(original, "_gt4py_pass_timing", False):
            continue

        def make_wrapper(original: Callable, cls_name: str) -> Callable:
            @functools.wraps(original)
            def wrapper(self: Any, sdfg: dace.SDFG, *args: Any, **kwargs: Any) -> Any:
                with timed(f"pass {cls_name} @{_caller()}", quiet=True) as info:
                    info["result"] = result = original(self, sdfg, *args, **kwargs)
                    if result is not None and not isinstance(result, (int, float, bool)):
                        info["result"] = type(result).__name__
                return result

            wrapper._gt4py_pass_timing = True  # type: ignore[attr-defined]
            return wrapper

        cls.apply_pass = make_wrapper(original, cls.__name__)


def _wrap_module_function(module: Any, name: str, label: Optional[str] = None) -> None:
    original = getattr(module, name)
    if getattr(original, "_gt4py_pass_timing", False):
        return

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        sdfg = next((a for a in (*args, *kwargs.values()) if isinstance(a, dace.SDFG)), None)
        with timed(label or name, sdfg) as info:
            result = original(*args, **kwargs)
            if isinstance(result, (int, float, bool, type(None))):
                info["result"] = result
        return result

    wrapper._gt4py_pass_timing = True  # type: ignore[attr-defined]
    setattr(module, name, wrapper)


def _log_stacks_periodically(thread_id: int, interval: float) -> None:
    while True:
        time.sleep(interval)
        frame = sys._current_frames().get(thread_id)
        if frame is None:
            return
        log("stack of the compiling thread:\n" + "".join(traceback.format_stack(frame)))


def enable_from_env() -> None:
    """Install the timing wrappers once, if `GT4PY_DACE_PASS_TIMING_LOG` is set."""
    global _stream, _min_seconds
    target = os.environ.get(_LOG_ENV_VAR)
    if _stream is not None or not target:
        return
    _stream = sys.stderr if target == "-" else open(target, "a", buffering=1)
    _min_seconds = float(os.environ.get(_MIN_SECONDS_ENV_VAR, _min_seconds))
    log(f"pass timing enabled (pid {os.getpid()}, quiet records >= {_min_seconds}s)")

    from gt4py.next.program_processors.runners.dace import transformations as gtx_transformations
    from gt4py.next.program_processors.runners.dace.transformations import (
        auto_optimize as gtx_auto_optimize,
    )

    for method_name in (
        "apply_transformations",
        "apply_transformations_repeated",
        "apply_transformations_once_everywhere",
    ):
        _wrap_sdfg_method(method_name)
    _wrap_apply_to()
    _wrap_match_patterns()
    original_hash = dace.SDFG.hash_sdfg

    def hash_sdfg(self: dace.SDFG, *args: Any, **kwargs: Any) -> Any:
        caller = _caller()
        with timed(f"hash_sdfg @{caller}", quiet=not caller.startswith("auto_optimize.py")):
            return original_hash(self, *args, **kwargs)

    dace.SDFG.hash_sdfg = hash_sdfg
    _wrap_passes()
    dace_propagation.propagate_memlets_sdfg = _counted(
        "propagate_memlets_sdfg", dace_propagation.propagate_memlets_sdfg
    )
    for name in dir(gtx_transformations):
        if name.startswith("gt_") and callable(getattr(gtx_transformations, name)):
            _wrap_module_function(gtx_transformations, name)
    for name in dir(gtx_auto_optimize):
        if name.startswith("_gt_auto_") and callable(getattr(gtx_auto_optimize, name)):
            _wrap_module_function(gtx_auto_optimize, name)
    _wrap_module_function(dace_codegen, "generate_code", "dace codegen")
    _wrap_module_function(dace_compiler, "configure_and_compile", "dace C++/CUDA compile")

    if interval := float(os.environ.get(_STACK_INTERVAL_ENV_VAR, "0")):
        threading.Thread(
            target=_log_stacks_periodically,
            args=(threading.get_ident(), interval),
            daemon=True,
            name="gt4py-dace-pass-timing-stacks",
        ).start()
