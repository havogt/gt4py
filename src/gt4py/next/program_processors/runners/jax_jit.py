# GT4Py - GridTools Framework
#
# Copyright (c) 2014-2024, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""JAX JIT backend for GT4Py field operators.

This backend wraps the embedded execution of field operators with ``jax.jit``,
enabling XLA compilation of GT4Py computations. It works by tracing through
the embedded field operator execution (which already uses ``jax.numpy``
operations on ``JaxArrayField`` pytrees) and compiling the resulting
computation graph.

Supported:
    - Direct calls to ``@field_operator`` (as a program, with ``out=``)
      on Cartesian grids.
    - Scan operators (loops are unrolled during tracing).

Not yet supported:
    - ``@program`` level JIT (programs call multiple field operators;
      supporting this requires changes to the ``Program.__call__`` dispatch).
    - Unstructured mesh operations (neighbor shifts, ``neighbor_sum``) because
      the embedded path uses ``jnp.nonzero`` for connectivity indexing, which
      requires concrete shapes incompatible with JAX tracing.
    - Scalar arguments to field operators (JAX tracers don't satisfy
      ``core_defs.is_scalar_type``).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from gt4py.next import backend as next_backend, common, utils
from gt4py.next.embedded import operators as embedded_operators
from gt4py.next.embedded.nd_array_field import jnp
from gt4py.next.ffront import stages as ffront_stages
from gt4py.next.otf import arguments, stages


if jnp is not None:
    import jax


def _make_executable(
    definition_stage: ffront_stages.DSLFieldOperatorDef,
    offset_provider: common.OffsetProvider,
) -> stages.ExecutableProgram:
    """
    Build an ``ExecutableProgram`` that runs the field operator under ``jax.jit``.

    The returned callable matches the protocol used by ``CompiledProgramsPool``::

        executable(*args, offset_provider=...)

    where ``args`` contains both input fields and the output field(s) as the
    last positional argument (the ``out`` parameter from the implicit program
    generated for a field operator).
    """
    definition = definition_stage.definition
    attributes = definition_stage.attributes

    # Build the right kind of embedded operator (regular vs scan).
    if attributes and all(k in attributes for k in ("init", "axis", "forward")):
        _make_op: Any = lambda: embedded_operators.ScanOperator(
            definition, attributes["forward"], attributes["init"], attributes["axis"]
        )
    else:
        _make_op = lambda: embedded_operators.EmbeddedOperator(definition)

    # Create a *pure function* that jax.jit can trace.
    #
    # Inside the trace, JaxArrayField objects carry abstract tracers as their
    # backing ndarrays.  The embedded operator uses jnp operations, which JAX
    # traces normally.  ``JaxArrayField.__setitem__`` stores the result tracer
    # on the Python object via ``object.__setattr__``, so after the call the
    # output field's ``.ndarray`` *is* the output tracer.
    #
    # ``offset_provider`` is captured by the closure; its connectivity arrays
    # become constants in the compiled XLA program (appropriate because the
    # ``CompiledProgramsPool`` already caches per offset-provider identity).
    def _pure(input_args: tuple, out: Any) -> Any:
        op = _make_op()
        embedded_operators.field_operator_call(
            op,
            input_args,
            {"out": out, "offset_provider": offset_provider},
        )
        return out

    jitted_pure = jax.jit(_pure)

    def _executable(
        *args: Any,
        offset_provider: Any,  # noqa: ARG001 [unused-function-argument]  already captured in _pure
        **kwargs: Any,  # noqa: ARG001 [unused-function-argument]
    ) -> None:
        # After arg canonicalization the last positional arg is ``out``.
        input_args = args[:-1]
        out = args[-1]

        result_out = jitted_pure(input_args, out)

        # Copy the concrete result arrays back into the *original* output
        # fields so the caller sees the updated data.
        flat_orig = utils.flatten_nested_tuple((out,)) if isinstance(out, tuple) else (out,)
        flat_result = (
            utils.flatten_nested_tuple((result_out,))
            if isinstance(result_out, tuple)
            else (result_out,)
        )
        for orig, result in zip(flat_orig, flat_result):
            if hasattr(orig, "_ndarray"):
                object.__setattr__(orig, "_ndarray", result.ndarray)

    return _executable


@dataclasses.dataclass(frozen=True)
class JaxJitBackend(next_backend.Backend):
    """
    Backend that compiles GT4Py field operators via ``jax.jit``.

    Instead of lowering through ITIR and generating C++/CUDA code, this
    backend reuses the embedded (Python) execution path — which already
    operates on ``JaxArrayField`` pytrees backed by ``jax.numpy`` — and
    wraps it with ``jax.jit`` so that JAX's XLA compiler can fuse and
    optimise the computation.

    Only direct field-operator calls (``field_op(..., out=..., offset_provider=...)``)
    are supported.  ``@program``-level compilation is not yet implemented.

    Usage::

        @gtx.field_operator(backend=jax_jit.run_jax_jit)
        def add(a: Field[[IDim], float64], b: Field[[IDim], float64]) -> Field[[IDim], float64]:
            return a + b

        add(inp1, inp2, out=result, offset_provider={})
    """

    def compile(  # type: ignore[override]  # intentionally wider input type
        self,
        program: ffront_stages.DSLFieldOperatorDef | ffront_stages.DSLProgramDef,
        compile_time_args: arguments.CompileTimeArgs,
    ) -> stages.ExecutableProgram:
        if not isinstance(program, ffront_stages.DSLFieldOperatorDef):
            raise NotImplementedError(
                f"JaxJitBackend only supports direct field-operator calls, "
                f"got {type(program).__name__}.  Use embedded execution for @program."
            )
        return _make_executable(program, compile_time_args.offset_provider)


# ---------------------------------------------------------------------------
# Pre-built backend instance
# ---------------------------------------------------------------------------
if jnp is not None:
    run_jax_jit = JaxJitBackend(
        name="jax_jit",
        executor=lambda inp: None,  # unused — compile() is overridden
        allocator=jnp,  # type: ignore[arg-type]  # jnp is a valid array-namespace allocator
        transforms=next_backend.DEFAULT_TRANSFORMS,
    )
else:
    run_jax_jit = None  # type: ignore[assignment]
