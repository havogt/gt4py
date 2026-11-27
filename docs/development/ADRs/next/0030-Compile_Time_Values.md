---
tags: [frontend, iterator, embedded, jax, torch]
---

# Compile-Time Values

- **Status**: valid
- **Authors**: Hannes Vogt (@havogt)
- **Created**: 2026-11-27
- **Updated**: 2026-11-27

In the context of field operators that index, shift or select by a variable, e.g. `f(I + m)` for a periodic halo of size `m`, facing that some values determine a type, a domain, a shape or control flow, we decided that such a value may be any expression whose value is known at compile time (a literal, a closed-over constant, or an expression of the parameters in `static_params`), to write grid-dependent operators without hardcoding values. We accept one compilation per distinct value, and errors at the first call rather than at definition time.

## Context

A value can be needed before run time for three reasons:

- **It determines a type.** `t[i]` on a tuple `(a, b)` of different types has the type of the `i`-th element. `I + 0.5` shifts to the staggered dimension (ADR 0026), `I + 1` does not.
- **It determines a domain.** Domain inference (`trace_shifts`, `SymbolicDomain.translate`) translates the accessed domain by the shift distance, and an offset index (`Ioff[i]`, `V2E[i]`) selects the shift distance or the connectivity column that is read.
- **It determines a shape or control flow, only under a tracing compiler.** The compiled backends take program domains and `concat_where` bounds at run time (ADR 0020). The tracing embedded backends (`jax_jit`, `torch_compile`) compute domains in Python at trace time, so a traced scalar can not determine a program domain, a `concat_where` bound or a Python `if`.

JAX draws the same line under `jax.jit`. A traced value may decide _where_ to read, e.g. the start of `lax.dynamic_slice`, `jnp.roll` or `x[k]`, but not a _shape_, e.g. the size of `lax.dynamic_slice`, `jnp.zeros(n)` or a slice `x[k:]`, nor the structure of a pytree. A static argument lifts all of these restrictions, because `jax.jit` traces once per binding of the static arguments.

## Decision

A value that determines a type, a domain, a shape or control flow must be known at compile time. Literals, closed-over constants and `static_params` all qualify; they are interchangeable. `static_params` is the same escape hatch as JAX's static arguments: `past_to_itir` substitutes the values as literals before any GTIR pass, and constant folding reduces an expression such as `m - 1` to a literal.

This covers:

| use                                     | determines                                             |
| --------------------------------------- | ------------------------------------------------------ |
| Cartesian shift distance, `f(I + m)`    | domain                                                 |
| offset index, `f(Ioff[m])`, `f(V2E[m])` | domain                                                 |
| tuple index, `t[m]`                     | type                                                   |
| half-integer shift distance, `f(I + h)` | type                                                   |
| program domain, `concat_where` bound    | shape, under `jax_jit` and `torch_compile` only        |
| Python `if` on a scalar                 | control flow, under `jax_jit` and `torch_compile` only |

Types and dimensions, e.g. the target of `astype`, the dimensions of `broadcast` or the `axis` of a scan, are static by construction. Every other scalar use, e.g. arithmetic, `astype(m, float64)` or a broadcast scalar mask in `where`, is a run time value on every backend.

A value that is not known at compile time is an error at the first call, which names the parameter and suggests `with_static_params`; `NonStaticShiftError` is the one for shift distances.

### Implementation

- **Value-independent types.** For an integral shift distance the type of the result does not depend on the value, so the frontend accepts any integral expression and lowers it symbolically. After static substitution and constant folding, `CanonicalizeShiftOffsets` turns the distance into an `OffsetLiteral`, or raises `NonStaticShiftError` if it still refers to a scalar argument. `jax_jit` and `torch_compile` raise the same error for a traced distance. Offset indices have value-independent types too and are to take the same path; for now the frontend still requires a literal (or closed-over constant) for them.
- **Value-dependent types.** A tuple index and a half-integer shift distance determine the type, so they need type deduction after static binding, once per specialization rather than once per definition. Until then, the frontend requires a literal (or closed-over constant) for them.

## Consequences

- Each distinct compile-time value is one compilation, as for every static parameter.
- Embedded NumPy and CuPy execution compute domains from the actual values and need no static value for a domain. The compiled backends and the tracing embedded backends do.
- Whether a value is known at compile time is only known once the call is bound, so the error is raised at the first call, not when the field operator is defined. With type deduction per specialization, this will also hold for type errors of value-dependent uses.

## Alternatives considered

### A static marker in the parameter's type

E.g. `m: Static[int32]`. Rejected: whether a value is static is a property of the call binding, not of the operator. The same field operator is called with a static value from one program and with a literal from another, and the program decides which of its parameters are static with `with_static_params`, as for every other compile-time specialization.

### Domain inference with symbolic shift distances

`SymbolicRange.translate` is `start + distance`, so a symbolic distance is not a conceptual obstacle for the compiled backends. It is not pursued: `jax_jit` and `torch_compile` would still need the value, and it would not help the uses that determine a type.
