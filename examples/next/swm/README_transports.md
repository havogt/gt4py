# Halo-exchange transports for the sharded SWM

FESOM2-JAX (arXiv:2608.01546, Sect. 2.2-2.3) writes the halo exchange *in JAX*, from
primitives whose transposes JAX already knows, so the distributed adjoint comes by
composition instead of by hand. This directory does the same on the structured SWM and
compares transports by adjoint exactness, wire volume and time.

Files:

| file | what |
|---|---|
| `halo_transports.py` | `Layout`, the `Transport` protocol, the registry, the `shard_map` and GT4Py-tracer shims, `collectives_in` / `bytes_moved` / `wire_cells` |
| `transport_allgather.py` | all-gather broadcast, registered as `"allgather"` -- the oracle |
| `transport_padded.py` | slot-padded dense `all_to_all`, registered as `"padded"` |
| `transport_coloured.py` | `ppermute` rounds, registered as `"coloured8"` (K=8) and `"coloured2ph"` (K=4) |
| `transport_ragged.py` | `ragged_all_to_all`, registered as `"ragged"`, plus its CPU-runnable emulation `"ragged_emul"` |
| `swm_sharded.py` | the sharded model, the single-device reference, and the test battery + CLI |
| `bench_transports.py` | the battery over a (transport, layout) grid -> one JSON row each; `--table` renders them |
| `santis_bench.sbatch` | Slurm template for CSCS Alps / Santis (GH200); untested on a cluster |
| `requirements-santis.txt` | primary Santis environment (jax 0.11.1 + CUDA 12, aarch64) |
| `requirements-santis-jax062.txt` | fallback: the laptop-proven jax 0.6.2 pin |
| `nb05_halo_transports.ipynb` | the comparison notebook: FESOM's design, the five transports, the adjoint, the two XLA findings |

## Running the battery

```
cd <gt4py>
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
    ./.venv/bin/python examples/next/swm/swm_sharded.py --transport allgather --layout 2x2 --steps 10
```

`--layout RxxRy` with `Rx*Ry <=` the device count, `16 % Rx == 0`, `16 % Ry == 0`. Exit
status is 0 iff every test passes. Every test is run under a guard: a transport that will
not compile reports `status: compile_error: ...` on that row and the rest of the table
still prints.

`--transport` resolves any registered name, including the ones that do not match their
module's file name (`coloured8`, `coloured2ph` live in `transport_coloured.py`;
`ragged_emul` in `transport_ragged.py`): the registry tries `transport_<name>` first and,
on a miss, imports every `transport_*.py` sibling and retries. An unknown name still gets
a `KeyError` listing what is registered.

Neither `swm_sharded.py` nor `bench_transports.py` pins `JAX_PLATFORMS` or the fake-device
flag; those belong on the command line. The device count comes from `jax.devices()`.

### The whole grid at once

```
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
    ./.venv/bin/python examples/next/swm/bench_transports.py \
        --transports all --layouts 1x1,2x1,1x2,2x2,4x2,2x4 --steps 10 --out results.jsonl

./.venv/bin/python examples/next/swm/bench_transports.py --table results.jsonl
```

One JSON row per (transport, layout), appended, each carrying the environment it was
measured in (jax version, platform, device kind, device and process counts, which
`shard_map` API, whether the GT4Py tracer shim fired) alongside every T0-T8 scalar, both
volume metrics, the collective counts and the timings. Laptop and cluster files therefore
concatenate, and `--table` renders any of them. Exit status is 1 while `ragged` is in the
set, because `ragged` cannot compile on CPU.

## Interface contract for a new transport

Create `transport_<name>.py` in this directory. It must import `halo_transports` and call
`register(...)` at import time; `get_transport("<name>")` imports it lazily by name, so a
module that does not exist or does not import cannot break the others.

```python
from halo_transports import Layout, register

class MyTransport:
    name = "mine"                       # must equal the <name> in the file name

    def prepare(self, layout: Layout):  # once, outside jit; returns anything
        return {...}                    # index arrays, permutations, slot maps, ...

    def exchange(self, a_local, tables, axis_name: str):
        ...                             # inside the shard_map body
        return a_refreshed

    def wire_cells(self, tables) -> int:   # optional
        return ...                         # cells this device puts on the wire

register(MyTransport())
```

`wire_cells` is optional but expected: the cells that cross the wire **per device** per
exchange, computed from the prepared tables. It is the **primary** volume metric (see
"Wire volume" below). Report the larger of the two directions: for `padded`, `coloured`
and `ragged` they are equal (`P*pad_slot`, the sum of the round slot widths, `send_max`),
but `allgather` is asymmetric -- it *sends* only `MLOC*NLOC`, which falls with `P`, and
*receives* `P*MLOC*NLOC`, which does not. `allgather.wire_cells` reports the receive
side, because taking delivery of the whole field is what makes it the O(P) transport.

`exchange` receives this device's `(MLOC+2h, NLOC+2h)` jnp array, with the interior at
`[h:-h, h:-h]`, and must return the same shape with **all** halo cells refreshed from the
owning neighbours -- four faces **and** four corners, periodic in both directions. It must
be pure jnp built only from `jax.lax` collectives and array ops: **no `pure_callback`, no
`custom_vjp`**, so that `jax.vjp` transposes it automatically. It must not depend on the
values in the input halo (the transports here overwrite the whole local box, interior
included, which is what makes the forward bit-equal to `np.pad(interior, h, "wrap")`).

### `Layout`

`Layout(M, N, Rx, Ry, h=1)`, all static, all NumPy, all computed outside jit:

- `MLOC = M // Rx`, `NLOC = N // Ry`, `P = Rx * Ry`, `local_shape = (MLOC+2h, NLOC+2h)`
- `coords(r) -> (rx, ry) = divmod(r, Ry)`, `rank(rx, ry) -> (rx % Rx) * Ry + (ry % Ry)`
- `origin(r)`, `block_slice(r)` -- rank `r` owns global rows `[rx*MLOC, (rx+1)*MLOC)` and
  columns `[ry*NLOC, (ry+1)*NLOC)`. This is exactly `halo_lib.Decomposition`'s ordering and
  `swm_ghex_2d.py`'s; the battery asserts it before using `halo_lib` as an oracle.
- `rx`, `ry` -- `(P,)` int32 coordinate tables
- `neighbour_tables` -- `{"E": (P,) int32, "W": ..., "N", "S", "NE", "NW", "SE", "SW"}`;
  axis 0 is `I` (x, E/W), axis 1 is `J` (y, N/S), so `E = (+1, 0)`, `N = (0, +1)`
- `neighbour(dx, dy) -> (P,) int32`
- `neighbours(r) -> {"E": rank, ...}` for one rank -- use it to assert that each round of a
  coloured/ppermute schedule is a partial permutation (no device sends or receives twice)
- `owner_lane() -> (src_dev, src_lane)`, both `(P, MLOC+2h, NLOC+2h)` int32: for every local
  cell of every rank, the owning rank and that cell's index in the owner's row-major
  flattened `(MLOC, NLOC)` interior. This is the FESOM all-gather index pair.

Blocking helpers (rank-major stacks, the shape `shard_map` wants with `in_specs=P("d")`):
`block(g, layout)` `(M,N) -> (P*MLOC, NLOC)`, `unblock`, `block_halo`/`unblock_halo` for
`(P, MLOC+2h, NLOC+2h) <-> (P*(MLOC+2h), NLOC+2h)`.

### The oracle

`allgather` is the oracle. `swm_sharded.oracle_gate(transport, layout)` is the gate every
new transport must pass, with FESOM's criteria: forward `np.array_equal` against
`allgather` (bit equality, not a tolerance) and, for `sum(w * exchange(x))` with random
`w`, `max|g - g_ref| / max|g_ref| < 1e-12`. It runs as **T0** of the battery.

## The model

`sharded_forward(transport, layout, u0, v0, p0, n_steps)` is one jitted program:
`shard_map` over the 1-D device axis `"d"`, `in_specs = out_specs = P("d")`, inputs and
outputs the rank-major stack of interior blocks. The body zero-pads to the halo shape,
calls `exchange`, wraps the local arrays as JAX-backed GT4Py fields on
`gtx.domain({I: (-1, MLOC+1), J: (-1, NLOC+1)})`, and runs `operators.timestep.definition`
with `M=MLOC, N=NLOC`: first step forward Euler with `dt` and `alpha=0`, then leapfrog with
`2*dt` under `jax.lax.scan`. GT4Py embedded execution survives `shard_map` **and** `scan`
on jax 0.6.2 -- no pure-jnp fallback was needed.

`timestep` calls `make_periodic` with the *local* sizes, so for `P > 1` the halos it writes
are wrong. They are never read: the next iteration's exchange overwrites `u, v, p`, and the
`old` fields' halos never reach an interior cell, because `uold_new = u + alpha*(unew - 2u
+ uold)` is evaluated on the pre-`make_periodic` `unew`, whose domain is the interior. Same
argument as `swm_ghex_2d.py`.

Parameters match `swm_ghex_2d.py` / nb01: `M = N = 16`, `dx = dy = 1e5`, `dt = 90`,
`a = 1e6`, `alpha = 1e-3`, `N_STEPS = 10`. `cost(fields) = sum(p**2)` over the global
interior.

### JAX versions, and the two shims

The harness runs unchanged on **jax 0.6.2** (the gt4py venv, Python 3.10) and on **jax
0.11.1** (Python 3.13), because `halo_transports.py` applies two compatibility shims at
import. Both are no-ops on 0.6.2.

1. **`shard_map`.** On jax >= 0.11 `jax.experimental.shard_map` is an empty module and the
   API is `jax.shard_map(f, mesh=, in_specs=, out_specs=, check_vma=)`; on 0.6.2 the
   working API is `jax.experimental.shard_map.shard_map(f, mesh, in_specs, out_specs,
   check_rep=)`. `halo_transports.shard_map(f, mesh, in_specs, out_specs, check_rep=True)`
   prefers the top-level form (0.6.2 also exposes it, with the new keyword names, and it
   gives bit-identical results there) and falls back to the experimental one. Every
   `shard_map` call in the harness goes through it; `shard_map_api()` says which is in use
   and the battery header prints it.
2. **GT4Py field construction under a JAX tracer.** GT4Py registers `jnp.ndarray` with
   `functools.singledispatch`. On jax >= 0.11 a tracer is still
   `isinstance(t, jax.Array)` but no longer *dispatches* through it, so `gtx.as_field`
   works eagerly and raises `NotImplementedError` from `common.py::_field` under
   `grad`/`jit`/`scan`. `halo_transports` registers `jax.core.Tracer` explicitly against
   `JaxArrayField.from_array` (and `JaxArrayConnectivityField.from_array`) when it is not
   already dispatchable. `GT4PY_TRACER_DISPATCH` reports `patched` / `not needed` /
   `no gt4py`. Nothing under `gt4py/src` is modified.

Verified 2026-09-09:

| environment | what was run | result |
|---|---|---|
| jax 0.6.2, gt4py venv, 8 fake CPU devices | full battery, 6 transports x 6 layouts | unchanged; see the table below |
| jax 0.11.1 + gt4py (`tmp/venv-011-gt4py`), 8 fake CPU devices | full battery, `allgather`/`padded`/`coloured8`/`coloured2ph`/`ragged_emul` on 2x2 and 4x2 | **ALL PASS**, 10/10; T3 exact with `max_ulp` 0.0, and T4, T5, T6 identical to the 0.6.2 numbers to every printed digit |
| jax 0.11.1, no gt4py (`.venv-jax`) | exchange-only T1+T2 (`tmp/exchange_only_0111.py`) on 2x2 and 4x2 | pass for all five; `ragged` fails to compile with the same message as on 0.6.2 |

Other 0.6.2 notes that still hold:

- Always `jax.jit` around the `shard_map`; the battery does. On 0.6.2 the bare
  `shard_map` also differentiates fine, with and without `jax.checkpoint` on the scan
  body (all four combinations checked), but `.lower(...).compile()` for T7/T8 needs the jit
  anyway.
- `check_rep=True` (the default, and `check_vma=True` through the shim) works for every
  transport here with `scan` + GT4Py. If a transport trips the replication check on scan
  carries, pass `check_rep=False`; it was verified to give bit-identical results.

## Test battery

| test | what | criterion |
|---|---|---|
| T0 `oracle_gate` | vs `allgather` | forward `array_equal`; adjoint rel < 1e-12 |
| T1 `exchange_forward` | `exchange` == `np.pad(interior, 1, "wrap")` per rank | exactly 0.0 |
| T2 `exchange_dotproduct` | `<Lx, y>` vs `<x, L^T y>` via `jax.vjp`; also vs `halo_lib.exchange_matrix(...).T` | rel <= 1e-14; dense <= 1e-12 |
| T3 `bit_identity_P1` | sharded on `1x1` vs `reference_forward` | exactly 0.0 (FESOM's invariant) |
| T4 `forward_P` | sharded on the requested layout vs reference | rel <= 1e-12 |
| T5 `gradient` | `grad(cost o sharded)` vs `grad(cost o reference)` | rel <= 1e-10 |
| T6 `taylor` | `\|J(x+hd) - J(x) - h<g,d>\|` over 6 halvings from `h=1e-2` | rate -> 2.00 |
| T7 `hlo` | `collectives_in` of 1 step, forward / cost / value_and_grad | reported |
| T8 `timing` | wall time, n_steps=10, min of 5 after warm-up | reported |

Three deliberate choices, all visible in the printed table:

- **T4/T5 use relative, not absolute, tolerances.** `p ~ 5e4`, so 1 ulp of `p` is 7e-12;
  an absolute 1e-12 on `p` is below the representable resolution of the field. The absolute
  numbers are printed too.
- **T5 prints a noise floor.** `noise_floor_*` is the same comparison between two
  *single-device* implementations that are forward bit-identical (`make_periodic` vs an
  explicit `jnp.pad(..., mode="wrap")`) and differ only in the order the adjoint
  accumulates. It sits at 4.3e-9 absolute / 2.8e-12 relative, i.e. one ulp of the
  pre-cancellation magnitude (~4.5e6) that the ~1.5e3 gradient entries are the difference
  of. The distributed gradient lands at 1.0-1.5e-8 / 6-10e-12, a small multiple of that
  floor. The gradient here is cancellation-limited, not transport-limited.
- **T3 is strict, and it is narrower than it looks.** Both sides of T3 call the *same*
  GT4Py `timestep`, so what it establishes is that **SPMD partitioning at P=1 changes
  nothing** -- a partitioner invariant -- not that the model is reproducible. A non-zero
  `max_ulp` of a few would be XLA instruction selection under SPMD partitioning, not the
  transport: it reproduces with a `shard_map` body containing no collective at all
  (observed on an earlier reassemble-by-reshape version of `allgather`, gone with the
  FESOM index-pair form). Report it, do not chase it.
- **T4 also reports `indep_max_rel_diff`,** which is the statement T3 cannot make: the
  sharded result against `roll_reference_forward`, a pure-`jnp.roll` single-device model
  in `swm_sharded.py` that shares **no code** with the GT4Py path and deliberately
  associates the arithmetic differently (`x/dx` rather than `(1/dx)*x`, one fused
  `0.25*(...)` rather than two nested `0.5*(...)`). It differs from `reference_forward` by
  5.55e-17 in `u` and `v` (1 ulp of re-association) and is bit-identical in `p`. Against
  it the sharded model lands at **1.45e-17 relative for `padded`, `coloured8`,
  `coloured2ph` and `ragged_emul` at every layout, and 6.82e-15 for `allgather`** -- the
  same split T4 shows against the GT4Py reference, now with an independent witness.

### Wire volume: the tables are the metric, the HLO is the cross-check

`T7` reports two numbers per transport and they answer different questions.

`table_bytes_per_step = 3 * wire_cells(tables) * 8` is what the transport's *design* puts
on the wire: the send-buffer size implied by the tables `prepare()` built, three fields,
f64. This is the primary metric, and it is FESOM's (`bench_halo_micro`'s `lanes` dict).
The reference printed next to it is the true halo rim,
`true_halo_bytes_per_step = 3 * (2h(MLOC+NLOC) + 4h^2) * 8`.

`fwd_fields_bytes` is `bytes_moved(collectives_in(hlo))`: the result size of every
defining collective instruction in the compiled module. Two corrections were needed before
it could be trusted at all, both from the same instruction:

- XLA:CPU compiles `lax.all_to_all(tiled=True)` into a **tuple**-typed instruction with one
  element per peer, `(f64[16], f64[16], f64[16], f64[16]) all-to-all(...)`. nb03's helper
  read only the first element, so every `all_to_all` volume was under-reported by a factor
  `P` (128 B instead of 512 B for `padded` at 2x2). `collectives_in` now counts every tuple
  element and renders it as `all-to-all[16]x4`.
- From tuple arity 6, XLA prints inline `/*index=5*/` markers **inside** the result type.
  The first version of the fix captured the type with `[^=]*?`, which that `=` terminates,
  so at `4x2`/`2x4` the instruction stopped matching at all and those rows silently
  reported *zero* collectives and *zero* bytes.

With both fixed the two metrics agree exactly for `allgather`, `padded` and both
`coloured`s at every layout. They disagree in exactly three places, all explicable:

| where | table | HLO | why |
|---|---|---|---|
| `padded` at `1x1` | 1632 | 0 | at `P=1` XLA deletes the single-device `all_to_all` entirely -- 0 B is the correct wire volume, 1632 B is the buffer it would have handed over |
| `ragged_emul`, `P>1` | = `ragged` | `P` x that | the emulation stands in for the primitive with an `all_gather`; its table number prices the **ragged design**, its HLO prices the **stand-in**. Never quote `ragged_emul`'s HLO as `ragged`'s |
| `ragged`, any layout | reported | none | it does not compile on XLA:CPU, so there is no module to read |

Two further HLO caveats that the numbers cannot fix. `all-gather`'s result is the full
gathered buffer, a slight over-count (a device does not receive its own shard).
`reduce-scatter`'s result is the per-device *output*, which under-counts the reduced volume
by roughly `P`, so `bwd_only_bytes` is a lower bound for `allgather` -- do not read its
forward/backward byte ratio literally. And one collective call is not one message: a dense
`all_to_all` is `P-1` messages per rank, which is what `messages_note` is for.

## Measured results

jax 0.6.2, CPU, 8 fake devices in one process
(`XLA_FLAGS=--xla_force_host_platform_device_count=8`), x64 on, `n_steps=10`, `M=N=16`.
Regenerate with

```
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
    ./.venv/bin/python examples/next/swm/bench_transports.py --transports all \
        --layouts 1x1,2x1,1x2,2x2,4x2,2x4 --repeats 50 --out results.jsonl
./.venv/bin/python examples/next/swm/bench_transports.py --table results.jsonl
```

`T0-T6` is `PASS` only if all seven correctness tests pass. Bytes are **per device per
step for all three fields**. `K/field` is collectives per field per exchange. `env` is
carried in every row so laptop and cluster files can be concatenated and still be told
apart. Correctness numbers are reproducible to the last bit -- five independent runs of
this grid agreed on all 23 non-timing fields of all 36 rows -- while the timings move by a
few percent between runs even at `--repeats 50`.

| env | transport | layout | P | MLOC x NLOC | T0-T6 | table B | HLO B | true B | K/field | fwd ms | grad ms | grad/fwd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jax 0.6.2 cpu x8/1p | `allgather` | 1x1 | 1 | 16x16 | PASS | 6144 | 6144 | 1632 | 1 | 0.0075 | 0.0357 | 4.74 |
| jax 0.6.2 cpu x8/1p | `coloured8` | 1x1 | 1 | 16x16 | PASS | 1632 | 1632 | 1632 | 8 | 0.0288 | 0.0686 | 2.38 |
| jax 0.6.2 cpu x8/1p | `coloured2ph` | 1x1 | 1 | 16x16 | PASS | 1632 | 1632 | 1632 | 4 | 0.0136 | 0.0467 | 3.45 |
| jax 0.6.2 cpu x8/1p | `padded` | 1x1 | 1 | 16x16 | PASS | 1632 | 0 | 1632 | 0 | 0.0070 | 0.0271 | 3.85 |
| jax 0.6.2 cpu x8/1p | `ragged` | 1x1 | 1 | 16x16 | compile_error | 1632 | - | 1632 | - | - | - | - |
| jax 0.6.2 cpu x8/1p | `ragged_emul` | 1x1 | 1 | 16x16 | PASS | 1632 | 1632 | 1632 | 1 | 0.0083 | 0.0333 | 4.02 |
| jax 0.6.2 cpu x8/1p | `allgather` | 2x1 | 2 | 8x16 | PASS | 6144 | 6144 | 1248 | 1 | 0.0231 | 0.0806 | 3.50 |
| jax 0.6.2 cpu x8/1p | `coloured8` | 2x1 | 2 | 8x16 | PASS | 1248 | 1248 | 1248 | 8 | 0.1646 | 0.3267 | 1.98 |
| jax 0.6.2 cpu x8/1p | `coloured2ph` | 2x1 | 2 | 8x16 | PASS | 1248 | 1248 | 1248 | 4 | 0.0762 | 0.1956 | 2.57 |
| jax 0.6.2 cpu x8/1p | `padded` | 2x1 | 2 | 8x16 | PASS | 1728 | 1728 | 1248 | 1 | 0.0236 | 0.0670 | 2.83 |
| jax 0.6.2 cpu x8/1p | `ragged` | 2x1 | 2 | 8x16 | compile_error | 1248 | - | 1248 | - | - | - | - |
| jax 0.6.2 cpu x8/1p | `ragged_emul` | 2x1 | 2 | 8x16 | PASS | 1248 | 2496 | 1248 | 1 | 0.0245 | 0.0851 | 3.48 |
| jax 0.6.2 cpu x8/1p | `allgather` | 1x2 | 2 | 16x8 | PASS | 6144 | 6144 | 1248 | 1 | 0.0316 | 0.0802 | 2.53 |
| jax 0.6.2 cpu x8/1p | `coloured8` | 1x2 | 2 | 16x8 | PASS | 1248 | 1248 | 1248 | 8 | 0.1531 | 0.3400 | 2.22 |
| jax 0.6.2 cpu x8/1p | `coloured2ph` | 1x2 | 2 | 16x8 | PASS | 1248 | 1248 | 1248 | 4 | 0.0768 | 0.2031 | 2.64 |
| jax 0.6.2 cpu x8/1p | `padded` | 1x2 | 2 | 16x8 | PASS | 1728 | 1728 | 1248 | 1 | 0.0247 | 0.0955 | 3.86 |
| jax 0.6.2 cpu x8/1p | `ragged` | 1x2 | 2 | 16x8 | compile_error | 1248 | - | 1248 | - | - | - | - |
| jax 0.6.2 cpu x8/1p | `ragged_emul` | 1x2 | 2 | 16x8 | PASS | 1248 | 2496 | 1248 | 1 | 0.0239 | 0.0788 | 3.29 |
| jax 0.6.2 cpu x8/1p | `allgather` | 2x2 | 4 | 8x8 | PASS | 6144 | 6144 | 864 | 1 | 0.0457 | 0.1256 | 2.75 |
| jax 0.6.2 cpu x8/1p | `coloured8` | 2x2 | 4 | 8x8 | PASS | 864 | 864 | 864 | 8 | 0.3451 | 0.8324 | 2.41 |
| jax 0.6.2 cpu x8/1p | `coloured2ph` | 2x2 | 4 | 8x8 | PASS | 864 | 864 | 864 | 4 | 0.1717 | 0.4186 | 2.44 |
| jax 0.6.2 cpu x8/1p | `padded` | 2x2 | 4 | 8x8 | PASS | 1536 | 1536 | 864 | 1 | 0.0603 | 0.1499 | 2.49 |
| jax 0.6.2 cpu x8/1p | `ragged` | 2x2 | 4 | 8x8 | compile_error | 864 | - | 864 | - | - | - | - |
| jax 0.6.2 cpu x8/1p | `ragged_emul` | 2x2 | 4 | 8x8 | PASS | 864 | 3456 | 864 | 1 | 0.0455 | 0.1415 | 3.11 |
| jax 0.6.2 cpu x8/1p | `allgather` | 4x2 | 8 | 4x8 | PASS | 6144 | 6144 | 672 | 1 | 0.1007 | 0.2387 | 2.37 |
| jax 0.6.2 cpu x8/1p | `coloured8` | 4x2 | 8 | 4x8 | PASS | 672 | 672 | 672 | 8 | 0.7681 | 1.8000 | 2.34 |
| jax 0.6.2 cpu x8/1p | `coloured2ph` | 4x2 | 8 | 4x8 | PASS | 672 | 672 | 672 | 4 | 0.3806 | 0.8635 | 2.27 |
| jax 0.6.2 cpu x8/1p | `padded` | 4x2 | 8 | 4x8 | PASS | 1536 | 1536 | 672 | 1 | 0.1151 | 0.3244 | 2.82 |
| jax 0.6.2 cpu x8/1p | `ragged` | 4x2 | 8 | 4x8 | compile_error | 672 | - | 672 | - | - | - | - |
| jax 0.6.2 cpu x8/1p | `ragged_emul` | 4x2 | 8 | 4x8 | PASS | 672 | 5376 | 672 | 1 | 0.1016 | 0.2410 | 2.37 |
| jax 0.6.2 cpu x8/1p | `allgather` | 2x4 | 8 | 8x4 | PASS | 6144 | 6144 | 672 | 1 | 0.1004 | 0.2274 | 2.26 |
| jax 0.6.2 cpu x8/1p | `coloured8` | 2x4 | 8 | 8x4 | PASS | 672 | 672 | 672 | 8 | 0.7762 | 1.9391 | 2.50 |
| jax 0.6.2 cpu x8/1p | `coloured2ph` | 2x4 | 8 | 8x4 | PASS | 672 | 672 | 672 | 4 | 0.4040 | 0.9051 | 2.24 |
| jax 0.6.2 cpu x8/1p | `padded` | 2x4 | 8 | 8x4 | PASS | 1536 | 1536 | 672 | 1 | 0.1090 | 0.2902 | 2.66 |
| jax 0.6.2 cpu x8/1p | `ragged` | 2x4 | 8 | 8x4 | compile_error | 672 | - | 672 | - | - | - | - |
| jax 0.6.2 cpu x8/1p | `ragged_emul` | 2x4 | 8 | 8x4 | PASS | 672 | 5376 | 672 | 1 | 0.0991 | 0.2521 | 2.55 |

### What the table says

- **Every transport except `ragged` passes every correctness test at every layout.** T3
  (P=1 bit identity) is exactly 0.0 with `max_ulp` 0.0 for all of them. T4 is exactly 0.0
  for `padded`, both `coloured`s and `ragged_emul` everywhere, and for `allgather` at
  `1x1` and `2x1`; only `allgather` at `1x2`, `2x2`, `4x2` and `2x4` shows 6.82e-15
  relative (see the note under section 7 of `nb05` -- it is a forward-only, scan-related
  XLA effect that appears from `n_steps >= 3`, not a cotangent-path difference). T5 is
  5.58e-12 to 9.66e-12 relative (8.55e-9 to 1.48e-8 absolute) against a measured noise
  floor of 2.80e-12 relative / 4.28e-9 absolute; T6 is 2.000 +/- 5e-5.
- **Volume.** `allgather` is flat at 6144 B, i.e. 3.8x the true halo at `P=1` and 9.1x at
  `P=8` -- the O(P) baseline. Both `coloured`s and `ragged` sit exactly on the true halo,
  ratio 1.00 at **every** layout -- the floor of FESOM's 1.0-1.4x, because on a torus every
  colour class is a full permutation with uniform chunks.
- **`padded`'s pad factor is a function of the LAYOUT, not of `P`.** It is the largest
  single-peer chunk times `P`, over the rim, so it grows with the *aspect ratio* of the
  block:

  | layout | 1x1 | 2x1 / 1x2 | 2x2 | 4x2 / 2x4 | 8x1 / 1x8 |
  |---|---|---|---|---|---|
  | P | 1 | 2 | 4 | 8 | 8 |
  | pad factor, h=1 | 1.00 | 1.38 | 1.78 | 2.29 | **3.60** |
  | pad factor, h=2 | 1.00 | 1.43 | 1.60 | 2.00 | **3.64** |

  FESOM measured 1.8x at `P=4` on their unstructured mesh -- the same number as `2x2`
  here -- and theirs keeps growing to 40.7x at `P=128`. On the torus it does not run away,
  but neither does it "saturate above 2": the two `P=8` layouts differ by 1.6x from each
  other, and `8x1`/`1x8` are exactly the layouts the 2-node sbatch block runs. At `h=2`,
  `8x1` is the case where the padding stops being a rounding error: `padded` moves 320
  cells against `allgather`'s 256 -- **worse than shipping the entire field**.
- **`coloured8` vs `coloured2ph` is the controlled experiment**: identical bytes at every
  layout, K = 8 vs K = 4, and the two-phase schedule is 1.92x-2.16x faster forward and
  1.47x-2.14x faster backward (`--repeats 50`; at the default `--repeats 5` the same
  ratios scatter over 1.3x-2.9x, which is how noisy these microsecond timings are). FESOM's "count the exchanges before you count the bytes",
  with the bytes pinned exactly equal. K does not collapse at `1x1`: XLA keeps the eight
  identity `collective-permute`s.
- **`ragged` never ran.** ``UNIMPLEMENTED: HLO opcode `ragged-all-to-all` is not supported
  by XLA:CPU ThunkEmitter`` -- note the backticks, they are in the message -- on every
  layout and on both JAX versions, raised as `XlaRuntimeError` on jax 0.6.2 and as
  `JaxRuntimeError` on 0.11.1. `ragged_emul` -- the same tables and the same pipeline with the
  primitive replaced by a pure-jnp transcription of its documented receive semantics --
  passes everything and validates the tables forward *and* adjoint.
- **Timings are dispatch-bound and rank nothing.** 8 fake CPU devices in one process, no
  interconnect, 16x16 grid: every transport gets *slower* as `P` grows while three of them
  move *less* per device. `grad/fwd` is 1.99-4.74 (FESOM measured 4.7x on real hardware).
  The only comparison that survives is the `coloured8`/`coloured2ph` one, because it holds
  volume exactly fixed.

### One battery in full: `coloured2ph` at 2x2

Verbatim except that the multi-line `note` rows of T3, T4 and T5 (reproduced in the
bullets under "Test battery") are elided. This is a **separate invocation** from the table
above, so its T8 rows differ from the table's `coloured2ph 2x2` row by a few percent of
run-to-run dispatch noise (0.1645 vs 0.1717 ms forward); everything else is identical.

```
==============================================================================
transport 'coloured2ph'   Layout(M=16, N=16, Rx=2, Ry=2, h=1, MLOC=8, NLOC=8, P=4)   n_steps=10
jax 0.6.2   platform cpu   devices 8   x64 True
shard_map via jax.shard_map(check_vma=)   gt4py tracer dispatch: not needed
==============================================================================

T0 oracle_gate        (vs allgather: bit-equal forward, 1e-12 adjoint)  [PASS]
      oracle                 allgather
      self_comparison        False
      fwd_array_equal        True
      fwd_max_abs_diff       0.000000
      grad_max_rel_diff      3.979303e-17

T1 exchange_forward   (exchange == wrap-pad, per rank)  [PASS]
      max_abs_diff           0.000000
      exact                  True

T2 exchange_dotproduct(<Lx,y> == <x,L^T y>, + halo_lib dense L^T)  [PASS]
      lhs                    21.529100
      rhs                    21.529100
      rel_diff               1.650191e-16
      halo_lib_order_matches True
      dense_fwd_diff         0.000000
      dense_vjp_diff         4.440892e-16

T3 bit_identity_P1    (sharded P=1 == single-device reference)  [PASS]
      max_abs_diff_u         0.000000
      max_abs_diff_v         0.000000
      max_abs_diff_p         0.000000
      max_abs_diff           0.000000
      max_ulp                0.000000
      exact                  True

T4 forward_P          (sharded vs reference)  [PASS]
      indep_max_rel_diff     1.445736e-17
      max_abs_diff_u         0.000000
      max_abs_diff_v         0.000000
      max_abs_diff_p         0.000000
      max_rel_diff_u         0.000000
      max_rel_diff_v         0.000000
      max_rel_diff_p         0.000000
      max_abs_diff           0.000000
      max_rel_diff           0.000000
      exact                  True

T5 gradient           (sharded grad vs reference grad)  [PASS]
      max_abs_diff_u         1.288367e-08
      max_rel_diff_u         8.411585e-12
      max_abs_diff_v         1.124420e-08
      max_rel_diff_v         7.341191e-12
      max_abs_diff_p         5.820766e-11
      max_rel_diff_p         5.820731e-16
      max_abs_diff           1.288367e-08
      max_rel_diff           8.411585e-12
      noise_floor_abs        4.284175e-09
      noise_floor_rel        2.797082e-12

T6 taylor             (2nd-order remainder rate -> 2)  [PASS]
      J0                     6.400000e+11
      slope                  4.050171e+10
      h                      [0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.000313, 0.000156]
      err2                   [2.72e+07, 6.81e+06, 1.7e+06, 4.26e+05, 1.07e+05, 2.66e+04, 6.66e+03]
      rate2                  [2, 2, 2, 2, 2, 2]
      rate2_last             1.999966

T7 hlo                (collectives, 1 step)
      n_steps                1
      messages_note          no all-to-all; one collective == one message here
      wire_cells             36
      table_bytes_per_step   864
      true_halo_bytes_per_step 864
      table_over_true        1.000000
      fwd_fields             ['collective-permute[8]', 'collective-permute[8]', 'collective-permute[8]', 'collective-permute[8]', 'collective-permute[8]', 'collective-permute[8]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[10]']
      fwd_fields_n           12
      fwd_fields_bytes       864
      fwd_cost               ['collective-permute[8]', 'collective-permute[8]', 'collective-permute[8]', 'collective-permute[8]', 'collective-permute[8]', 'collective-permute[8]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[10]', 'all-reduce[]']
      fwd_cost_n             13
      fwd_cost_bytes         872
      grad                   ['collective-permute[8]', 'collective-permute[8]', 'collective-permute[8]', 'collective-permute[8]', 'collective-permute[8]', 'collective-permute[8]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[10]', 'all-reduce[]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[8]', 'collective-permute[8]', 'collective-permute[8]', 'collective-permute[8]', 'collective-permute[8]', 'collective-permute[8]']
      grad_n                 25
      grad_bytes             1736
      bwd_only               ['collective-permute[10]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[10]', 'collective-permute[8]', 'collective-permute[8]', 'collective-permute[8]', 'collective-permute[8]', 'collective-permute[8]', 'collective-permute[8]']
      bwd_only_n             12
      bwd_only_bytes         864
      missing_in_grad        []

T8 timing             (CPU, 8 fake devices in one process)
      n_steps                10
      repeats                50
      fwd_total_s            0.001645
      fwd_per_step_ms        0.164500
      grad_total_s           0.004087
      grad_per_step_ms       0.408738
      grad_over_fwd          2.484732

------------------------------------------------------------------------------
summary: ALL PASS
------------------------------------------------------------------------------
```

The adjoint is 12 `collective-permute`s carrying exactly the same 864 B as the forward's
12, and it is symmetric round for round: `ppermute` transposes to the inverse `ppermute`,
in reverse phase order. No adjoint code exists anywhere in `transport_coloured.py` -- nor
in any other transport module; `nb05` §4 asserts that mechanically.

## Environment on Santis (CSCS Alps, GH200, 4 GPUs/node, aarch64)

The laptop run establishes correctness and the interface. The measurements that matter --
the real `ragged_all_to_all` at `P >= 4`, and a latency-vs-bandwidth ranking that CPU
timings structurally cannot produce -- happen there.

- **`requirements-santis.txt`** is the primary environment: `jax[cuda12]==0.11.1`, Python
  3.12 or 3.13, gt4py editable from the repo root. jaxlib and the CUDA plugin publish
  `manylinux_2_27_aarch64` wheels for cp312-cp315 at that version. It needs **both** shims
  above, and both were verified on jax 0.11.1 + gt4py on CPU (full battery ALL PASS).
  FESOM2-JAX recorded jax 0.10.1 and also ran 0.11.0, so this pin is the one directly
  comparable to their `ragged` result.
- **`requirements-santis-jax062.txt`** is the fallback: `jax[cuda12]==0.6.2`, Python
  3.10-3.13, `manylinux2014_aarch64` wheels. This is the version every number above was
  measured with; on aarch64 only the CUDA 12 plugin exists at 0.6.2.
- **`santis_bench.sbatch`** is the Slurm template: 1 node x 4 GH200 with
  `--layouts 4x1,2x2,1x4`, plus a commented block for 2 nodes x 8 GPUs
  (`8x1,4x2,2x4,1x8`), where `4x2` and `2x4` put four ranks per node so half of every
  exchange crosses Slingshot instead of NVLink. Fill in `--account`, the uenv image and
  the venv path.
- `XLA_PYTHON_CLIENT_PREALLOCATE=false` was needed on the laptop's 8 GiB GPU or NCCL
  failed to allocate; the template sets it.

### `--distributed`: what it does and how far it is tested

It calls `jax.distributed.initialize()` (JAX auto-detects Slurm) when `SLURM_NTASKS > 1`,
requires every layout to satisfy `Rx*Ry == jax.device_count()` -- a mesh over a subset of
the devices does not span all the processes, and its collectives would hang rather than
fail -- and writes rows (and the summary line) from process 0 only. The battery itself is
ported, not merely guarded:

- inputs are built with `jax.make_array_from_process_local_data`: every process computes
  the whole (tiny) array redundantly from the same seed and contributes only the blocks
  its own devices own (`swm_sharded.put_global`);
- results come back through `jax.experimental.multihost_utils.process_allgather`
  (`swm_sharded.get_global`), because `np.asarray` on an array that spans non-addressable
  devices raises;
- the single-device references are computed redundantly on every process;
- **T3 is skipped** under `jax.process_count() > 1` with
  `status: skipped (multi-process): a P=1 mesh does not span the processes` -- it is by
  definition a one-device test, and a one-device mesh inside a multi-process program is a
  hang on NCCL, not an error.

Tested here with **two real processes over gloo on CPU** (a faked Slurm step: `SLURM_JOB_ID`,
`SLURM_STEP_NODELIST`, `SLURM_NTASKS=2`, `SLURM_PROCID`, `SLURM_LOCALID`, plus
`JAX_CPU_COLLECTIVES_IMPLEMENTATION=gloo`), with 1 device per process at layout `2x1` and
2 devices per process at `4x1`, `2x2`, `1x4`: all five working transports report ALL PASS
on both processes, and T4/T5 match the single-process values exactly. **Untested:** more
than one node, NCCL, and GPUs. The single-process path is unchanged -- five runs of the
laptop grid agree on all 23 non-timing fields of all 36 rows.

Merge and re-render across machines:

```
cat laptop_results.jsonl santis_*.jsonl > all.jsonl
python examples/next/swm/bench_transports.py --table all.jsonl
```

## Where this connects

`nb05_halo_transports.ipynb` is the comparison: FESOM's design and what changes on a
torus, the battery, one section per transport with the load-bearing `where` masks, the
adjoint (zero `custom_vjp` anywhere), `coloured8` vs `coloured2ph`, the `ragged` story
including the finding that JAX's transpose *rule* is algebraically correct, the two XLA
findings, the timing caveat, and what a GT4Py-level exchange would emit. It reads its
numbers from a `bench_transports.py` results file.

`nb01` (the adjoint of a halo exchange in the field view), `nb02` (Route A `shard_map` vs
Route B MPI + `custom_vjp`), `nb03` (where the collectives and their transposes come from,
and the GSPMD-automatic baseline this round did **not** implement -- it cannot run inside
`shard_map`, so it is a different program shape, not a fifth transport), `nb04` (the 2-D
exchange as explicit messages, and the dense `L^T` that T2 checks against).
