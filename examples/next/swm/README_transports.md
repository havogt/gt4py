# Halo-exchange transports for the sharded SWM

FESOM2-JAX (arXiv:2608.01546, Sect. 2.2-2.3) writes the halo exchange *in JAX*, from
primitives whose transposes JAX already knows, so the distributed adjoint comes by
composition instead of by hand. This directory does the same on the structured SWM and
compares transports by adjoint exactness, wire volume and time.

Files:

| file | what |
|---|---|
| `halo_transports.py` | `Layout`, the `Transport` protocol, the registry, the blocking helpers |
| `jax_compat.py` | `shard_map` across jax versions; `patch_gt4py_tracer_dispatch()` for GT4Py fields from jax tracers |
| `hlo_accounting.py` | `collectives_in(hlo_text)`, `bytes_moved(collectives)` |
| `transport_allgather.py` | all-gather broadcast, registered as `"allgather"` -- the oracle |
| `transport_padded.py` | slot-padded dense `all_to_all`, registered as `"padded"` |
| `transport_coloured.py` | `ppermute` rounds, registered as `"coloured8"` (K=8) and `"coloured2ph"` (K=4) |
| `transport_ragged.py` | `ragged_all_to_all`, registered as `"ragged"`, plus its CPU-runnable emulation `"ragged_emul"` |
| `swm_sharded.py` | the sharded model and the single-device references |
| `swm_battery.py` | the test battery T0-T8 and its CLI |
| `bench_transports.py` | the battery over a (transport, layout) grid -> one JSON row each; `--table` renders them |
| `santis_bench.sbatch` | Slurm template for CSCS Alps / Santis (GH200); never yet run on a cluster |
| `requirements-santis.txt` | primary Santis environment (jax 0.11.1 + CUDA 12, aarch64) |
| `requirements-santis-jax062.txt` | fallback: the laptop-proven jax 0.6.2 pin |
| `nb05_halo_transports.ipynb` | the comparison notebook: FESOM's design, the five transports, the adjoint, the two XLA findings |

## Running the battery

```
cd <gt4py>
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
    python examples/next/swm/swm_battery.py --transport allgather --layout 2x2 --steps 10
```

```
usage: swm_battery.py [--transport {allgather,coloured8,coloured2ph,padded,ragged,ragged_emul}]
                      [--layout RxxRy] [--steps N] [--repeats N]
```

`--layout RxxRy` needs `Rx*Ry <=` the device count and `M % Rx == 0`, `N % Ry == 0`
(`M = N = 16` in `swm_sharded.py`). Exit status is 0 iff every test passes. Every test
runs under a guard: a transport that will not compile reports `status: compile_error: ...`
on that row and the rest of the table still prints. `--transport` accepts any registered
name; the registry imports every `transport_*.py` sibling, so `coloured8`/`coloured2ph`
(in `transport_coloured.py`) and `ragged_emul` (in `transport_ragged.py`) resolve too.

Neither script pins `JAX_PLATFORMS` or the fake-device flag; those belong on the command
line. The device count comes from `jax.devices()`.

### The whole grid at once

```
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
    python examples/next/swm/bench_transports.py \
        --transports all --layouts 1x1,2x1,1x2,2x2,4x2,2x4 --steps 10 --repeats 50 --out results.jsonl

python examples/next/swm/bench_transports.py --table results.jsonl
```

```
usage: bench_transports.py [--transports all|name,name,...] [--layouts 1x1,2x1,...] [--steps N]
                           [--repeats N] [--out RESULTS.JSONL] [--distributed]
                           [--table RESULTS.JSONL]
```

One JSON row per (transport, layout), appended, each carrying the environment it was
measured in (jax version, platform, device kind, device and process counts, which
`shard_map` API, whether the GT4Py tracer patch fired) alongside every T0-T8 scalar, both
volume metrics, the collective counts and the timings. Laptop and cluster files therefore
concatenate, and `--table` renders any of them:

```
cat laptop_results.jsonl $SCRATCH/santis_*.jsonl > all.jsonl
python examples/next/swm/bench_transports.py --table all.jsonl
```

Exit status is 1 while `ragged` is in the set, because `ragged` cannot compile on CPU.

## Interface contract for a new transport

Create `transport_<anything>.py` in this directory. It must import `halo_transports` and
call `register(...)` at import time; `get_transport(name)` imports every `transport_*.py`
sibling on a registry miss, so a module that does not import cannot break the others.

```python
from halo_transports import Layout, register

class MyTransport:
    name = "mine"

    def prepare(self, layout: Layout):  # once, outside jit; returns anything
        return {...}                    # index arrays, permutations, slot maps, ...

    def exchange(self, a_local, tables, axis_name: str):
        ...                             # inside the shard_map body
        return a_refreshed

    def wire_cells(self, tables) -> int:
        return ...                      # cells this device puts on the wire per exchange

register(MyTransport())
```

`exchange` receives this device's `(MLOC+2h, NLOC+2h)` jnp array, with the interior at
`[h:-h, h:-h]`, and must return the same shape with **all** halo cells refreshed from the
owning neighbours -- four faces **and** four corners, periodic in both directions. It must
be built only from `jax.lax` collectives and array ops: **no `pure_callback`, no
`custom_vjp`**, so that `jax.vjp` transposes it automatically. It must not depend on the
values in the input halo.

`wire_cells` is the larger of the send and receive sides. For `padded`, `coloured` and
`ragged` they are equal; `allgather` *sends* only `MLOC*NLOC`, which falls with `P`, and
*receives* `P*MLOC*NLOC`, which does not, so it reports the receive side -- taking delivery
of the whole field is what makes it the O(P) transport.

### `Layout`

`Layout(M, N, Rx, Ry, h=1)`: rank `r` owns the block at `(rx, ry) = divmod(r, Ry)`, i.e.
global rows `[rx*MLOC, (rx+1)*MLOC)` and columns `[ry*NLOC, (ry+1)*NLOC)`; axis 0 is `I`
(x, E/W), axis 1 is `J` (y, N/S). It provides the coordinate and neighbour tables, the
FESOM all-gather index pair (`owner_lane()`), the per-peer rim chunks (`halo_chunks()`)
and the blocking helpers `block`/`unblock`/`block_halo`/`unblock_halo` between global
`(M, N)` arrays and the rank-major stacks `shard_map` wants with `in_specs=P("d")`. See the
docstrings in `halo_transports.py`. The ordering is `halo_lib.Decomposition`'s and
`swm_ghex_2d.py`'s; T2 reports `halo_lib_order_matches` and uses `halo_lib`'s dense
transpose only when it is `True`.

### The oracle

`allgather` is the oracle. `swm_battery.oracle_gate(transport, layout)` is the gate every
new transport must pass, with FESOM's criteria: forward `np.array_equal` against
`allgather` (bit equality, not a tolerance) and, for `sum(w * exchange(x))` with random
`w`, `max|g - g_ref| / max|g_ref| < 1e-12`. It runs as **T0** of the battery.

## The model

`swm_sharded.sharded_forward(transport, layout, u0, v0, p0, n_steps)` runs one jitted
program (`sharded_program`, cached per transport, layout and step count): `shard_map` over
the 1-D device axis `"d"`, `in_specs = out_specs = P("d")`, inputs and outputs the
rank-major stack of interior blocks. The body zero-pads to the halo shape, calls
`exchange`, wraps the local arrays as JAX-backed GT4Py fields on
`gtx.domain({I: (-1, MLOC+1), J: (-1, NLOC+1)})`, and runs `operators.timestep.definition`
with `M=MLOC, N=NLOC`: first step forward Euler with `dt` and `alpha=0`, then leapfrog with
`2*dt` under `jax.lax.scan`. The module docstring says why `timestep`'s own `make_periodic`
on the local sizes is harmless.

Three single-device references on the global grid, all jitted and cached per `n_steps`:
`reference_program` (the same GT4Py step, halos from `make_periodic`),
`wrap_reference_program` (halos from an explicit `jnp.pad(mode="wrap")`; forward
bit-identical to `reference_program`, adjoint accumulated in a different order) and
`roll_reference_program` (pure `jnp.roll`, no GT4Py, the arithmetic deliberately
associated differently).

Parameters match `swm_ghex_2d.py` / nb01: `M = N = 16`, `dx = dy = 1e5`, `dt = 90`,
`a = 1e6`, `alpha = 1e-3`, `N_STEPS = 10`. `cost(fields) = sum(p**2)` over the global
interior.

### JAX versions

The harness runs on **jax 0.6.2** and **jax 0.11.1** through `jax_compat.py`:

1. `shard_map(f, mesh, in_specs, out_specs, check_rep=True)` calls
   `jax.shard_map(..., check_vma=)` where that exists and
   `jax.experimental.shard_map.shard_map(..., check_rep=)` otherwise; `shard_map_api()`
   says which, and the battery header prints it.
2. `patch_gt4py_tracer_dispatch()` registers `jax.core.Tracer` with GT4Py's
   `singledispatch` field constructor: on jax >= 0.11 a tracer no longer dispatches through
   `jax.Array`, so without it `gtx.as_field` raises `NotImplementedError` under
   `grad`/`jit`/`scan`. `swm_sharded.py` calls it before importing `gt4py.next` and keeps
   the result (`patched` / `not needed` / `no gt4py`) as `GT4PY_TRACER_DISPATCH`, which the
   battery header and every JSON row report. Nothing under `gt4py/src` is modified.

Verified on 8 fake CPU devices: jax 0.6.2 (Python 3.10) with the full grid of six
transports x six layouts; jax 0.11.1 (Python 3.13, same gt4py checkout) with `allgather`,
`padded` and `coloured8` on 2x4 -- ALL PASS, with every T0-T6 number identical to the
0.6.2 run to the last printed digit. `ragged` fails to compile on both with the same
message.

Two rules that hold on both versions: always `jax.jit` around the `shard_map` (T7/T8 need
the jit for `.lower(...).compile()` anyway), and `check_rep=True` (the default) works for
every transport here with `scan` + GT4Py -- if a new transport trips the replication check
on scan carries, pass `check_rep=False`.

## Test battery

| test | what | criterion |
|---|---|---|
| T0 `oracle_gate` | vs `allgather` | forward `array_equal`; adjoint rel < 1e-12 |
| T1 `exchange_forward` | `exchange` == `np.pad(interior, 1, "wrap")` per rank | exactly 0.0 |
| T2 `exchange_dotproduct` | `<Lx, y>` vs `<x, L^T y>` via `jax.vjp`; also vs `halo_lib.exchange_matrix(...).T` | rel <= 1e-14; dense <= 1e-12 |
| T3 `bit_identity_P1` | T4 on `1x1` | exactly 0.0 (FESOM's invariant) |
| T4 `forward_P` | sharded on the requested layout vs `reference_program`; also vs `roll_reference_program` | rel <= 1e-12 |
| T5 `gradient` | `grad(cost o sharded)` vs `grad(cost o reference)` | rel <= 1e-10, relative to the largest gradient component |
| T6 `taylor` | `\|J(x+hd) - J(x) - h<g,d>\|` over 6 halvings from `h=1e-2` | rate -> 2.00 |
| T7 `hlo` | `collectives_in` of 1 step, forward / cost / value_and_grad, plus the table-derived volume | reported |
| T8 `timing` | wall time of forward and `value_and_grad`, min of `--repeats` after warm-up | reported |

Four deliberate choices, all visible in the printed table:

- **T4/T5 use relative, not absolute, tolerances.** `p ~ 5e4`, so 1 ulp of `p` is 7e-12;
  an absolute 1e-12 on `p` is below the representable resolution of the field. The absolute
  numbers are printed too. T5 gates on `max_rel_diff_common`, the worst absolute error over
  `u, v, p` divided by the largest component of the whole reference gradient (the `p`
  adjoint, ~1e5); the per-field `max_rel_diff_*` are informational, since `max|du|`
  shrinks with the grid while the absolute error is set by the `p` adjoint.
- **T5 prints a noise floor.** `noise_floor_*` is the same comparison between
  `reference_program` and `wrap_reference_program`, which are forward bit-identical and
  differ only in the order the adjoint accumulates. It sits at 4.3e-9 absolute / 2.8e-12
  relative, one ulp of the pre-cancellation magnitude (~4.5e6) that the ~1.5e3 gradient
  entries are the difference of. The distributed gradient lands at 1.0-1.5e-8 / 6-10e-12,
  a small multiple of that floor: cancellation-limited, not transport-limited.
- **T3 is strict, and it is narrower than it looks.** Both sides call the *same* GT4Py
  `timestep`, so what it establishes is that **SPMD partitioning at P=1 changes nothing**
  -- a partitioner invariant -- not that the model is reproducible. A non-zero `max_ulp` of
  a few would be XLA instruction selection under SPMD partitioning, not the transport.
  Report it, do not chase it.
- **T4 also reports `indep_max_rel_diff`,** the statement T3 cannot make: the sharded
  result against `roll_reference_program`, which shares no code with the GT4Py path. It
  differs from `reference_program` by 5.55e-17 in `u` and `v` (1 ulp of re-association)
  and is bit-identical in `p`. Against it the sharded model lands at **1.45e-17 relative
  for `padded`, `coloured8`, `coloured2ph` and `ragged_emul` at every layout, and 6.82e-15
  for `allgather`** -- the same split T4 shows against the GT4Py reference, now with an
  independent witness.

### Wire volume: the tables are the metric, the HLO is the cross-check

T7 reports three byte counts per transport, all per device per step for the three f64
fields, and they answer different questions:

- `table_bytes = 3 * wire_cells(tables) * 8` is what the transport's *design* puts on the
  wire, the buffer size implied by the tables `prepare()` built. This is the primary
  metric, and it is FESOM's (`bench_halo_micro`'s `lanes` dict).
- `true_halo_bytes = 3 * (2h(MLOC+NLOC) + 4h^2) * 8` is the halo rim, the reference.
- `hlo_fwd_bytes = bytes_moved(collectives_in(hlo))` is the result size of every defining
  collective instruction in the compiled module. `collectives_in` counts every element of
  a tuple-typed instruction (XLA:CPU compiles `lax.all_to_all(tiled=True)` into one, with
  one element per peer, rendered as `all-to-all[16]x4`) and tolerates the inline
  `/*index=k*/` markers XLA prints inside the type from tuple arity 6; both were needed
  before the number could be trusted.

The three names are the same in the T7 rows, the JSON rows and the `--table` header. The
table and HLO numbers agree exactly for `allgather`, `padded` and both `coloured`s at every
layout. They disagree in exactly three places, all explicable:

| where | table | HLO | why |
|---|---|---|---|
| `padded` at `1x1` | 1632 | 0 | at `P=1` XLA deletes the single-device `all_to_all` entirely -- 0 B is the correct wire volume, 1632 B is the buffer it would have handed over |
| `ragged_emul`, `P>1` | = `ragged` | `P` x that | the emulation stands in for the primitive with an `all_gather`; its table number prices the **ragged design**, its HLO prices the **stand-in**. Never quote `ragged_emul`'s HLO as `ragged`'s |
| `ragged`, any layout | reported | none | it does not compile on XLA:CPU, so there is no module to read |

Two further HLO caveats. `all-gather`'s result is the full gathered buffer, a slight
over-count (a device does not receive its own shard). `reduce-scatter`'s result is the
per-device *output*, which under-counts the reduced volume by roughly `P`, so
`hlo_bwd_bytes` is a lower bound for `allgather` -- do not read its forward/backward byte
ratio literally. And one collective call is not one message: a dense `all_to_all` is `P-1`
messages per rank, which is what `messages_note` is for.

## Measured results

jax 0.6.2, CPU, 8 fake devices in one process
(`XLA_FLAGS=--xla_force_host_platform_device_count=8`), x64 on, `n_steps=10`, `M=N=16`,
`--repeats 50`; the full grid is the `bench_transports.py` command above, the `2x2` rows:

| env | transport | layout | P | MLOC x NLOC | T0-T6 | table_bytes | hlo_fwd_bytes | true_halo_bytes | K/field | fwd ms | grad ms | grad/fwd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jax 0.6.2 cpu x8/1p | `allgather` | 2x2 | 4 | 8x8 | PASS | 6144 | 6144 | 864 | 1 | 0.0457 | 0.1256 | 2.75 |
| jax 0.6.2 cpu x8/1p | `coloured8` | 2x2 | 4 | 8x8 | PASS | 864 | 864 | 864 | 8 | 0.3451 | 0.8324 | 2.41 |
| jax 0.6.2 cpu x8/1p | `coloured2ph` | 2x2 | 4 | 8x8 | PASS | 864 | 864 | 864 | 4 | 0.1717 | 0.4186 | 2.44 |
| jax 0.6.2 cpu x8/1p | `padded` | 2x2 | 4 | 8x8 | PASS | 1536 | 1536 | 864 | 1 | 0.0603 | 0.1499 | 2.49 |
| jax 0.6.2 cpu x8/1p | `ragged` | 2x2 | 4 | 8x8 | compile_error | 864 | - | 864 | - | - | - | - |
| jax 0.6.2 cpu x8/1p | `ragged_emul` | 2x2 | 4 | 8x8 | PASS | 864 | 3456 | 864 | 1 | 0.0455 | 0.1415 | 3.11 |

`T0-T6` is `PASS` only if all seven correctness tests pass. `K/field` is collectives per
field per exchange. Correctness numbers are reproducible to the last bit -- five
independent runs of the full grid agreed on all 23 non-timing fields of all 36 rows --
while the timings move by a few percent between runs even at `--repeats 50`.

### What the grid says

- **Every transport except `ragged` passes every correctness test at every layout.** T3 is
  exactly 0.0 with `max_ulp` 0.0 for all of them. T4 is exactly 0.0 for `padded`, both
  `coloured`s and `ragged_emul` everywhere, and for `allgather` at `1x1` and `2x1`; only
  `allgather` at `1x2`, `2x2`, `4x2` and `2x4` shows 6.82e-15 relative (see the note under
  section 7 of `nb05` -- it is a forward-only, scan-related XLA effect that appears from
  `n_steps >= 3`, not a cotangent-path difference). T5 is 5.58e-12 to 9.66e-12 relative
  (8.55e-9 to 1.48e-8 absolute) against a measured noise floor of 2.80e-12 relative /
  4.28e-9 absolute; T6 is 2.000 +/- 5e-5.
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
  other, and `8x1`/`1x8` are exactly the layouts the 2-node sbatch run uses. At `h=2`,
  `8x1` is the case where the padding stops being a rounding error: `padded` moves 320
  cells against `allgather`'s 256 -- **worse than shipping the entire field**.
- **`coloured8` vs `coloured2ph` is the controlled experiment**: identical bytes at every
  layout, K = 8 vs K = 4, and the two-phase schedule is 1.92x-2.16x faster forward and
  1.47x-2.14x faster backward (`--repeats 50`; at the default `--repeats 5` the same ratios
  scatter over 1.3x-2.9x, which is how noisy these microsecond timings are). FESOM's
  "count the exchanges before you count the bytes", with the bytes pinned exactly equal. K
  does not collapse at `1x1`: XLA keeps the eight identity `collective-permute`s. In T7 the
  adjoint of `coloured2ph` is 12 `collective-permute`s carrying exactly the forward's 864 B,
  round for round in reverse phase order: `ppermute` transposes to the inverse `ppermute`.
- **`ragged` never ran.** ``UNIMPLEMENTED: HLO opcode `ragged-all-to-all` is not supported
  by XLA:CPU ThunkEmitter`` -- note the backticks, they are in the message -- on every
  layout and on both JAX versions, raised as `XlaRuntimeError` on jax 0.6.2 and as
  `JaxRuntimeError` on 0.11.1. `ragged_emul` -- the same tables and the same pipeline with
  the primitive replaced by a pure-jnp transcription of its documented receive semantics --
  passes everything and validates the tables forward *and* adjoint.
- **Timings are dispatch-bound and rank nothing.** 8 fake CPU devices in one process, no
  interconnect, 16x16 grid: every transport gets *slower* as `P` grows while three of them
  move *less* per device. `grad/fwd` is 1.99-4.74 (FESOM measured 4.7x on real hardware).
  The only comparison that survives is the `coloured8`/`coloured2ph` one, because it holds
  volume exactly fixed.

## Environment on Santis (CSCS Alps, GH200, 4 GPUs/node, aarch64)

The laptop run establishes correctness and the interface. The measurements that matter --
the real `ragged_all_to_all` at `P >= 4`, and a latency-vs-bandwidth ranking that CPU
timings structurally cannot produce -- happen there.

- **`requirements-santis.txt`** is the primary environment: `jax[cuda12]==0.11.1`, Python
  3.12-3.13, gt4py editable from the repo root. FESOM2-JAX recorded jax 0.10.1 and also
  ran 0.11.0, so this pin is the one directly comparable to their `ragged` result.
- **`requirements-santis-jax062.txt`** is the fallback: `jax[cuda12]==0.6.2`, the version
  every number above was measured with; on aarch64 only the CUDA 12 plugin exists at 0.6.2.
- **`santis_bench.sbatch`** is the Slurm template: 1 node x 4 GH200 with
  `LAYOUTS=4x1,2x2,1x4`; resubmit with `--nodes=2` and `LAYOUTS=8x1,4x2,2x4,1x8` for 8
  GPUs, where `4x2` and `2x4` put four ranks per node so half of every exchange crosses
  Slingshot instead of NVLink. Fill in `--account`, the uenv image, `GT4PY` and the venv
  path; results go to `$SCRATCH/santis_<jobid>.jsonl`.
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

Tested with **two real processes over gloo on CPU**, a faked Slurm step on one machine
(`SLURM_JOB_ID` picks the coordinator port, `SLURM_STEP_NODELIST` its host):

```
cd <gt4py>
for r in 0 1; do
  JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=1 \
  JAX_CPU_COLLECTIVES_IMPLEMENTATION=gloo \
  SLURM_JOB_ID=4242 SLURM_STEP_NODELIST=localhost SLURM_NTASKS=2 SLURM_STEP_NUM_NODES=1 \
  SLURM_PROCID=$r SLURM_LOCALID=$r \
  python examples/next/swm/bench_transports.py --distributed \
      --transports padded --layouts 2x1 --out mp_$r.jsonl > mp_$r.log 2>&1 &
done
wait
```

With 1 device per process at `2x1` and 2 devices per process at `4x1`, `2x2`, `1x4`, all
five working transports report ALL PASS on both processes, and T4/T5 match the
single-process values exactly. **Untested:** more than one node, NCCL, and GPUs.

## Where this connects

`nb05_halo_transports.ipynb` is the comparison: FESOM's design and what changes on a
torus, the battery, one section per transport, the adjoint (zero `custom_vjp` anywhere),
`coloured8` vs `coloured2ph`, the `ragged` story including the finding that JAX's
transpose *rule* is algebraically correct, the two XLA findings, the timing caveat, and
what a GT4Py-level exchange would emit. It reads its numbers from a `bench_transports.py`
results file.

`nb01` (the adjoint of a halo exchange in the field view), `nb02` (Route A `shard_map` vs
Route B MPI + `custom_vjp`), `nb03` (where the collectives and their transposes come from,
and the GSPMD-automatic baseline this round did **not** implement -- it cannot run inside
`shard_map`, so it is a different program shape, not a fifth transport), `nb04` (the 2-D
exchange as explicit messages, and the dense `L^T` that T2 checks against).
