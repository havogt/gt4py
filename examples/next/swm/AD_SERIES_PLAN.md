<!-- Working notes for the AD notebook series in this directory (nb00-nb04,
swm_ghex*.py, halo_lib.py). Written during development, so they reference one
developer's machine: absolute paths under /home/vogtha, two local venvs, and a
laptop-specific MPI workaround. Kept for the verified findings and the record of
what was checked how, not as user documentation. -->

# Plan — AD of halo exchanges in GT4Py

## Goal

A set of instructive Jupyter notebooks that explain **how automatic differentiation
behaves at a halo exchange**, using GT4Py `next` in embedded mode with JAX-backed
fields and the NCAR shallow water model (SWM) as the vehicle.

This round covers halo exchange only. It is designed as the first installment of a
larger GT4Py AD tutorial series, so numbering and framing leave room to fold in the
already-existing Taylor-test / 4D-Var / hybrid-NN material later.

## Locked decisions

| Question | Decision |
|---|---|
| Base branch | `swm_2026_halo_update` on `havogt/gt4py` (newest of the four `swm_2026_*`) |
| Execution mode | Embedded only — JAX-backed `Field`s. The `jax_jit` runner is out of scope. |
| Distributed coverage | Both `shard_map`/`ppermute` **and** GHEX/MPI + `custom_vjp` |
| Model | NCAR SWM, `examples/next/swm/` |

## Working tree

```
gt4py_autodiff/
  PLAN.md          this file
  STATUS.md        rolling state — read this first when resuming
  gt4py/           havogt/gt4py, branch `ad_halo` (off swm_2026_halo_update)
                   origin = havogt/gt4py, upstream = GridTools/gt4py
  swm/             havogt/SWM, for reference only (older AD experiments)
                   origin = havogt/SWM, upstream = NCAR/SWM
```

Notebooks land in `gt4py/examples/next/swm/`, next to the existing
`example_4dvar.ipynb`.

## Deliverables

- `nb00_ad_foundations.ipynb` — AD from first principles (added after the original
  plan, on request: analytical -> finite differences -> Jacobians -> chain rule -> AD)
- `nb01_halo_exchange_adjoint.ipynb` — the core teaching notebook
- `nb02_distributed_ad.ipynb` — the two multi-rank routes
- `adjoint_operators.py` — hand-written DSL adjoint of `make_periodic`, imported by both
- `mpi_halo_exchange.py` — mpi4py + `custom_vjp` script

Existing notebooks (`example_4dvar.ipynb`, `operators.ipynb`) stay untouched this
round; renumbering them into the series is a later step.

## nb00 outline (added on request)

Foundations, assumed by nb01/nb02. Vehicles: the Tetens saturation-vapour-pressure
formula for the scalar sections, Lorenz-63 for everything vector-valued.

1. One function differentiated by hand
2. Finite differences and the truncation/roundoff V-curve; the cost objection
3. The complex-step trick, as the bridge
4. Dual numbers: forward mode implemented from scratch
5. The Jacobian; analytic Lorenz-63 Jacobian vs `jacfwd`; RK4 as the point where
   hand-differentiation dies
6. The chain rule as a matrix product; associativity gives forward and reverse mode
7. Cost asymmetry, measured
8. Reverse mode from scratch: a tape and a backward pass
9. JAX's API and the AD <-> NWP terminology dictionary
10. The two verification tests: Taylor and dot-product
11. Kinks, memory/checkpointing, and chaos
12. Bridge to nb01

## The teaching arc (nb01)

1. **Framing.** Why halo exchange is the interesting case for AD. Pointwise and
   stencil ops have unsurprising adjoints; the halo update is where the adjoint
   *reverses communication direction* and turns *assignment into accumulation*.
   It is the classic bug site in hand-written adjoint models.

2. **The smallest instance.** 1D periodic field, one halo cell per side, `M=4`.
   Build it with a single `concat_where`. Print the Jacobian as an explicit 0/1
   matrix with `jax.jacrev`. Transpose it by hand on the page.

3. **The aha.** Read the transpose out loud: *accumulate each halo cotangent into
   its owner, then zero the halo*. Note that the halo entry of the input is dead —
   no output depends on it — so its cotangent is structurally zero. That zeroing is
   the thing hand-written adjoints forget.

4. **Linearity.** The halo update is exactly linear, so its tangent-linear model is
   itself and its adjoint is its transpose — no linearization, no Taylor test
   needed here. Verify with the dot-product test `<L x, y> == <x, Lt y>`.
   (Taylor test returns in step 8 for the full nonlinear timestep.)

5. **2D and corners.** Move to the real `make_periodic` from `operators.py`: four
   sequential `concat_where`s, X phase then Y phase. Corners are filled implicitly
   because the J stages read the already-updated I halos. Show that reverse mode
   replays the stages in reverse order, so corner cotangents flow back *through*
   the X phase — the correct ordering for a two-phase exchange, derived not
   specified.

6. **Visualization.** Seed the cotangent of a single corner halo cell with 1.0 and
   plot where the mass lands in the interior. Then seed an edge halo cell. Two
   heatmaps that make the transpose concrete.

7. **The instructive failure.** Implement a plausible-but-wrong adjoint — apply the
   forward halo exchange again instead of its transpose — and let the dot-product
   test catch it. Show the residual. This cell is the point of the notebook.

8. **Adjoint inside the DSL.** Hand-write `make_periodic_adjoint` as a
   `@gtx.field_operator` (four `concat_where`s, reverse order, accumulate
   semantics). Dot-product-test it against `jax.vjp(make_periodic)`. Conclusion:
   the adjoint of a halo exchange stays inside field view — it does not need a
   scatter primitive. Then Taylor-test the full nonlinear `timestep` (which
   contains `make_periodic`) to close the loop.

9. **The MPI dictionary.** A table mapping each forward construct to its adjoint:
   `halo <- owner` to `owner_bar += halo_bar`; `=` to `+=`; halo zeroing; stage
   order reversal. Explain the double-counting failure mode concretely.

## The distributed arc (nb02)

10. **Route A — `shard_map` + `lax.ppermute`.** `ppermute` is a linear collective
    with a transpose rule, so a ring halo exchange written with it should
    differentiate out of the box across devices, with no adjoint code at all.
    *Unverified claim — the first task in this notebook is a 20-line check.*

11. **Route B — GHEX/MPI + `jax.custom_vjp`.** JAX cannot see through MPI, so the
    exchange must be wrapped: forward fills halos, backward accumulates halo
    cotangents into owners and zeroes. The single-rank result from nb01 is the
    *specification* for that backward pass. `mpi4jax` is only
    partial prior art: verified 2026-09-04 against its source, it defines AD rules for
    `sendrecv` and `allreduce` (SUM) only — plain `send`/`recv` have none. The
    point-to-point transpose is exactly what it does not give you.

12. **Oracle design.** Whichever route, the single-rank periodic run is the
    reference: same problem, same Taylor test, distributed gradient must match the
    single-rank gradient to roundoff.

13. **Honest limits.** What single-rank periodic does *not* exercise:
    - owner and halo live in the same array, so the accumulate is a local `+=`;
      no reduction crosses the wire
    - the tape holds a full field per `concat_where` stage, where a distributed
      implementation would tape only halo buffers — a real memory difference
    - `f(I + M)` is a global shift within one field; distributed it is a
      rank-boundary crossing with a completely different domain-inference story

## Phases and verification

| # | Phase | Verify |
|---|---|---|
| 0 | Environment: `uv sync` with the jax extra, run the existing `example_4dvar.ipynb` end to end | 4D-Var notebook executes; cost decreases; recovered IC resembles truth |
| 1 | 1D halo adjoint: explicit Jacobian, hand transpose, `jax.vjp` agreement | printed matrices match; dot-product test residual at roundoff |
| 2 | 2D `make_periodic`: transpose structure, ordering, corner visualizations | dot-product test passes; corner cotangent lands on the diagonally opposite interior cell |
| 3 | The wrong adjoint | dot-product test *fails* with a residual of order 1, and the notebook explains why |
| 4 | `make_periodic_adjoint` as a field operator | matches `jax.vjp(make_periodic)` to roundoff on random cotangents |
| 5 | Taylor test on the full `timestep` including halos | second-order remainder converges at rate ~2 over several halvings |
| 6 | nb01 assembled, prose written, runs top to bottom from a clean kernel | clean run, no stale state |
| 7 | Route A: `ppermute` transpose check, then a sharded SWM step | sharded gradient matches single-rank to roundoff |
| 8 | Route B: `custom_vjp` skeleton with its backward derived from nb01 | dot-product test on the wrapped exchange; multi-rank run if MPI is available |
| 9 | nb02 assembled | clean run |

## Risks and unknowns

- **Branch is off an older `upstream/main`.** It carries substantial local changes
  to `nd_array_field.py`, `common.py`, `foast_to_gtir.py`, `trace_shifts.py`.
  Do not rebase this round — work on the branch as-is.
- **`Dimension.__eq__` returns a `Domain`** (`common.py`, added on this branch) so
  that `I == -1` reads as a mask. Invasive: it breaks the usual `__eq__` contract
  and may surprise dict/set use of `Dimension`. Watch for it; it is a known cost of
  the `concat_where` sugar, not a bug to fix here.
- **Python 3.14 / jax pinning.** The branch's `pyproject.toml` has
  `jax = ['jax>=0.4.26']` without the newer 3.14 guard. If `uv sync` resolves to a
  jax without a 3.14 wheel, pin the venv to 3.13.
- **`concat_where` embedded differentiability** is inferred from reading
  `_concat_where`/`_concat` (slice + `jnp.concatenate`, no masking), not yet run.
  Phase 1 confirms it.
- **`ppermute` transpose rule** is asserted from memory. Phase 7 confirms it before
  anything is built on it.
- **Machine limits.** 32 GB laptop, RTX 2000 Ada. Keep grids small (M=N=16..64) and
  windows short. `GT4PY_BUILD_JOBS` is irrelevant here — embedded mode compiles
  nothing.

## Out of scope this round

- The `jax_jit` runner branch (`claude/jax-jit-backend-integration-oJXQP`)
- Compiled backends (gtfn, dace) — JAX AD does not reach them
- Source-to-source adjoint generation in the GT4Py compiler
- Consolidating or rebasing the four `swm_2026_*` branches
- Unstructured meshes


---

# Round 2 — GHEX-distributed SWM with JAX AD (2026-08-21)

## Goal

A shallow water model distributed over MPI ranks with **GHEX** doing the halo exchange, running
GT4Py field operators on JAX-backed fields per rank, and differentiable in forward and reverse
mode through the exchange. This is nb02's "Route B" made real.

## Steps (as requested)

1. Find out what is missing to run multi-rank MPI on this machine.
2. Build the GHEX-distributed SWM (forward).
3. Make it work forward and backward with JAX + GHEX.

## Design decisions

- **1-D decomposition along I**, a ring of R ranks, each owning `M // R` rows. Periodicity in J
  stays local (`periodic_j` via `concat_where`); periodicity in I is the GHEX exchange with a
  1-row halo and no corners. Chosen because the adjoint trick below is only valid without corners.
- **The exchange is opaque to JAX** (`jax.pure_callback`), wrapped in `jax.custom_vjp`.
- **The reverse exchange reuses the forward GHEX pattern on a scratch buffer.** Put the halo
  cotangents into the interior edge rows of a zero scratch field, run the ordinary forward
  exchange, then accumulate what arrives in the halos into the interior edge cotangents and zero
  the halo cotangent. Valid because a 1-D ring is symmetric; with corner halos an interior corner
  cell would have to send different cotangents to different neighbours through one buffer slot,
  which this trick cannot express.
- Validation is against the **single-rank `make_periodic` model** from nb01, gathered to rank 0:
  forward to ~1e-12, gradient to ~1e-12, plus a distributed dot-product test on the exchange
  alone and a distributed Taylor test on a cost functional.

## Execution

Run as a dynamic workflow (`ghex-swm-ad`): MPI diagnosis (up to two rounds) in parallel with
GHEX install → single-rank forward model; then multi-rank validation only if MPI is unblocked;
then the backward pass (multi-rank if possible, n=1 otherwise); then two adversarial reviewers
(adjoint correctness; reproducibility/honesty) who re-run everything.

## Known risks

- MPI: the hang is in MPICH itself (a pure-C hello also hangs and ignores SIGTERM). Leads: the
  hostname resolves to 127.0.1.1 not loopback; MPICH 4.3's ch4/OFI provider selection.
- GHEX is a source build from PyPI; Boost 1.90, cmake 4.4, g++ 15 are present.
- `jax.pure_callback` under `jax.lax.scan` may or may not survive; fall back to a Python loop.


---

# Round 2b — 2-D decomposition of the GHEX SWM, forward only (2026-09-01)

## Goal

`swm_ghex_2d.py`: the same GHEX-distributed SWM on an Rx x Ry decomposition, forward mode only.
Dropping the adjoint requirement removes the reason round 2 was 1-D (the scratch-buffer adjoint
trick is only valid without corners, nb04 §3.3), so 2-D is now straightforward.

## Design

- Rank r owns the global box `[rx*MLOC,(rx+1)*MLOC) x [ry*NLOC,(ry+1)*NLOC)`; local halo domain
  `{I: (-1, MLOC+1), J: (-1, NLOC+1)}`; the exchanged array is `(MLOC+2, NLOC+2)`.
- **One GHEX exchange fills both face halos and the four corners** (halos `((1,1),(1,1))`,
  periodicity `(True, True)`); no `periodic_j`, no `make_periodic` in the distributed step.
  `operators.timestep` still calls `make_periodic` with the local sizes; those halos are wrong
  for R>1 but are overwritten by the next exchange before anything reads them (nb04's reviewer
  NaN-poisoned them and confirmed nothing reads them).
- The SWM stencils read `p(I+1)(J+1)`, so corners are genuinely consumed: a negative control
  that zeroes the corner halos after each exchange must produce a WRONG result (nb04: ~4e-4).
- Validation: 1x1 (singleton), 2x1, 1x2, 2x2, 2x4/4x2 under `HWLOC_COMPONENTS=-gl mpirun`,
  against `swm_ghex.run_forward_reference` (expect bit-identical, as in 1-D) and against
  nb01's scan forward (~1e-11). Non-divisible rank counts must exit loudly.

## Out of scope

Backward mode. A 2-D adjoint would need a genuine reverse exchange (reversed two-phase, or a
scatter-accumulate the library does not offer) — nb04 shows exactly why; not attempted.
