<!-- Working notes for the AD notebook series in this directory (nb00-nb04,
swm_ghex*.py, halo_lib.py). Written during development, so they reference one
developer's machine: absolute paths under /home/vogtha, two local venvs, and a
laptop-specific MPI workaround. Kept for the verified findings and the record of
what was checked how, not as user documentation. -->

# Status

**Read this first when resuming.** `PLAN.md` holds the goal, the notebook outline
and the phase list; this file holds where we actually are and what has already been
established so it does not get re-derived.

Last updated: 2026-09-01

## Current phase

**The original plan is complete.** Phase 8's multi-rank MPI was unblocked in round 2 (see "MPI: fixed"). Three further pieces were added on request afterwards (nb00, the TL/adjoint
regularization section, nb03). Four notebooks and two modules exist, all executed with
outputs; **12 files are untracked and nothing has been committed** (see Deliverables).

Note nb03 runs in a *different* venv from the rest — see its section below.

| # | Phase | State |
|---|---|---|
| 0 | Environment | done — `uv sync --extra jax`, Python 3.10.16, jax 0.6.2 (CPU) |
| 1 | 1D halo adjoint, explicit Jacobian | done — dot-product residual 0.0 |
| 2 | 2D `make_periodic`, corners, visualisation | done |
| 3 | The wrong adjoints | done — two variants, both caught loudly |
| 4 | `halo_exchange_adjoint` as a field operator | done — bit-identical to `jax.vjp` |
| 5 | Taylor test on the full timestep | done — rates 1.00 / 2.00 over 8 halvings |
| 6 | nb01 assembled and executed | done |
| 7 | Route A: `shard_map` + `ppermute` | done — sharded gradient matches to 8.9e-16 |
| 8 | Route B: MPI + `custom_vjp` | done — backward rule verified on an emulated 4-rank decomposition in nb02, and on REAL 2-rank MPI via `mpi_halo_exchange.py` once MPI was fixed in round 2 |
| 9 | nb02 assembled and executed | done |

### Scope added after the original plan

| added | state |
|---|---|
| `nb00_ad_foundations.ipynb` — AD from first principles | done, reviewed by two agents, all findings fixed |
| nb00 §11 — how TL/adjoint models regularize `max(x,0)` kinks | done, grounded in ECMWF Tech. Memo. 666 |
| `nb03_jax_distributed_arrays.ipynb` — halo-focused | done, reviewed by one agent, all findings fixed |
| MPI/XLA "collective" terminology note (nb03 §5) | done |
| nb03 §9b — `lap(lap(u))` halo-width aggregation | done |

## Deliverables

In `gt4py/examples/next/swm/` on branch `ad_halo`:

| file | cells | figures | env |
|---|---|---|---|
| `nb00_ad_foundations.ipynb` | 48 (24 code) | 6 | gt4py venv |
| `nb01_halo_exchange_adjoint.ipynb` | 31 (13 code) | 2 | gt4py venv |
| `nb02_distributed_ad.ipynb` | 16 (7 code) | 0 | gt4py venv |
| `nb03_jax_distributed_arrays.ipynb` | 41 (20 code) | 1 | **`.venv-jax`** |
| `nb04_halo_exchange_patterns_2d.ipynb` | 54 (24 code) | 2 | gt4py venv |

Plus modules and the round-2 GHEX code:

- `adjoint_operators.py` — `halo_exchange`, its DSL adjoint, the two broken
  variants, `periodic_1d`
- `mpi_halo_exchange.py` — standalone mpi4py + `custom_vjp` script with a
  distributed dot-product test. Passes on 2 and 4 real ranks with `HWLOC_COMPONENTS=-gl`.
- `halo_lib.py` — nb04's illustrative emulated-rank exchange library
- `swm_ghex.py`, `README_ghex.md` — GHEX-distributed SWM, forward, 1-D ring (round 2)
- `swm_ghex_2d.py` — the same on an Rx x Ry decomposition, forward (round 2b)
- `swm_ghex_ad.py` — its backward pass; completed before the forward-only cut, not built on

**12 files untracked on `ad_halo`; nothing committed.**

Reading order is nb00 → nb01 → nb02 → nb04, with nb03 as a companion to nb02.

## MPI: fixed (was the one blocker)

**`HWLOC_COMPONENTS=-gl mpirun -n N ...`** — that is the whole fix. Verified by me on
2026-09-01: pure-C hello on 2 and 4 ranks (rc=0), control without the variable still hangs.
The workflow's MPI agent verified it for mpi4py and for `mpi_halo_exchange.py` (dot-product
`2.09e-16 / PASS` on 2 ranks).

Root cause (strace-level, by the workflow agent; consistent with everything observed before):
`hydra_pmi_proxy` loads hwloc; hwloc's `gl` component probes X displays `:0, :1, ...` for the
NVIDIA NV-CONTROL extension regardless of `$DISPLAY`; `:1` is a lazy Xwayland listener owned by
gnome-shell whose accept backlog (1) is full and never accepts, so `connect()` blocks forever and
every rank waits for its PMI `get kvsname` reply forever. This affects EVERY hydra launch,
`mpirun -n 1` included (the reviewer confirmed n=1 hangs too); only MPI *singleton* init —
plain `python script.py` with no `mpirun` — bypasses hydra and works. **Not** the 127.0.1.1 hostname, not
UCX/OFI (MPICH here is ch4:ucx and never reaches netmod init), not Wave's cgroup. My earlier
hostname lead was a red herring; the agent correctly discarded it.

Operational rules that still stand: wrap `mpirun` in `timeout -s KILL`; clean up with
`killall -q -9 mpiexec.hydra hydra_pmi_proxy`; never `pkill -f`. A 4-rank JAX run under the
`heavy` 12 GB cap was OOM-killed once — prefer 2 ranks under `heavy`, or run 4 without it.
`mpi4py` (and now `ghex`) are installed with `uv pip install`, not in `pyproject.toml`; `uv sync`
drops them. Saved to memory as `mpirun-needs-hwloc-gl-disabled`.

## Housekeeping: notebook generators are gone

The nbformat generator scripts (`gen_nb00..03.py`) lived in the session scratchpad under
/tmp and were wiped when the day rolled over. **The .ipynb files in the repo are the only
source of truth now.** Edit them in place with nbformat (as was done to revive nb02's MPI
cell) and re-execute with `tmp/run_nb.py <notebook>` (gt4py venv; nb03 needs `.venv-jax`
and kernel `jax-latest`). Prefix the executor with `HWLOC_COMPONENTS=-gl` for any notebook
that launches `mpirun`.

## Established findings

### Verified by running

- **`jax.vjp` works through embedded `concat_where`.** No special handling needed.
  Fields must be built with `gtx.as_field(domain, array, allocator=jnp)` — passing a
  jnp array without `allocator=jnp` silently produces a `NumPyArrayField`, which
  `jax.vjp` then rejects as "not a valid JAX type".
- **The hand-written DSL adjoint is bit-identical to `jax.vjp`** (max diff exactly
  0.0), and the dot-product test holds at 7e-16.
- **`jax.lax.ppermute` has a transpose rule**, confirmed directly: the transpose of a
  ring shift is the inverse permutation. A `shard_map`ped GT4Py pipeline over 4 CPU
  devices differentiates with no adjoint code and matches the single-device gradient
  to 8.9e-16.
- **Taylor test** through 10 leapfrog steps (so 10 halo exchanges in the backward
  pass): second-order rate 2.00, clean over 8 halvings.
- **Both adjoint bugs are caught by the dot-product test** — relative errors 1.6 and
  9.4, not roundoff.
- **nb00 numerics** (all self-verifying in the notebook): dual-number and tape
  derivatives exact to 0.0 against the analytic Tetens derivative; analytic Lorenz-63
  Jacobian matches `jacfwd` exactly; product of 200 step Jacobians matches
  `jacfwd` of the whole run to 1.2e-14; Lyapunov exponent converges to 0.900 at
  t=2000 (published 0.906); `jacfwd` vs `jacrev` for R^n -> R is ~31x at n=200
  (the "95x" first measured was a benchmarking artifact — see the nb00 review below).
- **JAX's derivative at a kink depends on the spelling.** Four ways of writing ReLU,
  identical values *everywhere including x=0*, but **three** distinct derivatives at the
  kink: `maximum(x,0)` gives 0.5, `where(x>0,x,0)` gives 0.0, and both `where(x>=0,x,0)`
  and `0.5*(x+abs(x))` give 1.0. This became the centrepiece of nb00 section 11.

### Established by reading (unchanged from before)

`operators.py:make_periodic` applies four sequential `concat_where` stages, X phase
then Y phase, on a field with domain `(-1, M+1) x (-1, N+1)`. Corners are filled
implicitly by the J stages reading already-updated I halos. Embedded `concat_where`
is slicing plus `jnp.concatenate` (`nd_array_field.py` ~990 and ~901), no masking.

Transpose of one stage `concat_where(J == N, f(J - N), f)`: `in_bar[0] += out_bar[N]`,
`in_bar[N] = 0`. Accumulate into owner, zero the halo — the MPI reverse exchange.

The base branch's `example_4dvar.ipynb` already ran 4D-Var through GT4Py field
operators with JAX AD; it avoids `@program`/`out=` by calling
`gtx_timestep.definition`.

### Design decision worth remembering

nb01's central object is **`halo_exchange` (halo → halo)**, not `make_periodic`
(interior → halo). Reason: with interior → halo the input has no halo, so the
*missing-zeroing* bug is invisible and the dot-product test passes anyway. Only the
halo → halo framing — which is what a distributed exchange actually does, discarding
the old halo — exposes both bugs one-shot.

## nb00 review outcome

Two independent reviewers audited the tutorial. Both reviews were substantive; the
notebook was regenerated with every finding addressed. The ones worth remembering:

**Errors that were in the notebook and are now fixed**

- The Magnus coefficients 610.94 / 17.625 / 243.04 are **Alduchov & Eskridge (1996)**,
  not Tetens (whose are 610.78 / 17.27 / 237.3). Was misattributed.
- Reverse-mode AD **predates** the meteorological adjoint work (Linnainmaa 1970/1976,
  Speelpenning 1980), and the adjoint reached atmospheric science as sensitivity
  analysis (Marchuk 1974; Hall, Cacuci & Schlesinger 1982) before 4D-Var. The original
  text had the chronology backwards.
- The kink section said "four different derivatives" — there are **three**
  (0.5, 0.0, 1.0, 1.0), and it also claimed the four spellings differ in *value* at
  x=0, which is the opposite of the point.
- Cost per pass "~1-3x forward" is impossible below 2x; Griewank & Walther give
  [2, 5/2] and [3, 4]. Now 2-3x / 3-4x.
- "AD never forms the Jacobian" contradicted section 9, where jacfwd/jacrev do exactly
  that. Now "never *needs* to".
- Prose said roundoff amplifies to eps|f|/h while the displayed formula said 2eps|f|/h.

**Methodological defects that were producing wrong numbers**

- The "95x" jacfwd/jacrev ratio was an artifact: `timeit` took the min over batch
  *means*, which destroys most of the min-filter's noise rejection. The stored figure
  showed jacrev *decreasing* from n=100 to n=200, which is impossible. With
  reps=300/batches=20 the ratio is ~31x and monotonicity now holds (asserted in the
  cell). Absolute timings are CPU-only and noisy; the notebook now says so.
- The finite-difference cell quoted `min()` over the h-sweep. That minimum is an
  accidental truncation/roundoff cancellation — it moved by two orders of magnitude
  with the number of sample points. Now reports the error at the *predicted* optimum
  plus a local median, and names the fluke.
- The checkpointing cell asserted "less memory" while only comparing gradients. Now
  measures it: 49,288 -> 12,872 bytes of XLA temporaries.
- The Lyapunov reference line was anchored at t=0.25, before the tangent vector aligns
  with the leading Lyapunov vector, so the data appeared to drift *below* e^(0.906t).
  Anchored at t=2 with evenly spaced lead times. (A log x-axis was tried and reverted —
  it bends the exponential and destroys the straight-line reading.)

**Latent bug the reviewers found in the teaching code**

The from-scratch reverse-mode `Var` class had no `__sub__`/`__rsub__`/`__neg__`, so it
worked only because the Tetens formula happens to contain no minus sign. Worse, the
demo used a function where every intermediate is consumed exactly once, so the `+=`
accumulation — the entire point of section 8, and the same accumulation the halo
adjoint relies on in nb01 — was never exercised. Both fixed: the operators are there
and a second cell differentiates a function with a thrice-used intermediate, checked
against `jax.grad`. (The reviewer independently confirmed the accumulation logic was
in fact correct on seven reuse patterns — the bug was in the *test*, not the tape.)

**Verified correct and left alone:** all five original references (right volume, year,
authors), the Lorenz-63 Jacobian entry by entry, the RK4 step Jacobian to 4e-19 against
a hand-derived chain rule, the 200-Jacobian product, the Taylor and dot-product tests,
every kink value under jit/float32/-0.0, the Lyapunov convergence toward 0.906, the
subdifferential claim, the 6-12h 4D-Var window, and the finite-difference figure's
inverted x-axis.

## nb00 section 11: TL/adjoint regularization (added on request)

Literature-grounded subsection on how operational models handle `max(x,0)`-type kinks.
Primary source: **ECMWF Tech. Memo. 666** (Janiskova & Lopez 2012), sections 3.2 and
5.1, read directly rather than summarised second-hand.

The finding that shaped the section: "replace max with a smooth function" is only one
of three techniques, and not the most used. The three are (1) flatten the offending
nonlinear function (Sundqvist autoconversion, their Fig. 1); (2) freeze the
perturbation -- vertical diffusion used `K' = 0` outright (Mahfouf 1999); (3) damp the
perturbation near the trouble, "more significantly reduced around the neutral state
(Ri close to zero) ... eased exponentially away from the neutral state". Techniques 2
and 3 alter **only the derivative**, leaving the nonlinear model untouched — so the
operational TL is deliberately not the derivative of the operational model.

Content added:
- the three standard surrogates (softplus, sqrt-smoothing, Gaussian) plus a figure of
  them and their derivatives
- the identity `E_xi[max(x + eps*xi, 0)] = x*Phi(x/eps) + eps*phi(x/eps)`, whose
  derivative is exactly `Phi(x/eps)` (verified: 1.1e-16 against the closed form, and by
  Monte Carlo to 6.7e-4). This makes the smoothed Heaviside the honest derivative of
  the grid-box-mean model, and connects to statistical cloud schemes
  (Sommeria & Deardorff 1977; Mellor 1977).
- `jax.custom_jvp` as the exact mechanism for "unchanged nonlinear model, regularized
  linearization"; note that it suffices for both modes since the tangent rule is linear
  and JAX transposes it. `custom_vjp` is needed only when the backward pass needs
  something the forward cannot express (nb02's MPI case).
- the price, measured: Taylor r2 rates are 1.00 for exact `max`, 2.00 for a smoothed
  *function*, and 1.00 for the `custom_jvp` regularized TL. Smoothing the derivative
  alone does not and cannot restore second order — the gradient is deliberately not
  grad J. Practical lesson recorded: check for deliberate regularization before
  debugging a first-order Taylor test.

An earlier attempt to demonstrate the benefit via a TL-vs-finite-difference crossover
was **discarded** — with a summed functional over N points the curvature term
(O(N h^2)) swamps the first-order term (O(sqrt(N))), so the apparent crossover was a
statistical artifact of the setup, not the physics. The ECMWF before/after result is
cited instead of being faked with a toy.

## nb03: JAX distributed arrays (added on request, halo-focused)

`gt4py/examples/next/swm/nb03_jax_distributed_arrays.ipynb` — 41 cells, executed.

**Runs in a SEPARATE venv**: `.venv-jax` (Python 3.13, jax 0.11.1), registered as Jupyter
kernel `jax-latest` / "Python (jax 0.11)". nb00-nb02 stay on the gt4py venv (Python 3.10,
jax 0.6.2) and are untouched — latest jax needs Python >= 3.12, and nb03 has no gt4py
dependency. Recreate with `uv venv --python 3.13 .venv-jax` +
`uv pip install --python .venv-jax jax matplotlib ipykernel rich`.

### Findings that shaped it

- **`jax.make_mesh` now defaults to `AxisType.Explicit`.** Under Explicit sharding
  `jnp.roll` along a sharded axis is a `ShardingTypeError`, as is a non-divisible slice.
  The "write global code, let GSPMD infer the halo" advice in older tutorials is rejected
  by default in current jax. `jax.sharding.Mesh(...)` still defaults to Auto.
- API moved: `jax.shard_map` is top level, `check_rep` -> `check_vma`, `jax.set_mesh`
  exists, `jax.experimental.shard_map` is empty.
- **Explicit mode does NOT reject everything that communicates.** `a[::-1]` is accepted and
  emits a whole-shard `collective-permute[8,16]`. What it checks is that sharding
  propagation is unambiguous *and implemented* (one of the three error messages literally
  ends "is not implemented" — a missing rule, not a safety rail).
- **`lap^n` costs 2n width-1 messages, linear not combinatorial**, and the byte total is
  already optimal. What is lost is message count and, crucially, *sequencing*: the n
  exchanges are dependent, so n latency hits instead of 1. Merging them is impossible at
  HLO level because the second exchange carries the *output* of the first stencil — you
  would have to change what is computed where (ghost-zone expansion / overlapped tiling).
  That is an iteration-domain transformation, which is exactly what GT4Py's domain
  inference does and flat HLO structurally cannot. Good DSL-vs-compiler argument.
- Multi-process works for real here: 2 processes, gloo, ring shift across the process
  boundary, `is_fully_addressable: False`.

### Review outcome (independent agent) — a serious methodology bug of mine

The first draft's `collectives()` helper matched the opcode **anywhere on an HLO line**,
so it counted every line that merely *consumed* a collective. GSPMD names instructions
after their opcode (`%collective-permute.1`) so consumer lines matched; `shard_map` names
them after JAX primitives (`%ppermute.6`) so they did not. **The over-count was
one-directional**, and it manufactured three false conclusions that I had already reported
to the user:

1. "manual `shard_map` emits strictly less communication than Auto" — false, they are
   byte-identical (256 B, same two ring permutes).
2. a `collective-permute[8,16]` whole-shard move in the five-point — does not exist.
3. "GSPMD degrades to whole-shard transfers for corners" — false. The 2-D nine-point is 8
   collectives, 0 whole-shard: a correct two-phase corner exchange. The real defect is much
   smaller — 2 of the 8 permutes are exact duplicates (a CSE miss).

Fixed helper anchors on the defining instruction (`%name = shape opcode(`) and skips
`-done` legs. Corrected numbers: 5-point 2 permutes / 256 B; edge-pad 8 permutes / 4,480 B
(~17x the necessary volume, via a `partition-id`/`iota`/`select`/`dynamic-slice` fallback);
2-D 5-point 4; 2-D 9-point 8.

Also fixed: `JAX_PLATFORMS=cpu` added (the notebook would have crashed on a GPU box, since
`xla_force_host_platform_device_count` only affects the CPU backend); the child script now
goes to `tempfile.mkdtemp()` instead of polluting the repo; timeout now kills orphans;
"touching a remote element raises" corrected (indexing works and is *collective* — the
deadlock hazard is the real caveat; what raises is host materialisation).

**Lesson worth keeping:** a measurement tool with a systematic one-directional bias is far
more dangerous than a noisy one. It produced clean, plausible, quotable numbers that all
pointed the same wrong way.

## Round 2: GHEX-distributed SWM (workflow `ghex-swm-ad`, run `wf_6bcbaaed-3d2`)

Scope was cut mid-run at the user's request to **forward only**; the backward agent had
already completed before the cut, so its results are recorded below but are NOT built on
further. Script (edited to forward-only, resumable):
`~/.claude/projects/-home-vogtha-claude-gt4py-autodiff-swm/0bcfdbcc-8eda-4022-b740-f503b0b9c375/workflows/scripts/ghex-swm-ad-wf_6bcbaaed-3d2.js`

Results from the workflow journal (each by its agent; MPI recipe re-verified by me):

| stage | result |
|---|---|
| MPI diagnosis | **fixed**: `HWLOC_COMPONENTS=-gl` (see "MPI: fixed"); 12 things tried, strace-level root cause |
| GHEX install | `ghex` 0.9.0 built from PyPI into the gt4py venv with CC/CXX=mpicc/mpicxx; `import ghex` ok; 1-rank exchange check passes (C- and F-order, mixed periodicity) — `tmp/ghex_single_rank_check.py` |
| forward SWM | `examples/next/swm/swm_ghex.py` + `README_ghex.md`; 1-D ring decomposition along I, `periodic_j` locally, GHEX I-halo via `jax.pure_callback`; single-rank matches the `make_periodic` reference, max diff **0** |
| multi-rank forward | 2 ranks match single-rank, max diff **0** |
| backward (pre-cut) | `swm_ghex_ad.py`; scratch-buffer reverse exchange as `custom_vjp`; distributed dot-product rel err **2.18e-16**, Taylor rate2 **2.00**, 2-rank gradient matches single-rank |
| review | **done, clean on substance**: from a clean shell the reviewer reproduced the MPI fix, single-rank bit-identity, 2/4 ranks — and additionally 8 and 16 ranks — bit-identical, 3 ranks failing loudly, `mpi_halo_exchange.py` on 2 and 4 ranks, and `ghex_exchange` purity under repeated eager calls and `jit`. Nothing claimed as tested was untested. |

Documentation-level findings, all applied and re-verified on 2 ranks (diff 0.0, rc=0):
- README and module docstring launch recipes lacked `HWLOC_COMPONENTS=-gl` (hung as written);
  README "Status" was stale. Fixed; Status now records 1/2/4/8/16 ranks.
- "bit-identical to the nb01 forward model" was overstated: it is bit-identical to an
  in-script Python-loop reference; against nb01's `lax.scan` forward it is ~1.5e-11 in p
  (different compilation path). README reworded.
- `ghex_exchange` now checks shape/dtype up front instead of failing inside the callback.
- Root-cause narrative claimed n=1 works without the fix — false, corrected above and in memory.
- The reviewer flagged nb02 as "modified by an unidentified step" — that was me, reviving its
  MPI cell with the fix (sanctioned).

Caveat carried from the MPI agent: a 4-rank JAX run under `heavy`'s 12 GB cap was
OOM-killed; 2 ranks under `heavy`, or 4 without it.

## nb04: 2-D halo-exchange patterns with an illustrative library (added on request)

`examples/next/swm/nb04_halo_exchange_patterns_2d.ipynb` (54 cells, 2 figures) +
`halo_lib.py` (pure-NumPy emulated-rank exchange library with explicit `Message` objects,
two-phase / single-phase / faces-only / ring patterns, `exchange_adjoint`,
`exchange_adjoint_scratch`, dense `exchange_matrix`). Generator survives in
`tmp/gen_nb04.py`. Independent review complete: **every numeric claim reproduced**;
14 findings (citation precision in §6, a `mloc >= halo` guard, an invisible figure guide,
a wrong explanation of the 5.8e-12 u/v gradient residual — it is cancellation of 1e5-scale
p cotangents, not message ordering) — **all 14 applied**, notebook regenerated and
re-executed clean (verified: 0 error outputs, every code cell executed). Adding one cell
shifted the shared RNG, so random-dependent numbers changed and prose was rewritten from
the new outputs (2-D scratch-trick dot-product error now 2.42; structural results
unchanged). `JAX_PLATFORMS="cpu"` is now pinned unconditionally there.

Established by nb04 (verified by author and reviewer):
- SWM on a **2x2 decomposition with corners** via `custom_vjp` + `pure_callback` is
  bit-identical to the single-rank model in forward mode (same-compilation-path caveat:
  eager vs scan differ at 7e-12 in p); gradient matches to 5.8e-12 (u,v) / 6e-16 (p);
  Taylor rate2 2.00. Faces-only exchange (no corners) gives WRONG u,v (~4e-4).
- **The scratch-buffer adjoint trick is exact on a ring iff every interior slot feeds
  one neighbour (mloc >= 2h), and wrong in 2-D**: rel error 2.42 on 3x2 (1.27 before the RNG shift); corner-halo
  cotangent lands nowhere, corner-adjacent face cotangent lands in three places; a ring of
  single-row blocks (mloc = h) also fails. This is the precise justification for round 2's
  1-D design and for `swm_ghex_ad.py`'s `MLOC >= 2` guard.
- Real-world section, source-verified: Atlas `HaloExchange::execute_adjoint` (pack halo,
  reverse send, accumulate into send region, `zero_halos`) — a shipped adjoint halo
  exchange doing exactly nb01's rule; ICON `exchange_data` has add-on-receive; GHEX has
  no adjoint/accumulate mode; IFS Part VI documents `SLCOMM1`/`SLCOMM2A` and `VMAX2`
  (so the routine name IS public); **ecTrans `trltomad`/`trmtolad` files are pack/unpack
  kernels, not adjoint transposition routines — the adjoint reuses the forward
  `TRLTOM`/`TRMTOL` in the opposite direction** (correction of an earlier statement).

## Round 2b: 2-D GHEX decomposition, forward only — done

`examples/next/swm/swm_ghex_2d.py` (+ a section in `README_ghex.md`). Rx x Ry blocks, one
GHEX exchange per field per step with halos `((1,1),(1,1))` and periodicity `(True, True)`;
no `periodic_j`, no `make_periodic` in the distributed step.

Established (build agent; 2x2, 1x3 and 4x2 re-verified by me after a fix):
- **GHEX's structured `HaloGenerator` fills the four corners across ranks in a single
  exchange.** 2x2 with interior `1000*i + j`: every halo cell of every rank, faces and
  corners, equals `np.pad(global, 1, mode="wrap")` (`tmp/ghex_2d_corner_check.py`). So no
  two-phase fallback was needed.
- Forward results are **bit-identical** (0.0) to the in-script reference on 1x1 (singleton),
  2x1, 1x2, 2x2, 4x2, 2x4, and 2.6e-14 (u,v) / 1.455e-11 (p) against nb01's `lax.scan`
  forward — the same numbers as the 1-D version, i.e. the decomposition is invisible.
- **Corners matter**: zeroing the four corner halos after each exchange (scratch copy) gives
  max abs diff 1.6e-3 in u,v and 0.27 in p on 2x2 after 10 steps. Compare nb04's faces-only
  ~4e-4: here corners are zero against p ~ 5e4, hence larger.
- Loud exits: `Rx*Ry != SIZE` and non-divisible layouts both `SystemExit` on every rank.

Fix applied after delivery: the agent had imported parameters and the reference from
`swm_ghex.py`, which runs that module's 1-D GHEX setup and its `M % SIZE` check at import —
making the 2-D divisibility check unreachable and giving `1 3` the wrong message. The 2-D
module is now standalone (≈15 duplicated lines); `1 3` reports
`M=16 N=16 is not divisible by the 1x3 layout`; 2x2 and 4x2 still 0.0. Not independently
reviewed beyond my own re-runs.

Backward for 2-D remains out of scope: it needs a genuine reverse exchange (nb04 §3.3).

## Literature: does reverse mode through halo exchanges make sense? (researched 2026-09-04)

Full survey by a research agent; the load-bearing claims below were re-verified by me.
**Short answer: yes, it is production-standard since ~2002; communication is verifiably
NOT the bottleneck; tape memory and checkpointing are.**

### The rule we derived is the canonical one

Heimbach, Hill & Giering (2005), *Future Gener. Comput. Syst.* 21(8):1356, Table 2 states
the duality verbatim: **"send & assign <-> receive & accumulate"** (also gather <-> scatter,
read & assign <-> write & accumulate). Utke et al., *Toward Adjoinable MPI* (PDSEC/IPDPS
2009) gives the formal rules including non-blocking, and its **worked case study is a
ghost-cell exchange on MITgcm** — literally this problem.

### Production practice: everyone hand-writes it

MITgcm/ECCO, ROMS (operational at NOAA), IFS 4D-Var and NEMOTAM all implement the reverse
exchange **by hand** and declare it to the AD tool, rather than differentiating through MPI.
The sharpest datapoint: the MITgcm-AD v2 authors *wrote* the AMPI paper and still declined
to use it (quote in the round-2 notes). NEMOTAM: "In order to handle properly the
multi-processor aspects and optimise computing cost, the current NEMOTAM is hand-coded."
So `swm_ghex_ad.py`'s hand-written `custom_vjp` is the production idiom, not a shortcut.

### Cost: 2-6x, and not because of communication

| code | adjoint / forward | note |
|---|---|---|
| MITgcm/ECCO (TAF) | **2.5x** adjoint sweep, **5.5x** with 3-level checkpointing | Heimbach 2005 |
| MITgcm verification cases (TAF) | 2.0x / 4.0x / 5.6x | MITgcm-AD v2 Table 3 |
| MITgcm `streamice` (Tapenade) | **7.0x -> 2.4x** from checkpoint *placement* alone | arXiv:2405.15590 |
| FESOM2-JAX | 4.7x | arXiv:2608.01546 |
| Tapenade + AMPI, CFD, untuned | 15.4x (3.9x of it checkpointing) | arXiv:1912.11717 |

Communication is unchanged, from three independent primary sources: Heimbach's analytic
cost model carries the exchange term over verbatim ("T_PSexch, which remains unchanged,
since the exchange pattern is unaltered"); MITgcm-AD v2 says the communication "should not
contribute to the adjoint slowdown"; and Cardesa's per-routine slowdown table is captioned
"None of the routines above include parallel communication calls within them". Enzyme's
analytic bound is 2x MPI calls, <=3x MPI memory.

**What dominates is tape memory and the recomputation bought to contain it.** ECCO needs
**96 cores for an adjoint where the forward runs on 12** — an 8x memory factor. Naive
MITgcm trajectory storage is ~1 TB. ECCO carries 1458 hand-inserted `CADJ` storage
directives.

### Two findings that cut against intuition

- **The adjoint often strong-scales BETTER than the forward**, because it has a higher
  compute-to-communication ratio. OpenFOAM+dco: the primal stops scaling at 192 threads
  while the adjoint keeps scaling to 768. PETSc TSAdjoint at 0.5 B DOF: forward is
  communication-limited, "For the adjoint solve, however, perfect linear scaling is
  observed."
- **`mpi4jax` is NOT prior art for point-to-point.** Verified by me against its source
  2026-09-04: only `sendrecv` and `allreduce` define AD rules; `send`, `recv`, `bcast`,
  `gather`, `scatter`, `alltoall` have none. An earlier claim of mine to the contrary was
  wrong and is corrected in PLAN.md.

### The closest analogue to this project

**FESOM2-JAX** (arXiv:2608.01546, Aug 2026): an ocean model on JAX `shard_map` where "the
adjoint of a halo read is an additive scatter back to the owner", with **four halo
transports compared by whether their adjoint is exact**, 4.7x measured, run to 256 GPUs.
It gets the transpose free by writing the exchange *in the differentiable language itself*
— architecturally what a GT4Py-level exchange would have to do. It also found
`lax.ragged_all_to_all` has a **defective reverse-mode rule in JAX 10.1**, usable
forward-only: the transpose of a communication primitive is where a framework's rules get
stressed.

Firedrake's "at the right abstraction the problem vanishes" (Farrell et al. 2013) has a
hard boundary that is exactly ours: user-defined PyOP2 kernels cannot be differentiated by
UFL, and a stencil DSL is precisely that regime.

### Tooling status 2026 (verified)

Tapenade **deprecated AMPI in April 2026** in favour of adMPI (no `MPI_Waitall`, no
one-sided, turn-point insertion still manual). Enzyme is the only tool doing non-blocking
reverse mode without annotation, but treats `MPI_Test` as inactive — complete a request
with Test instead of Wait and you silently get no adjoint communication.

## Open for a future round## Open for a future round

- The MPI hang (environment, not code)
- Overlapped / non-blocking exchange schedules — the adjoint must reverse the
  *ordering*, and nothing here tests that
- Unstructured halos: transpose of a gather through an index list is a scatter-add
  with duplicate targets
- Compiled backends (`gtfn`, `dace`) — JAX AD does not reach them; would need a
  `custom_vjp` whose backward is a generated adjoint program
- Folding the existing Taylor-test / 4D-Var / hybrid-NN material into a numbered
  tutorial series alongside nb01/nb02

## Log

- **2026-08-20** — Cloned `havogt/gt4py` and `havogt/SWM`. Surveyed the four
  `swm_2026_*` branches. Created branch `ad_halo` off `swm_2026_halo_update`. Wrote
  `PLAN.md` and this file.
- **2026-08-20** — Implemented the whole plan. Environment up; all numerical claims
  verified; both notebooks written and executed with outputs; MPI blocked and
  documented. Files left untracked on `ad_halo`, not committed.
- **2026-08-20** — Added `nb00_ad_foundations.ipynb`, the AD introduction, on
  request. Also installed `jupyterlab` into the venv with `uv pip install` (like
  `mpi4py`, not in `pyproject.toml`). Still nothing committed.
- **2026-09-01** — Round 2 workflow: MPI fixed (`HWLOC_COMPONENTS=-gl`), GHEX 0.9.0
  installed, forward + 2-rank SWM match to 0, backward completed before the forward-only
  cut. nb02's MPI cell revived (4 real ranks, PASS). nb04 written and reviewed; fixes in
  progress. Process crashed once; workflow and agents resumed from transcripts, nothing lost.
  Scratchpad generators for nb00-03 lost; `.ipynb` files are the source of truth.
- **2026-09-01** — Round 2b: `swm_ghex_2d.py`, 2-D decomposition forward-only. GHEX fills
  corners across ranks in one exchange; bit-identical on 2x1..4x2. Import coupling to the
  1-D module removed after delivery.
- **2026-08-20** — Two independent review agents audited nb00 (one on maths/prose/
  references, one re-deriving all numerics). 20 findings, all addressed; nb00 is now
  40 cells. See "nb00 review outcome" below.
