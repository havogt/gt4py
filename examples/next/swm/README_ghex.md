# SWM distributed with GHEX (`swm_ghex.py`)

Forward-mode shallow water model on a 1-D ring decomposition along `I`, halo
exchange through [GHEX](https://github.com/ghex-org/GHEX) 0.9.0, local state as
JAX-backed `gt4py.next` fields. Structured so the exchange can later be
differentiated: `ghex_exchange` is a `jax.custom_vjp` around `jax.pure_callback`
whose backward rule currently raises `NotImplementedError`.

## Requirements

`mpi4py` and `ghex` in the venv (both built against the system MPICH):

```
CC=mpicc CXX=mpicxx MPICC=mpicc MPICXX=mpicxx CMAKE_BUILD_PARALLEL_LEVEL=4 heavy ./.venv/bin/pip install ghex
```

## Run

From `gt4py/` (paths relative to the repo root):

```
cd examples/next/swm
../../../.venv/bin/python swm_ghex.py                       # single rank, MPI singleton init (no mpirun)
HWLOC_COMPONENTS=-gl timeout -s KILL 120 mpirun -n 4 ../../../.venv/bin/python swm_ghex.py   # R ranks, R must divide M
killall -q -9 mpiexec.hydra hydra_pmi_proxy                 # only needed if a run was killed
```

`HWLOC_COMPONENTS=-gl` is required on this machine for *every* `mpirun` launch,
including `-n 1`: hydra's hwloc topology probe otherwise blocks on a stale X display
socket and `MPI_Init` never returns (see `STATUS.md`, "MPI: fixed"). Wrap `mpirun` in
`timeout -s KILL`; hung MPICH ranks ignore SIGTERM.

Rank 0 prints, per prognostic field, the max abs difference between the gathered
distributed result and an in-script single-process reference (`make_periodic` on the
full global domain, stepped with the same Python loop), then `PASS`/`FAIL`. Against
nb01's `jax.lax.scan`-based forward the distributed result agrees to ~1e-11 in `p`
(roundoff from a different compilation path), not bit-for-bit.

Parameters are hard-coded to the nb01 values: `M = N = 16`, `dx = dy = 1e5`,
`dt = 90`, `a = 1e6`, `alpha = 1e-3`, 10 steps.

## Layout

- rank `r` owns global rows `[r*MLOC, (r+1)*MLOC)`, `MLOC = M // R`, all `N` columns
- local field domain `{I: (-1, MLOC+1), J: (-1, N+1)}`; the exchanged array is the
  full `(MLOC+2, N+2)` halo array, registered with GHEX as a block of an
  `M x (N+2)` global grid with halo `(1, 0)` and periodicity `(True, False)`
- one step: `ghex_exchange` (I halos) -> `periodic_j` (J halos, incl. corners) for
  `u, v, p` -> `operators.timestep` with `M = MLOC`
- `make_periodic` inside `timestep` leaves locally-periodic `I` halo rows on the
  outputs; they are wrong for `R > 1` but are overwritten by the next exchange
  before anything reads them
- `gather_to_root` collects the interior blocks on rank 0 (`comm.gather`)

## Status

Verified: 1 rank (singleton init) and 1, 2, 4, 8, 16 ranks under `mpirun` all reproduce
the in-script reference bit-for-bit (max abs diff 0.0). 3 ranks exits with
`M=16 is not divisible by 3 ranks` before any GHEX/JAX work. The exchange behaves as a
pure function under repeated eager calls and under `jit`. Independently re-verified by a
review agent from a clean shell.

The time loop is a plain Python loop (not `jax.lax.scan`/`jit`) so the three
`pure_callback` exchanges per step execute in program order on every rank; the
reference uses the same loop so the comparison is bit-exact.

## 2-D decomposition (`swm_ghex_2d.py`)

Same model on `Rx x Ry` blocks; both periodic directions are one GHEX exchange.
Parameters and the single-process reference are the same as in `swm_ghex.py`
(duplicated rather than imported, so this module does not run the 1-D module's GHEX
setup and rank check at import time).

### Run

```
cd examples/next/swm
../../../.venv/bin/python swm_ghex_2d.py 1 1                    # single rank, MPI singleton init
HWLOC_COMPONENTS=-gl timeout -s KILL 300 mpirun -n 4 ../../../.venv/bin/python swm_ghex_2d.py 2 2   # Rx Ry, Rx*Ry == ranks
killall -q -9 mpiexec.hydra hydra_pmi_proxy                     # only needed if a run was killed
```

Without arguments the layout is `SIZE x 1`. `Rx*Ry != SIZE`, or a layout that does not
divide `M x N`, exits with a `SystemExit` message on every rank naming the offending
dimension (e.g. `1 3` on 3 ranks: `M=16 N=16 is not divisible by the 1x3 layout`).

### Layout

- rank `r` owns block `(rx, ry) = divmod(r, Ry)`: rows `[rx*MLOC, (rx+1)*MLOC)`,
  columns `[ry*NLOC, (ry+1)*NLOC)`, `MLOC = M // Rx`, `NLOC = N // Ry`
- local field domain `{I: (-1, MLOC+1), J: (-1, NLOC+1)}`; the exchanged array is the
  full `(MLOC+2, NLOC+2)` halo array, registered with GHEX as that block of the `M x N`
  global grid with halos `((1,1),(1,1))`, periodicity `(True, True)`, offsets `(1, 1)`
- one step: `ghex_exchange` for `u, v, p` -> `operators.timestep` with
  `M = MLOC, N = NLOC`; no local periodic step (no `periodic_j`, no `make_periodic`)
- `make_periodic` inside `timestep` leaves locally periodic halos on the outputs; wrong
  for `Rx > 1` or `Ry > 1`, overwritten by the next exchange before anything reads them
- `gather_to_root` places the interior blocks by `(rx, ry)` on rank 0

### Corners

`timestep` reads the diagonal neighbour (`avg_x(avg_y(p))` at `(MLOC-1, NLOC-1)` reads
`p(MLOC, NLOC)`), so the four corner halo cells must carry the diagonal block's values.
GHEX's structured `HaloGenerator` with halos in both dimensions fills them in the same
exchange: on 2x2 with the interior set to `1000*i + j`, every halo cell of every rank,
faces and all four corners, equals `np.pad(global, 1, mode="wrap")` (scratch check
`tmp/ghex_2d_corner_check.py` at the project root). Negative control: zeroing the four
corner cells after each exchange in a scratch copy gives max abs diff 1.6e-3 in `u`, `v`
and 2.7e-1 in `p` on 2x2 after 10 steps (`FAIL`); nb04's faces-only exchange measured
~4e-4 in `u`, `v`.

### Status

Verified: `1 1` (singleton init) and, under `mpirun`, `2 1`, `1 2`, `2 2`, `4 2`, `2 4`
reproduce the in-script reference bit-for-bit (max abs diff 0.0) and agree with the nb01
`lax.scan` forward to 2.6e-14 in `u`, `v` and 1.5e-11 in `p` (printed alongside).
`3 1` and `1 3` on 3 ranks and `2 1` on 4 ranks exit with a message on every rank
(rc 1). Re-verified after making the module standalone: `2 2` and `4 2` still 0.0. Forward only; `ghex_exchange`'s backward rule raises `NotImplementedError`.
