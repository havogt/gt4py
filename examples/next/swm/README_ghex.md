# SWM distributed with GHEX (`swm_ghex.py`, `swm_ghex_2d.py`, `swm_ghex_ad.py`)

Shallow water model with the halo exchange through [GHEX](https://github.com/ghex-org/GHEX)
0.9.0 and the local state as JAX-backed `gt4py.next` fields. Three programs:

- `swm_ghex.py` — forward model on a 1-D ring decomposition along `I`; periodicity in
  `J` stays local (`periodic_j`).
- `swm_ghex_2d.py` — forward model on an `RX x RY` decomposition; both periodic
  directions, corners included, come from one GHEX exchange.
- `swm_ghex_ad.py` — reverse-mode AD of the 1-D model: a hand-written backward rule for
  the exchange, checked by a distributed dot-product test, against `jax.grad` of the
  single-process reference, and by a distributed Taylor test.

`swm_ghex.make_exchange` builds the GHEX context and pattern for one rank and returns the
exchange as a `jax.custom_vjp` around `jax.pure_callback`; the backward rule is optional
(`bwd=`) and raises `NotImplementedError` when absent. The 2-D and AD scripts import it
together with the model constants, the time loop and the single-process reference. The
layouts are described in the module docstrings.

## Requirements

`mpi4py` and `ghex` in the venv (both built against the system MPICH):

```
CC=mpicc CXX=mpicxx MPICC=mpicc MPICXX=mpicxx CMAKE_BUILD_PARALLEL_LEVEL=4 heavy uv pip install ghex
```

## Run

From `examples/next/swm`:

```
../../../.venv/bin/python swm_ghex.py                                                      # single rank, MPI singleton init
HWLOC_COMPONENTS=-gl timeout -s KILL 180 mpirun -n 4 ../../../.venv/bin/python swm_ghex.py         # R ranks, R must divide M
HWLOC_COMPONENTS=-gl timeout -s KILL 180 mpirun -n 4 ../../../.venv/bin/python swm_ghex_2d.py 2 2  # RX RY, RX*RY == ranks
HWLOC_COMPONENTS=-gl timeout -s KILL 180 mpirun -n 4 ../../../.venv/bin/python swm_ghex_ad.py      # R must divide M, MLOC >= 2
killall -q -9 mpiexec.hydra hydra_pmi_proxy                                                # only after a killed run
```

`HWLOC_COMPONENTS=-gl` is required on this machine for *every* `mpirun` launch,
including `-n 1`: hydra's hwloc topology probe otherwise blocks on a stale X display
socket and `MPI_Init` never returns (`AD_SERIES_STATUS.md`, "MPI: fixed"). Wrap `mpirun`
in `timeout -s KILL`; hung MPICH ranks ignore SIGTERM.

The forward scripts print, on rank 0, the max abs difference per prognostic field between
the gathered distributed result and the single-process reference, then `PASS`/`FAIL`
(threshold 1e-9). Both run the time loop as a plain Python loop, so the three
`pure_callback` exchanges per step execute in program order and the comparison is
bit-exact. A layout that does not match the rank count or does not divide `M x N`
exits with a message on every rank.

Parameters: `M = N = 16`, `dx = dy = 1e5`, `dt = 90`, `a = 1e6`, `alpha = 1e-3`;
10 steps in the forward scripts, 5 in the AD script.

## Corners in 2-D

`timestep` reads the diagonal neighbour (`avg_x(avg_y(p))` at `(MLOC-1, NLOC-1)` reads
`p(MLOC, NLOC)`), so the four corner halo cells must carry the diagonal block's values.
GHEX's structured `HaloGenerator` with halos in both dimensions fills them in the same
exchange, which is why there is no local periodic step in 2-D.

## Adjoint of the exchange (`swm_ghex_ad.py`)

GHEX has no reverse pattern and no accumulate-on-receive, so the backward rule reuses the
*forward* pattern on a scratch buffer: the halo cotangents are placed on the two boundary
interior rows, exchanged, and the halo rows that come back are added to the owning
interior rows; the incoming halo cotangent itself is dropped because the forward exchange
overwrote the halo. This is exact only on a ring with `MLOC >= 2` (nb04 §3 shows why it
fails in 2-D), which is why the AD script is 1-D.

The gradient is `jax.grad` of the *local* cost on every rank: the adjoint of the global
allreduce only seeds every rank with 1, and the cross-rank terms arrive through the
exchange adjoint, which all ranks run in lockstep. The gradient check threshold is 1e-10
relative rather than roundoff: `p_bar ~ 1e5` enters the stencil transpose as differences
~10, so the reordered accumulation at rank boundaries costs ~1e-12.

## Expected to pass

`swm_ghex.py` on 1, 2, 4, 8, 16 ranks; `swm_ghex_2d.py` on `1 1`, `2 1`, `1 2`, `2 2`,
`4 2`, `2 4`; `swm_ghex_ad.py` on 2 and 4 ranks.
