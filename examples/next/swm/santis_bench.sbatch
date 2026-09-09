#!/bin/bash -l
# Transport benchmark on CSCS Alps / Santis (GH200, 4 GPUs per node, aarch64).
#
# TEMPLATE, NEVER RUN ON A CLUSTER. --distributed itself is tested -- two real processes
# over gloo on CPU, layouts 2x1/4x1/2x2/1x4, all transports ALL PASS on both processes --
# but nothing here has run on more than one node, on NCCL, or on a GPU. Treat the Slurm
# directives, the uenv line and the 2-node block as unverified.
#
# Fill in: --account, the uenv image, and the venv path.
#
#SBATCH --job-name=swm-transports
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:20:00
#SBATCH --output=swm-transports-%j.out
#SBATCH --uenv=<UENV_IMAGE>            # e.g. prgenv-gnu/24.11:v1
#SBATCH --view=default

set -eu

GT4PY=<PATH_TO>/gt4py
VENV=<PATH_TO>/venv-santis           # see requirements-santis.txt
cd "$GT4PY"
source "$VENV/bin/activate"

# One process per GPU. JAX auto-detects Slurm in jax.distributed.initialize() and pins each
# process to its local GPU itself (clusters/slurm_cluster.py -> local_device_ids =
# [SLURM_LOCALID] -> jax_cuda_visible_devices), so do NOT set CUDA_VISIBLE_DEVICES here.
# Do not set JAX_PLATFORMS either: the harness takes the device count from jax.devices().
# Needed on the laptop's 8 GiB GPU or NCCL failed to allocate its buffers; on a 96 GiB
# GH200 it only matters when several jobs share a device.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

OUT="santis_$SLURM_JOB_ID.jsonl"

# --- 1 node, 4 GPUs: Rx*Ry must equal the total device count (4) --------------------------
# The four transports that run everywhere. `set -e` is live, so this step must succeed.
srun python examples/next/swm/bench_transports.py \
    --distributed --transports allgather,padded,coloured8,coloured2ph,ragged_emul \
    --layouts 4x1,2x2,1x4 --steps 10 --out "$OUT"

# `ragged` gets its own step: bench_transports.py exits 1 whenever a transport fails, and
# on any backend that cannot lower ragged-all-to-all it always does. `|| true` keeps the
# job alive so the rows still land in $OUT.
srun python examples/next/swm/bench_transports.py \
    --distributed --transports ragged \
    --layouts 4x1,2x2,1x4 --steps 10 --out "$OUT" || true

# --- 2 nodes, 8 GPUs ----------------------------------------------------------------------
# Submit as a separate job (Slurm directives are per job), or resubmit this file with
#     sbatch --nodes=2 santis_bench.sbatch
# and swap the layouts below in. With --nodes=2 --ntasks-per-node=4 --gpus-per-node=4 the
# total device count is 8, so the legal layouts are 8x1, 4x2, 2x4, 1x8, and 4x2/2x4 put four
# ranks per node -- the interesting case, because half of every exchange then crosses the
# node boundary (Slingshot) instead of staying on NVLink. Note that 8x1 and 1x8 are also
# the worst layouts for `padded` (pad factor 3.60x, above allgather at h=2), so they are
# worth running for that reason too.
#
# srun python examples/next/swm/bench_transports.py \
#     --distributed --transports allgather,padded,coloured8,coloured2ph,ragged_emul \
#     --layouts 8x1,4x2,2x4,1x8 --steps 10 --out "$OUT"
# srun python examples/next/swm/bench_transports.py \
#     --distributed --transports ragged --layouts 8x1,4x2,2x4,1x8 --out "$OUT" || true

# --- what to look at ----------------------------------------------------------------------
# `ragged` is the transport that matters here. On XLA:CPU it does not compile at all
# (UNIMPLEMENTED: HLO opcode `ragged-all-to-all` is not supported by XLA:CPU ThunkEmitter --
# XlaRuntimeError on jax 0.6.2, JaxRuntimeError on 0.11.1), and on the laptop GPU it only
# ran at P=1, where the exchange is the identity and an axis_size-fold transpose over-count
# is 1x and therefore invisible. At P >= 4 on real NCCL both questions are answerable:
#   * does T1 still hold (forward correct against the allgather oracle)?
#   * does T2 hold -- <Lx,y> == <x,L^T y> through jax.vjp? FESOM measured an O(1) violation
#     with f^T(ones) = axis_size x correct on A100. This round found the *Python* transpose
#     rule algebraically correct at P=1,2,4,8 against a CPU emulation of the primitive, so
#     if the over-count reproduces here it is below the rule, in XLA:GPU / NCCL. If T2
#     passes at P=8 it does not reproduce on this JAX+XLA, and FESOM's forward-only
#     restriction can be lifted for the version pinned in requirements-santis.txt.
# Second question: the latency-vs-bandwidth ranking (coloured2ph K=4 vs coloured8 K=8 vs
# padded's single all_to_all vs allgather's O(P) volume), which CPU timings cannot resolve.
#
# Merge with the laptop rows and regenerate the table (rows carry their environment):
#   cat tmp/laptop_results.jsonl santis_*.jsonl > all.jsonl
#   python examples/next/swm/bench_transports.py --table all.jsonl
