#!/bin/bash -l
# Transport benchmark on CSCS Alps / Santis (GH200, 4 GPUs per node, aarch64).
#
# Never yet run on a cluster; the Slurm directives are unverified. --distributed itself is
# tested -- two real processes over gloo on CPU, layouts 2x1/4x1/2x2/1x4, all transports ALL
# PASS on both processes -- but nothing here has run on more than one node, on NCCL, or on a
# GPU.
#
# Fill in: --account, the uenv image, GT4PY, and the venv path.

#SBATCH --job-name=swm-transports
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --output=swm-transports-%j.out
#SBATCH --uenv=<UENV_IMAGE>            # e.g. prgenv-gnu/24.11:v1
#SBATCH --view=default

set -eu

GT4PY=<PATH_TO>/gt4py
VENV=<PATH_TO>/venv-santis           # must be created with the uenv's python; see requirements-santis.txt
cd "$GT4PY"
source "$VENV/bin/activate"

# JAX auto-pins each process to its local GPU via Slurm env vars, so do NOT set
# CUDA_VISIBLE_DEVICES or JAX_PLATFORMS here.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

OUT="$SCRATCH/santis_$SLURM_JOB_ID.jsonl"

LAYOUTS=4x1,2x2,1x4  # for 2 nodes (8 GPUs): resubmit with --nodes=2 and LAYOUTS=8x1,4x2,2x4,1x8

# The four transports that run everywhere. `set -e` is live, so this step must succeed.
srun python examples/next/swm/bench_transports.py \
    --distributed --transports allgather,padded,coloured8,coloured2ph,ragged_emul \
    --layouts "$LAYOUTS" --steps 10 --repeats 50 --out "$OUT"

# `ragged` gets its own step: bench_transports.py exits 1 whenever a transport fails, and
# on any backend that cannot lower ragged-all-to-all it always does. `|| true` keeps the
# job alive so the rows still land in $OUT.
srun python examples/next/swm/bench_transports.py \
    --distributed --transports ragged \
    --layouts "$LAYOUTS" --steps 10 --out "$OUT" || true
