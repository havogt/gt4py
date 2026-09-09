# sourced by the sbatch scripts; ROOT is where setup.sh put the checkout and the venv
ROOT=$SCRATCH/tmp/gt4py_ad_2026_09_09
RESULTS=$ROOT/results
mkdir -p "$RESULTS"
cd "$ROOT/gt4py"
source "$ROOT/venv/bin/activate"
# JAX picks its GPU from SLURM_LOCALID; do not set CUDA_VISIBLE_DEVICES or JAX_PLATFORMS.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
SWM=examples/next/swm
