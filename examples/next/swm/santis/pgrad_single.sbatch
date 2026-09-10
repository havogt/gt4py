#!/bin/bash -l
# Plain (stored-residual) gradient one size past the default-allocator OOM boundary, with XLA
# preallocation on: the default allocator fragments and OOMs one size early (see 5d).
#SBATCH --job-name=swm-pgrad-single
#SBATCH --account=csstaff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --uenv=icon/26.7:v1
#SBATCH --view=default
set -u
source "$SLURM_SUBMIT_DIR/common.sh"
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
T=padded,coloured8,coloured2ph,allgather,ragged
timeout -s KILL 40m srun python $SWM/scaling_bench.py --sizes 2048,4096 --layouts 1x1 --transports $T --modes grad,ref_grad \
    --steps 20 --repeats 20 --out "$RESULTS/pgrad_single_$SLURM_JOB_ID.jsonl" || echo "STEP TIMED OUT OR FAILED"
timeout -s KILL 15m srun python $SWM/scaling_bench.py --sizes 4096,8192 --layouts 2x1 --transports $T --modes grad \
    --steps 20 --repeats 20 --out "$RESULTS/pgrad_single_$SLURM_JOB_ID.jsonl" || echo "STEP TIMED OUT OR FAILED"
python $SWM/scaling_bench.py --table "$RESULTS/pgrad_single_$SLURM_JOB_ID.jsonl"
