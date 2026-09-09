#!/bin/bash -l
# Four processes, one GPU each (jax.distributed over NCCL): battery via bench_transports
# --distributed, then the same small sweep for the P=4 layouts.
#SBATCH --job-name=swm-smoke-multi
#SBATCH --account=csstaff
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --uenv=icon/26.7:v1
#SBATCH --view=default
set -u
source "$SLURM_SUBMIT_DIR/common.sh"

OUT=$RESULTS/smoke_multi_$SLURM_JOB_ID.jsonl
srun python $SWM/bench_transports.py --distributed \
    --transports padded,coloured8,coloured2ph,allgather,ragged,ragged_emul \
    --layouts 2x2,4x1 --steps 10 --repeats 5 --out "$OUT" || true
python $SWM/bench_transports.py --table "$OUT"

OUT=$RESULTS/smoke_multi_scaling_$SLURM_JOB_ID.jsonl
srun python $SWM/scaling_bench.py --distributed --sizes 64,256,1024 --layouts 2x2,4x1 \
    --transports padded,allgather,coloured8,ragged --modes fwd,grad \
    --steps 10 --repeats 3 --out "$OUT"
python $SWM/scaling_bench.py --table "$OUT"
