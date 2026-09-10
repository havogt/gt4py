#!/bin/bash -l
# Rematerialised gradient, one process, 4 GPUs: weak part only (split off for the time limit).
#SBATCH --job-name=swm-remat-single-weak
#SBATCH --account=csstaff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --uenv=icon/26.7:v1
#SBATCH --view=default
set -u
source "$SLURM_SUBMIT_DIR/common.sh"
TRANSPORTS=${TRANSPORTS:-padded,coloured8,coloured2ph,allgather,ragged}
srun python $SWM/scaling_bench.py --weak --sizes 32,64,128,256,512,1024,2048,4096 \
    --layouts 1x1,2x1,2x2,4x1 --transports "$TRANSPORTS" --modes grad_remat,ref_grad_remat \
    --steps 20 --repeats 20 --out "$RESULTS/rweak_single_$SLURM_JOB_ID.jsonl"
for k in rweak; do python $SWM/scaling_bench.py --table "$RESULTS/${k}_single_$SLURM_JOB_ID.jsonl"; done
