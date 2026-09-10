#!/bin/bash -l
# One step per call (no scan) vs the 20-step scan used everywhere else: is the scan loop itself
# costing anything per step? 1 GPU, debug partition.
#SBATCH --job-name=swm-scanstep
#SBATCH --account=csstaff
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --uenv=icon/26.7:v1
#SBATCH --view=default
set -u
source "$SLURM_SUBMIT_DIR/common.sh"
OUT=$RESULTS/scanstep_single_$SLURM_JOB_ID.jsonl
for s in 1 2 5 20; do
    srun python $SWM/scaling_bench.py --sizes 2048,4096,8192,16384 --layouts 1x1 --transports padded --modes fwd,ref,grad_remat,ref_grad_remat \
        --steps $s --repeats 20 --out "$OUT"
done
python $SWM/scaling_bench.py --table "$OUT"
