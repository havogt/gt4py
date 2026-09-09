#!/bin/bash -l
# Exchange-only timings (one exchange call and its VJP), one process, 4 GPUs.
#SBATCH --job-name=swm-exch-single
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
OUT=$RESULTS/exch_single_$SLURM_JOB_ID.jsonl
srun python $SWM/scaling_bench.py --sizes 32,64,128,256,512,1024,2048,4096,8192,16384 \
    --layouts 1x1,2x1,2x2,4x1 --transports padded,coloured8,coloured2ph,allgather,ragged \
    --modes exch,exch_grad --repeats 20 --out "$OUT"
python $SWM/scaling_bench.py --table "$OUT"
