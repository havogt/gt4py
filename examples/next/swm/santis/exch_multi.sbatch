#!/bin/bash -l
# Exchange-only timings, one process per GPU.
#   sbatch --export=ALL,LAYOUTS=2x2:4x1 --ntasks-per-node=4 exch_multi.sbatch
#   sbatch --export=ALL,LAYOUTS=2x1     --ntasks-per-node=2 exch_multi.sbatch
#SBATCH --job-name=swm-exch-multi
#SBATCH --account=csstaff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --uenv=icon/26.7:v1
#SBATCH --view=default
set -u
source "$SLURM_SUBMIT_DIR/common.sh"
LAYOUTS=${LAYOUTS:-2x2:4x1}; LAYOUTS=${LAYOUTS//:/,}  # colon-separated: sbatch --export splits on commas
OUT=$RESULTS/exch_multi${SLURM_NTASKS}_$SLURM_JOB_ID.jsonl
srun python $SWM/scaling_bench.py --distributed --sizes 32,64,128,256,512,1024,2048,4096,8192,16384 \
    --layouts "$LAYOUTS" --transports padded,coloured8,coloured2ph,allgather,ragged \
    --modes exch,exch_grad --repeats 20 --out "$OUT"
python $SWM/scaling_bench.py --table "$OUT"
