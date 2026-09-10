#!/bin/bash -l
# One process per GPU (jax.distributed, NCCL). P must equal the task count:
#   sbatch --export=ALL,KIND=strong,LAYOUTS=2x2:4x1 --ntasks-per-node=4 sweep_multi.sbatch
#   sbatch --export=ALL,KIND=strong,LAYOUTS=2x1     --ntasks-per-node=2 sweep_multi.sbatch
#SBATCH --job-name=swm-sweep-multi
#SBATCH --account=csstaff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --uenv=icon/26.7:v1
#SBATCH --view=default
set -u
source "$SLURM_SUBMIT_DIR/common.sh"
KIND=${KIND:-strong}
LAYOUTS=${LAYOUTS:-2x2:4x1}; LAYOUTS=${LAYOUTS//:/,}  # colon-separated: sbatch --export splits on commas
TRANSPORTS=${TRANSPORTS:-padded,coloured8,coloured2ph,allgather,ragged}
STEPS=${STEPS:-20}
REPEATS=${REPEATS:-20}
if [ "$KIND" = weak ]; then
    SIZES=${SIZES:-32,64,128,256,512,1024,2048,4096}; WEAK=--weak
else
    SIZES=${SIZES:-32,64,128,256,512,1024,2048,4096,8192,16384}; WEAK=
fi
OUT=$RESULTS/${KIND}_multi${SLURM_NTASKS}_$SLURM_JOB_ID.jsonl
srun python $SWM/scaling_bench.py --distributed $WEAK --sizes "$SIZES" --layouts "$LAYOUTS" \
    --transports "$TRANSPORTS" --modes fwd,grad --steps "$STEPS" --repeats "$REPEATS" --out "$OUT"
python $SWM/scaling_bench.py --table "$OUT"
