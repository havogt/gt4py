#!/bin/bash -l
# One process, 4 GPUs. KIND=strong: global M=N per size; KIND=weak: per-device block edge.
#   sbatch --export=ALL,KIND=strong sweep_single.sbatch
#SBATCH --job-name=swm-sweep-single
#SBATCH --account=csstaff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
#SBATCH --uenv=icon/26.7:v1
#SBATCH --view=default
set -u
source "$SLURM_SUBMIT_DIR/common.sh"
KIND=${KIND:-strong}
TRANSPORTS=${TRANSPORTS:-padded,coloured8,coloured2ph,allgather,ragged}
STEPS=${STEPS:-20}
REPEATS=${REPEATS:-20}
if [ "$KIND" = weak ]; then
    SIZES=${SIZES:-32,64,128,256,512,1024,2048,4096}; WEAK=--weak
else
    SIZES=${SIZES:-32,64,128,256,512,1024,2048,4096,8192,16384}; WEAK=
fi
OUT=$RESULTS/${KIND}_single_$SLURM_JOB_ID.jsonl
srun python $SWM/scaling_bench.py $WEAK --sizes "$SIZES" --layouts 1x1,2x1,2x2,4x1 \
    --transports "$TRANSPORTS" --modes fwd,grad,ref --steps "$STEPS" --repeats "$REPEATS" --out "$OUT"
python $SWM/scaling_bench.py --table "$OUT"
