#!/bin/bash -l
# Sanity checks, one process, 4 GPUs: correctness battery at 2048^2, step-count linearity,
# allocator sensitivity of the grad OOM boundary, and a roofline rerun of the 1x1/2x2 points.
#SBATCH --job-name=swm-sanity-single
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
J=$SLURM_JOB_ID

echo "### battery at 2048^2, 2x2"
for t in padded coloured2ph allgather ragged; do
    srun python $SWM/swm_battery.py --size 2048 --transport $t --layout 2x2 --steps 10 --repeats 5 || true
done

echo "### step-count linearity (64^2 and 4096^2, 1x1): per-call overhead vs per-step cost"
srun python $SWM/scaling_bench.py --sizes 64,4096 --layouts 1x1 --transports padded --modes fwd,grad_remat,ref,ref_grad_remat \
    --steps 10 --repeats 20 --out "$RESULTS/lin10_single_$J.jsonl"
srun python $SWM/scaling_bench.py --sizes 64,4096 --layouts 1x1 --transports padded --modes fwd,grad_remat,ref,ref_grad_remat \
    --steps 40 --repeats 20 --out "$RESULTS/lin40_single_$J.jsonl"

echo "### grad OOM boundary with default preallocation"
XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
srun python $SWM/scaling_bench.py --sizes 2048,4096,8192 --layouts 1x1 --transports padded --modes grad,ref_grad \
    --steps 20 --repeats 5 --out "$RESULTS/prealloc_single_$J.jsonl"

echo "### roofline rerun (xla cost analysis), 1x1 and 2x2"
srun python $SWM/scaling_bench.py --sizes 32,64,128,256,512,1024,2048,4096,8192,16384 --layouts 1x1,2x2 \
    --transports padded,allgather --modes fwd,grad,ref,grad_remat,ref_grad_remat \
    --steps 20 --repeats 20 --out "$RESULTS/roof_single_$J.jsonl"
for k in lin10 lin40 prealloc roof; do python $SWM/scaling_bench.py --table "$RESULTS/${k}_single_$J.jsonl"; done
