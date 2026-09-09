#!/bin/bash -l
# Sanity checks on 8 nodes / 32 GPUs: NCCL network confirmation, correctness battery at
# 2048^2 on 32x1, and a repeat of the largest strong-scaling points for cross-job agreement.
#SBATCH --job-name=swm-sanity-nodes
#SBATCH --account=csstaff
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
#SBATCH --uenv=icon/26.7:v1
#SBATCH --view=default
set -u
source "$SLURM_SUBMIT_DIR/common.sh"
J=$SLURM_JOB_ID
export NCCL_NET_PLUGIN=ofi
export NCCL_NET="AWS Libfabric"
export NCCL_CROSS_NIC=1
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd

echo "### NCCL network in use"
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
srun python $SWM/scaling_bench.py --distributed --sizes 512 --layouts 32x1 --transports padded --modes fwd \
    --steps 2 --repeats 2 --out "$RESULTS/ncclcheck_multi32_$J.jsonl" 2>&1 | grep -m 8 -i "Using network\|NET/\|nccl-net\|error"

echo "### battery at 2048^2 on 32x1 (T3 skipped under multi-process)"
srun python $SWM/bench_transports.py --distributed --size 2048 --layouts 32x1 --transports padded,coloured2ph,allgather,ragged \
    --steps 10 --repeats 5 --out "$RESULTS/battery2048_multi32_$J.jsonl" || true
python $SWM/bench_transports.py --table "$RESULTS/battery2048_multi32_$J.jsonl"

echo "### repeat of the largest strong points"
srun python $SWM/scaling_bench.py --distributed --sizes 8192,16384 --layouts 32x1,8x4 --transports padded,coloured2ph \
    --modes fwd,grad_remat --steps 20 --repeats 20 --out "$RESULTS/repeat_multi32_$J.jsonl"
python $SWM/scaling_bench.py --table "$RESULTS/repeat_multi32_$J.jsonl"
