#!/bin/bash -l
# Multi-node: one process per GPU, 4 per node, NCCL over Slingshot through the uenv's
# aws-ofi-nccl plugin. Strong, weak and exchange-only sweeps in one job:
#   sbatch --nodes=2 --export=ALL,LAYOUTS=8x1,4x2   sweep_nodes.sbatch
#   sbatch --nodes=4 --export=ALL,LAYOUTS=16x1,4x4  sweep_nodes.sbatch
#   sbatch --nodes=8 --export=ALL,LAYOUTS=32x1,8x4  sweep_nodes.sbatch
#SBATCH --job-name=swm-sweep-nodes
#SBATCH --account=csstaff
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
#SBATCH --uenv=icon/26.7:v1
#SBATCH --view=default
set -u
source "$SLURM_SUBMIT_DIR/common.sh"
LAYOUTS=${LAYOUTS:?}
TRANSPORTS=${TRANSPORTS:-padded,coloured8,coloured2ph,allgather,ragged}
STEPS=${STEPS:-20}
REPEATS=${REPEATS:-20}
STRONG=32,64,128,256,512,1024,2048,4096,8192,16384
WEAK=32,64,128,256,512,1024,2048,4096
# NCCL loads libnccl-net-ofi.so from the uenv view; NCCL_NET makes a socket fallback an error.
export NCCL_NET_PLUGIN=ofi
export NCCL_NET="AWS Libfabric"
export NCCL_CROSS_NIC=1
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
P=$SLURM_NTASKS
srun python $SWM/scaling_bench.py --distributed --sizes "$STRONG" --layouts "$LAYOUTS" \
    --transports "$TRANSPORTS" --modes fwd,grad --steps "$STEPS" --repeats "$REPEATS" \
    --out "$RESULTS/strong_multi${P}_$SLURM_JOB_ID.jsonl"
srun python $SWM/scaling_bench.py --distributed --weak --sizes "$WEAK" --layouts "$LAYOUTS" \
    --transports "$TRANSPORTS" --modes fwd,grad --steps "$STEPS" --repeats "$REPEATS" \
    --out "$RESULTS/weak_multi${P}_$SLURM_JOB_ID.jsonl"
srun python $SWM/scaling_bench.py --distributed --sizes "$STRONG" --layouts "$LAYOUTS" \
    --transports "$TRANSPORTS" --modes exch,exch_grad --repeats "$REPEATS" \
    --out "$RESULTS/exch_multi${P}_$SLURM_JOB_ID.jsonl"
for k in strong weak exch; do python $SWM/scaling_bench.py --table "$RESULTS/${k}_multi${P}_$SLURM_JOB_ID.jsonl"; done
