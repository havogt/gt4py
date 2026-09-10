#!/bin/bash -l
# All sweeps for one multi-process layout list (colon-separated), any node count; appends new
# result files (the analysis merges all files per run):
#   sbatch --nodes=1 --export=ALL,LAYOUTS=4x1 layouts2d.sbatch
#   sbatch --nodes=2 --export=ALL,LAYOUTS=4x2 layouts2d.sbatch   (4 nodes: 4x4, 8 nodes: 8x4)
#SBATCH --job-name=swm-layouts2d
#SBATCH --account=csstaff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out
#SBATCH --uenv=icon/26.7:v1
#SBATCH --view=default
set -u
source "$SLURM_SUBMIT_DIR/common.sh"
LAYOUTS=${LAYOUTS:?}; LAYOUTS=${LAYOUTS//:/,}
T=padded,coloured8,coloured2ph,allgather,ragged
STRONG=32,64,128,256,512,1024,2048,4096,8192,16384
WEAK=32,64,128,256,512,1024,2048,4096
export NCCL_NET_PLUGIN=ofi NCCL_NET="AWS Libfabric" NCCL_CROSS_NIC=1 FI_CXI_DISABLE_HOST_REGISTER=1 FI_MR_CACHE_MONITOR=userfaultfd
P=$SLURM_NTASKS; J=$SLURM_JOB_ID
run() { timeout -s KILL 25m srun python $SWM/scaling_bench.py --distributed "$@" --repeats 20 || echo "STEP TIMED OUT OR FAILED: $*"; }
run --sizes $STRONG --layouts "$LAYOUTS" --transports $T --modes fwd,grad --steps 20 --out "$RESULTS/strong_multi${P}_$J.jsonl"
run --weak --sizes $WEAK --layouts "$LAYOUTS" --transports $T --modes fwd,grad --steps 20 --out "$RESULTS/weak_multi${P}_$J.jsonl"
run --sizes $STRONG --layouts "$LAYOUTS" --transports $T --modes grad_remat --steps 20 --out "$RESULTS/rstrong_multi${P}_$J.jsonl"
run --weak --sizes $WEAK --layouts "$LAYOUTS" --transports $T --modes grad_remat --steps 20 --out "$RESULTS/rweak_multi${P}_$J.jsonl"
run --sizes $STRONG --layouts "$LAYOUTS" --transports $T --modes exch,exch_grad --out "$RESULTS/exch_multi${P}_$J.jsonl"
