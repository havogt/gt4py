#!/bin/bash -l
# Plain gradient at 16384^2 past the default-allocator boundary, preallocation on, P per GPU:
#   sbatch --nodes=1 --export=ALL,LAYOUTS=2x2:4x1 pgrad_multi.sbatch
#   sbatch --nodes=2 --export=ALL,LAYOUTS=8x1:4x2 pgrad_multi.sbatch
#SBATCH --job-name=swm-pgrad-multi
#SBATCH --account=csstaff
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out
#SBATCH --uenv=icon/26.7:v1
#SBATCH --view=default
set -u
source "$SLURM_SUBMIT_DIR/common.sh"
LAYOUTS=${LAYOUTS:?}; LAYOUTS=${LAYOUTS//:/,}  # colon-separated: sbatch --export splits on commas
export XLA_PYTHON_CLIENT_PREALLOCATE=true XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export NCCL_NET_PLUGIN=ofi NCCL_NET="AWS Libfabric" NCCL_CROSS_NIC=1 FI_CXI_DISABLE_HOST_REGISTER=1 FI_MR_CACHE_MONITOR=userfaultfd
P=$SLURM_NTASKS
timeout -s KILL 30m srun python $SWM/scaling_bench.py --distributed --sizes 8192,16384 --layouts "$LAYOUTS" \
    --transports padded,coloured8,coloured2ph,allgather,ragged --modes grad \
    --steps 20 --repeats 20 --out "$RESULTS/pgrad_multi${P}_$SLURM_JOB_ID.jsonl" || echo "STEP TIMED OUT OR FAILED"
python $SWM/scaling_bench.py --table "$RESULTS/pgrad_multi${P}_$SLURM_JOB_ID.jsonl"
