#!/bin/bash -l
# Rematerialised gradient, one process per GPU, any node count:
#   sbatch --nodes=1 --ntasks-per-node=2 --export=ALL,LAYOUTS=2x1      remat_multi.sbatch
#   sbatch --nodes=1 --ntasks-per-node=4 --export=ALL,LAYOUTS=2x2,4x1  remat_multi.sbatch
#   sbatch --nodes=2 --export=ALL,LAYOUTS=8x1,4x2   remat_multi.sbatch   (and 4/8 nodes)
#SBATCH --job-name=swm-remat-multi
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
LAYOUTS=${LAYOUTS:?}
TRANSPORTS=${TRANSPORTS:-padded,coloured8,coloured2ph,allgather,ragged}
export NCCL_NET_PLUGIN=ofi
export NCCL_NET="AWS Libfabric"
export NCCL_CROSS_NIC=1
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
P=$SLURM_NTASKS
srun python $SWM/scaling_bench.py --distributed --sizes 32,64,128,256,512,1024,2048,4096,8192,16384 \
    --layouts "$LAYOUTS" --transports "$TRANSPORTS" --modes grad_remat \
    --steps 20 --repeats 20 --out "$RESULTS/rstrong_multi${P}_$SLURM_JOB_ID.jsonl"
srun python $SWM/scaling_bench.py --distributed --weak --sizes 32,64,128,256,512,1024,2048,4096 \
    --layouts "$LAYOUTS" --transports "$TRANSPORTS" --modes grad_remat \
    --steps 20 --repeats 20 --out "$RESULTS/rweak_multi${P}_$SLURM_JOB_ID.jsonl"
for k in rstrong rweak; do python $SWM/scaling_bench.py --table "$RESULTS/${k}_multi${P}_$SLURM_JOB_ID.jsonl"; done
