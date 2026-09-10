#!/bin/bash -l
# Fill-in for a timed-out single-process job: appends to the original results file.
# Sizes are capped below the per-device size at which the single-process 4-device gradient
# hangs (2x2 grad at 8192^2, grad_remat at 16384^2); each step is killed after 45 min.
#SBATCH --job-name=swm-fill-strong4x1
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
timeout -s KILL 45m srun python $SWM/scaling_bench.py --sizes 32,64,128,256,512,1024,2048,4096,8192,16384 --layouts 4x1 --transports padded,coloured8,coloured2ph,allgather,ragged --modes fwd --steps 20 --repeats 20 --out "$RESULTS/strong_single_855947.jsonl" || echo "STEP TIMED OUT OR FAILED: --sizes 32,64,128,256,512,1024,2048,4096,8192,16384 --layouts 4x1 --transports padded,coloured8,coloured2ph,allgather,ragged --modes fwd"
timeout -s KILL 45m srun python $SWM/scaling_bench.py --sizes 32,64,128,256,512,1024,2048,4096 --layouts 4x1 --transports padded,coloured8,coloured2ph,allgather,ragged --modes grad --steps 20 --repeats 20 --out "$RESULTS/strong_single_855947.jsonl" || echo "STEP TIMED OUT OR FAILED: --sizes 32,64,128,256,512,1024,2048,4096 --layouts 4x1 --transports padded,coloured8,coloured2ph,allgather,ragged --modes grad"
