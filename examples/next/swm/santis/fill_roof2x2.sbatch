#!/bin/bash -l
# Fill-in for a timed-out single-process job: appends to the original results file.
# Sizes are capped below the per-device size at which the single-process 4-device gradient
# hangs (2x2 grad at 8192^2, grad_remat at 16384^2); each step is killed after 45 min.
#SBATCH --job-name=swm-fill-roof2x2
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
timeout -s KILL 45m srun python $SWM/scaling_bench.py --sizes 32,64,128,256,512,1024,2048,4096,8192 --layouts 2x2 --transports padded --modes grad_remat --steps 20 --repeats 20 --out "$RESULTS/roof_single_855980.jsonl" || echo "STEP TIMED OUT OR FAILED: --sizes 32,64,128,256,512,1024,2048,4096,8192 --layouts 2x2 --transports padded --modes grad_remat"
timeout -s KILL 45m srun python $SWM/scaling_bench.py --sizes 32,64,128,256,512,1024,2048,4096 --layouts 2x2 --transports allgather --modes grad --steps 20 --repeats 20 --out "$RESULTS/roof_single_855980.jsonl" || echo "STEP TIMED OUT OR FAILED: --sizes 32,64,128,256,512,1024,2048,4096 --layouts 2x2 --transports allgather --modes grad"
timeout -s KILL 45m srun python $SWM/scaling_bench.py --sizes 32,64,128,256,512,1024,2048,4096,8192 --layouts 2x2 --transports allgather --modes grad_remat --steps 20 --repeats 20 --out "$RESULTS/roof_single_855980.jsonl" || echo "STEP TIMED OUT OR FAILED: --sizes 32,64,128,256,512,1024,2048,4096,8192 --layouts 2x2 --transports allgather --modes grad_remat"
