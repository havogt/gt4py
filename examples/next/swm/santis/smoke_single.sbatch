#!/bin/bash -l
# One process driving the node's 4 GPUs: correctness battery per transport on 2x2, then a
# small strong-scaling sweep including the 1x1 (single-GPU) layout.
#SBATCH --job-name=swm-smoke-single
#SBATCH --account=csstaff
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --uenv=icon/26.7:v1
#SBATCH --view=default
set -u
source "$SLURM_SUBMIT_DIR/common.sh"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
srun python -c "import jax; print('jax', jax.__version__, jax.devices())"

for t in padded coloured8 coloured2ph allgather ragged ragged_emul; do
    srun python $SWM/swm_battery.py --transport $t --layout 2x2 --steps 10 --repeats 5 || true
done

OUT=$RESULTS/smoke_single_$SLURM_JOB_ID.jsonl
srun python $SWM/scaling_bench.py --sizes 64,256,1024 --layouts 1x1,2x1,2x2,4x1 \
    --transports padded,allgather,coloured8,ragged --modes fwd,grad,ref \
    --steps 10 --repeats 3 --out "$OUT"
python $SWM/scaling_bench.py --table "$OUT"
