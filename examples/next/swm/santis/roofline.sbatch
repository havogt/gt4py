#!/bin/bash -l
# Per-kernel roofline inputs (ncu): DRAM bytes, FP64 instruction counts, duration; one step per
# call at 16384^2 and 2048^2 on one GPU, for the reference step and the padded sharded step.
#SBATCH --job-name=swm-roofline
#SBATCH --account=csstaff
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --uenv=icon/26.7:v1
#SBATCH --view=default
set -u
source "$SLURM_SUBMIT_DIR/common.sh"
P=$RESULTS/roofline_$SLURM_JOB_ID; mkdir -p "$P"
M=gpu__time_duration.sum,dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum
M=$M,smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum
M=$M,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
for mode in ref fwd ref_grad_remat grad_remat; do
  for size in 16384 2048; do
    echo "### ncu roofline $mode $size"
    srun ncu --metrics $M --launch-count 400 --csv --log-file "$P/ncu_${mode}_${size}.csv" \
        python $SWM/santis/profile_step.py $mode $size 2 2>&1 | grep -v NoTagError | tail -1
  done
done
ls -la "$P"
