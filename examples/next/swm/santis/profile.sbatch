#!/bin/bash -l
# Per-kernel profile of one model step at 16384^2 on one GPU: nsys kernel summary, then
# ncu speed-of-light (DRAM/SM throughput) for every kernel of a few calls.
#SBATCH --job-name=swm-profile
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
P=$RESULTS/profile_$SLURM_JOB_ID; mkdir -p "$P"
for mode in ref fwd ref_grad_remat grad_remat; do
    echo "### nsys $mode 16384"
    srun nsys profile -t cuda --cuda-graph-trace=node -o "$P/nsys_$mode" --force-overwrite true \
        python $SWM/santis/profile_step.py $mode 16384 5 2>&1 | grep -v "NoTagError" | tail -3
    nsys stats --report cuda_gpu_kern_sum --format csv --output "$P/nsys_$mode" "$P/nsys_$mode.nsys-rep" > /dev/null 2>&1
    nsys stats --report cuda_gpu_kern_sum "$P/nsys_$mode.nsys-rep" 2>/dev/null | head -25
done
for mode in ref fwd; do
    echo "### ncu $mode 16384 (all kernels of 1 warm-up + 2 calls)"
    srun ncu --section SpeedOfLight --launch-count 400 --csv --log-file "$P/ncu_$mode.csv" \
        python $SWM/santis/profile_step.py $mode 16384 2 2>&1 | grep -v NoTagError | tail -2
    wc -l "$P/ncu_$mode.csv"
done
echo "### ncu ref 2048 (launch-floor regime)"
srun ncu --section SpeedOfLight --launch-count 400 --csv --log-file "$P/ncu_ref_2048.csv" \
    python $SWM/santis/profile_step.py ref 2048 2 2>&1 | grep -v NoTagError | tail -2
