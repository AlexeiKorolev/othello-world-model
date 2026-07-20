#!/bin/bash
#SBATCH --job-name=oth_gen_full
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --output=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/gen_full_%j.out
#SBATCH --error=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/gen_full_%j.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com

# Full-scale (23M game) synthetic Othello data generation, CPU-only. Runs on
# the free/preemptible ckpt-all queue (no --account needed) targeting a large
# idle node (128 cpus) rather than stf's fully-occupied priority `compute`
# partition. gen_data.py is resumable at shard granularity, and QOS
# PreemptMode=REQUEUE + --requeue above means a preemption auto-resubmits
# this job, which then skips shards already written.
set -e
PROFILE=full
ROOT=/mmfs1/home/jdfelo/BilinearOthelloGPT

source /gscratch/stf/jdfelo/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/stf/jdfelo/envs/othello

mkdir -p $ROOT/logs $ROOT/data
echo "==== gen_data (full) ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME  Restart count: $SLURM_RESTART_COUNT"
echo "Profile: $PROFILE  CPUs: $SLURM_CPUS_PER_TASK  Start: $(date)"

cd $ROOT/src
srun python gen_data.py --config $ROOT/configs/config.yaml --profile $PROFILE

echo "Finished: $(date)"
