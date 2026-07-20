#!/bin/bash
#SBATCH --job-name=oth_bench_h200
#SBATCH --partition=ckpt-all
#SBATCH --gpus-per-node=h200:1
#SBATCH --requeue
#SBATCH --output=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/bench_h200_%j.out
#SBATCH --error=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/bench_h200_%j.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com

# Batch-size / throughput sweep on an H200 via the free, preemptible ckpt-all
# queue (no --account needed; QOS PreemptMode=REQUEUE + --requeue above means
# Slurm auto-resubmits this exact job if preempted). bench_speed.py persists
# a state file in results/ after every batch size measured, so a requeued run
# resumes the sweep instead of starting over.
#   sbatch slurm/bench_speed_h200.slurm            # profile=pilot (default)
#   TAG=h200_bf16_compile sbatch slurm/bench_speed_h200.slurm   # separate state file, no clash with an older sweep
set -e
PROFILE="${PROFILE:-pilot}"
TAG="${TAG:-h200}"
ROOT=/mmfs1/home/jdfelo/BilinearOthelloGPT

source /gscratch/stf/jdfelo/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/stf/jdfelo/envs/othello
module load cuda

mkdir -p $ROOT/logs $ROOT/results
echo "==== bench_speed (H200) ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME  Restart count: $SLURM_RESTART_COUNT"
echo "Profile: $PROFILE  Tag: $TAG  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

cd $ROOT/src
srun python bench_speed.py --config $ROOT/configs/config.yaml --profile $PROFILE --tag $TAG

echo "Finished: $(date)"
