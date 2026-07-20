#!/bin/bash
#SBATCH --job-name=oth_devstudy_full
#SBATCH --partition=ckpt-all
#SBATCH --gpus-per-node=h200:1
#SBATCH --requeue
#SBATCH --output=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/dev_study_full_%j.out
#SBATCH --error=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/dev_study_full_%j.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com

# Full-scale (paper-scale, ~20M game) GELU Othello-GPT training on an H200,
# with a FULL-FIDELITY Nanda mine/yours probe + causal intervention check
# every 2 epochs (--full_fidelity_probe: same probe_epochs/max_probe_games/
# n_cases/steps as nanda_probe.py/intervene_nanda.py, all 5 layers in the
# intervene range -- not the cheap pilot-study settings). Needs
# data/full_shard_*.pkl to already exist (submit with
# --dependency=afterok:<gen_data_full jobid> if it doesn't yet).
# Preemptible (ckpt-all, no --account): QOS PreemptMode=REQUEUE + --requeue
# above auto-resubmits on preemption; DevStudyTrainer/ResumableTrainer
# checkpoint every epoch (tag-specific .state file) so a resumed run picks up
# training from the last completed epoch. (One caveat: a probe/intervene
# round itself isn't checkpointed mid-round -- if preempted mid-round, that
# epoch's data point is skipped rather than retried, training just continues.)
set -e
TAG="${TAG:-full_h200}"
ROOT=/mmfs1/home/jdfelo/BilinearOthelloGPT

source /gscratch/stf/jdfelo/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/stf/jdfelo/envs/othello
module load cuda

mkdir -p $ROOT/logs $ROOT/ckpts $ROOT/results
echo "==== dev_study (full, H200) ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME  Restart count: $SLURM_RESTART_COUNT"
echo "Tag: $TAG  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

cd $ROOT/src
srun python dev_study.py --config $ROOT/configs/config.yaml --profile full --tag $TAG --full_fidelity_probe

echo "Finished: $(date)"
