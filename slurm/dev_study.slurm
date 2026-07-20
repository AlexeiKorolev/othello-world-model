#!/bin/bash
#SBATCH --job-name=oth_devstudy
#SBATCH --account=stf
#SBATCH --partition=gpu-l40s
#SBATCH --gpus-per-node=l40s:1
#SBATCH --output=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/dev_study_%j.out
#SBATCH --error=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/dev_study_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com

# Train the normal (GELU) Othello-GPT while interleaving a lightweight Nanda
# mine/yours probe + causal intervention check every other epoch, to see when
# the linear world model arises over training. Produces the normal
# gpt_synthetic_<profile>.ckpt (this run replaces a plain train_gpt.slurm run)
# plus results/dev_study_<profile>.{json,png}, and finishes by running the
# full-scale nanda_probe.py / intervene_nanda.py on the finished checkpoint.
#   sbatch slurm/dev_study.slurm                # profile=pilot (default)
set -e
PROFILE="${PROFILE:-pilot}"
ROOT=/mmfs1/home/jdfelo/BilinearOthelloGPT

source /gscratch/stf/jdfelo/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/stf/jdfelo/envs/othello
module load cuda

mkdir -p $ROOT/logs $ROOT/ckpts $ROOT/results
echo "==== dev_study ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  GPUs: $CUDA_VISIBLE_DEVICES  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

cd $ROOT/src
srun python dev_study.py --config $ROOT/configs/config.yaml --profile $PROFILE

echo "Finished: $(date)"
