#!/bin/bash
#SBATCH --job-name=oth_nintv
#SBATCH --output=/scratch/network/ak9088/Gemma/othello-gemma/logs/intervene_nanda_%j.out
#SBATCH --error=/scratch/network/ak9088/Gemma/othello-gemma/logs/intervene_nanda_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com,ak9088@princeton.edu

# Causal intervention using Nanda's LINEAR mine/yours probe (cements the linear
# result by showing the linear direction is causally used). Needs the trained GPT.
#   sbatch slurm/intervene_nanda.slurm            # profile=full (default)
set -e
PROFILE="${PROFILE:-full}"
ROOT=/scratch/network/ak9088/Gemma/othello-gemma

source /scratch/network/ak9088/anaconda3/etc/profile.d/conda.sh
conda activate $ROOT/env

mkdir -p $ROOT/logs $ROOT/results
echo "==== intervene_nanda ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  Start: $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

cd $ROOT/src
srun python intervene_nanda.py --config $ROOT/configs/config.yaml --profile $PROFILE

echo "Finished: $(date)"
