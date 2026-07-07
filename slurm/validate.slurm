#!/bin/bash
#SBATCH --job-name=oth_valid
#SBATCH --output=/scratch/network/ak9088/Gemma/othello-gemma/logs/validate_%j.out
#SBATCH --error=/scratch/network/ak9088/Gemma/othello-gemma/logs/validate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com,ak9088@princeton.edu

# Anchor check on the authors' released synthetic weights (already downloaded
# to ckpts/pretrained/synthetic_model.pth on the login node).
set -e
PROFILE="${PROFILE:-pilot}"
ROOT=/scratch/network/ak9088/Gemma/othello-gemma

source /scratch/network/ak9088/anaconda3/etc/profile.d/conda.sh
conda activate $ROOT/env

mkdir -p $ROOT/logs $ROOT/results
echo "==== validate_pretrained ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  Start: $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader || true

cd $ROOT/src
srun python validate_pretrained.py --config $ROOT/configs/config.yaml --profile $PROFILE

echo "Finished: $(date)"
