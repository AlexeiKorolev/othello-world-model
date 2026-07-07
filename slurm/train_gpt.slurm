#!/bin/bash
#SBATCH --job-name=oth_train
#SBATCH --output=/scratch/network/ak9088/Gemma/othello-gemma/logs/train_%j.out
#SBATCH --error=/scratch/network/ak9088/Gemma/othello-gemma/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com,ak9088@princeton.edu

# Train Othello-GPT. Pilot fits on 1 GPU. For the full (paper) run, request
# more GPUs and bump time, e.g.:
#   sbatch --export=ALL,PROFILE=full --gres=gpu:4 --time=4-00:00:00 \
#          --cpus-per-task=16 slurm/train_gpt.slurm
# (the upstream Trainer uses nn.DataParallel, so it uses every visible GPU.)
set -e
PROFILE="${PROFILE:-pilot}"
ROOT=/scratch/network/ak9088/Gemma/othello-gemma

source /scratch/network/ak9088/anaconda3/etc/profile.d/conda.sh
conda activate $ROOT/env

mkdir -p $ROOT/logs $ROOT/ckpts $ROOT/results
echo "==== train_gpt ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  GPUs: $CUDA_VISIBLE_DEVICES  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

cd $ROOT/src
srun python train_gpt.py --config $ROOT/configs/config.yaml --profile $PROFILE

echo "Finished: $(date)"
