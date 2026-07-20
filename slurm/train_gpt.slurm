#!/bin/bash
#SBATCH --job-name=oth_train
#SBATCH --account=stf
#SBATCH --partition=gpu-l40s
#SBATCH --gpus-per-node=l40s:1
#SBATCH --output=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/train_%j.out
#SBATCH --error=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com

# Train Othello-GPT. Pilot fits on 1 GPU. For the full (paper) run, request
# more GPUs and bump time, e.g.:
#   sbatch --export=ALL,PROFILE=full --gpus-per-node=l40s:4 --time=4-00:00:00 \
#          --cpus-per-task=16 slurm/train_gpt.slurm
# (the upstream Trainer uses nn.DataParallel, so it uses every visible GPU;
#  the stf account's priority share on gpu-l40s tops out around 10 GPUs total.)
set -e
PROFILE="${PROFILE:-pilot}"
ROOT=/mmfs1/home/jdfelo/BilinearOthelloGPT

source /gscratch/stf/jdfelo/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/stf/jdfelo/envs/othello
module load cuda

mkdir -p $ROOT/logs $ROOT/ckpts $ROOT/results
echo "==== train_gpt ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  GPUs: $CUDA_VISIBLE_DEVICES  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

cd $ROOT/src
srun python train_gpt.py --config $ROOT/configs/config.yaml --profile $PROFILE

echo "Finished: $(date)"
