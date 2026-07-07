#!/bin/bash
#SBATCH --job-name=oth_bilin
#SBATCH --output=/scratch/network/ak9088/Gemma/othello-gemma/logs/train_bilinear_%j.out
#SBATCH --error=/scratch/network/ak9088/Gemma/othello-gemma/logs/train_bilinear_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
# MUST pin A100-80GB: batch 4096 needs ~31GB/GPU and OOMs on the 32GB V100s
# (adroit-h11g3). nvidia_a100:2 forces adroit-h11g1 (4x A100-80GB), like the baseline.
#SBATCH --gres=gpu:nvidia_a100:2
#SBATCH --partition=gpu
#SBATCH --time=47:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com,ak9088@princeton.edu

# Full-scale training of Othello-GPT with BILINEAR MLPs (arXiv:2410.08417),
# param- AND FLOP-matched to the GELU baseline (expansion 8/3 -> d_hidden 1365,
# ~25.3M params). Single job: 2xA100-80GB, 20 epochs, ~41h, fits the 48h GPU QOS
# cap. --resume (no wall budget) is pure crash-insurance: it checkpoints full
# state each epoch so a late failure can be resumed by re-running this same sbatch;
# with no --wall_hours it simply trains straight through all 20 epochs, then evals.
# Writes ckpts/gpt_synthetic_full_bilinear.ckpt.
#   sbatch slurm/train_gpt_bilinear.slurm
set -e
PROFILE="${PROFILE:-full}"
ROOT=/scratch/network/ak9088/Gemma/othello-gemma

source /scratch/network/ak9088/anaconda3/etc/profile.d/conda.sh
conda activate $ROOT/env

mkdir -p $ROOT/logs $ROOT/ckpts $ROOT/results
echo "==== train_gpt (bilinear) ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  GPUs: $CUDA_VISIBLE_DEVICES  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

cd $ROOT/src
srun python train_gpt.py --config $ROOT/configs/config.yaml --profile $PROFILE \
     --arch bilinear --expansion 2.6666667 --resume

echo "Finished: $(date)"
