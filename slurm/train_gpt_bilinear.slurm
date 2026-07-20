#!/bin/bash
#SBATCH --job-name=oth_bilin
#SBATCH --account=stf
#SBATCH --partition=gpu-l40s
# L40S has 48GB/GPU, comfortably above the ~31GB/GPU that batch 4096 needs
# (the Adroit original pinned 2xA100-80GB because its only other option was
# 32GB V100s, which OOM'd; klone's L40S allocation clears the bar directly).
#SBATCH --gpus-per-node=l40s:2
#SBATCH --output=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/train_bilinear_%j.out
#SBATCH --error=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/train_bilinear_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=47:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com

# Full-scale training of Othello-GPT with BILINEAR MLPs (arXiv:2410.08417),
# param- AND FLOP-matched to the GELU baseline (expansion 8/3 -> d_hidden 1365,
# ~25.3M params). Single job: 2xL40S-48GB, 20 epochs, ~41h. The stf account's
# gpu-l40s QOS has no MaxWall cap, so a single 47h job fits without chunking.
# --resume (no --wall_hours) is pure crash-insurance: it checkpoints full
# state each epoch so a late failure can be resumed by re-running this same sbatch;
# with no --wall_hours it simply trains straight through all 20 epochs, then evals.
# Writes ckpts/gpt_synthetic_full_bilinear.ckpt.
#   sbatch slurm/train_gpt_bilinear.slurm
set -e
PROFILE="${PROFILE:-full}"
ROOT=/mmfs1/home/jdfelo/BilinearOthelloGPT

source /gscratch/stf/jdfelo/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/stf/jdfelo/envs/othello
module load cuda

mkdir -p $ROOT/logs $ROOT/ckpts $ROOT/results
echo "==== train_gpt (bilinear) ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  GPUs: $CUDA_VISIBLE_DEVICES  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

cd $ROOT/src
srun python train_gpt.py --config $ROOT/configs/config.yaml --profile $PROFILE \
     --arch bilinear --expansion 2.6666667 --resume

echo "Finished: $(date)"
