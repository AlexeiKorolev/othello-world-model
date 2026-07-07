#!/bin/bash
#SBATCH --job-name=oth_bchain
#SBATCH --output=/scratch/network/ak9088/Gemma/othello-gemma/logs/train_bilinear_chain_%j.out
#SBATCH --error=/scratch/network/ak9088/Gemma/othello-gemma/logs/train_bilinear_chain_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com,ak9088@princeton.edu

# One 12h chunk of the full bilinear training, checkpoint/resume, NO self-submit.
# Pre-submit a chain of these with afterany dependencies (see submit_bilinear_chain.sh):
# each chunk resumes from ckpts/gpt_synthetic_full_bilinear.ckpt(.state), trains until
# the 11.3h wall budget can't fit another epoch, saves, and exits 0 (paused) so the
# next queued chunk runs. When all 20 epochs are done it evals and exits 0; any extra
# queued chunks then detect completion, re-eval (idempotent), and exit fast.
PROFILE="${PROFILE:-full}"
ROOT=/scratch/network/ak9088/Gemma/othello-gemma

source /scratch/network/ak9088/anaconda3/etc/profile.d/conda.sh
conda activate $ROOT/env

mkdir -p $ROOT/logs $ROOT/ckpts $ROOT/results
echo "==== train_gpt bilinear CHAIN chunk ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  GPUs: $CUDA_VISIBLE_DEVICES  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

cd $ROOT/src
python train_gpt.py --config $ROOT/configs/config.yaml --profile $PROFILE \
     --arch bilinear --expansion 2.6666667 --resume --wall_hours 11.3
code=$?

# exit 64 = clean pause (more epochs remain); treat as success so the next
# dependency chunk runs and this chunk isn't flagged FAILED.
if [ $code -eq 64 ]; then
    echo "[chain] budget hit, paused for next chunk to resume ($(date))"
    exit 0
elif [ $code -ne 0 ]; then
    echo "[chain] python failed with code $code ($(date))"
    exit $code
fi
echo "[chain] TRAINING COMPLETE + evaluated ($(date))"
