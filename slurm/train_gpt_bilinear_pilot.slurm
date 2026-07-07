#!/bin/bash
#SBATCH --job-name=oth_bpilot
#SBATCH --output=/scratch/network/ak9088/Gemma/othello-gemma/logs/train_bilinear_pilot_%j.out
#SBATCH --error=/scratch/network/ak9088/Gemma/othello-gemma/logs/train_bilinear_pilot_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com,ak9088@princeton.edu

# Small bilinear-MLP training (PILOT profile: ~120k games, 40 epochs, 1 GPU) to
# confirm the bilinear GPT learns Othello, while the full run is queued. Compare
# results/gpt_pilot_bilinear_eval.json against the GELU pilot (gpt_pilot_eval.json,
# legal-move error ~2.84%). Writes ckpts/gpt_synthetic_pilot_bilinear.ckpt.
set -e
PROFILE="${PROFILE:-pilot}"
ROOT=/scratch/network/ak9088/Gemma/othello-gemma

source /scratch/network/ak9088/anaconda3/etc/profile.d/conda.sh
conda activate $ROOT/env

mkdir -p $ROOT/logs $ROOT/ckpts $ROOT/results
echo "==== train_gpt bilinear PILOT ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  GPUs: $CUDA_VISIBLE_DEVICES  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

cd $ROOT/src
srun python train_gpt.py --config $ROOT/configs/config.yaml --profile $PROFILE \
     --arch bilinear --expansion 2.6666667

echo "Finished: $(date)"
