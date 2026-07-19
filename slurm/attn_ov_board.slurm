#!/bin/bash
#SBATCH --job-name=oth_ov
#SBATCH --output=/scratch/network/ak9088/Gemma/othello-gemma/logs/attn_ov_board_%j.out
#SBATCH --error=/scratch/network/ak9088/Gemma/othello-gemma/logs/attn_ov_board_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com,ak9088@princeton.edu

# Do attention-head WEIGHTS map the board? OV-circuit board-alignment + tile x tile
# transfer matrices. Runs for the arch in OV_ARGS (bilinear or gelu).
#   sbatch --export=ALL,OV_ARGS='--arch bilinear' slurm/attn_ov_board.slurm
#   sbatch --export=ALL,OV_ARGS='--arch gelu'     slurm/attn_ov_board.slurm
set -e
PROFILE="${PROFILE:-full}"
ROOT=/scratch/network/ak9088/Gemma/othello-gemma

source /scratch/network/ak9088/anaconda3/etc/profile.d/conda.sh
conda activate $ROOT/env

mkdir -p $ROOT/logs $ROOT/results
echo "==== attn_ov_board ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  Args: $OV_ARGS  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

cd $ROOT/src
srun python attn_ov_board.py --config $ROOT/configs/config.yaml --profile $PROFILE $OV_ARGS

echo "Finished: $(date)"
