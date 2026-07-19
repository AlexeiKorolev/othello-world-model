#!/bin/bash
#SBATCH --job-name=oth_bmap
#SBATCH --output=/scratch/network/ak9088/Gemma/othello-gemma/logs/board_map_%j.out
#SBATCH --error=/scratch/network/ak9088/Gemma/othello-gemma/logs/board_map_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com,ak9088@princeton.edu

# Map of WHERE the board world-model is stored: exact linear decomposition of the
# mine/yours readout into per-component contributions (embedding, every attention
# head, every MLP). Runs for the arch given in MAP_ARGS (bilinear or gelu).
#   sbatch --export=ALL,MAP_ARGS='--arch bilinear' slurm/board_map.slurm
#   sbatch --export=ALL,MAP_ARGS='--arch gelu'     slurm/board_map.slurm
set -e
PROFILE="${PROFILE:-full}"
ROOT=/scratch/network/ak9088/Gemma/othello-gemma

source /scratch/network/ak9088/anaconda3/etc/profile.d/conda.sh
conda activate $ROOT/env

mkdir -p $ROOT/logs $ROOT/results
echo "==== board_map ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  Args: $MAP_ARGS  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

cd $ROOT/src
srun python board_map.py --config $ROOT/configs/config.yaml --profile $PROFILE $MAP_ARGS

echo "Finished: $(date)"
