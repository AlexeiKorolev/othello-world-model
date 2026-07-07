#!/bin/bash
#SBATCH --job-name=oth_diag
#SBATCH --output=/scratch/network/ak9088/Gemma/othello-gemma/logs/diag_%j.out
#SBATCH --error=/scratch/network/ak9088/Gemma/othello-gemma/logs/diag_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com,ak9088@princeton.edu
set -e
ROOT=/scratch/network/ak9088/Gemma/othello-gemma
source /scratch/network/ak9088/anaconda3/etc/profile.d/conda.sh
conda activate $ROOT/env
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME  Start: $(date)"
cd $ROOT/src
srun python probe_diag.py --config $ROOT/configs/config.yaml --layer 6
echo "Finished: $(date)"
