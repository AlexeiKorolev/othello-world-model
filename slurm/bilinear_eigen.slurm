#!/bin/bash
#SBATCH --job-name=oth_eigen
#SBATCH --output=/scratch/network/ak9088/Gemma/othello-gemma/logs/bilinear_eigen_%j.out
#SBATCH --error=/scratch/network/ak9088/Gemma/othello-gemma/logs/bilinear_eigen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com,ak9088@princeton.edu

# Weight-space board features in the bilinear Othello-GPT (arXiv:2410.08417):
# fold each layer's bilinear MLP against the mine/yours probe readout directions,
# eigendecompose the interaction matrices, and measure low-rank structure +
# alignment with the true board. Reads ckpts/gpt_synthetic_full_bilinear.ckpt.
#   sbatch slurm/bilinear_eigen.slurm
set -e
PROFILE="${PROFILE:-full}"
ROOT=/scratch/network/ak9088/Gemma/othello-gemma

source /scratch/network/ak9088/anaconda3/etc/profile.d/conda.sh
conda activate $ROOT/env

mkdir -p $ROOT/logs $ROOT/results
echo "==== bilinear_eigen ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  Args: $EIGEN_ARGS  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

cd $ROOT/src
srun python bilinear_eigen.py --config $ROOT/configs/config.yaml --profile $PROFILE $EIGEN_ARGS

echo "Finished: $(date)"
