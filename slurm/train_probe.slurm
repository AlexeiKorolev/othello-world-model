#!/bin/bash
#SBATCH --job-name=oth_probe
#SBATCH --output=/scratch/network/ak9088/Gemma/othello-gemma/logs/probe_%j.out
#SBATCH --error=/scratch/network/ak9088/Gemma/othello-gemma/logs/probe_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com,ak9088@princeton.edu

# Linear + nonlinear board-state probes across all layers.
#   sbatch slurm/train_probe.slurm                 # probe trained model
#   sbatch --export=ALL,PROBE_ARGS=--random slurm/train_probe.slurm   # control
set -e
PROFILE="${PROFILE:-pilot}"
ROOT=/scratch/network/ak9088/Gemma/othello-gemma

source /scratch/network/ak9088/anaconda3/etc/profile.d/conda.sh
conda activate $ROOT/env

mkdir -p $ROOT/logs $ROOT/results
echo "==== train_probe ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  Args: $PROBE_ARGS  Start: $(date)"

cd $ROOT/src
srun python train_probe.py --config $ROOT/configs/config.yaml --profile $PROFILE $PROBE_ARGS

echo "Finished: $(date)"
