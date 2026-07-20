#!/bin/bash
#SBATCH --job-name=oth_probe
#SBATCH --account=stf
#SBATCH --partition=gpu-l40s
#SBATCH --gpus-per-node=l40s:1
#SBATCH --output=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/probe_%j.out
#SBATCH --error=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/probe_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com

# Linear + nonlinear board-state probes across all layers.
#   sbatch slurm/train_probe.slurm                 # probe trained model
#   sbatch --export=ALL,PROBE_ARGS=--random slurm/train_probe.slurm   # control
set -e
PROFILE="${PROFILE:-pilot}"
ROOT=/mmfs1/home/jdfelo/BilinearOthelloGPT

source /gscratch/stf/jdfelo/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/stf/jdfelo/envs/othello
module load cuda

mkdir -p $ROOT/logs $ROOT/results
echo "==== train_probe ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  Args: $PROBE_ARGS  Start: $(date)"

cd $ROOT/src
srun python train_probe.py --config $ROOT/configs/config.yaml --profile $PROFILE $PROBE_ARGS

echo "Finished: $(date)"
