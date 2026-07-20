#!/bin/bash
#SBATCH --job-name=oth_nanda
#SBATCH --account=stf
#SBATCH --partition=gpu-l40s
#SBATCH --gpus-per-node=l40s:1
#SBATCH --output=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/nanda_%j.out
#SBATCH --error=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/nanda_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com

# Nanda's linear mine/yours board probe vs the absolute-encoding linear baseline.
#   sbatch slurm/nanda_probe.slurm                       # profile=full (default)
#   PROFILE=pilot sbatch slurm/nanda_probe.slurm
set -e
PROFILE="${PROFILE:-full}"
ROOT=/mmfs1/home/jdfelo/BilinearOthelloGPT

source /gscratch/stf/jdfelo/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/stf/jdfelo/envs/othello
module load cuda

mkdir -p $ROOT/logs $ROOT/results
echo "==== nanda_probe ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  Args: $NANDA_ARGS  Start: $(date)"

cd $ROOT/src
srun python nanda_probe.py --config $ROOT/configs/config.yaml --profile $PROFILE $NANDA_ARGS

echo "Finished: $(date)"
