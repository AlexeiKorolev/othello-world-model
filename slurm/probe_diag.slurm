#!/bin/bash
#SBATCH --job-name=oth_diag
#SBATCH --account=stf
#SBATCH --partition=gpu-l40s
#SBATCH --gpus-per-node=l40s:1
#SBATCH --output=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/diag_%j.out
#SBATCH --error=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/diag_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com
set -e
ROOT=/mmfs1/home/jdfelo/BilinearOthelloGPT
source /gscratch/stf/jdfelo/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/stf/jdfelo/envs/othello
module load cuda
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME  Start: $(date)"
cd $ROOT/src
srun python probe_diag.py --config $ROOT/configs/config.yaml --layer 6
echo "Finished: $(date)"
