#!/bin/bash
#SBATCH --job-name=oth_gen
#SBATCH --output=/scratch/network/ak9088/Gemma/othello-gemma/logs/gen_%j.out
#SBATCH --error=/scratch/network/ak9088/Gemma/othello-gemma/logs/gen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --partition=all
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com,ak9088@princeton.edu

# Synthetic Othello data generation (CPU, multiprocessing).
# Override profile with:  sbatch --export=ALL,PROFILE=full slurm/gen_data.slurm
set -e
PROFILE="${PROFILE:-pilot}"
ROOT=/scratch/network/ak9088/Gemma/othello-gemma

source /scratch/network/ak9088/anaconda3/etc/profile.d/conda.sh
conda activate $ROOT/env

mkdir -p $ROOT/logs
echo "==== gen_data ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  CPUs: $SLURM_CPUS_PER_TASK  Start: $(date)"

cd $ROOT/src
srun python gen_data.py --config $ROOT/configs/config.yaml --profile $PROFILE

echo "Finished: $(date)"
