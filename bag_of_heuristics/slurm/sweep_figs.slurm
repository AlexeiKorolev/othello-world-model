#!/bin/bash
#SBATCH --job-name=oth_sfigs
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --output=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/sfigs_%j.out
#SBATCH --error=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/sfigs_%j.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --mail-type=FAIL

# Data-scaling sweep figures. Pure matplotlib over existing JSON.
set -e
ROOT=/mmfs1/home/jdfelo/BilinearOthelloGPT

source /gscratch/stf/jdfelo/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/stf/jdfelo/envs/othello

mkdir -p $ROOT/bag_of_heuristics/results/figs
echo "==== sweep_figs ==== Job $SLURM_JOB_ID  Start: $(date)"
cd $ROOT/bag_of_heuristics/src
srun python sweep_figs.py --results $ROOT/results \
    --ood $ROOT/bag_of_heuristics/results \
    --out $ROOT/bag_of_heuristics/results/figs
echo "Finished: $(date)"
