#!/bin/bash
#SBATCH --job-name=oth_figs_web
#SBATCH --partition=ckpt-all
#SBATCH --output=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/figsweb_%j.out
#SBATCH --error=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/figsweb_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
set -e
ROOT=/mmfs1/home/jdfelo/BilinearOthelloGPT
source /gscratch/stf/jdfelo/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/stf/jdfelo/envs/othello
mkdir -p $ROOT/results/figs_web
cd $ROOT/src
srun python report_figs.py --results $ROOT/results --out $ROOT/results/figs_web --dpi 92 --themes light
