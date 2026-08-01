#!/bin/bash
#SBATCH --job-name=oth_figs
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --output=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/figs_%j.out
#SBATCH --error=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/figs_%j.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --mail-type=FAIL

# Report figures for REPORT.md. Pure matplotlib over the existing results/*.json
# -- no model, no data shards.
#   sbatch slurm/report_figs.slurm
set -e
ROOT=/mmfs1/home/jdfelo/BilinearOthelloGPT

source /gscratch/stf/jdfelo/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/stf/jdfelo/envs/othello

mkdir -p $ROOT/logs $ROOT/results/figs
echo "==== report_figs ===="
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME  Start: $(date)"

cd $ROOT/src
srun python report_figs.py --results $ROOT/results --out $ROOT/results/figs

echo "Finished: $(date)"
