#!/bin/bash
#SBATCH --job-name=oth_floor
#SBATCH --partition=ckpt-all
#SBATCH --requeue
#SBATCH --output=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/floor_%j.out
#SBATCH --error=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/floor_%j.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com

# Irreducible floors for the training loss and the intervention legality metric.
# CPU-only: pure board replay, no model is loaded.
#   sbatch --export=ALL,PROFILE=full slurm/entropy_floor.slurm
set -e
PROFILE="${PROFILE:-full}"
# 20k games -> ~1.2M target positions, SEM ~5e-4 nats. Far more precision than
# the comparison needs (the floor only has to be pinned to ~5e-3), and board
# replay costs ~2.3 s/game/core so this lands well inside the walltime.
NGAMES="${NGAMES:-20000}"
ROOT=/mmfs1/home/jdfelo/BilinearOthelloGPT

source /gscratch/stf/jdfelo/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/stf/jdfelo/envs/othello

mkdir -p $ROOT/logs $ROOT/results
echo "==== entropy_floor ===="
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  n_games: $NGAMES  Start: $(date)"

cd $ROOT/src
srun python entropy_floor.py --config $ROOT/configs/config.yaml \
     --profile $PROFILE --n-games $NGAMES --tag ${PROFILE}_h200

echo "Finished: $(date)"
