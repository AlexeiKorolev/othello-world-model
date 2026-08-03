#!/bin/bash
#SBATCH --job-name=oth_sweep
#SBATCH --partition=ckpt-all
#SBATCH --gpus-per-node=a40:1
#SBATCH --requeue
#SBATCH --output=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/sweep_%j.out
#SBATCH --error=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/sweep_%j.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT

# One point of the data-scaling sweep. Preemptible + resumable: the trainer
# checkpoints every epoch and exits 64 when the wall budget is nearly spent, and
# this script resubmits itself to continue.
#
#   NGAMES=20000 sbatch bag_of_heuristics/slurm/train_sweep.slurm
#
# Every point trains for the same number of PRESENTATIONS (and so the same
# number of optimiser steps at a fixed batch), so the only variable is how much
# unique data there is.
set -e
ROOT=/mmfs1/home/jdfelo/BilinearOthelloGPT
NGAMES="${NGAMES:?set NGAMES}"
PRESENTATIONS="${PRESENTATIONS:-40000000}"
BATCH="${BATCH:-1024}"
PROFILE="${PROFILE:-full}"
TAG="${TAG:-sweep_n${NGAMES}}"
WALL="${WALL:-7.0}"
WD="${WD:-}"
if [ -n "$WD" ]; then WD_ARG="--weight-decay $WD"; else WD_ARG=""; fi

source /gscratch/stf/jdfelo/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/stf/jdfelo/envs/othello
module load cuda

mkdir -p $ROOT/logs $ROOT/results
echo "==== train_sweep ===="
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME  Start: $(date)"
echo "n_games=$NGAMES presentations=$PRESENTATIONS batch=$BATCH tag=$TAG wd=${WD:-cfg}"

cd $ROOT/src
set +e
srun python train_gpt.py --config $ROOT/configs/config.yaml --profile $PROFILE \
    --max-train-games $NGAMES --presentations $PRESENTATIONS \
    --batch-size $BATCH --tag $TAG $WD_ARG --resume --wall_hours $WALL
RC=$?
set -e

if [ $RC -eq 64 ]; then
    echo "Wall budget spent, resubmitting to continue: $(date)"
    NGAMES=$NGAMES PRESENTATIONS=$PRESENTATIONS BATCH=$BATCH PROFILE=$PROFILE \
        TAG=$TAG WALL=$WALL WD=$WD sbatch $ROOT/bag_of_heuristics/slurm/train_sweep.slurm
    exit 0
fi

echo "Finished rc=$RC: $(date)"
exit $RC
