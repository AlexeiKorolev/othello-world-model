#!/bin/bash
#SBATCH --job-name=oth_psweep
#SBATCH --partition=ckpt-all
#SBATCH --gpus-per-node=a40:1
#SBATCH --requeue
#SBATCH --output=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/psweep_%j.out
#SBATCH --error=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/psweep_%j.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --mail-type=FAIL,TIME_LIMIT

# Analysis for one data-scaling sweep point: the Nanda linear mine/yours probe
# (vs the absolute-colour baseline, all 8 layers) plus the OOD generalisation
# suite. Both read the same fixed val split, so every point is comparable.
#
#   NGAMES=20000 sbatch bag_of_heuristics/slurm/probe_sweep.slurm
set -e
ROOT=/mmfs1/home/jdfelo/BilinearOthelloGPT
NGAMES="${NGAMES:?set NGAMES}"
PROFILE="${PROFILE:-full}"
TAG="${TAG:-sweep_n${NGAMES}}"
CKPT="${CKPT:-$ROOT/ckpts/gpt_synthetic_${TAG}.ckpt}"

source /gscratch/stf/jdfelo/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/stf/jdfelo/envs/othello
module load cuda

mkdir -p $ROOT/logs $ROOT/results $ROOT/bag_of_heuristics/results
echo "==== probe_sweep ===="
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME  Start: $(date)"
echo "tag=$TAG ckpt=$CKPT"

echo "---- nanda_probe ----"
cd $ROOT/src
srun python nanda_probe.py --config $ROOT/configs/config.yaml \
    --profile $PROFILE --ckpt $CKPT --tag $TAG

echo "---- ood_eval ----"
cd $ROOT/bag_of_heuristics/src
srun python ood_eval.py --config $ROOT/configs/config.yaml \
    --profile $PROFILE --ckpt $CKPT --tag $TAG \
    --n-games 300 --n-policy-games 300

echo "Finished: $(date)"
