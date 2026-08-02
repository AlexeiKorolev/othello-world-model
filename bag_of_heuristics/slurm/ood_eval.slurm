#!/bin/bash
#SBATCH --job-name=oth_ood
#SBATCH --partition=ckpt-all
#SBATCH --gpus-per-node=a40:1
#SBATCH --requeue
#SBATCH --output=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/ood_%j.out
#SBATCH --error=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/ood_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com

# Bag-of-heuristics generalisation suite: orientation equivariance + policy shift.
#   sbatch bag_of_heuristics/slurm/ood_eval.slurm                  # gelu, full
#   ARCH=bilinear sbatch bag_of_heuristics/slurm/ood_eval.slurm
#   OOD_ARGS="--ckpt /path/to/retrained.ckpt --tag myrun" sbatch ...
set -e
PROFILE="${PROFILE:-full}"
ARCH="${ARCH:-gelu}"
ROOT=/mmfs1/home/jdfelo/BilinearOthelloGPT

source /gscratch/stf/jdfelo/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/stf/jdfelo/envs/othello
module load cuda

mkdir -p $ROOT/logs $ROOT/bag_of_heuristics/results
echo "==== ood_eval ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  Arch: $ARCH  Args: $OOD_ARGS  Start: $(date)"

cd $ROOT/bag_of_heuristics/src
srun python ood_eval.py --config $ROOT/configs/config.yaml \
    --profile $PROFILE --arch $ARCH $OOD_ARGS

echo "Finished: $(date)"
