#!/bin/bash
#SBATCH --job-name=oth_xfer
#SBATCH --partition=ckpt-all
#SBATCH --gpus-per-node=a40:1
#SBATCH --requeue
#SBATCH --output=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/xfer_%j.out
#SBATCH --error=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/xfer_%j.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com

# Cross-epoch probe transfer: does the linear world model rotate during training?
#   sbatch --export=ALL,PROFILE=full slurm/probe_transfer.slurm
#   sbatch --time=00:20:00 --export=ALL,MAXGAMES=40,PROBE_EPOCHS=4,LAYERS=7,TAG=smoketest \
#          slurm/probe_transfer.slurm     # cheap end-to-end validation
set -e
PROFILE="${PROFILE:-full}"
LAYERS="${LAYERS:-4,7}"
MAXGAMES="${MAXGAMES:-2000}"
PROBE_EPOCHS="${PROBE_EPOCHS:-150}"
CONTROL_EPOCHS="${CONTROL_EPOCHS:-4,20}"
CONTROL_SEEDS="${CONTROL_SEEDS:-1,2}"
TAG="${TAG:-${PROFILE}_h200}"
ROOT=/mmfs1/home/jdfelo/BilinearOthelloGPT

source /gscratch/stf/jdfelo/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/stf/jdfelo/envs/othello
module load cuda

mkdir -p $ROOT/logs $ROOT/results
echo "==== probe_transfer ===="
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  layers: $LAYERS  games: $MAXGAMES  tag: $TAG  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd $ROOT/src
srun python probe_transfer.py --config $ROOT/configs/config.yaml \
     --profile $PROFILE --layers $LAYERS --max-games $MAXGAMES \
     --probe-epochs $PROBE_EPOCHS --control-epochs $CONTROL_EPOCHS \
     --control-seeds $CONTROL_SEEDS --tag $TAG

echo "Finished: $(date)"
