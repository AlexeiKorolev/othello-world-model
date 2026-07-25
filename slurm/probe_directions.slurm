#!/bin/bash
#SBATCH --job-name=oth_dirs
#SBATCH --partition=ckpt-all
#SBATCH --gpus-per-node=a40:1
#SBATCH --requeue
#SBATCH --output=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/dirs_%j.out
#SBATCH --error=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/dirs_%j.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com

# Export per-epoch board directions + shared 3D PCA embedding for animation.
#   sbatch --export=ALL,PROFILE=full slurm/probe_directions.slurm
set -e
PROFILE="${PROFILE:-full}"
LAYERS="${LAYERS:-4,7}"
MAXGAMES="${MAXGAMES:-2000}"
PROBE_EPOCHS="${PROBE_EPOCHS:-150}"
TAG="${TAG:-${PROFILE}_h200}"
ROOT=/mmfs1/home/jdfelo/BilinearOthelloGPT

source /gscratch/stf/jdfelo/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/stf/jdfelo/envs/othello
module load cuda

mkdir -p $ROOT/logs $ROOT/results
echo "==== probe_directions ===="
echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  layers: $LAYERS  tag: $TAG  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

cd $ROOT/src
srun python probe_directions.py --config $ROOT/configs/config.yaml \
     --profile $PROFILE --layers $LAYERS --max-games $MAXGAMES \
     --probe-epochs $PROBE_EPOCHS --tag $TAG

echo "Finished: $(date)"
