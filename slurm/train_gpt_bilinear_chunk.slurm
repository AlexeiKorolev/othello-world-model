#!/bin/bash
#SBATCH --job-name=oth_bchunk
#SBATCH --account=stf
#SBATCH --partition=gpu-l40s
#SBATCH --gpus-per-node=l40s:2
#SBATCH --output=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/train_bilinear_chunk_%j.out
#SBATCH --error=/mmfs1/home/jdfelo/BilinearOthelloGPT/logs/train_bilinear_chunk_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=alexeikorolev7@gmail.com

# One 12h chunk of the full bilinear training, with checkpoint/resume. Trains
# until the wall budget (11.3h) can't fit another epoch, saves state, exits 64,
# and RESUBMITS itself. Repeats until all 20 epochs are done (then evals, exit 0).
# Kick off the chain with a single:   sbatch slurm/train_gpt_bilinear_chunk.slurm
# State lives in ckpts/gpt_synthetic_full_bilinear.ckpt(.state); safe to re-run.
PROFILE="${PROFILE:-full}"
ROOT=/mmfs1/home/jdfelo/BilinearOthelloGPT
SELF=$ROOT/slurm/train_gpt_bilinear_chunk.slurm

source /gscratch/stf/jdfelo/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/stf/jdfelo/envs/othello
module load cuda

mkdir -p $ROOT/logs $ROOT/ckpts $ROOT/results
echo "==== train_gpt bilinear CHUNK ===="; echo "Job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Profile: $PROFILE  GPUs: $CUDA_VISIBLE_DEVICES  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

cd $ROOT/src
python train_gpt.py --config $ROOT/configs/config.yaml --profile $PROFILE \
     --arch bilinear --expansion 2.6666667 --resume --wall_hours 11.3
code=$?

if [ $code -eq 64 ]; then
    echo "[chunk] budget hit, more epochs remain -> resubmitting ($(date))"
    sbatch $SELF
    exit 0
elif [ $code -ne 0 ]; then
    echo "[chunk] python failed with code $code ($(date))"
    exit $code
fi
echo "[chunk] TRAINING COMPLETE + evaluated ($(date))"
