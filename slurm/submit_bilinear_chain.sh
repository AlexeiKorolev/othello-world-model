#!/bin/bash
# Pre-submit a dependency chain of bilinear training chunks (no self-resubmission).
# Each chunk resumes from the last saved epoch; afterany means an unexpected kill
# still triggers the next chunk. Extra chunks after completion exit fast.
#   bash slurm/submit_bilinear_chain.sh [N]     # N chunks (default 6)
set -e
ROOT=/mmfs1/home/jdfelo/BilinearOthelloGPT
N="${1:-6}"
SCRIPT=$ROOT/slurm/train_gpt_bilinear_chain.slurm

jid=$(sbatch --parsable $SCRIPT)
echo "chunk 1: $jid"
for i in $(seq 2 $N); do
    jid=$(sbatch --parsable --dependency=afterany:$jid $SCRIPT)
    echo "chunk $i: $jid (afterany previous)"
done
echo "submitted $N-chunk chain"
