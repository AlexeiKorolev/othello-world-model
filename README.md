# othello-gemma — Othello-GPT reproduction on Adroit

A faithful re-creation of **Li et al. 2023, "Emergent World Representations:
Exploring a Sequence Model Trained on a Synthetic Task"** (ICLR oral,
[arXiv:2210.13382](https://arxiv.org/abs/2210.13382)), run on the Princeton
Adroit cluster.

We train an 8-layer GPT *from scratch* to predict legal Othello moves, then show
it develops an internal model of the board: decodable by probes and **causally
used** (editing the internal board changes the model's predictions).

> Separate from the sibling `../gemma_world_model` project. This one trains a
> tiny GPT; it does not use Gemma weights.

## What's here

| Path | Role |
|---|---|
| `upstream/` | vendored [likenneth/othello_world](https://github.com/likenneth/othello_world) (data gen, minGPT model, probe heads, intervention) — unmodified |
| `src/common.py` | config, canonical 61-token vocab, shard I/O, fixed-vocab dataset |
| `src/gen_data.py` | synthetic game generation (random legal play) |
| `src/train_gpt.py` | train Othello-GPT + legal-move evaluation |
| `src/validate_pretrained.py` | anchor: convert authors' TransformerLens weights → minGPT, re-run our harness |
| `src/train_probe.py` | linear + nonlinear board-state probes per layer |
| `src/intervene.py` | causal intervention benchmark + saliency maps |
| `src/plot_results.py` | summary figures |
| `slurm/*.slurm` | one job per stage (all email on completion) |
| `configs/config.yaml` | every knob; `profile: pilot|full` |

## Pipeline (run order)

```bash
ROOT=/scratch/network/ak9088/Gemma/othello-gemma
cd $ROOT

# 1. synthetic data (CPU)
sbatch slurm/gen_data.slurm

# 2. train GPT (GPU)            3. anchor on authors' weights (GPU, parallel)
TJ=$(sbatch --parsable slurm/train_gpt.slurm)
sbatch slurm/validate.slurm

# 4. probes (after training)
PJ=$(sbatch --parsable --dependency=afterok:$TJ slurm/train_probe.slurm)

# 5. intervention (after training)
sbatch --dependency=afterok:$TJ slurm/intervene.slurm

# 6. figures (after probes + intervention)
cd src && python plot_results.py --config ../configs/config.yaml
```

**Profiles.** `pilot` (~100k games, 40 epochs, 1 GPU) smoke-tests the whole
pipeline in minutes–hours. For the paper-scale run:

```bash
sbatch --export=ALL,PROFILE=full slurm/gen_data.slurm
sbatch --export=ALL,PROFILE=full --gres=gpu:4 --time=4-00:00:00 --cpus-per-task=16 \
       slurm/train_gpt.slurm     # upstream Trainer uses nn.DataParallel
# then probe / intervene with --export=ALL,PROFILE=full
```

## Expected numbers (synthetic, from the paper)

| Quantity | Paper | Where |
|---|---|---|
| Legal-move error | ~0.01% | `results/gpt_<profile>_eval.json` |
| Linear board probe error | ~20–27% (never <20%) | `results/probe_<profile>.json` |
| Nonlinear (2-layer MLP) probe error | ~1.8% (best ~layer 6) | same |
| Intervention: baseline → intervened error | ~2.68 → ~0.12 | `results/intervention_<profile>.json` |

The **pilot** uses far less data/training, so its legal-move and probe errors
will be higher than the paper; the **full** profile targets the paper numbers.

## Notes / deviations
- Probes are extracted over held-out **synthetic** val games (upstream used
  championship games); the representational question is unchanged.
- The intervention legality-error metric is **reconstructed** (the exact scoring
  was not shipped with the upstream repo); see the docstring in `src/intervene.py`.
- Authors' checkpoint (`ckpts/pretrained/synthetic_model.pth`) is TransformerLens
  format; `validate_pretrained.py` converts it to the minGPT layout. The
  conversion self-checks via legal-move error (~0 if correct).

## Environment
Conda env at `$ROOT/env` (torch 2.4.1, in `/scratch` to spare the `/home`
quota). Cluster conventions live in the `princeton-adroit` skill.
