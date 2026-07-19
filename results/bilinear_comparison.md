# Bilinear vs GELU Othello-GPT — task & probe comparison

Same architecture (8L / 8H / 512d, block 59, ~25.3M params) trained from scratch
on the same synthetic Othello set. Only the MLP differs:

- **GELU** — standard minGPT MLP: `Linear → GELU → Linear`, expansion 4.
- **Bilinear** — gated linear unit with the activation removed (Dooms et al.
  2024, arXiv:2410.08417): `MLP(x) = P·((Wx) ⊙ (Vx))`, no biases, **expansion 8/3**
  chosen so the model is **param- and FLOP-matched** to the GELU baseline (both
  ~25.3M). The whole layer is a bilinear form → amenable to weight-based
  (eigendecomposition) interpretability.

Trained 2026-07-07, single 2×A100-80GB job, 20 epochs.

## Task performance (legal-move error, held-out synthetic val, 58 951 moves)

| Model | Legal-move error |
|---|---|
| Authors' released ckpt (anchor) | 0.017% |
| GELU (ours) | 0.032% |
| **Bilinear (ours)** | **0.036%** |

Bilinear solves the task as well as GELU — the interpretable MLP costs nothing
behaviorally.

## Board-state probes (best layer, error = fraction of tiles mis-decoded)

Probes trained on held-out synthetic val activations, all 8 layers; numbers below
are the min over layers. Standard probe = absolute black/white/empty
(`probe_full*.json`). Nanda = a **single linear probe** trained across both
black- and white-to-move plies, comparing the absolute frame vs the
player-relative **mine/yours** frame (`nanda_probe_full*.json`).

| Probe | GELU | Bilinear |
|---|---|---|
| Linear, absolute b/w/empty | 17.60% (L6) | **16.78%** (L6) |
| Nonlinear (MLP), absolute | **1.41%** (L7) | 2.60% (L7) |
| Linear, **mine/yours** (Nanda) | **0.99%** (L7) | 2.03% (L7) |
| linearity gap (absolute ÷ mine-yours, linear) | **17.7×** | **8.3×** |

## Findings

1. **Nanda's linearity reproduces in the bilinear model.** A single linear probe
   in the player-relative mine/yours frame decodes the board ~8× better than the
   same linear probe in the absolute frame (2.03% vs 16.8%). The bilinear MLP
   builds the same *linearly-decodable, player-relative* world model — the
   linearity is not a GELU artifact.

2. **The bilinear representation is ~2× harder for these probes to read**, despite
   identical task performance: nonlinear 2.60% vs 1.41%, mine/yours 2.03% vs
   0.99%. The one exception is the linear *absolute* probe, marginally better on
   bilinear (16.8% vs 17.6%). So bilinear reaches the same behavior via a
   representation whose board state is present and linear-in-the-right-frame, but
   a touch less cleanly separable overall.

3. **Depth profile matches:** both models peak at layers 6–7 on every probe; the
   bilinear nonlinear probe plateaus (~2.6%) across L6–L8 where GELU dips to 1.4%
   at L7.

### Caveat
Probe hyperparameters (mid_dim, epochs, lr) were tuned on the GELU model. The
**mine/yours-vs-absolute gap is robust** (it's a within-model contrast, same
probe capacity both frames). The absolute GELU-vs-bilinear *magnitude* comparison
could shift with bilinear-tuned probes — the qualitative ordering (bilinear a bit
harder to read) is the safe claim.

## Files
- Model: `ckpts/gpt_synthetic_full_bilinear.ckpt`; eval `results/gpt_full_bilinear_eval.json`
- Probes: `results/probe_full_bilinear.json`, `results/nanda_probe_full_bilinear.{json,png}`
- GELU baselines: `results/probe_full.json`, `results/nanda_probe_full.json`
- Code: `src/bilinear_model.py`, `src/train_gpt.py --arch bilinear`,
  `src/train_probe.py`/`src/nanda_probe.py` (`--arch bilinear`)
