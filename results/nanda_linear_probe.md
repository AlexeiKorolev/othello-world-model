# Nanda's linear "mine/yours" board probe — result

**Date:** 2026-07-01
**Model probed:** our from-scratch full-scale Othello-GPT
(`ckpts/gpt_synthetic_full.ckpt`, 8L/8H/512d, legal-move error 0.032%).
**Script:** `src/nanda_probe.py` · **Job:** SLURM 3293272 (adroit-h11g3, ~40 min).
**Artifacts:** `results/nanda_probe_full.json`, `results/nanda_probe_full.png`.

## Background

Li et al. 2023 (arXiv:2210.13382) reported that a **linear** probe decodes the
board poorly (~20–27% error) in the absolute white/blank/black encoding, and
concluded the world model is **nonlinear** (only a 2-layer MLP probe reached
~1.8%). Neel Nanda's follow-up, *"Actually, Othello-GPT Has A Linear Emergent
World Representation"* (2023), showed the representation is in fact **linear** —
once each tile is labelled **relative to the player about to move** (mine /
yours / empty) instead of by absolute colour.

## The "single linear probe" point (what Nanda was after)

The crux of Nanda's claim is that **one single linear probe**, with **one weight
matrix per (tile, class)**, decodes the board at **every** position — both
black-to-move and white-to-move plies — with no per-turn switching. That is
exactly what we trained:

- We flatten **all** board positions from the validation games (both parities of
  side-to-move) into a **single** training set and fit **one**
  `BatteryProbeClassification` linear head per layer. There is no separate
  black-to-move vs white-to-move probe, and no parity feature fed to the probe.
- The turn information is absorbed entirely into the **labels**: a tile is
  `mine` if it belongs to the side about to move, `yours` if the opponent's,
  `empty` otherwise.

Why a single *absolute* linear probe fails but a single *mine/yours* one
succeeds: in the absolute frame, the direction that means "black piece here"
must serve as "mine" on black's turn and "yours" on white's turn — opposite
roles the model encodes with opposite-sign activity, which no single linear
direction can match. The mine/yours relabel makes the target consistent across
turns, so one linear direction suffices.

## Encoding (this codebase's conventions, verified)

- `OthelloBoardState.get_state()` returns `state + 1` → **white=0, blank=1,
  black=2** (absolute).
- Black moves first; at sequence position `t` (0-indexed move index) the board
  reflects moves `0..t`, and the side to move **next** is black iff `t` is odd.
- **mine/yours = absolute with the two non-empty classes swapped (`v → 2−v`) on
  the positions where the side-to-move is white (even `t`)**, so the mover's
  colour is always class 2 ("mine"). Verified against reconstructed boards (at
  `t=0`, white-to-move-next, the lone white piece is correctly labelled "mine").
  Which parity is flipped is arbitrary — the other choice is a global
  mine↔yours relabel that a linear probe fits identically; what matters is that
  the flip is **player-relative (parity-based)**, not absolute.

For each layer we train, on the **same** activations, a linear probe on
mine/yours labels and a linear probe on absolute labels, to isolate the effect
of the encoding alone.

## Results

Board-reconstruction error (1 − per-tile 3-class accuracy), 118k val positions:

| Layer | Linear **mine/yours** | Linear **absolute** |
|------:|----------------------:|--------------------:|
| 1 | 9.56% | 19.28% |
| 2 | 6.03% | 18.50% |
| 3 | 3.59% | 18.22% |
| 4 | 2.38% | 18.18% |
| 5 | 1.56% | 17.70% |
| 6 | 1.12% | 17.60% |
| **7** | **0.99%** ← best | 17.72% |
| 8 | 1.03% | 17.76% |

## Interpretation

- The encoding change **alone** — identical activations, identical linear probe
  architecture — flips the board from "linearly undecodable" (~17–19%, matching
  Li et al.'s conclusion) to "linearly decodable" (~1%). This reproduces Nanda's
  correction: Othello-GPT **does** carry a linear emergent world model, in the
  player-relative (mine/yours) frame rather than the absolute-colour frame.
- The best single linear mine/yours probe (0.99% @ L7) even edges out our
  nonlinear *absolute* probe (1.41% @ L7), consistent with the representation
  being fundamentally **linear** once the correct frame is used.
- Error falls monotonically through the early/middle layers and bottoms out
  around layers 6–7, mirroring the depth at which the absolute nonlinear probe
  peaks in the original paper — the board state is built up over the stack.

## Causal intervention (the linear direction is *used*, not just present)

Decodability alone does not prove the model computes with this representation.
We therefore intervene: edit the residual stream along the **linear mine/yours**
probe direction to flip one tile's owner (layers 4–8, propagated layer-by-layer
via `GPTforProbeIA`), then measure whether the model's legal-move predictions
follow the counterfactual board. Identical benchmark (200 cases), layers, and
legality metric as the original nonlinear intervention (`intervene.py`) — only
the probe changes (linear mine/yours vs nonlinear absolute).

| Intervention probe | Baseline err | Intervened err | Reduction |
|---|---:|---:|---:|
| **Linear mine/yours** (Nanda) | 1.365 | **0.695** | **49.1%** |
| Nonlinear absolute (Li et al. style) | 1.365 | 0.925 | 32.2% |

The linear mine/yours direction is not only decodable — it is a **stronger
causal handle** on the model's behaviour than the nonlinear absolute direction.
This cements the linear result: Othello-GPT genuinely computes with a linear,
player-relative board representation.

Code `src/intervene_nanda.py`, `slurm/intervene_nanda.slurm`; data
`results/intervention_nanda_full.json` (vs `results/intervention_full.json`).

## Reproduce

```bash
sbatch slurm/nanda_probe.slurm            # profile=full (default), ckpt = full model
# or:  PROFILE=pilot sbatch slurm/nanda_probe.slurm
sbatch slurm/intervene_nanda.slurm        # linear mine/yours causal intervention
```
