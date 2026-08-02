# Othello-GPT: A Linear, Causal Board Model and How It Emerges

**Research report — outline form.** Last updated 2026-08-02.
Repo: `BilinearOthelloGPT` · all numbers below trace to a file in `results/`.

---

## 0. One-paragraph summary

We reproduce Othello-GPT (Li et al. 2023) at full scale, then push past
replication on three fronts. The board world-model is **linear in the
player-relative (mine/yours) frame** and **causally used** — and we calibrate
just *how* causally used, against oracle bounds computed from the game engine.
Training-time tracing then shows the board appears very early (~epoch 2–4 of 20)
and its *causal* usefulness saturates well before its *decodability* does, while
the optimal linear readout of it keeps drifting for the entire run.
Swapping the MLPs for bilinear MLPs solves the task equally well but leaves the
board measurably harder to read.

---

## 1. Headline findings (the four things worth remembering)

| # | Finding | Key number | Source |
|---|---|---|---|
| 1 | Replication holds at full scale | legal-move error **0.046%** (authors' ckpt 0.017%) | `gpt_full_h200_eval.json` |
| 2 | Board rep is **linear** in the mine/yours frame | **1.12%** err @ L7 vs **17.7%** absolute-color — a 15.8× gap | `nanda_probe_full_h200.json` |
| 3 | The board is **causally used**, and we now know how much | 1.36 → **0.67** illegal moves; calibrated floor 0.0 / ceiling 1.39 ⇒ intervention closes **~51%** of the gap | `intervention_nanda_full_h200.json`, `floors_full_h200.json` |
| 4 | The **linear readout drifts** through training even after decodability plateaus | probe-direction cosine ep2↔ep20 = **0.53** (same-epoch, different-seed: **0.999**) | `probe_transfer_full_h200.json` |

---

## 2. Setup

- **Task/model**: 8L / 8H / 512d GPT (~25.3M params), block size 59, 61-token
  vocab, trained from scratch on synthetic random-legal Othello games. minGPT
  under the hood, upstream `likenneth/othello_world` vendored unmodified.
- **Two architectures, param- and FLOP-matched**:
  - `gelu` — standard baseline.
  - `bilinear` — bilinear MLPs (Dooms et al. 2024, arXiv:2410.08417),
    expansion 8/3. A drop-in MLP replacement with no elementwise nonlinearity.
- **Anchor**: the authors' released checkpoint, converted TransformerLens →
  minGPT, re-scored through our harness (self-check: ~0 legal-move error).
- **Compute**: UW Hyak *klone*. Main runs on H200/A100; analysis jobs on A40.
  Full bilinear train = one 47h job on 2×A100-80GB.

---

## 3. Result 1 — Replication (the baseline everything else stands on)

- Legal-move error, full scale: **GELU 0.032%**, **GELU/H200 rerun 0.046%**,
  **bilinear 0.036%** — vs authors' anchor 0.017%. All three architectures solve
  the task equivalently.
- Classic probe picture reproduces: **linear absolute-color probe never beats
  ~17.6%**, while a **nonlinear (MLP) probe hits 1.41% @ L7**. This is the
  original paper's "the board is there but not linearly readable" result.
- **New calibration — an entropy floor for the loss.** Under random-legal play
  the irreducible next-move cross-entropy is
  **2.011 nats** (train; mean 8.62 legal moves/position). Any reported loss
  should be read against this, not against 0.

## 4. Result 2 — The board is linear, in the right frame

- Replicating Nanda's reframing: label tiles **mine/yours** (relative to the
  player to move) instead of black/white, and train a *single* linear probe
  across both plies.
- **1.12% error @ L7** vs **17.7%** for the absolute-color linear probe.
- Interpretation: the earlier "nonlinearity" was a **coordinate-frame artifact**.
  The model stores a player-relative board, and it stores it linearly.
- Layer profile: error falls monotonically L1→L7 (9.5% → 1.12%) and ticks back
  up slightly at L8 (1.17%) — the board is *built* through the mid stack.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figs/fig1_linearity_dark.png">
  <img alt="Linear probe error by layer, log scale. The absolute black/white probe stays flat near 18% across all 8 layers. The player-relative mine/yours probe falls from 9.5% at layer 1 to 1.17% at layer 8, a 15.8x gap at layer 7." src="results/figs/fig1_linearity_light.png" width="720">
</picture>

## 5. Result 3 — Causal use, now with a calibrated scale

- **Intervention**: train mine/yours probes at L4–L8, edit one tile in the
  residual stream, propagate the edit layer-by-layer, score how many of the
  model's top-N moves are illegal *on the edited board*.
- Measured: **baseline 1.36 → intervened 0.67** illegal moves (200 cases).
- **New: what do those numbers mean?** We computed reference points directly
  from the game engine:
  - **0.000** = an oracle that fully honours the edit (floor).
  - **1.388 ± 0.547** = an oracle that completely *ignores* the edit and plays
    the true board (ceiling).
- So: the un-intervened model sits **exactly at the "ignores the edit" ceiling**
  (1.36 vs 1.388 — as it must), and the intervention moves it **~51% of the way
  to a perfect edit-follower**. The edit is genuinely causal, and it's
  genuinely partial. This replaces a previously uncalibrated metric.
- Both the floor/ceiling and the intervention are computed over the **same 200
  benchmark cases** (`bench[:n_cases]` from one config key), so the comparison
  is not confounded by case selection.
- The floor is a *perfect* edit-follower, not a perfect version of this model.
  This model's own intrinsic error contributes ~0.005 expected illegal moves at
  N≈10, so the distinction moves the 51% by well under a percentage point.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figs/fig2_intervention_dark.png">
  <img alt="Horizontal bars of illegal moves in the model's top-N. Un-intervened model 1.36, sitting on the oracle ceiling of 1.39 that ignores the edit. After the board edit 0.67, roughly halfway to the oracle floor of 0.0. The edit closes 51% of the gap." src="results/figs/fig2_intervention_light.png" width="760">
</picture>

## 6. Result 4 — Bilinear MLPs: same task error, slightly worse-read board

- Bilinear matches GELU on the task (**0.036% vs 0.032%** legal-move error), so
  the architecture swap costs nothing on the objective.
- But its board is consistently **~2× harder to read**:

  | Probe | GELU (`full`) | Bilinear (`full`) |
  |---|---|---|
  | nonlinear @ L7 | 1.41% | 2.60% |
  | linear mine/yours @ L7 | 0.99% | 2.03% |

  **Run bookkeeping**: this table compares the two `full`-profile checkpoints,
  which is the matched pair — the bilinear model was trained under that recipe.
  §4's headline 1.12% comes from the separate `full_h200` GELU run. The two GELU
  numbers (0.99% vs 1.12%) are the same architecture trained twice, and their
  spread is a useful reminder of the run-to-run noise floor on these probes.
- The mine/yours linearity result of §4 reproduces in bilinear too (2.03%
  mine/yours vs **16.97%** absolute @ L7 = **8.4×** gap), so this is a difference
  in how *sharply* the board is written, not in what kind of representation it is.
- **The board code is about half as redundant under bilinear.** Decomposing the
  linear mine/yours readout over all 64 attention heads, the number that
  correlate with it at **> 0.3** is:

  | | GELU | Bilinear |
  |---|---|---|
  | heads with board_corr > 0.3 | **18 / 64** | **9 / 64** |
  | strongest heads | A5.h5 0.61, A6.h5 0.54, A7.h5 0.45 | A5.h3 0.58, A6.h0 0.57, A5.h6 0.36 |

  The *peak* is essentially unchanged (0.61 vs 0.58) — bilinear is not missing a
  strong board signal, it just spreads it over half as many heads. That the
  sparser code is also the harder-to-read one is suggestive, but with one seed
  per architecture it is a correlation between two numbers, not a demonstrated
  cause. Source: `board_map_full{,_bilinear}.json`.
- We did not pin down what drives the readability gap; it is reported here as an
  observation, not an explanation.
- Detail and the supporting per-layer numbers: `results/bilinear_comparison.md`.

## 7. Result 5 — Developmental trace: when does the world model appear?

Probing + intervening at every 2nd epoch of a 20-epoch full run:

| Epoch | best mine/yours probe err | intervention (base → intervened) |
|---|---|---|
| 0 (init) | 22.4% | 8.14 → 8.18 *(no effect — control)* |
| 2 | 5.2% | 1.41 → 1.05 |
| 4 | 2.65% | 1.37 → 0.69 |
| 8 | 1.49% | 1.37 → 0.64 |
| 20 | 1.12% | 1.36 → 0.69 |

- **The board arrives early**: 22% → 5% within 2 epochs, → 2.65% by epoch 4.
- **Causal saturation precedes decodability saturation**: intervention efficacy
  is fully mature by **epoch 4** and never improves again, while probe error
  keeps falling 2.65% → 1.12% over the remaining 16 epochs. The last 60% of
  probe improvement buys **zero** additional causal control.
- Noise check: with n=200 and std ≈0.53, the SEM on each intervention number is
  ≈0.037. The epoch-4→20 intervened values span 0.605–0.685, i.e. ~2 SEM with no
  trend — flat within noise, rather than measurably improving.
- Reading: late training refines a representation the model was *already* using
  as much as it ever will — a decodability/causality dissociation.
- Epoch-0 numbers double as the random-init control (22% probe error is the
  class-prior floor; intervention does nothing).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figs/fig3_devtrace_dark.png">
  <img alt="Two stacked panels over training epochs 0 to 20, log scale. Top: best-layer probe error falls from 22.4% at random init to 2.65% by epoch 4 and continues down to 1.12% by epoch 20. Bottom: the un-intervened and intervened illegal-move counts separate by epoch 2 and are flat from epoch 4 onward at about 1.37 versus 0.68." src="results/figs/fig3_devtrace_light.png" width="720">
</picture>

## 8. Result 6 — The optimal linear readout keeps drifting *(newest result)*

Probe-transfer study: train a probe on epoch *i* activations, apply it to
epoch *j*; also compare probe weight directions directly.

- **Cross-epoch cosine (L7)**, between trained probe directions: adjacent late
  epochs ≈ 0.99, falling smoothly and monotonically to **ep2 ↔ ep20 = 0.53**.
  Same pattern at L4. (The epoch-0 row sits at ≈0.02 against everything, but
  carries no weight: at random init there is no board, the probe fits noise at
  the 22% class prior, and its direction is arbitrary. It is excluded from the
  argument below.)
- **Transfer error (L7, per-tile recalibrated)**: on-diagonal 1.09% @ ep20, but
  the epoch-2 probe applied at epoch 20 gives **8.3%**. On raw (un-recalibrated)
  logits cross-epoch transfer collapses to ~50%+ — so a large part of the naive
  failure is scale/offset, consistent with the 7–14× growth in activation norm,
  and only the recalibrated gap is evidence of anything directional.
- **Seed control**: two probes trained at the *same* epoch with different seeds
  agree at **cosine 0.997–0.999**, subspace 0.996–0.999, identical error. The
  drift is therefore not probe-fitting noise — the optimization is reproducible.

**What this does and does not establish.** It establishes that the *optimal
linear readout* of the board moves through training, well after decodability has
plateaued: a probe is a statement about one checkpoint and does not transfer.

It does **not** establish that the model's board encoding rotates, and an earlier
version of this report wrongly claimed it did. A discriminative probe solves
roughly `w ∝ Σ⁻¹Δμ`, so its direction depends both on where the board is written
(`Δμ`) and on the activation covariance `Σ` — which changes enormously over these
same epochs (mean ‖h‖ grows 12.1→130.7 at L4, 16.3→225.8 at L7, monotonically).
A fixed encoding under a drifting covariance produces exactly this signature. The
seed control does not separate the two, because both probes see identical
activations, so `Δμ` and `Σ` are held fixed together.

**The test that would separate them** (not yet run): compare *generative*
directions `Δμ = mean(h | mine) − mean(h | yours)` per tile, which involve no
`Σ⁻¹`, on unit-normalized activations. If `Δμ` cosines stay high while probe
cosines fall, the drift is a readout/whitening effect and the encoding is stable;
if they fall together, the encoding itself is moving.

- **Geometry side-note**: the 64 tile directions are **not** low-dimensional —
  top-3 PCs explain only **9.2% (L4) / 10.0% (L7)**, against ~4.7% for 64
  perfectly isotropic directions. No compact 3D board manifold. (These are
  probe directions too, so the `Σ` caveat applies, but the conclusion is robust
  to it: a covariance artifact does not manufacture rank this high.)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/figs/fig4_transfer_dark.png">
  <img alt="An 11 by 11 heatmap of cosine similarity between layer-7 probe directions trained at different epochs. The diagonal is 1 by construction; values decay smoothly away from it, from 0.99 between adjacent late epochs down to 0.53 between epoch 2 and epoch 20, and about 0.02 for anything against epoch 0." src="results/figs/fig4_transfer_light.png" width="660">
</picture>

---

## 9. What this adds up to

1. **Frame matters more than nonlinearity.** The original "needs a nonlinear
   probe" result dissolves under the player-relative frame.
2. **Causal claims need a scale.** "1.36 → 0.67" means nothing until you know
   the oracle bounds; against them it is a real but partial (~51%) effect.
3. **Emergence is early, and the refinement afterwards is causally inert.**
   Probe error and causal control come apart after epoch 4.
4. **A probe is a statement about one checkpoint.** The board stays decodable
   while the optimal linear readout of it keeps moving — whether the encoding
   moves with it is still open (§8).

## 10. Limitations / caveats

- Probes are trained on held-out **synthetic** games (upstream used championship
  games). Representational question unchanged; absolute numbers not directly
  comparable to the paper's championship track.
- The intervention legality metric is **reconstructed** (upstream never shipped
  the exact scoring) — §5's floor/ceiling calibration is the fix for that, but
  cross-paper comparison of the raw number is still unsafe.
- Developmental study: epochs 0–14 recorded only the best layer's probe error
  (a preempted-job bug, fixed after that run); full per-layer breakdown exists
  from epoch 16 on.
- Intervention n = 200 cases; transfer/PCA studies use 2,000 games
  (117,961 positions), single seed for the model itself.
- Everything is one model size (25.3M) on one synthetic distribution.
- Every "direction" in this report is a **discriminative probe weight**, not a
  measured encoding direction. Where a claim depends on the distinction it is
  flagged (§8); elsewhere the claims are about decodability, which is what a
  probe legitimately measures.
- §6's GELU/bilinear comparison is one training run per architecture. The two
  GELU runs available (0.99% vs 1.12% mine/yours @ L7) bound the run-to-run
  noise at ~0.13pp, well under the ~1pp GELU/bilinear gap, but a single seed per
  arm still cannot separate architecture from initialization.

## 11. Open threads (ranked)

1. **Why does causal efficacy cap at ~51%** of the calibrated floor? Is the
   remaining half a probe-quality limit, or does the model also read the board
   from somewhere the edit doesn't reach?
2. **Settle §8**: generative `Δμ` directions on normalized activations, to say
   whether the encoding moves or only the readout does. Cheap — reuses the
   cached activations, no probe training.
3. **Multi-seed models** to test whether the drift converges to a
   seed-independent endpoint, or is genuinely seed-specific.
4. **What makes the bilinear board less sharp** at equal task error (§6).
5. **Championship-game track** to make the numbers directly comparable to the
   paper's.

## 12. Reproduction map

| Result | Code | Job | Output |
|---|---|---|---|
| Training / eval | `src/train_gpt.py` | `slurm/train_gpt*.slurm` | `gpt_*_eval.json` |
| Anchor | `src/validate_pretrained.py` | `slurm/validate.slurm` | `validate_pretrained.json` |
| Classic probes | `src/train_probe.py` | `slurm/train_probe.slurm` | `probe_*.json` |
| Mine/yours probe | `src/nanda_probe.py` | `slurm/nanda_probe.slurm` | `nanda_probe_*.json` |
| Intervention | `src/intervene_nanda.py` | `slurm/intervene_nanda.slurm` | `intervention_nanda_*.json` |
| Floors/calibration | `src/entropy_floor.py` | `slurm/entropy_floor.slurm` | `floors_full_h200.json` |
| Developmental study | `src/dev_study.py` | `slurm/dev_study_full_h200.slurm` | `dev_study_full_h200.json` |
| Probe transfer | `src/probe_transfer.py` | `slurm/probe_transfer.slurm` | `probe_transfer_full_h200.json` |
| Probe geometry | `src/probe_directions.py` | `slurm/probe_directions.slurm` | `probe_pca3d_full_h200.json` |
| Board-code redundancy (§6) | `src/board_map.py` | `slurm/board_map.slurm` | `board_map_full{,_bilinear}.json` |
| Report figures | `src/report_figs.py` | `slurm/report_figs.slurm` | `results/figs/*.png` |

Longer-form writeups: `results/nanda_linear_probe.md`,
`results/bilinear_comparison.md`.

**Figures.** The four above are regenerated from the result JSONs by
`sbatch slurm/report_figs.slurm` (light + dark variants; the report serves each
via `<picture>`). Additional per-study plots already in `results/`:
`nanda_probe_full*.png`, `probe_transfer_full_h200_L{4,7}.png`,
`dev_study_full_h200.png`.

**Side analyses not part of this study.** The repo also contains a
weight-space thread (`src/attn_ov_board.py`, `src/bilinear_eigen.py` and their
outputs, written up in `results/weight_space_interpretability.md`). It is kept
for reference and is not claimed as a result here: that the board is carried by
attention rather than the position-wise MLPs is close to definitional, since
attention is the only component that moves information across tokens. §6 uses
the same decomposition only to *count* board-carrying heads across the two
architectures, which is a claim about the architecture swap, not about where the
board lives.

## 13. References

- Li et al. 2023, *Emergent World Representations* — arXiv:2210.13382
- Nanda et al. 2023, *Emergent Linear Representations in World Models*
- Dooms et al. 2024, *Bilinear MLPs enable weight-based mechanistic
  interpretability* — arXiv:2410.08417

*Compute on UW Hyak "klone" under Jason Hendrick's account, used with
permission.*
