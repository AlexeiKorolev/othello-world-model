# Othello-GPT: Where the World Model Lives, and How It Gets There

**Research report — outline form.** Last updated 2026-08-01.
Repo: `BilinearOthelloGPT` · all numbers below trace to a file in `results/`.

---

## 0. One-paragraph summary

We reproduce Othello-GPT (Li et al. 2023) at full scale, then push past
replication with four new lines of evidence. The board world-model is **linear in
the player-relative (mine/yours) frame**, **causally used**, and — the main
finding — **stored in attention, not in MLPs**: it is recoverable directly from
attention OV *weights* as a near-diagonal tile→tile copy map, while the MLPs
(including exactly-analysable bilinear MLPs) contribute essentially nothing to the
board readout. Training-time tracing shows the board appears very early
(~epoch 2–4 of 20) and its *causal* usefulness saturates before its
*decodability* does, while the probe directions themselves keep slowly rotating
for the entire run.

---

## 1. Headline findings (the five things worth remembering)

| # | Finding | Key number | Source |
|---|---|---|---|
| 1 | Replication holds at full scale | legal-move error **0.046%** (authors' ckpt 0.017%) | `gpt_full_h200_eval.json` |
| 2 | Board rep is **linear** in the mine/yours frame | **1.12%** err @ L7 vs **17.7%** absolute-color — a 15.8× gap | `nanda_probe_full_h200.json` |
| 3 | The board is **causally used**, and we now know how much | 1.36 → **0.67** illegal moves; calibrated floor 0.0 / ceiling 1.39 ⇒ intervention closes **~51%** of the gap | `intervention_nanda_full_h200.json`, `floors_full_h200.json` |
| 4 | The world model lives in **attention OV circuits**, not MLPs | OV board→board mapping ~**12× chance**, transfer matrix ~**10× diagonal**; 0 of 8 MLPs reach board_corr 0.3 | `attn_ov_board_full*.json`, `board_map_full*.json` |
| 5 | The representation **drifts throughout training** even after it stops improving | probe cosine ep2↔ep20 = **0.53**, but cross-seed control at fixed epoch = **0.999** | `probe_transfer_full_h200.json` |

---

## 2. Setup

- **Task/model**: 8L / 8H / 512d GPT (~25.3M params), block size 59, 61-token
  vocab, trained from scratch on synthetic random-legal Othello games. minGPT
  under the hood, upstream `likenneth/othello_world` vendored unmodified.
- **Two architectures, param- and FLOP-matched**:
  - `gelu` — standard baseline.
  - `bilinear` — bilinear MLPs (Dooms et al. 2024, arXiv:2410.08417),
    expansion 8/3. Chosen because a bilinear MLP has *no elementwise
    nonlinearity*, so its output is an exact quadratic form → weights are
    directly eigendecomposable.
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
- Layer profile: error falls monotonically L1→L7 (9.5% → 1.12%), flat at L8 —
  the board is *built* through the mid stack and *consumed* at the top.

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

## 6. Result 4 — Bilinear vs GELU: same task error, worse-read board

- Bilinear matches GELU on the task (0.036% vs 0.032%) but its board is
  consistently **~2× harder to read**:

  | Probe | GELU | Bilinear |
  |---|---|---|
  | nonlinear @ L7 | 1.41% | 2.60% |
  | linear mine/yours @ L7 | 0.99% | 2.03% |

- Nanda linearity reproduces in bilinear too (2.03% mine/yours vs 16.8%
  absolute = 8.3× gap), so this is a *sharpness* difference, not a
  representational-kind difference.
- **Mechanistic explanation found later (§7.2)**: bilinear MLPs inject more
  board-orthogonal variance and the attention board-code is less redundant
  (9 vs 18 board-carrying heads) — so the same board is present but noisier.

## 7. Result 5 — Where the world model actually lives *(the main line)*

### 7.1 It is not in the MLP weights — even when the MLPs are exactly analysable

- Bilinear MLPs fold into an exact quadratic form
  `u·MLP(x) = xᵀQ_u x = Σ λᵢ(qᵢ·x)²` (identity verified to 1e-6), so we can
  eigendecompose the MLP against any output direction.
- **Against board directions**: at the layers where the board is *most*
  decodable (L6–L8, ~2% probe error), the MLP's contribution to the board
  readout is ≈ **0** (|corr| 0.029 / 0.075 / 0.008).
- **Against the model's own legality logits** (native basis): the MLP is
  strikingly **low-rank** — median effective rank **1.2** of 512,
  reconstruction@2 = 0.62. The clean structure the bilinear literature predicts
  *does* exist — but **only in the native basis**, and those legality
  eigenfeatures are **board-anti-aligned** (0.054 vs chance 0.125).
- Conclusion: bilinear MLPs do clean, low-rank, *corrective* work **orthogonal
  to the board**. Imposing the board as a lens is what made prior weight-space
  attempts look structureless.

### 7.2 It is in attention — shown twice, independently

- **Activation side (`board_map.py`)**: exact additive decomposition of the
  linear mine/yours readout into embedding + all 64 heads + all 8 MLPs
  (reconstruction corr = 1.000). Heads with board_corr > 0.3:
  **bilinear 9/64, GELU 18/64. MLPs: 0/8** (max 0.22). The MLPs' large *share*
  of variance (0.69 bilinear / 0.27 GELU) is board-**orthogonal**.
- **Weight side (`attn_ov_board.py`)**: compute `W_OV = W_O W_V` as a
  board→board map, no activations involved.
  - Top heads map board→board at **~12× chance**.
  - Weight-derived `io_frac` correlates **0.6–0.7** with activation-derived
    `board_corr` across all 64 heads — two independent methods agree.
  - The tile×tile transfer matrix is **~10× diagonal** (off-diagonal ~0.08):
    these heads **copy tile state to the same tile** — they are
    **board-maintenance heads**, not flip-computation heads.
  - GELU crisper than bilinear (diagonal 11.5× vs 9.7×) — matching §6.
- **Named circuit lead**: GELU's "h5 family" — A5.h5 / A6.h5 / A7.h5
  (0.61 / 0.54 / 0.45); bilinear's A5.h3 / A6.h0.
- **Headline**: the world model **is** weight-recoverable — just from attention
  OV circuits, not from MLPs.

## 8. Result 6 — Developmental trace: when does the world model appear?

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
- Reading: late training refines a representation the model was *already* using
  as much as it ever will — a decodability/causality dissociation.
- Epoch-0 numbers double as the random-init control (22% probe error is the
  class-prior floor; intervention does nothing).

## 9. Result 7 — The representation keeps rotating *(newest result)*

Probe-transfer study: train a probe on epoch *i* activations, apply it to
epoch *j*; also compare probe directions directly.

- **Cross-epoch cosine (L7)**: adjacent late epochs ≈ 0.99, but
  **ep2 ↔ ep20 = 0.53**, and **ep0 ↔ anything ≈ 0.02**. A smooth monotone decay
  away from the diagonal, at both L4 and L7.
- **Transfer error (L7, per-tile recalibrated)**: on-diagonal 1.09% @ ep20, but
  the epoch-2 probe applied at epoch 20 gives **8.3%** — and on raw (un-recalibrated)
  logits, cross-epoch transfer collapses to ~50%+ error.
- **Critical control**: two probes trained at the *same* epoch with different
  seeds agree at **cosine 0.997–0.999**, subspace 0.996–0.999, identical error.
  So the drift is a property of the **model**, not probe-training noise.
- Interpretation: decodability plateaus by ~epoch 10, but the **basis in which
  the board is written keeps slowly rotating for the whole run**. Probes are
  checkpoint-specific; "the board direction" is not a fixed object across
  training.
- **Geometry side-note**: the 64 tile directions are **not** low-dimensional —
  top-3 PCs explain only **9.2% (L4) / 10.0% (L7)**. No compact 3D board
  manifold; the board is a genuinely high-rank code.

---

## 10. What this adds up to

1. **Frame matters more than nonlinearity.** The original "needs a nonlinear
   probe" result dissolves under the player-relative frame.
2. **Component matters more than architecture.** Making the MLPs bilinear
   changed the *sharpness* of the board code, not its *location* — it was
   already in attention.
3. **The world model is maintenance, not computation.** OV circuits copy board
   state forward tile-to-tile; the flip logic is not (yet) localised.
4. **Weight-space interpretability works when you use the model's own basis.**
   Board-lens = structureless; legality-lens = rank ~1.
5. **Emergence is early and the refinement afterwards is causally inert.**

## 11. Limitations / caveats

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

## 12. Open threads (ranked)

1. **QK circuits** — we know *which* heads carry the board and *that* they copy
   it; we don't know *which positions they read*. Do the h5-family heads attend
   along the 8 lines through the just-played move?
2. **Where is the flip computation?** OV off-diagonal is weak (~0.08) — flips
   must be computed somewhere. Check whether off-diagonal structure has line
   geometry.
3. **Why does causal efficacy cap at ~51%** of the calibrated floor? Is the
   remaining half a probe-quality limit or a genuine "the model also reads the
   board from elsewhere" effect?
4. **Mid-layer (L5–L7) unembed-mode eigendecomposition** via propagated output
   directions — the native-basis trick only ran on L8.
5. **Multi-seed models** to test whether the rotation in §9 converges to a
   seed-independent endpoint.

## 13. Reproduction map

| Result | Code | Job | Output |
|---|---|---|---|
| Training / eval | `src/train_gpt.py` | `slurm/train_gpt*.slurm` | `gpt_*_eval.json` |
| Anchor | `src/validate_pretrained.py` | `slurm/validate.slurm` | `validate_pretrained.json` |
| Classic probes | `src/train_probe.py` | `slurm/train_probe.slurm` | `probe_*.json` |
| Mine/yours probe | `src/nanda_probe.py` | `slurm/nanda_probe.slurm` | `nanda_probe_*.json` |
| Intervention | `src/intervene_nanda.py` | `slurm/intervene_nanda.slurm` | `intervention_nanda_*.json` |
| Floors/calibration | `src/entropy_floor.py` | `slurm/entropy_floor.slurm` | `floors_full_h200.json` |
| Bilinear eigendecomp | `src/bilinear_eigen.py` | `slurm/bilinear_eigen.slurm` | `bilinear_eigen_*.json` |
| Board decomposition | `src/board_map.py` | `slurm/board_map.slurm` | `board_map_*.json` |
| OV circuits | `src/attn_ov_board.py` | `slurm/attn_ov_board.slurm` | `attn_ov_board_*.json` |
| Developmental study | `src/dev_study.py` | `slurm/dev_study_full_h200.slurm` | `dev_study_full_h200.json` |
| Probe transfer | `src/probe_transfer.py` | `slurm/probe_transfer.slurm` | `probe_transfer_full_h200.json` |
| Probe geometry | `src/probe_directions.py` | `slurm/probe_directions.slurm` | `probe_pca3d_full_h200.json` |

Longer-form writeups: `results/nanda_linear_probe.md`,
`results/bilinear_comparison.md`, `results/weight_space_interpretability.md`.

## 14. References

- Li et al. 2023, *Emergent World Representations* — arXiv:2210.13382
- Nanda et al. 2023, *Emergent Linear Representations in World Models*
- Dooms et al. 2024, *Bilinear MLPs enable weight-based mechanistic
  interpretability* — arXiv:2410.08417

*Compute on UW Hyak "klone" under Jason Hendrick's account, used with
permission.*
