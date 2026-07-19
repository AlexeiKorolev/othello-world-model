# othello-gemma — session handoff

Faithful reproduction of Othello-GPT (Li et al. 2023, arXiv:2210.13382) on
Adroit. Train a small GPT from scratch on synthetic Othello games; show the
board state is decodable (probes) and causally used (intervention). Separate
from `../gemma_world_model`.

## Status (2026-06-20)
- Env, vendored upstream, all `src/` scripts + `slurm/` jobs written.
- **Pilot launched**: gen_data ✅ (120k games, 101s). train_gpt (3280429),
  validate_pretrained (3280430) running; train_probe (3280431) queued on
  `afterok` train. Intervention not yet submitted (waits on trained ckpt).
- Authors' weights downloaded + converter verified (~100% legal on spot-check).

## Layout / pipeline
See `README.md`. Order: gen_data → (train_gpt ∥ validate) → probe → intervene →
plot_results. Profiles `pilot` (1 GPU, fast) and `full` (paper scale, multi-GPU).

## Key facts (so you don't re-derive)
- **Canonical vocab** (`src/common.py`): 61 tokens = `-100` (pad → idx 0,
  matches GPT loss `ignore_index=0`) + the 60 non-centre tiles. Pinned so pilot
  / full / authors' checkpoints all share one stoi.
- **Architecture**: 8L/8H/512d, block_size 59, ~25.3M params. minGPT, torch-2.x
  clean. Upstream `Trainer` auto-wraps `nn.DataParallel` → request more GPUs to
  scale the full run.
- **Authors' ckpt** `ckpts/pretrained/synthetic_model.pth` is **TransformerLens
  format** (folded LN, separate `b_U`). `validate_pretrained.convert_tl_to_mingpt`
  maps it to minGPT (LN γ=1/β=0; head bias attached separately). Self-checks via
  legal-move error.
- **Probe deviation**: we probe over held-out synthetic val games (upstream used
  championship). Linear + nonlinear (mid_dim 64) per layer.
- **Intervention** (`src/intervene.py`): trains mid_dim-128 probes at layers 4–8,
  propagates a tile-flip edit layer-by-layer (port of the attribution notebook),
  scores legality error vs the intervened board. Metric is **reconstructed**
  (exact upstream scoring not shipped). Target: baseline ~2.68 → intervened ~0.12.

## Adroit conventions (see `princeton-adroit` skill)
- GPU jobs: `--partition=gpu --gres=gpu:1` (nodes: 4×A100-80, 8×A100-MIG-20,
  4×V100). CPU jobs: `--partition=all`. **No `cpu` partition.**
- Env in `/scratch` (`$ROOT/env`); activate via
  `/scratch/network/ak9088/anaconda3/etc/profile.d/conda.sh`. `/home` is over
  quota — keep everything in `/scratch`.
- Every job emails on completion: `--mail-type=END,FAIL,TIME_LIMIT`,
  `--mail-user=alexeikorolev7@gmail.com,ak9088@princeton.edu`.
- Bash cwd resets between calls → absolute paths only.

## Follow-up experiments (post-replication)
- **Nanda linear mine/yours probe** — DONE. A *single* linear probe (both
  black- and white-to-move plies, mine/yours labels) decodes the board at
  **0.99% @ L7** vs **~17.7% absolute** — the representation is linear in the
  player-relative frame. Writeup `results/nanda_linear_probe.md`; data
  `results/nanda_probe_full.{json,png}`; code `src/nanda_probe.py`,
  `slurm/nanda_probe.slurm`.
- **Bilinear-MLP GPT** — TRAINED (2026-07-07). Same arch, bilinear MLPs
  (arXiv:2410.08417), expansion 8/3 (param+FLOP-matched, 25.3M).
  `src/bilinear_model.py`, `src/train_gpt.py --arch bilinear`. Pilot verified it
  learns (3.54% vs GELU 2.84%). **Full run done: legal-move error 0.036%**
  (`results/gpt_full_bilinear_eval.json`) — matches GELU baseline 0.032%
  (authors' anchor 0.017%), so bilinear solves the task as well as GELU.
  ckpt `ckpts/gpt_synthetic_full_bilinear.ckpt`. **Probe comparison DONE** →
  writeup `results/bilinear_comparison.md`. Nanda linearity reproduces (mine/yours
  linear 2.03% @L7 vs absolute 16.8% = 8.3× gap); but bilinear board rep is ~2×
  harder to read than GELU (nonlinear 2.60% vs 1.41%, mine/yours 2.03% vs 0.99%)
  despite identical task error. Outputs `results/probe_full_bilinear.json`,
  `results/nanda_probe_full_bilinear.{json,png}` vs GELU `probe_full.json`,
  `nanda_probe_full.json`. (Gotcha: train_probe.slurm reads `PROBE_ARGS`,
  nanda_probe.slurm reads `NANDA_ARGS` — different env vars.)
- **Weight-space interpretability** — DONE. `src/bilinear_eigen.py`
  (`--output-mode {board,unembed}`), `slurm/bilinear_eigen.slurm`. Folds each
  bilinear MLP into the exact quadratic form `u.MLP(x)=x^T Q_u x=sum lambda_i
  (q_i.x)^2` (identity verified 1e-6) and eigendecomposes vs an output direction.
  **Finding (writeup `results/weight_space_interpretability.md`):** the emergent
  board world-model is a RESIDUAL/ATTENTION phenomenon, NOT in the MLP weights.
  board mode: deep-MLP contribution to board dirs ~0 (L6-8 board|corr| 0.03/0.08/
  0.008) even though board is ~2% probe-decodable there. unembed mode (final layer
  vs move-legality logits, the model-native basis): MLP is strikingly low-rank
  (eff_rank med 1.2, recon@2 0.62 — the paper's structure, appears ONLY in native
  basis) BUT legality eigenfeatures are board-ANTI-aligned (0.054 vs chance 0.125)
  and weakly predict legality (0.19). So bilinear MLPs do clean low-rank corrective
  ops orthogonal to the board; the world model isn't in them. Outputs
  `results/bilinear_eigen_full_bilinear{,_unembed}.json` + PNGs. **Next: attention
  (OV/QK) circuits** — untested component, prime suspect for board-building /
  board->move routing. Also open: mid-layer (5-7) unembed via propagated directions.
- **Board storage map** — DONE. `src/board_map.py` (`--arch {bilinear,gelu}`),
  `slurm/board_map.slurm`. Exact additive decomposition of the linear mine/yours
  readout into every component (emb, all 64 attn heads, all 8 MLPs; reconstruction
  corr 1.000). **Finding: the board lives in ATTENTION, both archs, never MLPs** —
  board_corr>0.3 only for attn heads (bilinear 9/64, GELU 18/64), 0/8 MLPs (max MLP
  0.22). Sparse mid-layer heads (4-7). MLPs' large *share* (bilinear 0.69, GELU
  0.27) is board-ORTHOGONAL variance. Making MLPs bilinear did NOT push board into
  attention (already there); it made attention board-code LESS redundant (9 vs 18)
  and MLPs inject more orthogonal variance → mechanistically explains bilinear's ~2x
  worse board readout (2.08% vs 1.03%) at equal task error. Lead: GELU "h5 family"
  A5.h5/A6.h5/A7.h5 (0.61/0.54/0.45) = candidate board-tracking head circuit. Next:
  attention patterns of these heads (do they attend along the 8 lines through the
  move?). Outputs `results/board_map_full{,_bilinear}.json` + PNGs; full synthesis
  in `results/weight_space_interpretability.md`.
- **OV-circuit board map** — DONE. `src/attn_ov_board.py`, `slurm/attn_ov_board.slurm`.
  Tests whether head WEIGHTS map the board: W_OV=W_O W_V board->board projection
  (io_frac, chance (r/D)^2) + tile×tile transfer T[t,s]=w_t·W_OV·enc_s. **Answer:
  YES.** Top heads (A5.h3/A6.h0 bilinear, A5.h5/A6.h5 GELU) map board->board at
  ~12x chance; weight io_frac correlates 0.6-0.7 with activation board_corr over 64
  heads (two independent methods agree); transfer matrix ~10x DIAGONAL -> heads COPY
  board state tile->same-tile = board-MAINTENANCE heads (not flip-computation;
  off-diag weak ~0.08). GELU crisper (diag 11.5x, 1.03%) than bilinear (9.7x, 2.08%).
  So the world model IS weight-recoverable — from attention OV circuits, not MLPs.
  Outputs `results/attn_ov_board_full{,_bilinear}.json` + _transfer.png. Open next:
  QK circuits (which positions these heads read) + whether OV off-diag has flip geometry.
  Full run was a **single 47h job on 2xA100-80GB**:
  `slurm/train_gpt_bilinear.slurm` (`sbatch` it). **MUST pin A100**
  (`--gres=gpu:nvidia_a100:2`): batch 4096 needs ~31GB/GPU and OOMs on the 32GB
  V100s (h11g3) — an unpinned `gpu:2` chain died this way. Uses `--resume` (no
  `--wall_hours`) purely as crash-insurance (checkpoints full state each epoch to
  `<ckpt>.state`; re-`sbatch` to resume). Done signal:
  `results/gpt_full_bilinear_eval.json`.
  Resumable/chunked infra kept for reuse but unused now: `src/resumable_trainer.py`
  (general `--resume --wall_hours`), `slurm/train_gpt_bilinear_chain.slurm` +
  `submit_bilinear_chain.sh` (afterany dependency chain),
  `slurm/train_gpt_bilinear_chunk.slurm` (self-resubmit variant).
  `--arch bilinear` is wired into `train_probe.py` + `nanda_probe.py`, so after
  training: `sbatch --export=ALL,PROBE_ARGS='--arch bilinear' slurm/train_probe.slurm`
  and the nanda equivalent to compare bilinear vs GELU board representations.

## Expected numbers (paper, synthetic)
legal-move ~0.01% · linear probe ~20–27% · nonlinear ~1.8% (≈layer 6) ·
intervention 2.68 → 0.12. Pilot will be worse (less data/training); use `full`
to hit paper numbers.

## Next steps
1. When pilot train done → probe runs automatically; submit
   `sbatch --dependency=afterok:<train> slurm/intervene.slurm`.
2. Check `results/*.json`; run `plot_results.py`. Sanity: anchor
   (`validate_pretrained.json`) should already match paper closely — if pilot is
   far off, that's expected (scale), if anchor is off, harness bug.
3. If pilot looks right, launch `full` profile (multi-GPU train; bump time to
   2–4 days). Then re-probe/intervene at `full`.
4. Open: pick final intervention metric; optional random-init probe control
   (`PROBE_ARGS=--random`); optional championship track.
