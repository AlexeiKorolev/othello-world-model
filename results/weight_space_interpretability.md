# Weight-space interpretability of the bilinear Othello-GPT

Goal: recover the emergent board world-model from the WEIGHTS (not activations),
using the bilinear-MLP eigendecomposition of Dooms et al. 2024 (arXiv:2410.08417).
Code `src/bilinear_eigen.py`; model `ckpts/gpt_synthetic_full_bilinear.ckpt`
(expansion 8/3, legal-move error 0.036%).

## Method

A bilinear MLP has no elementwise nonlinearity, so its contribution to any output
direction u is an exact quadratic form of the MLP input x = ln2(residual):

    u . MLP(x) = x^T Q_u x ,   Q_u = W^T diag(P^T u) V   (symmetrized)
               = sum_i lambda_i (q_i . x)^2    (eigendecomposition)

The eigenvectors q_i are input directions read from weights; lambda_i their signed
importance. Identity verified to 1e-6 (corr 1.0). Two choices of output direction u:

- **board mode**: u = mine/yours probe logit-difference per tile (an IMPOSED lens).
- **unembed mode**: u = ln_f.weight (o) head.weight[m], the move-legality logits
  (the model's NATIVE output), decomposing the final block; eigenfeatures then
  projected onto the board subspace to see if legality is built FROM the board.

Extracted over 117,961 held-out synthetic positions.

## Result 1 — the board is the wrong basis (board mode)

| Layer | mine/yours probe err | board\|corr\| (MLP→board dir) | recon@2 |
|---|---|---|---|
| L4 | 3.15% | 0.240 | 0.29 |
| L5 | 2.44% | 0.122 | 0.33 |
| L6 | 2.04% | **0.029** | 0.38 |
| L7 | 2.03% | **0.075** | 0.29 |
| L8 | 2.08% | **0.008** | **0.834** |

Where the board is MOST decodable from activations (L6-7, ~2% probe error), the
MLP's contribution to the board-tile readout is ~zero. The deep MLPs do not write
the board. L8 is cleanly rank-2 (recon@2 0.83) but board-orthogonal (corr 0.008):
the low-rank structure exists, it just isn't the board. Board-tile readouts are
also NOT low-rank (effective rank ~13-35) — expected: a tile's legality relevance
depends on line-scans in up to 8 directions, no reason to be rank-2.

## Result 2 — native basis is clean & low-rank, but board-orthogonal (unembed mode)

Final layer (L8), 54 move tokens (0.02 < legal-rate < 0.98):

| Metric | Value | Chance/ref |
|---|---|---|
| effective rank per move | median **1.2** (1.1-3.2) | out of 512 |
| reconstruction @ k=1/2/4 | 0.38 / 0.62 / 0.73 | — |
| legality eigenfeatures in board subspace | **0.054** (max 0.070) | chance 0.125 |
| L8 MLP contribution vs true legality | median 0.19 (max 0.32) | — |

In the model's OWN output basis the MLP is strikingly low-rank (~1 eigenfeature per
move) — the paper's headline structure, which appeared ONLY once we stopped
imposing the board. But those legality eigenfeatures sit at **0.4x chance** overlap
with the board subspace (uniformly, across all 54 moves): the final-layer legality
computation actively avoids the mine/yours board directions, and its contribution
only weakly tracks true legality (the residual stream already carries most of it).

## Conclusion

**The emergent board world-model in this Othello-GPT is a residual-stream /
attention phenomenon, not an MLP-weight one.** The bilinear MLPs perform clean,
very low-rank corrective operations on the move logits; they neither store the
board (board mode) nor read it to compute legality (unembed mode). Weight-based
bilinear interpretability recovers real low-rank MLP features here — but the world
model is not inside the MLPs, so eigendecomposing them alone does not surface it.
Li et al. / Nanda read the board from ACTIVATIONS for the same reason: that is
where it lives.

### Caveat & next step
Only the FINAL layer is analyzed in the native basis (the only MLP whose output
reaches the logits through just ln_f+head). The board->legality conversion likely
happens in mid layers (5-7) and/or in ATTENTION, and is mostly spent by L8. The
untested component — and the natural mechanism for gathering board tiles along the
8 lines through a move — is **attention (OV/QK circuits)**. That, not more MLP
eigendecomposition, is where to look next for the world model in the weights.

## Result 3 — the storage map: exact residual decomposition (board_map.py)

Because mine/yours is *linearly* decodable, the readout w.resid decomposes EXACTLY
and additively into per-component contributions (embedding, all n_head*n_layer
attention heads, all n_layer MLPs); reconstruction corr 1.000. Per component we
score `board_corr` (does its contribution track the true board?) and `share`
(fraction of the readout's variance it supplies). Run on bilinear AND the GELU
baseline (identical except the MLP).

| | Bilinear | GELU |
|---|---|---|
| linear board readout err | 2.08% | 1.03% |
| attn share / mlp share | 0.31 / 0.69 | 0.72 / 0.27 |
| components board_corr>0.3 | **9/64 heads, 0/8 MLPs** | **18/64 heads, 0/8 MLPs** |
| max board_corr attn / mlp | 0.58 / 0.20 | 0.61 / 0.22 |
| top board heads | A5.h3, A6.h0, A5.h6, A7.h5 | A5.h5, A6.h5, A7.h5, A5.h7 |

**The board lives in ATTENTION, in both architectures, never in the MLPs.** Every
component with real board signal (board_corr>0.3) is an attention head; not one MLP
exceeds 0.22 in either model. A sparse set of MID-LAYER heads (4-7) carries it. The
MLPs' large *share* is board-ORTHOGONAL variance (high share, low board_corr) --
they drive the readout axis magnitude, not its board content, exactly matching the
eigendecomposition (MLPs = low-rank board-orthogonal ops).

**Making the MLPs bilinear did NOT push the board into attention** -- it was already
there in GELU. If anything bilinear made the attention board-code LESS redundant
(9 vs 18 heads) while its MLPs inject MORE board-orthogonal variance onto the
readout (share 0.69 vs 0.27). This mechanistically explains why the bilinear board
is ~2x harder to read (2.08% vs 1.03%) despite identical task error (0.036%): the
board signal still sits in a few heads, but it is less redundantly encoded and the
bilinear MLPs crowd the readout axis with orthogonal signal, hurting linear
separability.

**Interpretability lead:** GELU shows a clean "h5" family -- A5.h5/A6.h5/A7.h5, the
same head index writing the board across three consecutive layers (0.61/0.54/0.45)
-- a candidate board-tracking head circuit. Next: inspect what these heads ATTEND
to (do they attend along the 8 lines through the current move -- the flip geometry?).

## Result 4 — the board IS in the attention weights: OV self-transport (attn_ov_board.py)

board_map showed board signal flows through certain heads at the ACTIVATION level.
This tests the WEIGHTS: a head's OV circuit W_OV = W_O^(h) W_V^(h) maps what it reads
(ln1 of the source residual) to what it writes. Metrics vs the mine/yours board
subspace (random-weight calibration exact: out 0.125=r/D, io 0.0156=(r/D)^2):
  out_frac = ||B_out^T W_OV||_F^2/||W_OV||_F^2       (write lands in board subspace)
  io_frac  = ||B_out^T W_OV B_in||_F^2/||W_OV||_F^2  (board -> board through weights)
  T[t,s]   = w_out_t . W_OV . enc_in_s               (inject src tile s -> read dst t)

| | Bilinear | GELU |
|---|---|---|
| top head by io_frac | A5.h3 = **12.5x chance** | A5.h5 = **12.2x chance** |
| 2nd | A6.h0 = 9.9x | A6.h5 = 9.2x |
| transfer diag / off-diag | 9.5-9.7x | 10.7-11.5x |
| corr(weight io_frac, activation board_corr), 64 heads | **0.71** | **0.64** |

**Yes -- the attention-head weights map the board.** (1) The OV weights map
board->board at ~12x chance, in exactly the heads board_map flagged (A5.h5/h3,
A6.h5/h0). (2) The purely-weight io_frac correlates 0.6-0.7 with the activation
board_corr across all 64 heads -- two independent methods pick the same heads.
(3) The tile x tile transfer matrix is ~10x DIAGONAL-dominant: the OV weights
transport board state tile -> SAME tile, i.e. they COPY the running board forward
along the sequence. These are board-MAINTENANCE heads, not flip-computation heads
(off-diagonal Othello-line structure is uniformly weak, ~0.08). GELU maintains the
board slightly more crisply (diag 11.5x, readout 1.03%) than bilinear (9.7x, 2.08%).

## Overall conclusion

Consistent across four analyses: the emergent board world-model is an ATTENTION
phenomenon. The MLPs (bilinear or GELU) do low-rank operations orthogonal to the
board and neither store nor read it. The board lives in a sparse set of mid-layer
attention heads, whose OV WEIGHTS implement a diagonal board self-transport --
copying the running board state forward position-to-position. This is directly
readable from the weights (io_frac ~12x chance, diagonal transfer matrix), and the
weight-based head ranking matches the activation-based one at r~0.7. So for this
Othello-GPT the world model IS recoverable from weights -- just from the attention
OV circuits, not the (bilinear) MLPs. Open next step: the QK circuits (which source
positions these heads read) and whether the weak OV off-diagonal carries flip
geometry.

## Files
- `results/bilinear_eigen_full_bilinear.json` (+ _recon/_effrank/_spectra.png) — board mode
- `results/bilinear_eigen_full_bilinear_unembed.json` (+ _recon.png) — unembed mode
- `results/board_map_full_bilinear.json`, `results/board_map_full.json` (+ _*.png heatmaps) — storage map
- `results/attn_ov_board_full_bilinear.json`, `results/attn_ov_board_full.json` (+ _transfer.png) — OV board map
- `src/bilinear_eigen.py --output-mode {board,unembed}`, `src/board_map.py --arch {bilinear,gelu}`,
  `src/attn_ov_board.py --arch {bilinear,gelu}`
- `slurm/bilinear_eigen.slurm`, `slurm/board_map.slurm`, `slurm/attn_ov_board.slurm`
