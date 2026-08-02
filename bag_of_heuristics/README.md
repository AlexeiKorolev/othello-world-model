# bag_of_heuristics — is our Othello-GPT's world model a bag of heuristics?

Follow-up to the probe / intervention work in the parent repo (see `../REPORT.md`).
The parent project established that a board state is *decodable* and *causally used*.
This subproject asks the next question: **how is it computed?**

## Source

Primary reference (a LessWrong / Alignment Forum report from Neel Nanda's MATS
stream, mid-2024 — not peer-reviewed):

- **OthelloGPT learned a bag of heuristics**
  https://www.lesswrong.com/posts/gcpNuEZnxAPayaKBY/othellogpt-learned-a-bag-of-heuristics-1
  (mirror: https://www.alignmentforum.org/posts/gcpNuEZnxAPayaKBY/othellogpt-learned-a-bag-of-heuristics-1)

Companions:

- Exploring how OthelloGPT computes its world model —
  https://www.lesswrong.com/posts/wezSznWnsMhpRF2QH/exploring-how-othellogpt-computes-its-world-model
- (Maybe) A Bag of Heuristics is All There Is —
  https://www.lesswrong.com/posts/azCuKGJTecCDLBEEj/maybe-a-bag-of-heuristics-is-all-there-is-and-a-bag-of
- Contrast / prior: Li et al. 2023, Emergent World Representations —
  https://arxiv.org/abs/2210.13382 ; Nanda et al., linear probes —
  https://arxiv.org/abs/2309.00941

## Claim under test

The board representation is real and causal, but it is produced by *many shallow,
partially redundant, board-position-specific rules* rather than one clean
algorithm. Reported signatures:

1. Individual neurons implement rules of the form
   "if square A was played recently AND square B is `mine` → square C is `theirs`",
   specific to one square and often to one flip direction.
2. Rules are many and redundant: ablating any one barely moves accuracy; the
   representation degrades gracefully rather than breaking.
3. The circuit does not generalise — it is tied to this board size / move
   ordering and does not compose into a reusable "flip" subroutine.

## Planned experiments

- [ ] **Neuron rule extraction.** For top board-relevant MLP neurons per layer,
      fit the "recently-played A ∧ state(B) → state(C)" rule form and score fit.
- [ ] **Redundancy / graceful degradation.** Ablate k neurons for k = 1..N,
      per-square; plot probe accuracy vs k. A bag predicts a slow linear decay,
      an algorithm a cliff.
- [ ] **Positional specificity.** Do the neurons for square C transfer to a
      translated square C'? Reuse `../src/probe_transfer.py` machinery.
- [ ] **Move-recency dependence.** Condition probe accuracy on how many moves ago
      the flip happened; heuristics should be recency-biased.
- [ ] **Bilinear angle (novel bit).** Our bilinear variant makes per-neuron
      quadratic forms readable — extract eigenvectors of the board-relevant
      bilinear terms (`../src/bilinear_eigen.py`) and check whether the "rules"
      show up as low-rank interaction terms. This is what the original report
      could not do with a standard GELU MLP.

## Layout

    src/      analysis scripts (import from ../src, don't fork it)
    slurm/    one job per experiment; all compute goes through Slurm
    results/  json / npz outputs, gitignored if large
    notes/    running lab notebook, one file per experiment

## Conventions

Same as the parent repo (see `../CLAUDE.md`). Nothing runs on the login node —
every experiment, including smoke tests, is submitted with `sbatch`.
Checkpoints and data come from `../ckpts` and `../data` symlinks; do not copy.

## Results so far

`src/ood_eval.py` (run: `sbatch slurm/ood_eval.slurm`) → `results/ood_eval_full.json`.
Baseline is the full GELU checkpoint, `gpt_synthetic_full_h200.ckpt`, 300 games
per condition. Re-run it against any retrained checkpoint with
`OOD_ARGS="--ckpt <path> --tag <name>"` to compare training methods.

**Suite 1 — orientation.** Only the stabiliser of the opening position maps
legal games to legal games; the strict legality check confirms this empirically
(1.000 vs 0.000), so the eight transforms split into two very different tests.

| transform | legal | err | TV | |
|---|---|---|---|---|
| identity  | 1.000 | 0.107% | 0.0000 | control |
| rot180    | 1.000 | 0.090% | 0.0248 | in-dist |
| transpose | 1.000 | 0.119% | 0.0253 | in-dist |
| antitrans | 1.000 | 0.096% | 0.0251 | in-dist |
| flip_v    | 0.000 | 2.012% | 0.1111 | colour-swapped |
| rot90     | 0.000 | 4.408% | 0.1363 | colour-swapped |
| rot270    | 0.000 | 6.042% | 0.1168 | colour-swapped |
| flip_h    | 0.000 | 10.348% | 0.1274 | colour-swapped |

- **Competence is orientation-invariant**: 0.090–0.119% across the legal group
  vs 0.107% for identity. No gross per-square blind spots.
- **But the function is not exactly equivariant**: TV = 0.025 where a shared rule
  would give 0. Identity gives exactly 0.0000, so the metric is calibrated.
- **Colour-swapped orientations cost 20–100×** (up to 10.3%), and unevenly —
  flip_h is 5× worse than flip_v despite being the same kind of transform.
  That asymmetry is the bag-of-heuristics signature: there is no reason for a
  rule-based model to find one reflection harder than another.

**Suite 2 — policy shift.** Legal games drawn from non-uniform play policies, so
the boards are legal but off the training distribution.

| policy | err | mean legal moves | vs control |
|---|---|---|---|
| uniform (control) | 0.034% | 8.55 | 1× |
| edge_avoid | 0.057% | 10.05 | 1.7× |
| max_flip | 0.085% | 8.58 | 2.5× |
| min_flip | 0.085% | 8.56 | 2.5× |
| **corner_first** | **0.554%** | 6.47 | **16×** |

Legal-move error rises 16× under corner-first play while the games stay entirely
legal. This is the headline generalisation gap and the number to move.

**Caveat on argmax agreement.** The JSON also stores `argmax_agree`, which sits
at 15–25% even for in-distribution transforms. That is expected, not a finding:
the training target is ~uniform over legal moves, so which legal move wins the
argmax is near-arbitrary. TV distance over the distribution is the metric to
read; argmax agreement is kept only as a secondary diagnostic.
