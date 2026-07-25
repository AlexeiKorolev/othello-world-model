"""Estimate the irreducible floors this project's two headline metrics sit on.

Both floors exist for the same reason: the synthetic dataset is uniformly random
*legal* play (`get_ood_game` -> `random.choice(ab.get_valid_moves())`), so at
every position the ground-truth next-move distribution is exactly uniform over
the legal moves. That pins the Bayes-optimal predictor analytically -- we are
not fitting anything here, just replaying boards and counting.

1. TRAINING-LOSS FLOOR.
   The Bayes-optimal predictor puts mass 1/n_legal on each legal move, so its
   cross-entropy at a position is exactly ln(n_legal). The floor is the mean of
   that over the same target positions the training loss averages over:
   CharDataset sets x = seq[:-1], y = seq[1:], pads with -100 -> index 0, and
   GPT's loss uses ignore_index=0. So the targets of a length-T game are moves
   1..T-1, i.e. the choice sets faced at prefixes 0..T-2. Move 0 is never a
   target and is excluded. Reduction is 'mean' over non-ignored targets, so we
   pool positions across games rather than averaging per game.

   Without this number "training loss 2.020" is uninterpretable, and the
   loss-vs-probe lead/lag question is unfalsifiable: what matters is the EXCESS
   loss over the floor, and the sign of the lead/lag conclusion flips depending
   on whether the floor is nearer 1.95 or 2.02.

2. INTERVENTION LEGALITY-ERROR FLOOR / CEILING.
   `intervene_nanda.legality_error` ranks the 60 move tokens by logit, takes the
   top N (N = number of legal moves on the COUNTERFACTUAL board), and counts how
   many are illegal there -- and it grades BOTH the baseline and the intervened
   run against that same counterfactual board. Two reference points bound the
   achievable range:

     floor   -- a model that perfectly knows the counterfactual board. Scores 0,
                except on cases where the edited board has no legal move at all
                (there N = max(1, 0) = 1 and any prediction is illegal), so the
                floor is slightly above 0 and we measure it rather than assume.
     ceiling -- a model that perfectly knows the REAL board but nothing about the
                edit. This is what the baseline column would read if the GPT were
                flawless, so it separates "irreducible given no knowledge of the
                edit" from "model error", and bounds how much of the baseline ->
                intervened gap the intervention could possibly close.

   A perfect-but-uninformed model is indifferent within its legal set, so its
   score depends on tie-breaking. We report the expectation under uniform random
   tie-breaking plus the best- and worst-case orderings.

SAMPLING NOTE (important): floor 1 is compared against TRAINING loss, so games
are sampled from the data shards, not from `common.get_val_games`. The val split
is `sorted(games)[:val_games]`, which makes all 10k val games share the
lexicographically smallest opening -- fine as a held-out set, useless as a
distributional sample. The val floor is computed too, purely to quantify that
skew.

    python entropy_floor.py --config ../configs/config.yaml --profile full
"""
import os
import json
import time
import math
import random
import argparse
import multiprocessing as mp
from collections import Counter

import numpy as np
from tqdm import tqdm

import common

VOCAB_TILES = 60  # legality_error ranks indices 1..60, i.e. the 60 non-centre tiles

_UPSTREAM = None


def _init_worker(upstream):
    """Each worker needs the vendored upstream package on its own sys.path."""
    global _UPSTREAM
    _UPSTREAM = upstream
    import sys
    if upstream not in sys.path:
        sys.path.insert(0, upstream)


def game_logns(game):
    """ln(n_legal) at every training-target position of one game.

    Replays the game; the choice set the generator drew move k from is
    `get_valid_moves()` on the prefix m[0..k-1]. Move 0 is not a target, so we
    start recording at k=1. Returns (sum, sum_sq, n, Counter(n_legal)).
    """
    from data.othello import OthelloBoardState
    ab = OthelloBoardState()
    s = ss = 0.0
    n = 0
    hist = Counter()
    for k, mv in enumerate(game):
        valid = ab.get_valid_moves()
        if not valid:
            break
        if k >= 1:
            ln = math.log(len(valid))
            s += ln
            ss += ln * ln
            n += 1
            hist[len(valid)] += 1
        ab.update([mv])
    return s, ss, n, hist


def bench_case(case):
    """Perfect-model reference points for one intervention benchmark case.

    Mirrors intervene_nanda.main() exactly in how the counterfactual board is
    built (replay history, force one tile, recompute legal moves).
    """
    from data.othello import OthelloBoardState
    history = case["history"]
    pos_int = case["pos_int"]
    ori_color = case["ori_color"]
    intervention_to = 2 - ori_color

    ab = OthelloBoardState()
    ab.update(history)
    R = set(ab.get_valid_moves())          # legal on the REAL board

    cb = OthelloBoardState()
    cb.update(history)
    r, c = pos_int // 8, pos_int % 8
    cb.state[r, c] = intervention_to - 1
    C = set(cb.get_valid_moves())          # legal on the COUNTERFACTUAL board

    N = max(1, len(C))
    nR, nC = len(R), len(C)
    both = len(R & C)

    # --- floor: perfect knowledge of the counterfactual board ---------------
    # top-N is exactly C -> 0 illegal, unless C is empty (N forced to 1).
    informed = 0.0 if nC > 0 else 1.0

    # --- ceiling: perfect knowledge of the REAL board, edit unknown ---------
    # Ranks all of R first (indifferent within R), then the other tiles.
    a = min(N, nR)                          # slots filled from R
    b = N - a                               # slots filled from the complement
    comp = VOCAB_TILES - nR                 # size of the complement
    c_only = len(C - R)                     # legal-under-C tiles outside R

    exp_R = a * ((nR - both) / nR) if nR else 0.0
    exp_c = b * ((comp - c_only) / comp) if comp else 0.0
    expected = exp_R + exp_c

    best = max(0, a - both) + max(0, b - c_only)
    worst = min(a, nR - both) + min(b, comp - c_only)

    return {
        "informed": informed,
        "expected": expected,
        "best": float(best),
        "worst": float(worst),
        "n_real": nR,
        "n_cf": nC,
        "n_both": both,
        "n_symdiff": len(R ^ C),
        "N": N,
    }


def run_floor(games, upstream, ncpu, desc):
    s = ss = 0.0
    n = 0
    hist = Counter()
    with mp.Pool(ncpu, initializer=_init_worker, initargs=(upstream,)) as pool:
        for gs, gss, gn, gh in tqdm(pool.imap_unordered(game_logns, games, chunksize=64),
                                    total=len(games), desc=desc, mininterval=30):
            s += gs
            ss += gss
            n += gn
            hist.update(gh)
    mean = s / n
    var = max(0.0, ss / n - mean * mean)
    return {
        "mean_ln_n_legal": mean,
        "std_ln_n_legal": math.sqrt(var),
        "sem_pooled": math.sqrt(var / n),
        "n_positions": n,
        "n_games": len(games),
        "mean_n_legal": sum(k * v for k, v in hist.items()) / n,
        "n_legal_hist": {str(k): v for k, v in sorted(hist.items())},
    }


def sample_train_games(cfg, profile, n_games, n_shards, seed):
    """Uniform sample of games from the shards, excluding the val split.

    Deliberately does NOT call common.load_all_games -- that materialises all
    ~23M games to dedup them, which is many GB for no benefit here. Shards are
    i.i.d. by construction (gen_data seeds each game independently), so a spread
    of shards is already a fair sample. Cross-shard duplicates are left in; at
    23M games the dedup rate is far below the precision we need.
    """
    import glob
    import pickle
    paths = sorted(glob.glob(os.path.join(cfg["paths"]["data_dir"],
                                          f"{profile}_shard_*.pkl")))
    if not paths:
        raise FileNotFoundError(f"no shards for profile '{profile}'")
    rng = random.Random(seed)
    picked = rng.sample(paths, min(n_shards, len(paths)))
    pool = []
    for p in picked:
        with open(p, "rb") as f:
            pool.extend(pickle.load(f))
    val = set(tuple(g) for g in common.get_val_games(cfg, profile))
    pool = [g for g in pool if tuple(g) not in val]
    rng.shuffle(pool)
    return pool[:n_games], [os.path.basename(p) for p in picked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--n-games", type=int, default=50000,
                    help="train games sampled for the loss floor")
    ap.add_argument("--n-shards", type=int, default=6,
                    help="shards to draw that sample from")
    ap.add_argument("--val-games", type=int, default=10000,
                    help="val games for the skew comparison (0 to skip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="full_h200")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    if args.profile:
        cfg["profile"] = args.profile
    common.add_upstream_to_path(cfg)
    profile = cfg["profile"]
    upstream = cfg["paths"]["upstream"]
    ncpu = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
    t0 = time.time()
    print(f"[floor] profile={profile} cpus={ncpu} n_games={args.n_games}", flush=True)

    out = {"profile": profile, "tag": args.tag, "n_cpu": ncpu, "seed": args.seed}

    # --- 1. training-loss floor -------------------------------------------
    games, shards = sample_train_games(cfg, profile, args.n_games,
                                       args.n_shards, args.seed)
    print(f"[floor] sampled {len(games)} train games from {shards}", flush=True)
    out["loss_floor_train"] = run_floor(games, upstream, ncpu, "train floor")
    out["loss_floor_train"]["shards"] = shards
    f = out["loss_floor_train"]
    print(f"[floor] TRAIN loss floor = {f['mean_ln_n_legal']:.5f} nats "
          f"(+/- {f['sem_pooled']:.5f}) over {f['n_positions']} positions; "
          f"mean legal moves {f['mean_n_legal']:.3f}", flush=True)

    if args.val_games:
        vg = common.get_val_games(cfg, profile)[:args.val_games]
        out["loss_floor_val"] = run_floor(vg, upstream, ncpu, "val floor")
        v = out["loss_floor_val"]
        print(f"[floor] VAL loss floor  = {v['mean_ln_n_legal']:.5f} nats "
              f"(skew check only -- val is sorted(games)[:N], all games share "
              f"the lexicographically smallest opening)", flush=True)

    # --- 2. intervention legality-error floor / ceiling --------------------
    import pickle
    bench_path = os.path.join(upstream, "intervention_benchmark.pkl")
    with open(bench_path, "rb") as fh:
        bench = pickle.load(fh)
    n_cases = min(cfg["intervene"]["n_cases"], len(bench))
    print(f"[floor] intervention reference points over {n_cases} cases "
          f"(of {len(bench)} in the benchmark)", flush=True)
    _init_worker(upstream)
    rows = [bench_case(c) for c in tqdm(bench[:n_cases], desc="bench", mininterval=30)]
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg["n_cases"] = n_cases
    agg["n_cases_empty_cf"] = int(sum(1 for r in rows if r["n_cf"] == 0))
    agg["expected_std"] = float(np.std([r["expected"] for r in rows]))
    out["legality_floor"] = agg
    print(f"[floor] legality floor (perfect, edit-aware)      = {agg['informed']:.3f}",
          flush=True)
    print(f"[floor] legality ceiling (perfect, edit-unaware)  = {agg['expected']:.3f} "
          f"[best {agg['best']:.3f}, worst {agg['worst']:.3f}]", flush=True)
    print(f"[floor] mean legal moves: real {agg['n_real']:.2f}  counterfactual "
          f"{agg['n_cf']:.2f}  shared {agg['n_both']:.2f}  symdiff {agg['n_symdiff']:.2f}",
          flush=True)

    res_dir = cfg["paths"]["results_dir"]
    os.makedirs(res_dir, exist_ok=True)
    dest = os.path.join(res_dir, f"floors_{args.tag}.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[floor] wrote {dest} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
