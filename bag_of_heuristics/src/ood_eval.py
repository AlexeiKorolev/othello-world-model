"""Does Othello-GPT generalise, or did it fit a bag of position-specific rules?

Two suites, both scored with the paper's legal-move metric (greedy top-1 next
move, is it legal). Both are pure evaluation -- no probes, no training -- so the
same script re-scores any checkpoint and the JSON is directly comparable across
training methods.

SUITE 1 -- ORIENTATION (equivariance).
Othello's rules are invariant under the 8 symmetries of the square, but the
opening position is not: it is two diagonals, so a 90 degree rotation maps the
black diagonal onto the white one and swaps whose turn it is. Only the stabiliser
{identity, rot180, transpose, anti-transpose} maps legal games to legal games,
and the script verifies this empirically rather than assuming it (`legal_frac`
per transform -- expect 1.0 for exactly those four).

For each legality-preserving transform g we report:
  * err       -- legal-move error on g(game). A model with one shared rule scores
                 the same on all four; a model with per-square rules does not.
  * equivar   -- fraction of prefixes where pred(g(game)) == g(pred(game)).
                 This is the sharper number: it does not ask whether the model is
                 RIGHT, it asks whether it computes the SAME FUNCTION in a
                 different orientation. A genuine algorithm is exactly
                 equivariant (1.0) even where it errs; a bag of heuristics is not.
Note the training distribution is itself symmetric under these four maps, so this
is not a distribution shift -- any gap is per-square fitting, not extrapolation.

SUITE 2 -- POLICY SHIFT (out of distribution).
Training data is uniformly random legal play. These games are still perfectly
legal Othello but drawn from different policies, so they visit board positions
the model has effectively never seen:
  uniform      -- control, matches the training distribution exactly
  max_flip     -- always take the move flipping the most discs (greedy)
  min_flip     -- always take the move flipping the fewest
  corner_first -- corners > edges > interior
  edge_avoid   -- interior > edges > corners
Flip counts are read off the engine (discs before/after, minus the placed one)
so no board conventions are assumed. `mean_len` / `mean_legal` quantify how far
each policy actually moved the distribution.

    python ood_eval.py --config ../../configs/config.yaml --profile full \
        --arch gelu --n-games 300 --n-policy-games 300
"""
import os
import sys
import json
import time
import copy
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))
import common


# ---------------------------------------------------------------- symmetries
def _rc(t):
    return t // 8, t % 8


def _t(r, c):
    return r * 8 + c


TRANSFORMS = {
    "identity":  lambda r, c: (r, c),
    "rot90":     lambda r, c: (c, 7 - r),
    "rot180":    lambda r, c: (7 - r, 7 - c),
    "rot270":    lambda r, c: (7 - c, r),
    "flip_h":    lambda r, c: (r, 7 - c),
    "flip_v":    lambda r, c: (7 - r, c),
    "transpose": lambda r, c: (c, r),
    "antitrans": lambda r, c: (7 - c, 7 - r),
}


def perm_of(name):
    """Tile permutation array: perm[t] is where tile t lands."""
    f = TRANSFORMS[name]
    return np.array([_t(*f(*_rc(t))) for t in range(64)], dtype=np.int64)


def apply_perm(game, perm):
    return [int(perm[m]) for m in game]


# ---------------------------------------------------------------- legality
def strict_legal_frac(games):
    """Fraction of games that are legal under STRICT turn order.

    Do NOT use `OthelloBoardState.update()` for this. Its `umpire` (upstream
    othello.py:230) treats a move that flips nothing for the side to move as a
    forfeited hand and silently flips the colour, so it accepts colour-swapped
    sequences that no standard game can produce. We check membership in
    get_valid_moves() before each move instead, which is the real constraint.
    """
    from data.othello import OthelloBoardState
    ok = 0
    for g in games:
        ab = OthelloBoardState()
        good = True
        for m in g:
            if m not in set(ab.get_valid_moves()):
                good = False
                break
            ab.update([m])
        ok += good
    return ok / max(1, len(games))


# ---------------------------------------------------------------- eval metric
@torch.no_grad()
def predict_probs(model, games, device):
    """One forward pass per game -> next-move probabilities at every prefix.
    Returns a list of [T, vocab] float32 arrays."""
    model.eval()
    out = []
    for game in tqdm(games, desc="forward", mininterval=30):
        if len(game) < 2:
            out.append(np.zeros((0, common.VOCAB_SIZE), dtype=np.float32))
            continue
        x = torch.tensor([common.STOI[m] for m in game[:-1]],
                         dtype=torch.long, device=device)[None, :]
        logits, _ = model(x)
        out.append(torch.softmax(logits[0].float(), dim=-1).cpu().numpy())
    return out


def legal_move_error(probs, games, per_square=False):
    """Paper metric: greedy top-1 next move at every prefix, is it legal.

    argmax over logits == mingpt's sample(sample=False), which is what
    train_gpt.py uses, so this matches the headline number. Legality is checked
    strictly against get_valid_moves() for the side to move (see
    strict_legal_frac for why update() is not trustworthy here). Pad-token
    predictions count as illegal.
    """
    from data.othello import OthelloBoardState
    total = ok = pad = 0
    sq_tot = np.zeros(64, dtype=np.int64)
    sq_bad = np.zeros(64, dtype=np.int64)
    all_preds = []

    for probs_g, game in zip(probs, games):
        if len(game) < 2:
            all_preds.append([])
            continue
        preds = [common.ITOS[int(i)] for i in probs_g.argmax(axis=-1)]
        all_preds.append(preds)

        ab = OthelloBoardState()
        for L in range(1, len(game)):
            ab.update([game[L - 1]])
            valid = set(ab.get_valid_moves())
            p = preds[L - 1]
            total += 1
            if p == -100:
                pad += 1
            elif p in valid:
                ok += 1
            if per_square and p != -100:
                # attribute the error to the square the model WANTED to play
                sq_tot[p] += 1
                if p not in valid:
                    sq_bad[p] += 1

    out = {"err": 1.0 - ok / max(1, total), "n_positions": total,
           "pad_preds": pad}
    if per_square:
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(sq_tot > 0, sq_bad / np.maximum(sq_tot, 1), np.nan)
        out["per_square_err"] = [None if np.isnan(v) else float(v) for v in rate]
        out["per_square_n"] = sq_tot.tolist()
    return out, all_preds


def equivariance(base_probs, tr_probs, perm):
    """How close is pred(g(x)) to g(pred(x))?

    Compared on DISTRIBUTIONS, not argmax. The training target is ~uniform over
    the legal moves, so which legal move wins the argmax is close to arbitrary
    and its agreement across orientations would mostly measure tie-breaking
    noise. The distribution is well determined, so total-variation distance is
    the meaningful quantity: 0 = exactly equivariant, 1 = disjoint support.
    argmax agreement is kept as a secondary, noisier readout.
    """
    # permutation on vocab indices induced by the tile permutation
    vperm = np.arange(common.VOCAB_SIZE)
    for t in range(64):
        if t in common.STOI and int(perm[t]) in common.STOI:
            vperm[common.STOI[t]] = common.STOI[int(perm[t])]

    tvs, agree, tot = [], 0, 0
    for bp, tp in zip(base_probs, tr_probs):
        n = min(len(bp), len(tp))
        if n == 0:
            continue
        moved = bp[:n][:, np.argsort(vperm)]      # g applied to the base dist
        tvs.append(0.5 * np.abs(moved - tp[:n]).sum(axis=-1))
        agree += int((moved.argmax(-1) == tp[:n].argmax(-1)).sum())
        tot += n
    tv = float(np.concatenate(tvs).mean()) if tvs else float("nan")
    return {"tv": tv, "argmax_agree": agree / max(1, tot), "n_compared": tot}


# ---------------------------------------------------------------- policies
def _flips(ab, move):
    """Discs flipped by `move`, convention-free: total discs gain 1 + flips."""
    before = int(np.count_nonzero(ab.state))
    nb = copy.deepcopy(ab)
    nb.update([move])
    return int(np.count_nonzero(nb.state)) - before - 1


CORNERS = {0, 7, 56, 63}


def _ring(t):
    r, c = _rc(t)
    if t in CORNERS:
        return 2
    if r in (0, 7) or c in (0, 7):
        return 1
    return 0


def pick(policy, ab, valid, rng):
    if policy == "uniform":
        return rng.choice(valid)
    if policy in ("max_flip", "min_flip"):
        f = [_flips(ab, m) for m in valid]
        best = (max if policy == "max_flip" else min)(f)
        return rng.choice([m for m, v in zip(valid, f) if v == best])
    if policy in ("corner_first", "edge_avoid"):
        k = [_ring(m) for m in valid]
        best = (max if policy == "corner_first" else min)(k)
        return rng.choice([m for m, v in zip(valid, k) if v == best])
    raise ValueError(policy)


def gen_games(policy, n, seed):
    """Legal games played to a leaf under `policy`. Mirrors get_ood_game's loop
    (play until the side to move has no legal move) so only the policy differs."""
    from data.othello import OthelloBoardState
    rng = random.Random(seed)
    out = []
    for _ in tqdm(range(n), desc=f"gen {policy}", mininterval=30):
        ab = OthelloBoardState()
        g = []
        valid = ab.get_valid_moves()
        while valid:
            m = pick(policy, ab, valid, rng)
            g.append(m)
            ab.update([m])
            valid = ab.get_valid_moves()
        out.append(g)
    return out


def dist_stats(games):
    """How far did the policy actually move the distribution?"""
    from data.othello import OthelloBoardState
    lens, legals = [], []
    for g in games[:200]:
        lens.append(len(g))
        ab = OthelloBoardState()
        for m in g:
            legals.append(len(ab.get_valid_moves()))
            ab.update([m])
    return {"mean_len": float(np.mean(lens)),
            "mean_legal": float(np.mean(legals))}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../../configs/config.yaml")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--arch", default="gelu", choices=["gelu", "bilinear"])
    ap.add_argument("--expansion", type=float, default=8.0 / 3.0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n-games", type=int, default=300,
                    help="val games for the orientation suite")
    ap.add_argument("--n-policy-games", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    if args.profile:
        cfg["profile"] = args.profile
    common.add_upstream_to_path(cfg)
    profile = cfg["profile"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from mingpt.model import GPT, GPTConfig
    mconf = GPTConfig(common.VOCAB_SIZE, common.BLOCK_SIZE,
                      n_layer=cfg["model"]["n_layer"], n_head=cfg["model"]["n_head"],
                      n_embd=cfg["model"]["n_embd"])
    suffix = "" if args.arch == "gelu" else f"_{args.arch}"
    if args.arch == "bilinear":
        from bilinear_model import make_bilinear_gpt
        model = make_bilinear_gpt(mconf, expansion=args.expansion)
    else:
        model = GPT(mconf)
    # checkpoint filenames carry a run tag (the full run is *_full_h200), so try
    # the plain name and fall back to the tagged variants actually on disk
    if args.ckpt:
        ckpt = args.ckpt
    else:
        cdir = cfg["paths"]["ckpt_dir"]
        cands = [f"gpt_synthetic_{profile}{suffix}.ckpt",
                 f"gpt_synthetic_{profile}_h200{suffix}.ckpt"]
        for cand in cands:
            ckpt = os.path.join(cdir, cand)
            if os.path.exists(ckpt):
                break
        else:
            raise SystemExit(f"[ood] no checkpoint found; tried {cands} in {cdir}")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device)
    tag = args.tag or (profile + suffix)
    print(f"[ood] loaded {ckpt} on {device}", flush=True)

    res = {"tag": tag, "arch": args.arch, "profile": profile, "ckpt": ckpt,
           "n_games": args.n_games, "n_policy_games": args.n_policy_games,
           "seed": args.seed}

    # ---- suite 1: orientation -------------------------------------------
    val = common.get_val_games(cfg, profile)[:args.n_games]
    from data.othello import OthelloBoardState

    print("\n[ood] === suite 1: orientation ===", flush=True)
    base_probs = predict_probs(model, val, device)
    base, _ = legal_move_error(base_probs, val, per_square=True)
    print(f"[ood] identity err = {base['err']*100:.4f}%  "
          f"({base['n_positions']} positions)", flush=True)

    orient = {}
    for name in TRANSFORMS:
        perm = perm_of(name)
        tg = [apply_perm(g, perm) for g in val]
        # STRICT legality -- update() would accept colour-swapped games (see
        # strict_legal_frac). Expect 1.0 for the stabiliser of the opening
        # {identity, rot180, transpose, antitrans} and ~0 for the other four.
        legal_frac = strict_legal_frac(tg)
        tp = predict_probs(model, tg, device)
        m, _ = legal_move_error(tp, tg, per_square=True)
        eq = equivariance(base_probs, tp, perm)
        entry = {"legal_frac": legal_frac, **m, **eq}
        orient[name] = entry
        kind = "in-dist" if legal_frac > 0.999 else "COLOUR-SWAPPED (ood)"
        print(f"[ood] {name:10s} legal={legal_frac:.3f}  err={m['err']*100:7.4f}%  "
              f"tv={eq['tv']:.4f}  argmax_agree={eq['argmax_agree']*100:6.2f}%  "
              f"{kind}", flush=True)
    res["orientation"] = orient

    # ---- suite 2: policy shift ------------------------------------------
    print("\n[ood] === suite 2: policy shift ===", flush=True)
    pol = {}
    for i, policy in enumerate(["uniform", "max_flip", "min_flip",
                                "corner_first", "edge_avoid"]):
        t0 = time.time()
        games = gen_games(policy, args.n_policy_games, args.seed + 1000 * i)
        stats = dist_stats(games)
        m, _ = legal_move_error(predict_probs(model, games, device), games,
                                per_square=True)
        m.update(stats)
        m["gen_seconds"] = round(time.time() - t0, 1)
        pol[policy] = m
        print(f"[ood] {policy:13s} err={m['err']*100:.4f}%  "
              f"mean_len={stats['mean_len']:.1f}  "
              f"mean_legal={stats['mean_legal']:.2f}", flush=True)
    res["policy"] = pol

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "results", f"ood_eval_{tag}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n[ood] wrote {os.path.abspath(out)}", flush=True)


if __name__ == "__main__":
    main()
