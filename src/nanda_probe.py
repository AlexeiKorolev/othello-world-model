"""Nanda's linear "mine/yours" board probe (follow-up to Li et al. 2023).

Li et al. reported that a *linear* probe decodes the board poorly (~20-27%) in
the absolute white/blank/black encoding, and concluded the world model is
nonlinear. Nanda (2023, "Actually, Othello-GPT Has A Linear Emergent World
Representation") showed the representation is in fact *linear* once you relabel
each tile relative to the player about to move: mine / yours / empty. Because
the side-to-move alternates every ply, a SINGLE linear probe trained on both
black-to-move and white-to-move positions with the mine/yours labels decodes the
board almost perfectly (~1-2%), on par with the nonlinear absolute probe.

Mechanics (this codebase's conventions, verified):
  * get_state() returns state+1 -> white=0, blank=1, black=2 (absolute).
  * black moves first; at sequence position t (0-indexed move index) the board
    reflects moves 0..t and the side to move NEXT is black iff t is odd.
  * mine/yours = absolute with the two non-empty classes swapped (v -> 2-v) on
    the positions where the side-to-move is white (even t). Which parity we flip
    is arbitrary -- the other choice is a global mine<->yours relabel that a
    linear probe fits identically; what matters is that the flip is player-
    relative (parity-based), not absolute.

For each layer we train, on the SAME activations:
  * one linear probe on mine/yours labels  (Nanda's result -> should be low),
  * one linear probe on absolute labels     (Li et al. baseline -> ~20%),
so the figure shows the encoding change alone flips the linear decodability.

    python nanda_probe.py --config ../configs/config.yaml --profile full
    python nanda_probe.py --config ... --ckpt <path> --tag pretrained
"""
import os
import json
import time
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import common
from train_probe import all_layer_acts, train_one_probe


@torch.no_grad()
def extract_myours(model, games, n_games, layers, device):
    """Per-position activations at each layer plus BOTH label encodings:
    absolute (white0/blank1/black2) and mine/yours (yours0/empty1/mine2)."""
    from data.othello import OthelloBoardState
    model.eval()
    acts = {L: [] for L in layers}
    ys_abs, ys_my = [], []
    for game in tqdm(games[:n_games], desc="extract acts", mininterval=30):
        moves = game[:common.BLOCK_SIZE]
        if len(moves) < 2:
            continue
        toks = torch.tensor([common.STOI[m] for m in moves],
                            dtype=torch.long, device=device)[None]
        states = OthelloBoardState().get_gt(moves, "get_state")   # list[T] of len-64
        states = torch.tensor(states, dtype=torch.long)           # [T, 64] absolute
        T = states.size(0)
        # side-to-move-next is white on even t -> flip (2 - v) there so that the
        # mover's colour becomes "mine" (=2) at every position.
        my = states.clone()
        even = (torch.arange(T) % 2 == 0)
        my[even] = 2 - my[even]
        layer_acts = all_layer_acts(model, toks)                  # after each block
        for L in layers:
            acts[L].append(layer_acts[L - 1].cpu())               # [T, f]
        ys_abs.append(states)
        ys_my.append(my)
    Y_abs = torch.cat(ys_abs, dim=0)                              # [N, 64]
    Y_my = torch.cat(ys_my, dim=0)                                # [N, 64]
    A = {L: torch.cat(acts[L], dim=0) for L in layers}            # [N, f]
    return A, Y_abs, Y_my


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--ckpt", default=None, help="GPT checkpoint to probe")
    ap.add_argument("--tag", default=None, help="label for output files")
    ap.add_argument("--arch", default="gelu", choices=["gelu", "bilinear"],
                    help="MLP type of the checkpoint being probed")
    ap.add_argument("--expansion", type=float, default=2.6666667,
                    help="bilinear d_hidden/d_model (must match the trained ckpt)")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    if args.profile:
        cfg["profile"] = args.profile
    common.add_upstream_to_path(cfg)
    profile = cfg["profile"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_embd = cfg["model"]["n_embd"]

    from mingpt.model import GPT, GPTConfig
    from mingpt.probe_model import BatteryProbeClassification

    mconf = GPTConfig(common.VOCAB_SIZE, common.BLOCK_SIZE,
                      n_layer=cfg["model"]["n_layer"], n_head=cfg["model"]["n_head"],
                      n_embd=n_embd)
    suffix = "" if args.arch == "gelu" else f"_{args.arch}"
    if args.arch == "bilinear":
        from bilinear_model import make_bilinear_gpt
        model = make_bilinear_gpt(mconf, expansion=args.expansion)
    else:
        model = GPT(mconf)
    tag = args.tag or (profile + suffix)
    ckpt = args.ckpt or os.path.join(cfg["paths"]["ckpt_dir"],
                                     f"gpt_synthetic_{profile}{suffix}.ckpt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device)
    print(f"[nanda] loaded {ckpt}", flush=True)

    val_games = common.get_val_games(cfg, profile)
    layers = cfg["probe"]["layers"]
    epochs = cfg["probe"]["epochs"]
    bs = cfg["probe"]["batch_size"]

    t0 = time.time()
    A, Y_abs, Y_my = extract_myours(model, val_games,
                                    cfg["probe"]["max_probe_games"], layers, device)
    print(f"[nanda] extracted {Y_my.shape[0]} positions x 64 tiles "
          f"in {time.time()-t0:.0f}s", flush=True)

    results = {"tag": tag, "profile": profile,
               "linear_mine_yours": {}, "linear_absolute": {}}
    for L in layers:
        p_my = BatteryProbeClassification(device, probe_class=3, num_task=64,
                                          input_dim=n_embd)
        err_my = train_one_probe(A[L], Y_my, p_my, epochs, bs, device)
        p_ab = BatteryProbeClassification(device, probe_class=3, num_task=64,
                                          input_dim=n_embd)
        err_ab = train_one_probe(A[L], Y_abs, p_ab, epochs, bs, device)
        results["linear_mine_yours"][L] = err_my
        results["linear_absolute"][L] = err_ab
        print(f"[nanda] layer {L}: linear mine/yours {err_my*100:.2f}%  "
              f"linear absolute {err_ab*100:.2f}%", flush=True)

    res_dir = cfg["paths"]["results_dir"]
    os.makedirs(res_dir, exist_ok=True)
    out = os.path.join(res_dir, f"nanda_probe_{tag}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    best_L = min(results["linear_mine_yours"], key=results["linear_mine_yours"].get)
    print(f"[nanda] best linear mine/yours: layer {best_L} "
          f"err {results['linear_mine_yours'][best_L]*100:.2f}%  -> {out}", flush=True)

    # figure: mine/yours vs absolute linear error across layers
    xs = layers
    plt.figure(figsize=(7, 4.5))
    plt.plot(xs, [results["linear_absolute"][L] * 100 for L in xs],
             "o-", color="#c44", label="linear, absolute (Li et al.)")
    plt.plot(xs, [results["linear_mine_yours"][L] * 100 for L in xs],
             "o-", color="#268", label="linear, mine/yours (Nanda)")
    plt.xlabel("layer"); plt.ylabel("board-reconstruction error (%)")
    plt.title(f"Linear probe: absolute vs mine/yours encoding ({tag})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    fig = os.path.join(res_dir, f"nanda_probe_{tag}.png")
    plt.savefig(fig, dpi=150)
    print(f"[nanda] wrote {fig}", flush=True)


if __name__ == "__main__":
    main()
