"""Probe Othello-GPT's internal board representation.

Reproduces the paper's central probing result: linear probes decode the board
poorly (~20-27% error) while a 2-layer MLP (nonlinear) probe decodes it almost
perfectly (~1.8% error, best around layer 6). For each transformer layer we
train BOTH a linear and a nonlinear probe (the original
`BatteryProbeClassification[TwoLayer]` heads) to predict the 3-class state
(white/blank/black) of all 64 tiles, and report test error vs layer.

Deviation from upstream (documented): the original `train_probe_othello.py`
extracts probe activations over *championship* games. Since this round is
synthetic-only, we extract over held-out *synthetic* validation games. The
representational question is unchanged.

    python train_probe.py --config ../configs/config.yaml --profile pilot
    python train_probe.py --config ... --ckpt <path> --tag pretrained
    python train_probe.py --config ... --random   # random-init control
"""
import os
import json
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm

import common


@torch.no_grad()
def all_layer_acts(model, x):
    """Return activations after each transformer block: list of [T, n_embd]."""
    b, t = x.size()
    tok = model.tok_emb(x)
    pos = model.pos_emb[:, :t, :]
    h = tok + pos                      # drop is identity in eval()
    acts = []
    for blk in model.blocks:
        h = blk(h)
        acts.append(h[0].detach())     # [T, n_embd]
    return acts


def extract(model, games, n_games, layers, device):
    """Run the GPT over games, collecting per-position activations at each
    requested layer plus the ground-truth board state (0/1/2 per tile)."""
    from data.othello import OthelloBoardState
    model.eval()
    acts = {L: [] for L in layers}
    ys = []
    for game in tqdm(games[:n_games], desc="extract acts", mininterval=30):
        moves = game[:common.BLOCK_SIZE]
        if len(moves) < 2:
            continue
        toks = torch.tensor([common.STOI[m] for m in moves],
                            dtype=torch.long, device=device)[None]
        states = OthelloBoardState().get_gt(moves, "get_state")  # len==len(moves)
        layer_acts = all_layer_acts(model, toks)                 # after each block
        for L in layers:
            acts[L].append(layer_acts[L - 1].cpu())              # [T, f]
        ys.append(torch.tensor(states, dtype=torch.long))        # [T, 64]
    Y = torch.cat(ys, dim=0)                                      # [N, 64]
    A = {L: torch.cat(acts[L], dim=0) for L in layers}           # [N, f]
    return A, Y


def train_one_probe(act, y, probe, epochs, batch_size, device, lr=1e-3):
    """Train a single probe head; return best test error (1 - tile accuracy)."""
    ds = TensorDataset(act, y)
    n_test = max(1, int(0.2 * len(ds)))
    n_train = len(ds) - n_test
    g = torch.Generator().manual_seed(0)
    train_ds, test_ds = random_split(ds, [n_train, n_test], generator=g)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Adam + cosine annealing. NB: the upstream ReduceLROnPlateau(patience=0)
    # collapses the LR after a few epochs and starves the nonlinear probe
    # (~16% error); cosine over many epochs reaches the paper's ~1.8%.
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_err = 1.0
    for ep in range(epochs):
        probe.train()
        for a, yy in train_ld:
            a, yy = a.to(device), yy.to(device)
            _, loss = probe(a, yy)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        # eval
        probe.eval()
        hits, total = 0, 0
        with torch.no_grad():
            for a, yy in test_ld:
                a, yy = a.to(device), yy.to(device)
                logits, _ = probe(a, yy)
                hits += (logits.argmax(dim=-1) == yy).sum().item()
                total += yy.numel()
        best_err = min(best_err, 1.0 - hits / total)
    return best_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--ckpt", default=None, help="GPT checkpoint to probe")
    ap.add_argument("--tag", default=None, help="label for output files")
    ap.add_argument("--random", action="store_true",
                    help="probe a random-init GPT (control)")
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

    from mingpt.model import GPTforProbing, GPTConfig  # noqa
    from mingpt.model import GPT
    from mingpt.probe_model import (BatteryProbeClassification,
                                    BatteryProbeClassificationTwoLayer)

    mconf = GPTConfig(common.VOCAB_SIZE, common.BLOCK_SIZE,
                      n_layer=cfg["model"]["n_layer"], n_head=cfg["model"]["n_head"],
                      n_embd=cfg["model"]["n_embd"])
    suffix = "" if args.arch == "gelu" else f"_{args.arch}"
    if args.arch == "bilinear":
        from bilinear_model import make_bilinear_gpt
        model = make_bilinear_gpt(mconf, expansion=args.expansion)
    else:
        model = GPT(mconf)
    default_tag = ("random_init" if args.random else profile) + suffix
    tag = args.tag or default_tag
    if args.random:
        model.apply(model._init_weights)
        print("[probe] using RANDOM-INIT GPT (control)", flush=True)
    else:
        ckpt = args.ckpt or os.path.join(cfg["paths"]["ckpt_dir"],
                                         f"gpt_synthetic_{profile}{suffix}.ckpt")
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"[probe] loaded {ckpt}", flush=True)
    model = model.to(device)

    val_games = common.get_val_games(cfg, profile)
    layers = cfg["probe"]["layers"]
    t0 = time.time()
    A, Y = extract(model, val_games, cfg["probe"]["max_probe_games"], layers, device)
    print(f"[probe] extracted {Y.shape[0]} positions x {Y.shape[1]} tiles "
          f"in {time.time()-t0:.0f}s", flush=True)

    results = {"tag": tag, "profile": profile, "linear": {}, "nonlinear": {}}
    for L in layers:
        lin = BatteryProbeClassification(device, probe_class=3, num_task=64,
                                         input_dim=cfg["model"]["n_embd"])
        err_lin = train_one_probe(A[L], Y, lin, cfg["probe"]["epochs"],
                                  cfg["probe"]["batch_size"], device)
        nl = BatteryProbeClassificationTwoLayer(device, probe_class=3, num_task=64,
                                                mid_dim=cfg["probe"]["mid_dim"],
                                                input_dim=cfg["model"]["n_embd"])
        err_nl = train_one_probe(A[L], Y, nl, cfg["probe"]["epochs"],
                                 cfg["probe"]["batch_size"], device)
        results["linear"][L] = err_lin
        results["nonlinear"][L] = err_nl
        print(f"[probe] layer {L}: linear err {err_lin*100:.2f}%  "
              f"nonlinear err {err_nl*100:.2f}%", flush=True)

    res_dir = cfg["paths"]["results_dir"]
    os.makedirs(res_dir, exist_ok=True)
    out = os.path.join(res_dir, f"probe_{tag}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    best_nl_layer = min(results["nonlinear"], key=results["nonlinear"].get)
    print(f"[probe] best nonlinear: layer {best_nl_layer} "
          f"err {results['nonlinear'][best_nl_layer]*100:.2f}%  -> {out}", flush=True)


if __name__ == "__main__":
    main()
