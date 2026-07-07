"""Diagnostic: why does the nonlinear probe plateau ~16% on the authors' model
when the paper reports ~1.8%? Sweep probe training hyperparameters on the
ANCHOR (authors' verified weights) at a fixed layer and report test error.
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

import common
from train_probe import extract
from validate_pretrained import convert_tl_to_mingpt


def train_probe(act, y, probe, epochs, batch_size, device, lr, schedule):
    ds = TensorDataset(act, y)
    n_test = int(0.2 * len(ds)); n_train = len(ds) - n_test
    tr, te = random_split(ds, [n_train, n_test], generator=torch.Generator().manual_seed(0))
    trl = DataLoader(tr, batch_size=batch_size, shuffle=True)
    tel = DataLoader(te, batch_size=batch_size, shuffle=False)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    if schedule == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.75, patience=0)
    elif schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        sched = None
    best = 1.0
    for ep in range(epochs):
        probe.train()
        for a, yy in trl:
            a, yy = a.to(device), yy.to(device)
            _, loss = probe(a, yy)
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval(); hits=tot=0; ls=[]
        with torch.no_grad():
            for a, yy in tel:
                a, yy = a.to(device), yy.to(device)
                lo, l = probe(a, yy); ls.append(l.item())
                hits += (lo.argmax(-1)==yy).sum().item(); tot += yy.numel()
        err = 1 - hits/tot; best = min(best, err)
        if schedule=="plateau": sched.step(np.mean(ls))
        elif schedule=="cosine": sched.step()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--n_games", type=int, default=3000)
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    common.add_upstream_to_path(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from mingpt.model import GPT, GPTConfig
    from mingpt.probe_model import (BatteryProbeClassification,
                                    BatteryProbeClassificationTwoLayer)
    d = cfg["model"]["n_embd"]
    pth = os.path.join(cfg["paths"]["ckpt_dir"], "pretrained", "synthetic_model.pth")
    sd = torch.load(pth, map_location="cpu", weights_only=True)
    body, _ = convert_tl_to_mingpt(sd, cfg["model"]["n_layer"], d)
    model = GPT(GPTConfig(common.VOCAB_SIZE, common.BLOCK_SIZE, n_layer=8, n_head=8, n_embd=d))
    model.load_state_dict(body, strict=False); model = model.to(device).eval()

    games = common.load_all_games(cfg["paths"]["data_dir"], cfg["profile"])
    _, val = common.split_train_val(games, cfg["data"]["val_games"])
    A, Y = extract(model, val, args.n_games, [args.layer], device)
    act = A[args.layer]
    print(f"[diag] layer {args.layer}: {act.shape[0]} samples", flush=True)

    print(f"\n{'probe':10s}{'epochs':>7}{'mid':>5}{'lr':>8}{'sched':>9}{'err%':>9}")
    configs = [
        ("linear",   16,  0,  1e-3, "plateau"),
        ("nonlin",   16, 64, 1e-3, "plateau"),   # current (buggy) setup
        ("nonlin",   60, 64, 1e-3, "none"),
        ("nonlin",  150, 128, 1e-3, "cosine"),
        ("nonlin",  150, 256, 1e-3, "cosine"),
        ("nonlin",  300, 256, 1e-3, "cosine"),
    ]
    for kind, ep, mid, lr, sch in configs:
        if kind == "linear":
            p = BatteryProbeClassification(device, 3, 64, input_dim=d)
        else:
            p = BatteryProbeClassificationTwoLayer(device, 3, 64, mid_dim=mid, input_dim=d)
        err = train_probe(act, Y, p, ep, cfg["probe"]["batch_size"], device, lr, sch)
        print(f"{kind:10s}{ep:>7}{mid:>5}{lr:>8.0e}{sch:>9}{err*100:>8.2f}%", flush=True)


if __name__ == "__main__":
    main()
