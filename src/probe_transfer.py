"""Does the linear world model ROTATE during training, or just sharpen in place?

Trains one mine/yours probe per dev-study checkpoint (epochs 0,2,...,20) and
cross-applies every probe to every checkpoint's activations, producing an
epoch x epoch transfer matrix. Because all checkpoints are read over the SAME
games in the SAME order, row k of every activation tensor is the same board
state, so probes and splits are directly comparable across epochs.

Three measurements, because transfer error alone cannot distinguish the cases:

  1. TRANSFER ERROR, under three output corrections.
       raw      -- apply probe i's logits to epoch j's activations unchanged.
       global   -- fit a scalar gain + per-class offset (4 params).
       pertile  -- fit a per-tile gain + per-tile-class offset (256 params).
     The bias must be per-class: adding a constant to all 3 class logits is a
     softmax no-op, so a single scalar bias would fit nothing. The gap between
     `raw` and `global` is exactly the part of the drift that is mere rescaling
     (the residual stream grows during training), and what survives `global` is
     the part that is not a scale artefact.

  2. DIRECTION COSINE. For each tile the identified quantity is the mine-minus-
     yours direction w_mine - w_yours in R^512 -- NOT the raw class weights,
     since softmax is invariant to adding a common vector to all three classes,
     so the individual class weights are not identified but their differences
     are. Cosine between epoch i's and epoch j's direction, averaged over the 64
     tiles, measures rotation directly and is immune to any rescaling.

  3. SUBSPACE OVERLAP. Principal angles between the 64-dimensional subspaces
     spanned by those directions. Individual tile directions can drift while the
     subspace they span is preserved (an internal change of coordinates); this
     separates "the board lives somewhere else now" from "the board lives in the
     same place, indexed differently".

SEED CONTROL (essential -- without it none of the above is interpretable):
probes are over-parameterised (512 inputs for a 3-way call), so two probes
trained on the SAME checkpoint with different seeds already disagree. That
disagreement is the noise floor for every off-diagonal number here. We train
extra probes at two epochs with different seeds and report their mutual cosine
and transfer error, so "cosine 0.85 between epoch 4 and 20" can be read against
"cosine 0.97 between two probes on epoch 20 itself".

Epoch 0 is a random-init model and is expected to be an outlier throughout; it
is kept as a lower reference, not treated as a real point on the trajectory.

    python probe_transfer.py --config ../configs/config.yaml --profile full
"""
import os
import json
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common
from nanda_probe import extract_myours

EPOCHS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
MODES = ["raw", "global", "pertile"]


def make_split(n, frac_test=0.2, seed=0):
    """One split reused for every epoch, so all matrix entries are scored on the
    identical set of board positions."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_test = max(1, int(frac_test * n))
    return perm[n_test:], perm[:n_test]


def train_probe(act, y, idx, epochs, bs, device, n_embd, seed):
    from mingpt.probe_model import BatteryProbeClassification
    torch.manual_seed(seed)
    probe = BatteryProbeClassification(device, probe_class=3, num_task=64,
                                       input_dim=n_embd)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = idx.numel()
    for _ in range(epochs):
        probe.train()
        order = idx[torch.randperm(n)]
        for s in range(0, n, bs):
            sel = order[s:s + bs]
            a = act[sel].to(device, non_blocking=True)
            yy = y[sel].to(device, non_blocking=True)
            _, loss = probe(a, yy)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    probe.eval()
    return probe


@torch.no_grad()
def probe_logits(probe, act, idx, device, bs=8192):
    out = []
    for s in range(0, idx.numel(), bs):
        sel = idx[s:s + bs]
        out.append(probe(act[sel].to(device))[0].detach())
    return torch.cat(out, 0)                      # [n, 64, 3]


def err_of(logits, y):
    return float(1.0 - (logits.argmax(-1).cpu() == y).float().mean().item())


def fit_affine(logits, y, mode, device, steps=300, lr=0.05, max_fit=20000):
    """Fit the output correction on the TARGET epoch's train split.

    Only 4 or 256 parameters, so a subsample is plenty and keeps 121 pairs cheap.
    """
    if mode == "raw":
        return None
    n = min(max_fit, logits.shape[0])
    z = logits[:n].to(device)
    t = y[:n].to(device)
    if mode == "global":
        a = torch.ones(1, device=device, requires_grad=True)
        b = torch.zeros(3, device=device, requires_grad=True)
    else:
        a = torch.ones(64, 1, device=device, requires_grad=True)
        b = torch.zeros(64, 3, device=device, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=lr)
    for _ in range(steps):
        loss = F.cross_entropy((a * z + b).reshape(-1, 3), t.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return a.detach(), b.detach()


def apply_affine(logits, ab):
    if ab is None:
        return logits
    a, b = ab
    return a * logits + b


def directions(probe):
    """Per-tile mine-minus-yours direction: the softmax-identified quantity.

    proj.weight is [3*64, 512] and forward reshapes to [-1, 64, 3], so rows are
    ordered tile-major. Classes are yours=0, empty=1, mine=2.
    """
    W = probe.proj.weight.detach().reshape(64, 3, -1)
    return (W[:, 2, :] - W[:, 0, :]).float()      # [64, 512]


def mean_cosine(D1, D2):
    return float(F.cosine_similarity(D1, D2, dim=1).mean().item())


def subspace_overlap(D1, D2):
    """Cosines of the principal angles between the two 64-dim row spaces."""
    Q1, _ = torch.linalg.qr(D1.T.double())
    Q2, _ = torch.linalg.qr(D2.T.double())
    s = torch.linalg.svdvals(Q1.T @ Q2).clamp(-1, 1)
    return float(s.mean().item()), float(s.min().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default="full")
    ap.add_argument("--ckpt-dir", default=None,
                    help="dir holding gpt_epoch###.ckpt (default: <ckpt_dir>/dev_study_full_h200)")
    ap.add_argument("--layers", default="4,7", type=str)
    ap.add_argument("--probe-epochs", type=int, default=150)
    ap.add_argument("--max-games", type=int, default=2000)
    ap.add_argument("--control-epochs", default="4,20",
                    help="epochs to retrain with extra seeds for the noise floor")
    ap.add_argument("--control-seeds", default="1,2")
    ap.add_argument("--tag", default="full_h200")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    cfg["profile"] = args.profile
    common.add_upstream_to_path(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_embd = cfg["model"]["n_embd"]
    bs = cfg["probe"]["batch_size"]
    layers = [int(x) for x in args.layers.split(",")]
    ck_dir = args.ckpt_dir or os.path.join(cfg["paths"]["ckpt_dir"], "dev_study_full_h200")
    ctrl_eps = [int(x) for x in args.control_epochs.split(",") if x]
    ctrl_seeds = [int(x) for x in args.control_seeds.split(",") if x]
    t0 = time.time()

    from mingpt.model import GPT, GPTConfig
    mconf = GPTConfig(common.VOCAB_SIZE, common.BLOCK_SIZE,
                      n_layer=cfg["model"]["n_layer"], n_head=cfg["model"]["n_head"],
                      n_embd=n_embd)
    val_games = common.get_val_games(cfg, args.profile)

    print(f"[transfer] layers={layers} epochs={EPOCHS} games={args.max_games} "
          f"probe_epochs={args.probe_epochs} device={device}", flush=True)

    # --- extract activations for every checkpoint over the SAME games ---------
    acts, Ymy, norms = {}, None, {}
    for ep in EPOCHS:
        path = os.path.join(ck_dir, f"gpt_epoch{ep:03d}.ckpt")
        model = GPT(mconf)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model = model.to(device)
        A, _, Y = extract_myours(model, val_games, args.max_games, layers, device)
        if Ymy is None:
            Ymy = Y
        else:
            assert torch.equal(Ymy, Y), "label tensors diverged across checkpoints"
        for L in layers:
            acts[(ep, L)] = A[L]
            norms[f"{ep}_L{L}"] = float(A[L].norm(dim=1).mean().item())
        del model, A
        torch.cuda.empty_cache()
        print(f"[transfer] epoch {ep}: acts {acts[(ep, layers[0])].shape} "
              f"mean|h| L{layers[0]}={norms[f'{ep}_L{layers[0]}']:.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    n_pos = Ymy.shape[0]
    tr_idx, te_idx = make_split(n_pos)
    print(f"[transfer] {n_pos} positions -> {tr_idx.numel()} train / {te_idx.numel()} test",
          flush=True)

    out = {"tag": args.tag, "epochs": EPOCHS, "layers": layers,
           "n_positions": int(n_pos), "probe_epochs": args.probe_epochs,
           "max_games": args.max_games, "act_norm": norms, "by_layer": {}}

    for L in layers:
        print(f"\n[transfer] ===== layer {L} =====", flush=True)
        probes = {}
        for ep in EPOCHS:
            probes[ep] = train_probe(acts[(ep, L)], Ymy, tr_idx,
                                     args.probe_epochs, bs, device, n_embd, seed=0)
            print(f"[transfer] L{L} trained probe on epoch {ep} "
                  f"({time.time()-t0:.0f}s)", flush=True)

        # seed control: same checkpoint, different probe init/shuffle
        ctrl = {}
        for ep in ctrl_eps:
            for sd in ctrl_seeds:
                ctrl[(ep, sd)] = train_probe(acts[(ep, L)], Ymy, tr_idx,
                                             args.probe_epochs, bs, device, n_embd,
                                             seed=sd)
                print(f"[transfer] L{L} control probe epoch {ep} seed {sd}", flush=True)

        # --- transfer matrices -------------------------------------------
        mats = {m: np.zeros((len(EPOCHS), len(EPOCHS))) for m in MODES}
        for i, ei in enumerate(EPOCHS):
            for j, ej in enumerate(EPOCHS):
                lg_tr = probe_logits(probes[ei], acts[(ej, L)], tr_idx, device)
                lg_te = probe_logits(probes[ei], acts[(ej, L)], te_idx, device)
                y_tr, y_te = Ymy[tr_idx], Ymy[te_idx]
                for m in MODES:
                    ab = fit_affine(lg_tr, y_tr, m, device)
                    mats[m][i, j] = err_of(apply_affine(lg_te, ab), y_te)
                del lg_tr, lg_te
            print(f"[transfer] L{L} row epoch {ei} done: diag={mats['global'][i,i]*100:.2f}% "
                  f"-> ep20={mats['global'][i,-1]*100:.2f}% ({time.time()-t0:.0f}s)",
                  flush=True)

        # --- geometry -----------------------------------------------------
        D = {ep: directions(probes[ep]) for ep in EPOCHS}
        cos = np.zeros((len(EPOCHS), len(EPOCHS)))
        sub_mean = np.zeros_like(cos)
        sub_min = np.zeros_like(cos)
        for i, ei in enumerate(EPOCHS):
            for j, ej in enumerate(EPOCHS):
                cos[i, j] = mean_cosine(D[ei], D[ej])
                sm, sn = subspace_overlap(D[ei], D[ej])
                sub_mean[i, j], sub_min[i, j] = sm, sn

        # --- control numbers ----------------------------------------------
        control = {}
        for ep in ctrl_eps:
            Dc = [D[ep]] + [directions(ctrl[(ep, sd)]) for sd in ctrl_seeds]
            pair_cos, pair_err, pair_sub = [], [], []
            for a in range(len(Dc)):
                for b in range(a + 1, len(Dc)):
                    pair_cos.append(mean_cosine(Dc[a], Dc[b]))
                    pair_sub.append(subspace_overlap(Dc[a], Dc[b])[0])
            plist = [probes[ep]] + [ctrl[(ep, sd)] for sd in ctrl_seeds]
            for a in range(len(plist)):
                for b in range(len(plist)):
                    if a == b:
                        continue
                    lg_tr = probe_logits(plist[a], acts[(ep, L)], tr_idx, device)
                    lg_te = probe_logits(plist[a], acts[(ep, L)], te_idx, device)
                    ab = fit_affine(lg_tr, Ymy[tr_idx], "global", device)
                    pair_err.append(err_of(apply_affine(lg_te, ab), Ymy[te_idx]))
            control[str(ep)] = {
                "seed_cosine_mean": float(np.mean(pair_cos)),
                "seed_cosine_min": float(np.min(pair_cos)),
                "seed_subspace_mean": float(np.mean(pair_sub)),
                "seed_transfer_err_mean": float(np.mean(pair_err)),
                "own_err": float(mats["global"][EPOCHS.index(ep), EPOCHS.index(ep)]),
            }
            c = control[str(ep)]
            print(f"[transfer] L{L} CONTROL epoch {ep}: cross-seed cosine "
                  f"{c['seed_cosine_mean']:.4f} subspace {c['seed_subspace_mean']:.4f} "
                  f"err {c['seed_transfer_err_mean']*100:.2f}% vs own "
                  f"{c['own_err']*100:.2f}%", flush=True)

        out["by_layer"][str(L)] = {
            "transfer_err": {m: mats[m].tolist() for m in MODES},
            "cosine": cos.tolist(),
            "subspace_mean": sub_mean.tolist(),
            "subspace_min": sub_min.tolist(),
            "control": control,
        }
        del probes, ctrl, D
        torch.cuda.empty_cache()

    res_dir = cfg["paths"]["results_dir"]
    os.makedirs(res_dir, exist_ok=True)
    dest = os.path.join(res_dir, f"probe_transfer_{args.tag}.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[transfer] wrote {dest}", flush=True)

    # --- figure -----------------------------------------------------------
    for L in layers:
        d = out["by_layer"][str(L)]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
        E = [str(e) for e in EPOCHS]
        for ax, (data, title, kw) in zip(axes, [
            (np.array(d["transfer_err"]["global"]) * 100,
             f"transfer error % (gain+bias fit), L{L}", dict(cmap="viridis")),
            (np.array(d["cosine"]),
             f"mean tile-direction cosine, L{L}", dict(cmap="RdBu_r", vmin=-1, vmax=1)),
            (np.array(d["subspace_mean"]),
             f"subspace overlap (mean principal cos), L{L}", dict(cmap="RdBu_r", vmin=-1, vmax=1)),
        ]):
            im = ax.imshow(data, **kw)
            ax.set_xticks(range(len(E))); ax.set_xticklabels(E, fontsize=7)
            ax.set_yticks(range(len(E))); ax.set_yticklabels(E, fontsize=7)
            ax.set_xlabel("activations from epoch"); ax.set_ylabel("probe trained on epoch")
            ax.set_title(title, fontsize=10)
            plt.colorbar(im, ax=ax, fraction=0.046)
        plt.tight_layout()
        fp = os.path.join(res_dir, f"probe_transfer_{args.tag}_L{L}.png")
        plt.savefig(fp, dpi=150); plt.close()
        print(f"[transfer] wrote {fp}", flush=True)

    print(f"[transfer] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
