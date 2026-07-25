"""Export per-epoch board directions + a shared 3D PCA embedding for animation.

`probe_transfer.py` computed the cosine/subspace matrices but kept only those
summaries; the direction vectors themselves were never written out. This job
regenerates them (importing probe_transfer's train_probe/make_split/directions
so the probes are bit-identical to the ones behind the published cosine matrix,
same seed and same split) and saves:

  * the full probe weights per epoch, so any future geometric analysis can be
    done offline without another GPU job;
  * a 3D PCA embedding of the 64 per-tile board directions at every epoch,
    in ONE SHARED BASIS, for animating how the representation moves.

Design decisions that the animation depends on:

  SHARED BASIS. The PCA is fitted once on the pooled trained-epoch directions
  and every epoch is projected into that same basis. Fitting per-epoch would
  give each frame its own axes (with arbitrary sign/rotation), and the apparent
  motion between frames would be an artefact of the basis, not the model.

  UNIT-NORMALISED. Each tile direction is scaled to unit length before the PCA.
  The residual stream norm grows ~4x over training (mean |h| 12 -> 130), so
  without this the animation would be dominated by that scale change rather
  than the rotation we care about. Norms are saved separately so the growth can
  still be displayed if wanted.

  FIT EXCLUDES EPOCH 0. Epoch 0 is a random-init model: its 64 directions are
  mutually near-orthogonal noise (cosine ~0.02 to everything, per the transfer
  run) and would soak up variance in the fit for no reason. It is projected
  into the trained basis afterwards as a reference blob, not used to build it.

  CENTERED. Standard PCA centering, so the embedding shows structure relative
  to the grand mean direction. The per-epoch centroid is saved separately,
  which is what carries the collective drift.

Explained variance is reported per component and must be read before trusting
the picture: 3 components out of 512 dimensions is a shadow of the real
geometry, and if PC1-3 capture little of the variance the animation is
suggestive rather than conclusive.

    python probe_directions.py --config ../configs/config.yaml --profile full
"""
import os
import json
import time
import argparse

import numpy as np
import torch

import common
from nanda_probe import extract_myours
from probe_transfer import EPOCHS, make_split, train_probe, directions


def tile_meta():
    """Board metadata per tile, for colouring/grouping in the animation."""
    meta = []
    for t in range(64):
        r, c = t // 8, t % 8
        ring = int(max(abs(r - 3.5), abs(c - 3.5)) - 0.5)   # 0 = centre 2x2 .. 3 = border
        meta.append({
            "tile": t, "row": r, "col": c, "ring": ring,
            "corner": bool(r in (0, 7) and c in (0, 7)),
            "edge": bool(r in (0, 7) or c in (0, 7)),
            "center4": bool(t in common.CENTER_TILES),
        })
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default="full")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--layers", default="4,7")
    ap.add_argument("--probe-epochs", type=int, default=150)
    ap.add_argument("--max-games", type=int, default=2000)
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
    res_dir = cfg["paths"]["results_dir"]
    os.makedirs(res_dir, exist_ok=True)
    t0 = time.time()

    from mingpt.model import GPT, GPTConfig
    mconf = GPTConfig(common.VOCAB_SIZE, common.BLOCK_SIZE,
                      n_layer=cfg["model"]["n_layer"], n_head=cfg["model"]["n_head"],
                      n_embd=n_embd)
    val_games = common.get_val_games(cfg, args.profile)
    print(f"[dirs] layers={layers} epochs={EPOCHS} device={device}", flush=True)

    # --- extract activations, train one probe per checkpoint -----------------
    acts, Ymy = {}, None
    for ep in EPOCHS:
        model = GPT(mconf)
        model.load_state_dict(torch.load(
            os.path.join(ck_dir, f"gpt_epoch{ep:03d}.ckpt"), map_location="cpu"))
        model = model.to(device)
        A, _, Y = extract_myours(model, val_games, args.max_games, layers, device)
        if Ymy is None:
            Ymy = Y
        else:
            assert torch.equal(Ymy, Y), "label tensors diverged across checkpoints"
        for L in layers:
            acts[(ep, L)] = A[L]
        del model, A
        torch.cuda.empty_cache()
        print(f"[dirs] epoch {ep} extracted ({time.time()-t0:.0f}s)", flush=True)

    tr_idx, _ = make_split(Ymy.shape[0])
    out = {"tag": args.tag, "epochs": EPOCHS, "layers": layers,
           "tiles": tile_meta(), "by_layer": {}}
    npz = {}

    for L in layers:
        W, D = {}, {}
        for ep in EPOCHS:
            probe = train_probe(acts[(ep, L)], Ymy, tr_idx, args.probe_epochs,
                                bs, device, n_embd, seed=0)
            W[ep] = probe.proj.weight.detach().cpu().numpy().reshape(64, 3, n_embd)
            D[ep] = directions(probe).cpu().numpy()          # [64, 512]
            print(f"[dirs] L{L} probe epoch {ep} done ({time.time()-t0:.0f}s)", flush=True)

        norms = np.stack([np.linalg.norm(D[ep], axis=1) for ep in EPOCHS])   # [E, 64]
        Dn = {ep: D[ep] / np.linalg.norm(D[ep], axis=1, keepdims=True) for ep in EPOCHS}

        # --- shared PCA basis, fitted on trained epochs only -----------------
        fit_eps = [e for e in EPOCHS if e != 0]
        X = np.concatenate([Dn[e] for e in fit_eps], axis=0)     # [10*64, 512]
        mu = X.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
        var = S ** 2 / (X.shape[0] - 1)
        evr = (var / var.sum()).tolist()
        basis = Vt[:3]                                            # [3, 512]

        coords = np.stack([(Dn[e] - mu) @ basis.T for e in EPOCHS])   # [E, 64, 3]
        centroid = coords.mean(axis=1)                                 # [E, 3]

        out["by_layer"][str(L)] = {
            "coords": coords.tolist(),
            "centroid": centroid.tolist(),
            "norms": norms.tolist(),
            "explained_variance_ratio": evr[:10],
            "evr_top3_sum": float(sum(evr[:3])),
            "fit_epochs": fit_eps,
        }
        npz[f"W_L{L}"] = np.stack([W[e] for e in EPOCHS])
        npz[f"dirs_L{L}"] = np.stack([D[e] for e in EPOCHS])
        npz[f"basis_L{L}"] = basis
        npz[f"mean_L{L}"] = mu
        print(f"[dirs] L{L} PCA: PC1-3 explain "
              f"{sum(evr[:3])*100:.1f}% "
              f"(PC1 {evr[0]*100:.1f}%, PC2 {evr[1]*100:.1f}%, PC3 {evr[2]*100:.1f}%)",
              flush=True)
        del W, D, Dn

    np.savez_compressed(os.path.join(res_dir, f"probe_dirs_{args.tag}.npz"),
                        epochs=np.array(EPOCHS), **npz)
    dest = os.path.join(res_dir, f"probe_pca3d_{args.tag}.json")
    with open(dest, "w") as f:
        json.dump(out, f)
    print(f"[dirs] wrote {dest} and probe_dirs_{args.tag}.npz "
          f"in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
