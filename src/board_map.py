"""Map of WHERE the board world-model is stored, via exact residual decomposition.

The mine/yours board is *linearly* decodable, so a linear probe over the residual
stream decomposes EXACTLY and additively into per-component contributions. The
final residual is a sum of every component's output:

    resid = emb + sum_L ( sum_h attn_head_out(L,h) + mlp_out(L) )

so for the probe direction w_t (tile t, mine - yours):

    w_t . resid = sum_component (w_t . contribution_c)          (exact)

We project every component (token+pos embedding, all n_head*n_layer attention
heads, all n_layer MLPs) into the probe direction and measure, per component:

  * board_corr : median over tiles of corr(w_t . comp , true signed board_t)
                 -> does this component WRITE board-aligned signal?
  * share      : mean over tiles of cov(w_t . comp, w_t . resid)/var(w_t . resid)
                 -> how much of the readout's actual variance it supplies
                 (shares of the variable components sum to ~1).

Run on the bilinear model and the GELU baseline to see whether making the MLPs
bilinear pushed the board representation into the attention weights.

    python board_map.py --config ../configs/config.yaml --profile full --arch bilinear
    python board_map.py --config ... --arch gelu
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
from train_probe import train_one_probe
from bilinear_eigen import extract_acts_and_labels, pearson


@torch.no_grad()
def component_projections(model, games, n_pos, Wdiff, n_layer, n_head, device):
    """Second pass: per position, split the residual into components and project
    each into the board-readout direction Wdiff [64, D]. Returns dict of
    per-component [N,64] arrays, plus the total readout and the signed board."""
    from data.othello import OthelloBoardState
    model.eval()
    D = Wdiff.shape[1]
    dh = D // n_head

    cap = {}
    handles = []
    def mk_proj_pre(L):
        def h(mod, inp):
            cap[("y", L)] = inp[0].detach()[0]          # concat heads [T, D]
        return h
    def mk_mlp_hook(L):
        def h(mod, inp, out):
            cap[("M", L)] = out.detach()[0]             # [T, D]
        return h
    def mk_block_hook(L):
        def h(mod, inp, out):
            cap[("resid", L)] = out.detach()[0]         # true post-block residual [T,D]
        return h
    Wo = {}
    for i, blk in enumerate(model.blocks, start=1):
        handles.append(blk.attn.proj.register_forward_pre_hook(mk_proj_pre(i)))
        handles.append(blk.mlp.register_forward_hook(mk_mlp_hook(i)))
        Wo[i] = blk.attn.proj.weight.detach()           # [D, D]
    handles.append(model.blocks[n_layer - 1].register_forward_hook(mk_block_hook(n_layer)))

    comps = ([f"A{L}.h{h}" for L in range(1, n_layer + 1) for h in range(n_head)]
             + [f"M{L}" for L in range(1, n_layer + 1)] + ["emb"])
    store = {c: [] for c in comps}
    tot, sgn = [], []
    Wd = Wdiff.to(device)                                # [64, D]

    total = 0
    for game in tqdm(games, desc="components", mininterval=30):
        if total >= n_pos:
            break
        moves = game[:common.BLOCK_SIZE]
        if len(moves) < 2:
            continue
        toks = torch.tensor([common.STOI[m] for m in moves],
                            dtype=torch.long, device=device)[None]
        states = OthelloBoardState().get_gt(moves, "get_state")
        states = torch.tensor(states, dtype=torch.long)
        T = states.size(0)
        my = states.clone()
        even = (torch.arange(T) % 2 == 0)
        my[even] = 2 - my[even]
        signed = (my.float() - 1.0)                     # [T,64]

        cap.clear()
        model(toks)
        emb = (model.tok_emb(toks) + model.pos_emb[:, :T, :])[0]   # [T,D]
        store["emb"].append((emb @ Wd.t()).cpu())
        for L in range(1, n_layer + 1):
            y = cap[("y", L)]                           # [T,D] concat heads
            for h in range(n_head):
                sl = slice(h * dh, (h + 1) * dh)
                headc = y[:, sl] @ Wo[L][:, sl].t()     # [T,D] head h's residual write
                store[f"A{L}.h{h}"].append((headc @ Wd.t()).cpu())
            store[f"M{L}"].append((cap[("M", L)] @ Wd.t()).cpu())
        # true total readout from the captured post-block residual (independent of
        # the component sum, so the reconstruction check is meaningful)
        tot.append((cap[("resid", n_layer)] @ Wd.t()).cpu())
        sgn.append(signed)
        total += T

    for hh in handles:
        hh.remove()
    out = {c: torch.cat(store[c], 0)[:n_pos] for c in comps}
    return out, torch.cat(tot, 0)[:n_pos], torch.cat(sgn, 0)[:n_pos]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--arch", default="bilinear", choices=["gelu", "bilinear"])
    ap.add_argument("--expansion", type=float, default=2.6666667)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n_pos", type=int, default=20000)
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    if args.profile:
        cfg["profile"] = args.profile
    common.add_upstream_to_path(cfg)
    profile = cfg["profile"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = cfg["model"]["n_embd"]
    n_layer = cfg["model"]["n_layer"]; n_head = cfg["model"]["n_head"]

    from mingpt.model import GPT, GPTConfig
    from mingpt.probe_model import BatteryProbeClassification
    mconf = GPTConfig(common.VOCAB_SIZE, common.BLOCK_SIZE,
                      n_layer=n_layer, n_head=n_head, n_embd=D)
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
    print(f"[map] loaded {ckpt} (arch={args.arch})", flush=True)

    # pass A: residual after the LAST block -> train the mine/yours linear probe
    t0 = time.time()
    val_games = common.get_val_games(cfg, profile)
    Post, _, Ymy, _ = extract_acts_and_labels(model, val_games,
                                              cfg["probe"]["max_probe_games"], [n_layer], device)
    probe = BatteryProbeClassification(device, probe_class=3, num_task=64, input_dim=D)
    err = train_one_probe(Post[n_layer], Ymy, probe, cfg["probe"]["epochs"],
                          cfg["probe"]["batch_size"], device)
    Wp = probe.proj.weight.detach()
    Wdiff = (Wp[2::3] - Wp[0::3])                        # [64, D] mine - yours
    print(f"[map] linear mine/yours probe on resid{n_layer}: {err*100:.2f}%  "
          f"({time.time()-t0:.0f}s)", flush=True)

    # pass B: decompose the residual into components, project into readout dirs
    comp, total, signed = component_projections(
        model, val_games[:cfg["probe"]["max_probe_games"]], args.n_pos,
        Wdiff, n_layer, n_head, device)
    signed = signed.to(device); total = total.to(device)

    # reconstruction sanity: sum of components ≈ total readout (up to const biases)
    ssum = sum(comp[c] for c in comp).to(device)
    rc = np.median([pearson(ssum[:, t], total[:, t]) for t in range(64)])
    print(f"[map] reconstruction corr (sum comps vs total), median tile: {rc:.4f}", flush=True)

    var_tot = torch.stack([total[:, t].var() for t in range(64)])      # [64]
    def scores(x):                                                     # x: [N,64] on device
        bc = np.median([abs(pearson(x[:, t], signed[:, t])) for t in range(64)])
        cov = torch.stack([((x[:, t]-x[:, t].mean())*(total[:, t]-total[:, t].mean())).mean()
                           for t in range(64)])
        share = float((cov / var_tot.clamp_min(1e-20)).mean().item())
        mag = float(x.pow(2).mean().sqrt().item())
        return float(bc), share, mag

    res = {"tag": tag, "arch": args.arch, "profile": profile,
           "probe_err": err, "reconstruction_corr": float(rc), "components": {}}
    for c in comp:
        x = comp[c].to(device)
        bc, share, mag = scores(x)
        res["components"][c] = {"board_corr": bc, "share": share, "rms": mag}

    # aggregates
    attn_share = sum(res["components"][f"A{L}.h{h}"]["share"]
                     for L in range(1, n_layer+1) for h in range(n_head))
    mlp_share = sum(res["components"][f"M{L}"]["share"] for L in range(1, n_layer+1))
    emb_share = res["components"]["emb"]["share"]
    res["aggregate"] = {"attn_share": attn_share, "mlp_share": mlp_share,
                        "emb_share": emb_share}
    # top components by |share|
    top = sorted(res["components"].items(), key=lambda kv: -abs(kv[1]["share"]))[:12]
    res["top_by_share"] = [{"comp": k, **v} for k, v in top]

    res_dir = cfg["paths"]["results_dir"]
    os.makedirs(res_dir, exist_ok=True)
    out = os.path.join(res_dir, f"board_map_{tag}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[map] attn share {attn_share:.2f}  mlp share {mlp_share:.2f}  "
          f"emb share {emb_share:.2f}", flush=True)
    print(f"[map] top components by share:", flush=True)
    for k, v in top:
        print(f"    {k:8s} share {v['share']:+.3f}  board_corr {v['board_corr']:.3f}", flush=True)
    print(f"[map] wrote {out}", flush=True)

    # heatmap: attention-head share grid (L x H)
    grid_share = np.array([[res["components"][f"A{L}.h{h}"]["share"]
                            for h in range(n_head)] for L in range(1, n_layer+1)])
    grid_corr = np.array([[res["components"][f"A{L}.h{h}"]["board_corr"]
                           for h in range(n_head)] for L in range(1, n_layer+1)])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, G, ttl in [(axs[0], grid_share, "attn-head share of board readout"),
                       (axs[1], grid_corr, "attn-head board |corr|")]:
        im = ax.imshow(G, aspect="auto", cmap="viridis",
                       extent=[0.5, n_head+0.5, n_layer+0.5, 0.5])
        ax.set_xlabel("head"); ax.set_ylabel("layer"); ax.set_title(ttl)
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"Where the board readout is stored ({tag}, arch={args.arch}) — "
                 f"attn {attn_share:.2f} / mlp {mlp_share:.2f} / emb {emb_share:.2f}")
    fig.tight_layout()
    f1 = os.path.join(res_dir, f"board_map_{tag}.png")
    fig.savefig(f1, dpi=150); plt.close()
    print(f"[map] wrote {f1}", flush=True)


if __name__ == "__main__":
    main()
