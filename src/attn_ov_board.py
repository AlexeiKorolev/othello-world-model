"""Do the attention-head WEIGHTS map the board? (OV-circuit analysis)

board_map.py showed board signal flows through certain heads at the ACTIVATION
level. This asks the stronger, weight-level question: does a head's OV circuit
    W_OV = W_O^(h) @ W_V^(h)   (D x D; maps ln1(resid) it reads -> residual it writes)
implement a board map? We measure, per head, using the mine/yours board subspace:

  out_frac = ||B_out^T W_OV||_F^2 / ||W_OV||_F^2
             fraction of the head's WRITE energy landing in the board-readout
             subspace (chance ~ r/D).
  io_frac  = ||B_out^T W_OV B_in||_F^2 / ||W_OV||_F^2
             board -> board: how much of W_OV maps the board subspace (at the
             head's input) into the board-readout subspace (chance ~ (r/D)^2).

and the tile x tile transfer matrix (weights + board directions, no black box):
  T_h[t,s] = w_out_t . W_OV . enc_in_s
             inject "source tile s is mine (vs yours)" -> read "dest tile t
             readout". Diagonal / Othello-line structure => the weights transport
             board state.

Frames (kept exact): W_V^(h) reads ln1(resid) directly, so the INPUT board basis
B_in and encoding dirs enc are estimated in each layer's ln1-output space (the
head's actual input). The head's write is added to the residual and reaches the
final readout additively, so the OUTPUT readout w_out (probe on the final block's
residual) dotted with the write is the head's exact contribution to the board
readout.

    python attn_ov_board.py --config ../configs/config.yaml --profile full --arch bilinear
    python attn_ov_board.py --config ... --arch gelu
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


@torch.no_grad()
def extract_ln1_and_final(model, games, n_games, n_layer, device):
    """Capture, per position: each block's ln1 OUTPUT (the head input space) and
    the final post-block residual, plus mine/yours labels."""
    from data.othello import OthelloBoardState
    model.eval()
    cap = {}
    handles = []
    def mk_ln1(L):
        def h(mod, inp, out): cap[("ln1", L)] = out.detach()[0]
        return h
    def mk_final():
        def h(mod, inp, out): cap["final"] = out.detach()[0]
        return h
    for i, blk in enumerate(model.blocks, start=1):
        handles.append(blk.ln1.register_forward_hook(mk_ln1(i)))
    handles.append(model.blocks[n_layer - 1].register_forward_hook(mk_final()))

    ln1 = {L: [] for L in range(1, n_layer + 1)}
    fin, ys = [], []
    for game in tqdm(games[:n_games], desc="extract", mininterval=30):
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
        cap.clear(); model(toks)
        for L in range(1, n_layer + 1):
            ln1[L].append(cap[("ln1", L)].cpu())
        fin.append(cap["final"].cpu()); ys.append(my)
    for hh in handles:
        hh.remove()
    Ln1 = {L: torch.cat(ln1[L], 0) for L in range(1, n_layer + 1)}
    return Ln1, torch.cat(fin, 0), torch.cat(ys, 0)


def encoding_dirs(X, Ymy):
    """Per-tile mine-minus-yours mean activation pattern (encoding direction).
    X: [N, D], Ymy: [N, 64] in {0=yours,1=empty,2=mine}. Returns [64, D]."""
    D = X.shape[1]
    E = torch.zeros(64, D)
    for t in range(64):
        yt = Ymy[:, t]
        mine = X[yt == 2]; yours = X[yt == 0]
        if len(mine) and len(yours):
            E[t] = mine.mean(0) - yours.mean(0)
    return E


def ortho_basis(M):
    """Orthonormal basis (D x r) for the row space of M [k, D]."""
    q, _ = torch.linalg.qr(M.t())
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--arch", default="bilinear", choices=["gelu", "bilinear"])
    ap.add_argument("--expansion", type=float, default=2.6666667)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    if args.profile:
        cfg["profile"] = args.profile
    common.add_upstream_to_path(cfg)
    profile = cfg["profile"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = cfg["model"]["n_embd"]; n_layer = cfg["model"]["n_layer"]; n_head = cfg["model"]["n_head"]
    dh = D // n_head

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
    print(f"[ov] loaded {ckpt} (arch={args.arch})", flush=True)

    t0 = time.time()
    val_games = common.get_val_games(cfg, profile)
    Ln1, Fin, Ymy = extract_ln1_and_final(model, val_games,
                                          cfg["probe"]["max_probe_games"], n_layer, device)
    print(f"[ov] extracted {Fin.shape[0]} positions in {time.time()-t0:.0f}s", flush=True)

    # OUTPUT board readout: mine/yours linear probe on the final residual
    probe = BatteryProbeClassification(device, probe_class=3, num_task=64, input_dim=D)
    err = train_one_probe(Fin, Ymy, probe, cfg["probe"]["epochs"], cfg["probe"]["batch_size"], device)
    Wp = probe.proj.weight.detach().cpu()
    Wout = (Wp[2::3] - Wp[0::3])                              # [64, D] readout (mine-yours)
    Bout = ortho_basis(Wout)                                  # [D, r]
    r = Bout.shape[1]; chance_out = r / D; chance_io = (r / D) ** 2
    print(f"[ov] final board readout {err*100:.2f}%; board rank {r}, "
          f"chance out {chance_out:.3f} io {chance_io:.4f}", flush=True)

    # INPUT board subspace + encoding dirs per layer (ln1-output space)
    Bin, Enc = {}, {}
    for L in range(1, n_layer + 1):
        E = encoding_dirs(Ln1[L], Ymy)                       # [64, D]
        Enc[L] = E
        Bin[L] = ortho_basis(E)                              # [D, r]

    Wout_d = Wout.to(device); Bout_d = Bout.to(device)

    # optional: board_corr per head from board_map (to label board heads)
    bc = {}
    bmap = os.path.join(cfg["paths"]["results_dir"], f"board_map_{tag}.json")
    if os.path.exists(bmap):
        d = json.load(open(bmap))
        for k, v in d["components"].items():
            if k.startswith("A"):
                bc[k] = v["board_corr"]

    results = {"tag": tag, "arch": args.arch, "board_rank": r,
               "chance_out": chance_out, "chance_io": chance_io,
               "final_readout_err": err, "heads": {}}
    rows = []
    for L in range(1, n_layer + 1):
        blk = model.blocks[L - 1]
        Wv = blk.attn.value.weight.detach()                  # [D, D]
        Wo = blk.attn.proj.weight.detach()                   # [D, D]
        Bin_d = Bin[L].to(device); Enc_d = Enc[L].to(device)
        for h in range(n_head):
            Wv_h = Wv[h*dh:(h+1)*dh, :]                       # [dh, D]
            Wo_h = Wo[:, h*dh:(h+1)*dh]                       # [D, dh]
            Wov = Wo_h @ Wv_h                                 # [D, D]
            fro2 = Wov.pow(2).sum().clamp_min(1e-20)
            out_frac = (Bout_d.t() @ Wov).pow(2).sum() / fro2
            io_frac = (Bout_d.t() @ Wov @ Bin_d).pow(2).sum() / fro2
            name = f"A{L}.h{h}"
            rec = {"out_frac": float(out_frac), "io_frac": float(io_frac),
                   "out_over_chance": float(out_frac / chance_out),
                   "io_over_chance": float(io_frac / chance_io),
                   "board_corr": bc.get(name)}
            results["heads"][name] = rec
            rows.append((name, rec))

    # rank heads by io_frac (board->board through the weights)
    rows.sort(key=lambda kv: -kv[1]["io_frac"])
    print("[ov] top heads by board->board io_frac (x chance):", flush=True)
    for name, rec in rows[:10]:
        bcs = f"{rec['board_corr']:.2f}" if rec["board_corr"] is not None else "  -"
        print(f"    {name:8s} io {rec['io_frac']:.3f} ({rec['io_over_chance']:5.1f}x)  "
              f"out {rec['out_frac']:.3f} ({rec['out_over_chance']:4.1f}x)  "
              f"board_corr {bcs}", flush=True)

    res_dir = cfg["paths"]["results_dir"]
    os.makedirs(res_dir, exist_ok=True)
    out = os.path.join(res_dir, f"attn_ov_board_{tag}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[ov] wrote {out}", flush=True)

    # tile x tile transfer matrices for the top-2 board->board heads
    top2 = rows[:2]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (name, rec) in zip(axs, top2):
        L = int(name[1:name.index(".")]); h = int(name[name.index("h")+1:])
        blk = model.blocks[L - 1]
        Wv_h = blk.attn.value.weight.detach()[h*dh:(h+1)*dh, :]
        Wo_h = blk.attn.proj.weight.detach()[:, h*dh:(h+1)*dh]
        Wov = (Wo_h @ Wv_h)
        T = (Wout_d @ Wov @ Enc[L].to(device).t()).cpu().numpy()   # [64 dst, 64 src]
        vlim = np.abs(T).max()
        im = ax.imshow(T, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        ax.set_title(f"{name}  io {rec['io_over_chance']:.0f}x chance\n"
                     f"T[dst tile, src tile] = w_dst . W_OV . enc_src")
        ax.set_xlabel("source tile (is-mine)"); ax.set_ylabel("dest tile (readout)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Do attention-head weights transport the board? ({tag})")
    fig.tight_layout()
    f1 = os.path.join(res_dir, f"attn_ov_board_{tag}_transfer.png")
    fig.savefig(f1, dpi=150); plt.close()
    print(f"[ov] wrote {f1}", flush=True)

    # diagonal dominance of the transfer matrices (board self-transport signal)
    for name, rec in top2:
        L = int(name[1:name.index(".")]); h = int(name[name.index("h")+1:])
        blk = model.blocks[L - 1]
        Wov = (blk.attn.proj.weight.detach()[:, h*dh:(h+1)*dh]
               @ blk.attn.value.weight.detach()[h*dh:(h+1)*dh, :])
        T = (Wout_d @ Wov @ Enc[L].to(device).t()).cpu()
        diag = T.diag().abs().mean().item()
        off = (T.abs().sum() - T.diag().abs().sum()).item() / (64*64 - 64)
        print(f"[ov] {name} transfer: mean|diag| {diag:.3f} vs mean|off-diag| {off:.3f} "
              f"(ratio {diag/max(off,1e-9):.2f})", flush=True)


if __name__ == "__main__":
    main()
