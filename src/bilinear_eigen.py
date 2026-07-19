"""Weight-based board features in the bilinear Othello-GPT (arXiv:2410.08417).

The whole point of swapping GELU -> bilinear MLPs was to read the world model out
of the WEIGHTS, not the activations. A bilinear MLP has no elementwise
nonlinearity:

    g(x)   = (W x) * (V x)          # x = ln2(residual), the MLP input
    MLP(x) = P g(x)                 # no biases

so its contribution to any output direction u is an EXACT quadratic form of x:

    u . MLP(x) = (P^T u) . g(x)
               = sum_a (P^T u)_a (w_a . x)(v_a . x)
               = x^T Q_u x ,   Q_u = sum_a (P^T u)_a  w_a v_a^T = W^T diag(P^T u) V

Only the symmetric part of Q_u survives x^T Q x, so we symmetrize and
eigendecompose (Dooms et al.'s "interaction matrix"):

    Q_u^sym = sum_i lambda_i q_i q_i^T   ->   u.MLP(x) = sum_i lambda_i (q_i . x)^2

The eigenvectors q_i are INPUT directions the layer reads; eigenvalues lambda_i
are their signed importance (positive = boosts u, negative = suppresses). This is
computed purely from weights -- no data.

Output direction u: the player-relative board readout. For each tile we take the
mine/yours linear probe's logit-difference direction u_t = w_probe[t, mine] -
w_probe[t, yours] (analogous to the paper's True-False direction). Then per tile:
  * eigendecompose Q_{u_t} -> spectrum + eigenfeatures, straight from weights;
  * effective rank of the spectrum (is the readout low-rank, as the paper finds?);
  * how well a k-eigenvector truncation reconstructs the full quadratic (the
    paper's "2 eigenvectors capture the feature" test);
  * whether the weight-derived quantity x^T Q x actually tracks the TRUE board
    (correlation with the mine/yours label) -- i.e. the representation the
    activation-probe learned is literally present in the MLP weights.

    python bilinear_eigen.py --config ../configs/config.yaml --profile full
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
def extract_acts_and_labels(model, games, n_games, layers, device, collect_legal=False):
    """One forward pass per game, capturing for each target layer L:
      * post-block residual (what the probe reads) -> to TRAIN the readout probe
      * ln2 output x (the MLP input)              -> for the weight-form analysis
    plus the mine/yours board label per position (yours0/empty1/mine2), and
    optionally the legal-next-move mask per position (over the token vocab)."""
    from data.othello import OthelloBoardState
    model.eval()

    cap = {}
    handles = []
    def mk_block_hook(L):
        def h(mod, inp, out):
            cap[("post", L)] = out.detach()[0]      # [T, D]
        return h
    def mk_ln2_hook(L):
        def h(mod, inp, out):
            cap[("x", L)] = out.detach()[0]         # [T, D]
        return h
    for i, blk in enumerate(model.blocks, start=1):
        if i in layers:
            handles.append(blk.register_forward_hook(mk_block_hook(i)))
            handles.append(blk.ln2.register_forward_hook(mk_ln2_hook(i)))

    post = {L: [] for L in layers}
    xin = {L: [] for L in layers}
    ys_my = []
    legal = []
    for game in tqdm(games[:n_games], desc="extract", mininterval=30):
        moves = game[:common.BLOCK_SIZE]
        if len(moves) < 2:
            continue
        toks = torch.tensor([common.STOI[m] for m in moves],
                            dtype=torch.long, device=device)[None]
        states = OthelloBoardState().get_gt(moves, "get_state")   # [T,64] absolute
        states = torch.tensor(states, dtype=torch.long)
        T = states.size(0)
        my = states.clone()
        even = (torch.arange(T) % 2 == 0)                         # white-to-move ply
        my[even] = 2 - my[even]                                   # -> player-relative
        if collect_legal:
            # legal next moves AFTER each played move -> the model's target at t
            valids = OthelloBoardState().get_gt(moves, "get_valid_moves")
            lm = torch.zeros(T, common.VOCAB_SIZE, dtype=torch.uint8)
            for t, vs in enumerate(valids):
                for mv in vs:
                    lm[t, common.STOI[mv]] = 1
            legal.append(lm)
        cap.clear()
        model(toks)
        for L in layers:
            post[L].append(cap[("post", L)].cpu())
            xin[L].append(cap[("x", L)].cpu())
        ys_my.append(my)

    for hh in handles:
        hh.remove()
    Post = {L: torch.cat(post[L], 0) for L in layers}             # [N, D]
    Xin = {L: torch.cat(xin[L], 0) for L in layers}               # [N, D]
    Ymy = torch.cat(ys_my, 0)                                     # [N, 64]
    Legal = torch.cat(legal, 0) if collect_legal else None        # [N, vocab]
    return Post, Xin, Ymy, Legal


def interaction_matrix(blk, u):
    """Fold a block's bilinear MLP against output direction u (residual space)
    -> symmetric interaction matrix Q (input/ln2 space). u: [D]."""
    W = blk.mlp.w.weight.detach()          # [H, D]
    V = blk.mlp.v.weight.detach()          # [H, D]
    P = blk.mlp.proj.weight.detach()       # [D, H]
    h = P.t() @ u                          # [H]  = P^T u
    Q = W.t() @ (h[:, None] * V)           # [D, D] = sum_a h_a w_a v_a^T
    return 0.5 * (Q + Q.t())


def eff_rank(evals):
    """Participation ratio of the squared spectrum: effective # of eigen-directions
    that carry comparable weight. (sum e^2)^2 / sum e^4."""
    e2 = evals ** 2
    return (e2.sum() ** 2 / (e2 ** 2).sum().clamp_min(1e-30)).item()


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-20)
    return (a @ b / denom).item()


def run_unembed(model, cfg, args, tag, device, D, epochs, bs, k_list):
    """Decompose the FINAL block's bilinear MLP against the model's OWN output
    directions -- the move-legality logits -- instead of an imposed board basis.

    For move token j the effective unembedding direction (residual space) is
    u_j = ln_f.weight (o) head.weight[j] (the ln_f gain folded in; the final MLP
    output feeds ln_f -> head with only these linear maps in between). Fold the
    last block's bilinear MLP against u_j -> Q_j, eigendecompose. This asks the
    model-native question 'what input interactions make move j legal?' with no
    board assumption. We then INTERPRET the eigenfeatures by measuring how much of
    each lives in the board subspace (mine/yours probe directions in the same
    ln2-input space) -- discovering, not assuming, the board."""
    from mingpt.probe_model import BatteryProbeClassification
    profile = cfg["profile"]
    Lf = cfg["model"]["n_layer"]                              # final block
    tag = tag + "_unembed"

    t0 = time.time()
    val_games = common.get_val_games(cfg, profile)
    Post, Xin, Ymy, Legal = extract_acts_and_labels(
        model, val_games, cfg["probe"]["max_probe_games"], [Lf], device,
        collect_legal=True)
    N = Ymy.shape[0]
    print(f"[eigen] extracted {N} positions in {time.time()-t0:.0f}s", flush=True)

    blk = model.blocks[Lf - 1]
    ln_f = model.ln_f
    head = model.head
    gamma = ln_f.weight.detach()                             # [D]
    Wu = head.weight.detach()                                # [vocab, D]

    # board subspace in the MLP-INPUT (ln2) space: train mine/yours probe on Xin[Lf]
    board_probe = BatteryProbeClassification(device, probe_class=3, num_task=64, input_dim=D)
    err_board = train_one_probe(Xin[Lf], Ymy, board_probe, epochs, bs, device)
    Wp = board_probe.proj.weight.detach()                    # [192, D]
    Bdir = (Wp[2::3] - Wp[0::3])                             # [64, D] mine-yours dirs
    Bq, _ = torch.linalg.qr(Bdir.t())                        # [D, r] orthonormal board basis
    r = Bq.shape[1]
    chance = r / D                                           # expected ||proj||^2 of a random unit vec
    print(f"[eigen] board subspace in ln2-space: rank {r}, chance frac {chance:.3f} "
          f"(board probe on MLP input: {err_board*100:.2f}%)", flush=True)

    # positions subsample + legality target
    idx = torch.randperm(N)[:min(args.n_eig_pos, N)]
    Xs = Xin[Lf][idx].to(device)                             # [n, D]
    Lg = Legal[idx].float().to(device)                       # [n, vocab]

    # score only moves with real variance in legality (skip near-constant tokens)
    freq = Lg.mean(0)                                        # [vocab]
    move_ids = [j for j in range(common.VOCAB_SIZE) if 0.02 < freq[j].item() < 0.98]
    print(f"[eigen] scoring {len(move_ids)} move tokens (0.02<legal-rate<0.98)", flush=True)

    per_move = []
    eff_ranks, legal_corr_full, align_top8 = [], [], []
    recon_corr = {k: [] for k in k_list}
    legal_corr_k = {k: [] for k in k_list}
    for j in move_ids:
        u = gamma * Wu[j]                                    # effective unembed dir [D]
        Q = interaction_matrix(blk, u)
        evals, evecs = torch.linalg.eigh(Q)
        proj = Xs @ evecs
        sq = proj ** 2
        pred_full = sq @ evals                              # x^T Q x = u.MLP contribution to logit j
        er = eff_rank(evals)
        lc = pearson(pred_full, Lg[:, j])
        eff_ranks.append(er); legal_corr_full.append(lc)
        order = torch.argsort(evals.abs(), descending=True)
        for k in k_list:
            sel = order[:k]
            pred_k = sq[:, sel] @ evals[sel]
            recon_corr[k].append(pearson(pred_k, pred_full))
            legal_corr_k[k].append(pearson(pred_k, Lg[:, j]))
        # board-alignment of the top-8 eigenfeatures: ||B^T q||^2 for unit q
        top8 = evecs[:, order[:8]]                          # [D, 8]
        frac = (Bq.t() @ top8).pow(2).sum(0)               # [8] fraction in board subspace
        align_top8.append(frac.mean().item())
        per_move.append({"token": j, "move_idx": int(common.ITOS[j]),
                         "legal_rate": round(freq[j].item(), 3),
                         "eff_rank": round(er, 1), "legal_corr": round(lc, 3),
                         "board_align_top8": round(frac.mean().item(), 3)})

    er_a = np.array(eff_ranks); lc_a = np.abs(np.array(legal_corr_full))
    al_a = np.array(align_top8)
    summ = {
        "tag": tag, "profile": profile, "final_layer": Lf,
        "n_moves_scored": len(move_ids),
        "board_subspace_rank": r, "board_align_chance": chance,
        "board_probe_err_on_mlp_input": err_board,
        "eff_rank_median": float(np.median(er_a)),
        "legal_corr_median": float(np.median(lc_a)),
        "legal_corr_mean_abs": float(lc_a.mean()),
        "board_align_top8_median": float(np.median(al_a)),
        "board_align_top8_over_chance": float(np.median(al_a) / chance),
        "recon_corr_median": {str(k): float(np.median(recon_corr[k])) for k in k_list},
        "legal_corr_median_k": {str(k): float(np.median(np.abs(legal_corr_k[k]))) for k in k_list},
        "per_move": per_move,
    }
    print(f"[eigen] UNEMBED final layer {Lf}: "
          f"legal|corr|(med) {summ['legal_corr_median']:.3f}  "
          f"eff_rank(med) {summ['eff_rank_median']:.1f}  "
          f"recon@2 {summ['recon_corr_median']['2']:.3f}  "
          f"legal@8 {summ['legal_corr_median_k']['8']:.3f}  "
          f"board_align(top8) {summ['board_align_top8_median']:.3f} "
          f"(={summ['board_align_top8_over_chance']:.1f}x chance {chance:.3f})", flush=True)

    res_dir = cfg["paths"]["results_dir"]
    os.makedirs(res_dir, exist_ok=True)
    out = os.path.join(res_dir, f"bilinear_eigen_{tag}.json")
    with open(out, "w") as f:
        json.dump(summ, f, indent=2)
    print(f"[eigen] wrote {out}", flush=True)

    # figure: legality recon + board-alignment story
    ks = k_list
    plt.figure(figsize=(7, 4.5))
    plt.plot(ks, [summ["recon_corr_median"][str(k)] for k in ks], "o-",
             color="#268", label="recon of u.MLP legality (self)")
    plt.plot(ks, [summ["legal_corr_median_k"][str(k)] for k in ks], "o-",
             color="#c44", label="corr with TRUE move legality")
    plt.axhline(0.75, ls="--", color="#999", lw=1)
    plt.xscale("log", base=2)
    plt.xlabel("# eigenvectors kept (by |lambda|)")
    plt.ylabel("median correlation over moves")
    plt.title(f"Weight-space move-legality readout, final layer {Lf} ({tag})")
    plt.ylim(0, 1); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    f1 = os.path.join(res_dir, f"bilinear_eigen_{tag}_recon.png")
    plt.savefig(f1, dpi=150); plt.close()
    print(f"[eigen] figure: {f1}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--expansion", type=float, default=2.6666667)
    ap.add_argument("--layers", default=None,
                    help="comma-list of layers to analyze (default: config probe layers)")
    ap.add_argument("--n_eig_pos", type=int, default=20000,
                    help="positions subsampled for the eigen reconstruction/correlation")
    ap.add_argument("--k_list", default="1,2,4,8,16,32",
                    help="eigenvector truncations to score")
    ap.add_argument("--output-mode", dest="output_mode", default="board",
                    choices=["board", "unembed"],
                    help="board: decompose against mine/yours tile readouts (imposed lens); "
                         "unembed: decompose the FINAL layer against move-legality logits "
                         "(model-native), then interpret eigenfeatures via the board subspace")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    if args.profile:
        cfg["profile"] = args.profile
    common.add_upstream_to_path(cfg)
    profile = cfg["profile"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    D = cfg["model"]["n_embd"]

    from mingpt.model import GPTConfig
    from mingpt.probe_model import BatteryProbeClassification
    from bilinear_model import make_bilinear_gpt

    mconf = GPTConfig(common.VOCAB_SIZE, common.BLOCK_SIZE,
                      n_layer=cfg["model"]["n_layer"], n_head=cfg["model"]["n_head"],
                      n_embd=D)
    model = make_bilinear_gpt(mconf, expansion=args.expansion)
    tag = args.tag or (profile + "_bilinear")
    ckpt = args.ckpt or os.path.join(cfg["paths"]["ckpt_dir"],
                                     f"gpt_synthetic_{profile}_bilinear.ckpt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device)
    print(f"[eigen] loaded {ckpt}", flush=True)

    layers = ([int(x) for x in args.layers.split(",")] if args.layers
              else cfg["probe"]["layers"])
    k_list = [int(x) for x in args.k_list.split(",")]
    epochs = cfg["probe"]["epochs"]
    bs = cfg["probe"]["batch_size"]

    if args.output_mode == "unembed":
        run_unembed(model, cfg, args, tag, device, D, epochs, bs, k_list)
        return

    t0 = time.time()
    val_games = common.get_val_games(cfg, profile)
    Post, Xin, Ymy, _ = extract_acts_and_labels(model, val_games,
                                                cfg["probe"]["max_probe_games"], layers, device)
    N = Ymy.shape[0]
    print(f"[eigen] extracted {N} positions x 64 tiles in {time.time()-t0:.0f}s", flush=True)

    # signed board target per tile: mine(2)->+1, empty(1)->0, yours(0)->-1
    signed = (Ymy.float() - 1.0)                                  # [N, 64]

    results = {"tag": tag, "profile": profile, "layers": {},
               "note": "u_t = mine/yours probe logit-diff (mine - yours); "
                       "Q_t folded from the block's bilinear MLP; "
                       "x = ln2 output (MLP input)."}

    for L in layers:
        blk = model.blocks[L - 1]
        # 1) train the mine/yours linear readout on post-block residual -> get u_t
        probe = BatteryProbeClassification(device, probe_class=3, num_task=64, input_dim=D)
        err_my = train_one_probe(Post[L], Ymy, probe, epochs, bs, device)
        Wp = probe.proj.weight.detach()                          # [192, D], row t*3+c
        U = (Wp[2::3] - Wp[0::3])                                # [64, D]  mine - yours

        # subsample positions for the eigen reconstruction/correlation
        idx = torch.randperm(N)[:min(args.n_eig_pos, N)]
        Xs = Xin[L][idx].to(device)                              # [n, D]
        Ss = signed[idx].to(device)                              # [n, 64]

        per_tile = []
        eff_ranks, board_corr_full = [], []
        recon_corr = {k: [] for k in k_list}
        board_corr_k = {k: [] for k in k_list}
        spectra_examples = {}
        for t in range(64):
            u = U[t]
            Q = interaction_matrix(blk, u)                       # [D, D] on device
            evals, evecs = torch.linalg.eigh(Q)                  # ascending; evecs cols
            # weight-only feature: activation of each eigen-direction on real x
            proj = Xs @ evecs                                    # [n, D]
            sq = proj ** 2                                       # [n, D]
            pred_full = sq @ evals                               # [n]  = x^T Q x  (u.MLP contribution)
            er = eff_rank(evals)
            bc = pearson(pred_full, Ss[:, t])
            eff_ranks.append(er)
            board_corr_full.append(bc)
            # order eigen-directions by |lambda|
            order = torch.argsort(evals.abs(), descending=True)
            for k in k_list:
                sel = order[:k]
                pred_k = sq[:, sel] @ evals[sel]
                recon_corr[k].append(pearson(pred_k, pred_full))
                board_corr_k[k].append(pearson(pred_k, Ss[:, t]))
            top = evals[order[:6]].tolist()
            per_tile.append({"tile": t, "eff_rank": er, "board_corr": bc,
                             "top6_abs_eigs": [round(v, 4) for v in top]})
            if t in (19, 20, 26, 37):                            # a few central tiles
                spectra_examples[str(t)] = evals.detach().cpu().tolist()

        er_arr = np.array(eff_ranks)
        bc_arr = np.array(board_corr_full)
        summ = {
            "mine_yours_probe_err": err_my,
            "eff_rank_median": float(np.median(er_arr)),
            "eff_rank_mean": float(er_arr.mean()),
            "board_corr_full_median": float(np.median(np.abs(bc_arr))),
            "board_corr_full_mean_abs": float(np.abs(bc_arr).mean()),
            "recon_corr_median": {str(k): float(np.median(recon_corr[k])) for k in k_list},
            "board_corr_median": {str(k): float(np.median(np.abs(board_corr_k[k]))) for k in k_list},
            "per_tile": per_tile,
            "spectra_examples": spectra_examples,
        }
        results["layers"][str(L)] = summ
        print(f"[eigen] L{L}: probe_err {err_my*100:.2f}%  "
              f"eff_rank(med) {summ['eff_rank_median']:.1f}  "
              f"board|corr|(med) {summ['board_corr_full_median']:.3f}  "
              f"recon@2 {summ['recon_corr_median']['2']:.3f}  "
              f"board@2 {summ['board_corr_median']['2']:.3f}", flush=True)

    res_dir = cfg["paths"]["results_dir"]
    os.makedirs(res_dir, exist_ok=True)
    out = os.path.join(res_dir, f"bilinear_eigen_{tag}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[eigen] wrote {out}", flush=True)

    # ---- figures ----
    Ls = sorted(int(L) for L in results["layers"])
    # (a) reconstruction + board correlation vs #eigenvectors, at best board layer
    best_L = max(Ls, key=lambda L: results["layers"][str(L)]["board_corr_full_median"])
    s = results["layers"][str(best_L)]
    ks = k_list
    plt.figure(figsize=(7, 4.5))
    plt.plot(ks, [s["recon_corr_median"][str(k)] for k in ks], "o-",
             color="#268", label="recon of x·(u.MLP) (self)")
    plt.plot(ks, [s["board_corr_median"][str(k)] for k in ks], "o-",
             color="#c44", label="corr with TRUE board (mine/yours)")
    plt.axhline(0.75, ls="--", color="#999", lw=1)
    plt.xscale("log", base=2)
    plt.xlabel("# eigenvectors kept (by |lambda|)")
    plt.ylabel("median correlation over 64 tiles")
    plt.title(f"Weight-space board readout, layer {best_L} ({tag})")
    plt.ylim(0, 1); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    f1 = os.path.join(res_dir, f"bilinear_eigen_{tag}_recon.png")
    plt.savefig(f1, dpi=150); plt.close()

    # (b) effective rank across layers
    plt.figure(figsize=(7, 4.5))
    med = [results["layers"][str(L)]["eff_rank_median"] for L in Ls]
    plt.plot(Ls, med, "o-", color="#268")
    plt.xlabel("layer"); plt.ylabel("median effective rank of Q (64 tiles)")
    plt.title(f"Board-readout interaction matrices are low-rank ({tag})")
    plt.grid(alpha=0.3); plt.tight_layout()
    f2 = os.path.join(res_dir, f"bilinear_eigen_{tag}_effrank.png")
    plt.savefig(f2, dpi=150); plt.close()

    # (c) example eigenvalue spectra at best layer
    plt.figure(figsize=(7, 4.5))
    for t, sp in s["spectra_examples"].items():
        ev = sorted(sp, key=abs, reverse=True)
        plt.plot(range(1, len(ev) + 1), ev, lw=1, label=f"tile {t}")
    plt.xlim(1, 40)
    plt.xlabel("eigenvalue rank (by |lambda|)"); plt.ylabel("eigenvalue")
    plt.title(f"Eigenvalue spectra of board readouts, layer {best_L} ({tag})")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    f3 = os.path.join(res_dir, f"bilinear_eigen_{tag}_spectra.png")
    plt.savefig(f3, dpi=150); plt.close()

    print(f"[eigen] figures: {f1}\n{f2}\n{f3}", flush=True)
    print(f"[eigen] best board layer {best_L}: "
          f"board|corr|(med) {s['board_corr_full_median']:.3f}, "
          f"eff_rank(med) {s['eff_rank_median']:.1f}, "
          f"board@2eig {s['board_corr_median']['2']:.3f}", flush=True)


if __name__ == "__main__":
    main()
