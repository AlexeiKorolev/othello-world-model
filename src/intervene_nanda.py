"""Causal intervention using Nanda's LINEAR mine/yours probe.

Cements the relevance of the linear result: it is not enough that the board is
linearly *decodable* in the player-relative (mine/yours) frame -- we show that
the linear probe direction is causally *used* by the model. We edit the residual
stream along the linear mine/yours probe direction to flip one tile's owner and
verify the model's legal-move predictions follow the counterfactual board.

Structurally identical to `intervene.py` (same benchmark, same layer-by-layer
propagation via GPTforProbeIA + mingpt.utils.intervene, same legality metric).
The ONLY changes:
  * the per-layer probes are LINEAR (BatteryProbeClassification), not the 2-layer
    MLP -- so a successful intervention shows a *linear* direction is causal;
  * probes are trained with mine/yours labels (player-relative), and the
    benchmark's absolute from/to colours are converted into mine/yours classes
    using the parity of the intervention position.

    python intervene_nanda.py --config ../configs/config.yaml --profile full
"""
import os
import json
import pickle
import argparse

import numpy as np
import torch
from tqdm import tqdm

import common
from nanda_probe import extract_myours


def train_linear_layer_probe(act, y, epochs, batch_size, device, n_embd):
    from mingpt.probe_model import BatteryProbeClassification
    from torch.utils.data import TensorDataset, DataLoader
    probe = BatteryProbeClassification(device, probe_class=3, num_task=64,
                                       input_dim=n_embd)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ld = DataLoader(TensorDataset(act, y), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        probe.train()
        for a, yy in ld:
            a, yy = a.to(device), yy.to(device)
            _, loss = probe(a, yy)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    probe.eval()
    return probe


def legality_error(logits_vocab, true_legal):
    """top-N tile prediction (N = #true legal moves); count how many are illegal.
    Same reconstructed metric as intervene.py so the two runs are comparable."""
    scores = [(common.ITOS[i], logits_vocab[i].item())
              for i in range(1, common.VOCAB_SIZE)]
    scores.sort(key=lambda t: t[1], reverse=True)
    N = max(1, len(true_legal))
    pred = [m for m, _ in scores[:N]]
    return sum(1 for m in pred if m not in true_legal)


def to_mine_yours(c_abs, side_black_to_move):
    """Map an absolute get_state class (white0/blank1/black2) to the mine/yours
    class at a position where `side_black_to_move` is the side about to move.
    mine=2, yours=0, empty=1. Identity when black is to move, else swap 0<->2."""
    return c_abs if side_black_to_move else (2 - c_abs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    if args.profile:
        cfg["profile"] = args.profile
    common.add_upstream_to_path(cfg)
    profile = cfg["profile"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ic = cfg["intervene"]
    layer_s, layer_e = ic["layer_s"], ic["layer_e"]
    n_embd = cfg["model"]["n_embd"]

    from mingpt.model import GPT, GPTConfig, GPTforProbeIA
    from mingpt.utils import intervene as intervene_step
    from data.othello import OthelloBoardState, permit_reverse

    mconf = GPTConfig(common.VOCAB_SIZE, common.BLOCK_SIZE, n_layer=cfg["model"]["n_layer"],
                      n_head=cfg["model"]["n_head"], n_embd=n_embd)
    ckpt = os.path.join(cfg["paths"]["ckpt_dir"], f"gpt_synthetic_{profile}.ckpt")
    sd = torch.load(ckpt, map_location="cpu")

    model_ia = GPTforProbeIA(mconf, probe_layer=layer_s)
    model_ia.load_state_dict(sd)
    model_ia = model_ia.to(device).eval()

    # --- 1. train a LINEAR mine/yours probe at each layer in [layer_s, layer_e) ---
    plain = GPT(mconf); plain.load_state_dict(sd); plain = plain.to(device).eval()
    val_games = common.get_val_games(cfg, profile)
    layers = list(range(layer_s, layer_e))
    A, _Yabs, Y_my = extract_myours(plain, val_games, cfg["probe"]["max_probe_games"],
                                    layers, device)
    probes = {}
    for L in layers:
        probes[L] = train_linear_layer_probe(A[L], Y_my, cfg["probe"]["epochs"],
                                              cfg["probe"]["batch_size"], device, n_embd)
        print(f"[intervene-nanda] trained LINEAR mine/yours probe at layer {L}", flush=True)
    del A, Y_my

    # --- 2-3. benchmark loop ---
    with open(os.path.join(cfg["paths"]["upstream"], "intervention_benchmark.pkl"), "rb") as f:
        bench = pickle.load(f)
    htd = {"lr": ic["lr"], "steps": ic["steps"], "reg_strg": ic["reg_strength"]}
    n_cases = min(ic["n_cases"], len(bench))

    intv_errs, base_errs = [], []
    for case in tqdm(bench[:n_cases], desc="nanda intervention", mininterval=30):
        history = case["history"]
        pos_int = case["pos_int"]
        ori_color = case["ori_color"]
        intervention_to = 2 - ori_color

        # side about to move at the intervention position (last token, index T-1):
        # black iff (T-1) is odd  <=>  len(history) is even.
        side_black = (len(history) % 2 == 0)
        from_my = to_mine_yours(ori_color, side_black)
        to_my = to_mine_yours(intervention_to, side_black)
        wtd = {"intervention_position": permit_reverse(pos_int),
               "intervention_from": from_my, "intervention_to": to_my}

        g = torch.tensor([common.STOI[s] for s in history], dtype=torch.long, device=device)

        # post-intervention ground-truth legal moves (absolute board; frame-agnostic)
        ab = OthelloBoardState(); ab.update(history)
        r, c = pos_int // 8, pos_int % 8
        ab.state[r, c] = intervention_to - 1
        true_legal = set(ab.get_valid_moves())

        # baseline: un-intervened prediction
        with torch.no_grad():
            x0 = model_ia.forward_1st_stage(g[None, :])
            outs = model_ia.forward_2nd_stage(x0, layer_s, end_layer=-1)
            base_logits, _ = model_ia.predict(outs[-1])
        base_errs.append(legality_error(base_logits[0, -1], true_legal))

        # intervened: edit at layer_s, propagate, re-intervene at each layer
        whole = model_ia.forward_1st_stage(g[None, :])
        p = probes[layer_s]
        mid = whole[0, -1]
        labels = p(mid[None, :])[0].squeeze(0).argmax(dim=-1)
        new_mid = intervene_step(p, mid.clone(), labels, wtd, htd, plot=False)
        whole = whole.clone(); whole[0, -1] = new_mid
        for L in range(layer_s, layer_e - 1):
            whole = model_ia.forward_2nd_stage(whole, L, L + 1)[0]
            p = probes[L + 1]
            mid = whole[0, -1]
            labels = p(mid[None, :])[0].squeeze(0).argmax(dim=-1)
            new_mid = intervene_step(p, mid.clone(), labels, wtd, htd, plot=False)
            whole = whole.clone(); whole[0, -1] = new_mid
        with torch.no_grad():
            intv_logits, _ = model_ia.predict(whole)
        intv_errs.append(legality_error(intv_logits[0, -1], true_legal))

    res = {
        "profile": profile, "n_cases": n_cases, "probe": "linear_mine_yours",
        "intervened_error_mean": float(np.mean(intv_errs)),
        "baseline_error_mean": float(np.mean(base_errs)),
        "intervened_error_std": float(np.std(intv_errs)),
        "baseline_error_std": float(np.std(base_errs)),
        "layer_s": layer_s, "layer_e": layer_e, "metric": "top-N illegal count",
    }
    out = os.path.join(cfg["paths"]["results_dir"], f"intervention_nanda_{profile}.json")
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[intervene-nanda] LINEAR mine/yours intervention: intervened err "
          f"{res['intervened_error_mean']:.3f} vs baseline {res['baseline_error_mean']:.3f}",
          flush=True)
    print(f"[intervene-nanda] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
