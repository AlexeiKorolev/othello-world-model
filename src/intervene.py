"""Causal intervention + latent saliency maps (paper Sec. 4-5).

Reproduces the central causal claim: editing the model's *internal* board
representation (via the probe direction, propagated through layers) changes the
model's move predictions to match the counterfactual board. We:

  1. train nonlinear probes (mid_dim 128) at layers layer_s..layer_e-1 on our
     trained GPT (the intervention propagation needs a probe per layer);
  2. for each benchmark case (curated in upstream intervention_benchmark.pkl),
     flip one tile's colour and propagate the edit layer-by-layer using
     GPTforProbeIA (port of plot_attribution_via_intervention notebook);
  3. score legality error of the post-edit prediction vs the *intervened*
     board's true legal moves, and compare to the un-intervened baseline;
  4. emit a latent-saliency-map figure for one example.

Legality-error metric (reconstructed -- exact scoring is not shipped upstream):
for a board with N true legal moves, take the model's top-N predicted tiles;
error = how many of them are illegal on that board. The paper reports
intervened ~0.12 vs baseline ~2.68.

    python intervene.py --config ../configs/config.yaml --profile pilot
"""
import os
import json
import pickle
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import common
from train_probe import extract


def train_layer_probe(act, y, mid_dim, epochs, batch_size, device, n_embd):
    from mingpt.probe_model import BatteryProbeClassificationTwoLayer
    from torch.utils.data import TensorDataset, DataLoader
    probe = BatteryProbeClassificationTwoLayer(device, probe_class=3, num_task=64,
                                               mid_dim=mid_dim, input_dim=n_embd)
    # cosine schedule + many epochs (see train_probe note): under-trained probes
    # give a useless probe direction and the intervention becomes inert.
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
    """logits over the 61-token vocab -> top-N tile prediction; count how many
    predicted tiles are illegal on the board (N = #true legal moves)."""
    scores = [(common.ITOS[i], logits_vocab[i].item())
              for i in range(1, common.VOCAB_SIZE)]   # skip pad idx 0
    scores.sort(key=lambda t: t[1], reverse=True)
    N = max(1, len(true_legal))
    pred = [m for m, _ in scores[:N]]
    return sum(1 for m in pred if m not in true_legal)


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
    from data.othello import OthelloBoardState, permit, permit_reverse

    mconf = GPTConfig(common.VOCAB_SIZE, common.BLOCK_SIZE, n_layer=cfg["model"]["n_layer"],
                      n_head=cfg["model"]["n_head"], n_embd=n_embd)
    ckpt = os.path.join(cfg["paths"]["ckpt_dir"], f"gpt_synthetic_{profile}.ckpt")
    sd = torch.load(ckpt, map_location="cpu")

    # GPTforProbeIA: staged forward (1st stage to layer_s, then block-by-block)
    model_ia = GPTforProbeIA(mconf, probe_layer=layer_s)
    model_ia.load_state_dict(sd)
    model_ia = model_ia.to(device).eval()

    # --- 1. train a probe at each layer in [layer_s, layer_e) on our model ---
    plain = GPT(mconf); plain.load_state_dict(sd); plain = plain.to(device).eval()
    val_games = common.get_val_games(cfg, profile)
    layers = list(range(layer_s, layer_e))
    A, Y = extract(plain, val_games, cfg["probe"]["max_probe_games"], layers, device)
    probes = {}
    for L in layers:
        probes[L] = train_layer_probe(A[L], Y, ic["mid_dim"], ic["probe_epochs"],
                                      cfg["probe"]["batch_size"], device, n_embd)
        print(f"[intervene] trained probe at layer {L}", flush=True)
    del A, Y

    # --- 2-3. benchmark loop ---
    with open(os.path.join(cfg["paths"]["upstream"], "intervention_benchmark.pkl"), "rb") as f:
        bench = pickle.load(f)
    htd = {"lr": ic["lr"], "steps": ic["steps"], "reg_strg": ic["reg_strength"]}
    n_cases = min(ic["n_cases"], len(bench))

    intv_errs, base_errs = [], []
    for case in tqdm(bench[:n_cases], desc="intervention benchmark", mininterval=30):
        history = case["history"]
        pos_int = case["pos_int"]
        ori_color = case["ori_color"]
        intervention_to = 2 - ori_color
        wtd = {"intervention_position": permit_reverse(pos_int),
               "intervention_from": ori_color, "intervention_to": intervention_to}

        g = torch.tensor([common.STOI[s] for s in history], dtype=torch.long, device=device)

        # post-intervention ground-truth legal moves
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

        # intervened: edit at layer_s, propagate through layers, re-intervene
        whole = model_ia.forward_1st_stage(g[None, :])      # [1, T, F]
        p = probes[layer_s]
        mid = whole[0, -1]
        labels = p(mid[None, :])[0].squeeze(0).argmax(dim=-1)
        new_mid = intervene_step(p, mid.clone(), labels, wtd, htd, plot=False)
        whole = whole.clone(); whole[0, -1] = new_mid
        for L in range(layer_s, layer_e - 1):
            whole = model_ia.forward_2nd_stage(whole, L, L + 1)[0]   # [1, T, F]
            p = probes[L + 1]
            mid = whole[0, -1]
            labels = p(mid[None, :])[0].squeeze(0).argmax(dim=-1)
            new_mid = intervene_step(p, mid.clone(), labels, wtd, htd, plot=False)
            whole = whole.clone(); whole[0, -1] = new_mid
        with torch.no_grad():
            intv_logits, _ = model_ia.predict(whole)
        intv_errs.append(legality_error(intv_logits[0, -1], true_legal))

    res = {
        "profile": profile, "n_cases": n_cases,
        "intervened_error_mean": float(np.mean(intv_errs)),
        "baseline_error_mean": float(np.mean(base_errs)),
        "intervened_error_std": float(np.std(intv_errs)),
        "baseline_error_std": float(np.std(base_errs)),
        "layer_s": layer_s, "layer_e": layer_e, "metric": "top-N illegal count",
    }
    out = os.path.join(cfg["paths"]["results_dir"], f"intervention_{profile}.json")
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[intervene] intervened err {res['intervened_error_mean']:.3f} vs "
          f"baseline {res['baseline_error_mean']:.3f}  (paper: ~0.12 vs ~2.68)",
          flush=True)
    print(f"[intervene] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
