"""Anchor check: run our eval + probe harness on the AUTHORS' released weights.

The released `synthetic_model.pth` (HF: NeelNanda/Othello-GPT-Transformer-Lens)
is in TransformerLens format with LayerNorm folded in. We convert it back to
the minGPT GPT layout so it loads into the exact architecture we train, then:
  1. measure legal-move error (paper: ~0.01%) -- this DOUBLES as a correctness
     check on the weight conversion: a wrong remap gives a high error.
  2. run the linear/nonlinear board probes (paper: linear ~20-27%, nonlin ~1.8%).

This is an independent anchor: if our from-scratch numbers later drift, this
tells us whether the harness or the training is at fault. No internet needed
(weights are pre-downloaded to ckpts/pretrained/ on the login node).

    python validate_pretrained.py --config ../configs/config.yaml --profile pilot
"""
import os
import json
import argparse

import torch

import common
from train_gpt import evaluate_legal_move_error
from train_probe import extract, train_one_probe


def convert_tl_to_mingpt(sd, n_layer, d_model):
    """TransformerLens (folded-LN) state dict -> minGPT GPT state dict.

    Folded LN means LN has no learnable scale/bias, so we set minGPT's LN
    weight=1, bias=0 (i.e. LayerNormPre). Attention heads are concatenated
    head-major (out index = head*d_head + d), matching minGPT's
    `.view(B,T,n_head, C//n_head)`.
    """
    new = {}
    new["tok_emb.weight"] = sd["embed.W_E"].clone()
    new["pos_emb"] = sd["pos_embed.W_pos"].unsqueeze(0).clone()
    for i in range(n_layer):
        p = f"blocks.{i}."
        new[p + "ln1.weight"] = torch.ones(d_model)
        new[p + "ln1.bias"] = torch.zeros(d_model)
        new[p + "ln2.weight"] = torch.ones(d_model)
        new[p + "ln2.bias"] = torch.zeros(d_model)
        # W_Q[head, in, d] -> query.weight[head*d + d, in]
        new[p + "attn.query.weight"] = sd[p + "attn.W_Q"].permute(0, 2, 1).reshape(d_model, d_model).contiguous()
        new[p + "attn.query.bias"] = sd[p + "attn.b_Q"].reshape(-1).clone()
        new[p + "attn.key.weight"] = sd[p + "attn.W_K"].permute(0, 2, 1).reshape(d_model, d_model).contiguous()
        new[p + "attn.key.bias"] = sd[p + "attn.b_K"].reshape(-1).clone()
        new[p + "attn.value.weight"] = sd[p + "attn.W_V"].permute(0, 2, 1).reshape(d_model, d_model).contiguous()
        new[p + "attn.value.bias"] = sd[p + "attn.b_V"].reshape(-1).clone()
        # W_O[head, d, out] -> proj.weight[out, head*d + d]
        new[p + "attn.proj.weight"] = sd[p + "attn.W_O"].permute(2, 0, 1).reshape(d_model, d_model).contiguous()
        new[p + "attn.proj.bias"] = sd[p + "attn.b_O"].clone()
        new[p + "mlp.0.weight"] = sd[p + "mlp.W_in"].t().contiguous()
        new[p + "mlp.0.bias"] = sd[p + "mlp.b_in"].clone()
        new[p + "mlp.2.weight"] = sd[p + "mlp.W_out"].t().contiguous()
        new[p + "mlp.2.bias"] = sd[p + "mlp.b_out"].clone()
    new["ln_f.weight"] = torch.ones(d_model)
    new["ln_f.bias"] = torch.zeros(d_model)
    new["head.weight"] = sd["unembed.W_U"].t().contiguous()
    head_bias = sd["unembed.b_U"].clone()   # minGPT head is bias-free; attach separately
    return new, head_bias


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default=None)
    ap.add_argument("--pth", default=None, help="TransformerLens .pth weights")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    if args.profile:
        cfg["profile"] = args.profile
    common.add_upstream_to_path(cfg)
    profile = cfg["profile"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from mingpt.model import GPT, GPTConfig
    import torch.nn as nn
    from mingpt.probe_model import (BatteryProbeClassification,
                                    BatteryProbeClassificationTwoLayer)

    pth = args.pth or os.path.join(cfg["paths"]["ckpt_dir"], "pretrained",
                                   "synthetic_model.pth")
    print(f"[validate] loading {pth}", flush=True)
    sd = torch.load(pth, map_location="cpu", weights_only=True)

    d_model = cfg["model"]["n_embd"]
    n_layer = cfg["model"]["n_layer"]
    body, head_bias = convert_tl_to_mingpt(sd, n_layer, d_model)

    model = GPT(GPTConfig(common.VOCAB_SIZE, common.BLOCK_SIZE,
                          n_layer=n_layer, n_head=cfg["model"]["n_head"],
                          n_embd=d_model))
    missing, unexpected = model.load_state_dict(body, strict=False)
    # only the attn causal-mask buffers should be "missing" (they're recreated)
    missing = [m for m in missing if not m.endswith("attn.mask")]
    assert not missing, f"unexpected missing keys: {missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    # attach the unembed bias the bias-free minGPT head can't hold
    new_head = nn.Linear(d_model, common.VOCAB_SIZE, bias=True)
    with torch.no_grad():
        new_head.weight.copy_(model.head.weight)
        new_head.bias.copy_(head_bias)
    model.head = new_head
    model = model.to(device)

    # --- 1. legal-move error (also validates the conversion) ---
    games = common.load_all_games(cfg["paths"]["data_dir"], profile)
    _, val_games = common.split_train_val(games, cfg["data"]["val_games"])
    ds = common.make_dataset(val_games[:50])  # only needs stoi/itos
    err, ok, total = evaluate_legal_move_error(
        model, ds, val_games, cfg["eval"]["n_val_games"], device)
    print(f"[validate] LEGAL-MOVE ERROR = {err*100:.4f}%  ({ok}/{total})", flush=True)
    if err > 0.05:
        print("[validate] WARNING: error >5% -- weight conversion is likely wrong!",
              flush=True)

    # --- 2. probes ---
    layers = cfg["probe"]["layers"]
    A, Y = extract(model, val_games, cfg["probe"]["max_probe_games"], layers, device)
    res = {"source": "authors_pretrained_synthetic", "legal_move_error": err,
           "linear": {}, "nonlinear": {}}
    for L in layers:
        lin = BatteryProbeClassification(device, 3, 64, input_dim=d_model)
        res["linear"][L] = train_one_probe(A[L], Y, lin, cfg["probe"]["epochs"],
                                           cfg["probe"]["batch_size"], device)
        nl = BatteryProbeClassificationTwoLayer(device, 3, 64,
                                                mid_dim=cfg["probe"]["mid_dim"],
                                                input_dim=d_model)
        res["nonlinear"][L] = train_one_probe(A[L], Y, nl, cfg["probe"]["epochs"],
                                              cfg["probe"]["batch_size"], device)
        print(f"[validate] layer {L}: linear {res['linear'][L]*100:.2f}%  "
              f"nonlinear {res['nonlinear'][L]*100:.2f}%", flush=True)

    out = os.path.join(cfg["paths"]["results_dir"], "validate_pretrained.json")
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[validate] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
