"""Generate summary figures from the result JSONs.

  - probe error vs layer (linear vs nonlinear), trained model + pretrained anchor
  - intervention: intervened vs baseline legality error

    python plot_results.py --config ../configs/config.yaml --profile pilot
"""
import os
import json
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common


def _load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--profile", default=None)
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    if args.profile:
        cfg["profile"] = args.profile
    profile = cfg["profile"]
    R = cfg["paths"]["results_dir"]
    os.makedirs(R, exist_ok=True)

    trained = _load(os.path.join(R, f"probe_{profile}.json"))
    anchor = _load(os.path.join(R, "validate_pretrained.json"))

    # --- Fig 1: probe error vs layer ---
    if trained:
        fig, ax = plt.subplots(figsize=(7, 5))
        layers = sorted(int(k) for k in trained["linear"])
        lin = [trained["linear"][str(L)] * 100 for L in layers]
        nl = [trained["nonlinear"][str(L)] * 100 for L in layers]
        ax.plot(layers, lin, "o-", label="linear probe (ours)", color="C0")
        ax.plot(layers, nl, "s-", label="nonlinear probe (ours)", color="C1")
        if anchor:
            al = [anchor["linear"][str(L)] * 100 for L in layers]
            an = [anchor["nonlinear"][str(L)] * 100 for L in layers]
            ax.plot(layers, al, "o--", label="linear (authors' ckpt)", color="C0", alpha=0.5)
            ax.plot(layers, an, "s--", label="nonlinear (authors' ckpt)", color="C1", alpha=0.5)
        ax.axhspan(20, 27, color="C0", alpha=0.08, label="paper linear ~20-27%")
        ax.axhline(1.8, color="C1", ls=":", alpha=0.6, label="paper nonlinear ~1.8%")
        ax.set_xlabel("transformer layer"); ax.set_ylabel("board-state error (%)")
        ax.set_title(f"Othello-GPT board probe ({profile})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        f1 = os.path.join(R, f"probe_error_vs_layer_{profile}.png")
        fig.tight_layout(); fig.savefig(f1, dpi=140); print("wrote", f1)

    # --- Fig 2: intervention ---
    intv = _load(os.path.join(R, f"intervention_{profile}.json"))
    if intv:
        fig, ax = plt.subplots(figsize=(5, 5))
        bars = ax.bar(["baseline\n(no edit)", "intervened"],
                      [intv["baseline_error_mean"], intv["intervened_error_mean"]],
                      color=["C3", "C2"])
        ax.bar_label(bars, fmt="%.2f")
        ax.axhline(2.68, color="C3", ls=":", alpha=0.6)
        ax.axhline(0.12, color="C2", ls=":", alpha=0.6)
        ax.set_ylabel("legality error (top-N illegal count)")
        ax.set_title(f"Causal intervention ({profile})\npaper: 2.68 -> 0.12")
        f2 = os.path.join(R, f"intervention_{profile}.png")
        fig.tight_layout(); fig.savefig(f2, dpi=140); print("wrote", f2)


if __name__ == "__main__":
    main()
