"""Figures for the data-scaling sweep.

Reuses the palette/styling helpers from ../../src/report_figs.py so these match
the figures already in REPORT.md (light + dark, served via <picture>).

    python sweep_figs.py --results ../../results --ood ../results \
        --out ../results/figs
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))
import report_figs as RF

SIZES = [2000000, 632000, 200000, 63000, 20000]
LABELS = {2000000: "2M", 632000: "632k", 200000: "200k", 63000: "63k",
          20000: "20k"}


def load(res, ood, prefix="sweep_n"):
    out = {}
    for n in SIZES:
        p = json.load(open(f"{res}/nanda_probe_{prefix}{n}.json"))
        e = json.load(open(f"{res}/gpt_{prefix}{n}_eval.json"))
        o = json.load(open(f"{ood}/ood_eval_{prefix}{n}.json"))
        my = np.array([p["linear_mine_yours"][str(L)] for L in range(1, 9)]) * 100
        ab = np.array([p["linear_absolute"][str(L)] for L in range(1, 9)]) * 100
        tv = float(np.mean([o["orientation"][k]["tv"]
                            for k in ("rot180", "transpose", "antitrans")]))
        out[n] = dict(my=my, ab=ab, task=e["legal_move_error"] * 100,
                      tv=tv,
                      ident=o["orientation"]["identity"]["err"] * 100,
                      unif=o["policy"]["uniform"]["err"] * 100,
                      corner=o["policy"]["corner_first"]["err"] * 100)
    return out


def ramp(t, k, n):
    """k-th of n colours along the sequential ramp, skipping the pale end."""
    seq = t["seq"]
    idx = np.linspace(len(seq) - 1, 1, n)[k]
    lo, hi = int(np.floor(idx)), int(np.ceil(idx))
    if lo == hi:
        return seq[lo]
    f = idx - lo
    c1 = np.array([int(seq[lo][i:i+2], 16) for i in (1, 3, 5)])
    c2 = np.array([int(seq[hi][i:i+2], 16) for i in (1, 3, 5)])
    c = (1 - f) * c1 + f * c2
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in c)


def fig_layers(D, out, tag):
    """THE result: where in the stack the board is best decodable, by data size."""
    t = RF.T
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    L = np.arange(1, 9)
    for k, n in enumerate(SIZES):
        c = ramp(t, k, len(SIZES))
        ax.plot(L, D[n]["my"], "-o", color=c, label=f"{LABELS[n]} games",
                zorder=3 - 0.1 * k)
        b = int(np.argmin(D[n]["my"]))
        ax.plot([L[b]], [D[n]["my"][b]], "o", ms=13, mfc="none", mec=c, mew=2.2,
                zorder=4)
        ax.annotate(f"L{b+1}", xy=(L[b], D[n]["my"][b]), xytext=(0, -16),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=c, fontweight="bold")
    ax.set_xlabel("layer")
    ax.set_ylabel("linear mine/yours probe error (%)")
    ax.set_xticks(L)
    ax.grid(True, axis="y", alpha=0.5)
    ax.set_axisbelow(True)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color(t["text2"])
    RF.title(ax, "The world model retreats toward the embedding as data shrinks",
             "Circled point = best layer. At 2M the board is built through the stack; "
             "below ~200k games the deeper layers destroy it.")
    RF.save(fig, out, f"figs1_layer_inversion_{tag}")


def fig_summary(D, out, tag):
    t = RF.T
    x = np.array(SIZES, dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))

    ax = axes[0]
    ax.plot(x, [D[n]["task"] for n in SIZES], "-o", color=t["s1"],
            label="legal-move error")
    ax.plot(x, [min(D[n]["my"]) for n in SIZES], "-s", color=t["s2"],
            label="best linear mine/yours")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("unique training games"); ax.set_ylabel("error (%)")
    leg = ax.legend(frameon=False, fontsize=9)
    for tx in leg.get_texts():
        tx.set_color(t["text2"])
    ax.set_title("task vs representation", loc="left", pad=8)

    ax = axes[1]
    gain = [min(D[n]["ab"]) / min(D[n]["my"]) for n in SIZES]
    ax.plot(x, gain, "-o", color=t["s1"])
    ax.axhline(1.0, color=t["muted"], lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("unique training games")
    ax.set_ylabel("absolute err / mine-yours err")
    ax.set_title("the linearity advantage collapses", loc="left", pad=8)
    ax.annotate("no advantage", xy=(x[-1], 1.0), xytext=(0, 8),
                textcoords="offset points", fontsize=8.5, color=t["muted"])

    ax = axes[2]
    ax.plot(x, [D[n]["tv"] for n in SIZES], "-o", color=t["s1"],
            label="equivariance TV")
    ax.set_xscale("log")
    ax.set_xlabel("unique training games")
    ax.set_ylabel("mean TV (0 = equivariant)")
    ax.set_title("orientation consistency", loc="left", pad=8)

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.45)
        ax.set_axisbelow(True)
    fig.suptitle("Less data buys a worse world model, not a simpler one",
                 x=0.077, y=1.10, ha="left", fontsize=13.5,
                 fontweight="semibold", color=t["text"])
    fig.text(0.077, 1.028,
             "All five points train for identical optimiser steps (40M "
             "presentations, batch 1024, weight decay 0) on a fixed val split.",
             ha="left", fontsize=9.5, color=t["text2"])
    RF.save(fig, out, f"figs2_sweep_summary_{tag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../../results")
    ap.add_argument("--ood", default="../results")
    ap.add_argument("--out", default="../results/figs")
    ap.add_argument("--themes", default="light,dark")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    D = load(a.results, a.ood)
    for tag in a.themes.split(","):
        RF.T = RF.THEMES[tag]
        RF.style(RF.T)
        fig_layers(D, a.out, tag)
        fig_summary(D, a.out, tag)
        print(f"[sweepfigs] wrote {tag}", flush=True)
    print("[sweepfigs] done ->", a.out)
