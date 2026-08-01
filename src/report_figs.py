"""Report figures for REPORT.md.

Renders the five headline figures from the existing results/*.json, in both a
light and a dark theme (REPORT.md serves them via <picture>, so GitHub picks
the one matching the reader's theme).

Palette: categorical slots 1-2 of the project's reference palette, validated
with the dataviz validator (light #2a78d6/#eb6834, dark #3987e5/#d95926 --
all checks PASS in both modes, worst CVD dE 24.7/26.8). Sequential heatmap
ramp is the single-hue blue 100->700 from the same palette.

    python report_figs.py --results ../results --out ../results/figs
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

THEMES = {
    "light": dict(
        surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#8a8880",
        grid="#e4e3df", s1="#2a78d6", s2="#eb6834",
        seq=["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
        # ink for cells at the far end of the ramp: light mode ends DARK -> white ink
        cell_hi="#ffffff",
    ),
    "dark": dict(
        surface="#1a1a19", text="#ffffff", text2="#c3c2b7", muted="#8a8880",
        grid="#333331", s1="#3987e5", s2="#d95926",
        seq=["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
        # dark mode ends LIGHT (low values recede into the surface) -> dark ink
        cell_hi="#101010",
    ),
}


def style(t):
    """Recessive axes/grid, text in ink tokens -- never in a series color."""
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "text.color": t["text"],
        "axes.labelcolor": t["text2"], "xtick.color": t["text2"],
        "ytick.color": t["text2"], "axes.edgecolor": t["grid"],
        "grid.color": t["grid"], "font.size": 10.5,
        "axes.titlesize": 12.5, "axes.titleweight": "semibold",
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 160, "savefig.bbox": "tight", "savefig.pad_inches": 0.28,
        "lines.linewidth": 2.0, "lines.markersize": 6.5,
    })


def title(ax, head, sub=None):
    # Offsets in POINTS, not axes fractions -- axes heights differ per figure and
    # a fractional subtitle offset collides with the title on the taller ones.
    ax.set_title(head, loc="left", pad=28 if sub else 8)
    if sub:
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 8),
                    textcoords="offset points", va="bottom", ha="left",
                    fontsize=9.5, color=T["text2"])


def save(fig, out, name):
    for ext in ("png",):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)


# ---------------------------------------------------------------- fig 1
def fig_linearity(R, out, tag):
    d = json.load(open(f"{R}/nanda_probe_full_h200.json"))
    layers = list(range(1, 9))
    my = [100 * d["linear_mine_yours"][str(l)] for l in layers]
    ab = [100 * d["linear_absolute"][str(l)] for l in layers]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_yscale("log")
    ax.grid(axis="y", lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.plot(layers, ab, color=T["s2"], marker="s", label="absolute (black/white)")
    ax.plot(layers, my, color=T["s1"], marker="o", label="mine/yours (player-relative)")

    # direct labels at the right end -- identity is never color-alone
    ax.annotate(f"{ab[-1]:.1f}%", (layers[-1], ab[-1]), xytext=(9, 0),
                textcoords="offset points", va="center", color=T["text"], fontsize=10)
    ax.annotate(f"{my[-1]:.2f}%", (layers[-1], my[-1]), xytext=(9, 0),
                textcoords="offset points", va="center", color=T["text"], fontsize=10)
    ax.annotate("", xy=(7, my[6]), xytext=(7, ab[6]),
                arrowprops=dict(arrowstyle="<->", color=T["muted"], lw=1.2))
    ax.text(6.85, np.sqrt(my[6] * ab[6]), "15.8x", ha="right", va="center",
            color=T["text"], fontsize=11, fontweight="semibold")

    ax.set_xlabel("layer")
    ax.set_ylabel("probe error (%, log scale)")
    ax.set_xticks(layers)
    ax.set_xlim(0.7, 8.9)
    ax.set_yticks([1, 2, 5, 10, 20])
    ax.set_yticklabels(["1", "2", "5", "10", "20"])
    ax.legend(frameon=False, loc="lower left", fontsize=10, labelcolor=T["text2"])
    title(ax, "The board is linear -- in the player-relative frame",
          "Linear probe error by layer, full-scale GELU model")
    save(fig, out, f"fig1_linearity_{tag}")


# ---------------------------------------------------------------- fig 2
def fig_intervention(R, out, tag):
    fl = json.load(open(f"{R}/floors_full_h200.json"))["legality_floor"]
    iv = json.load(open(f"{R}/intervention_nanda_full_h200.json"))
    floor, ceil = fl["informed"], fl["expected"]
    base, post = iv["baseline_error_mean"], iv["intervened_error_mean"]
    closed = 100 * (base - post) / (base - floor)

    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    names = ["model, un-intervened", "model, after board edit"]
    vals = [base, post]
    y = [1, 0]
    ax.barh(y, vals, height=0.44, color=[T["s2"], T["s1"]], zorder=3)
    for yy, v in zip(y, vals):
        ax.text(v + 0.035, yy, f"{v:.2f}", va="center", color=T["text"], fontsize=11,
                fontweight="semibold")

    for x, lab in ((floor, "oracle floor\n(honours the edit)"),
                   (ceil, "oracle ceiling\n(ignores the edit)")):
        ax.axvline(x, color=T["muted"], ls=(0, (4, 3)), lw=1.4, zorder=4)
        ax.text(x, 1.62, f"  {lab}  ", ha="left" if x == floor else "right",
                va="bottom", color=T["text2"], fontsize=9, linespacing=1.35)

    ax.annotate("", xy=(post, -0.52), xytext=(base, -0.52),
                arrowprops=dict(arrowstyle="<->", color=T["text2"], lw=1.2))
    ax.text((post + base) / 2, -0.68, f"closes {closed:.0f}% of the gap",
            ha="center", va="top", color=T["text"], fontsize=10, fontweight="semibold")

    ax.set_yticks(y)
    ax.set_yticklabels(names, color=T["text"])
    ax.set_xlabel("illegal moves in the model's top-N  (lower = follows the edited board)")
    ax.set_xlim(-0.02, 1.72)
    ax.set_ylim(-0.95, 1.95)
    ax.grid(axis="x", lw=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    title(ax, "The board edit is causal -- and now calibrated",
          f"n={iv['n_cases']} cases, probes at L{iv['layer_s']}-L{iv['layer_e']-1}")
    save(fig, out, f"fig2_intervention_{tag}")


# ---------------------------------------------------------------- fig 3
def fig_dev(R, out, tag):
    d = json.load(open(f"{R}/dev_study_full_h200.json"))
    ep = [e["epoch"] for e in d["epochs"]]
    probe = [100 * min(e["probe_myours_err"].values()) for e in d["epochs"]]
    base = [e["intervention_baseline_err"] for e in d["epochs"]]
    post = [e["intervention_intervened_err"] for e in d["epochs"]]

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True,
                                 gridspec_kw=dict(hspace=0.40))
    for ax in (a1, a2):
        ax.set_yscale("log")
        ax.grid(axis="y", lw=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        ax.axvspan(4, 20.6, color=T["muted"], alpha=0.085, lw=0, zorder=0)

    a1.plot(ep, probe, color=T["s1"], marker="o")
    a1.set_ylabel("probe error (%, log)")
    a1.set_yticks([1, 2, 5, 10, 20])
    a1.set_yticklabels(["1", "2", "5", "10", "20"])
    a1.annotate(f"{probe[0]:.1f}%\n(random init)", (ep[0], probe[0]), xytext=(8, -6),
                textcoords="offset points", color=T["text2"], fontsize=9)
    a1.annotate(f"{probe[-1]:.2f}%", (ep[-1], probe[-1]), xytext=(-4, 10),
                textcoords="offset points", color=T["text"], fontsize=10, ha="right")
    title(a1, "Decodability keeps improving...",
          "Best-layer mine/yours probe error over training")

    a2.plot(ep, base, color=T["s2"], marker="s", label="un-intervened")
    a2.plot(ep, post, color=T["s1"], marker="o", label="after board edit")
    a2.set_ylabel("illegal moves (log)")
    a2.set_xlabel("training epoch")
    a2.set_yticks([0.5, 1, 2, 5, 10])
    a2.set_yticklabels(["0.5", "1", "2", "5", "10"])
    a2.set_xticks(ep)
    a2.legend(frameon=False, fontsize=10, labelcolor=T["text2"], loc="upper right")
    title(a2, "...but causal control is done by epoch 4",
          "Intervention effect over training (shaded: no further gain)")
    save(fig, out, f"fig3_devtrace_{tag}")


# ---------------------------------------------------------------- fig 4
def fig_transfer(R, out, tag):
    d = json.load(open(f"{R}/probe_transfer_full_h200.json"))
    ep = d["epochs"]
    M = np.array(d["by_layer"]["7"]["cosine"])
    ctl = d["by_layer"]["7"]["control"]["20"]["seed_cosine_mean"]

    cmap = LinearSegmentedColormap.from_list("seq", T["seq"])
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, origin="lower")
    for i in range(len(ep)):
        for j in range(len(ep)):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}"[1:] if v < 1 else "1", ha="center", va="center",
                    fontsize=7.4, color=T["cell_hi"] if v > 0.62 else T["text"])
    ax.set_xticks(range(len(ep)), ep)
    ax.set_yticks(range(len(ep)), ep)
    ax.set_xlabel("activations from epoch")
    ax.set_ylabel("probe trained on epoch")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cb.set_label("mean cosine between probe directions", color=T["text2"])
    cb.outline.set_visible(False)
    title(ax, "The board directions rotate for the whole run",
          f"Layer 7. Same-epoch, different-seed control: {ctl:.3f}")
    save(fig, out, f"fig4_transfer_{tag}")


# ---------------------------------------------------------------- fig 5
def fig_where(R, out, tag):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), sharey=True,
                             gridspec_kw=dict(wspace=0.08))
    for ax, (arch, fn) in zip(axes, [("GELU", "board_map_full.json"),
                                     ("bilinear", "board_map_full_bilinear.json")]):
        comps = json.load(open(f"{R}/{fn}"))["components"]
        heads = [(int(k[1]), v["board_corr"]) for k, v in comps.items() if k.startswith("A")]
        mlps = [(int(k[1]), v["board_corr"]) for k, v in comps.items() if k.startswith("M")]
        rng = np.random.default_rng(0)
        hx = [l + rng.uniform(-0.22, 0.22) for l, _ in heads]
        ax.grid(axis="y", lw=0.8, alpha=0.9)
        ax.set_axisbelow(True)
        ax.axhline(0.3, color=T["muted"], ls=(0, (4, 3)), lw=1.3)
        ax.scatter(hx, [c for _, c in heads], s=46, color=T["s1"], marker="o",
                   alpha=0.85, edgecolor=T["surface"], linewidth=0.8,
                   label="attention head", zorder=3)
        ax.scatter([l for l, _ in mlps], [c for _, c in mlps], s=95, color=T["s2"],
                   marker="D", edgecolor=T["surface"], linewidth=0.9,
                   label="MLP", zorder=4)
        n_hi = sum(c > 0.3 for _, c in heads)
        ax.set_title(f"{arch}   -   {n_hi}/64 heads above 0.3, 0/8 MLPs",
                     loc="left", fontsize=10.5, color=T["text2"], pad=6)
        ax.set_xlabel("layer")
        ax.set_xticks(range(1, 9))
        ax.set_xlim(0.4, 8.6)
    axes[0].set_ylabel("correlation with the board readout")
    axes[0].text(0.55, 0.315, "board_corr = 0.3", color=T["text2"], fontsize=9)
    axes[0].legend(frameon=False, fontsize=10, labelcolor=T["text2"], loc="upper left")
    fig.suptitle("The world model lives in attention, not in MLPs", x=0.125, y=1.015,
                 ha="left", fontsize=12.5, fontweight="semibold", color=T["text"])
    fig.text(0.125, 0.965, "Contribution of every component to the linear board readout "
             "(exact decomposition, reconstruction corr 1.000)",
             ha="left", fontsize=9.5, color=T["text2"])
    save(fig, out, f"fig5_where_{tag}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="../results")
    p.add_argument("--out", default="../results/figs")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for tag, theme in THEMES.items():
        T = theme
        globals()["T"] = theme
        style(theme)
        fig_linearity(a.results, a.out, tag)
        fig_intervention(a.results, a.out, tag)
        fig_dev(a.results, a.out, tag)
        fig_transfer(a.results, a.out, tag)
        fig_where(a.results, a.out, tag)
        print(f"[figs] wrote {tag} theme")
    print("[figs] done ->", a.out)
