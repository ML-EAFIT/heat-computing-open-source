#!/usr/bin/env python3
"""Plot RTL sampled histograms against the exact Boltzmann law.

Reads build/stats.json written by compare.py and writes, per case,
<out>/hist_<case>.png: left, state probabilities sorted by exact probability
(exact = line, RTL = dots, golden = small dots, log y); right, the
magnetisation distribution (RTL vs exact).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

# reference categorical palette, slots 1-3 (light mode)
C_EXACT, C_RTL, C_GOLD = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_case(r, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw=dict(width_ratios=[1.6, 1]))
    title = (f"{r['case']}  N={r['n']}  Q{7 - r['wfrac']}.{r['wfrac']}  beta={r['beta_q']:.4f}  "
             f"{r['samples']} RTL samples")
    fig.suptitle(title, color=INK, fontsize=10, x=0.01, ha="left")
    if r["hist_rtl"] is not None:
        ex = np.array(r["exact_q"])
        order = np.argsort(-ex)
        x = np.arange(len(ex))
        a1.plot(x, ex[order], color=C_EXACT, linewidth=2, label="exact Boltzmann (programmed params)")
        rt = np.array(r["hist_rtl"])[order]
        gl = np.array(r["golden"])[order]
        m = rt > 0
        a1.scatter(x[m], rt[m], s=10, color=C_RTL, edgecolors="white", linewidths=0.4, label="RTL (Verilator)", zorder=3)
        mg = gl > 0
        a1.scatter(x[mg], gl[mg], s=4, color=C_GOLD, label="golden float model", zorder=2)
        a1.set_yscale("log")
        a1.set_xlabel("state, sorted by exact probability", color=MUTED, fontsize=8)
        a1.set_ylabel("probability", color=MUTED, fontsize=8)
        a1.set_title(f"TV(RTL, exact) = {r['tv_rtl_exact_q']:.4f}   KL = {r['kl_rtl_exact_q']:.4f}   "
                     f"iid floor {r['tv_iid_floor']:.4f}", fontsize=8, color=INK, loc="left")
        a1.legend(fontsize=7, frameon=False, labelcolor=INK)
    else:
        a1.text(0.5, 0.5, f"2^{r['n']} states: full histogram not plotted\n(see magnetisation panel)",
                ha="center", va="center", color=MUTED, fontsize=9, transform=a1.transAxes)
        a1.set_axis_off()
    k = np.arange(r["n"] + 1)
    w = 0.38
    a2.bar(k - w / 2 - 0.02, r["mag_exact_q"], width=w, color=C_EXACT, label="exact")
    a2.bar(k + w / 2 + 0.02, r["mag_rtl"], width=w, color=C_RTL, label="RTL")
    a2.set_xlabel("number of +1 spins", color=MUTED, fontsize=8)
    a2.set_ylabel("probability", color=MUTED, fontsize=8)
    a2.set_title(f"magnetisation TV = {r['mag_tv_rtl_exact_q']:.4f}", fontsize=8, color=INK, loc="left")
    a2.legend(fontsize=7, frameon=False, labelcolor=INK)
    for a in (a1, a2):
        style(a)
    fig.tight_layout()
    path = os.path.join(out, f"hist_{r['case']}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"plot_hist: wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="build/stats.json")
    ap.add_argument("--out", default="build")
    a = ap.parse_args()
    with open(a.stats) as f:
        for r in json.load(f):
            plot_case(r, a.out)


if __name__ == "__main__":
    main()
