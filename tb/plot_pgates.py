#!/usr/bin/env python3
"""Plot the p-gate truth-table statistics written by compare.py --pgates.

Per case, <out>/pgate_<case>.png: one bar group per clamp scenario, showing
the probability of every (inputs | outputs) row, RTL (bars) vs exact law of
the quantized kernel (ticks). Rows of the Boolean function are coloured;
invalid rows are grey. factor6 is shown as a = b grid per product instead.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

from plot_hist import C_EXACT, C_RTL, GRID, INK, MUTED, style  # noqa: E402

C_BAD = "#c9c7bf"


def png_name(case):
    return (case if case.startswith("pgate_") else f"pgate_{case}") + ".png"


def row_label(r, nin):
    return f"{r[:nin]}|{r[nin:]}"


def plot_gate(r, out, valid_rows):
    sc = r["scenarios"]
    nin, nw = r["nin"], len(r["wires"])
    rows = [format(k, f"0{nw}b")[::-1] for k in range(1 << nw)] if nw <= 5 else None
    show = sc if len(sc) <= 7 else [s for s in sc if s["label"] == "free" or s["label"].startswith("bwd")]
    fig, axes = plt.subplots(1, len(show), figsize=(2.3 * len(show) + 0.6, 3.2), sharey=True, squeeze=False)
    fig.suptitle(f"{r['case']}  rows ({','.join(r['wires'][:nin])} | {','.join(r['wires'][nin:])})  "
                 f"beta={r['beta']:.4f}   bars = RTL, ticks = exact", color=INK, fontsize=10, x=0.01, ha="left")
    for ax, s in zip(axes[0], show):
        keys = rows if rows else sorted(set(s["emp"]) | set(s["ref"]), key=lambda k: -s["ref"].get(k, 0))[:12]
        keys = [k for k in keys if s["emp"].get(k, 0) > 0.002 or s["ref"].get(k, 0) > 0.002] or keys[:1]
        x = np.arange(len(keys))
        emp = [s["emp"].get(k, 0) for k in keys]
        ref = [s["ref"].get(k, 0) for k in keys]
        col = [C_RTL if tuple(int(c) for c in k) in valid_rows else C_BAD for k in keys]
        ax.bar(x, emp, color=col, width=0.7)
        ax.scatter(x, ref, marker="_", s=180, color=C_EXACT, linewidths=2, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([row_label(k, nin) for k in keys], rotation=90, fontsize=7, family="monospace")
        ax.set_title(f"{s['label']}  P(ok)={s['p_ok']:.2f}", fontsize=8, color=INK)
        ax.set_ylim(0, 1.0)
        style(ax)
    axes[0][0].set_ylabel("probability", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, png_name(r["case"])), dpi=130)
    plt.close(fig)


def plot_factor(r, out):
    sc = r["scenarios"]
    fig, axes = plt.subplots(1, len(sc), figsize=(2.6 * len(sc) + 0.6, 3.0), squeeze=False)
    fig.suptitle(f"factor6: 2x2-bit multiplier run backwards, beta={r['beta']:.4f}. "
                 "Cell = P(a, b) sampled by the RTL with the product clamped", color=INK, fontsize=10, x=0.01,
                 ha="left")
    for ax, s in zip(axes[0], sc):
        grid = np.zeros((4, 4))
        for k, v in s["emp"].items():
            a = int(k[0]) + 2 * int(k[1])
            b = int(k[2]) + 2 * int(k[3])
            grid[a, b] += v
        ax.imshow(grid, cmap="Oranges", vmin=0, vmax=1)
        for a in range(4):
            for b in range(4):
                if grid[a, b] > 0.01:
                    ax.text(b, a, f"{grid[a, b]:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if grid[a, b] > 0.5 else INK)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xlabel("b", color=MUTED, fontsize=8)
        ax.set_ylabel("a", color=MUTED, fontsize=8)
        ax.set_title(f"{s['label']}: P(a*b={s['label'].split()[1]})={s['p_ok']:.2f}", fontsize=8, color=INK)
        ax.tick_params(colors=MUTED, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color(GRID)
    fig.tight_layout()
    fig.savefig(os.path.join(out, png_name(r["case"])), dpi=130)
    plt.close(fig)


def main():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
    import cases as cs
    import pgates as pg
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="build/pgates.json")
    ap.add_argument("--out", default="build")
    a = ap.parse_args()
    by = {c["name"]: c for c in cs.all_cases()}
    for r in json.load(open(a.stats)):
        if r["case"] == "factor6":
            plot_factor(r, a.out)
        else:
            plot_gate(r, a.out, pg.rows_valid(by[r["case"]]))
        print(f"wrote {a.out}/{png_name(r['case'])}")


if __name__ == "__main__":
    main()
