#!/usr/bin/env python3
"""Statistics of the continuous p-mode cell (pmode_array.v) against exact densities.

  pmode_study.py --rtl pm_well=build/pm_well_rtl_trace.hex pm_pair=... --out build/pmode.json

For each RTL trace (Verilator, long run):
  * the trace must equal model/pmode.PmodeArray step for step (bit-exact);
  * the histogram of x (bins 1/16) is compared with
      exact : pi(x) ~ exp(-V(x)/T) by quadrature (1 or 2 sites),
      ULA   : float overdamped Langevin with the same eps (torx LangevinGate,
              metropolis=False), many chains -- it carries the same O(eps) bias,
    so TV(rtl, ULA) isolates the fixed-point / LUT effects and TV(ULA, exact)
    the step-size bias;
  * moments: <x^2>, <|x|>, P(x > 0); for the pair also <sign x0 sign x1>.
Then an eps scan (k = 3..7, eps = 2^-k) of the double well with the bit-exact
model shows how the step-size bias shrinks.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
import export_vectors as ev  # noqa: E402
import fixed as fx           # noqa: E402
import pmode as pm           # noqa: E402

BURN = 2000
BINS = np.arange(-4.0, 4.0 + 1e-9, 1 / 16)


def read_trace(path, m):
    with open(path) as f:
        rows = [int(x, 16) for x in f.read().split()]
    x = np.array([[(r >> (16 * i)) & 0xFFFF for i in range(m)] for r in rows], dtype=np.int64)
    return np.where(x >= 32768, x - 65536, x)


def model_trace(c, x0, steps):
    arr = pm.PmodeArray(c["Aq"], c["hq"], c["lamq"], c["k"], c["sig"], x0=x0)
    out = np.empty((steps, c["m"]), dtype=np.int64)
    for t in range(steps):
        arr.step()
        out[t] = arr.x
    return out


def real_params(c):
    """The parameters the hardware actually holds (quantized), in real units."""
    A = np.array(c["Aq"], float) / 16
    h = np.array(c["hq"], float) / 16
    lam = c["lamq"] / 16
    eps = 2.0 ** -c["k"]
    T = (c["sig"] / 4096) ** 2 / (2 * eps)
    return A, h, lam, eps, T


def exact_grid(c, centers):
    A, h, lam, _, T = real_params(c)
    if c["m"] == 1:
        p = np.exp(-pm.potential(centers[:, None], A, h, lam) / T)
        return p / p.sum()
    X0, X1 = np.meshgrid(centers, centers, indexing="ij")
    p = np.exp(-pm.potential(np.stack([X0, X1], -1), A, h, lam) / T)
    return p / p.sum()


def hist1(x):
    hh, _ = np.histogram(x, BINS)
    return hh / max(1, hh.sum())


def moments(x):
    r = dict(x2=float(np.mean(x[:, 0] ** 2)), absx=float(np.mean(np.abs(x[:, 0]))), ppos=float(np.mean(x[:, 0] > 0)))
    if x.shape[1] > 1:
        r["sgn01"] = float(np.mean(np.sign(x[:, 0]) * np.sign(x[:, 1])))
    return r


def analyse(c, xr, rng):
    centers = 0.5 * (BINS[1:] + BINS[:-1])
    A, h, lam, eps, T = real_params(c)
    p_ex = exact_grid(c, centers)
    p_ex1 = p_ex if c["m"] == 1 else p_ex.sum(1)
    ula = pm.float_ula(A, h, lam, eps, T, 3000, 2000, rng)[500:].reshape(-1, c["m"])
    r = dict(case=c["name"], steps=len(xr), eps=eps, T=T,
             tv_rtl_exact=0.5 * float(np.abs(hist1(xr[:, 0]) - p_ex1).sum()),
             tv_ula_exact=0.5 * float(np.abs(hist1(ula[:, 0]) - p_ex1).sum()),
             tv_rtl_ula=0.5 * float(np.abs(hist1(xr[:, 0]) - hist1(ula[:, 0])).sum()),
             rtl=moments(xr), ula=moments(ula))
    # exact moments of site 0 by quadrature
    if c["m"] == 1:
        w, x0g, x1g = p_ex1, centers, None
    else:
        X0, X1 = np.meshgrid(centers, centers, indexing="ij")
        w, x0g, x1g = p_ex.ravel(), X0.ravel(), X1.ravel()
    ex = dict(x2=float(w @ x0g ** 2), absx=float(w @ np.abs(x0g)), ppos=float(w @ (x0g > 0)))
    if x1g is not None:
        ex["sgn01"] = float(w @ (np.sign(x0g) * np.sign(x1g)))
    r["exact"] = ex
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtl", nargs="*", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--scan-steps", type=int, default=200000)
    a = ap.parse_args()
    rng = np.random.default_rng(11)
    rc, out = 0, dict(rtl=[], scan=[])
    print("1) RTL traces vs bit-exact model, exact density and float ULA (same eps)")
    for spec in a.rtl:
        name, path = spec.split("=")
        c = pm.case(name[3:])
        x0 = [fx.to_q(v, pm.XF, 16) for v in ev.PM_X0[c["name"]]]
        xr = read_trace(path, c["m"])
        ref = model_trace(c, x0, len(xr))
        mism = int(np.sum(np.any(xr != ref, axis=1)))
        rc |= int(mism != 0)
        r = analyse(c, xr[BURN:] / 4096, rng)
        r["mismatches"] = mism
        out["rtl"].append(r)
        print(f"  {name:10s} {len(xr)} steps, eps = {r['eps']:.4f}, T = {r['T']:.3f}; steps differing from model: {mism}")
        print(f"    TV(rtl, exact) {r['tv_rtl_exact']:.4f}   TV(ULA, exact) {r['tv_ula_exact']:.4f}   "
              f"TV(rtl, ULA) {r['tv_rtl_ula']:.4f}")
        keys = list(r["exact"])
        print("    " + "  ".join(f"{k}: rtl {r['rtl'][k]:+.4f} ULA {r['ula'][k]:+.4f} exact {r['exact'][k]:+.4f}"
                                for k in keys))
    print(f"2) step-size scan, double well, bit-exact model, {a.scan_steps} steps per eps")
    print(f"  {'eps':>8s} {'TV(fixed, exact)':>16s} {'TV(ULA, exact)':>14s} {'<x^2> fixed':>11s} {'<x^2> exact':>11s}")
    base = pm.case("well")
    for k in (3, 4, 5, 6, 7):
        eps = 2.0 ** -k
        c = dict(base, k=k, eps=eps, sig=fx.to_q(math.sqrt(2 * base["T"] * eps), pm.XF, 16, signed=False))
        xr = model_trace(c, [0], a.scan_steps)[BURN:] / 4096
        r = analyse(c, xr, rng)
        out["scan"].append(dict(k=k, **{kk: r[kk] for kk in ("eps", "tv_rtl_exact", "tv_ula_exact", "rtl", "ula", "exact")}))
        unstable = r["rtl"]["x2"] > 10
        print(f"  {eps:8.4f} {r['tv_rtl_exact']:16.4f} {r['tv_ula_exact']:14.4f} {r['rtl']['x2']:11.4f} "
              f"{r['exact']['x2']:11.4f}" + ("   unstable: eps * 3 lambda x^2 > 2, x pinned at the rails"
                                             if unstable else ""))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
        plot(out, os.path.dirname(a.out), rng)
    print("pmode study: PASS" if rc == 0 else "pmode study: FAIL")
    sys.exit(rc)


def plot(out, d, rng):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in out["rtl"] if r["case"] in ("well", "tilt")]
    if not rows:
        return
    fig, ax = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 3.2))
    ax = np.atleast_1d(ax)
    centers = 0.5 * (BINS[1:] + BINS[:-1])
    for axi, r in zip(ax, rows):
        c = pm.case(r["case"])
        trace = read_trace(os.path.join(d, f"pm_{r['case']}_rtl_trace.hex"), 1)[BURN:, 0] / 4096
        axi.step(centers, hist1(trace) * 16, where="mid", color="#2a6fdb", label="RTL")
        axi.plot(centers, exact_grid(c, centers) * 16, color="black", lw=1, label="exact e^{-V/T}")
        axi.set_xlim(-2.5, 2.5)
        axi.set_xlabel("x")
        axi.set_title(f"{r['case']}: TV {r['tv_rtl_exact']:.3f}", fontsize=9)
    ax[0].set_ylabel("density")
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "pmode_hist.png"), dpi=120)


if __name__ == "__main__":
    main()
