#!/usr/bin/env python3
"""Noise study: finite RNG correlation time (tau0) and correlated RNG lanes.

  noise_study.py --sweeps 100000 --out build/noise.json [--rtl case=trace ...]

1. q-scan. sparse_array.v with NOISE_CTL = 1 and nz_tau: per update the RNG has
   decorrelated with probability q (else the cell keeps its spin). For
   q in {1, 1/2, 1/4, 1/8, 1/16} on ring8 and grid16 (P = 4) it reports
     - TV of the sampled law vs the exact stationary law of the quantized
       kernel (fixed.fixed_stationary): the law must not move;
     - tau_int of the energy (Sokal window): mixing must slow down ~ 1/q;
     - TV / iid floor with the effective sample size n / (2 tau_int).
   Runs model/fixed.SparseParallelSampler, which is bit-exact with the RTL
   (make sparse: sp_*_tau cases; make script: noise_sparse session).
2. shared noise. nz_share gives every lane of a round the same draw. TV vs
   the exact law shows the bias that correlated RNGs introduce.
3. --rtl case=trace: long Verilator traces of sp_grid16_p4_tau / _share are
   checked bit-exact against the model over the full run and analysed the
   same way.
[DOCUMENTED motivation] tau0 and RNG autocorrelation [data/An efficient ...
2510.23972v2.pdf, p.6 Fig. 4b]; correlated noise as a hardware risk
[data/Diffusion_Based_Integer_Factorization ... p.76 Sec. 22.4].
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
import cases as cs           # noqa: E402
import export_vectors as ev  # noqa: E402
import fixed as fx           # noqa: E402

BURN = 1000
QS = [None, 32768, 16384, 8192, 4096]


def run_trace(c, P, sweeps, q16=None, share=False):
    lay = ev.sparse_layout(c, P)
    sc = ev.case_schedule(c)
    smp = fx.SparseParallelSampler(lay["nbr"], c["hq"], c["colors"], fx.Schedule(**sc), P=P,
                                   wfrac=c["wfrac"], q16=q16, share=share)
    _, _, tr = fx.run(smp, sweeps, trace=True)
    return np.array(tr, dtype=np.int64)


def energies(c, n_states):
    """E(s) of the quantized parameters for every state index (bit i = node i)."""
    n = c["n"]
    idx = np.arange(n_states)
    s = 2 * ((idx[:, None] >> np.arange(n)) & 1) - 1
    J = np.asarray(c["Jq"], float) / (1 << c["wfrac"])
    h = np.asarray(c["hq"], float) / (1 << c["wfrac"])
    return -0.5 * np.einsum("si,ij,sj->s", s, J, s) - s @ h


def tau_int(x, c=5.0):
    """Integrated autocorrelation time, Sokal's automatic window (tau = 1/2 + sum rho)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    if acf[0] == 0:
        return 0.5
    rho = acf / acf[0]
    tau = 0.5
    for m in range(1, n):
        tau += rho[m]
        if m >= c * tau:
            break
    return tau


def analyse(c, tr, pi, E):
    seg = tr[BURN:]
    n = len(seg)
    emp = np.bincount(seg, minlength=len(pi)) / n
    tv = 0.5 * np.abs(emp - pi).sum()
    t = tau_int(E[seg])
    rng = np.random.default_rng(7)
    neff = max(1, int(n / (2 * t)))
    floor = np.mean([0.5 * np.abs(np.bincount(rng.choice(len(pi), neff, p=pi), minlength=len(pi)) / neff - pi).sum()
                     for _ in range(5)])
    return dict(tv=float(tv), tau=float(t), neff=neff, floor=float(floor), ratio=float(tv / floor),
                e_mean=float(np.mean(E[seg])), e_exact=float(E @ pi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", type=int, default=100000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rtl", nargs="*", default=[], help="sp_<case>_p4_<tau|share>=<rtl trace file>")
    a = ap.parse_args()
    out = dict(sweeps=a.sweeps, scan=[], share=[], rtl=[])
    rc = 0
    cs_list = [cs.ring8(), cs.grid16()]
    print(f"1) RNG persistence (nz_tau), P = 4, {a.sweeps} sweeps, burn-in {BURN}")
    print(f"  {'case':7s} {'q':>7s} {'TV exact':>9s} {'tau_int(E)':>10s} {'tau*q':>6s} {'iid floor':>9s} "
          f"{'TV/floor':>8s} {'<E> rtl':>8s} {'<E> exact':>9s}")
    for c in cs_list:
        pi = fx.fixed_stationary(c["Jq"], c["hq"], c["order"], c["beta_q"], wfrac=c["wfrac"])
        E = energies(c, len(pi))
        for q in QS:
            r = analyse(c, run_trace(c, 4, a.sweeps, q16=q), pi, E)
            qf = 1.0 if q is None else q / 65536
            r.update(case=c["name"], q=qf)
            out["scan"].append(r)
            print(f"  {c['name']:7s} {qf:7.4f} {r['tv']:9.4f} {r['tau']:10.2f} {r['tau'] * qf:6.2f} "
                  f"{r['floor']:9.4f} {r['ratio']:8.2f} {r['e_mean']:8.3f} {r['e_exact']:9.3f}")
            if r["ratio"] > 3.0:
                rc = 1
                print("    ^ law moved (TV > 3x the iid floor at the effective sample size)")
    print("2) shared noise (nz_share): one draw per round for all lanes")
    for c in cs_list:
        pi = fx.fixed_stationary(c["Jq"], c["hq"], c["order"], c["beta_q"], wfrac=c["wfrac"])
        E = energies(c, len(pi))
        for P in (1, 2, 4):
            r = analyse(c, run_trace(c, P, a.sweeps, share=True), pi, E)
            r.update(case=c["name"], P=P)
            out["share"].append(r)
            print(f"  {c['name']:7s} P={P}  TV {r['tv']:.4f}  iid floor {r['floor']:.4f}  ratio {r['ratio']:6.2f}  "
                  f"<E> {r['e_mean']:7.3f} vs exact {r['e_exact']:7.3f}")
    if a.rtl:
        print("3) RTL traces (Verilator), bit-exact vs model over the full run")
    for spec in a.rtl:
        name, path = spec.split("=")
        cname, mode = name[3:].rsplit("_p4_", 1)
        c = {"ring8": cs.ring8, "grid16": cs.grid16}[cname]()
        q16, share = {"tau": (8192, False), "share": (None, True), "tsh": (32768, True)}[mode]
        with open(path) as f:
            rtl = np.array([int(x, 16) for x in f.read().split()], dtype=np.int64)
        ref = run_trace(c, 4, len(rtl), q16=q16, share=share)
        mism = int(np.sum(rtl != ref))
        pi = fx.fixed_stationary(c["Jq"], c["hq"], c["order"], c["beta_q"], wfrac=c["wfrac"])
        r = analyse(c, rtl, pi, energies(c, len(pi)))
        r.update(case=name, sweeps=len(rtl), mismatches=mism)
        out["rtl"].append(r)
        print(f"  {name:20s} {len(rtl)} sweeps, mismatches vs model {mism}; TV {r['tv']:.4f}, "
              f"tau_int {r['tau']:.2f}, TV/floor {r['ratio']:.2f}")
        rc |= int(mism != 0)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
        plot(out, os.path.dirname(a.out))
    print("noise study: PASS" if rc == 0 else "noise study: FAIL")
    sys.exit(rc)


def plot(out, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for name, col in (("ring8", "#2a6fdb"), ("grid16", "#d9792b")):
        rows = [r for r in out["scan"] if r["case"] == name]
        q = [r["q"] for r in rows]
        ax[0].loglog(q, [r["tau"] for r in rows], "o-", color=col, label=name)
        ax[1].semilogx(q, [r["ratio"] for r in rows], "o-", color=col, label=name)
    q = np.array([1 / 16, 1])
    ax[0].loglog(q, 1.2 / q, ":", color="gray", label="∝ 1/q")
    ax[0].set_xlabel("q = P(RNG decorrelated per update)")
    ax[0].set_ylabel("τ_int of energy (sweeps)")
    ax[1].axhline(1, color="gray", lw=0.8)
    ax[1].set_xlabel("q")
    ax[1].set_ylabel("TV vs exact / iid floor")
    ax[0].legend(frameon=False)
    ax[1].set_ylim(0, 3)
    fig.suptitle("RNG persistence: same law, slower mixing", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "noise_tau.png"), dpi=120)


if __name__ == "__main__":
    main()
