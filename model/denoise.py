#!/usr/bin/env python3
"""Toy denoising-thermodynamic-model (DTM) chain, bit-accurate mirror of rtl/denoise_demo.v.

What comes from the sources (DTM = data/An efficient ... 2510.23972v2.pdf):
  * reverse process = chain of T EBMs, step t samples x^{t-1} given clamped x^t
        [DTM p.4 Eqs. 5, 7-8; Fig. 3b]
  * forward energy = one pairwise coupling per variable, strength Gamma(t)/2
        [DTM p.23 Eq. D1], Gamma(t) = ln((1+(M-1)e^{-gamma t})/(1-e^{-gamma t}))
        [DTM p.16 Eq. B15], M = 2 for binary variables
  * the x^t nodes stay fixed during Gibbs sampling [DTM p.24 Fig. 9b]
Our choices ([FPGA-APPROX]; DTM gives no small trained model):
  * marginal energy E^theta = Hopfield/Hebbian couplings storing 3 4x4 patterns,
    no latent variables (DTM uses a trained latent grid, App. D.2)
  * per-step durations dt = 0.25, 0.5, 0.75, 1.0 (gamma = 1) -> 4 steps
  * clamping x^t == bias offset (Gamma_t/2) * x^t_i on node i [INFERRED, Eq. 11 + D1]
  * sign: ferromagnetic binding of x^{t-1} to x^t (see docs/01_evidence.md)
"""
import argparse
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixed as fx  # noqa: E402

G = 4
N = G * G
T = 4
K = 16                       # Gibbs sweeps per denoising step
BETA_Q = 16                  # beta = 1.0
DT = [0.25, 0.5, 0.75, 1.0]  # step t = 1..T uses DT[t-1]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors", "denoise")

PATTERNS = [
    "X..X"
    ".XX."
    ".XX."
    "X..X",
    "XXXX"
    "X..X"
    "X..X"
    "XXXX",
    "XX.."
    "XX.."
    "..XX"
    "..XX",
]


def pattern_spins(p):
    return [1 if ch == "X" else -1 for ch in p]


def gamma_fn(t, m=2, gamma=1.0):
    return math.log((1 + (m - 1) * math.exp(-gamma * t)) / (1 - math.exp(-gamma * t)))


def model():
    xi = [pattern_spins(p) for p in PATTERNS]
    Jf = [[0.0 if i == j else sum(x[i] * x[j] for x in xi) / len(xi) for j in range(N)] for i in range(N)]
    Jq = [[fx.to_q(v * 0.75, fx.WFRAC, fx.WB) for v in row] for row in Jf]   # scale 0.75
    hq = [0] * N
    gq = [fx.to_q(gamma_fn(dt) / 2, fx.WFRAC, fx.WB) for dt in DT]
    colors = [[i for i in range(N) if (i // G + i % G) % 2 == p] for p in (0, 1)]
    order = [i for c in colors for i in c]            # dense J: any fixed order is a valid scan
    return Jq, hq, gq, order, xi


def sat8(v):
    return max(-128, min(127, v))


class Denoiser:
    def __init__(self, seed=fx.LFSR_DEFAULT_SEED):
        self.Jq, self.hq, self.gq, self.order, self.xi = model()
        self.smp = fx.FixedSampler(self.Jq, list(self.hq), self.order, fx.Schedule(0, BETA_Q, BETA_Q), seed=seed)

    def run(self, xT_bits):
        """xT_bits: int, bit i = node i. Returns list of x^{t-1} after each step (t = T..1)."""
        s = self.smp
        s.s = [(xT_bits >> i) & 1 for i in range(N)]
        outs = []
        for t in range(T, 0, -1):
            g = self.gq[t - 1]
            s.h = [sat8(self.hq[i] + (g if s.s[i] else -g)) for i in range(N)]
            for _ in range(K):
                s.sweep()
            outs.append(s.state_index())
        return outs


def classify(bits, xi):
    """Hamming distance to the nearest stored pattern or its negation."""
    v = [1 if (bits >> i) & 1 else -1 for i in range(N)]
    best = N
    for x in xi:
        d = sum(1 for a, b in zip(v, x) if a != b)
        best = min(best, d, N - d)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--runs", type=int, default=2000)
    a = ap.parse_args()
    Jq, hq, gq, order, xi = model()
    rnd = random.Random(7)
    print(f"denoise: N={N} T={T} K={K} beta={BETA_Q/16} Gamma_t/2 (Q3.4) per step t=1..T: {gq} "
          f"= {[round(g / 16, 4) for g in gq]}")
    d = Denoiser()
    hits, exact = 0, 0
    base_hits = 0
    for _ in range(a.runs):
        xT = rnd.getrandbits(N)
        x0 = d.run(xT)[-1]
        dist = classify(x0, xi)
        hits += dist <= 1
        exact += dist == 0
        base_hits += classify(rnd.getrandbits(N), xi) <= 1
    print(f"denoise: {a.runs} chains from uniform x^T: x^0 within Hamming 1 of a stored pattern "
          f"(or its negation) in {100 * hits / a.runs:.1f}% (exact {100 * exact / a.runs:.1f}%); "
          f"uniform random baseline {100 * base_hits / a.runs:.1f}%")
    if a.export:
        os.makedirs(OUT, exist_ok=True)
        stride = 1 << fx.clog2(N)
        ev = lambda v: f"{v & 0xFF:02x}"   # noqa: E731
        with open(os.path.join(OUT, "weights.hex"), "w") as f:
            f.write("\n".join(ev(Jq[i][j]) for i in range(stride) for j in range(stride)) + "\n")
        with open(os.path.join(OUT, "bias.hex"), "w") as f:
            f.write("\n".join(ev(v) for v in hq) + "\n")
        with open(os.path.join(OUT, "order.hex"), "w") as f:
            f.write("\n".join(f"{i:x}" for i in order) + "\n")
        with open(os.path.join(OUT, "gamma.hex"), "w") as f:
            f.write("\n".join(ev(g) for g in gq) + "\n")
        # 8 bit-exact runs for the RTL testbench (one sampler, LFSR continues)
        d = Denoiser()
        rnd = random.Random(99)
        lines = []
        for _ in range(8):
            xT = rnd.getrandbits(N)
            outs = d.run(xT)
            lines.append(f"{xT:04x}" + "".join(f"{o:04x}" for o in outs))
        with open(os.path.join(OUT, "runs.hex"), "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"denoise: exported vectors to {OUT}")


if __name__ == "__main__":
    main()
