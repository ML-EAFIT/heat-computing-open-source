#!/usr/bin/env python3
"""Continuous p-mode cell: fixed-point Langevin soft spins, bit-exact spec of rtl/pmode_array.v.

Mirror of torx's LangevinGate [torx/helpers/_langevin.py:L46-143]:
    V(x) = -1/2 s^T A s - h^T s + (lambda/4) sum_i (x_i^2 - 1)^2,   s = tanh(x)
    x'   = x - eps * dV/dx + sqrt(2 T eps) * xi,   xi ~ N(0, 1)   (ULA, all sites at once)
    dV/dx_i = -(1 - s_i^2) (sum_j A_ij s_j + h_i) + lambda x_i (x_i^2 - 1)
(beta folded into A, as for the p-bits). Its invariant law is approximately
exp(-V/T) with an O(eps) bias (no Metropolis step) [DOCUMENTED, same file].

Fixed point [FPGA-APPROX] (Python >> on ints = arithmetic shift = Verilog >>>):
    x      Q4.12, 16-bit signed, saturated          (range [-8, 8))
    s15    tanh(x) in Q.15 = LUT[sat((x + 32) >> 6)] - 32768
           (tanh x = 2 sigma(2x) - 1, the p-bit's sigmoid LUT, step 1/64 in x)
    A, h   Q3.4 (8-bit), lambda Q4.4 (8-bit unsigned)
    F      ((h << 15) + sum_j A_ij s15_j) >> 7               Q.12
    g1     (2^30 - s15^2) >> 15                              Q.15  (= 1 - s^2)
    cpl    (F * g1) >> 15                                    Q.12
    x2     (x * x) >> 12;  q = (x * (x2 - 4096)) >> 12;  quart = (lambda * q) >> 4
    grad   quart - cpl
    noise  (sig * xi) >> 12, sig = sqrt(2 T eps) in Q4.12 (16-bit unsigned),
           xi = GAUSS[r >> 6], r = 16-bit draw of noise lane 0 (one per site)
    x'     sat16(x - (grad >> k) + noise),  eps = 2^-k
GAUSS: 1024-entry inverse-CDF table, xi_k = Phi^-1((k + 1/2) / 1024) rescaled to
unit variance, Q3.12 (|xi| <= 3.3) [FPGA-APPROX].
Update order: tanh of every site, then sites 0..M-1 (one draw each, from the
old state), then all x' are committed together (Jacobi, as the torx gate).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixed as fx          # noqa: E402

XF = 12                      # x fraction bits
GN = 1024


def gauss_table():
    from statistics import NormalDist
    nd = NormalDist()
    z = [nd.inv_cdf((k + 0.5) / GN) for k in range(GN)]
    sd = math.sqrt(sum(v * v for v in z) / GN)
    return [int(round(v / sd * (1 << XF))) for v in z]


def sat(v, bits=16):
    return fx.sat(v, bits)


class PmodeArray:
    def __init__(self, A, h, lam, k, sig, x0=None, seed=fx.LFSR_DEFAULT_SEED):
        m = len(h)
        self.m, self.A, self.h, self.lam, self.k, self.sig = m, A, list(h), lam, k, sig
        self.x = list(x0) if x0 else [0] * m
        self.lut = fx.sigmoid_lut()
        self.gauss = gauss_table()
        self.lfsr = fx.lane_seed(seed, 0)

    def tanh15(self, x):
        return self.lut[sat((x + 32) >> 6, fx.LUT_AW) + (1 << (fx.LUT_AW - 1))] - 32768

    def step(self):
        s = [self.tanh15(x) for x in self.x]
        new = []
        for i in range(self.m):
            acc = self.h[i] << 15
            for j in range(self.m):
                acc += self.A[i][j] * s[j]
            F = acc >> 7
            g1 = ((1 << 30) - s[i] * s[i]) >> 15
            cpl = (F * g1) >> 15
            x = self.x[i]
            x2 = (x * x) >> XF
            q = (x * (x2 - (1 << XF))) >> XF
            quart = (self.lam * q) >> 4
            grad = quart - cpl
            self.lfsr, r = fx.lfsr_draw16(self.lfsr, fx.LFSR_MASKS[0])
            noise = (self.sig * self.gauss[r >> 6]) >> XF
            new.append(sat(x - (grad >> self.k) + noise))
        self.x = new

    def xs(self):
        return [v / (1 << XF) for v in self.x]


# ------------------------------------------------------------------ references
def potential(x, A, h, lam):
    """V(x) in real units (A, h, lam real), x: (..., M) array."""
    import numpy as np
    s = np.tanh(x)
    return -0.5 * np.einsum("...i,ij,...j->...", s, A, s) - s @ h + 0.25 * lam * ((x * x - 1) ** 2).sum(-1)


def float_ula(A, h, lam, eps, T, steps, chains, rng, x0=None):
    """Float ULA of the same dynamics (torx LangevinGate with metropolis=False), many chains.

    x is clipped to [-8, 8) like the hardware's Q4.12 saturation, so an unstable
    step size (explicit Euler on the stiff quartic: eps * 3 lambda x^2 > 2)
    shows up as a pinned chain instead of an overflow.
    """
    import numpy as np
    m = len(h)
    x = np.zeros((chains, m)) if x0 is None else np.array(x0, float)
    out = []
    for _ in range(steps):
        s = np.tanh(x)
        grad = -(1 - s * s) * (s @ A.T + h) + lam * x * (x * x - 1)
        x = np.clip(x - eps * grad + math.sqrt(2 * T * eps) * rng.standard_normal(x.shape), -8.0, 8.0 - 2.0 ** -XF)
        out.append(x.copy())
    return np.array(out)


def case(name):
    """Named p-mode configurations: real parameters + their fixed-point words."""
    T = 1.0
    if name == "well":           # one double well, lambda = 4, no field
        A, h, lam, k = [[0.0]], [0.0], 4.0, 5
    elif name == "tilt":         # tilted double well
        A, h, lam, k = [[0.0]], [0.5], 4.0, 5
    elif name == "pair":         # two wells coupled ferromagnetically
        A, h, lam, k = [[0.0, 1.0], [1.0, 0.0]], [0.0, 0.0], 4.0, 5
    elif name == "chain4":       # 4-site ring, mixed couplings and fields (bit-exact test case)
        A = [[0, 0.75, 0, -0.5], [0.75, 0, 1.0, 0], [0, 1.0, 0, 0.5], [-0.5, 0, 0.5, 0]]
        h, lam, k = [0.25, -0.25, 0.0, 0.125], 2.0, 4
    else:
        raise KeyError(name)
    eps = 2.0 ** -k
    words = dict(Aq=[[fx.to_q(v, 4, 8) for v in r] for r in A], hq=[fx.to_q(v, 4, 8) for v in h],
                 lamq=fx.to_q(lam, 4, 8, signed=False), k=k,
                 sig=fx.to_q(math.sqrt(2 * T * eps), XF, 16, signed=False))
    return dict(name=name, A=A, h=h, lam=lam, eps=eps, T=T, m=len(h), **words)


if __name__ == "__main__":
    g = gauss_table()
    var = sum((v / 4096) ** 2 for v in g) / GN
    print(f"gauss table: {GN} entries, max {max(g) / 4096:.3f}, variance {var:.4f}")
    c = case("well")
    p = PmodeArray(c["Aq"], c["hq"], c["lamq"], c["k"], c["sig"])
    xs = []
    for t in range(20000):
        p.step()
        xs.append(p.xs()[0])
    print(f"well: <|x|> = {sum(abs(v) for v in xs[1000:]) / len(xs[1000:]):.3f}, "
          f"P(x > 0) = {sum(v > 0 for v in xs[1000:]) / len(xs[1000:]):.3f}")
