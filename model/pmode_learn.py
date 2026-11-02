#!/usr/bin/env python3
"""Learn a p-mode circuit from its samples, with the p-mode chip as the sampler.

A hidden "teacher" chip (pmode.case(target): A, h in Q3.4, lambda, eps) produces soft
spins x(t). A "student" chip with the same lambda, eps and T, but A = h = 0, learns
A and h by maximum likelihood. The model is an exponential family in s = tanh(x),
    log p(x) = (1/2 s.A.s + h.s - lambda/4 sum (x^2 - 1)^2) / T - log Z,
so the gradient is a moment difference (the continuous version of model/train.py):
    dA_ij = lr (<s_i s_j>_data - <s_i s_j>_model),   dh_i = lr (<s_i>_data - <s_i>_model)
The negative phase runs the student persistently: 64 copies of the bit-exact model of
rtl/pmode_array.v (PmodeBank, the chip has one; a bank of boards or more time does the same); the float master weights are rounded to Q3.4 words before every run, as the
chip stores them. Teacher and student share the same fixed-point dynamics, so the O(eps)
Langevin bias cancels and the teacher's words are the fixed point.
[FPGA-APPROX] learning rate, run lengths.

  python3 model/pmode_learn.py      train on chain4, print the recovered couplings
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixed as fx          # noqa: E402
import pmode as pm          # noqa: E402

QMAX = 127 / 16


def q34(v):
    return fx.to_q(float(v), 4, 8)


class PmodeBank:
    """K independent copies of pmode.PmodeArray in numpy (int64, same shifts), one LFSR per copy.

    Copy k is bit-exact with PmodeArray(..., seed=seeds[k]) (checked in main()); the bank only
    makes the sampler fast enough to train on (one step costs the same for any K)."""

    def __init__(self, A, h, lam, k, sig, seeds, x0=None):
        self.m, self.lam, self.k, self.sig = len(h), lam, k, sig
        self.set_words(A, h)
        self.K = len(seeds)
        self.x = np.zeros((self.K, self.m), np.int64) if x0 is None else np.array(x0, np.int64)
        self.lut = np.array(fx.sigmoid_lut(), np.int64)
        self.gauss = np.array(pm.gauss_table(), np.int64)
        self.lfsr = np.array([fx.lane_seed(s, 0) for s in seeds], np.int64)
        self.mask = np.int64(fx.LFSR_MASKS[0])
        self.lo, self.hi = -(1 << 15), (1 << 15) - 1
        self.alo, self.ahi = -(1 << (fx.LUT_AW - 1)), (1 << (fx.LUT_AW - 1)) - 1

    def set_words(self, A, h):
        self.A, self.h = np.array(A, np.int64), np.array(h, np.int64)

    def draw16(self):
        r = np.zeros(self.K, np.int64)
        st = self.lfsr
        for b in range(16):
            out = st & 1
            st = (st >> 1) ^ (out * self.mask)
            r |= out << b
        self.lfsr = st
        return r

    def step(self):
        idx = np.clip((self.x + 32) >> 6, self.alo, self.ahi) + (1 << (fx.LUT_AW - 1))
        s = self.lut[idx] - 32768                                       # (K, m) Q.15
        new = np.empty_like(self.x)
        for i in range(self.m):
            F = ((self.h[i] << 15) + s @ self.A[i]) >> 7
            g1 = ((1 << 30) - s[:, i] * s[:, i]) >> 15
            cpl = (F * g1) >> 15
            x = self.x[:, i]
            x2 = (x * x) >> pm.XF
            q = (x * (x2 - (1 << pm.XF))) >> pm.XF
            grad = ((self.lam * q) >> 4) - cpl
            noise = (self.sig * self.gauss[self.draw16() >> 6]) >> pm.XF
            new[:, i] = np.clip(x - (grad >> self.k) + noise, self.lo, self.hi)
        self.x = new

    def xs(self):
        return self.x / (1 << pm.XF)


class PmodeTrainer:
    def __init__(self, target="chain4", chains=64, data_steps=1500, thin=4, steps=200, lr=0.5, lr_tau=30,
                 epochs=120, seed=7):
        c = pm.case(target)
        self.c, self.m = c, c["m"]
        self.steps, self.lr, self.lr_tau, self.target_epochs = steps, lr, lr_tau, epochs
        seeds = lambda base: [(base * 7919 + 104729 * i) & 0xFFFFFFFF or 1 for i in range(chains)]  # noqa: E731
        teacher = PmodeBank(c["Aq"], c["hq"], c["lamq"], c["k"], c["sig"], seeds(seed))
        for _ in range(300):
            teacher.step()
        xs = []
        for t in range(data_steps * thin):
            teacher.step()
            if t % thin == 0:
                xs.append(teacher.xs())
        self.x_data = np.concatenate(xs)
        S = np.tanh(self.x_data)
        self.md, self.Cd = S.mean(0), S.T @ S / len(S)
        self.A_true = np.array(c["Aq"], float) / 16
        self.h_true = np.array(c["hq"], float) / 16
        self.A, self.h = np.zeros((self.m, self.m)), np.zeros(self.m)
        self.student = PmodeBank(*self.words(), c["lamq"], c["k"], c["sig"], seeds(seed + 101))
        self.t = 0
        self.hist = []
        self.x_model = None
        self.record(self.moments(self.run(self.steps)))

    def words(self):
        return [[q34(v) for v in r] for r in self.A], [q34(v) for v in self.h]

    def run(self, steps):
        self.student.set_words(*self.words())
        xs = []
        for _ in range(steps):
            self.student.step()
            xs.append(self.student.xs())
        self.x_model = np.concatenate(xs[steps // 4:])         # the first quarter re-equilibrates
        return self.x_model

    @staticmethod
    def moments(x):
        S = np.tanh(x)
        return S.mean(0), S.T @ S / len(S)

    def step(self):
        mm, Cm = self.moments(self.run(self.steps))
        lr = self.lr / (1 + self.t / self.lr_tau)
        self.t += 1
        G = self.Cd - Cm
        np.fill_diagonal(G, 0)
        self.A = np.clip(self.A + lr * G, -QMAX, QMAX)
        self.h = np.clip(self.h + lr * (self.md - mm), -QMAX, QMAX)
        self.record((mm, Cm))

    def record(self, mom):
        mm, Cm = mom
        off = ~np.eye(self.m, dtype=bool)
        Aw, hw = self.words()
        self.hist.append(dict(
            moment_err=float(max(np.abs(self.Cd - Cm)[off].max(), np.abs(self.md - mm).max())),
            param_err=float(max(np.abs(np.array(Aw) / 16 - self.A_true)[off].max(), np.abs(np.array(hw) / 16 - self.h_true).max()))))

    def histograms(self, bins=40, lo=-2.5, hi=2.5):
        edges = np.linspace(lo, hi, bins + 1)
        dens = lambda x: [np.histogram(x[:, i], edges, density=True)[0].tolist() for i in range(self.m)]   # noqa: E731
        return dict(edges=edges.tolist(), data=dens(self.x_data), model=dens(self.x_model))

    def status(self):
        Aw, hw = self.words()
        return dict(epoch=self.t, target=self.target_epochs, hist=self.hist, A=(np.array(Aw) / 16).tolist(),
                    h=[v / 16 for v in hw], A_true=self.A_true.tolist(), h_true=self.h_true.tolist(),
                    lam=self.c["lam"], eps=self.c["eps"], T=self.c["T"], names=[f"x{i}" for i in range(self.m)],
                    n_data=len(self.x_data), steps=self.steps, chains=self.student.K, **self.histograms())


def bank_check(steps=300):
    """PmodeBank copy k == pmode.PmodeArray with the same seed, bit for bit."""
    c = pm.case("chain4")
    seeds = [0xACE1F00D, 12345, 777]
    bank = PmodeBank(c["Aq"], c["hq"], c["lamq"], c["k"], c["sig"], seeds)
    refs = [pm.PmodeArray(c["Aq"], c["hq"], c["lamq"], c["k"], c["sig"], seed=s) for s in seeds]
    for _ in range(steps):
        bank.step()
        for r in refs:
            r.step()
    return all(bank.x[k].tolist() == r.x for k, r in enumerate(refs))


def main():
    import time
    ok_bank = bank_check()
    print(f"pmode_learn: vectorised bank == pmode.PmodeArray bit for bit: {'ok' if ok_bank else 'FAIL'}")
    t0 = time.time()
    tr = PmodeTrainer()
    for ep in range(tr.target_epochs):
        tr.step()
        if (ep + 1) % 25 == 0:
            h = tr.hist[-1]
            print(f"  epoch {ep + 1:4d}  moment error {h['moment_err']:.3f}  parameter error {h['param_err']:.3f}")
    st = tr.status()
    print("learned A =", np.round(st["A"], 4).tolist(), " h =", st["h"])
    print("true    A =", st["A_true"], " h =", st["h_true"])
    lsb = st["hist"][-1]["param_err"] * 16
    print(f"pmode_learn: {len(tr.x_data)} data samples, {tr.target_epochs} epochs in {time.time() - t0:.1f} s; "
          f"worst parameter off by "
          f"{lsb:.0f} LSB (1 LSB = 1/16)")
    ok = ok_bank and lsb <= 1
    print("pmode_learn: PASS" if ok else "pmode_learn: FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
