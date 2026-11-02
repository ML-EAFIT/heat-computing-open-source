#!/usr/bin/env python3
"""Parallel tempering (replica exchange) for the dense p-bit array: bit-exact spec of rtl/temper_ctrl.v.

K <= 4 replicas of the spin state share ONE p-bit engine (pbit_array.v, noise
lane 0). Rung r of the ladder has inverse temperature ladder[r] (Q4.4);
rung 0 is the target and is the one read out. One round:
  1. sweep replica 0, 1, ..., K-1 in turn, each at the beta of its rung;
  2. swap phase: even rounds try rung pairs (0,1), (2,3), odd rounds (1,2).
     For pair (a, b) holding replicas x, y:
        ln r = (beta_a - beta_b) * (E_x - E_y)       (exchange ratio)
     accepted with the Barker probability sigma(ln r). With e = E/2 that is
        p = LUT[logit_q5(beta_a - beta_b, e_x - e_y)]
     i.e. a swap decision IS a p-bit: field = energy difference, beta =
     ladder gap. Signs are flipped so the beta argument is >= 0, the field is
     saturated to the p-bit's AW bits, and the draw comes from noise lane 1
     (mask LFSR_MASKS[1], seed lane_seed(seed, 1)). An accepted swap exchanges
     the rung labels (not the spins).
Energies are tracked incrementally: when p-bit i flips, e += s_old * field_i
(field_i = h_i + sum_j J_ij s_j, s = +-1), exact in integer units. They are
relative to the common start state (all replicas equal after a reset or a
resync), so the unknown offset cancels in e_x - e_y. Changing h or J
invalidates the offsets: resync() copies the rung-0 replica into every
replica and zeroes e (pgate.v does this on every start, top.v on 'Z').

[FPGA-APPROX] Barker instead of Metropolis acceptance (both satisfy detailed
balance; Barker reuses the sigmoid LUT). Replica exchange itself is the
standard remedy for the slow mixing seen in docs/05 (factor 3: TV 0.47).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixed as fx          # noqa: E402

SWAP_LANE = 1


def pairs_of(t, K):
    return [(a, a + 1) for a in ((0, 2) if t % 2 == 0 else (1,)) if a + 1 < K]


class Tempering:
    """Replica bookkeeping and the swap phase; sweeps are supplied by the caller."""

    def __init__(self, ladder, n, seed=fx.LFSR_DEFAULT_SEED, aw=14):
        self.ladder = list(ladder)
        self.K, self.n, self.aw = len(ladder), n, aw
        self.lut = fx.sigmoid_lut()
        self.reset(seed)

    def reset(self, seed):
        self.reps = [[0] * self.n for _ in range(self.K)]
        self.e = [0] * self.K
        self.rep_at = list(range(self.K))           # rung -> replica
        self.t = 0
        self.lfsr = fx.lane_seed(seed, SWAP_LANE)
        self.tries = [0] * max(1, self.K - 1)
        self.accepts = [0] * max(1, self.K - 1)

    def resync(self):
        r0 = list(self.reps[self.rep_at[0]])
        self.reps = [list(r0) for _ in range(self.K)]
        self.e = [0] * self.K

    def rung_of(self, k):
        return self.rep_at.index(k)

    def swap_phase(self):
        for a, b in pairs_of(self.t, self.K):
            x, y = self.rep_at[a], self.rep_at[b]
            db, de = self.ladder[a] - self.ladder[b], self.e[x] - self.e[y]
            if db < 0:
                db, de = -db, -de
            p = self.lut[fx.logit_q5(db, fx.sat(de, self.aw)) + (1 << (fx.LUT_AW - 1))]
            self.lfsr, r = fx.lfsr_draw16(self.lfsr, fx.LFSR_MASKS[SWAP_LANE])
            self.tries[a] += 1
            if r < p:
                self.rep_at[a], self.rep_at[b] = y, x
                self.accepts[a] += 1
        self.t += 1

    def round(self, sweep):
        """sweep(spins, beta) mutates spins (one sweep) and returns sum of s_old * field over flips."""
        for k in range(self.K):
            self.e[k] += sweep(self.reps[k], self.ladder[self.rung_of(k)])
        self.swap_phase()

    def spins(self):
        return self.reps[self.rep_at[0]]


class TemperingSampler:
    """Dense sampler with tempering, the mirror of pgate.v (K > 1).

    Same update as fixed.FixedSampler: visit order, lane-0 draw per update.
    """

    def __init__(self, J, h, order, ladder, seed=fx.LFSR_DEFAULT_SEED, wfrac=fx.WFRAC, aw=None):
        n = len(h)
        self.n, self.J, self.h, self.order, self.wfrac = n, J, list(h), list(order), wfrac
        self.lut = fx.sigmoid_lut()
        self.lfsr = fx.lane_seed(seed, 0)
        self.pt = Tempering(ladder, n, seed, aw or fx.acc_width(n))

    def sweep_one(self, s, beta):
        de = 0
        for i in self.order:
            acc = self.h[i]
            row = self.J[i]
            for j in range(self.n):
                acc += row[j] if s[j] else -row[j]
            p = self.lut[fx.logit_q5(beta, acc, self.wfrac) + (1 << (fx.LUT_AW - 1))]
            self.lfsr, r = fx.lfsr_draw16(self.lfsr, fx.LFSR_MASKS[0])
            new = 1 if r < p else 0
            if new != s[i]:
                de += acc if s[i] else -acc
            s[i] = new
        return de

    def sweep(self):
        """One round (K sweeps + swaps); named sweep so fixed.run() works."""
        self.pt.round(self.sweep_one)

    @property
    def s(self):
        return self.pt.spins()

    def state_index(self):
        return sum(b << i for i, b in enumerate(self.s))

    def energy2(self, s):
        """E/2 of a state in integer units (for tests of the incremental tracking)."""
        sp = [1 if b else -1 for b in s]
        e = -sum(self.h[i] * sp[i] for i in range(self.n))
        e2 = -sum(self.J[i][j] * sp[i] * sp[j] for i in range(self.n) for j in range(self.n))
        return (2 * e + e2) / 4


def _selftest():
    """Incremental energies equal the true E/2 differences; replicas keep their labels consistent."""
    import random
    rnd = random.Random(3)
    n = 6
    J = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            J[i][j] = J[j][i] = rnd.randint(-20, 20)
    h = [rnd.randint(-10, 10) for _ in range(n)]
    smp = TemperingSampler(J, h, list(range(n)), [40, 24, 12, 4])
    e0 = smp.energy2([0] * n)
    for _ in range(300):
        smp.sweep()
        for k in range(4):
            assert abs((smp.energy2(smp.pt.reps[k]) - e0) - smp.pt.e[k]) < 1e-9, "energy tracking"
    assert sorted(smp.pt.rep_at) == [0, 1, 2, 3]
    print("tempering selftest: incremental energies exact over 300 rounds; acceptance",
          [f"{a / max(1, t):.2f}" for a, t in zip(smp.pt.accepts, smp.pt.tries)])


if __name__ == "__main__":
    _selftest()
