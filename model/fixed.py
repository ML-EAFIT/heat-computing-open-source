#!/usr/bin/env python3
"""Bit-accurate fixed-point model of the FPGA p-bit sampler.

This file is the *specification* that rtl/*.v must match bit for bit:
same Q-formats, same LFSR, same sigmoid LUT, same update order.

Source of the algorithm (see docs/01_evidence.md for full citations):
  * p-bit update P(x_i=+1) = sigma(2*beta*(sum_j J_ij x_j + h_i))
        [data/An efficient ... 2510.23972v2.pdf, p.5, Eq. 11]
  * MAC with sign-corrected neighbour states (x_j XOR s_j)
        [same pdf, p.25 Fig. 10, p.26 Eq. E7]  -> here: +J or -J
  * chromatic Gibbs, colour blocks updated in turn
        [same pdf, p.5; torx/06_ising_sampling_contrastive_divergence.py:L198-208]
  * annealing = scaling weights by beta on a schedule
        [thrml/03_codon_optimization.py:L311-344]
Everything numeric below (Q-formats, LFSR, LUT) is an [FPGA-APPROX] choice.
"""
import math

# ---------------------------------------------------------------- formats
WB = 8            # weight/bias width, signed two's complement
WFRAC = 4         # default Q3.4 -> range [-8, 7.9375], step 1/16 (parameter)
BB = 8            # beta width, unsigned
BFRAC = 4         # Q4.4  -> range [0, 15.9375]
LUT_AW = 10       # sigmoid LUT address width (1024 entries)
LFRAC = 5         # logit format Q5.5 -> range [-16, 16), step 1/32
PB = 16           # probability / random word width
# logit_q5 = (beta_q4 * acc + round) >>> shift  (2*beta*l in Q.5, rounded)
#   beta*acc has BFRAC+wfrac frac bits; times 2 and down to 5 => shift below
def logit_shift(wfrac=WFRAC):
    s = BFRAC + wfrac - 1 - LFRAC
    assert s >= 1, "wfrac must be >= 3"
    return s


LOGIT_SHIFT = logit_shift()
assert LOGIT_SHIFT == 2

# Galois right-shift masks verified maximal-length (period 2^32-1) by
# lfsr_is_maximal() below; lane k of noise_bank.v uses LFSR_MASKS[k].
LFSR_MASKS = [0xB4BCD35C, 0xA3000000, 0x80200003, 0x80000057]
LFSR_DEFAULT_SEED = 0xACE1F00D


def clog2(n):
    return max(1, (n - 1).bit_length())


def acc_width(n):
    """Accumulator width: |h + sum_j +-J| <= (n+1)*2^(WB-1) never overflows."""
    return WB + clog2(n + 1) + 1


def to_q(x, frac, bits, signed=True):
    """Round a float to fixed point and saturate."""
    v = int(round(x * (1 << frac)))
    lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
    return max(lo, min(hi, v))


# ---------------------------------------------------------------- LFSR
def lfsr_step(state, mask):
    """One Galois right-shift step. Returns (new_state, output_bit)."""
    out = state & 1
    state >>= 1
    if out:
        state ^= mask
    return state, out


def lfsr_draw16(state, mask):
    """Leap 16 steps; output bit k of the draw = LFSR output of step k."""
    r = 0
    for k in range(PB):
        state, b = lfsr_step(state, mask)
        r |= b << k
    return state, r


def lfsr_is_maximal(mask, width=32):
    """Check period 2^width-1 via the companion-matrix order over GF(2)."""
    full = (1 << width) - 1
    cols = [lfsr_step(1 << k, mask)[0] for k in range(width)]
    ident = [1 << k for k in range(width)]

    def apply(a, v):
        r, k = 0, 0
        while v:
            if v & 1:
                r ^= a[k]
            v >>= 1
            k += 1
        return r

    def mpow(a, e):
        res = ident[:]
        while e:
            if e & 1:
                res = [apply(a, c) for c in res]
            a = [apply(a, c) for c in a]
            e >>= 1
        return res

    if mpow(cols, full) != ident:
        return False
    # 2^32-1 = 3 * 5 * 17 * 257 * 65537
    return all(mpow(cols, full // p) != ident for p in (3, 5, 17, 257, 65537))


def lane_seed(seed, k):
    """Per-lane seed (never zero), identical to noise_bank.v."""
    s = (seed ^ ((0x9E3779B9 * (k + 1)) & 0xFFFFFFFF)) & 0xFFFFFFFF
    return s if s else 1


# ---------------------------------------------------------------- sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_lut():
    """1024 x 16-bit table; entry k is sigma((k-512)/32), so LUT[512] = 0.5."""
    lut = []
    for k in range(1 << LUT_AW):
        x = (k - (1 << (LUT_AW - 1))) / (1 << LFRAC)
        lut.append(min((1 << PB) - 1, int(round(sigmoid(x) * (1 << PB)))))
    return lut


def sigmoid_pwl(logit_q5):
    """PLAN-style piecewise-linear sigmoid (shifts/adds only), 16-bit out.

    Segments (|x| in real units): >=5 -> 1; [2.375,5) -> x/32+0.84375;
    [1,2.375) -> x/8+0.625; [0,1) -> x/4+0.5; mirrored for x<0.
    [FPGA-APPROX] standard approximation, not from the sources.
    """
    a = -logit_q5 if logit_q5 < 0 else logit_q5          # Q.5, a >= 0
    one = 1 << PB                                        # 1.0 in Q0.16
    if a >= 5 * 32:
        y = one - 1
    elif a >= 76:                                        # 2.375 * 32
        y = (a << (PB - 5 - 5)) + 55296                  # x/32 + 0.84375
    elif a >= 32:
        y = (a << (PB - 3 - 5)) + 40960                  # x/8 + 0.625
    else:
        y = (a << (PB - 2 - 5)) + 32768                  # x/4 + 0.5
    y = min(y, one - 1)
    return (one - y) if logit_q5 < 0 else y


def sat(v, bits):
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    return max(lo, min(hi, v))


def logit_q5(beta_q, acc, wfrac=WFRAC):
    """Round-to-nearest (beta*acc + 2^(s-1)) >>> s, saturated to LUT_AW bits."""
    s = logit_shift(wfrac)
    return sat((beta_q * acc + (1 << (s - 1))) >> s, LUT_AW)


def p_one(beta_q, acc, lut, wfrac=WFRAC):
    return lut[logit_q5(beta_q, acc, wfrac) + (1 << (LUT_AW - 1))]


# ---------------------------------------------------------------- schedule
class Schedule:
    """Mirror of schedule_ctrl.v.

    mode 0: constant beta_start
    mode 1: linear, beta += delta every sps sweeps, saturating at beta_end
    mode 2: table, beta = table[k], k advances every sps sweeps (holds last)
    """

    def __init__(self, mode=0, beta_start=16, beta_end=16, delta=0, sps=1, table=None):
        self.mode, self.start, self.end, self.delta, self.sps = mode, beta_start, beta_end, delta, max(1, sps)
        self.table = table or [beta_start]
        self.reset()

    def reset(self):
        self.cnt, self.k = 0, 0
        self.beta = self.table[0] if self.mode == 2 else self.start

    def sweep_done(self):
        self.cnt += 1
        if self.cnt < self.sps:
            return
        self.cnt = 0
        if self.mode == 1:
            self.beta = min(self.end, self.beta + self.delta)
        elif self.mode == 2:
            self.k = min(len(self.table) - 1, self.k + 1)
            self.beta = self.table[self.k]


# ---------------------------------------------------------------- sampler
class FixedSampler:
    """Systematic-scan Gibbs with one time-multiplexed p-bit engine.

    J: N x N signed ints (Q3.4) with zero diagonal, h: N ints, order: visit
    list (colour 0 nodes then colour 1 ...). Spin bit 1 = +1, 0 = -1.
    One 16-bit noise draw (lane 0) per p-bit update.
    """

    def __init__(self, J, h, order, schedule, seed=LFSR_DEFAULT_SEED, pwl=False, wfrac=WFRAC):
        n = len(h)
        assert all(J[i][i] == 0 for i in range(n)), "diagonal must be zero"
        self.n, self.J, self.h, self.order = n, J, h, list(order)
        self.sched, self.pwl, self.wfrac = schedule, pwl, wfrac
        self.lut = sigmoid_lut()
        self.lfsr = lane_seed(seed, 0)
        self.mask = LFSR_MASKS[0]
        self.s = [0] * n
        self.aw = acc_width(n)

    def field(self, i):
        acc = self.h[i]
        row, s = self.J[i], self.s
        for j in range(self.n):
            acc += row[j] if s[j] else -row[j]
        return acc

    def update(self, i):
        acc = self.field(i)
        assert -(1 << (self.aw - 1)) <= acc < (1 << (self.aw - 1))
        lg = logit_q5(self.sched.beta, acc, self.wfrac)
        p = sigmoid_pwl(lg) if self.pwl else self.lut[lg + (1 << (LUT_AW - 1))]
        self.lfsr, r = lfsr_draw16(self.lfsr, self.mask)
        self.s[i] = 1 if r < p else 0

    def sweep(self):
        for i in self.order:
            self.update(i)
        self.sched.sweep_done()

    def state_index(self):
        return sum(b << i for i, b in enumerate(self.s))   # bit i = node i


def dense_to_nbr(J):
    """Neighbour lists [(j, J_ij), ...] of the non-zero couplings of a dense J."""
    return [[(j, w) for j, w in enumerate(row) if w] for row in J]


class SparseParallelSampler:
    """Mirror of sparse_array.v: P lock-step engines, sparse neighbour lists.

    nbr[i]: up to D entries (j, Jq). blocks: colour blocks (node lists). In
    block b the node at position k is updated by engine k % P, which draws its
    16-bit noise from lane k % P (own polynomial, see noise_bank.v). Nodes of
    one block must not be coupled (valid colouring), so the parallel update
    equals any sequential order: bit-exact independent of engine timing.
    With P = 1 and blocks = colour blocks this is FixedSampler.
    [INFERRED] block-parallel update [data/An efficient ... 2510.23972v2.pdf, p.5]
    [FPGA-APPROX] P engines instead of one cell per node.

    Noise options (sparse_array.v NOISE_CTL = 1, UART 'Y'):
      q16   : RNG persistence (finite tau0). Each update first draws d from its
              lane; if d >= q16 the cell keeps its spin and takes no second
              draw, else it draws r and sets s = r < p. A lazy Gibbs kernel:
              same stationary law, autocorrelation ~ (1 - q16/65536)^t.
              [DOCUMENTED motivation: RNG autocorrelation tau0, DTM p.6 Fig. 4b]
      share : every lane uses lane 0's draws (one d and one r per round), so the
              cells of a round get identical noise: correlated RNGs.
    """

    def __init__(self, nbr, h, blocks, schedule, P=1, seed=LFSR_DEFAULT_SEED, pwl=False, wfrac=WFRAC,
                 q16=None, share=False):
        n = len(h)
        assert 1 <= P <= len(LFSR_MASKS), "P must be 1..%d (noise lanes)" % len(LFSR_MASKS)
        assert sorted(i for b in blocks for i in b) == list(range(n)), "blocks must partition the nodes"
        for b in blocks:
            bs = set(b)
            for i in b:
                assert all(not (j in bs and w) for j, w in nbr[i]), f"node {i}: coupled inside its colour block"
                assert all(j != i for j, w in nbr[i] if w), f"node {i}: self coupling"
        self.n, self.nbr, self.h, self.blocks, self.P = n, nbr, list(h), [list(b) for b in blocks], P
        self.sched, self.pwl, self.wfrac = schedule, pwl, wfrac
        self.lut = sigmoid_lut()
        self.lfsr = [lane_seed(seed, e) for e in range(P)]
        self.s = [0] * n
        self.deg = max(len(r) for r in nbr)          # D needed by the RTL
        self.q16, self.share = q16, share

    def field(self, i):
        acc = self.h[i]
        for j, w in self.nbr[i]:
            acc += w if self.s[j] else -w
        return acc

    def p_of(self, i):
        lg = logit_q5(self.sched.beta, self.field(i), self.wfrac)
        return sigmoid_pwl(lg) if self.pwl else self.lut[lg + (1 << (LUT_AW - 1))]

    def draw(self, e):
        self.lfsr[e], r = lfsr_draw16(self.lfsr[e], LFSR_MASKS[e])
        return r

    def update(self, i, e):
        p = self.p_of(i)
        if self.q16 is not None and self.draw(e) >= self.q16:
            return                                   # RNG has not decorrelated: keep
        self.s[i] = 1 if self.draw(e) < p else 0

    def sweep(self):
        for b in self.blocks:
            if not self.share:
                for k, i in enumerate(b):
                    self.update(i, k % self.P)
                continue
            for r0 in range(0, len(b), self.P):      # one round = one shared draw
                chunk = b[r0:r0 + self.P]
                if self.q16 is not None and self.draw(0) >= self.q16:
                    continue
                ps = [self.p_of(i) for i in chunk]
                r = self.draw(0)
                for i, p in zip(chunk, ps):
                    self.s[i] = 1 if r < p else 0
        self.sched.sweep_done()

    def state_index(self):
        return sum(b << i for i, b in enumerate(self.s))


def engine_layout(blocks, P):
    """Static round layout of sparse_array.v.

    Each colour block is cut into rounds of P nodes; in round r lane e updates
    the node at block position r*P + e, or idles (None) in a short last round.
    Returns rounds[r][e] = node or None. All lanes share the round pointer, a
    round never mixes two blocks, so its nodes are never coupled. Same visit
    order and lane assignment as SparseParallelSampler.sweep().
    """
    rounds = []
    for b in blocks:
        for r0 in range(0, len(b), P):
            chunk = b[r0:r0 + P]
            rounds.append(chunk + [None] * (P - len(chunk)))
    return rounds


def run(sampler, sweeps, burn_in=0, hist_bits=None, trace=False):
    """Run sweeps; record after each sweep >= burn_in (like readout_counter.v)."""
    n = sampler.n
    hist = [0] * (1 << hist_bits) if hist_bits else None
    up = [0] * n
    tr = []
    for t in range(sweeps):
        sampler.sweep()
        if trace:
            tr.append(sampler.state_index())
        if t >= burn_in:
            idx = sampler.state_index()
            if hist is not None:
                hist[idx & ((1 << hist_bits) - 1)] += 1
            for i in range(n):
                up[i] += sampler.s[i]
    return hist, up, tr


# ------------------------------------------------- exact chain of the fixed model
def fixed_stationary(J, h, order, beta_q, lut=None, tol=1e-13, max_iter=20000, wfrac=WFRAC, pwl=False):
    """Exact stationary law of the *quantized* sweep kernel (N <= ~16).

    Each single-site kernel uses the LUT probability p/65536. pi is found by
    power iteration of pi <- pi K_order[0] ... K_order[N-1] (each K applied in
    O(2^N)). Separates quantization error from Monte-Carlo error (docs/06).
    """
    import numpy as np
    lut = lut or sigmoid_lut()
    n = len(h)
    idx = np.arange(1 << n)
    bits = (idx[:, None] >> np.arange(n)) & 1
    Jm = np.asarray(J, dtype=np.int64)
    if pwl:
        lut = [sigmoid_pwl(k - (1 << (LUT_AW - 1))) for k in range(1 << LUT_AW)]
    lutv = np.asarray(lut, dtype=np.float64) / 65536.0
    sh = logit_shift(wfrac)
    pone = []
    for i in range(n):
        acc = h[i] + ((2 * bits - 1) * Jm[i][None, :]).sum(1)
        lg = np.clip((beta_q * acc + (1 << (sh - 1))) >> sh, -512, 511)
        pone.append(lutv[lg + 512])
    pi = np.full(1 << n, 1.0 / (1 << n))
    for _ in range(max_iter):
        old = pi
        for i in order:
            m = 1 << i
            # state st goes to st|m w.p. p(st) and to st&~m w.p. 1-p(st)
            new = np.zeros_like(pi)
            np.add.at(new, idx | m, pi * pone[i])
            np.add.at(new, idx & ~m, pi * (1.0 - pone[i]))
            pi = new
        if np.abs(pi - old).sum() < tol:
            break
    return pi / pi.sum()


if __name__ == "__main__":
    for m in LFSR_MASKS:
        print(f"mask {m:#010x} maximal: {lfsr_is_maximal(m)}")
    lut = sigmoid_lut()
    print("LUT[0], LUT[512], LUT[1023] =", lut[0], lut[512], lut[1023])
    worst = max(abs(sigmoid_pwl(k - 512) - lut[k]) for k in range(1024)) / 65536
    print(f"PWL vs LUT max abs error: {worst:.4f}")
