#!/usr/bin/env python3
"""Statistics of raw noise draws ('N' command of top.v), for the RO_JITTER study.

  entropy.py --dump ../tb/build/script_noise_dense/dump.hex --skip 3 --count 300 --lfsr-check
  entropy.py --raw noise.bin                       (bytes as read from the board, LE draws)
  host.py entropy --draws 100000 --out noise.bin   (board: reads 'N', then runs this)

Per bit stream (16 bits per draw, bit k = LFSR step k):
  * bias P(1) and its z-score;
  * serial autocorrelation at lags 1..64 (max |r| vs the 3/sqrt(n) band);
  * most-common-value min-entropy with the 99 % upper bound on p
    (NIST SP 800-90B Sec. 6.3.1 style);
  * Markov min-entropy per bit from the first-order transition matrix, maximised
    over 128-bit paths (SP 800-90B Sec. 6.3.3 style, no confidence bounds);
  * chi-square of the byte histogram (Wilson-Hilferty p-value).
--lfsr-check: the draws must equal model/fixed.lfsr_draw16 of lane 0 seeded like
noise_bank.v (RO_JITTER = 0). An LFSR passes these tests but has zero true
entropy: the tests are necessary, not sufficient
[DOCUMENTED: min-entropy estimate asked for in data/Diffusion_Based_Integer_
Factorization ... p.76 Sec. 22.4]. [FPGA-APPROX] everything here.
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import fixed as fx          # noqa: E402


def draws_from_bytes(b):
    b = list(b)
    return [b[2 * k] | (b[2 * k + 1] << 8) for k in range(len(b) // 2)]


def bits_of(draws):
    d = np.asarray(draws, dtype=np.int64)
    return ((d[:, None] >> np.arange(16)) & 1).reshape(-1)


def mcv_entropy(bits):
    n = len(bits)
    p = max(bits.mean(), 1 - bits.mean())
    pu = min(1.0, p + 2.576 * math.sqrt(p * (1 - p) / (n - 1)))
    return -math.log2(pu)


def markov_entropy(bits, L=128):
    b0, b1 = bits[:-1], bits[1:]
    P = np.zeros((2, 2))
    for a in (0, 1):
        m = b0 == a
        tot = max(1, m.sum())
        P[a, 1] = (b1[m] == 1).sum() / tot
        P[a, 0] = 1 - P[a, 1]
    init = np.array([1 - bits.mean(), bits.mean()])
    with np.errstate(divide="ignore"):
        lp, lP = np.log2(init), np.log2(P)
    best = lp.copy()
    for _ in range(L - 1):
        best = np.max(best[:, None] + lP, axis=0)
    return float(-best.max() / L), P


def chi2_bytes(draws):
    by = np.array([x for d in draws for x in (d & 255, d >> 8)])
    cnt = np.bincount(by, minlength=256)
    e = len(by) / 256
    x2 = float(((cnt - e) ** 2 / e).sum())
    k = 255
    z = ((x2 / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    pval = 0.5 * math.erfc(z / math.sqrt(2))
    return x2, pval


def autocorr(bits, lags=64):
    x = 2.0 * bits - 1
    x = x - x.mean()
    v = (x * x).mean()
    return np.array([(x[:-k] * x[k:]).mean() / v for k in range(1, lags + 1)])


def report(draws, label="noise"):
    bits = bits_of(draws)
    n = len(bits)
    p1 = bits.mean()
    z = (p1 - 0.5) / math.sqrt(0.25 / n)
    ac = autocorr(bits)
    band = 3 / math.sqrt(n)
    hm = mcv_entropy(bits)
    hk, P = markov_entropy(bits)
    x2, pv = chi2_bytes(draws)
    print(f"{label}: {len(draws)} draws, {n} bits")
    print(f"  bias        P(1) = {p1:.5f}  (z = {z:+.2f})")
    print(f"  autocorr    max |r| over lags 1..64 = {np.abs(ac).max():.4f} at lag {int(np.abs(ac).argmax()) + 1}"
          f"  (3/sqrt(n) band {band:.4f}; {int((np.abs(ac) > band).sum())} lags outside)")
    print(f"  min-entropy MCV {hm:.4f} bit/bit, Markov {hk:.4f} bit/bit "
          f"(P(1|0) = {P[0, 1]:.4f}, P(1|1) = {P[1, 1]:.4f})")
    print(f"  bytes       chi2 = {x2:.1f} (255 dof), p = {pv:.3f}")
    return dict(draws=len(draws), p1=float(p1), z=float(z), ac_max=float(np.abs(ac).max()), band=band,
                h_mcv=hm, h_markov=hk, chi2=x2, chi2_p=pv)


def lfsr_ref(n, seed=fx.LFSR_DEFAULT_SEED):
    s = fx.lane_seed(seed, 0)
    out = []
    for _ in range(n):
        s, r = fx.lfsr_draw16(s, fx.LFSR_MASKS[0])
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", help="hex byte dump (one byte per line), e.g. a tb_top_script dump")
    ap.add_argument("--raw", help="binary file of LE 16-bit draws")
    ap.add_argument("--skip", type=int, default=0, help="bytes to skip before the draws")
    ap.add_argument("--count", type=int, default=None, help="number of draws")
    ap.add_argument("--lfsr-check", action="store_true")
    ap.add_argument("--seed", type=lambda x: int(x, 0), default=fx.LFSR_DEFAULT_SEED)
    a = ap.parse_args()
    if a.dump:
        with open(a.dump) as f:
            b = [int(x, 16) for x in f.read().split()]
    else:
        with open(a.raw, "rb") as f:
            b = list(f.read())
    b = b[a.skip:]
    draws = draws_from_bytes(b)
    if a.count is not None:
        draws = draws[:a.count]
    report(draws, os.path.basename(a.dump or a.raw))
    if a.lfsr_check:
        ref = lfsr_ref(len(draws), a.seed)
        nm = sum(x != y for x, y in zip(draws, ref))
        print(f"  lfsr check  {nm} of {len(draws)} draws differ from fixed.lfsr_draw16 (lane 0)")
        print("entropy: PASS" if nm == 0 else "entropy: FAIL")
        sys.exit(1 if nm else 0)


if __name__ == "__main__":
    main()
