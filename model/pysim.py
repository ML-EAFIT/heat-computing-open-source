#!/usr/bin/env python3
"""Ideal Python simulator of a p-bit array: float weights, any size, many parallel chains.

The chip model (uart_script.TopModel) mirrors rtl/top.v bit for bit. This simulator drops
everything the hardware has to approximate:
  * Q3.4 words (|J|, |h| <= 7.94 in 1/16 steps), the 16-bit sigmoid LUT, LFSR noise;
  * the 16 / 256 p-bit limits and the sparse chip's degree <= 12, <= 64-round layout.
It keeps the algorithm: chromatic Gibbs sweeps with P(s_i = +1) = sigma(2 beta (h_i + sum_j J_ij s_j)),
s = +-1, H = -1/2 s.J.s - h.s (golden.chromatic_gibbs, with persistence, clamps and sparse J).
Pinned p-bits are hard clamps here (on the chip they are extreme biases, docs/09).
C chains run in lock step as one (C, n) numpy array; J is a scipy CSR matrix above 1024 nodes.

Used by web/server.py (the 'pysim' chip of the Learn tab), web/learn.py and model/ml.py:
  Gibbs        the sampler
  SimTrainer   train.Trainer with this sampler instead of the chip (same learning rule)
  RBM          restricted Boltzmann machine (digits demo: 64 pixels + 10 labels + hidden)
  grid_denoise checkerboard MRF denoising on an S x S image (any S)

  python3 model/pysim.py        self-check against exact enumeration + timing
"""
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as spm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden as gd         # noqa: E402
import train                # noqa: E402

DENSE_MAX = 1024


def colour_classes(J):
    """Independent sets covering the coupling graph (greedy), as index arrays."""
    if spm.issparse(J):
        J = J.tocsr()
        n = J.shape[0]
        color = -np.ones(n, int)
        for i in np.argsort(-np.diff(J.indptr), kind="stable"):
            used = set(color[J.indices[J.indptr[i]:J.indptr[i + 1]]].tolist())
            c = 0
            while c in used:
                c += 1
            color[i] = c
        return [np.flatnonzero(color == c) for c in range(color.max() + 1)]
    return [np.array(c) for c in gd.greedy_coloring(np.asarray(J).tolist())]


class Gibbs:
    """C parallel chromatic Gibbs chains over s in {-1, +1}^n."""

    def __init__(self, J, h, beta=1.0, chains=64, colors=None, seed=0):
        self.rng = np.random.default_rng(seed)
        self.n = len(h)
        self.set_weights(J, h)
        self.beta = float(beta)
        self.colors = [np.asarray(c, int) for c in (colors if colors is not None else colour_classes(self.J))]
        self.s = self.rng.choice([-1.0, 1.0], size=(chains, self.n))
        self.fixed = np.zeros(self.n, bool)

    def set_weights(self, J, h):
        if spm.issparse(J) or self.n > DENSE_MAX:
            self.J = spm.csr_matrix(J)
        else:
            self.J = np.asarray(J, float)
        self.h = np.asarray(h, float)

    @property
    def chains(self):
        return self.s.shape[0]

    def clamp(self, idx=(), bits=None):
        """Hold nodes idx at bits (same for every chain: shape (k,); per chain: shape (C, k)). idx=() frees all."""
        idx = np.asarray(idx, int)
        self.fixed[:] = False
        self.fixed[idx] = True
        if len(idx):
            self.s[:, idx] = 2.0 * np.asarray(bits, float) - 1.0
        self._blocks = [c[~self.fixed[c]] for c in self.colors]

    def field(self, idx):
        if spm.issparse(self.J):
            return (self.J[idx] @ self.s.T).T + self.h[idx]          # J symmetric
        return self.s @ self.J[:, idx] + self.h[idx]

    def sweep(self, k=1):
        if not hasattr(self, "_blocks"):
            self.clamp()
        for _ in range(k):
            for idx in self._blocks:
                if len(idx):
                    p = 1.0 / (1.0 + np.exp(-2.0 * self.beta * self.field(idx)))
                    self.s[:, idx] = np.where(self.rng.random(p.shape) < p, 1.0, -1.0)
        return self.s

    def run(self, sweeps, burn=0):
        """burn sweeps, then sweeps sweeps; returns every chain's state after each counted sweep, (sweeps*C, n)."""
        self.sweep(burn)
        return np.concatenate([self.sweep().copy() for _ in range(sweeps)]) if sweeps else np.zeros((0, self.n))

    def marginals(self, sweeps, burn=0):
        """P(s_i = +1) averaged over chains and sweeps."""
        acc = np.zeros(self.n)
        self.sweep(burn)
        for _ in range(sweeps):
            acc += (self.sweep() > 0).mean(0)
        return acc / max(sweeps, 1)


def states(S):
    """(T, n) +-1 array -> state indices (bit k = 1 <=> s_k = +1), n <= 62."""
    return ((S > 0).astype(np.int64) << np.arange(S.shape[1], dtype=np.int64)).sum(1)


# ------------------------------------------------------------------ training (general BM, n <= ~20)
class _NoChip:
    def __init__(self, n):
        self.n = n


class SimTrainer(train.Trainer):
    """train.Trainer with the ideal sampler in place of the chip.

    Same learning rule, data and exact KL; the weights stay float (no Q3.4 rounding or
    +-7.94 clipping), pins are hard clamps, and the phases use many parallel chains:
      positive (hidden units only): every data vector clamped in `rep` chains at once;
      negative: `chains` persistent chains, n_neg samples per epoch.
    """

    def __init__(self, nv, nh, data, chains=200, rep=20, **kw):
        self.n_chains, self.rep = chains, rep
        super().__init__(_NoChip(nv + nh), nv, nh, data, **kw)

    def setup(self, seed):
        order = [[i] for i in range(self.n)]                 # dense BM: one node per colour (sequential Gibbs)
        self.neg = Gibbs(self.J, self.h, chains=self.n_chains, colors=order, seed=seed)
        if self.nh:
            vis = np.array(list(self.data))
            self.pos_w = np.repeat(np.array(list(self.data.values())), self.rep) / self.rep
            self.pos = Gibbs(self.J, self.h, chains=len(vis) * self.rep, colors=order, seed=seed + 1)
            self.pos.clamp(range(self.nv), np.repeat(vis, self.rep, axis=0))

    def upload(self, h_override=None):
        self.neg.set_weights(self.J, self.h)
        if self.nh:
            self.pos.set_weights(self.J, self.h)

    def positive(self):
        if self.nh == 0:
            return super().positive()
        self.pos.sweep(self.burn // 2)
        m, C = np.zeros(self.n), np.zeros((self.n, self.n))
        k = max(1, self.n_pos // 10)
        for _ in range(k):
            S = self.pos.sweep()
            m += self.pos_w @ S / k
            C += (S * self.pos_w[:, None]).T @ S / k
        return m, C

    def negative(self):
        k = max(1, self.n_neg // self.n_chains)
        S = self.neg.run(k, burn=2)
        return S.mean(0), (S.T @ S) / len(S)

    def step(self):
        mp, Cp = self.positive()
        mn, Cn = self.negative()
        lr = self.lr / (1 + self.t / self.lr_tau) if self.lr_tau else self.lr
        self.t += 1
        self.J += lr * ((Cp - Cn) * self.mask - self.decay * self.J)
        self.h += lr * (mp - mn)
        self.upload()

    def quantized(self):                                     # nothing to quantize: the exact KL uses float weights
        return self.J.copy(), self.h.copy()


# ------------------------------------------------------------------ restricted Boltzmann machine
class RBM:
    """Bipartite BM on +-1 units: H = -v.W.g - b.v - c.g (v visible, g hidden).

    Positive phase exact (<g | v> = tanh(c + W^T v)); negative phase from persistent
    block-Gibbs chains (the two colours of the bipartite graph) on the same Gibbs sampler.
    Free energy F(v) = -b.v - sum_j log 2cosh(c_j + W_j.v) gives exact P(label | image).
    """

    def __init__(self, nv, nh, chains=100, lr=0.01, seed=0, init=None):
        rng = np.random.default_rng(seed)
        self.nv, self.nh, self.lr, self.t = nv, nh, lr, 0
        self.W = rng.normal(0, 0.01, (nv, nh))
        self.b = np.zeros(nv) if init is None else np.arctanh(np.clip(init, -0.95, 0.95))
        self.c = np.zeros(nh)
        J, h = self.ising()
        self.gibbs = Gibbs(J, h, chains=chains, colors=[np.arange(nv), np.arange(nv, nv + nh)], seed=seed + 1)

    @property
    def n(self):
        return self.nv + self.nh

    def ising(self):
        J = np.zeros((self.n, self.n))
        J[:self.nv, self.nv:] = self.W
        J[self.nv:, :self.nv] = self.W.T
        return J, np.concatenate([self.b, self.c])

    def fit_batch(self, V, k=1, lr=None):
        """One PCD-k update from a (B, nv) +-1 batch."""
        lr = self.lr if lr is None else lr
        G = np.tanh(V @ self.W + self.c)
        self.gibbs.sweep(k)
        S = self.gibbs.s
        Vn, Gn = S[:, :self.nv], np.tanh(S[:, :self.nv] @ self.W + self.c)
        self.W += lr * (V.T @ G / len(V) - Vn.T @ Gn / len(Vn))
        self.b += lr * (V.mean(0) - Vn.mean(0))
        self.c += lr * (G.mean(0) - Gn.mean(0))
        self.gibbs.set_weights(*self.ising())
        self.t += 1

    def free_energy(self, V):
        x = V @ self.W + self.c
        return -V @ self.b - (np.abs(x) + np.log1p(np.exp(-2 * np.abs(x)))).sum(-1)    # log 2cosh, stable

    def sampler(self, chains=64, seed=0):
        J, h = self.ising()
        return Gibbs(J, h, chains=chains, colors=[np.arange(self.nv), np.arange(self.nv, self.n)], seed=seed)


# ------------------------------------------------------------------ MRF denoising on an S x S grid
def grid_denoise(noisy_bits, side, J=0.5, lam=0.867, beta=1.0, sweeps=300, burn=None, chains=8, seed=0):
    """Ising posterior on the 4-neighbour grid with evidence h = lam * y: P(pixel = 1 | y) per pixel.

    Checkerboard updates with numpy shifts (no coupling matrix), so S = 256 (65 536 p-bits) is fine.
    """
    rng = np.random.default_rng(seed)
    y = 2.0 * np.asarray(noisy_bits, float).reshape(side, side) - 1.0
    h = lam * y
    s = np.broadcast_to(y, (chains, side, side)).copy()            # start from the noisy image
    yy, xx = np.mgrid[0:side, 0:side]
    masks = [((yy + xx) % 2) == c for c in (0, 1)]
    acc = np.zeros((side, side))
    burn = max(10, sweeps // 5) if burn is None else burn
    for t in range(burn + sweeps):
        for m in masks:
            nb = np.zeros_like(s)
            nb[:, 1:, :] += s[:, :-1, :]
            nb[:, :-1, :] += s[:, 1:, :]
            nb[:, :, 1:] += s[:, :, :-1]
            nb[:, :, :-1] += s[:, :, 1:]
            p = 1.0 / (1.0 + np.exp(-2.0 * beta * (J * nb + h)))
            new = np.where(rng.random(s.shape) < p, 1.0, -1.0)
            s = np.where(m, new, s)
        if t >= burn:
            acc += (s > 0).mean(0)
    return (acc / sweeps).ravel()


# ------------------------------------------------------------------ self-check
def _check():
    rng = np.random.default_rng(3)
    n = 10
    J = np.triu(rng.normal(0, 0.6, (n, n)), 1)
    J = J + J.T
    h = rng.normal(0, 0.4, n)
    p, _ = gd.exact_boltzmann(J, h, 1.0)
    g = Gibbs(J, h, chains=400, seed=1)
    S = g.run(100, burn=20)
    q = np.bincount(states(S), minlength=1 << n) / len(S)
    tv = 0.5 * np.abs(p - q).sum()
    print(f"pysim: n = {n} random dense model, {len(S)} samples, TV vs exact {tv:.4f}")
    # clamped conditional vs exact
    import ml
    free, pf, mf = ml.cond_dist(J, h, {0: 1, 3: 0})
    g.clamp([0, 3], [1, 0])
    m = g.marginals(200, burn=20)
    err = max(abs(m[i] - mf[k]) for k, i in enumerate(free))
    print(f"pysim: clamped marginals vs exact conditional, max error {err:.4f}")
    # sparse path: 64 x 64 grid as an explicit CSR matrix
    side = 64
    N = side * side
    rows, cols = [], []
    for i in range(N):
        if (i % side) + 1 < side:
            rows += [i, i + 1]
            cols += [i + 1, i]
        if i + side < N:
            rows += [i, i + side]
            cols += [i + side, i]
    Js = spm.csr_matrix((np.full(len(rows), 0.3), (rows, cols)), shape=(N, N))
    t0 = time.time()
    g = Gibbs(Js, np.zeros(N), chains=16, seed=2)
    g.sweep(50)
    print(f"pysim: {N}-node sparse grid, {len(g.colors)} colours, 50 sweeps x 16 chains in {time.time() - t0:.2f} s")
    ok = tv < 0.05 and err < 0.03
    print("pysim: PASS" if ok else "pysim: FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _check() else 1)
