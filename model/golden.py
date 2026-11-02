#!/usr/bin/env python3
"""Floating-point golden model: chromatic Gibbs p-bit sampler + exact Boltzmann.

Reused logic (cited):
  * chromatic two-colour Gibbs sweep, per-site sigma(2*beta*field)
        [torx/helpers/_plots_sampling.py:L474-493]  (generalised to dense J)
  * exact enumeration of all 2^N states and Boltzmann weights e^{-beta H}
        [torx/06_ising_sampling_contrastive_divergence.py:L216-235]
  * energy H(s) = -sum_{(i,j)} J_ij s_i s_j - sum_i h_i s_i
        [torx/06_ising_sampling_contrastive_divergence.py:L99-108]
        [data/An efficient ... 2510.23972v2.pdf, p.5, Eq. 10]
  * colouring into blocks (DSATUR) [thrml/02_spin_models.py:L129-141]
  * total variation as in torx/06 [torx/06_ising_sampling_contrastive_divergence.py:L297]
State index convention (ours, also used by RTL): bit i of the index = node i.
"""
import numpy as np


def edges_from_J(J):
    n = len(J)
    return [(i, j) for i in range(n) for j in range(i + 1, n) if J[i][j] != 0]


def energy_all(J, h):
    """H(s) for every state; J symmetric with zero diagonal (pair counted once)."""
    J = np.asarray(J, float)
    h = np.asarray(h, float)
    n = len(h)
    idx = np.arange(1 << n)
    spins = 2 * ((idx[:, None] >> np.arange(n)) & 1) - 1
    pair = -0.5 * np.einsum("si,ij,sj->s", spins, J, spins)
    return pair - spins @ h, spins


def exact_boltzmann(J, h, beta):
    e, spins = energy_all(J, h)
    w = np.exp(-beta * (e - e.min()))
    return w / w.sum(), spins


def greedy_coloring(J):
    """Greedy colouring of the coupling graph (DSATUR in thrml; greedy suffices)."""
    n = len(J)
    color = [-1] * n
    for i in sorted(range(n), key=lambda k: -sum(1 for j in range(n) if J[k][j])):
        used = {color[j] for j in range(n) if J[i][j] and color[j] >= 0}
        c = 0
        while c in used:
            c += 1
        color[i] = c
    ncol = max(color) + 1
    return [[i for i in range(n) if color[i] == c] for c in range(ncol)]


def chromatic_gibbs(J, h, beta, colors, sweeps, chains, rng, burn_in=0):
    """Float chromatic Gibbs over many parallel chains; returns state-index samples."""
    J = np.asarray(J, float)
    h = np.asarray(h, float)
    n = len(h)
    s = rng.choice([-1.0, 1.0], size=(chains, n))
    out = []
    for t in range(sweeps):
        for block in colors:
            field = s @ J[:, block] + h[block]          # local field of block nodes
            p = 1.0 / (1.0 + np.exp(-2.0 * beta * field))
            s[:, block] = np.where(rng.random(p.shape) < p, 1.0, -1.0)
        if t >= burn_in:
            out.append(((s > 0).astype(np.int64) << np.arange(n)).sum(1))
    return np.concatenate(out)


def histogram(indices, n):
    return np.bincount(np.asarray(indices), minlength=1 << n).astype(float)


def tv(p, q):
    p = np.asarray(p, float) / np.sum(p)
    q = np.asarray(q, float) / np.sum(q)
    return 0.5 * np.abs(p - q).sum()


def kl(p_emp, q_ref, eps=1e-12):
    """KL(p_emp || q_ref) in nats; zero-count bins contribute 0."""
    p = np.asarray(p_emp, float) / np.sum(p_emp)
    q = np.asarray(q_ref, float) / np.sum(q_ref)
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / np.maximum(q[m], eps))))


if __name__ == "__main__":
    # torx/06 ring, original float parameters
    # [torx/06_ising_sampling_contrastive_divergence.py:L203-208]
    N, beta = 8, 1.5
    Jr = [0.65, -0.40, 0.55, 0.45, -0.35, 0.50, 0.30, -0.45]
    hr = [0.10, -0.15, 0.05, 0.12, -0.10, 0.08, -0.04, 0.02]
    J = np.zeros((N, N))
    for e, (i, j) in enumerate([(i, (i + 1) % N) for i in range(N)]):
        J[i, j] = J[j, i] = Jr[e]
    exact, _ = exact_boltzmann(J, hr, beta)
    rng = np.random.default_rng(0)
    smp = chromatic_gibbs(J, hr, beta, [list(range(0, N, 2)), list(range(1, N, 2))],
                          sweeps=240, chains=6000, rng=rng, burn_in=239)
    print(f"golden ring8: peak={exact.max():.4f}  TV(6000 chains, 240 sweeps)={tv(histogram(smp, N), exact):.4f}")
