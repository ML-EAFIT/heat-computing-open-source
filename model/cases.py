#!/usr/bin/env python3
"""Test problems shared by the exporter, the testbench comparison and the FPGA host.

Each case returns a dict with float parameters (the "physics") and the
quantized integer parameters actually programmed into the RTL.
"""
import networkx as nx

import fixed as fx
import golden as gd
import pgates as pg


def _dense(n, edges, weights):
    J = [[0.0] * n for _ in range(n)]
    for (i, j), w in zip(edges, weights):
        J[i][j] = J[j][i] = w
    return J


def _quantize(Jf, hf, wfrac):
    Jq = [[fx.to_q(v, wfrac, fx.WB) for v in row] for row in Jf]
    hq = [fx.to_q(v, wfrac, fx.WB) for v in hf]
    return Jq, hq


def _order(colors):
    return [i for block in colors for i in block]


def ring8(wfrac=fx.WFRAC, name="ring8"):
    """8-site Ising ring of torx/06 [torx/06_ising_sampling_contrastive_divergence.py:L203-208]."""
    n, beta = 8, 1.5
    Jr = [0.65, -0.40, 0.55, 0.45, -0.35, 0.50, 0.30, -0.45]
    hr = [0.10, -0.15, 0.05, 0.12, -0.10, 0.08, -0.04, 0.02]
    Jf = _dense(n, [(i, (i + 1) % n) for i in range(n)], Jr)
    colors = [list(range(0, n, 2)), list(range(1, n, 2))]
    return _finish(name, Jf, hr, beta, colors, wfrac)


def maxcut(n, seed, beta, name):
    """Random 3-regular Max-Cut graph as in [torx/helpers/_nb09_maxcut.py:L31-44].

    Antiferromagnetic J_ij = -1 on edges so low energy = large cut
    (cut = sum (1 - s_i s_j)/2 [torx/helpers/_nb09_maxcut.py:L52-60]).
    """
    trial = seed
    g = nx.random_regular_graph(3, n, seed=trial)
    while not nx.is_connected(g):
        trial += 1
        g = nx.random_regular_graph(3, n, seed=trial)
    edges = sorted((min(i, j), max(i, j)) for i, j in g.edges())
    Jf = _dense(n, edges, [-1.0] * len(edges))
    colors = gd.greedy_coloring(Jf)
    c = _finish(name, Jf, [0.0] * n, beta, colors)
    c["edges"] = edges
    return c


def grid16():
    """4x4 periodic ferromagnet of torx/16 [torx/16_gibbs_sampling_factor_graph.py:L136-145]."""
    g = 4
    n = g * g
    edges = set()
    for i in range(n):
        r, c = divmod(i, g)
        for rr, cc in ((r + 1, c), (r, c + 1)):
            edges.add(tuple(sorted((i, (rr % g) * g + cc % g))))
    edges = sorted(edges)
    Jf = _dense(n, edges, [1.0] * len(edges))
    # checkerboard colouring [torx/helpers/_plots_schematics.py:L1668-1674]
    colors = [[i for i in range(n) if (i // g + i % g) % 2 == p] for p in (0, 1)]
    return _finish("grid16", Jf, [0.0] * n, 0.45, colors)


# DTM grid connectivity rules [data/An efficient ... 2510.23972v2.pdf, p.24 Table II]
G_PATTERNS = {
    "G8": [(0, 1), (4, 1)],
    "G12": [(0, 1), (4, 1), (9, 10)],
    "G16": [(0, 1), (4, 1), (8, 7), (14, 9)],
}


def g_edges(L, rules):
    """Rule (a, b) links (x, y) to (x+a, y+b), (x-b, y+a), (x-a, y-b), (x+b, y-a);
    edges leaving the grid are not formed [same pdf, p.24 App. D.2]. Every rule
    has a+b odd, so the graph is bipartite under the checkerboard colouring."""
    edges = set()
    for x in range(L):
        for y in range(L):
            for a, b in rules:
                for dx, dy in ((a, b), (-b, a), (-a, -b), (b, -a)):
                    u, v = x + dx, y + dy
                    if 0 <= u < L and 0 <= v < L:
                        edges.add(tuple(sorted((x * L + y, u * L + v))))
    return sorted(edges)


def gridG(L, pattern, kind, beta, seed=1, name=None):
    """L x L DTM-style grid: ferromagnet (J = +0.25) or +-0.5 spin glass, h = 0.

    [DOCUMENTED] pattern and open boundary (DTM p.24); [FPGA-APPROX] the
    weights: DTM trains them, the sources give no weight file (README Q6).
    """
    import random
    n = L * L
    edges = g_edges(L, G_PATTERNS[pattern])
    rnd = random.Random(seed)
    w = [0.25] * len(edges) if kind == "ferro" else [rnd.choice((-0.5, 0.5)) for _ in edges]
    Jf = _dense(n, edges, w)
    colors = [[i for i in range(n) if (i // L + i % L) % 2 == p] for p in (0, 1)]
    c = _finish(name or f"{pattern.lower()}_{kind}{L}", Jf, [0.0] * n, beta, colors)
    c["edges"] = edges
    return c


def sparse_cases():
    """(case, P) pairs run on sparse_array.v. Small ones have an exact Boltzmann law."""
    small = [ring8(), maxcut(12, 7, 0.5, "maxcut12"), grid16()]
    big = [gridG(16, "G12", "ferro", 0.5), gridG(16, "G12", "glass", 1.0)]
    return [(c, p) for c in small for p in (1, 2, 4)] + [(c, p) for c in big for p in (1, 4)]


def noise_cases():
    """sparse_array.v noise options (NOISE_CTL = 1): (case, P, q16, share, suffix).
    q16 = probability the RNG has decorrelated per update, in 1/65536."""
    r8, g16 = ring8(), grid16()
    return [(r8, 4, 16384, False, "_tau"), (g16, 4, 8192, False, "_tau"),
            (g16, 4, None, True, "_share"), (g16, 4, 32768, True, "_tsh")]


def pgate_case(name, circ, beta, ins, outs, clamps=None, ladder=None):
    """A p-gate circuit as an array case (docs/05). clamps: {wire: bit}.
    ladder: beta_q per rung (rung 0 = target) -> pgate.v with K = len(ladder)
    replicas (parallel tempering, model/tempering.py)."""
    Jf, hf = circ.dense()
    if clamps:
        hf = pg.clamp(hf, [circ.nodes.index(w) for w in clamps], list(clamps.values()))
    c = _finish(name, Jf, hf, beta, gd.greedy_coloring(Jf))
    c["nodes"] = list(circ.nodes)
    c["clamps"] = dict(clamps or {})
    c["ins"], c["outs"] = list(ins), list(outs)
    c["ladder"] = list(ladder) if ladder else None
    return c


def pgate_cases():
    """Free-running gates sample their own truth table; factor6 runs the multiplier backwards."""
    p4 = ["p0", "p1", "p2", "p3"]
    return [pgate_case("pgate_not", pg.gate("NOT"), 1.0, "A", "Y"),
            pgate_case("pgate_and", pg.gate("AND"), 1.0, "AB", "Y"),
            pgate_case("pgate_or", pg.gate("OR"), 1.0, "AB", "Y"),
            pgate_case("pgate_xor", pg.gate("XOR"), 1.5, "AB", "Y"),
            pgate_case("fulladder", pg.full_adder(), 1.5, ["a", "b", "cin"], ["s", "cout"]),
            pgate_case("factor6", pg.mult2x2(), 2.0, ["a0", "a1", "b0", "b1"], p4,
                       dict(zip(p4, (0, 1, 1, 0)))),
            # same multiplier and script, K = 4 replicas at beta 2, 1.25, 0.75, 0.375
            pgate_case("factor6_pt", pg.mult2x2(), 2.0, ["a0", "a1", "b0", "b1"], p4,
                       dict(zip(p4, (0, 1, 1, 0))), ladder=PT_LADDER)]


PT_LADDER = [32, 20, 12, 6]


def _finish(name, Jf, hf, beta, colors, wfrac=fx.WFRAC):
    Jq, hq = _quantize(Jf, hf, wfrac)
    bq = fx.to_q(beta, fx.BFRAC, fx.BB, signed=False)
    return dict(name=name, n=len(hf), Jf=Jf, hf=list(hf), beta=beta, Jq=Jq, hq=hq, beta_q=bq, wfrac=wfrac,
                colors=colors, order=_order(colors))


def all_cases():
    return [ring8(), ring8(6, "ring8_q6"), maxcut(12, 7, 0.5, "maxcut12"), grid16(),
            maxcut(16, 3, 0.5, "maxcut16")] + pgate_cases()
