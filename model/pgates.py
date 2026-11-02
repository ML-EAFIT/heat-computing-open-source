#!/usr/bin/env python3
"""p-gates: Boolean gates as small Ising problems, and how they compose.

A p-gate is a set of p-bits with couplings J and biases h chosen so that the
lowest-energy states of H(s) = -sum_{i<j} J_ij s_i s_j - sum_i h_i s_i are
exactly the rows of the gate's truth table (bit 1 <-> spin +1). Sampling the
Boltzmann law P ~ exp(-beta H) with the p-bit array then puts most of the
probability on valid rows; beta sets how much.

  * compose : gates sharing a wire share a node, J and h of the gates add
  * clamp   : a large bias on a node pins it (forward = clamp inputs,
              backward = clamp outputs -> the circuit is invertible)

Tags:
  [DOCUMENTED] the energy / Boltzmann convention and sigma(2 beta field)
      [data/An efficient ... 2510.23972v2.pdf, p.5, Eqs. 10-11]
  [DOCUMENTED] "AND, OR, NOT ... by composing energy functions" is mentioned
      without coefficients [data/A Blueprint ... 2607.16183v2.pdf, p.3]
  [NOT IN SOURCES] the gadget coefficients below. AND/OR/COPY/NOT are the
      standard Ising gadgets; XOR was found by brute force over integer
      J, h in [-2, 2] with one auxiliary spin (search kept in check_xor()).
  [INFERRED] clamping = bias offset, as for the denoising chain (docs/02 2.3)

Run it to print every gate's truth table under the exact Boltzmann law.
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import golden as gd       # noqa: E402

# ------------------------------------------------------------------ gadgets
# Each gadget: ordered terminal names, J as {(a, b): w}, h as {a: w}.
# Terminal names are local; compose() renames them to circuit wires.
GADGETS = {
    "COPY": dict(terms=("A", "Y"), J={("A", "Y"): +1}, h={}),
    "NOT":  dict(terms=("A", "Y"), J={("A", "Y"): -1}, h={}),
    # valid rows score H = -3, every invalid row H >= +1 (gap 4)
    "AND":  dict(terms=("A", "B", "Y"),
                 J={("A", "B"): -1, ("A", "Y"): +2, ("B", "Y"): +2},
                 h={"A": +1, "B": +1, "Y": -2}),
    # OR(A,B) = NOT AND(NOT A, NOT B): flipping every spin negates h only
    "OR":   dict(terms=("A", "B", "Y"),
                 J={("A", "B"): -1, ("A", "Y"): +2, ("B", "Y"): +2},
                 h={"A": -1, "B": -1, "Y": +2}),
    # one auxiliary spin X; valid rows H = -7 (min over X), invalid >= -5
    "XOR":  dict(terms=("A", "B", "Y", "X"),
                 J={("A", "B"): -1, ("A", "Y"): -1, ("B", "Y"): -1,
                    ("A", "X"): -2, ("B", "X"): -2, ("Y", "X"): -2},
                 h={"A": -1, "B": -1, "Y": -1, "X": -2}),
}

TRUTH = {
    "COPY": lambda a: (a,),
    "NOT":  lambda a: (1 - a,),
    "AND":  lambda a, b: (a & b,),
    "OR":   lambda a, b: (a | b,),
    "XOR":  lambda a, b: (a ^ b,),
}
N_IN = {"COPY": 1, "NOT": 1, "AND": 2, "OR": 2, "XOR": 2}


class Circuit:
    """Named nodes + accumulated J, h. add() places a gadget on wires."""

    def __init__(self, inputs=()):
        self.nodes = []
        self.J = {}
        self.h = {}
        self.gates = []
        for w in inputs:
            self.node(w)

    def node(self, name):
        if name not in self.nodes:
            self.nodes.append(name)
        return self.nodes.index(name)

    def add(self, kind, *wires, scale=1):
        g = GADGETS[kind]
        if len(wires) == len(g["terms"]) - 1 and kind == "XOR":
            wires = wires + (f"{wires[-1]}_aux",)          # private aux spin
        assert len(wires) == len(g["terms"]), (kind, wires)
        ren = dict(zip(g["terms"], wires))
        for w in wires:
            self.node(w)
        for (a, b), v in g["J"].items():
            k = tuple(sorted((ren[a], ren[b]), key=self.nodes.index))
            self.J[k] = self.J.get(k, 0) + scale * v
        for a, v in g["h"].items():
            self.h[ren[a]] = self.h.get(ren[a], 0) + scale * v
        self.gates.append((kind, wires))
        return self

    def dense(self):
        n = len(self.nodes)
        Jm = [[0.0] * n for _ in range(n)]
        for (a, b), v in self.J.items():
            i, j = self.nodes.index(a), self.nodes.index(b)
            Jm[i][j] = Jm[j][i] = float(v)
        hv = [float(self.h.get(w, 0)) for w in self.nodes]
        return Jm, hv


def gate(kind, scale=1):
    """Single gate with wires named after its terminals (aux last)."""
    c = Circuit()
    c.add(kind, *GADGETS[kind]["terms"], scale=scale)
    return c


def half_adder():
    return Circuit(("a", "b")).add("XOR", "a", "b", "s").add("AND", "a", "b", "c")


def full_adder():
    """s = a ^ b ^ cin, cout = ab | cin(a ^ b), from 2 XOR, 2 AND, 1 OR."""
    c = Circuit(("a", "b", "cin"))
    c.add("XOR", "a", "b", "t").add("XOR", "t", "cin", "s")
    c.add("AND", "a", "b", "g").add("AND", "t", "cin", "k").add("OR", "g", "k", "cout")
    return c


def mult2x2():
    """2-bit x 2-bit multiplier, p = a * b (4 bits). 14 p-bits.

    p0 = a0 b0; p1 = a0 b1 ^ a1 b0, c1 = a0 b1 & a1 b0; p2 = a1 b1 ^ c1; p3 = a1 b1 & c1
    """
    c = Circuit(("a0", "a1", "b0", "b1"))
    c.add("AND", "a0", "b0", "p0").add("AND", "a0", "b1", "q01")
    c.add("AND", "a1", "b0", "q10").add("AND", "a1", "b1", "q11")
    c.add("XOR", "q01", "q10", "p1").add("AND", "q01", "q10", "c1")
    c.add("XOR", "q11", "c1", "p2").add("AND", "q11", "c1", "p3")
    return c


# clamp bias = the extremes of the Q3.4 bias word (+127/16, -128/16). Every
# terminal here has sum_j |J_ij| <= 6, so a clamped node's field keeps its sign.
CLAMP_HI, CLAMP_LO = 127 / 16, -8.0


def clamp(h, idx, bits):
    """Pin node idx[k] to bits[k] by overriding its bias with the extreme value."""
    h = list(h)
    for i, b in zip(idx, bits):
        h[i] = CLAMP_HI if b else CLAMP_LO
    return h


# ------------------------------------------------------------------ scenarios
def scenarios(c, sweeps=2000):
    """Clamp experiments for tb_pgates.v: list of (label, {wire: bit}, sweeps).

    free run, every input row forward, every output row backward. For the
    multiplier: run backwards on the products 6, 3, 9, 4 (factoring).
    """
    ins, outs = c["ins"], c["outs"]
    out = []
    if c["name"].startswith("factor6"):
        # with K replicas a round costs K sweeps: same sweep budget per scenario
        rounds = 2 * sweeps // len(c.get("ladder") or [0])
        for p in (6, 3, 9, 4):
            out.append((f"factor {p}", {w: (p >> k) & 1 for k, w in enumerate(outs)}, rounds))
        return out
    out.append(("free", {}, sweeps))
    for r in itertools.product((0, 1), repeat=len(ins)):
        out.append(("fwd " + "".join(map(str, r)), dict(zip(ins, r)), sweeps))
    for r in itertools.product((0, 1), repeat=len(outs)):
        out.append(("bwd " + "".join(map(str, r)), dict(zip(outs, r)), sweeps))
    return out


def rows_valid(c):
    """Valid (ins + outs) rows of the case's Boolean function."""
    name = c["name"]
    if name.startswith("pgate_"):
        return truth_rows(name[6:].upper())
    if name == "fulladder":
        return {(a, b, ci, (a + b + ci) & 1, int(a + b + ci >= 2))
                for a in (0, 1) for b in (0, 1) for ci in (0, 1)}
    rows = set()
    for a in range(4):
        for b in range(4):
            p = a * b
            rows.add((a & 1, a >> 1, b & 1, b >> 1) + tuple((p >> k) & 1 for k in range(4)))
    return rows


# ------------------------------------------------------------------ checks
def state_bits(n):
    idx = np.arange(1 << n)
    return (idx[:, None] >> np.arange(n)) & 1          # bit i = node i


def ground_rows(c, wires):
    """Set of projections onto `wires` of all minimum-energy states."""
    Jm, hv = c.dense()
    e, _ = gd.energy_all(Jm, hv)
    bits = state_bits(len(hv))
    cols = [c.nodes.index(w) for w in wires]
    return {tuple(int(x) for x in bits[s, cols]) for s in np.nonzero(e == e.min())[0]}


def marginal(c, wires, beta, h=None):
    """Exact Boltzmann law of the sub-state on `wires` (dict row -> prob)."""
    Jm, hv = c.dense()
    p, _ = gd.exact_boltzmann(Jm, h if h is not None else hv, beta)
    bits = state_bits(len(hv))
    cols = [c.nodes.index(w) for w in wires]
    out = {}
    for s in range(len(p)):
        k = tuple(int(x) for x in bits[s, cols])
        out[k] = out.get(k, 0.0) + p[s]
    return out


def truth_rows(kind):
    n = N_IN[kind]
    return {r + TRUTH[kind](*r) for r in itertools.product((0, 1), repeat=n)}


def check_gates():
    for kind in GADGETS:
        c = gate(kind)
        wires = list(GADGETS[kind]["terms"][:N_IN[kind] + 1])
        g = ground_rows(c, wires)
        assert g == truth_rows(kind), (kind, g)
    sums = {(a, b, ci): (a + b + ci) & 1 for a in (0, 1) for b in (0, 1) for ci in (0, 1)}
    fa = full_adder()
    g = ground_rows(fa, ["a", "b", "cin", "s", "cout"])
    assert g == {(a, b, ci, sums[a, b, ci], int(a + b + ci >= 2)) for a, b, ci in sums}, g
    m = mult2x2()
    g = ground_rows(m, ["a0", "a1", "b0", "b1", "p0", "p1", "p2", "p3"])
    want = set()
    for a in range(4):
        for b in range(4):
            p = a * b
            want.add((a & 1, a >> 1, b & 1, b >> 1, p & 1, (p >> 1) & 1, (p >> 2) & 1, p >> 3))
    assert g == want, g


def check_xor():
    """Brute force: XOR + one aux spin with integer J, h in [-2, 2] (not run by default)."""
    vals = range(-2, 3)
    st = np.array(list(itertools.product((-1, 1), repeat=4)))
    pairs = list(itertools.combinations(range(4), 2))
    found = []
    for Jv in itertools.product(vals, repeat=6):
        for hv in itertools.product(vals, repeat=4):
            e = -sum(Jv[k] * st[:, i] * st[:, j] for k, (i, j) in enumerate(pairs)) - st @ np.array(hv)
            e3 = {}
            for s, v in zip(st, e):
                k = tuple(s[:3])
                e3[k] = min(e3.get(k, 99), v)
            val = [v for k, v in e3.items() if (k[2] == 1) == ((k[0] == 1) != (k[1] == 1))]
            inv = [v for k, v in e3.items() if (k[2] == 1) != ((k[0] == 1) != (k[1] == 1))]
            if max(val) == min(val) and min(inv) > max(val):
                found.append((Jv, hv, min(inv) - max(val)))
    return found


def main():
    check_gates()
    beta = 1.0
    print(f"exact Boltzmann law of each gate at beta = {beta} (rows: inputs -> output)")
    for kind in GADGETS:
        c = gate(kind)
        wires = list(GADGETS[kind]["terms"][:N_IN[kind] + 1])
        m = marginal(c, wires, beta)
        valid = truth_rows(kind)
        pv = sum(p for r, p in m.items() if r in valid)
        print(f"  {kind:4s} P(valid row) = {pv:.4f}   " +
              "  ".join(f"{''.join(map(str, r))}:{m[r]:.3f}" for r in sorted(valid)))
    for name, c in (("half_adder", half_adder()), ("full_adder", full_adder()), ("mult2x2", mult2x2())):
        Jm, hv = c.dense()
        deg = max(sum(abs(v) for v in row) for row in Jm)
        print(f"  {name:10s} nodes={len(c.nodes):2d} max|h|={max(abs(v) for v in hv):.0f} "
              f"max sum|J| per node={deg:.0f} colours={len(gd.greedy_coloring(Jm))}")
    print("all gadget ground states == truth tables (gates, full adder, 2x2 multiplier)")


if __name__ == "__main__":
    main()
