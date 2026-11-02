#!/usr/bin/env python3
"""User problem files -> array cases (the "program" of the p-bit chip).

The chip has no instruction set: a program is an energy function
    H(s) = -sum_{i<j} J_ij s_i s_j - sum_i h_i s_i,   s_i in {-1, +1}
plus an update order and a beta schedule. Sampling at inverse temperature
beta puts most probability on low-H states; annealing beta up finds them.
Bit k of a read-back state is 1 <-> s_k = +1.

A problem file is JSON with a "type":

  ising    {"type": "ising", "n": 4, "edges": [[0, 1, 0.5], ...], "h": [0.1, ...], "beta": 1.0}
  maxcut   {"type": "maxcut", "n": 3, "edges": [[0, 1], [1, 2, 2.0], ...], "beta": 1.0}
           weight w on edge (i, j) becomes J_ij = -w, so low energy = large cut
  qubo     {"type": "qubo", "n": 3, "Q": [[i, j, q], ...], "beta": 2.0}
           minimise sum_k q_k x_i x_j over x in {0, 1} (i == j: linear term)
  circuit  {"type": "circuit", "gates": [["AND", "a", "b", "y"], ...],
            "inputs": ["a", "b"], "outputs": ["y"], "clamp": {"y": 1},
            "show": {"a": ["a0", "a1"]}, "beta": 2.0}
           gates from model/pgates.GADGETS (COPY NOT AND OR XOR); a clamp pins
           a wire, so clamping outputs runs the circuit backwards (docs/05).
           "show" prints groups of wires as little-endian integers.

Optional for every type:
  "schedule": {"beta_start": 0.5, "beta_end": 2.0, "sweeps_per_step": 8}
      anneal used by host.py anneal / solve / pgate: 16 geometric steps
      beta_start -> beta_end, sweeps_per_step sweeps each, then hold.
      Default: beta / 4 -> beta, 8 sweeps per step.
  "scale": s   multiply J and h by s before quantization (ising / maxcut / qubo;
      qubo defaults to the largest s <= 1 that fits the Q3.4 range).

Run  python3 model/problem.py file.json  to print the quantized program and,
for n <= 16, the exact Boltzmann ground states.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cases as cs        # noqa: E402
import fixed as fx        # noqa: E402
import golden as gd       # noqa: E402
import pgates as pg       # noqa: E402

QMAX = (2 ** (fx.WB - 1) - 1) / (1 << fx.WFRAC)     # 7.9375 for Q3.4


_WARNINGS = []     # warnings of the current build(), returned as c["warnings"]


def _warn(msg):
    _WARNINGS.append(msg)
    print(f"problem: warning: {msg}", file=sys.stderr)


def _schedule(spec, beta):
    s = spec or {}
    lo, hi = float(s.get("beta_start", beta / 4)), float(s.get("beta_end", beta))
    sps = int(s.get("sweeps_per_step", 8))
    table = [fx.to_q(lo * (hi / lo) ** (k / 15), fx.BFRAC, fx.BB, signed=False) for k in range(16)]
    return dict(mode=2, beta_start=table[0], beta_end=table[-1], delta=0, sps=sps, table=table)


def _check_range(Jf, hf, name):
    big = max([abs(v) for row in Jf for v in row] + [abs(v) for v in hf] + [0])
    if big > QMAX:
        _warn(f"{name}: |coefficient| up to {big:g} saturates at {QMAX} (Q3.4); set \"scale\"")
    err = max([abs(fx.to_q(v, fx.WFRAC, fx.WB) / 16 - v) for row in Jf for v in row] +
              [abs(fx.to_q(v, fx.WFRAC, fx.WB) / 16 - v) for v in hf] + [0])
    if err > 1e-9:
        _warn(f"{name}: quantization to 1/16 steps changes a coefficient by up to {err:.4f}")


def _ising(name, n, edges, h, beta, scale):
    Jf = [[0.0] * n for _ in range(n)]
    for i, j, w in edges:
        if not (0 <= i < n and 0 <= j < n) or i == j:
            raise ValueError(f"{name}: bad edge ({i}, {j}) for n = {n}")
        Jf[i][j] += scale * w
        Jf[j][i] = Jf[i][j]
    hf = [scale * v for v in h]
    _check_range(Jf, hf, name)
    return cs._finish(name, Jf, hf, beta, gd.greedy_coloring(Jf))


def qubo_to_ising(n, Q):
    """x = (1 + s)/2: sum q x_i x_j -> -sum J s_i s_j - sum h s_i + const."""
    J, h, const = {}, [0.0] * n, 0.0
    for i, j, q in Q:
        if i == j:
            h[i] -= q / 2
            const += q / 2
        else:
            a, b = min(i, j), max(i, j)
            J[(a, b)] = J.get((a, b), 0.0) - q / 4
            h[a] -= q / 4
            h[b] -= q / 4
            const += q / 4
    return [(a, b, w) for (a, b), w in J.items()], h, const


def load(path):
    with open(path) as f:
        spec = json.load(f)
    name = os.path.splitext(os.path.basename(path))[0]
    return build(spec, name)


def build(spec, name="problem"):
    del _WARNINGS[:]
    kind = spec.get("type")
    beta = float(spec.get("beta", 1.0))
    if kind == "circuit":
        circ = pg.Circuit(spec.get("inputs", ()))
        for g in spec["gates"]:
            if g[0] not in pg.GADGETS:
                raise ValueError(f"{name}: unknown gate {g[0]!r} (have {', '.join(pg.GADGETS)})")
            circ.add(g[0], *g[1:])
        clamp = {w: int(b) for w, b in spec.get("clamp", {}).items()}
        for w in clamp:
            if w not in circ.nodes:
                raise ValueError(f"{name}: clamp on unknown wire {w!r}")
        c = cs.pgate_case(name, circ, beta, spec.get("inputs", []), spec.get("outputs", []), clamp)
        c["gates"] = list(circ.gates)
        c["show"] = spec.get("show", {})
        c["hf_free"] = circ.dense()[1]           # biases before the clamps (live edits, web/)
    elif kind in ("ising", "maxcut"):
        n = int(spec["n"])
        edges = [(e[0], e[1], e[2] if len(e) > 2 else 1.0) for e in spec["edges"]]
        if kind == "maxcut":
            c = _ising(name, n, [(i, j, -w) for i, j, w in edges], [0.0] * n, beta, spec.get("scale", 1.0))
            c["edges"] = [(min(i, j), max(i, j)) for i, j, _ in edges]
            c["cut_w"] = [w for _, _, w in edges]
        else:
            c = _ising(name, n, edges, spec.get("h", [0.0] * n), beta, spec.get("scale", 1.0))
    elif kind == "qubo":
        n = int(spec["n"])
        Q = [(int(i), int(j), float(q)) for i, j, q in spec["Q"]]
        edges, h, const = qubo_to_ising(n, Q)
        big = max([abs(w) for *_, w in edges] + [abs(v) for v in h] + [1e-12])
        scale = spec.get("scale", min(1.0, QMAX / big))
        c = _ising(name, n, edges, h, beta, scale)
        c["Q"], c["qubo_scale"] = Q, scale
    else:
        raise ValueError(f"{name}: \"type\" must be ising, maxcut, qubo or circuit, got {kind!r}")
    c["kind"] = kind
    c["schedule"] = _schedule(spec.get("schedule"), beta)
    c["warnings"] = list(_WARNINGS)
    return c


# ------------------------------------------------------------------ decoding
def energy(c, s):
    """H of state index s with the quantized (programmed) J, h, in J units."""
    n = c["n"]
    x = [1 if (s >> i) & 1 else -1 for i in range(n)]
    e = -sum(c["hq"][i] * x[i] for i in range(n))
    e -= sum(c["Jq"][i][j] * x[i] * x[j] for i in range(n) for j in range(i + 1, n))
    return e / (1 << c["wfrac"])


def decode(c, s):
    """Human-readable lines for state index s (bit k = 1 <-> s_k = +1)."""
    n = c["n"]
    bit = [(s >> i) & 1 for i in range(n)]
    out = [f"energy {energy(c, s):+.4f}"]
    kind = c.get("kind")
    if kind == "circuit":
        w = dict(zip(c["nodes"], bit))
        wires = c["ins"] + c["outs"]
        out.append("  ".join(f"{k}={w[k]}" for k in wires))
        for g, ws in c.get("show", {}).items():
            out.append(f"{g} = {sum(w[x] << k for k, x in enumerate(ws))}")
        bad = []
        for gk, gw in c.get("gates", []):
            k = pg.N_IN[gk]
            if tuple(w[x] for x in gw[k:k + 1]) != pg.TRUTH[gk](*[w[x] for x in gw[:k]]):
                bad.append(f"{gk}{tuple(gw)}")
        clamp_bad = [x for x, b in c["clamps"].items() if w[x] != b]
        if c.get("gates"):
            out.append("all gates satisfied" if not bad else f"violated: {', '.join(bad)}")
        if clamp_bad:
            out.append(f"clamp not held on: {', '.join(clamp_bad)}")
    elif kind == "maxcut":
        cut = sum(wt for (i, j), wt in zip(c["edges"], c["cut_w"]) if bit[i] != bit[j])
        out.append(f"partition {''.join(map(str, bit))}  cut {cut:g}")
    elif kind == "qubo":
        val = sum(q * bit[i] * bit[j] for i, j, q in c["Q"])
        out.append(f"x = {bit}  objective {val:g}")
    else:
        out.append("s = [" + ", ".join("+1" if b else "-1" for b in bit) + "]")
    return out


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: problem.py file.json")
    c = load(sys.argv[1])
    n = c["n"]
    print(f"{c['name']}: {c['kind']}, n = {n} p-bits, beta = {c['beta']} (Q4.4 {c['beta_q']}), "
          f"colours {len(c['colors'])}, order {c['order']}")
    if "nodes" in c:
        print("nodes:", " ".join(f"{k}:{w}" for k, w in enumerate(c["nodes"])))
    print("h (x16):", c["hq"])
    print("J (x16) nonzero:", [(i, j, c["Jq"][i][j]) for i in range(n) for j in range(i + 1, n) if c["Jq"][i][j]])
    if n <= 16:
        import numpy as np
        p, _ = gd.exact_boltzmann(np.array(c["Jq"]) / 16, np.array(c["hq"]) / 16, c["beta_q"] / 16)
        e = [energy(c, s) for s in range(1 << n)]
        emin = min(e)
        gs = [s for s in range(1 << n) if math.isclose(e[s], emin)]
        print(f"exact: {len(gs)} ground state(s), P(ground) = {sum(p[s] for s in gs):.3f} at beta")
        for s in gs[:8]:
            print("  " + " | ".join(decode(c, s)))


if __name__ == "__main__":
    main()
