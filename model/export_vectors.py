#!/usr/bin/env python3
"""Export reference vectors for the RTL testbenches into model/vectors/.

Files (all $readmemh-compatible, one entry per line):
  sigmoid_lut.hex            1024 x 16b probability table (used by RTL too)
  lfsr_lane<k>.hex           first 256 16-bit draws of noise lane k
  pbit_vectors.hex           {acc16, beta8, rand16, logit10(sx16), p16, bit4}
  pwl_vectors.hex            {logit16, p_pwl16}
  mac_vectors.hex            {J row 8x8b, h8, spins8, acc16}   (N = 8)
  sched_vectors.hex          beta after each of 64 sweeps for 3 schedule modes
  <case>/weights.hex         signed 8b, address {i, j} = i*2^clog2(N) + j (zero padded)
  <case>/bias.hex            N signed 8b
  <case>/order.hex           N node indices (visit order = colour blocks)
  <case>/beta_table.hex      16 x 8b beta table (schedule mode 2)
  <case>/case.vh             localparams for the testbench
  <case>/trace.hex           expected spin state (bit i = node i) after each sweep
  <case>/trace_pwl.hex       same with the piecewise-linear sigmoid
  <pgate>/pg_script.hex      clamp scenarios {mask16, val16, sweeps32} for tb_pgates.v
  <pgate>/pg_trace.hex       expected spins after each sweep of the whole script
  <pgate>/pg.vh              localparams for tb_pgates.v
  gauss_lut.hex              1024 x 16b inverse-CDF Gaussian table (pmode_array.v)
  pm_<case>/                 pmode_array.v case: a.hex {i, j}, h.hex, x0.hex, trace.hex
                             (all M x 16-bit states after every step), pm.vh
  sp_<case>_p<P>/            sparse_array.v case with P lanes:
    node_e<k>.hex            lane k, per round: {valid, node, h8} (valid = 0 -> lane idles)
    nbr_e<k>.hex             lane k, address {round, slot}: {j, J8} (unused slots 0)
    case.vh, beta_table.hex, trace.hex   as above (trace of SparseParallelSampler)
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cases as cs        # noqa: E402
import fixed as fx        # noqa: E402
import pgates as pg       # noqa: E402
import pmode as pm        # noqa: E402
import tempering as tp    # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors")
TRACE_SWEEPS = 2000


def w8(v):
    return f"{v & 0xFF:02x}"


def dump(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def case_schedule(c):
    """Constant beta for sampling cases; geometric anneal for maxcut16.
    Cases from problem files (model/problem.py) carry their own schedule."""
    if c.get("schedule"):
        return c["schedule"]
    if c["name"] == "maxcut16":
        lo, hi = 0.25, 4.0
        table = [fx.to_q(lo * (hi / lo) ** (k / 15), fx.BFRAC, fx.BB, signed=False) for k in range(16)]
        return dict(mode=2, beta_start=table[0], beta_end=table[-1], delta=0, sps=8, table=table)
    if c["name"] == "factor6":
        # geometric warm-up 0.5 -> 2.0 over 128 sweeps, then hold (button demo, docs/05)
        lo, hi = 0.5, 2.0
        table = [fx.to_q(lo * (hi / lo) ** (k / 15), fx.BFRAC, fx.BB, signed=False) for k in range(16)]
        return dict(mode=2, beta_start=table[0], beta_end=table[-1], delta=0, sps=8, table=table)
    b = c["beta_q"]
    return dict(mode=0, beta_start=b, beta_end=b, delta=0, sps=1, table=[b] * 16)


def export_case(c):
    d = os.path.join(OUT, c["name"])
    os.makedirs(d, exist_ok=True)
    n = c["n"]
    stride = 1 << fx.clog2(n)
    dump(os.path.join(d, "weights.hex"),
         [w8(c["Jq"][i][j]) if (i < n and j < n) else "00" for i in range(stride) for j in range(stride)])
    odig = (fx.clog2(n) + 3) // 4
    dump(os.path.join(d, "bias.hex"), [w8(v) for v in c["hq"]] + ["00"] * (stride - n))
    dump(os.path.join(d, "order.hex"), [f"{i:0{odig}x}" for i in c["order"]] + ["0" * odig] * (stride - n))
    sc = case_schedule(c)
    dump(os.path.join(d, "beta_table.hex"), [w8(v) for v in sc["table"]])
    smp = fx.FixedSampler(c["Jq"], c["hq"], c["order"], fx.Schedule(**sc), wfrac=c["wfrac"])
    _, _, trace = fx.run(smp, TRACE_SWEEPS, trace=True)
    hexw = (n + 3) // 4
    dump(os.path.join(d, "trace.hex"), [f"{t:0{hexw}x}" for t in trace])
    # same case through the piecewise-linear sigmoid (USE_PWL = 1)
    smp = fx.FixedSampler(c["Jq"], c["hq"], c["order"], fx.Schedule(**sc), wfrac=c["wfrac"], pwl=True)
    _, _, trace = fx.run(smp, TRACE_SWEEPS, trace=True)
    dump(os.path.join(d, "trace_pwl.hex"), [f"{t:0{hexw}x}" for t in trace])
    vh = [
        f"localparam integer CASE_N        = {n};",
        f"localparam integer CASE_WFRAC    = {c['wfrac']};",
        f"localparam integer CASE_MODE     = {sc['mode']};",
        f"localparam integer CASE_BETA0    = {sc['beta_start']};",
        f"localparam integer CASE_BETA1    = {sc['beta_end']};",
        f"localparam integer CASE_DELTA    = {sc['delta']};",
        f"localparam integer CASE_SPS      = {sc['sps']};",
        f"localparam integer CASE_SWEEPS   = {TRACE_SWEEPS};",
        f"localparam [31:0]  CASE_SEED     = 32'h{fx.LFSR_DEFAULT_SEED:08x};",
    ]
    dump(os.path.join(d, "case.vh"), vh)
    return sc


def sparse_layout(c, P):
    """Everything sparse_array.v needs for case c with P lanes (also used by fpga/host.py)."""
    n = c["n"]
    nbr = fx.dense_to_nbr(c["Jq"])
    D = max(1, max(len(r) for r in nbr))
    rounds = fx.engine_layout(c["colors"], P)
    return dict(n=n, P=P, D=D, IW=fx.clog2(n), DW=fx.clog2(D), LW=fx.clog2(len(rounds)),
                R=len(rounds), rounds=rounds, nbr=nbr)


def sparse_words(c, lay, e):
    """RAM images of lane e: node words {valid, node, h} and neighbour words {j, J}."""
    iw, dw, lw, D = lay["IW"], lay["DW"], lay["LW"], lay["D"]
    node, nbrw = [0] * (1 << lw), [0] * (1 << (lw + dw))
    for r, row in enumerate(lay["rounds"]):
        i = row[e]
        if i is None:
            continue
        node[r] = (1 << (iw + 8)) | (i << 8) | (c["hq"][i] & 0xFF)
        for d, (j, w) in enumerate(lay["nbr"][i]):
            assert d < D
            nbrw[(r << dw) | d] = (j << 8) | (w & 0xFF)
    return node, nbrw


def export_sparse(c, P, q16=None, share=False, suffix=""):
    """vectors/sp_<case>_p<P><suffix>/: lane RAM images + expected trace of SparseParallelSampler.

    q16 / share: noise options of sparse_array.v (NOISE_CTL = 1), see fixed.py.
    """
    name = f"sp_{c['name']}_p{P}{suffix}"
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    lay = sparse_layout(c, P)
    iw = lay["IW"]
    for e in range(P):
        node, nbrw = sparse_words(c, lay, e)
        dump(os.path.join(d, f"node_e{e}.hex"), [f"{v:0{(iw + 12) // 4}x}" for v in node])
        dump(os.path.join(d, f"nbr_e{e}.hex"), [f"{v:0{(iw + 11) // 4}x}" for v in nbrw])
    sc = case_schedule(c)
    dump(os.path.join(d, "beta_table.hex"), [w8(v) for v in sc["table"]])
    sweeps = TRACE_SWEEPS if c["n"] <= 16 else 300
    smp = fx.SparseParallelSampler(lay["nbr"], c["hq"], c["colors"], fx.Schedule(**sc), P=P, wfrac=c["wfrac"],
                                   q16=q16, share=share)
    _, _, trace = fx.run(smp, sweeps, trace=True)
    hexw = (c["n"] + 3) // 4
    dump(os.path.join(d, "trace.hex"), [f"{t:0{hexw}x}" for t in trace])
    dump(os.path.join(d, "case.vh"), [
        f"localparam integer CASE_N        = {c['n']};",
        f"localparam integer CASE_P        = {P};",
        f"localparam integer CASE_D        = {lay['D']};",
        f"localparam integer CASE_DW       = {lay['DW']};",
        f"localparam integer CASE_LW       = {lay['LW']};",
        f"localparam integer CASE_ROUNDS   = {lay['R']};",
        f"localparam integer CASE_WFRAC    = {c['wfrac']};",
        f"localparam integer CASE_MODE     = {sc['mode']};",
        f"localparam integer CASE_BETA0    = {sc['beta_start']};",
        f"localparam integer CASE_BETA1    = {sc['beta_end']};",
        f"localparam integer CASE_DELTA    = {sc['delta']};",
        f"localparam integer CASE_SPS      = {sc['sps']};",
        f"localparam integer CASE_SWEEPS   = {sweeps};",
        f"localparam [31:0]  CASE_SEED     = 32'h{fx.LFSR_DEFAULT_SEED:08x};",
        f"localparam integer CASE_NZ_TAU   = {int(q16 is not None)};",
        f"localparam integer CASE_NZ_SHARE = {int(share)};",
        f"localparam integer CASE_NZ_Q     = {q16 or 0};",
    ])
    return name, lay


def pgate_script(c):
    """Clamp scenarios as (label, mask, val, sweeps) with node-index bit masks."""
    out = []
    for label, cl, sw in pg.scenarios(c):
        mask = sum(1 << c["nodes"].index(w) for w in cl)
        val = sum(b << c["nodes"].index(w) for w, b in cl.items())
        out.append((label, mask, val, sw))
    return out


def pgate_trace(c, script, sampler=False):
    """model/fixed.py replay of tb_pgates.v: bias RAM rewritten per scenario, state kept.

    With a ladder (K > 1, model/tempering.py) every scenario starts with a
    resync and runs sw rounds; the trace is the rung-0 replica per round.
    """
    sc = case_schedule(c)
    beta = sc["beta_end"]                       # pgate.v runs at a constant beta
    if c.get("ladder"):
        beta = c["ladder"][0]
        smp = tp.TemperingSampler(c["Jq"], list(c["hq"]), c["order"], c["ladder"], wfrac=c["wfrac"])
    else:
        smp = fx.FixedSampler(c["Jq"], list(c["hq"]), c["order"], fx.Schedule(0, beta, beta), wfrac=c["wfrac"])
    tr = []
    for _, mask, val, sw in script:
        smp.h = [(127 if (val >> i) & 1 else -128) if (mask >> i) & 1 else c["hq"][i] for i in range(c["n"])]
        if c.get("ladder"):
            smp.pt.resync()
        for _ in range(sw):
            smp.sweep()
            tr.append(smp.state_index())
    return (beta, tr, smp) if sampler else (beta, tr)


def export_pgate(c):
    d = os.path.join(OUT, c["name"])
    script = pgate_script(c)
    beta, tr = pgate_trace(c, script)
    dump(os.path.join(d, "pg_script.hex"), [f"{m:04x}{v:04x}{sw:08x}" for _, m, v, sw in script])
    hexw = (c["n"] + 3) // 4
    dump(os.path.join(d, "pg_trace.hex"), [f"{t:0{hexw}x}" for t in tr])
    dump(os.path.join(d, "pg.vh"), [
        f"localparam integer PG_SCENARIOS = {len(script)};",
        f"localparam integer PG_SWEEPS    = {len(tr)};",
        f"localparam integer PG_BETA      = {beta};",
        f"localparam integer PG_K         = {len(c['ladder']) if c.get('ladder') else 1};",
        f"localparam [31:0]  PG_LADDER    = 32'h{ladder_word(c.get('ladder') or [16]):08x};",
    ])


def ladder_word(ladder):
    """{b3, b2, b1, b0} as packed for temper_ctrl.v LADDER (missing rungs 0)."""
    return sum((b & 0xFF) << (8 * k) for k, b in enumerate(ladder))


PM_X0 = {"well": [0.0], "tilt": [-1.0], "pair": [1.0, -1.0], "chain4": [0.5, -0.25, 1.0, -1.5]}
PM_STEPS = 2000


def export_pmode(name):
    """vectors/pm_<name>/: RAM images and expected trace of model/pmode.PmodeArray."""
    c = pm.case(name)
    m = c["m"]
    mw = fx.clog2(m)
    d = os.path.join(OUT, f"pm_{name}")
    os.makedirs(d, exist_ok=True)
    x0 = [fx.to_q(v, pm.XF, 16) for v in PM_X0[name]]
    dump(os.path.join(d, "a.hex"), [w8(c["Aq"][i][j]) if (i < m and j < m) else "00"
                                    for i in range(1 << mw) for j in range(1 << mw)])
    dump(os.path.join(d, "h.hex"), [w8(v) for v in c["hq"]] + ["00"] * ((1 << mw) - m))
    dump(os.path.join(d, "x0.hex"), [f"{v & 0xFFFF:04x}" for v in x0] + ["0000"] * ((1 << mw) - m))
    arr = pm.PmodeArray(c["Aq"], c["hq"], c["lamq"], c["k"], c["sig"], x0=x0)
    lines = []
    for _ in range(PM_STEPS):
        arr.step()
        lines.append("".join(f"{arr.x[i] & 0xFFFF:04x}" for i in reversed(range(m))))
    dump(os.path.join(d, "trace.hex"), lines)
    dump(os.path.join(d, "pm.vh"), [
        f"localparam integer PM_M     = {m};",
        f"localparam integer PM_MW    = {mw};",
        f"localparam integer PM_LAM   = {c['lamq']};",
        f"localparam integer PM_K     = {c['k']};",
        f"localparam integer PM_SIG   = {c['sig']};",
        f"localparam integer PM_STEPS = {PM_STEPS};",
        f"localparam [31:0]  PM_SEED  = 32'h{fx.LFSR_DEFAULT_SEED:08x};",
    ])
    return c


def export_units():
    rnd = random.Random(1234)
    dump(os.path.join(OUT, "sigmoid_lut.hex"), [f"{v:04x}" for v in fx.sigmoid_lut()])
    dump(os.path.join(OUT, "gauss_lut.hex"), [f"{v & 0xFFFF:04x}" for v in pm.gauss_table()])
    for k, m in enumerate(fx.LFSR_MASKS):
        s = fx.lane_seed(fx.LFSR_DEFAULT_SEED, k)
        lines = []
        for _ in range(256):
            s, r = fx.lfsr_draw16(s, m)
            lines.append(f"{r:04x}")
        dump(os.path.join(OUT, f"lfsr_lane{k}.hex"), lines)
    lut = fx.sigmoid_lut()
    lines = []
    for _ in range(4096):
        acc = rnd.randint(-2000, 2000)
        beta = rnd.randint(0, 255)
        r = rnd.randint(0, 65535)
        lg = fx.logit_q5(beta, acc)
        p = lut[lg + 512]
        lines.append(f"{acc & 0xFFFF:04x}{beta:02x}{r:04x}{lg & 0xFFFF:04x}{p:04x}{int(r < p):01x}")
    dump(os.path.join(OUT, "pbit_vectors.hex"), lines)
    dump(os.path.join(OUT, "pwl_vectors.hex"),
         [f"{(k - 512) & 0xFFFF:04x}{fx.sigmoid_pwl(k - 512):04x}" for k in range(1024)])
    lines = []
    for _ in range(512):
        row = [rnd.randint(-128, 127) for _ in range(8)]
        h = rnd.randint(-128, 127)
        sp = rnd.randint(0, 255)
        acc = h + sum(row[j] if (sp >> j) & 1 else -row[j] for j in range(8))
        rowhex = "".join(w8(row[j]) for j in reversed(range(8)))  # J_i7 ... J_i0
        lines.append(f"{rowhex}{w8(h)}{sp:02x}{acc & 0xFFFF:04x}")
    dump(os.path.join(OUT, "mac_vectors.hex"), lines)
    lines = []
    for sc in (dict(mode=0, beta_start=20, beta_end=20, delta=0, sps=1),
               dict(mode=1, beta_start=4, beta_end=60, delta=3, sps=5),
               dict(mode=2, beta_start=0, beta_end=0, delta=0, sps=3,
                    table=[2 + 9 * k for k in range(16)])):
        s = fx.Schedule(**sc)
        for _ in range(64):
            lines.append(w8(s.beta))
            s.sweep_done()
    dump(os.path.join(OUT, "sched_vectors.hex"), lines)


def main():
    os.makedirs(OUT, exist_ok=True)
    export_units()
    for c in cs.all_cases():
        sc = export_case(c)
        if "nodes" in c:
            export_pgate(c)
        print(f"{c['name']:9s} N={c['n']:2d} Q{7 - c['wfrac']}.{c['wfrac']} beta_q={c['beta_q']:3d} ({c['beta_q'] / 16:.4f}) "
              f"mode={sc['mode']} colours={len(c['colors'])} order={c['order']}")
    for c, p in cs.sparse_cases():
        name, lay = export_sparse(c, p)
        print(f"{name:20s} N={lay['n']:3d} P={p} D={lay['D']:2d} rounds={lay['R']}")
    for name in PM_X0:
        c = export_pmode(name)
        print(f"pm_{name:16s} M={c['m']} lambda={c['lam']} eps=2^-{c['k']} sig={c['sig']}")
    for c, p, q16, share, suf in cs.noise_cases():
        name, lay = export_sparse(c, p, q16=q16, share=share, suffix=suf)
        print(f"{name:20s} N={lay['n']:3d} P={p} q16={q16} share={share}")


if __name__ == "__main__":
    main()
