#!/usr/bin/env python3
"""Compare RTL results with the Python models.

  compare.py --cases ring8 maxcut12 --sweeps 200000 --out build/stats.json
      For each case: (1) the RTL post-sweep trace must equal model/fixed.py
      bit for bit over the full run; (2) the RTL histogram is compared (TV, KL)
      with the exact Boltzmann law of the programmed (quantized) parameters,
      the exact Boltzmann law of the original float parameters, the exact
      stationary law of the quantized sweep kernel, and the float golden
      chromatic-Gibbs model (model/golden.py).

  compare.py --sparse-stats sp_ring8_p4 sp_g12_ferro16_p4 --out build/sparse_stats.json
      sparse_array.v runs (build/<case>_rtl_trace.hex): the trace must equal
      model/fixed.SparseParallelSampler over the full run. N <= 16: TV/KL vs
      the exact Boltzmann law and the iid floor. N = 256 (no exact law):
      energy per spin and |m| vs a float chromatic-Gibbs reference (many
      independent chains), with batch-means standard errors.

  compare.py --top-dump build/top_uart_dump.txt
      Replays the tb_top.v UART session in model/fixed.py and requires the
      received byte stream to match byte for byte.

  compare.py --top-sparse-dump build/top_sparse_uart_dump.txt
      Same for tb_top_sparse.v (SPARSE top, 16 x 16 G12 spin glass, P = 4),
      replayed in model/fixed.SparseParallelSampler.

  compare.py --anneal
      Max-Cut annealing quality of the maxcut16 schedule (Python mirror of
      the bit-exact RTL) against brute force.

  compare.py --pgates pgate_and fulladder factor6 --out build/pgates.json
      Truth-table statistics of the tb_pgates.v dumps (build/pgate_<case>_dump.hex):
      per clamp scenario, the law of the (inputs, outputs) row from the RTL
      against the exact stationary law of the quantized kernel, P(valid row),
      and the logic check (forward: most likely output = truth table;
      backward / factoring: the inputs land on pre-images of the clamp).
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
import cases as cs          # noqa: E402
import export_vectors as ev  # noqa: E402
import fixed as fx          # noqa: E402
import golden as gd         # noqa: E402
import pgates as pg         # noqa: E402

BURN_IN = 1000


def case_by_name(name):
    return {c["name"]: c for c in cs.all_cases()}[name]


def read_trace(path):
    with open(path) as f:
        return np.array([int(x, 16) for x in f.read().split()], dtype=np.int64)


def python_trace(c, sweeps):
    smp = fx.FixedSampler(c["Jq"], c["hq"], c["order"], fx.Schedule(**ev.case_schedule(c)), wfrac=c["wfrac"])
    _, _, tr = fx.run(smp, sweeps, trace=True)
    return np.array(tr, dtype=np.int64)


def mag_dist(p, n):
    pop = np.array([bin(i).count("1") for i in range(1 << n)])
    return np.bincount(pop, weights=p, minlength=n + 1)


def stats_case(name, sweeps, rng):
    c = case_by_name(name)
    n = c["n"]
    rtl = read_trace(os.path.join(HERE, "build", f"{name}_rtl_trace.hex"))
    py = python_trace(c, len(rtl))
    mism = int(np.sum(rtl != py))
    samples = rtl[BURN_IN:]
    m = len(samples)
    hist = gd.histogram(samples, n)
    scale = 1 << c["wfrac"]
    Jq = np.array(c["Jq"]) / scale
    hq = np.array(c["hq"]) / scale
    bq = c["beta_q"] / 16.0
    exact_q, _ = gd.exact_boltzmann(Jq, hq, bq)
    exact_f, _ = gd.exact_boltzmann(c["Jf"], c["hf"], c["beta"])
    stat_fx = fx.fixed_stationary(c["Jq"], c["hq"], c["order"], c["beta_q"], wfrac=c["wfrac"])
    # golden float model with the same programmed parameters, many short chains
    chains = 2000
    keep = max(1, m // chains)
    gold = gd.chromatic_gibbs(Jq, hq, bq, c["colors"], sweeps=200 + keep, chains=chains, rng=rng, burn_in=200)
    ghist = gd.histogram(gold, n)
    iid = gd.histogram(rng.choice(1 << n, size=m, p=exact_q), n)
    r = dict(
        case=name, n=n, wfrac=c["wfrac"], beta=c["beta"], beta_q=bq, sweeps=int(len(rtl)), samples=m,
        trace_mismatches=mism,
        tv_rtl_exact_q=gd.tv(hist, exact_q), kl_rtl_exact_q=gd.kl(hist, exact_q),
        tv_rtl_exact_float=gd.tv(hist, exact_f),
        tv_rtl_fixed_stationary=gd.tv(hist, stat_fx),
        tv_fixed_stationary_exact_q=gd.tv(stat_fx, exact_q),
        tv_exact_q_exact_float=gd.tv(exact_q, exact_f),
        tv_rtl_golden=gd.tv(hist, ghist), tv_golden_exact_q=gd.tv(ghist, exact_q),
        golden_samples=int(len(gold)),
        tv_iid_floor=gd.tv(iid, exact_q),
        mag_tv_rtl_exact_q=gd.tv(mag_dist(hist / hist.sum(), n), mag_dist(exact_q, n)),
    )
    r["hist_rtl"] = (hist / hist.sum()).tolist() if n <= 12 else None
    r["exact_q"] = exact_q.tolist() if n <= 12 else None
    r["golden"] = (ghist / ghist.sum()).tolist() if n <= 12 else None
    r["mag_rtl"] = mag_dist(hist / hist.sum(), n).tolist()
    r["mag_exact_q"] = mag_dist(exact_q, n).tolist()
    return r


def sparse_case(tag):
    """'sp_<case>_p<P>' -> (case dict, P)."""
    name, p = tag[3:].rsplit("_p", 1)
    return {c["name"]: c for c, _ in cs.sparse_cases()}[name], int(p)


def read_trace_big(path):
    with open(path) as f:
        return [int(x, 16) for x in f.read().split()]


def spins_pm(trace, n):
    """State integers -> (samples, n) array of +-1."""
    bits = np.array([[(t >> i) & 1 for i in range(n)] for t in trace], dtype=np.int8)
    return 2.0 * bits - 1.0


def energy_mag(S, J, h):
    """Per-sample energy per spin E/N = -(s.J.s/2 + h.s)/N and |m|."""
    n = S.shape[1]
    return -(0.5 * np.einsum("ki,ij,kj->k", S, J, S) + S @ h) / n, np.abs(S.mean(1))


def batch_se(x, nb=20):
    b = np.array_split(x, nb)
    return float(np.std([v.mean() for v in b], ddof=1) / np.sqrt(nb))


def sparse_stats_case(tag, rng):
    c, P = sparse_case(tag)
    n = c["n"]
    rtl = read_trace_big(os.path.join(HERE, "build", f"{tag}_rtl_trace.hex"))
    lay = ev.sparse_layout(c, P)
    smp = fx.SparseParallelSampler(lay["nbr"], c["hq"], c["colors"], fx.Schedule(**ev.case_schedule(c)),
                                   P=P, wfrac=c["wfrac"])
    _, _, py = fx.run(smp, len(rtl), trace=True)
    mism = sum(1 for a, b in zip(rtl, py) if a != b)
    scale = 1 << c["wfrac"]
    Jq, hq, bq = np.array(c["Jq"]) / scale, np.array(c["hq"]) / scale, c["beta_q"] / 16.0
    r = dict(case=tag, n=n, P=P, D=lay["D"], sweeps=len(rtl), trace_mismatches=mism)
    if n <= 16:
        samples = np.array(rtl[BURN_IN:], dtype=np.int64)
        hist = gd.histogram(samples, n)
        exact_q, _ = gd.exact_boltzmann(Jq, hq, bq)
        iid = gd.histogram(rng.choice(1 << n, size=len(samples), p=exact_q), n)
        r.update(samples=len(samples), tv_rtl_exact_q=gd.tv(hist, exact_q), kl_rtl_exact_q=gd.kl(hist, exact_q),
                 tv_iid_floor=gd.tv(iid, exact_q))
    else:
        burn = len(rtl) // 10
        e, m = energy_mag(spins_pm(rtl[burn:], n), Jq, hq)
        # float reference: independent chains, random start, same programmed parameters
        chains, gsweeps, gburn = 256, 3000, 1000
        S = rng.choice([-1.0, 1.0], size=(chains, n))
        ge, gm = [], []
        for t in range(gsweeps):
            for block in c["colors"]:
                f = S @ Jq[:, block] + hq[block]
                pr = 1.0 / (1.0 + np.exp(-2.0 * bq * f))
                S[:, block] = np.where(rng.random(pr.shape) < pr, 1.0, -1.0)
            if t >= gburn and t % 10 == 0:
                a, b = energy_mag(S, Jq, hq)
                ge.append(a)
                gm.append(b)
        ge, gm = np.array(ge), np.array(gm)
        r.update(samples=len(e), e_rtl=float(e.mean()), e_rtl_se=batch_se(e),
                 m_rtl=float(m.mean()), m_rtl_se=batch_se(m),
                 e_gold=float(ge.mean()), e_gold_se=float(ge.mean(0).std(ddof=1) / np.sqrt(chains)),
                 m_gold=float(gm.mean()), m_gold_se=float(gm.mean(0).std(ddof=1) / np.sqrt(chains)))
    return r


def do_sparse_stats(args):
    rng = np.random.default_rng(2026)
    out = []
    for tag in args.sparse_stats:
        r = sparse_stats_case(tag, rng)
        out.append(r)
        head = f"{tag:20s} N={r['n']:3d} P={r['P']} D={r['D']:2d} sweeps={r['sweeps']:6d} trace mism={r['trace_mismatches']}"
        if "tv_rtl_exact_q" in r:
            print(f"{head}  TV rtl/exact={r['tv_rtl_exact_q']:.4f} KL={r['kl_rtl_exact_q']:.4f} "
                  f"TV iid floor={r['tv_iid_floor']:.4f}")
        else:
            print(f"{head}  E/N rtl={r['e_rtl']:+.4f}+-{r['e_rtl_se']:.4f} gold={r['e_gold']:+.4f}+-{r['e_gold_se']:.4f}  "
                  f"|m| rtl={r['m_rtl']:.4f}+-{r['m_rtl_se']:.4f} gold={r['m_gold']:.4f}+-{r['m_gold_se']:.4f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f)
    bad = [r["case"] for r in out if r["trace_mismatches"]]
    print("compare --sparse-stats: PASS (all RTL traces bit-exact)" if not bad
          else f"compare --sparse-stats: FAIL trace mismatch in {bad}")
    return 0 if not bad else 1


def do_stats(args):
    rng = np.random.default_rng(2026)
    out = []
    print(f"{'case':9s} {'fmt':5s} {'samples':>8s} {'trace':>6s} {'TV rtl/exact':>12s} {'KL':>8s} "
          f"{'TV rtl/fxstat':>13s} {'TV rtl/golden':>13s} {'TV gold/exact':>13s} {'TV iid floor':>12s} "
          f"{'TV quant':>8s} {'TV mag':>7s}")
    for name in args.cases:
        r = stats_case(name, args.sweeps, rng)
        out.append(r)
        print(f"{r['case']:9s} Q{7 - r['wfrac']}.{r['wfrac']} {r['samples']:8d} {r['trace_mismatches']:6d} "
              f"{r['tv_rtl_exact_q']:12.4f} {r['kl_rtl_exact_q']:8.4f} {r['tv_rtl_fixed_stationary']:13.4f} "
              f"{r['tv_rtl_golden']:13.4f} {r['tv_golden_exact_q']:13.4f} {r['tv_iid_floor']:12.4f} "
              f"{r['tv_exact_q_exact_float']:8.4f} {r['mag_tv_rtl_exact_q']:7.4f}")
    print("columns: trace = RTL vs fixed.py post-sweep mismatches over the whole run; "
          "TV quant = exact(quantized params) vs exact(original float params); "
          "TV mag = magnetisation-distribution TV")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f)
    bad = [r["case"] for r in out if r["trace_mismatches"]]
    print("compare: PASS (all RTL traces bit-exact)" if not bad else f"compare: FAIL trace mismatch in {bad}")
    return 0 if not bad else 1


def le(v, nbytes):
    return [(v >> (8 * k)) & 0xFF for k in range(nbytes)]


def do_top(path):
    """Mirror of tb_top.v. N = 16, HIST_BITS = 12."""
    c = case_by_name("grid16")
    n, hb = 16, 12
    expected = [n, hb, ord("K")] + [ord("K")] * (256 + 32) + [ord("K")] * 3   # P, W, H/O, B X C
    sched = fx.Schedule(mode=0, beta_start=7, beta_end=7, delta=0, sps=1)
    smp = fx.FixedSampler(c["Jq"], c["hq"], c["order"], sched, seed=0x12345678, wfrac=4)
    smp.s = [0] * n
    hist = [0] * (1 << hb)
    up = [0] * n
    ns = 0

    def sweeps(k, stream=False):
        nonlocal ns
        outb = []
        for _ in range(k):
            smp.sweep()
            idx = smp.state_index()
            hist[idx & ((1 << hb) - 1)] += 1
            for i in range(n):
                up[i] += smp.s[i]
            ns += 1
            if stream:
                outb += le(idx, 2)
        return outb

    sweeps(2000)
    expected += [ord("K")]                                     # R
    expected += le(ns, 4) + sum((le(u, 4) for u in up), []) + [ord("K")]   # U
    expected += le(smp.state_index(), 2) + [ord("K")]          # Q
    expected += sweeps(5, stream=True) + [ord("K")]            # S 5
    expected += sum((le(h, 4) for h in hist), []) + [ord("K")]  # D
    sched.reset()                                              # button S2
    sweeps(128)
    expected += le(smp.state_index(), 2) + [ord("K")]          # Q
    expected += [ord("?")]
    with open(path) as f:
        got = [int(x, 16) for x in f.read().split()]
    nm = sum(1 for a, b in zip(got, expected) if a != b) + abs(len(got) - len(expected))
    print(f"top UART replay: received {len(got)} bytes, expected {len(expected)}, mismatching bytes: {nm}")
    print(f"  ping reply N={got[0] if got else '-'} HIST_BITS={got[1] if len(got) > 1 else '-'}; "
          f"final spins 0x{smp.state_index():04x}; samples counted {ns - 128 - 5}+5+128")
    print("compare --top-dump: PASS" if nm == 0 else "compare --top-dump: FAIL")
    return 0 if nm == 0 else 1


def do_top_sparse(path):
    """Mirror of tb_top_sparse.v. N = 256, P = 4, HIST_BITS = 12."""
    c = {x["name"]: x for x, _ in cs.sparse_cases()}["g12_glass16"]
    P, hb = 4, 12
    lay = ev.sparse_layout(c, P)
    n, nb = c["n"], c["n"] // 8
    n_e = sum(1 for e in range(P) for r in range(lay["R"]) if lay["rounds"][r][e] is not None
              for _ in lay["nbr"][lay["rounds"][r][e]])
    expected = le(n, 1) + [0x80 | hb, n >> 8, P, lay["D"], lay["LW"], ord("K")]       # P
    expected += [ord("K")] * (1 + P * lay["R"] + n_e + 3)                              # G M E B X C
    beta = ev.case_schedule(c)["beta_start"]
    sched = fx.Schedule(mode=0, beta_start=beta, beta_end=beta, delta=0, sps=1)
    smp = fx.SparseParallelSampler(lay["nbr"], c["hq"], c["colors"], sched, P=P, seed=0x12345678, wfrac=c["wfrac"])
    hist = [0] * (1 << hb)
    up = [0] * n
    ns = 0

    def sweeps(k, stream=False):
        nonlocal ns
        outb = []
        for _ in range(k):
            smp.sweep()
            idx = smp.state_index()
            hist[idx & ((1 << hb) - 1)] += 1
            for i in range(n):
                up[i] += smp.s[i]
            ns += 1
            if stream:
                outb += le(idx, nb)
        return outb

    sweeps(300)
    expected += [ord("K")]                                                     # R
    expected += le(ns, 4) + sum((le(u, 4) for u in up), []) + [ord("K")]       # U
    expected += le(smp.state_index(), nb) + [ord("K")]                         # Q
    expected += sweeps(3, stream=True) + [ord("K")]                            # S 3
    expected += sum((le(h, 4) for h in hist), []) + [ord("K")]                 # D
    expected += le(smp.state_index(), nb) + [ord("K")]                         # Q
    expected += [ord("?")]
    with open(path) as f:
        got = [int(x, 16) for x in f.read().split()]
    nm = sum(1 for a, b in zip(got, expected) if a != b) + abs(len(got) - len(expected))
    print(f"top sparse UART replay: received {len(got)} bytes, expected {len(expected)}, mismatching bytes: {nm}")
    print(f"  ping {got[:7]}; {n_e} neighbour words programmed; final magnetisation "
          f"{sum(2 * b - 1 for b in smp.s)}")
    print("compare --top-sparse-dump: PASS" if nm == 0 else "compare --top-sparse-dump: FAIL")
    return 0 if nm == 0 else 1


def do_anneal():
    """Annealing quality on maxcut16 (schedule of model/export_vectors.case_schedule)."""
    c = case_by_name("maxcut16")
    n = c["n"]
    cuts, opt, nopt = gd_cuts(c)
    sc = ev.case_schedule(c)
    total = 16 * sc["sps"]
    found = []
    for seed in range(1, 201):
        smp = fx.FixedSampler(c["Jq"], c["hq"], c["order"], fx.Schedule(**sc), seed=seed)
        best = 0
        for _ in range(total + 32):
            smp.sweep()
            best = max(best, cuts[smp.state_index()])
        found.append((best, cuts[smp.state_index()]))
    best = np.array([b for b, _ in found])
    final = np.array([f for _, f in found])
    print(f"maxcut16: |E|={len(c['edges'])}, brute-force optimum cut = {opt} ({nopt} optimal states of 65536)")
    print(f"  {len(found)} anneal runs ({total} sweeps beta {sc['table'][0]/16:.2f}->{sc['table'][-1]/16:.2f} + 32 at max): "
          f"final state optimal in {np.mean(final == opt) * 100:.1f}% of runs, "
          f"best-seen optimal in {np.mean(best == opt) * 100:.1f}%, mean final cut {final.mean():.2f}")
    rand = cuts.mean()
    print(f"  reference: mean cut of a uniformly random partition = {rand:.2f}")
    return dict(opt=int(opt), final_opt_rate=float(np.mean(final == opt)), best_opt_rate=float(np.mean(best == opt)),
                mean_final=float(final.mean()), random_mean=float(rand))


def gd_cuts(c):
    n = c["n"]
    idx = np.arange(1 << n)
    spins = 2 * ((idx[:, None] >> np.arange(n)) & 1) - 1
    cut = np.zeros(1 << n)
    for i, j in c["edges"]:
        cut += (1 - spins[:, i] * spins[:, j]) / 2           # [torx/helpers/_nb09_maxcut.py:L52-60]
    opt = int(cut.max())
    return cut.astype(int), opt, int((cut == opt).sum())


PG_BURN_IN = 200


def pgate_case(name):
    c = case_by_name(name)
    script = ev.pgate_script(c)
    beta, _ = ev.pgate_trace(c, script[:1])
    rtl = read_trace(os.path.join(HERE, "build", f"pgate_{name}_dump.hex"))
    wires = c["ins"] + c["outs"]
    cols = [c["nodes"].index(w) for w in wires]
    nin = len(c["ins"])
    valid = pg.rows_valid(c)
    bits_all = pg.state_bits(c["n"])
    rows_all = [tuple(int(x) for x in r) for r in bits_all[:, cols]]
    res, pos, ok_all = [], 0, True
    for label, mask, val, sw in script:
        seg = rtl[pos + PG_BURN_IN:pos + sw]
        pos += sw
        hq = [(127 if (val >> i) & 1 else -128) if (mask >> i) & 1 else c["hq"][i] for i in range(c["n"])]
        pi = fx.fixed_stationary(c["Jq"], hq, c["order"], beta, wfrac=c["wfrac"])
        segbits = (seg[:, None] >> np.array(cols)) & 1
        emp, ref = {}, {}
        for r in map(tuple, segbits.tolist()):
            emp[r] = emp.get(r, 0) + 1.0 / len(seg)
        for st, r in enumerate(rows_all):
            ref[r] = ref.get(r, 0.0) + pi[st]
        keys = sorted(set(emp) | set(ref))
        tv = 0.5 * sum(abs(emp.get(k, 0) - ref.get(k, 0)) for k in keys)
        clamp = {wires.index(c["nodes"][i]): (val >> i) & 1 for i in range(c["n"])
                 if (mask >> i) & 1 and c["nodes"][i] in wires}
        # rows of the Boolean function compatible with the clamp
        target = {r for r in valid if all(r[k] == b for k, b in clamp.items())}
        p_ok = sum(v for r, v in emp.items() if r in target)
        p_ok_ref = sum(v for r, v in ref.items() if r in target)
        top = max(emp, key=emp.get)
        if label.startswith("fwd"):
            ok = top in target                       # most likely row = truth table
        else:
            ok = top in target and p_ok > 0.5        # backward / factoring
        ok_all &= ok
        res.append(dict(label=label, p_ok=p_ok, p_ok_exact=p_ok_ref, tv=tv, ok=bool(ok),
                        top="".join(map(str, top)), emp={"".join(map(str, k)): v for k, v in emp.items()},
                        ref={"".join(map(str, k)): v for k, v in ref.items()}))
    out = dict(case=name, wires=wires, nin=nin, beta=beta / 16, scenarios=res, ok=bool(ok_all))
    if c.get("ladder"):
        # swap statistics from the bit-exact replay (the RTL trace equals it)
        _, _, smp = ev.pgate_trace(c, script, sampler=True)
        out["ladder"] = [b / 16 for b in c["ladder"]]
        out["swap_accept"] = [a / max(1, t) for a, t in zip(smp.pt.accepts, smp.pt.tries)]
    return out


def do_pgates(args):
    out = []
    for name in args.pgates:
        r = pgate_case(name)
        out.append(r)
        print(f"{name}  rows = ({','.join(r['wires'][:r['nin']])} | {','.join(r['wires'][r['nin']:])})  "
              f"beta = {r['beta']:.4f}")
        print(f"  {'scenario':12s} {'P(ok) RTL':>9s} {'P(ok) exact':>11s} {'TV':>7s}  most likely row   check")
        for s in r["scenarios"]:
            print(f"  {s['label']:12s} {s['p_ok']:9.4f} {s['p_ok_exact']:11.4f} {s['tv']:7.4f}  "
                  f"{s['top'][:r['nin']]}|{s['top'][r['nin']:]:12s}  {'ok' if s['ok'] else 'FAIL'}")
        if "ladder" in r:
            print(f"  tempering: K = {len(r['ladder'])}, ladder beta = {r['ladder']}, swap acceptance per rung pair = "
                  f"{[round(x, 3) for x in r['swap_accept']]}; one round = {len(r['ladder'])} sweeps "
                  f"(same sweep budget per scenario as the K = 1 case)")
    print("columns: P(ok) = probability of rows consistent with the gate and the clamp; exact = stationary law "
          "of the quantized kernel; TV = RTL vs exact over (inputs|outputs) rows")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f)
    bad = [r["case"] for r in out if not r["ok"]]
    print("pgates: PASS" if not bad else f"pgates: FAIL {bad}")
    return 0 if not bad else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=[])
    ap.add_argument("--sweeps", type=int, default=200000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--top-dump", default=None)
    ap.add_argument("--top-sparse-dump", default=None)
    ap.add_argument("--sparse-stats", nargs="*", default=[])
    ap.add_argument("--anneal", action="store_true")
    ap.add_argument("--pgates", nargs="*", default=[])
    a = ap.parse_args()
    rc = 0
    if a.cases:
        rc |= do_stats(a)
    if a.top_dump:
        rc |= do_top(a.top_dump)
    if a.sparse_stats:
        rc |= do_sparse_stats(a)
    if a.top_sparse_dump:
        rc |= do_top_sparse(a.top_sparse_dump)
    if a.anneal:
        do_anneal()
    if a.pgates:
        rc |= do_pgates(a)
    sys.exit(rc)


if __name__ == "__main__":
    main()
