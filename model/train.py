#!/usr/bin/env python3
"""Learning loop: train a Boltzmann machine with the p-bit chip as the sampler.

The trainer talks to the chip ONLY through top.v's UART commands
(W/H/O/B/X/S), so the same code runs on three transports:
  * ModelChip  : model/uart_script.TopModel (bit-exact Python mirror of top.v)
  * the RTL    : the ModelChip session is recorded as a Script and replayed by
                 tb/tb_top_script.v; byte equality = the RTL produces the same
                 samples, hence the same gradients (make train)
  * the board  : fpga/host.py train (serial; not run here, no board)

Learning rule (maximum likelihood / contrastive divergence)
[DOCUMENTED pattern: thrml/02_spin_models.py, torx/06 positive / negative phase]:
    dJ_ij = lr * (<s_i s_j>_data - <s_i s_j>_model)
    dh_i  = lr * (<s_i>_data     - <s_i>_model),        s = +-1
  positive phase: exact from the data for visible-only models; with hidden
                  units the visibles are clamped (bias +127 / -128, as in
                  pgates.clamp) and the chip samples the hiddens;
  negative phase: free run of the chip; it continues from the previous chain
                  (persistent CD) after a short burn-in.
Float master weights; Q3.4 words are uploaded (only those that changed).
beta is fixed at 1 (B 0 16 16): the learned J, h already include beta.
Unused array nodes keep J = 0 and h = -128 (pinned to -1).
Evaluation: exact KL(data || model) over the visibles, from the quantized
weights (enumeration of all 2^(nv+nh) states).
[FPGA-APPROX] all choices (learning rate, sample sizes, smoothing).
"""
import argparse
import itertools
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cases as cs          # noqa: E402
import fixed as fx          # noqa: E402
import golden as gd         # noqa: E402
import uart_script as us    # noqa: E402

WMAX = 127 / 16


class ModelChip:
    """The Python mirror of a dense top.v (N = 16, HIST_BITS = 12); records the session."""

    def __init__(self, n=16, iw=4):
        self.m = us.TopModel(N=n, IW=iw, AW=14, W_FILE="", H_FILE="", O_FILE="", T_FILE="")
        self.script = us.Script(self.m)
        self.n = n

    def cmd(self, c, *args):
        return self.script.cmd(c, *args)


class Trainer:
    def __init__(self, chip, nv, nh, data, lr=0.1, burn=20, n_neg=1000, n_pos=60, seed=1, decay=1e-3,
                 visible_couplings=True, lr_tau=None):
        self.chip, self.nv, self.nh, self.n = chip, nv, nh, nv + nh
        self.N = chip.n
        self.iw = fx.clog2(self.N)
        self.nb = (self.N + 7) // 8
        self.data = data                                  # {visible tuple (bits): weight}, sums to 1
        self.lr, self.burn, self.n_neg, self.n_pos, self.decay = lr, burn, n_neg, n_pos, decay
        self.lr_tau, self.t = lr_tau, 0                   # lr / (1 + t / lr_tau): averages out sampling noise
        rng = np.random.default_rng(seed)
        self.J = np.zeros((self.n, self.n))
        if nh:                                            # break the hidden-unit symmetry
            w = rng.normal(0, 1.0, (self.n, self.n))
            self.J = np.triu(w, 1) + np.triu(w, 1).T
        self.mask = 1 - np.eye(self.n)
        if not visible_couplings:                         # restricted BM: no visible-visible terms
            self.mask[:nv, :nv] = 0
        self.J *= self.mask
        self.h = np.zeros(self.n)
        self.wq, self.hq = {}, {}
        self.setup(seed)

    # ------------------------------------------------------------ chip I/O
    def setup(self, seed):
        self.chip.cmd("B", 0, 16, 16, 0, 0, 1)            # constant beta = 1
        # the weight cache below assumes an all-zero weight RAM: clear whatever program the
        # chip holds (board power-on case, previous web session) before the first upload
        for i in range(self.N):
            for j in range(self.N):
                a = (i << self.iw) | j
                self.chip.cmd("W", a >> 8, a & 0xFF, 0)
        for k in range(self.N):
            self.chip.cmd("O", k, k)
        for i in range(self.n, self.N):
            self.chip.cmd("H", i, 0x80)                   # unused node pinned to -1
        self.chip.cmd32 = lambda c, v: self.chip.cmd(c, *[(v >> s) & 0xFF for s in (24, 16, 8, 0)])
        self.upload()
        self.chip.cmd32("X", seed)

    def upload(self, h_override=None):
        h = self.h if h_override is None else h_override
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                q = fx.to_q(self.J[i, j], 4, 8)
                if self.wq.get((i, j), 0) != q:
                    a = (i << self.iw) | j
                    self.chip.cmd("W", a >> 8, a & 0xFF, q & 0xFF)
                    self.wq[(i, j)] = q
        for i in range(self.n):
            q = h[i] if isinstance(h[i], (int, np.integer)) else fx.to_q(h[i], 4, 8)
            if self.hq.get(i) != q:
                self.chip.cmd("H", i, q & 0xFF)
                self.hq[i] = q

    def stream(self, k):
        rep = self.chip.cmd32("S", k)
        raw = rep[:-1]
        idx = [int.from_bytes(bytes(raw[t * self.nb:(t + 1) * self.nb]), "little") for t in range(k)]
        return 2 * ((np.array(idx)[:, None] >> np.arange(self.n)) & 1) - 1

    # ------------------------------------------------------------ phases
    def positive(self):
        m, C = np.zeros(self.n), np.zeros((self.n, self.n))
        if self.nh == 0:
            for v, w in self.data.items():
                s = 2 * np.array(v) - 1
                m += w * s
                C += w * np.outer(s, s)
            return m, C
        for v, w in self.data.items():
            hq = [(127 if b else -128) for b in v] + [fx.to_q(x, 4, 8) for x in self.h[self.nv:]]
            self.upload(h_override=hq)
            self.stream(self.burn // 2)
            S = self.stream(self.n_pos)
            m += w * S.mean(0)
            C += w * (S.T @ S) / len(S)
        self.upload()
        return m, C

    def negative(self):
        self.stream(self.burn)
        S = self.stream(self.n_neg)
        return S.mean(0), (S.T @ S) / len(S)

    def step(self):
        mp, Cp = self.positive()
        mn, Cn = self.negative()
        lr = self.lr / (1 + self.t / self.lr_tau) if self.lr_tau else self.lr
        self.t += 1
        self.J += lr * ((Cp - Cn) * self.mask - self.decay * self.J)
        self.h += lr * (mp - mn)
        self.J = np.clip(self.J, -WMAX, WMAX)
        self.h = np.clip(self.h, -WMAX, WMAX)
        self.upload()

    # ------------------------------------------------------------ evaluation
    def quantized(self):
        Jq = np.array([[fx.to_q(self.J[i, j], 4, 8) / 16 for j in range(self.n)] for i in range(self.n)])
        hq = np.array([fx.to_q(x, 4, 8) / 16 for x in self.h])
        return Jq, hq

    def model_visible(self):
        Jq, hq = self.quantized()
        p, _ = gd.exact_boltzmann(Jq, hq, 1.0)
        idx = np.arange(1 << self.n)
        vis = idx & ((1 << self.nv) - 1)
        return np.bincount(vis, weights=p, minlength=1 << self.nv)

    def kl(self):
        q = self.model_visible()
        return sum(w * math.log(w / max(q[sum(b << k for k, b in enumerate(v))], 1e-300))
                   for v, w in self.data.items() if w > 0)


# ------------------------------------------------------------------ data sets
def gate_data(fn, eps=0.02):
    """Truth table of a 2-input gate as (A, B, Y), smoothed by eps over all 8 rows."""
    d = {}
    for v in itertools.product((0, 1), repeat=3):
        d[v] = eps / 8 + ((1 - eps) / 4 if v[2] == fn(v[0], v[1]) else 0.0)
    return d


def ring8_data():
    """Exact law of the torx/06 ring at beta 1.5 (infinite data set)."""
    c = cs.ring8()
    p, _ = gd.exact_boltzmann(np.array(c["Jf"]), np.array(c["hf"]), c["beta"])
    return {tuple((k >> i) & 1 for i in range(8)): float(w) for k, w in enumerate(p)}, c


DEMOS = {
    "and":     dict(nv=3, nh=0, data=lambda: gate_data(lambda a, b: a & b), epochs=200, lr=0.3),
    "xor":     dict(nv=3, nh=0, data=lambda: gate_data(lambda a, b: a ^ b), epochs=150, lr=0.15),
    "xor_h":   dict(nv=3, nh=1, data=lambda: gate_data(lambda a, b: a ^ b), epochs=250, lr=0.4),
    "ring8":   dict(nv=8, nh=0, data=lambda: ring8_data()[0], epochs=300, lr=0.3, lr_tau=60, decay=0.0),
    # model/ml.py: 3x3 glyphs T L X O + one-hot label (classifier), 3x3 bars and stripes (+ 6 hidden)
    "glyphs":  dict(nv=13, nh=0, data=lambda: _ml().glyph_data(), epochs=120, lr=0.3, lr_tau=40),
    "bars3":   dict(nv=9, nh=6, data=lambda: _ml().bars_data(), epochs=150, lr=0.3, lr_tau=40),
}


def _ml():
    import ml
    return ml


def run_demo(name, epochs=None, chip=None, quiet=False, log_every=10):
    d = DEMOS[name]
    chip = chip or ModelChip()
    tr = Trainer(chip, d["nv"], d["nh"], d["data"](), lr=d["lr"], lr_tau=d.get("lr_tau"),
                 decay=d.get("decay", 1e-3), n_pos=d.get("n_pos", 60), n_neg=d.get("n_neg", 1000))
    hist = [tr.kl()]
    for ep in range(epochs if epochs is not None else d["epochs"]):
        tr.step()
        hist.append(tr.kl())
        if not quiet and (ep + 1) % log_every == 0:
            print(f"  {name:6s} epoch {ep + 1:4d}  KL(data||model) = {hist[-1]:.4f} nats")
    return tr, hist


def report(name, tr, hist):
    out = dict(demo=name, kl=hist, J=tr.J.tolist(), h=tr.h.tolist())
    Jq, hq = tr.quantized()
    if name == "and":
        # hand gadget of docs/05: J_AB = -1, J_AY = J_BY = +2, h = (+1, +1, -2) (any positive scale)
        g = np.array([-1, 2, 2, 1, 1, -2], float)
        l = np.array([Jq[0, 1], Jq[0, 2], Jq[1, 2], hq[0], hq[1], hq[2]])
        cos = float(g @ l / np.linalg.norm(g) / np.linalg.norm(l))
        print(f"  and: learned (J_AB, J_AY, J_BY, h_A, h_B, h_Y) = {np.round(l, 3).tolist()}")
        print(f"       hand gadget direction (-1, 2, 2, 1, 1, -2): cosine similarity {cos:.3f}")
        out["cos_gadget"] = cos
    if name == "ring8":
        _, c = ring8_data()
        true = c["beta"] * np.array(c["Jf"])
        ring = [(i, (i + 1) % 8) for i in range(8)]
        err_ring = max(abs(Jq[i, j] - true[i, j]) for i, j in ring)
        off = max(abs(Jq[i, j]) for i in range(8) for j in range(8) if i != j and (i, j) not in ring
                  and (j, i) not in ring)
        print(f"  ring8: ring couplings learned vs beta*J_true: max error {err_ring:.3f} "
              f"(Q3.4 step 0.0625); largest off-ring coupling {off:.3f}")
        print(f"         learned ring J = {[round(float(Jq[i, j]), 3) for i, j in ring]}")
        print(f"         true  beta*J   = {[round(float(true[i, j]), 3) for i, j in ring]}")
        # floor: the true parameters rounded to Q3.4 (best any Q3.4 upload could do near the truth)
        tq = np.vectorize(lambda x: fx.to_q(x, 4, 8) / 16)
        pq, _ = gd.exact_boltzmann(tq(true), tq(c["beta"] * np.array(c["hf"])), 1.0)
        floor = sum(w * math.log(w / pq[k]) for k, w in enumerate(ring8_data()[0].values()))
        print(f"         KL of the true model rounded to Q3.4 (quantization floor): {floor:.4f} nats")
        out.update(err_ring=float(err_ring), off_ring=float(off), kl_floor=float(floor))
    q = tr.model_visible()
    pv = {v: w for v, w in tr.data.items()}
    if tr.nv == 3:
        rows = " ".join(f"{''.join(map(str, v))}:{q[sum(b << k for k, b in enumerate(v))]:.3f}"
                        for v in sorted(pv))
        print(f"  {name}: model P(A,B,Y) = {rows}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", nargs="*", default=["and", "xor", "xor_h", "ring8"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--rtl-script", default=None, help="write the first --rtl-epochs of xor_h as a tb_top_script session")
    ap.add_argument("--rtl-epochs", type=int, default=3)
    a = ap.parse_args()
    if a.rtl_script:
        chip = ModelChip()
        tr, hist = run_demo("xor_h", epochs=a.rtl_epochs, chip=chip, quiet=True)
        chip.script.write(a.rtl_script, us.tb_params(chip.m))
        print(f"train: xor_h, {a.rtl_epochs} epochs recorded: "
              f"{sum(1 for w in chip.script.ops if w < (1 << 24))} bytes sent, {len(chip.script.expected)} reply bytes "
              f"-> {a.rtl_script} (KL {hist[0]:.4f} -> {hist[-1]:.4f})")
        return
    res, ok = [], True
    for name in a.demos:
        print(f"demo {name}: nv={DEMOS[name]['nv']} nh={DEMOS[name]['nh']}")
        tr, hist = run_demo(name, a.epochs)
        r = report(name, tr, hist)
        res.append(r)
        print(f"  {name}: KL {hist[0]:.4f} -> {hist[-1]:.4f} nats")
    kl = {r["demo"]: r["kl"][-1] for r in res}
    checks = []
    if "and" in kl:
        checks.append(("and learned (KL < 0.1)", kl["and"] < 0.1))
    if "xor" in kl and "xor_h" in kl:
        checks.append(("xor needs a hidden unit (KL visible-only > 0.5, with one hidden < 0.2)",
                       kl["xor"] > 0.5 and kl["xor_h"] < 0.2))
    if "ring8" in kl:
        r8 = [r for r in res if r["demo"] == "ring8"][0]
        checks.append(("ring8 recovered (ring J within 1 LSB, KL < 5 x Q3.4 floor)",
                       r8["err_ring"] <= 0.0625 and kl["ring8"] < 5 * r8["kl_floor"]))
    for label, c in checks:
        print(f"check {label}: {'ok' if c else 'FAIL'}")
        ok &= c
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f)
        plot(res, os.path.dirname(a.out))
    print("train: PASS" if ok else "train: FAIL")
    sys.exit(0 if ok else 1)


def plot(res, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    cols = {"and": "#2a6fdb", "xor": "#c0392b", "xor_h": "#27ae60", "ring8": "#d9792b"}
    lab = {"and": "AND (3 visible)", "xor": "XOR, visible only", "xor_h": "XOR + 1 hidden", "ring8": "ring8 recovery"}
    for r in res:
        ax.semilogy(np.maximum(r["kl"], 1e-4), color=cols[r["demo"]], label=lab[r["demo"]])
    ax.set_xlabel("epoch (one positive + one negative phase on the chip)")
    ax.set_ylabel("KL(data || model), nats")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "train_kl.png"), dpi=120)


if __name__ == "__main__":
    main()
