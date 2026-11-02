#!/usr/bin/env python3
"""Byte-level model of rtl/top.v and UART scripts for tb/tb_top_script.v.

TopModel(cfg) mirrors top.v at the level of its UART: feed it the bytes a host
sends, it returns the bytes top.v answers. It is built from the bit-exact
primitives of fixed.py (LFSR lanes, logit, sigmoid LUT, Schedule) and mirrors
the RAMs of pbit_array.v (dense) or sparse_array.v (SPARSE = 1) word for word,
so arbitrary host sessions (programming, runs, readout, noise options,
tempering) can be checked byte for byte against the RTL.

Script(cfg) records a session: each cmd() appends the bytes to send and, from
the model, the number of reply bytes to wait for. write() produces
  <dir>/script.hex    32-bit words: 00_0000xx send byte xx,
                                    01_nnnnnn wait until n more bytes arrived,
                                    ff_000000 end
  <dir>/script.vh     top.v parameters for tb_top_script.v
  <dir>/expected.hex  the reply bytes predicted by TopModel
Button presses are not modelled (the scripts never press S2).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixed as fx          # noqa: E402
import tempering as tp      # noqa: E402

VEC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors")

# top.v defaults (power-on program = maxcut16, table schedule)
DEFAULTS = dict(N=16, IW=4, WFRAC=4, AW=14, HIST_BITS=12, SEED=fx.LFSR_DEFAULT_SEED,
                DEF_MODE=2, DEF_BETA0=8, DEF_BETA1=64, DEF_DELTA=0, DEF_SPS=8,
                W_FILE="maxcut16/weights.hex", H_FILE="maxcut16/bias.hex",
                O_FILE="maxcut16/order.hex", T_FILE="maxcut16/beta_table.hex",
                SPARSE=0, P=4, D=4, DW=2, LW=2, ROUNDS=4, UP_RAM=0, NOISE_CTL=0,
                TEMPER_K=1, TEMPER_LADDER=0x0C142030)


def _s8(v):
    return v - 256 if v & 0x80 else v


def _readhex(name, n, default=0):
    vals = [default] * n
    if name:
        with open(os.path.join(VEC, name)) as f:
            for k, tok in enumerate(f.read().split()[:n]):
                vals[k] = int(tok, 16)
    return vals


def le(v, nbytes):
    return [(v >> (8 * k)) & 0xFF for k in range(nbytes)]


NARGS = {"W": 3, "E": 5, "M": 5, "G": 2, "H": 2, "O": 2, "T": 2, "B": 6, "X": 4, "R": 4,
         "S": 4, "N": 4, "Y": 3, "L": 2}


class TopModel:
    def __init__(self, **cfg):
        c = dict(DEFAULTS)
        c.update(cfg)
        self.c = c
        n, iw = c["N"], c["IW"]
        self.n, self.iw, self.hb = n, iw, c["HIST_BITS"]
        self.nb = (n + 7) // 8
        # dense RAMs (pbit_array.v): weight word {i, j}, bias, order
        self.W = [_s8(v) for v in _readhex(c["W_FILE"], 1 << (2 * iw))]
        self.H = [_s8(v) for v in _readhex(c["H_FILE"], 1 << iw)]
        self.O = _readhex(c["O_FILE"], 1 << iw)
        if not c["O_FILE"]:
            self.O = list(range(1 << iw))
        self.table = _readhex(c["T_FILE"], 16)
        # sparse RAMs (sparse_array.v): per lane node words and neighbour words
        P, lw, dw = c["P"], c["LW"], c["DW"]
        self.node = [[(0, 0, 0)] * (1 << lw) for _ in range(P)]
        self.nbr = [[(0, 0)] * (1 << (lw + dw)) for _ in range(P)]
        self.rounds = c["ROUNDS"]
        # schedule_ctrl.v configuration
        self.mode, self.b0, self.b1, self.dl, self.sps = (c["DEF_MODE"], c["DEF_BETA0"], c["DEF_BETA1"],
                                                          c["DEF_DELTA"], c["DEF_SPS"])
        self.restart()
        self.nzf, self.nzq = 0, 0
        self.de = 0
        self.pt = None
        if c["TEMPER_K"] > 1:
            lad = [(c["TEMPER_LADDER"] >> (8 * k)) & 0xFF for k in range(c["TEMPER_K"])]
            self.pt = tp.Tempering(lad, n, c["SEED"], c["AW"])
        self.reseed(c["SEED"], spins=False)
        self.s = [0] * n
        self.clear()

    # ------------------------------------------------------------ state
    def restart(self):
        self.sched = fx.Schedule(self.mode, self.b0, self.b1, self.dl, self.sps, list(self.table))

    def reseed(self, seed, spins=True):
        self.seed = seed
        self.lfsr = [fx.lane_seed(seed, e) for e in range(len(fx.LFSR_MASKS))]
        self.raw = fx.lane_seed(seed, 0)
        if spins:
            self.s = [0] * self.n
        self.temper_reset()

    def temper_reset(self):
        if self.pt:
            self.pt.reset(self.seed)
            self.s = self.pt.spins()

    def clear(self):
        self.hist = [0] * (1 << self.hb)
        self.up = [0] * self.n
        self.ns = 0

    def draw(self, e):
        self.lfsr[e], r = fx.lfsr_draw16(self.lfsr[e], fx.LFSR_MASKS[e])
        return r

    def p_of(self, beta, acc):
        lut = self._lut()
        return lut[fx.logit_q5(beta, acc, self.c["WFRAC"]) + (1 << (fx.LUT_AW - 1))]

    def _lut(self):
        if not hasattr(self, "_lut_cache"):
            self._lut_cache = fx.sigmoid_lut()
        return self._lut_cache

    def state_index(self):
        return sum(b << i for i, b in enumerate(self.s))

    def record(self):
        idx = self.state_index()
        self.hist[idx & ((1 << self.hb) - 1)] += 1
        for i in range(self.n):
            self.up[i] += self.s[i]
        self.ns += 1

    # ------------------------------------------------------------ sweeps
    def dense_sweep(self, beta):
        n, iw = self.n, self.iw
        for k in range(n):
            i = self.O[k]
            acc = self.H[i]
            for j in range(n):
                w = self.W[(i << iw) | j]
                acc += w if self.s[j] else -w
            p = self.p_of(beta, acc)
            new = 1 if self.draw(0) < p else 0
            self.on_update(i, self.s[i], new, acc)
            self.s[i] = new

    def on_update(self, i, old, new, acc):
        if new != old:                              # tempering energy tap (e = E/2)
            self.de += acc if old else -acc

    def sparse_sweep(self, beta):
        c = self.c
        P, D, dw = c["P"], c["D"], c["DW"]
        tau = c["NOISE_CTL"] and (self.nzf & 1)
        share = c["NOISE_CTL"] and (self.nzf & 2)
        for r in range(self.rounds):
            lanes = []
            for e in range(P):
                v, i, h = self.node[e][r]
                if not v:
                    continue
                acc = h
                for sl in range(D):
                    j, w = self.nbr[e][(r << dw) | sl]
                    acc += w if self.s[j] else -w
                lanes.append((e, i, self.p_of(beta, acc)))
            keep = {}
            if tau:
                if share:
                    d = self.draw(0)
                    keep = {e: d >= self.nzq for e, _, _ in lanes}
                else:
                    keep = {e: self.draw(e) >= self.nzq for e, _, _ in lanes}
            if share:
                r0 = None
                if not keep.get(0, False):
                    r0 = self.draw(0)
                for e, i, p in lanes:
                    if not keep.get(e, False):
                        self.s[i] = 1 if r0 < p else 0
            else:
                for e, i, p in lanes:
                    if not keep.get(e, False):
                        self.s[i] = 1 if self.draw(e) < p else 0

    def sweep(self):
        beta = self.sched.beta
        if self.c["SPARSE"]:
            self.sparse_sweep(beta)
        else:
            self.dense_sweep(beta)
        self.sched.sweep_done()

    def run_unit(self):
        """One recorded unit of 'R' / 'S': a sweep (a round when tempering)."""
        if not self.pt:
            self.sweep()
            return

        def sweep_rep(s, beta):
            self.s, self.de = s, 0
            self.dense_sweep(beta)
            return self.de
        self.pt.round(sweep_rep)
        self.s = self.pt.spins()

    # ------------------------------------------------------------ commands
    def command(self, cmd, args):
        """Execute one command (args = list of argument bytes); return the reply bytes."""
        c = self.c
        K = [ord("K")]
        a = 0
        for b in args:
            a = (a << 8) | b
        if cmd == "P":
            if c["SPARSE"]:
                return le(self.n, 1) + [0x80 | self.hb, self.n >> 8, c["P"], c["D"], c["LW"]] + K
            return [self.n & 0xFF, self.hb] + K
        if cmd == "W":
            if not c["SPARSE"]:
                self.W[((args[0] << 8) | args[1]) & ((1 << (2 * self.iw)) - 1)] = _s8(args[2])
            return K
        if cmd == "H":
            if not c["SPARSE"]:
                self.H[args[0] & ((1 << self.iw) - 1)] = _s8(args[1])
            return K
        if cmd == "O":
            if not c["SPARSE"]:
                self.O[args[0] & ((1 << self.iw) - 1)] = args[1] & ((1 << self.iw) - 1)
            return K
        if cmd == "E":
            if c["SPARSE"]:
                l, r, sl, j, d = args
                self.nbr[l & 3][((r & ((1 << c["LW"]) - 1)) << c["DW"]) | (sl & ((1 << c["DW"]) - 1))] = \
                    (j & ((1 << self.iw) - 1), _s8(d))
            return K
        if cmd == "M":
            if c["SPARSE"]:
                l, r, f, i, d = args
                self.node[l & 3][r & ((1 << c["LW"]) - 1)] = (f & 1, i & ((1 << self.iw) - 1), _s8(d))
            return K
        if cmd == "G":
            if c["SPARSE"]:
                self.rounds = a & ((1 << (c["LW"] + 1)) - 1)
            return K
        if cmd == "T":
            self.table[args[0] & 15] = args[1]
            return K
        if cmd == "B":
            self.mode, self.b0, self.b1, self.dl = args[0] & 3, args[1], args[2], args[3]
            self.sps = (args[4] << 8) | args[5]
            self.restart()
            return K
        if cmd == "Z":
            self.restart()
            if self.pt:
                self.pt.resync()
                self.s = self.pt.spins()
            return K
        if cmd == "X":
            self.reseed(a)
            return K
        if cmd == "Y":
            self.nzf, self.nzq = args[0] & 3, (args[1] << 8) | args[2]
            return K
        if cmd == "L":
            if self.pt and (args[0] & 3) < c["TEMPER_K"]:
                self.pt.ladder[args[0] & 3] = args[1]
            return K
        if cmd == "R":
            for _ in range(a):
                self.run_unit()
                self.record()
            return K
        if cmd == "S":
            out = []
            for _ in range(a):
                self.run_unit()
                self.record()
                out += le(self.state_index(), self.nb)
            return out + K
        if cmd == "N":
            out = []
            for _ in range(a):
                self.raw, r = fx.lfsr_draw16(self.raw, fx.LFSR_MASKS[0])
                out += le(r, 2)
            return out + K
        if cmd == "C":
            self.clear()
            return K
        if cmd == "D":
            return sum((le(h, 4) for h in self.hist), []) + K
        if cmd == "U":
            return le(self.ns, 4) + sum((le(u, 4) for u in self.up), []) + K
        if cmd == "Q":
            return le(self.state_index(), self.nb) + K
        return [ord("?")]


def reply_len(cmd, args, n, hist_bits, sparse=False):
    """Bytes top.v answers to a command (for serial transports; mirrors TopModel.command)."""
    a = 0
    for b in args:
        a = (a << 8) | b
    nb = (n + 7) // 8
    if cmd == "P":
        return 7 if sparse else 3
    if cmd == "S":
        return a * nb + 1
    if cmd == "N":
        return 2 * a + 1
    if cmd == "D":
        return 4 * (1 << hist_bits) + 1
    if cmd == "U":
        return 4 + 4 * n + 1
    if cmd == "Q":
        return nb + 1
    if cmd in NARGS or cmd in "ZC":
        return 1
    return 1                                        # '?'


class Script:
    """Records a host session and the model's replies (see module docstring)."""

    def __init__(self, model):
        self.m = model
        self.ops = []
        self.expected = []

    def cmd(self, c, *args):
        args = list(args)
        assert len(args) == NARGS.get(c, 0), (c, args)
        for b in [ord(c)] + args:
            self.ops.append(b & 0xFF)
        reply = self.m.command(c, args)
        if reply:
            self.ops.append((1 << 24) | len(reply))
        self.expected += reply
        return reply

    def cmd32(self, c, v):
        return self.cmd(c, *[(v >> s) & 0xFF for s in (24, 16, 8, 0)])

    def write(self, d, params):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "script.hex"), "w") as f:
            f.write("\n".join(f"{w:08x}" for w in self.ops + [0xFF000000]) + "\n")
        with open(os.path.join(d, "expected.hex"), "w") as f:
            f.write("\n".join(f"{b:02x}" for b in self.expected) + "\n")
        lines = [f"localparam integer SCRIPT_LEN = {len(self.ops) + 1};"]
        for k, v in params.items():
            if isinstance(v, str):
                lines.append(f'localparam        TOP_{k} = "{v}";')
            else:
                lines.append(f"localparam integer TOP_{k} = {v};")
        with open(os.path.join(d, "script.vh"), "w") as f:
            f.write("\n".join(lines) + "\n")


def compare_dump(expected_path, got_path, label):
    with open(expected_path) as f:
        exp = [int(x, 16) for x in f.read().split()]
    with open(got_path) as f:
        got = [int(x, 16) for x in f.read().split()]
    nm = sum(1 for a, b in zip(got, exp) if a != b) + abs(len(got) - len(exp))
    print(f"{label}: received {len(got)} bytes, expected {len(exp)}, mismatching bytes: {nm}")
    return nm


# ------------------------------------------------------------------ sessions
def tb_params(m):
    """top.v parameters of a TopModel for script.vh (file paths relative to tb/)."""
    c = m.c
    out = {k: c[k] for k in ("N", "IW", "WFRAC", "AW", "HIST_BITS", "DEF_MODE", "DEF_BETA0", "DEF_BETA1",
                             "DEF_DELTA", "DEF_SPS", "SPARSE", "P", "D", "DW", "LW", "ROUNDS", "UP_RAM",
                             "NOISE_CTL", "TEMPER_K", "TEMPER_LADDER")}
    out["SEED"] = c["SEED"]
    for k in ("W_FILE", "H_FILE", "O_FILE", "T_FILE"):
        out[k] = "../model/vectors/" + c[k] if c[k] else ""
    return out


def program_sparse(sc, case, P):
    """G / M / E commands for a sparse case (layout of export_vectors.sparse_layout)."""
    import export_vectors as ev
    lay = ev.sparse_layout(case, P)
    sc.cmd("G", lay["R"] >> 8, lay["R"] & 255)
    for e in range(P):
        node, nbrw = ev.sparse_words(case, lay, e)
        for r in range(lay["R"]):
            w = node[r]
            sc.cmd("M", e, r, (w >> (lay["IW"] + 8)) & 1, (w >> 8) & ((1 << lay["IW"]) - 1), w & 255)
            for sl in range(lay["D"]):
                ew = nbrw[(r << lay["DW"]) | sl]
                if ew:
                    sc.cmd("E", e, r, sl, ew >> 8, ew & 255)
    return lay


def session_noise_sparse():
    """SPARSE top (grid16, P = 4, NOISE_CTL = 1): every noise mode, raw noise, readout."""
    import cases as cs
    g16 = cs.grid16()
    m = TopModel(N=16, IW=4, AW=12, SPARSE=1, P=4, D=4, DW=2, LW=2, ROUNDS=1, NOISE_CTL=1, UP_RAM=0,
                 W_FILE="", H_FILE="", O_FILE="", T_FILE="")
    sc = Script(m)
    sc.cmd("P")
    program_sparse(sc, g16, 4)
    sc.cmd("B", 0, g16["beta_q"], g16["beta_q"], 0, 0, 1)
    sc.cmd32("X", 0x12345678)
    sc.cmd("C")
    sc.cmd32("R", 200)                         # plain
    sc.cmd("Y", 1, 0x20, 0x00)                 # tau, q = 1/8
    sc.cmd32("R", 200)
    sc.cmd32("S", 3)
    sc.cmd("Y", 2, 0, 0)                       # shared noise
    sc.cmd32("R", 200)
    sc.cmd("Y", 3, 0x80, 0x00)                 # both
    sc.cmd32("S", 3)
    sc.cmd("Y", 0, 0, 0)                       # back to plain
    sc.cmd32("R", 50)
    sc.cmd("U")
    sc.cmd("D")
    sc.cmd32("N", 40)                          # raw lane
    sc.cmd32("X", 0xCAFEF00D)
    sc.cmd32("N", 8)
    sc.cmd("Q")
    sc.cmd("y")
    return m, sc


def session_noise_dense():
    """Dense default top (maxcut16 power-on program): 'N' stream, 'Y' accepted, sampling unchanged."""
    m = TopModel()
    sc = Script(m)
    sc.cmd("P")
    sc.cmd32("N", 300)
    sc.cmd("Y", 1, 0x10, 0)                    # ignored without NOISE_CTL
    sc.cmd32("R", 20)
    sc.cmd("Q")
    sc.cmd32("X", 1)
    sc.cmd32("N", 4)
    sc.cmd("U")
    return m, sc


def program_dense(sc, case, h=None):
    """W (non-zero), H and O commands for a dense case on a top with IW address bits."""
    iw, n = sc.m.iw, case["n"]
    for i in range(n):
        for j in range(n):
            if case["Jq"][i][j]:
                a = (i << iw) | j
                sc.cmd("W", a >> 8, a & 255, case["Jq"][i][j] & 255)
    for i, v in enumerate(h or case["hq"]):
        sc.cmd("H", i, v & 255)
    for k, i in enumerate(case["order"]):
        sc.cmd("O", k, i)


def session_temper_dense():
    """Dense top, TEMPER_K = 4: the 2x2 multiplier clamped to 3, then to 6 (resync with 'Z')."""
    import cases as cs
    c = {x["name"]: x for x in cs.pgate_cases()}["factor6_pt"]
    p4 = ["p0", "p1", "p2", "p3"]

    def clamp(prod):
        return [(127 if (prod >> p4.index(w)) & 1 else -128) if w in p4 else c["hq"][i]
                for i, w in enumerate(c["nodes"])]
    m = TopModel(N=16, IW=4, AW=14, TEMPER_K=4, W_FILE="", H_FILE="", O_FILE="", T_FILE="")
    sc = Script(m)
    sc.cmd("P")
    program_dense(sc, c, clamp(3))
    for i in range(c["n"], 16):                # unused nodes still get visited
        sc.cmd("O", i, i)
    for k, b in enumerate(c["ladder"]):
        sc.cmd("L", k, b)
    sc.cmd32("X", 0x2468ACE1)
    sc.cmd("C")
    sc.cmd32("R", 120)
    sc.cmd("Q")
    sc.cmd32("S", 3)
    for i, v in enumerate(clamp(6)):
        sc.cmd("H", i, v & 255)
    sc.cmd("Z")
    sc.cmd32("R", 60)
    sc.cmd("L", 3, 2)                          # hotter top rung
    sc.cmd32("R", 20)
    sc.cmd("U")
    sc.cmd("D")
    sc.cmd("Q")
    return m, sc


SESSIONS = {"noise_sparse": session_noise_sparse, "noise_dense": session_noise_dense,
            "temper_dense": session_temper_dense}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["write", "check"])
    ap.add_argument("session")
    ap.add_argument("--dir", required=True)
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()
    if a.action == "write":
        m, sc = SESSIONS[a.session]()
        sc.write(a.dir, tb_params(m))
        print(f"script {a.session}: {sum(1 for w in sc.ops if w < (1 << 24))} bytes to send, "
              f"{len(sc.expected)} reply bytes expected -> {a.dir}")
    else:
        nm = compare_dump(os.path.join(a.dir, "expected.hex"), a.dump or os.path.join(a.dir, "dump.hex"),
                          f"script {a.session} UART replay")
        print(f"script {a.session}: PASS" if nm == 0 else f"script {a.session}: FAIL")
        sys.exit(1 if nm else 0)


if __name__ == "__main__":
    main()
