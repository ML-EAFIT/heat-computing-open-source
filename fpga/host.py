#!/usr/bin/env python3
"""Host-side driver for rtl/top.v on the Tang Nano 9K (UART 115200 8N1).

Examples
  python3 fpga/host.py --port /dev/ttyUSB1 ping
  python3 fpga/host.py --port /dev/ttyUSB1 sample --case maxcut12 --sweeps 100000
  python3 fpga/host.py --port /dev/ttyUSB1 anneal --case maxcut16 --runs 20
  python3 fpga/host.py --port /dev/ttyUSB1 stream --sweeps 50
  SPARSE build (make sparse):
  python3 fpga/host.py --port /dev/ttyUSB1 sample --case grid16 --sweeps 20000
  python3 fpga/host.py --port /dev/ttyUSB1 mag --case g12_glass16 --sweeps 2000
  Learning loop (dense build; model/train.py, the board is the sampler):
  python3 fpga/host.py --port /dev/ttyUSB1 train --case xor_h
  Raw noise of the 'N' lane (build with RO_JITTER = 0 / 1, fpga/entropy.py):
  python3 fpga/host.py --port /dev/ttyUSB1 entropy --sweeps 100000 --out noise.bin
  Parallel tempering build (TEMPER_K = 4, make case CASE=factor6_pt):
  python3 fpga/host.py --port /dev/ttyUSB1 temper --case factor6_pt --clamp p0=1,p1=1,p2=0,p3=0
  Your own program (model/problem.py file format, docs/09_programming.md):
  python3 fpga/host.py --port /dev/ttyUSB1 solve --problem examples/factor6.json
  python3 fpga/host.py --sim solve --problem examples/qubo_small.json   # no board: Python model

The board enumerates two serial ports (BL702 JTAG + UART); the UART is
usually the second one (/dev/ttyUSB1). Protocol: see the header of rtl/top.v.
Requires pyserial (pip install --user pyserial). --sim needs no board: the
bytes are answered by model/uart_script.TopModel, which mirrors the default
dense top.v (N = 16) byte for byte (slow: about a thousand sweeps per second).
"""
import argparse
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
import cases as cs          # noqa: E402
import export_vectors as ev  # noqa: E402
import fixed as fx          # noqa: E402
import golden as gd         # noqa: E402
import problem as pr        # noqa: E402
import uart_script as us    # noqa: E402


class SimSerial:
    """Serial-port stand-in: every write is one command, answered by TopModel."""

    def __init__(self, **cfg):
        self.m = us.TopModel(**cfg)
        self.buf = bytearray()

    def reset_input_buffer(self):
        self.buf.clear()

    def write(self, data):
        self.buf += bytes(self.m.command(chr(data[0]), list(data[1:])))

    def read(self, n):
        out, self.buf = bytes(self.buf[:n]), self.buf[n:]
        return out


class Board:
    def __init__(self, port, baud=115200, timeout=5.0, sim=False, transport=None):
        if transport is not None:       # any object with write / read / reset_input_buffer (web/server.py)
            self.s = transport
        elif sim:
            self.s = SimSerial()
        else:
            import serial
            self.s = serial.Serial(port, baud, timeout=timeout)
        self.s.reset_input_buffer()

    def cmd(self, c, *args):
        self.s.write(bytes([ord(c)] + list(args)))

    def read(self, n):
        b = self.s.read(n)
        if len(b) != n:
            raise IOError(f"timeout: wanted {n} bytes, got {len(b)}")
        return b

    def ack(self):
        b = self.read(1)
        if b != b"K":
            raise IOError(f"expected 'K', got {b!r}")

    def ping(self):
        """-> (N, HIST_BITS, sparse) with sparse = dict(P, D, LW) or None (dense build)."""
        self.cmd("P")
        n, hb = self.read(2)
        sparse = None
        if hb & 0x80:
            n_hi, p, d, lw = self.read(4)
            n, hb = n | (n_hi << 8), hb & 0x7F
            sparse = dict(P=p, D=d, LW=lw)
        self.ack()
        return n, hb, sparse

    def load_sparse(self, c, n_board, sp):
        """Program a case into a SPARSE build (layout = model/export_vectors.sparse_layout)."""
        lay = ev.sparse_layout(c, sp["P"])
        assert c["n"] <= n_board and lay["D"] <= sp["D"] and lay["R"] <= (1 << sp["LW"]), \
            f"case needs N={c['n']} D={lay['D']} rounds={lay['R']}, board has N={n_board} {sp}"
        self.cmd("G", lay["R"] >> 8, lay["R"] & 0xFF)
        nack = 1
        for e in range(sp["P"]):
            for r, row in enumerate(lay["rounds"]):
                i = row[e]
                nb = lay["nbr"][i] if i is not None else []
                self.cmd("M", e, r, int(i is not None), i or 0, (c["hq"][i] if i is not None else 0) & 0xFF)
                for slot in range(sp["D"]):              # also clears stale slots
                    j, w = nb[slot] if slot < len(nb) else (0, 0)
                    self.cmd("E", e, r, slot, j, w & 0xFF)
                nack += 1 + sp["D"]
        for _ in range(nack):
            self.ack()
        return lay

    def load(self, c, n_board):
        """Program weights/bias/order of a case (must have n <= board N)."""
        n = c["n"]
        assert n <= n_board, "case larger than the synthesised array"
        stride = 1 << fx.clog2(n_board)
        for i in range(n_board):
            for j in range(n_board):
                v = c["Jq"][i][j] if (i < n and j < n) else 0
                a = i * stride + j
                self.cmd("W", a >> 8, a & 0xFF, v & 0xFF)
        for i in range(n_board):
            self.cmd("H", i, (c["hq"][i] if i < n else 0) & 0xFF)
        order = c["order"] + list(range(n, n_board))      # unused nodes stay -1 (J=0, h=0)
        for k, i in enumerate(order):
            self.cmd("O", k, i)
        for _ in range(n_board * n_board + 2 * n_board):
            self.ack()

    def schedule(self, mode, b0, b1=0, delta=0, sps=1, table=None):
        if table is not None:
            for k, b in enumerate(table):
                self.cmd("T", k, b)
                self.ack()
        self.cmd("B", mode, b0, b1, delta, sps >> 8, sps & 0xFF)
        self.ack()

    def reseed(self, seed):
        self.cmd("X", *struct.pack(">I", seed))
        self.ack()

    def run(self, n):
        self.cmd("C")
        self.ack()
        self.cmd("R", *struct.pack(">I", n))
        self.ack()

    def hist(self, hb):
        self.cmd("D")
        raw = self.read(4 << hb)
        self.ack()
        return np.frombuffer(raw, dtype="<u4").astype(float)

    def spins(self, n):
        self.cmd("Q")
        raw = self.read((n + 7) // 8)
        self.ack()
        return int.from_bytes(raw, "little")

    def xfer(self, c, *args, n=16, hb=12, sparse=False):
        """Send a command and read its whole reply (length from uart_script.reply_len)."""
        self.cmd(c, *args)
        return list(self.read(us.reply_len(c, list(args), n, hb, sparse)))

    def stream(self, n_sweeps, n):
        nb = (n + 7) // 8
        self.cmd("S", *struct.pack(">I", n_sweeps))
        out = [int.from_bytes(self.read(nb), "little") for _ in range(n_sweeps)]
        self.ack()
        return out


def solve(b, c, n_board, runs, seed):
    """Anneal `runs` times from seeds seed, seed + 1, ...; keep the lowest-energy final state."""
    sc = ev.case_schedule(c)
    sweeps = 16 * sc["sps"] + 32
    best, seen = None, {}
    for r in range(runs):
        b.schedule(sc["mode"], sc["beta_start"], sc["beta_end"], sc["delta"], sc["sps"],
                   sc.get("table") if r == 0 else None)
        b.reseed(seed + r)
        b.run(sweeps)
        s = b.spins(n_board) & ((1 << c["n"]) - 1)
        seen[s] = seen.get(s, 0) + 1
        if best is None or pr.energy(c, s) < pr.energy(c, best):
            best = s
    e0 = pr.energy(c, best)
    hits = sum(k for st, k in seen.items() if pr.energy(c, st) == e0)
    return dict(schedule=sc, sweeps=sweeps, best=best, seen=seen, hits=hits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB1")
    ap.add_argument("--sim", action="store_true", help="no board: answer with the Python model of top.v")
    ap.add_argument("action", choices=["ping", "sample", "anneal", "stream", "pgate", "mag", "train", "entropy",
                                       "temper", "solve"])
    ap.add_argument("--problem", default=None, help="JSON problem file (model/problem.py), instead of --case")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--case", default="maxcut12")
    ap.add_argument("--sweeps", type=int, default=100000)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed", type=lambda x: int(x, 0), default=0x12345678)
    ap.add_argument("--clamp", default="", help="pgate: wire=bit list, e.g. Y=1 or p0=0,p1=1,p2=1,p3=0")
    a = ap.parse_args()
    c = pr.load(a.problem) if a.problem else None   # parse before touching the port
    b = Board(a.port, sim=a.sim)
    n_board, hb, sp = b.ping()
    print(f"board: N={n_board} HIST_BITS={hb}" + (f" SPARSE P={sp['P']} D={sp['D']} LW={sp['LW']}" if sp else ""))
    if a.action == "ping":
        return
    if a.action == "entropy":
        import entropy
        raw = b.xfer("N", *struct.pack(">I", a.sweeps), n=n_board, hb=hb, sparse=bool(sp))
        if raw[-1] != ord("K"):
            raise IOError("no 'K' after the noise stream")
        if a.out:
            with open(a.out, "wb") as f:
                f.write(bytes(raw[:-1]))
        entropy.report(entropy.draws_from_bytes(raw[:-1]), f"board 'N' x {a.sweeps}")
        return
    if a.action == "train":
        # same trainer as model/train.py, the board replaces the Python chip
        import train
        if sp or n_board != 16:
            raise SystemExit("train expects the default dense N = 16 build")

        class BoardChip:
            n = n_board

            def cmd(self, c, *args):
                return b.xfer(c, *args, n=n_board, hb=hb)
        tr, hist = train.run_demo(a.case, a.epochs, chip=BoardChip())
        train.report(a.case, tr, hist)
        print(f"{a.case}: KL {hist[0]:.4f} -> {hist[-1]:.4f} nats on the board")
        return
    if a.action == "temper":
        # TEMPER_K build: program the circuit, the ladder and the clamp, stream rung-0 rounds
        import tempering  # noqa: F401  (documents the mirror)
        c = {x["name"]: x for x in cs.pgate_cases()}[a.case]
        cl = dict(kv.split("=") for kv in a.clamp.split(",") if kv)
        b.load(c, n_board)
        for w, bit in cl.items():
            b.cmd("H", c["nodes"].index(w), 0x7F if int(bit) else 0x80)
            b.ack()
        for k, beta in enumerate(c["ladder"]):
            b.cmd("L", k, beta)
            b.ack()
        b.reseed(a.seed)
        wires = c["ins"] + c["outs"]
        cols = [c["nodes"].index(w) for w in wires]
        rows = {}
        for s in b.stream(a.sweeps, n_board)[200:]:
            r = "".join(str((s >> k) & 1) for k in cols)
            rows[r] = rows.get(r, 0) + 1
        tot = sum(rows.values())
        nin = len(c["ins"])
        print(f"{c['name']} clamp {cl or 'none'}  rows ({','.join(c['ins'])} | {','.join(c['outs'])}), rung 0")
        for r, k in sorted(rows.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {r[:nin]}|{r[nin:]}  {k / tot:.3f}  " + "#" * int(40 * k / tot))
        return
    if c is None:
        allc = {x["name"]: x for x in cs.all_cases()}
        allc.update({x["name"]: x for x, _ in cs.sparse_cases()})
        if a.case not in allc:
            raise SystemExit(f"unknown case {a.case!r}; built-in: {', '.join(sorted(allc))} (or use --problem)")
        c = allc[a.case]
    if c["n"] > n_board:
        raise SystemExit(f"{c['name']} needs {c['n']} p-bits, the board has N = {n_board}"
                         + ("" if sp else " (dense); a SPARSE build (make sparse) has up to 256"))
    if c["wfrac"] != 4:
        print("note: board is synthesised with WFRAC=4; this case needs a WFRAC=6 build")
    if sp:
        lay = b.load_sparse(c, n_board, sp)
        if a.action not in ("sample", "mag", "stream", "solve"):
            raise SystemExit("SPARSE build: actions sample, mag, stream, solve")
    else:
        if a.action == "mag":
            raise SystemExit("mag needs a SPARSE build")
        b.load(c, n_board)
    if sp and a.action in ("sample", "mag"):
        # sparse: idle nodes are never visited, so no padding draws; the
        # board must equal SparseParallelSampler bit for bit
        b.schedule(0, c["beta_q"], c["beta_q"])
        b.reseed(a.seed)
        t = time.time()
        b.run(a.sweeps)
        dt = time.time() - t
        h = b.hist(hb)
        b.cmd("U")
        raw = b.read(4 * (n_board + 1))
        b.ack()
        cnt = np.frombuffer(raw, dtype="<u4")
        ns, up = int(cnt[0]), cnt[1:1 + c["n"]]
        m = 2 * up / max(ns, 1) - 1
        print(f"{a.sweeps} sweeps in {dt:.2f} s ({a.sweeps / dt:.0f} sweeps/s incl. UART); "
              f"mean |<s_i>| = {np.abs(m).mean():.3f}, <m> = {m.mean():+.3f}")
        if c["n"] <= hb:
            hh = h.reshape(-1, 1 << c["n"]).sum(0) if c["n"] < hb else h
            exact, _ = gd.exact_boltzmann(np.array(c["Jq"]) / 16, np.array(c["hq"]) / 16, c["beta_q"] / 16)
            print(f"TV(board, exact) = {gd.tv(hh, exact):.4f}, KL = {gd.kl(hh, exact):.4f}")
        if a.sweeps <= 20000:
            smp = fx.SparseParallelSampler(lay["nbr"], c["hq"], c["colors"],
                                           fx.Schedule(0, c["beta_q"], c["beta_q"]), P=sp["P"], seed=a.seed)
            ph, pu, _ = fx.run(smp, a.sweeps, hist_bits=hb)
            same = np.array_equal(h, np.array(ph, float)) and list(up) == pu
            print("bit-exact vs model/fixed.SparseParallelSampler:", "YES" if same else "NO")
    elif a.action == "sample":
        b.schedule(0, c["beta_q"], c["beta_q"])
        b.reseed(a.seed)
        t = time.time()
        b.run(a.sweeps)
        dt = time.time() - t
        h = b.hist(hb)
        b_full = h.copy()
        n = c["n"]
        if n > hb:
            print(f"histogram covers only the low {hb} spins of {n}")
            return
        if n < hb:                          # marginalise padding spins out
            h = h.reshape(-1, 1 << n).sum(0)
        exact, _ = gd.exact_boltzmann(np.array(c["Jq"]) / 16, np.array(c["hq"]) / 16, c["beta_q"] / 16)
        print(f"{a.sweeps} sweeps in {dt:.2f} s ({a.sweeps / dt:.0f} sweeps/s incl. UART), "
              f"TV(board, exact) = {gd.tv(h, exact):.4f}, KL = {gd.kl(h, exact):.4f}")
        # the board must be bit-exact with the Python model of the *padded*
        # n_board-node array (padding nodes also consume noise draws)
        if a.sweeps <= 20000:
            Jp = [[c["Jq"][i][j] if (i < n and j < n) else 0 for j in range(n_board)] for i in range(n_board)]
            hp = [c["hq"][i] if i < n else 0 for i in range(n_board)]
            op = c["order"] + list(range(n, n_board))
            smp = fx.FixedSampler(Jp, hp, op, fx.Schedule(0, c["beta_q"], c["beta_q"]), seed=a.seed)
            ph, _, _ = fx.run(smp, a.sweeps, hist_bits=hb)
            same = np.array_equal(b_full, np.array(ph, float))
            print("bit-exact vs model/fixed.py:", "YES" if same else "NO")
    elif a.action == "anneal":
        sc = ev.case_schedule(c)
        b.schedule(sc["mode"], sc["beta_start"], sc["beta_end"], sc["delta"], sc["sps"], sc.get("table"))
        idx = np.arange(1 << c["n"])
        spins = 2 * ((idx[:, None] >> np.arange(c["n"])) & 1) - 1
        cut = sum((1 - spins[:, i] * spins[:, j]) / 2 for i, j in c.get("edges", []))
        for r in range(a.runs):
            b.reseed(a.seed + r)
            b.schedule(sc["mode"], sc["beta_start"], sc["beta_end"], sc["delta"], sc["sps"])
            b.run(16 * sc["sps"] + 32)
            s = b.spins(n_board) & ((1 << c["n"]) - 1)
            print(f"run {r}: spins 0x{s:04x} cut {int(cut[s])} (optimum {int(cut.max())})")
    elif a.action == "pgate":
        # p-gate (docs/05): clamp wires with the extreme bias, stream sweeps, print the row law
        if "nodes" not in c:
            raise SystemExit("pgate needs a p-gate case (pgate_and, fulladder, factor6, ...)")
        cl = dict(kv.split("=") for kv in a.clamp.split(",") if kv)
        hq = list(c["hq"])
        for w, bit in cl.items():
            hq[c["nodes"].index(w)] = 127 if int(bit) else -128
        for i, v in enumerate(hq):
            b.cmd("H", i, v & 0xFF)
            b.ack()
        beta = ev.case_schedule(c)["beta_end"]
        b.schedule(0, beta, beta)
        wires = c["ins"] + c["outs"]
        cols = [c["nodes"].index(w) for w in wires]
        rows = {}
        for s in b.stream(a.sweeps, n_board)[200:]:
            r = "".join(str((s >> k) & 1) for k in cols)
            rows[r] = rows.get(r, 0) + 1
        tot = sum(rows.values())
        nin = len(c["ins"])
        print(f"{c['name']} clamp {cl or 'none'}  rows ({','.join(c['ins'])} | {','.join(c['outs'])})")
        for r, k in sorted(rows.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {r[:nin]}|{r[nin:]}  {k / tot:.3f}  " + "#" * int(40 * k / tot))
    elif a.action == "solve":
        r = solve(b, c, n_board, a.runs, a.seed)
        sc, best, seen = r["schedule"], r["best"], r["seen"]
        print(f"{c['name']}: {a.runs} anneals x {r['sweeps']} sweeps (beta {sc['beta_start'] / 16:g} -> "
              f"{sc['beta_end'] / 16:g}); {r['hits']}/{a.runs} reached the lowest energy found")
        for line in pr.decode(c, best):
            print("  " + line)
        others = sorted((k for k in seen if k != best), key=lambda k: pr.energy(c, k))[:3]
        for k in others:
            print(f"  also ({seen[k]}x): " + " | ".join(pr.decode(c, k)))
    elif a.action == "stream":
        b.schedule(0, c["beta_q"], c["beta_q"])
        for k, s in enumerate(b.stream(a.sweeps, n_board)):
            print(f"{k:5d} {s:0{(n_board + 3) // 4}x}")


if __name__ == "__main__":
    main()
