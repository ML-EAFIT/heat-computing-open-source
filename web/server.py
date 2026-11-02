#!/usr/bin/env python3
"""Web viewer for the p-bit chips: one page, three engines behind the same UART protocol.

  /usr/bin/python3 web/server.py [--port 8000] [--host 127.0.0.1]
  then open http://localhost:8000

Backends (all driven through fpga/host.Board, byte for byte the same commands):
  model  model/uart_script.TopModel, the bit-exact Python mirror of rtl/top.v
  rtl    rtl/top.v compiled by Verilator (web/build_rtl.py), bytes piped through
         web/rtl/harness.cpp, which bit-bangs the chip's UART pins
  board  a Tang Nano 9K on a serial port (pyserial, 115200 8N1)
The p-mode chip has no UART; its page runs model/pmode.PmodeArray directly.
The 'pysim' chip is model/pysim.py, the ideal float simulator (Learn tab only).
Standard library only (plus numpy / networkx / pyserial already used by the repo).
"""
import argparse
import collections
import json
import math
import os
import select
import struct
import subprocess
import sys
import threading
import time
import traceback
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, os.path.join(ROOT, "fpga"), os.path.join(ROOT, "model"), os.path.join(ROOT, "tools")):
    sys.path.insert(0, p)
import build_rtl            # noqa: E402
import cases as cs          # noqa: E402
import chips                # noqa: E402
import export_vectors as ev  # noqa: E402
import fixed as fx          # noqa: E402
import golden as gd         # noqa: E402
import host                 # noqa: E402
import pgates as pg         # noqa: E402
import ml                   # noqa: E402
import pmode as pm          # noqa: E402
import pmode_learn as pml   # noqa: E402
import problem as pr        # noqa: E402
import pysim                # noqa: E402

EXAMPLES = os.path.join(ROOT, "examples")
STATIC = os.path.join(HERE, "static")
SERVE_DIRS = {"static": STATIC, "docs": os.path.join(ROOT, "docs"), "examples": EXAMPLES,
              "rtl": os.path.join(ROOT, "rtl"), "model": os.path.join(ROOT, "model")}
MIME = {".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml",
        ".png": "image/png", ".json": "application/json", ".md": "text/plain; charset=utf-8",
        ".v": "text/plain; charset=utf-8"}
# node positions of the multiplier, shared with docs/explainer.html (tools/build_explainer.py)
MULT_POS = {"a0": (40, 60), "a1": (40, 130), "b0": (40, 220), "b1": (40, 290),
            "p0": (200, 40), "q01": (200, 125), "q10": (200, 210), "q11": (200, 300),
            "p1": (360, 150), "p1_aux": (360, 70), "c1": (360, 240),
            "p2": (520, 250), "p2_aux": (520, 170), "p3": (520, 320)}


# ------------------------------------------------------------------ transports
class Wire:
    """Records every command sent and the bytes received after it (the 'Wire' tab)."""

    def __init__(self, inner, log):
        self.inner, self.log = inner, log

    def reset_input_buffer(self):
        self.inner.reset_input_buffer()

    def write(self, data):
        self.log.append(dict(t=time.time(), cmd=chr(data[0]), tx=bytes(data).hex(), rx="", nrx=0, crc=0))
        self.inner.write(data)

    def read(self, n):
        b = self.inner.read(n)
        if self.log:
            e = self.log[-1]
            e["nrx"] += len(b)
            e["crc"] = zlib.crc32(b, e["crc"])
            if len(e["rx"]) < 96:
                e["rx"] = (e["rx"] + b.hex())[:96]
        return b


class RtlSerial:
    """Byte pipe to the Verilator build of rtl/top.v (web/rtl/harness.cpp)."""

    def __init__(self, chip_id, timeout=30.0):
        exe = build_rtl.ensure(chip_id)
        self.p = subprocess.Popen([exe], cwd=HERE, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        self.timeout = timeout

    def reset_input_buffer(self):
        while select.select([self.p.stdout], [], [], 0)[0]:
            if not os.read(self.p.stdout.fileno(), 65536):
                break

    def write(self, data):
        self.p.stdin.write(bytes(data))

    def read(self, n):
        out, t0 = b"", time.time()
        while len(out) < n:
            left = self.timeout - (time.time() - t0)
            if left <= 0 or not select.select([self.p.stdout], [], [], left)[0]:
                break
            chunk = os.read(self.p.stdout.fileno(), n - len(out))
            if not chunk:
                break
            out += chunk
        return out

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=2)
        except Exception:
            self.p.kill()


def open_transport(chip, backend, port=None):
    if backend == "model":
        return host.SimSerial(**chips.model_cfg(chip))
    if backend == "rtl":
        return RtlSerial(chip["id"])
    if backend == "board":
        import serial
        s = serial.Serial(port, 115200, timeout=5.0)
        s.reset_input_buffer()
        return s
    raise ValueError(f"unknown backend {backend!r}")


# ------------------------------------------------------------------ programs
def builtin_cases():
    allc = {x["name"]: x for x in cs.all_cases()}
    allc.update({x["name"]: x for x, _ in cs.sparse_cases()})
    return allc


def normalize(c):
    """Give built-in cases the fields problem.decode() uses, plus the live-edit state:
    hf_free (biases without pins) and pins {node index: bit} (circuit clamps become pins)."""
    c.setdefault("hf_free", list(c["hf"]))
    names = c.get("nodes")
    c["pins"] = {names.index(w): int(b) for w, b in c.get("clamps", {}).items()} if names else {}
    c.setdefault("warnings", [])
    if "kind" not in c:
        if "nodes" in c:
            c.update(kind="circuit", gates=[], show={})
        elif "edges" in c and c["name"].startswith("maxcut"):
            c.update(kind="maxcut", cut_w=[1.0] * len(c["edges"]))
        else:
            c["kind"] = "ising"
    return c


def load_program(req, chip):
    """Request -> case dict. req: {example: name} | {case: name} | {problem: {...}}."""
    if req.get("problem") is not None:
        spec = req["problem"]
        if isinstance(spec, str):
            spec = json.loads(spec)
        c = pr.build(spec, req.get("name") or "custom")
    elif req.get("example"):
        c = pr.load(os.path.join(EXAMPLES, os.path.basename(req["example"]) + ".json"))
    else:
        name = req.get("case") or chip["program"]
        if name == "factor6_pt":       # the example file has the gate list and the integer view
            c = pr.load(os.path.join(EXAMPLES, "factor6.json"))
            c.update(name="factor6_pt", ladder=list(cs.PT_LADDER))
        elif name == "factor6":
            c = pr.load(os.path.join(EXAMPLES, "factor6.json"))
        else:
            allc = builtin_cases()
            if name not in allc:
                raise ValueError(f"unknown case {name!r}")
            c = dict(allc[name])
    return normalize(c)


def layout(c):
    n = c["n"]
    names = c.get("nodes") or [str(i) for i in range(n)]
    if c.get("nodes") and all(w in MULT_POS for w in c["nodes"]):
        pos = [MULT_POS[w] for w in c["nodes"]]
        return names, [(x / 560, y / 360) for x, y in pos]
    side = int(round(math.sqrt(n)))
    image_like = c.get("grid") or c["name"].startswith(("g12", "grid", "mrf", "hopfield"))
    if side * side == n and n >= 16 and image_like:
        return names, [((i % side) / (side - 1), (i // side) / (side - 1)) for i in range(n)]
    if n == 1:
        return names, [(0.5, 0.5)]
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from((i, j) for i in range(n) for j in range(i + 1, n) if c["Jq"][i][j])
    p = nx.circular_layout(g) if c["kind"] in ("maxcut", "ising") and n <= 16 else nx.spring_layout(g, seed=1)
    xs = np.array([p[i] for i in range(n)])
    lo, hi = xs.min(0), xs.max(0)
    xs = (xs - lo) / np.where(hi - lo > 0, hi - lo, 1)
    return names, [tuple(map(float, v)) for v in xs]


def all_energies(c):
    """H of every state index with the programmed (quantized) J, h; numpy version of problem.energy."""
    n = c["n"]
    idx = np.arange(1 << n)
    sp = 2 * ((idx[:, None] >> np.arange(n)) & 1) - 1
    J = np.array(c["Jq"], float) / (1 << c["wfrac"])
    h = np.array(c["hq"], float) / (1 << c["wfrac"])
    return -0.5 * np.einsum("si,ij,sj->s", sp, J, sp) - sp @ h


def program_summary(c, rev=0):
    n = c["n"]
    if "_pos" not in c:                 # positions stay put while the program is edited live
        c["_names"], c["_pos"] = layout(c)
    names, pos = c["_names"], c["_pos"]
    edges = [[i, j, c["Jq"][i][j] / 16] for i in range(n) for j in range(i + 1, n) if c["Jq"][i][j]]
    out = dict(name=c["name"], kind=c["kind"], n=n, names=names, pos=pos, edges=edges,
               h=[v / 16 for v in c["hq"]], h_free=list(c["hf_free"]), beta=c["beta_q"] / 16, colors=c["colors"],
               ins=c.get("ins", []), outs=c.get("outs", []), clamps=c.get("clamps", {}),
               pins={str(i): b for i, b in c["pins"].items()}, show=c.get("show", {}),
               gates=[[k, *w] for k, w in c.get("gates", [])], warnings=c.get("warnings", []),
               edited=c.get("edited", False), rev=rev,
               schedule=ev.case_schedule(c), ladder=c.get("ladder"))
    if n <= 16:
        e = all_energies(c)
        emin = float(e.min())
        gs = np.flatnonzero(np.isclose(e, emin)).tolist()
        out["ground"] = dict(energy=emin, states=[dict(state=f"{s:x}", lines=pr.decode(c, s)) for s in gs[:8]],
                             count=len(gs))
    return out


# ------------------------------------------------------------------ session
class Session:
    def __init__(self):
        self.lock = threading.RLock()
        self.wire = collections.deque(maxlen=400)
        self.chip = self.backend = self.board = self.case = None
        self.pm = self.ml = None
        self.sim = False
        self.status = "not connected"
        self.rev = 0
        self.last_req = {}

    def close(self):
        if self.board is not None and hasattr(self.board.s.inner, "close"):
            try:
                self.board.s.inner.close()
            except Exception:
                pass
        self.board = None

    # -------------------------------------------------- connect / program
    def connect(self, chip_id, backend, port=None):
        chip = chips.BY_ID[chip_id]
        if backend not in chip["backends"]:
            raise ValueError(f"{chip['title']}: backend {backend} unavailable. {chip.get('note', '')}")
        self.close()
        self.wire.clear()
        self.chip, self.backend, self.case, self.ml = chip, backend, None, None
        self.sim = chip_id == "pysim"
        if self.sim:
            self.status = "ideal Python simulator ready"
            return dict(chip=chip_id, backend=backend, info="model/pysim.py: float chromatic Gibbs, any size")
        if chip_id == "pmode_chain4":
            self.pm_reset()
            self.status = "p-mode model ready"
            return dict(chip=chip_id, backend=backend, info="Python model of pmode_array.v (M = 4)")
        t0 = time.time()
        self.board = host.Board(None, transport=Wire(open_transport(chip, backend, port), self.wire))
        n, hb, sp = self.board.ping()
        self.n_board, self.hb, self.sp = n, hb, sp
        want = chip["params"]
        warn = []
        if n != want["N"] or hb != want["HIST_BITS"] or bool(sp) != bool(want.get("SPARSE")):
            warn.append(f"the {backend} answers N={n} HIST_BITS={hb} sparse={bool(sp)}, "
                        f"but {chip['id']} has N={want['N']} HIST_BITS={want['HIST_BITS']}: "
                        "is the right bitstream flashed?")
        self.status = f"{backend}: N={n} HIST_BITS={hb}" + (f" SPARSE P={sp['P']} D={sp['D']}" if sp else "")
        info = self.program({})
        return dict(chip=chip_id, backend=backend, ping=dict(N=n, HIST_BITS=hb, sparse=sp), warnings=warn,
                    connect_s=round(time.time() - t0, 2), program=info)

    def need_board(self):
        if self.board is None:
            raise ValueError("the ideal simulator has no UART: Program / Run / Results use the FPGA chips "
                             "(the Learn tab works here)" if self.sim else "connect first")

    def need_engine(self):
        """A sampler for the Learn tab: a UART chip or the ideal simulator."""
        if not self.sim:
            self.need_board()

    def need_case(self):
        self.need_board()
        if self.case is None:
            raise ValueError("no program loaded: a training demo owns the chip; press 'Use trained model' "
                             "(Learn tab) or load a program")

    def program(self, req):
        self.need_board()
        c = load_program(req, self.chip)
        self.last_req = {k: v for k, v in req.items() if k in ("case", "example", "problem", "name")}
        return self.install(c)

    def install(self, c, schedule=None):
        """Program case c into the chip and make it the session program."""
        if self.sim:                                    # the ideal simulator samples c["Jf"], c["hf_free"] itself
            self.case = c
            self.rev += 1
            return program_summary(c, self.rev)
        self.need_board()
        if c["n"] > self.n_board:
            raise ValueError(f"{c['name']} needs {c['n']} p-bits; this chip has {self.n_board}")
        if c.get("ladder") and not self.chip["params"].get("TEMPER_K"):
            c = dict(c, ladder=None)
        b = self.board
        if self.sp:
            b.load_sparse(c, self.n_board, self.sp)
        else:
            b.load(c, self.n_board)
        if self.chip["params"].get("TEMPER_K", 1) > 1:
            for k, beta in enumerate(c.get("ladder") or cs.PT_LADDER):
                b.cmd("L", k, beta)
                b.ack()
        self.case = c
        self.rev += 1
        self.restart(schedule or {"mode": "anneal"})
        return program_summary(c, self.rev)

    # -------------------------------------------------- live edits (spins and schedule keep running)
    def _rebuild(self, c, struct):
        """Derived words from the free problem: hq from hf_free + pins, Jq from Jf, colours if the graph changed."""
        n, wf = c["n"], c["wfrac"]
        c["hf"] = [pg.CLAMP_HI if c["pins"].get(i) == 1 else pg.CLAMP_LO if c["pins"].get(i) == 0 else c["hf_free"][i]
                   for i in range(n)]
        c["hq"] = [fx.to_q(v, wf, fx.WB) for v in c["hf"]]
        c["Jq"] = [[fx.to_q(v, wf, fx.WB) for v in row] for row in c["Jf"]]
        if struct:
            c["colors"] = gd.greedy_coloring(c["Jf"])
            c["order"] = cs._order(c["colors"])
        names = c.get("nodes")
        if names:
            c["clamps"] = {names[i]: b for i, b in sorted(c["pins"].items())}
        if c["kind"] == "maxcut":               # keep the cut readout consistent with the edited graph
            c["edges"] = [(i, j) for i in range(n) for j in range(i + 1, n) if c["Jf"][i][j] < 0]
            c["cut_w"] = [-c["Jf"][i][j] for i, j in c["edges"]]

    def _edit(self, fn, bias_only):
        """Apply fn(c) to a copy of the program, push it to the engine, keep it only if that worked."""
        self.need_engine()
        import copy
        old = self.case
        c = copy.deepcopy(old)
        struct = fn(c)
        self._rebuild(c, struct)
        c["edited"] = True
        b = self.board
        if self.sim:
            pass
        elif self.sp:
            lay = ev.sparse_layout(c, self.sp["P"])
            if lay["D"] > self.sp["D"] or lay["R"] > (1 << self.sp["LW"]):
                raise ValueError(f"this edit does not fit the sparse chip: it needs degree {lay['D']} and "
                                 f"{lay['R']} rounds ({len(c['colors'])} colours), the build has degree <= "
                                 f"{self.sp['D']} and <= {1 << self.sp['LW']} rounds. Keep the graph "
                                 "two-colourable (e.g. change existing couplings instead of adding one).")
            b.load_sparse(c, self.n_board, self.sp)     # h lives in the node words: reload the lanes
        elif bias_only:
            changed = [i for i in range(c["n"]) if c["hq"][i] != old["hq"][i]]
            for i in changed:
                b.cmd("H", i, c["hq"][i] & 0xFF)
            for _ in changed:
                b.ack()
        else:
            b.load(c, self.n_board)
        if not self.sim and self.chip["params"].get("TEMPER_K", 1) > 1:
            b.cmd("Z")                                   # resync replicas after W / H changes (top.v)
            b.ack()
        self.case = c
        self.rev += 1
        return program_summary(c, self.rev)

    def _node(self, c, i):
        i = int(i)
        if not 0 <= i < c["n"]:
            raise ValueError(f"node {i} out of range 0..{c['n'] - 1}")
        return i

    def pin(self, req):
        """Pin p-bit `node` to 1 / 0 with the extreme bias (like a clamp), or free it (value null)."""
        def fn(c):
            i, v = self._node(c, req["node"]), req.get("value")
            if v is None:
                c["pins"].pop(i, None)
            else:
                c["pins"][i] = 1 if int(v) else 0
            return False
        return self._edit(fn, bias_only=True)

    def pin_group(self, req):
        """Pin a circuit group (show: {"p": ["p0", ...]}) to an integer, little-endian, or free it."""
        def fn(c):
            g = req["group"]
            wires = c.get("show", {}).get(g)
            if not wires:
                raise ValueError(f"no group {g!r}; groups: {', '.join(c.get('show', {})) or 'none'}")
            idx = [c["nodes"].index(w) for w in wires]
            v = req.get("value")
            if v is None or v == "":
                for i in idx:
                    c["pins"].pop(i, None)
                return False
            v = int(v)
            if not 0 <= v < (1 << len(idx)):
                raise ValueError(f"{g} has {len(idx)} bits: value must be 0..{(1 << len(idx)) - 1}")
            for k, i in enumerate(idx):
                c["pins"][i] = (v >> k) & 1
            return False
        return self._edit(fn, bias_only=True)

    def bias(self, req):
        def fn(c):
            i, h = self._node(c, req["i"]), float(req["h"])
            if abs(h) > pr.QMAX:
                raise ValueError(f"|h| must be <= {pr.QMAX} (Q3.4)")
            c["hf_free"][i] = h
            return False
        return self._edit(fn, bias_only=True)

    def coupling(self, req):
        """Set J_ij (0 removes the edge). A new or removed edge recolours the graph."""
        def fn(c):
            i, j, w = self._node(c, req["i"]), self._node(c, req["j"]), float(req["J"])
            if i == j:
                raise ValueError("a coupling needs two different p-bits")
            if abs(w) > pr.QMAX:
                raise ValueError(f"|J| must be <= {pr.QMAX} (Q3.4)")
            struct = (c["Jf"][i][j] == 0) != (w == 0)
            c["Jf"][i][j] = c["Jf"][j][i] = w
            if c["kind"] == "qubo":                     # the x-objective no longer matches the edited energy
                c["kind"] = "ising"
            q = fx.to_q(w, c["wfrac"], fx.WB) / (1 << c["wfrac"])
            c["warnings"] = [f"J[{i}][{j}] = {w:g} is stored as {q:g} (1/16 steps)"] if abs(q - w) > 1e-9 else []
            return struct
        return self._edit(fn, bias_only=False)

    def reset_program(self, req):
        return self.program(dict(self.last_req))

    def restart(self, req):
        """Reseed and (re)start a schedule: mode 'anneal' (the program's schedule) or 'const' at beta."""
        self.need_case()
        c, b = self.case, self.board
        if req.get("mode") == "const":
            bq = fx.to_q(float(req.get("beta", c["beta_q"] / 16)), fx.BFRAC, fx.BB, signed=False)
            sc = dict(mode=0, beta_start=bq, beta_end=bq, delta=0, sps=1, table=None)
        elif req.get("mode") == "ramp":
            # warm start: 16 geometric steps from beta/8 up to beta (sps sweeps each), then hold beta;
            # lets the chain leave the all -1 reset state before it is counted
            b1 = float(req.get("beta", c["beta_q"] / 16))
            sps = int(req.get("sps", 4))
            table = [max(1, fx.to_q(b1 / 8 * 8 ** (k / 15), fx.BFRAC, fx.BB, signed=False)) for k in range(16)]
            sc = dict(mode=2, beta_start=table[0], beta_end=table[-1], delta=0, sps=sps, table=table)
        else:
            sc = ev.case_schedule(c)
        b.schedule(sc["mode"], sc["beta_start"], sc["beta_end"], sc["delta"], sc["sps"], sc.get("table"))
        b.reseed(int(req.get("seed", 0x12345678)))
        if self.chip["params"].get("TEMPER_K", 1) > 1:
            b.cmd("Z")
            b.ack()
        self.sched = fx.Schedule(sc["mode"], sc["beta_start"], sc["beta_end"], sc["delta"], sc["sps"],
                                 sc.get("table"))
        self.t = 0
        return dict(schedule=sc)

    # -------------------------------------------------- runs
    def stream(self, req):
        self.need_case()
        c = self.case
        k = max(1, min(int(req.get("sweeps", 50)), 4000 if c["n"] <= 16 else 500))
        states = self.board.stream(k, self.n_board)
        temper = self.chip["params"].get("TEMPER_K", 1) > 1
        out = []
        mask = (1 << c["n"]) - 1
        for s in states:
            s &= mask
            beta = (c.get("ladder") or cs.PT_LADDER)[0] if temper else self.sched.beta
            out.append(dict(t=self.t, s=f"{s:x}", e=pr.energy(c, s), beta=beta / 16))
            self.t += 1
            if not temper:
                self.sched.sweep_done()
        last = states[-1] & mask
        return dict(samples=out, leds=self.leds(last), busy=False, decode=pr.decode(c, last))

    def leds(self, s):
        return [(s >> k) & 1 for k in range(5)]

    def sample(self, req):
        self.need_case()
        c, b = self.case, self.board
        sweeps = max(1, min(int(req.get("sweeps", 20000)), 2_000_000))
        self.restart(dict(mode="const", beta=req.get("beta", c["beta_q"] / 16), seed=req.get("seed", 0x12345678)))
        t0 = time.time()
        b.run(sweeps)
        dt = time.time() - t0
        h = b.hist(self.hb)
        b.cmd("U")
        raw = b.read(4 * (self.n_board + 1))
        b.ack()
        cnt = np.frombuffer(raw, dtype="<u4")
        ns, up = int(cnt[0]), cnt[1:1 + c["n"]]
        mag = (2 * up / max(ns, 1) - 1).tolist()
        out = dict(sweeps=sweeps, seconds=round(dt, 3), rate=round(sweeps / max(dt, 1e-9)), mag=mag, n=c["n"],
                   hist_bits=self.hb)
        n = c["n"]
        if n <= self.hb:
            hh = h.reshape(-1, 1 << n).sum(0) if n < self.hb else h
            bq = fx.to_q(float(req.get("beta", c["beta_q"] / 16)), fx.BFRAC, fx.BB, signed=False)
            exact, _ = gd.exact_boltzmann(np.array(c["Jq"]) / 16, np.array(c["hq"]) / 16, bq / 16)
            top = np.argsort(-exact)[:24]
            emp = hh / max(hh.sum(), 1)
            out.update(tv=round(float(gd.tv(hh, exact)), 5), kl=round(float(gd.kl(hh, exact)), 5),
                       top=[dict(state=f"{int(s):x}", exact=float(exact[s]), chip=float(emp[s]),
                                 energy=pr.energy(c, int(s))) for s in top])
        else:
            top = np.argsort(-h)[:24]
            emp = h / max(h.sum(), 1)
            out["top"] = [dict(state=f"{int(s):x}", chip=float(emp[s])) for s in top if h[s] > 0]
            out["note"] = f"histogram covers the low {self.hb} of {n} spins"
        return out

    def solve(self, req):
        self.need_case()
        c = self.case
        runs = max(1, min(int(req.get("runs", 10)), 200))
        r = host.solve(self.board, c, self.n_board, runs, int(req.get("seed", 0x12345678)))
        seen = sorted(r["seen"].items(), key=lambda kv: (pr.energy(c, kv[0]), -kv[1]))
        self.restart({"mode": "anneal"})
        return dict(runs=runs, sweeps=r["sweeps"], hits=r["hits"], best=pr.decode(c, r["best"]),
                    states=[dict(state=f"{s:x}", count=k, energy=pr.energy(c, s), lines=pr.decode(c, s))
                            for s, k in seen[:8]])

    # -------------------------------------------------- machine learning (model/ml.py, model/train.py)
    def ml_check(self, demo):
        if demo not in ML_DEMOS:
            raise ValueError(f"unknown demo {demo!r}; demos: {', '.join(ML_DEMOS)}")
        d = ML_DEMOS[demo]
        if self.chip is None:
            raise ValueError("connect first")
        if self.chip["id"] not in d["chips"]:
            raise ValueError(f"{d['title']} needs {d['needs']}; connect that chip first")
        if self.board is not None:                      # the bitstream that answered must fit too
            p = self.chip["params"]
            if d["sparse"] != bool(self.sp) or p.get("TEMPER_K", 1) > 1 or self.n_board < d["n"]:
                raise ValueError(f"{d['title']} needs {d['needs']}; connect that chip first")
        elif not self.sim and self.chip["id"] != "pmode_chain4":
            self.need_board()
        return d

    def ml_start(self, req):
        demo = req.get("demo")
        d = self.ml_check(demo)
        seed = int(req.get("seed", 1))
        self.ml = dict(demo=demo)
        if demo in ("glyphs", "bars3"):
            import train
            td = train.DEMOS[demo]
            kw = dict(lr=td["lr"], lr_tau=td.get("lr_tau"), decay=td.get("decay", 1e-3), seed=seed)
            if self.sim:
                tr = pysim.SimTrainer(td["nv"], td["nh"], td["data"](), **kw)
            else:
                b, n, hb = self.board, self.n_board, self.hb

                class BoardChip:                # train.Trainer talks to the chip with .cmd only
                    def __init__(self):
                        self.n = n

                    def cmd(self, c, *args):
                        return b.xfer(c, *args, n=n, hb=hb)
                tr = train.Trainer(BoardChip(), td["nv"], td["nh"], td["data"](), **kw)
            self.ml.update(trainer=tr, hist=[tr.kl()], target=td["epochs"])
            self.case = None
            return self.ml_status()
        if demo == "digits":
            self.ml.update(digits=ml.Digits(seed=seed - 1))
            return self.ml_status()
        if demo == "pmode":
            self.ml.update(pmt=pml.PmodeTrainer(seed=6 + seed))
            return self.ml_status()
        if demo == "hopfield":
            c = normalize(ml.hopfield_case(letters=ML_LETTERS))
            out = self.install(c, {"mode": "const", "beta": c["beta"]})
            return dict(demo=demo, program=out, letters=ML_LETTERS, engine=self.engine(),
                        patterns={l: list(ml.bits(ml.HOPFIELD[l])) for l in ML_LETTERS})
        if demo == "mrf":
            return dict(demo=demo, images=ml.IMAGES16, sides=MRF_SIDES if self.sim else [16],
                        **self.ml_denoise(dict(req, sweeps=200)))
        raise ValueError(demo)

    def engine(self):
        return "ideal simulator (float)" if self.sim else f"{self.chip['id']} · {self.backend}"

    def ml_status(self):
        m = self.ml
        if not m or not any(k in m for k in ("trainer", "digits", "pmt")):
            raise ValueError("start a training demo first")
        if "digits" in m:
            d = m["digits"]
            return dict(demo="digits", epoch=d.epoch, target=d.target, engine=self.engine(),
                        curves=[dict(name="test (357 held-out)", values=[x["test"] for x in d.hist]),
                                dict(name="train (1440)", values=[x["train"] for x in d.hist])],
                        curve_title="classification accuracy per epoch (exact, free energy)",
                        metrics={"test accuracy": d.hist[-1]["test"], "train accuracy": d.hist[-1]["train"]},
                        filters=d.filters(), nv=d.rbm.nv, nh=d.rbm.nh)
        if "pmt" in m:
            st = m["pmt"].status()
            last = st["hist"][-1]
            return dict(demo="pmode", engine=self.engine(), **st,
                        curves=[dict(name="worst parameter error", values=[h["param_err"] * 16 for h in st["hist"]])],
                        curve_title="worst |learned − true| over A and h, in LSB (1/16)",
                        metrics_raw={"worst parameter error (LSB)": round(last["param_err"] * 16),
                                     "worst moment error": round(last["moment_err"], 3)})
        tr = m["trainer"]
        Jq, hq = tr.quantized()
        out = dict(demo=m["demo"], epoch=len(m["hist"]) - 1, target=m["target"], kl=m["hist"], engine=self.engine(),
                   J=Jq.tolist(), h=hq.tolist(), nv=tr.nv, nh=tr.nh, names=self.ml_names())
        if m["demo"] == "glyphs":
            acc, nt = ml.glyph_accuracy(Jq, hq)
            out["metrics"] = {"classification accuracy (exact, 40 test images)": acc}
        else:
            mass, comp = ml.bars_quality(Jq, hq)
            out["metrics"] = {"model mass on the 14 valid images": mass, "valid bottom-row completions": comp}
        return out

    def ml_names(self):
        tr = self.ml["trainer"]
        if self.ml["demo"] == "glyphs":
            return [f"x{k}" for k in range(9)] + [f"L_{l}" for l in ml.LABELS] + [f"h{k}" for k in range(tr.nh)]
        return [f"x{k}" for k in range(tr.nv)] + [f"h{k}" for k in range(tr.nh)]

    def ml_train(self, req):
        m = self.ml
        if not m or not any(k in m for k in ("trainer", "digits", "pmt")):
            raise ValueError("start a training demo first")
        for _ in range(max(1, min(int(req.get("epochs", 5)), 50))):
            if "digits" in m:
                m["digits"].train_epoch()
            elif "pmt" in m:
                m["pmt"].step()
            else:
                m["trainer"].step()
                m["hist"].append(m["trainer"].kl())
        return self.ml_status()

    def ml_use(self, req):
        """The trained Boltzmann machine becomes the session program (Run tab, pins, inference)."""
        m = self.ml
        if not m or "trainer" not in m:
            raise ValueError("train a model first (glyphs or bars & stripes)")
        c = normalize(ml.trained_case(m["trainer"], m["demo"], f"trained_{m['demo']}"))
        out = self.install(c, {"mode": "const", "beta": 1.0})
        return dict(program=out, **self.ml_status())

    def ml_infer(self, req):
        """Pin nodes (and optionally set evidence biases), sample at constant beta, return marginals."""
        self.need_engine()
        c = self.case
        if c is None:
            raise ValueError("no program: start a demo (and use the trained model) first")
        pins = {int(i): int(v) for i, v in (req.get("pins") or {}).items()}
        evidence = {int(i): float(v) for i, v in (req.get("evidence") or {}).items()}

        def fn(cc):
            cc["pins"] = pins
            for i, v in evidence.items():
                cc["hf_free"][i] = v if self.sim else max(-pr.QMAX, min(pr.QMAX, v))
            return False
        self._edit(fn, bias_only=True)
        c = self.case
        beta = float(req.get("beta", c["beta_q"] / 16))
        sweeps = max(10, min(int(req.get("sweeps", 1000)), 20000))
        if self.sim:
            return self._sim_infer(c, pins, beta, sweeps, req)
        sps = max(1, min(int(req.get("ramp_sps", 4)), 64))
        self.restart({"mode": "ramp", "beta": beta, "sps": sps, "seed": int(req.get("seed", 0x12345678))})
        ramp = 16 * sps
        n = c["n"]
        out = dict(sweeps=sweeps, beta=beta, warmup=ramp + 20)
        if n <= 16:
            self.stream({"sweeps": ramp + 20})
            st = []
            while len(st) < sweeps:
                st += [int(x["s"], 16) for x in self.stream({"sweeps": min(sweeps - len(st), 4000)})["samples"]]
            arr = np.array(st)
            out.update(self._sample_stats(arr, n, req))
            # exact P(bit = 1 | pins) from the programmed words (pinned nodes clamped, free biases)
            Jq = np.array(c["Jq"], float) / (1 << c["wfrac"])
            hq = np.array(self._free_hq(c), float) / (1 << c["wfrac"])
            out["marg_exact"] = self._exact_marg(Jq, hq, pins, beta, n)
        else:
            self.board.run(ramp + 20)
            self.board.run(sweeps)
            self.board.cmd("U")
            raw = self.board.read(4 * (self.n_board + 1))
            self.board.ack()
            cnt = np.frombuffer(raw, dtype="<u4")
            out["marg"] = (cnt[1:1 + n] / max(int(cnt[0]), 1)).tolist()
        return out

    @staticmethod
    def _sample_stats(arr, n, req):
        vals, cnt = np.unique(arr, return_counts=True)
        top = np.argsort(-cnt)[:max(1, min(int(req.get("top", 12)), 1 << 16))]
        return dict(marg=(((arr[:, None] >> np.arange(n)) & 1).mean(0)).tolist(),
                    top=[dict(state=f"{int(vals[k]):x}", count=int(cnt[k])) for k in top],
                    recent=[f"{int(v):x}" for v in arr[-12:]])

    @staticmethod
    def _exact_marg(J, h, pins, beta, n):
        free, _, mf = ml.cond_dist(J, h, pins, beta)
        ex = [None] * n
        for i, v in pins.items():
            ex[i] = float(v)
        for k, i in enumerate(free):
            ex[i] = float(mf[k])
        return ex

    def _sim_infer(self, c, pins, beta, sweeps, req):
        """Ideal simulator: float J and biases, hard pins, 50 parallel chains, no warm-up ramp needed."""
        n = c["n"]
        J, h = np.array(c["Jf"], float), np.array(c["hf_free"], float)
        g = pysim.Gibbs(J, h, beta=beta, chains=50, colors=c["colors"], seed=int(req.get("seed", 0x12345678)) & 0xFFFF)
        g.clamp(sorted(pins), [pins[i] for i in sorted(pins)])
        warm = 50
        S = g.run(-(-sweeps // g.chains), burn=warm)
        out = dict(sweeps=len(S), beta=beta, warmup=warm, chains=g.chains, engine=self.engine())
        if n <= 30:
            out.update(self._sample_stats(pysim.states(S), n, req))
        else:
            out["marg"] = (S > 0).mean(0).tolist()
        if n - len(pins) <= 20:
            out["marg_exact"] = self._exact_marg(J, h, pins, beta, n)
        return out

    def _free_hq(self, c):
        """Quantized biases without the pins (the exact conditional clamps the pinned nodes itself)."""
        return [fx.to_q(v, c["wfrac"], fx.WB) for v in c["hf_free"]]

    def ml_hopfield(self, req):
        """Noisy stored letter -> evidence bias lam * y -> sample -> recalled image (P(bit) > 1/2)."""
        c = self.case
        if c is None or c.get("name") != "hopfield_letters":
            self.ml_start({"demo": "hopfield"})
        letter = req.get("letter", ML_LETTERS[0])
        xi = list(ml.bits(ml.HOPFIELD[letter]))
        rng = np.random.default_rng(int(req.get("seed", 0)))
        p, lam = float(req.get("noise", 0.15)), float(req.get("lam", 0.5))
        trials = max(1, min(int(req.get("trials", 1)), 50))
        ok, last = 0, None
        for t in range(trials):
            y = req["input"] if (req.get("input") and t == 0) else ml.noisy(xi, p, rng)
            r = self.ml_infer(dict(evidence={i: lam * (2 * b - 1) for i, b in enumerate(y)}, pins={},
                                   sweeps=int(req.get("sweeps", 300)), beta=self.case["beta_q"] / 16,
                                   ramp_sps=int(req.get("ramp_sps", 8)), seed=0x1234 + t))
            rec = [int(m > 0.5) for m in r["marg"]]
            ok += rec == xi
            if last is None:
                last = dict(noisy=list(y), recalled=rec, marg=r["marg"], marg_exact=r["marg_exact"])
        return dict(letter=letter, stored=xi, trials=trials, successes=ok, engine=self.engine(), **last,
                    noisy_errors=sum(a != b for a, b in zip(last["noisy"], xi)),
                    recalled_errors=sum(a != b for a, b in zip(last["recalled"], xi)))

    def ml_denoise(self, req):
        """MRF posterior P(x_i = 1 | y) per pixel: the sparse chip's up-counters, or the ideal simulator."""
        self.ml_check("mrf")
        name = req.get("image", "disk")
        side = int(req.get("side", 16)) if self.sim else 16
        if side not in MRF_SIDES:
            raise ValueError(f"side must be one of {MRF_SIDES}")
        truth = ml.image(name, side)
        npx = side * side
        p = min(0.45, max(0.01, float(req.get("noise", 0.15))))
        rng = np.random.default_rng(int(req.get("seed", 7)))
        y = ml.noisy(truth, p, rng)
        Jc = float(req.get("J", 0.5))
        lam = 0.5 * math.log((1 - p) / p)
        sweeps = max(20, min(int(req.get("sweeps", 300)), 20000))
        prog = None
        t0 = time.time()
        if self.sim:
            if side == 16:                              # same program as the sparse chip, for the Run graph
                prog = self.install(normalize(ml.mrf_case(truth, y, J=Jc, p=p)))
            marg = pysim.grid_denoise(y, side, J=Jc, lam=lam, sweeps=sweeps, chains=4,
                                      seed=int(req.get("seed", 7))).tolist()
            dt = time.time() - t0
        else:
            c = normalize(ml.mrf_case(truth, y, J=Jc, p=p))
            prog = self.install(c, {"mode": "ramp", "beta": 1.0, "sps": 4})
            b = self.board
            b.run(16 * 4 + max(10, sweeps // 5))         # warm-up ramp to beta = 1 + burn-in, then count
            t0 = time.time()
            b.run(sweeps)
            dt = time.time() - t0
            b.cmd("U")
            raw = b.read(4 * (self.n_board + 1))
            b.ack()
            cnt = np.frombuffer(raw, dtype="<u4")
            marg = (cnt[1:257] / max(int(cnt[0]), 1)).tolist()
        den = [int(m > 0.5) for m in marg]
        err = lambda a: sum(u != v for u, v in zip(a, truth)) / npx       # noqa: E731
        return dict(image=name, side=side, truth=truth, noisy=y, denoised=den, marg=marg, noise=p, J=Jc,
                    lam=lam, sweeps=sweeps, seconds=round(dt, 2), engine=self.engine(),
                    chains=4 if self.sim else 1,
                    error_noisy=err(y), error_denoised=err(den), program=prog)

    def ml_digits(self, req):
        """digits demo: classify an image, generate from a label, complete a partial image."""
        m = self.ml
        if not m or "digits" not in m:
            raise ValueError("start the digits demo first")
        d = m["digits"]
        act = req.get("action")
        seed = int(req.get("seed", 0)) & 0xFFFF
        if act == "tests":
            rng = np.random.default_rng(seed)
            k = rng.choice(len(d.yte), size=min(int(req.get("k", 24)), 64), replace=False)
            return dict(images=d.Xte[k].tolist(), labels=d.yte[k].tolist(), index=k.tolist())
        if act == "classify":
            img = [int(v) for v in req["image"]]
            marg, _ = d.sample({i: b for i, b in enumerate(img)}, sweeps=int(req.get("sweeps", 200)), seed=seed)
            return dict(exact=d.label_probs([img])[0].tolist(), sampled=marg[64:].tolist())
        if act == "generate":
            c = int(req["label"])
            marg, st = d.sample({64 + k: int(k == c) for k in range(10)}, sweeps=int(req.get("sweeps", 200)), seed=seed)
            return dict(label=c, marg=marg[:64].tolist(), samples=st[:12, :64].tolist())
        if act == "complete":
            pins = {int(i): int(v) for i, v in req["pins"].items()}
            marg, _ = d.sample(pins, sweeps=int(req.get("sweeps", 200)), seed=seed)
            known = [pins.get(i) for i in range(64)]
            img = [known[i] if known[i] is not None else int(marg[i] > 0.5) for i in range(64)]
            return dict(marg=marg[:64].tolist(), labels=marg[64:].tolist(), completed=img,
                        exact=d.label_probs([img])[0].tolist())
        raise ValueError(f"unknown digits action {act!r}")

    # -------------------------------------------------- p-modes
    def pm_reset(self, seed=fx.LFSR_DEFAULT_SEED):
        c = pm.case("chain4")
        x0 = [fx.to_q(v, pm.XF, 16) for v in ev.PM_X0["chain4"]]
        self.pm = pm.PmodeArray(c["Aq"], c["hq"], c["lamq"], c["k"], c["sig"], x0=x0, seed=seed)
        self.pm_case, self.pm_t, self.pm_hist = c, 0, np.zeros((c["m"], 64))
        self.pm_edges = np.linspace(-2.5, 2.5, 65)

    def pm_step(self, req):
        if self.pm is None:
            raise ValueError("connect to the p-mode chip first")
        k = max(1, min(int(req.get("steps", 200)), 20000))
        every = max(1, k // 400)
        traj = []
        for i in range(k):
            self.pm.step()
            xs = self.pm.xs()
            for m, x in enumerate(xs):
                b = np.searchsorted(self.pm_edges, x) - 1
                if 0 <= b < 64:
                    self.pm_hist[m, b] += 1
            if i % every == 0:
                traj.append(dict(t=self.pm_t, x=xs))
            self.pm_t += 1
        return dict(traj=traj, hist=self.pm_hist.tolist(), edges=self.pm_edges.tolist(),
                    exact=pm_exact(), leds=[int(x > 0) for x in self.pm.xs()])

    def pm_info(self):
        c = self.pm_case
        return dict(m=c["m"], A=c["A"], h=c["h"], lam=c["lam"], eps=c["eps"], T=c["T"],
                    x0=ev.PM_X0["chain4"], names=[f"x{i}" for i in range(c["m"])])


_PM_EXACT = None


def pm_exact():
    """Per-site marginals of e^(-V/T) for chain4 on the histogram grid (4-D grid sum, cached)."""
    global _PM_EXACT
    if _PM_EXACT is None:
        c = pm.case("chain4")
        g = np.linspace(-2.5, 2.5, 65)
        mid = (g[:-1] + g[1:]) / 2
        X = np.stack(np.meshgrid(*[mid] * c["m"], indexing="ij"), -1)
        V = pm.potential(X, np.array(c["A"]), np.array(c["h"]), c["lam"])
        w = np.exp(-(V - V.min()) / c["T"])
        w /= w.sum()
        _PM_EXACT = [w.sum(axis=tuple(a for a in range(c["m"]) if a != m)).tolist() for m in range(c["m"])]
    return _PM_EXACT


# ------------------------------------------------------------------ compare model vs RTL
def compare(chip_id, sweeps=2000):
    """Same session on Model and RTL; every reply byte must match."""
    chip = chips.BY_ID[chip_id]
    if "rtl" not in chip["backends"]:
        raise ValueError(chip.get("note", "no RTL"))
    logs = {}
    t = {}
    for be in ("model", "rtl"):
        log = collections.deque()
        tr = open_transport(chip, be)
        t0 = time.time()
        b = host.Board(None, transport=Wire(tr, log))
        n, hb, sp = b.ping()
        c = load_program({}, chip)
        if sp:
            b.load_sparse(c, n, sp)
        else:
            b.load(c, n)
        sc = ev.case_schedule(c)
        b.schedule(sc["mode"], sc["beta_start"], sc["beta_end"], sc["delta"], sc["sps"], sc.get("table"))
        b.reseed(0x12345678)
        b.run(sweeps if not sp else max(1, sweeps // 10))
        b.hist(hb)
        b.cmd("U")
        b.read(4 * (n + 1))
        b.ack()
        b.stream(50, n)
        b.spins(n)
        t[be] = round(time.time() - t0, 2)
        if hasattr(tr, "close"):
            tr.close()
        logs[be] = list(log)
    rows = collections.OrderedDict()
    first = None
    full = {}
    for be in logs:
        full[be] = [(e["cmd"], e["nrx"], e["crc"], e["rx"]) for e in logs[be]]
    for k, (a, b) in enumerate(zip(full["model"], full["rtl"])):
        r = rows.setdefault(a[0], dict(cmd=a[0], count=0, bytes=0, equal=0))
        r["count"] += 1
        r["bytes"] += a[1]
        same = a == b
        r["equal"] += same
        if not same and first is None:
            first = dict(index=k, cmd=a[0], model=a[3], rtl=b[3])
    ok = first is None and len(full["model"]) == len(full["rtl"])
    return dict(chip=chip_id, equal=ok, commands=len(full["model"]), rows=list(rows.values()), first_mismatch=first,
                seconds=t, note="every reply byte is compared (length + CRC-32 per command)")


ML_LETTERS = ["H", "C", "J"]
MRF_SIDES = [16, 32, 64, 128]
ML_DEMOS = {
    "glyphs": dict(title="Glyph classifier", train=True, sparse=False, n=13, chips=["dense16", "factor6", "pysim"],
                   needs="a dense chip with >= 13 p-bits (Dense N = 16, the multiplier chip) or the ideal simulator",
                   summary="Learns the joint law of 3x3 glyphs T, L, X, O and a one-hot label from clean and "
                           "1-pixel-noise examples. Classify: pin the image, read the label p-bits. Generate: pin a label."),
    "bars3": dict(title="Bars & stripes (hidden units)", train=True, sparse=False, n=15, chips=["dense16", "pysim"],
                  needs="the Dense N = 16 chip or the ideal simulator",
                  summary="Boltzmann machine with 9 visible + 6 hidden p-bits trained on the 14 3x3 bars-and-stripes "
                          "images. Generate new images; complete occluded ones."),
    "hopfield": dict(title="Hopfield associative memory", train=False, sparse=False, n=16, chips=["dense16", "pysim"],
                     needs="the Dense N = 16 chip or the ideal simulator",
                     summary="One-shot Hebbian couplings store 3 letters on 4x4 p-bits. A noisy letter enters as "
                             "soft evidence (bias); sampling recalls the stored letter."),
    "mrf": dict(title="Image denoising (MRF)", train=False, sparse=True, n=256, chips=["sparse256", "pysim"],
                needs="the Sparse N = 256 chip or the ideal simulator",
                summary="Ising prior on the pixel grid + evidence from a noisy image: the sampler draws from "
                        "P(image | noisy), and the per-pixel majority is the denoised image. 16x16 on the sparse "
                        "chip; up to 128x128 (16 384 p-bits) on the ideal simulator."),
    "digits": dict(title="Handwritten digits (RBM, 174 p-bits)", train=True, sparse=False, n=174, chips=["pysim"],
                   needs="the ideal simulator (174 densely coupled p-bits do not fit a chip)",
                   summary="Restricted Boltzmann machine over 8x8 digits (scikit-learn, 1440 training images) "
                           "+ a one-hot label: 74 visible + 100 hidden p-bits. Classify, generate a digit from "
                           "its label, complete a half-hidden digit."),
    "pmode": dict(title="Learn a p-mode circuit from its samples", train=True, sparse=False, n=4,
                  chips=["pmode_chain4"], needs="the continuous p-mode chip",
                  summary="A hidden teacher chip emits soft spins; a student chip with A = h = 0 matches the "
                          "moments <tanh x_i tanh x_j> by Langevin sampling and recovers the teacher's "
                          "couplings and fields (to 1/16)."),
}
for _k, _d in ML_DEMOS.items():
    _d["home"] = _d["chips"][0]


# ------------------------------------------------------------------ HTTP
S = Session()


def list_programs():
    ex = []
    for f in sorted(os.listdir(EXAMPLES)):
        if f.endswith(".json"):
            spec = json.load(open(os.path.join(EXAMPLES, f)))
            ex.append(dict(name=f[:-5], type=spec.get("type"), comment=spec.get("comment", ""), spec=spec))
    allc = builtin_cases()
    return dict(examples=ex, cases=[dict(name=k, n=v["n"]) for k, v in sorted(allc.items())])


def list_ports():
    try:
        from serial.tools import list_ports
        return [dict(device=p.device, description=p.description) for p in list_ports.comports()]
    except Exception as e:      # pyserial missing
        return [dict(device="", description=f"pyserial unavailable: {e}")]


ROUTES_GET = {
    "/api/chips": lambda q: [chips.public(c) for c in chips.CHIPS],
    "/api/ml/list": lambda q: [dict(id=k, **{x: v[x] for x in ("title", "train", "sparse", "n", "needs", "summary",
                                                                "chips", "home")})
                               for k, v in ML_DEMOS.items()],
    "/api/programs": lambda q: list_programs(),
    "/api/ports": lambda q: list_ports(),
    "/api/wire": lambda q: list(S.wire)[-int(q.get("n", 200)):],
    "/api/status": lambda q: dict(status=S.status, chip=S.chip and S.chip["id"], backend=S.backend,
                                  program=S.case and program_summary(S.case, S.rev),
                                  pmode=S.pm and S.pm_info()),
}
ROUTES_POST = {
    "/api/connect": lambda r: S.connect(r["chip"], r.get("backend", "model"), r.get("port")),
    "/api/program": lambda r: S.program(r),
    "/api/restart": lambda r: S.restart(r),
    "/api/pin": lambda r: S.pin(r),
    "/api/pin_group": lambda r: S.pin_group(r),
    "/api/bias": lambda r: S.bias(r),
    "/api/coupling": lambda r: S.coupling(r),
    "/api/reset_program": lambda r: S.reset_program(r),
    "/api/ml/start": lambda r: S.ml_start(r),
    "/api/ml/train": lambda r: S.ml_train(r),
    "/api/ml/use": lambda r: S.ml_use(r),
    "/api/ml/infer": lambda r: S.ml_infer(r),
    "/api/ml/hopfield": lambda r: S.ml_hopfield(r),
    "/api/ml/denoise": lambda r: S.ml_denoise(r),
    "/api/ml/digits": lambda r: S.ml_digits(r),
    "/api/stream": lambda r: S.stream(r),
    "/api/sample": lambda r: S.sample(r),
    "/api/solve": lambda r: S.solve(r),
    "/api/pmode/step": lambda r: S.pm_step(r),
    "/api/pmode/reset": lambda r: (S.pm_reset(int(r.get("seed", fx.LFSR_DEFAULT_SEED))), S.pm_info())[1],
    "/api/pmode/info": lambda r: S.pm_info(),
    "/api/compare": lambda r: compare(r.get("chip") or S.chip["id"], int(r.get("sweeps", 2000))),
    "/api/build_rtl": lambda r: dict(binary=build_rtl.ensure(r["chip"])),
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else "") and "/api/wire" not in args[0] and "/api/stream" not in args[0]:
            sys.stderr.write("%s\n" % (fmt % args))

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def handle_api(self, fn, arg):
        try:
            with S.lock:
                self.send_json(fn(arg))
        except Exception as e:
            traceback.print_exc()
            self.send_json(dict(error=f"{type(e).__name__}: {e}"), 400)

    def do_GET(self):
        path, _, qs = self.path.partition("?")
        q = dict(kv.split("=", 1) for kv in qs.split("&") if "=" in kv)
        if path in ROUTES_GET:
            return self.handle_api(ROUTES_GET[path], q)
        if path in ("/", "/index.html"):
            path = "/static/index.html"
        parts = path.strip("/").split("/", 1)
        base = SERVE_DIRS.get(parts[0])
        if base and len(parts) == 2:
            f = os.path.realpath(os.path.join(base, parts[1]))
            if f.startswith(os.path.realpath(base) + os.sep) and os.path.isfile(f):
                data = open(f, "rb").read()
                self.send_response(200)
                self.send_header("Content-Type", MIME.get(os.path.splitext(f)[1], "application/octet-stream"))
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
        self.send_json(dict(error="not found"), 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError as e:
            return self.send_json(dict(error=f"bad JSON: {e}"), 400)
        if self.path in ROUTES_POST:
            return self.handle_api(ROUTES_POST[self.path], req)
        self.send_json(dict(error="not found"), 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"p-bit chip viewer: http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        S.close()


if __name__ == "__main__":
    main()
