#!/usr/bin/env python3
"""Draw the basic-unit diagrams of docs/04 and docs/05 as SVG files.

  python3 tools/make_diagrams.py            -> docs/img/units/*.svg
  python3 tools/make_diagrams.py --fragments DIR
                                            -> same drawings without the <style>
                                               block, for docs/explainer.html

Numbers in the drawings come from the code, not from hand-typing:
p-gate J / h and truth-table probabilities from model/pgates.py, measured
factoring probabilities from tb/build/pgates.json (make pgates), cycle counts
from the RTL headers (docs/06), FPGA utilisation from docs/07.

Colour code used in every drawing:
  warm (orange) = the physical / thermodynamic side (what the papers describe)
  cool (teal)   = the digital FPGA emulation (this repo's Verilog)
  green         = valid truth-table rows, grey = invalid rows
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "model"))
import pgates as pg      # noqa: E402

# Palettes for the standalone SVGs. They are written as literal colours (no CSS
# custom properties) because many viewers (librsvg, Inkscape, PDF converters)
# do not support var() and would paint everything black.
LIGHT = dict(bg="#faf9f6", ink="#1d2329", mut="#66707a", line="#8f99a3", warm="#c4501d",
             warmf="#fbe5d8", cool="#16708a", coolf="#dbeef3", ok="#2c8a4b", okf="#ddf0e3",
             bad="#b9b5ac", badf="#efede8", panel="#f1efea")
DARK = dict(bg="#14181c", ink="#e5e8eb", mut="#9aa4ae", line="#64707b", warm="#f08a5c",
            warmf="#3b2418", cool="#5cc1d9", coolf="#12303a", ok="#6dca8b", okf="#163322",
            bad="#6d6a64", badf="#24272a", panel="#1b2025")

# class -> presentation (works with the page CSS of explainer.html too)
CSS_CLASSES = """
.bg{fill:var(--bg)} .panel{fill:var(--panel)}
.t{fill:var(--ink);font:13px 'IBM Plex Sans',system-ui,sans-serif}
.ts{fill:var(--mut);font:11px 'IBM Plex Sans',system-ui,sans-serif}
.tm{fill:var(--ink);font:11.5px 'IBM Plex Mono',ui-monospace,monospace}
.tms{fill:var(--mut);font:10.5px 'IBM Plex Mono',ui-monospace,monospace}
.th{fill:var(--ink);font:600 15px 'IBM Plex Sans',system-ui,sans-serif}
.tb{fill:var(--ink);font:600 13px 'IBM Plex Sans',system-ui,sans-serif}
.tw{fill:var(--warm)} .tc{fill:var(--cool)} .tok{fill:var(--ok)}
.warm{fill:var(--warmf);stroke:var(--warm);stroke-width:1.4}
.cool{fill:var(--coolf);stroke:var(--cool);stroke-width:1.4}
.ok{fill:var(--okf);stroke:var(--ok);stroke-width:1.4}
.bad{fill:var(--badf);stroke:var(--bad);stroke-width:1.2}
.plain{fill:var(--bg);stroke:var(--line);stroke-width:1.2}
.ln{stroke:var(--line);stroke-width:1.4;fill:none}
.lw{stroke:var(--warm);stroke-width:1.6;fill:none}
.lc{stroke:var(--cool);stroke-width:1.6;fill:none}
.lk{stroke:var(--ink);stroke-width:1.6;fill:none}
.dash{stroke-dasharray:4 4}
.fw{fill:var(--warm)} .fc{fill:var(--cool)} .fk{fill:var(--ink)} .fl{fill:var(--line)}
"""


def resolve(css, pal):
    """Replace every var(--name) in css with the literal colour pal[name]."""
    for k in sorted(pal, key=len, reverse=True):
        css = css.replace(f"var(--{k})", pal[k])
    return css


def standalone_style():
    return ("<style>" + resolve(CSS_CLASSES, LIGHT)
            + "@media (prefers-color-scheme:dark){" + resolve(CSS_CLASSES, DARK) + "}\n</style>")


class Svg:
    def __init__(self, w, h, label):
        self.w, self.h, self.label, self.parts = w, h, label, []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, cls="t", anchor="start", extra=""):
        s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}" {extra}>{s}</text>')

    def rect(self, x, y, w, h, cls="plain", r=6):
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{r}" class="{cls}"/>')

    def box(self, x, y, w, h, title, sub=None, cls="plain", tcls="tb", r=6):
        self.rect(x, y, w, h, cls, r)
        cy = y + h / 2 + (4 if sub is None else -3)
        self.text(x + w / 2, cy, title, tcls, "middle")
        if sub:
            for k, line in enumerate(sub.split("\n")):
                self.text(x + w / 2, cy + 15 + 13 * k, line, "ts", "middle")

    def circle(self, x, y, r, cls="plain"):
        self.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" class="{cls}"/>')

    def line(self, x1, y1, x2, y2, cls="ln", head=True, label=None, lcls="ts", loff=(0, -5), anchor="middle"):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="{cls}"/>')
        if head:
            self.head(x1, y1, x2, y2, cls)
        if label:
            self.text((x1 + x2) / 2 + loff[0], (y1 + y2) / 2 + loff[1], label, lcls, anchor)

    def poly(self, pts, cls="ln", head=True, label=None, lpos=None, lcls="ts", anchor="middle"):
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.add(f'<polyline points="{d}" class="{cls}"/>')
        if head:
            (x1, y1), (x2, y2) = pts[-2], pts[-1]
            self.head(x1, y1, x2, y2, cls)
        if label and lpos:
            self.text(lpos[0], lpos[1], label, lcls, anchor)

    def head(self, x1, y1, x2, y2, cls):
        a = math.atan2(y2 - y1, x2 - x1)
        L, W = 8, 4
        bx, by = x2 - L * math.cos(a), y2 - L * math.sin(a)
        p = [(x2, y2), (bx - W * math.sin(a), by + W * math.cos(a)), (bx + W * math.sin(a), by - W * math.cos(a))]
        fill = {"lw": "fw", "lc": "fc", "lk": "fk"}.get(cls.split()[0], "fl")
        self.add('<polygon points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in p) + f'" class="{fill}"/>')

    def path(self, d, cls="ln"):
        self.add(f'<path d="{d}" class="{cls}"/>')

    def render(self, style=True):
        size = f'width="{self.w}" height="{self.h}" ' if style else ""
        head = (f'<svg xmlns="http://www.w3.org/2000/svg" {size}viewBox="0 0 {self.w} {self.h}" role="img" '
                f'aria-label="{self.label}" font-family="IBM Plex Sans, system-ui, sans-serif">')
        st = standalone_style() if style else ""
        bg = f'<rect width="{self.w}" height="{self.h}" class="bg"/>'
        return "\n".join([head, st, bg] + self.parts + ["</svg>"])


def curve(s, x0, y0, w, h, fn, cls, n=60, xr=(-4, 4)):
    pts = []
    for k in range(n + 1):
        u = xr[0] + (xr[1] - xr[0]) * k / n
        pts.append((x0 + w * k / n, y0 + h - h * fn(u)))
    s.poly(pts, cls, head=False)


def sig(u):
    return 1 / (1 + math.exp(-u))


# ------------------------------------------------------------ 01 mapping
def d_mapping():
    rows = [
        ("Transistor", "switch: on / off", "Noise source", "shot noise / LFSR bits", "lfsr.v  noise_bank.v"),
        ("Logic gate", "AND / OR / NOT", "p-gate", "J, h whose ground states = truth table",
         "pgates.v"),
        ("Comparator", "V > Vref", "p-bit", "P(1) = σ(2β·field)", "pbit.v"),
        ("Adder / ALU", "sums numbers", "Synapse MAC", "field = h + Σ J·x", "weight_mac.v"),
        ("Flip-flop", "holds a bit", "Latched spin", "latches on clock edge", "pbit_array.v (s[i])"),
        ("Clock + PC", "next instruction", "Gibbs scheduler", "colour block 0, 1, 0, 1 …", "pbit_array.v (FSM)"),
        ("Clock / voltage", "speed knob", "β annealing", "temperature knob", "schedule_ctrl.v"),
        ("RAM / ROM", "program + data", "J / h memory", "the program is energy", "BSRAM in pbit_array.v"),
        ("Bus / wires", "connect units", "Coupling graph", "who feels whom (J ≠ 0)", "weight RAM {i, j}"),
        ("Output port", "reads one answer", "Readout", "counts samples → histogram", "readout_counter.v"),
    ]
    W, top, rh = 980, 150, 52
    s = Svg(W, top + rh * len(rows) + 30,
            "Classical computer blocks next to their thermodynamic counterparts and the Verilog file of each")
    s.text(24, 34, "Classical computer", "th")
    s.text(24, 54, "output = f(input): the same answer every time", "ts")
    s.text(420, 34, "Thermodynamic / probabilistic computer", "th tw")
    s.text(420, 54, "P(state) ∝ exp(−β·E(state)): answers are samples", "ts")
    s.text(800, 34, "Our Verilog", "th tc")
    s.text(800, 54, "rtl/ on the Tang Nano 9K", "ts")
    # step vs sigmoid mini plots
    x0, y0 = 24, 72
    s.rect(x0, y0, 150, 54, "panel", 4)
    s.poly([(x0 + 10, y0 + 46), (x0 + 75, y0 + 46), (x0 + 75, y0 + 8), (x0 + 140, y0 + 8)], "lk", head=False)
    s.text(x0 + 160, y0 + 24, "step: input decides", "ts")
    s.text(x0 + 160, y0 + 38, "the output exactly", "ts")
    x1 = 420
    s.rect(x1, y0, 150, 54, "panel", 4)
    for b, cls in ((0.6, "lw dash"), (1.5, "lw")):
        curve(s, x1 + 10, y0 + 8, 130, 38, lambda u, b=b: sig(2 * b * u), cls, xr=(-3, 3))
    s.text(x1 + 160, y0 + 24, "sigmoid: input tilts the", "ts")
    s.text(x1 + 160, y0 + 38, "odds (dashed = hotter, low β)", "ts")
    for k, (c1, c1s, c2, c2s, f) in enumerate(rows):
        y = top + k * rh
        s.box(24, y, 230, 40, c1, None, "plain")
        s.text(262, y + 25, c1s, "ts")
        s.line(400, y + 20, 416, y + 20, "ln")
        s.rect(420, y, 360, 40, "warm")
        s.text(434, y + 25, c2, "tb")
        s.text(560, y + 25, c2s, "ts")
        s.rect(800, y, 160, 40, "cool")
        s.text(810, y + 25, f, "tms")
    s.text(24, top + rh * len(rows) + 14,
           "Every right-hand unit is built from the p-bit: noise + bias + comparator + latch (see 02_pbit_anatomy).",
           "ts")
    return s


# ------------------------------------------------------------ 02 p-bit anatomy
def d_pbit():
    W, H = 1000, 520
    s = Svg(W, H, "Anatomy of one p-bit: the physical sampling cell above, the FPGA emulation below, "
                  "with matching stages aligned")
    s.text(24, 32, "One p-bit, two implementations", "th")
    s.text(24, 52, "Same five stages. Top: the analog cell the DTM paper describes (Fig. 10, App. K). "
                   "Bottom: rtl/pbit_array.v + rtl/pbit.v.", "ts")
    # stage columns
    cols = [(24, "1 inputs"), (196, "2 weigh + sum"), (390, "3 to probability"), (590, "4 random decision"),
            (790, "5 hold + broadcast")]
    for x, lab in cols:
        s.text(x, 84, lab.upper(), "ts", extra='letter-spacing="1"')
    # physical lane
    yP = 104
    s.rect(14, yP - 8, 972, 170, "panel", 10)
    s.text(24, yP + 10, "PHYSICAL (thermodynamic chip)", "ts tw", extra='letter-spacing="1"')
    for k, lab in enumerate(("x1", "x2", "xn")):
        yy = yP + 34 + k * 38
        s.circle(46, yy, 13, "warm")
        s.text(46, yy + 4, lab, "tm", "middle")
        s.circle(98, yy, 11, "plain")
        s.text(98, yy + 4, "⊕", "t", "middle")
        s.line(59, yy, 86, yy, "lw")
        s.text(98, yy - 15, f"s{lab[1]}", "tms", "middle")
        # resistor zigzag
        zx = 120
        pts = [(zx, yy)] + [(zx + 8 + 8 * i, yy + (-5 if i % 2 == 0 else 5)) for i in range(6)] + [(zx + 64, yy)]
        s.poly(pts, "lw", head=False)
        s.text(zx + 32, yy - 10, f"G{lab[1]}", "tms", "middle")
        s.line(zx + 64, yy, 250, yP + 72, "lw", head=False)
    s.circle(256, yP + 72, 7, "warm")
    s.text(256, yP + 98, "V_b node", "ts", "middle")
    s.line(256, yP + 79, 256, yP + 112, "ln", head=False)
    s.line(244, yP + 112, 268, yP + 112, "lk", head=False)
    s.line(244, yP + 118, 268, yP + 118, "lk", head=False)
    s.text(276, yP + 122, "C", "tms")
    s.line(263, yP + 72, 400, yP + 72, "lw", label="bias voltage", loff=(0, -6))
    s.box(404, yP + 40, 150, 64, "noise shifted", "Gaussian shot noise\n+ bias voltage", "warm")
    s.line(554, yP + 72, 600, yP + 72, "lw")
    s.box(604, yP + 40, 150, 64, "comparator", "noise > threshold ?", "warm")
    s.line(754, yP + 72, 800, yP + 72, "lw", label="0/1", loff=(0, -6))
    s.box(804, yP + 40, 150, 64, "latch", "freezes on clock", "warm")
    s.line(879, yP + 104, 879, yP + 140, "lw", head=False)
    s.poly([(879, yP + 140), (46, yP + 140), (46, yP + 125)], "lw dash", label="x_i to all neighbours",
           lpos=(470, yP + 136))
    s.text(879, yP + 30, "clock", "tms", "middle")
    s.line(879, yP + 18 + 16, 879, yP + 40, "ln")
    # FPGA lane
    yF = 300
    s.rect(14, yF - 8, 972, 170, "panel", 10)
    s.text(24, yF + 10, "FPGA EMULATION (this repo)", "ts tc", extra='letter-spacing="1"')
    s.box(24, yF + 30, 150, 84, "spins s[ ] + J row", "J RAM row i (BSRAM)\nh RAM entry h_i", "cool")
    s.line(174, yF + 72, 196, yF + 72, "lc")
    s.box(200, yF + 30, 170, 84, "weight MAC", "acc += s_j ? +J : −J\nN cycles, no multiplier\nfield ℓ (AW bits)",
          "cool")
    s.line(370, yF + 72, 400, yF + 72, "lc", label="ℓ", loff=(0, -6))
    s.box(404, yF + 30, 150, 84, "× β, σ LUT", "shift-add 8 cycles\n1024×16 ROM\np = 65536·σ(2βℓ)", "cool")
    s.line(554, yF + 72, 600, yF + 72, "lc", label="p16", loff=(0, -6))
    s.box(604, yF + 30, 150, 84, "rand16 < p16", "LFSR32 draws 16 bits\nfresh every update", "cool")
    s.line(754, yF + 72, 800, yF + 72, "lc", label="bit", loff=(0, -6))
    s.box(804, yF + 30, 150, 84, "spin register", "s[i] ← bit\nread by next MAC", "cool")
    s.poly([(879, yF + 114), (879, yF + 140), (99, yF + 140), (99, yF + 114)], "lc dash",
           label="next update reads the new s[i]", lpos=(470, yF + 136))
    # correspondence
    for x in (99, 285, 479, 679, 879):
        s.line(x, yP + 160, x, yF - 6, "ln dash", head=False)
    # sigmoid inset
    yS = 482
    s.text(24, yS, "Stage 3 + 4 together: P(x_i = 1) = σ(2β(Σ_j J_ij x_j + h_i))  "
                   "[DTM p.5 Eq. 11]. The comparator on noise is what makes the bit random; the bias only tilts "
                   "the odds.", "ts")
    s.text(24, yS + 18, "Physical: one cell per node, all cells of a colour flip at once (~100 ns). "
                        "FPGA: one engine visits nodes in turn, N + 18 cycles per update.", "ts")
    return s


# ------------------------------------------------------------ 03 p-gates
def gate_panel(s, x, y, kind, beta, pos, title, note):
    g = pg.GADGETS[kind]
    s.rect(x, y, 300, 360, "panel", 10)
    s.text(x + 16, y + 26, title, "th")
    s.text(x + 16, y + 44, note, "ts")
    terms = g["terms"]
    P = {t: (x + px, y + py) for t, (px, py) in pos.items()}
    for (a, b), w in g["J"].items():
        (x1, y1), (x2, y2) = P[a], P[b]
        cls = "lw" if w > 0 else "lc"
        s.line(x1, y1, x2, y2, cls + ("" if w > 0 else " dash"), head=False)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        s.rect(mx - 15, my - 9, 30, 17, "plain", 8)
        s.text(mx, my + 4, f"{w:+d}", "tms", "middle")
    for t in terms:
        px, py = P[t]
        aux = t == "X"
        s.circle(px, py, 17, "plain" if aux else ("warm" if t == "Y" else "cool"))
        s.text(px, py + 4, t, "tb", "middle")
        hv = g["h"].get(t, 0)
        left = px - x < 150
        s.text(px + (-24 if left else 24), py + 4, f"h={hv:+d}" if hv else "h=0", "tms",
               "end" if left else "start")
    # truth table with exact probabilities
    nin = pg.N_IN[kind]
    c = pg.gate(kind)
    wires = list(terms[:nin + 1])
    m = pg.marginal(c, wires, beta)
    valid = pg.truth_rows(kind)
    ty = y + 206
    hdr = " ".join(wires[:nin]) + " → " + wires[nin]
    s.text(x + 16, ty, hdr, "tms")
    s.text(x + 284, ty, f"P at β={beta:g}", "tms", "end")
    rows = sorted(m, key=lambda r: (r not in valid, r))
    rows = [r for r in rows if r in valid] + sorted([r for r in rows if r not in valid], key=lambda r: -m[r])[:2]
    for k, r in enumerate(rows):
        yy = ty + 12 + k * 19
        ok = r in valid
        s.rect(x + 12, yy, 276, 16, "ok" if ok else "bad", 3)
        s.text(x + 22, yy + 12, " ".join(map(str, r[:nin])) + "  →  " + str(r[nin]), "tm")
        bw = 110 * m[r] / max(m.values())
        s.rect(x + 120, yy + 3, max(1, bw), 10, "ok" if ok else "bad", 2)
        s.text(x + 284, yy + 12, f"{m[r]:.3f}", "tm", "end")
    pv = sum(p for r, p in m.items() if r in valid)
    s.text(x + 16, y + 344, f"P(valid row) = {pv:.3f}", "ts")
    if len(m) > len(valid) + 1:
        s.text(x + 284, y + 344, "grey: 2 likeliest invalid rows", "ts", "end")


def d_pgates():
    W, H = 1260, 500
    s = Svg(W, H, "p-gates NOT, AND, OR and XOR as small Ising graphs with their biases, couplings and exact "
                  "truth-table probabilities")
    s.text(24, 32, "p-gates: Boolean gates as energy landscapes", "th")
    s.text(24, 52, "Circles are p-bits (blue = input, orange = output, grey = auxiliary). Edge = coupling J "
                   "(solid +: wants equal, dashed −: wants opposite). The truth-table rows are the lowest-energy "
                   "states, so a free-running sampler mostly visits them.", "ts")
    y = 76
    gate_panel(s, 24, y, "NOT", 1.0, {"A": (70, 120), "Y": (230, 120)}, "p-NOT", "1 coupling, J = −1")
    gate_panel(s, 334, y, "AND", 1.0, {"A": (60, 90), "B": (60, 160), "Y": (230, 125)}, "p-AND",
               "3 p-bits, gap 4 between valid and invalid")
    gate_panel(s, 644, y, "OR", 1.0, {"A": (60, 90), "B": (60, 160), "Y": (230, 125)}, "p-OR",
               "AND with every bias negated")
    gate_panel(s, 954, y, "XOR", 1.5, {"A": (60, 85), "B": (60, 165), "Y": (240, 85), "X": (240, 165)},
               "p-XOR", "needs 1 auxiliary p-bit X")
    s.text(24, 464, "E(s) = −Σ J_ij s_i s_j − Σ h_i s_i, logic 1 = spin +1. Probabilities: exact Boltzmann law "
                    "from model/pgates.py; the RTL reproduces them (docs/05).", "ts")
    s.text(24, 482, "Coefficients: standard Ising gadgets [NOT IN SOURCES]. The papers only say gates can be built "
                    "by composing energies [BP p.3].", "ts")
    return s


# ------------------------------------------------------------ 04 composition
def d_compose(pgj):
    W, H = 1180, 600
    s = Svg(W, H, "How p-gates compose: shared wires merge into one p-bit and their energies add; the 2 by 2 "
                  "multiplier built this way runs backwards to factor 6")
    s.text(24, 32, "Composition: wire = shared p-bit, circuit energy = sum of gate energies", "th")
    # (a) merging rule
    s.rect(14, 50, 400, 250, "panel", 10)
    s.text(28, 74, "a) Two gates share wire t", "tb")
    s.circle(70, 120, 15, "cool")
    s.text(70, 124, "a", "tb", "middle")
    s.circle(70, 180, 15, "cool")
    s.text(70, 184, "b", "tb", "middle")
    s.circle(190, 150, 17, "warm")
    s.text(190, 154, "t", "tb", "middle")
    s.circle(300, 120, 15, "cool")
    s.text(300, 124, "c", "tb", "middle")
    s.circle(300, 200, 15, "warm")
    s.text(300, 204, "y", "tb", "middle")
    for (x1, y1, x2, y2) in ((85, 122, 173, 147), (85, 178, 173, 153), (82, 132, 82, 168)):
        s.line(x1, y1, x2, y2, "lc", head=False)
    for (x1, y1, x2, y2) in ((207, 147, 285, 122), (204, 160, 287, 192), (300, 135, 300, 185)):
        s.line(x1, y1, x2, y2, "lw", head=False)
    s.text(110, 222, "gate 1 (teal)", "ts tc", "middle")
    s.text(260, 236, "gate 2 (orange)", "ts tw", "middle")
    s.text(28, 262, "h_t = h_t(gate 1) + h_t(gate 2)", "tm")
    s.text(28, 280, "J adds the same way. Nothing else is needed.", "ts")
    # (b) direction
    s.rect(14, 316, 400, 270, "panel", 10)
    s.text(28, 340, "b) Clamping chooses the direction", "tb")
    s.text(28, 360, "clamp = extreme bias ±8 on a p-bit (h RAM write)", "ts")
    s.box(40, 380, 110, 44, "inputs", "a, b", "cool")
    s.box(270, 380, 110, 44, "outputs", "p = a·b", "warm")
    s.line(150, 394, 270, 394, "lc", label="forward: clamp a, b", loff=(0, -7))
    s.line(270, 414, 150, 414, "lw", label="backward: clamp p", loff=(0, 18))
    s.text(28, 470, "Forward = an ordinary (noisy) circuit.", "ts")
    s.text(28, 488, "Backward = the sampler searches for inputs that", "ts")
    s.text(28, 504, "produce the clamped output: factoring, SAT,", "ts")
    s.text(28, 520, "inversion. A deterministic circuit cannot do this.", "ts")
    s.text(28, 548, "Same hardware, same J; only 4 bias words change.", "tb")
    # (c) multiplier
    X0, Y0 = 430, 50
    s.rect(X0, Y0, 736, 536, "panel", 10)
    s.text(X0 + 16, Y0 + 24, "c) 2 × 2-bit multiplier = 6 AND + 2 XOR = 14 p-bits (model/pgates.mult2x2)", "tb")
    ins = [("a0", 90), ("a1", 150), ("b0", 230), ("b1", 290)]
    P = {}
    for nm, yy in ins:
        P[nm] = (X0 + 50, Y0 + yy)
    ands = [("p0", "a0·b0", 70), ("q01", "a0·b1", 150), ("q10", "a1·b0", 230), ("q11", "a1·b1", 320)]
    for nm, lab, yy in ands:
        P[nm] = (X0 + 230, Y0 + yy)
    P["p1"] = (X0 + 420, Y0 + 170)
    P["c1"] = (X0 + 420, Y0 + 250)
    P["p2"] = (X0 + 580, Y0 + 270)
    P["p3"] = (X0 + 580, Y0 + 350)
    wires = [("a0", "p0"), ("b0", "p0"), ("a0", "q01"), ("b1", "q01"), ("a1", "q10"), ("b0", "q10"),
             ("a1", "q11"), ("b1", "q11"), ("q01", "p1"), ("q10", "p1"), ("q01", "c1"), ("q10", "c1"),
             ("q11", "p2"), ("c1", "p2"), ("q11", "p3"), ("c1", "p3")]
    for a, b in wires:
        (x1, y1), (x2, y2) = P[a], P[b]
        s.line(x1 + 18, y1, x2 - 18, y2, "ln", head=False)
    gate_of = {"p0": "AND", "q01": "AND", "q10": "AND", "q11": "AND", "p1": "XOR", "c1": "AND", "p2": "XOR",
               "p3": "AND"}
    clamp6 = {"p0": 0, "p1": 1, "p2": 1, "p3": 0}
    for nm, (x, y) in P.items():
        is_out = nm in clamp6
        s.circle(x, y, 17, "warm" if is_out else ("cool" if nm[0] in "ab" else "plain"))
        s.text(x, y + 4, nm, "tm", "middle")
        if nm in gate_of:
            s.text(x, y - 23, gate_of[nm] + (" (+aux)" if gate_of[nm] == "XOR" else ""), "tms", "middle")
        if is_out:
            s.rect(x + 22, y - 10, 46, 20, "warm", 10)
            s.text(x + 45, y + 4, f"= {clamp6[nm]}", "tm", "middle")
    s.text(X0 + 50, Y0 + 330, "inputs", "ts", "middle")
    s.text(X0 + 670, Y0 + 110, "product clamped to 6 = 0110", "ts tw", "middle")
    s.text(X0 + 670, Y0 + 126, "(p3 p2 p1 p0)", "tms", "middle")
    # measured result
    yr = Y0 + 400
    sc = None
    if pgj:
        f6 = next((r for r in pgj if r["case"] == "factor6"), None)
        if f6:
            sc = {x["label"]: x for x in f6["scenarios"]}
    s.text(X0 + 16, yr, "Measured on the RTL (tb_pgates.v, β = 2, 1800 sweeps after burn-in per product):", "ts")
    if sc:
        for k, lab in enumerate(("factor 6", "factor 3", "factor 9", "factor 4")):
            r = sc[lab]
            grid = {}
            for key, v in r["emp"].items():
                a = int(key[0]) + 2 * int(key[1])
                b = int(key[2]) + 2 * int(key[3])
                grid[(a, b)] = grid.get((a, b), 0) + v
            best = sorted(grid.items(), key=lambda kv: -kv[1])[:2]
            xx = X0 + 16 + k * 180
            s.rect(xx, yr + 12, 168, 96, "plain", 8)
            s.text(xx + 12, yr + 34, f"clamp p = {lab.split()[1]}", "tb")
            for j, ((a, b), v) in enumerate(best):
                okk = a * b == int(lab.split()[1])
                s.text(xx + 12, yr + 56 + 18 * j, f"a={a} b={b}", "tm")
                s.text(xx + 156, yr + 56 + 18 * j, f"{v:.2f}", "tm tok" if okk else "tm", "end")
            s.text(xx + 12, yr + 98, f"P(a·b = p) = {r['p_ok']:.2f}", "ts")
    else:
        s.text(X0 + 16, yr + 30, "(run `make pgates` in tb/ to fill in the measured numbers)", "ts")
    return s


# ------------------------------------------------------------ 05 hierarchy
def d_stack():
    W, H = 1000, 580
    s = Svg(W, H, "Layers of the thermodynamic computer from noise up to the host program, with the classical "
                  "analogue and the Verilog file for each layer")
    s.text(24, 32, "How the units stack into a computer", "th")
    s.text(24, 52, "Read bottom-up: each layer is made only of the layer below it.", "ts")
    layers = [
        ("Host program", "anneal · sample · denoise · factor", "software", "fpga/host.py over UART"),
        ("Instruction set", "W H O T B R S C D U Q X P", "ISA / bus", "top.v command FSM"),
        ("Programs as energies", "Max-Cut · p-gate circuits · denoising chain", "compiled code",
         "model/cases.py → hex"),
        ("Sweep + schedule", "colour blocks in turn; β rises over time", "clock + PC", "pbit_array.v · schedule_ctrl.v"),
        ("Ising array", "N p-bits + coupling graph J, biases h", "datapath + RAM", "pbit_array.v · coupling_row.v"),
        ("p-bit", "field → σ → compare with noise → latch", "gate + flip-flop", "pbit.v · weight_mac.v · sigmoid_lut.v"),
        ("Noise", "fresh random bits every update", "transistor", "lfsr.v · noise_bank.v"),
    ]
    y0, lh = 96, 64
    for k, (name, what, cl, f) in enumerate(layers):
        y = y0 + k * lh
        cls = "warm" if k >= 5 else ("cool" if k >= 3 else "plain")
        s.rect(150, y, 560, lh - 16, cls, 8)
        s.text(166, y + 20, name, "tb")
        s.text(166, y + 37, what, "ts")
        s.text(24, y + 28, cl, "ts")
        s.text(976, y + 28, f, "tms", "end")
        if k < len(layers) - 1:
            s.line(430, y + lh - 2, 430, y + lh - 14, "ln", label="makes up", loff=(8, 4), anchor="start")
    s.text(24, y0 - 8, "CLASSICAL ANALOGUE", "ts", extra='letter-spacing="1"')
    s.text(976, y0 - 8, "WHERE IT LIVES", "ts", "end", extra='letter-spacing="1"')
    s.text(24, H - 22, "Orange = units with a physical counterpart in the papers; teal = control the papers "
                       "describe at system level; neutral = our host interface.", "ts")
    return s


# ------------------------------------------------------------ 06 timeline
def d_timeline():
    N = 16
    W, H = 1060, 470
    s = Svg(W, H, "Timing of one p-bit update, N plus 18 clock cycles, and of a chromatic Gibbs sweep over a "
                  "4 by 4 grid")
    s.text(24, 32, "Time: one update = N + 18 cycles, one sweep = N updates", "th")
    s.text(24, 52, f"Phases of the update FSM in rtl/pbit_array.v, drawn to scale for N = {N} "
                   f"({N + 18} cycles = 1.26 µs at 27 MHz).", "ts")
    phases = [("ORD", 1, "read order[k]"), ("BIAS", 1, "read h_i"), ("MAC", 1, "start"),
              ("MACW", N + 2, "Σ ±J_ij over j"), ("PBW", 12, "×β (8) · round · σ · compare"),
              ("WR", 1, "s[i] ← bit")]
    x, y, sc = 24, 80, (W - 48) / (N + 18)
    for nm, cyc, what in phases:
        w = cyc * sc
        cls = "cool" if nm in ("MACW", "PBW") else "plain"
        s.rect(x, y, w, 40, cls, 3)
        if w > 50:
            s.text(x + w / 2, y + 18, f"{nm}", "tb", "middle")
            s.text(x + w / 2, y + 33, f"{cyc} cycles", "ts", "middle")
        if w > 50:
            s.text(x + w / 2, y + 58, what, "ts", "middle")
        elif nm == "WR":
            s.text(x + w, y + 58, f"{nm}: {what}", "ts", "end")
        else:
            s.text(x, y + 58 + 14 * (nm == "BIAS") + 28 * (nm == "MAC"), f"{nm}: {what}", "ts")
        x += w
    # clock ticks
    for k in range(N + 19):
        xx = 24 + k * sc
        s.line(xx, 124, xx, 130, "ln", head=False)
    # sweep on 4x4 grid
    y2 = 220
    s.text(24, y2, "One sweep on the 4 × 4 ferromagnet (case grid16): colour 0 nodes, then colour 1", "tb")
    s.text(24, y2 + 18, "Same-colour nodes never touch, so updating them one by one equals updating them all "
                        "at once (what the chip does in ~100 ns).", "ts")
    for ph in range(2):
        gx, gy = 40 + ph * 330, y2 + 40
        s.text(gx, gy + 6, f"phase {ph + 1}: colour {ph} updates", "ts")
        for i in range(16):
            r, c = divmod(i, 4)
            cx, cy = gx + 20 + c * 44, gy + 30 + r * 44
            col = (r + c) % 2
            for dr, dc in ((0, 1), (1, 0)):
                if c + dc < 4 and r + dr < 4:
                    s.line(cx, cy, cx + 44 * dc, cy + 44 * dr, "ln", head=False)
            s.circle(cx, cy, 13, ("warm" if col == ph else "plain"))
            s.text(cx, cy + 4, str(i), "tms", "middle")
    s.text(720, y2 + 70, "Engine time per sweep", "tb")
    rows = [("N = 16", "16 × 34 = 544 cycles", "20.1 µs"), ("N = 64", "64 × 82 = 5248 cycles", "194 µs"),
            ("chip (DTM)", "2 colour phases × τ0", "≈ 0.2 µs")]
    for k, (a, b, c) in enumerate(rows):
        yy = y2 + 96 + 26 * k
        s.text(720, yy, a, "tm")
        s.text(810, yy, b, "tms")
        s.text(1036, yy, c, "tm", "end")
    s.text(720, y2 + 190, "FPGA is ~100× slower per sweep but exact", "ts")
    s.text(720, y2 + 206, "and bit-reproducible (docs/06, docs/08 B.4).", "ts")
    return s


# ------------------------------------------------------------ 07 FPGA board
def d_board():
    W, H = 1100, 560
    s = Svg(W, H, "Tang Nano 9K system: host PC over USB UART into the GW1NR-9 FPGA, the blocks inside it, "
                  "their memory and the pins, with measured utilisation")
    s.text(24, 32, "The thermodynamic computer on the Tang Nano 9K", "th")
    s.text(24, 52, "Default build (N = 16, maxcut16 at power-on). Measured: LUT4 2534/8640 (29 %), "
                   "DFF 1293, BSRAM 10/26, 0 DSP, Fmax 62.67 MHz vs 27 MHz needed.", "ts")
    s.box(24, 110, 140, 70, "Host PC", "fpga/host.py", "plain")
    s.rect(200, 80, 880, 450, "panel", 14)
    s.text(216, 102, "Tang Nano 9K board", "ts", extra='letter-spacing="1"')
    s.box(216, 116, 120, 60, "BL702", "USB ↔ UART", "plain")
    s.line(164, 145, 216, 145, "lk", label="USB-C", loff=(0, -7))
    s.box(216, 420, 120, 50, "27 MHz osc", "pin 52", "plain")
    s.box(216, 300, 120, 80, "buttons", "pin 4 reset\npin 3 run demo", "plain")
    s.rect(360, 112, 700, 400, "cool", 12)
    s.text(376, 134, "GW1NR-9 FPGA (rtl/top.v)", "tb tc")
    s.box(380, 150, 130, 50, "uart_rx", "pin 18, 115200", "plain")
    s.box(380, 214, 130, 50, "uart_tx", "pin 17", "plain")
    s.line(336, 140, 380, 168, "lk", head=True)
    s.line(380, 238, 336, 156, "lk", head=True)
    s.box(540, 150, 160, 114, "command FSM", "W H O T: program\nB Z: schedule\nR S: run sweeps\nD U Q: read out",
          "plain")
    s.line(510, 175, 540, 175, "lk")
    s.line(540, 238, 510, 238, "lk")
    # array
    s.rect(380, 290, 470, 200, "warm", 10)
    s.text(394, 310, "pbit_array: the sampler", "tb tw")
    s.box(394, 322, 130, 54, "J RAM", "N×N × 8 bit (1 BSRAM)", "plain")
    s.box(394, 386, 130, 40, "h + order RAM", None, "plain")
    s.box(540, 322, 150, 104, "p-bit engine", "MAC → ×β → σ LUT\n→ rand < p\n(1 BSRAM LUT)", "plain")
    s.box(706, 322, 130, 54, "spin reg", "s[0..N−1]", "plain")
    s.box(540, 436, 150, 44, "noise_bank", "4 × LFSR32", "plain")
    s.line(524, 349, 540, 349, "lk")
    s.line(524, 406, 540, 406, "lk")
    s.line(615, 436, 615, 426, "lk")
    s.line(690, 349, 706, 349, "lk")
    s.poly([(771, 376), (771, 400), (690, 400)], "lk", head=True)
    s.box(706, 160, 140, 70, "schedule_ctrl", "β: const / linear\n/ table", "plain")
    s.line(776, 230, 776, 290, "lk", label="β", loff=(10, 0))
    s.line(620, 264, 620, 290, "lk", label="write J, h; run", loff=(56, 4))
    s.box(870, 290, 170, 110, "readout_counter", "histogram 4096 × 32\n(8 BSRAM)\n+ per-spin counters",
          "plain")
    s.line(836, 349, 870, 349, "lk", label="sweep_done", loff=(-4, 22))
    s.poly([(955, 290), (955, 140), (620, 140), (620, 150)], "lk", label="histogram dump", lpos=(965, 220),
           anchor="start")
    s.box(870, 420, 170, 60, "6 LEDs", "spins 0-4 + busy\npins 10-16", "plain")
    s.line(836, 365, 870, 440, "lk")
    s.line(336, 445, 380, 445, "lk", label="clk", loff=(0, -6))
    s.line(336, 340, 380, 340, "lk", label="reset / run", loff=(0, -6))
    s.text(24, 548, "Orange box = the thermodynamic core; everything else is ordinary digital plumbing that "
                    "plays the role of XTR-0's host + FPGA [SDP p.9 Fig. 6].", "ts")
    return s


DRAWINGS = [
    ("01_classical_vs_thermo", d_mapping),
    ("02_pbit_anatomy", d_pbit),
    ("03_pgates", d_pgates),
    ("04_composition", None),
    ("05_hierarchy", d_stack),
    ("06_gibbs_timeline", d_timeline),
    ("07_fpga_board", d_board),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "img", "units"))
    ap.add_argument("--fragments", default=None, help="also write style-less copies here")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pj = os.path.join(ROOT, "tb", "build", "pgates.json")
    pgj = json.load(open(pj)) if os.path.exists(pj) else None
    for name, fn in DRAWINGS:
        s = d_compose(pgj) if name == "04_composition" else fn()
        with open(os.path.join(a.out, name + ".svg"), "w") as f:
            f.write(s.render(style=True))
        if a.fragments:
            os.makedirs(a.fragments, exist_ok=True)
            with open(os.path.join(a.fragments, name + ".svg"), "w") as f:
                f.write(s.render(style=False))
        print(f"wrote {name}.svg ({s.w}x{s.h})")


if __name__ == "__main__":
    main()
