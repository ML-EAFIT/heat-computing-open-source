#!/usr/bin/env python3
"""Builds the wiring / mechanism map of the p-bit thermodynamic sampler as one HTML page."""
import html
import sys

E = html.escape
OUT = sys.argv[1] if len(sys.argv) > 1 else ""
WARN = []
CH_T, CH_S = 7.5, 6.3          # px per char: 12.5px and 10.5px IBM Plex Mono


# ----------------------------------------------------------------- SVG helpers
class Fig:
    def __init__(s, fid, w, h, aria):
        s.id, s.w, s.h, s.aria = fid, w, h, aria
        s.p = []

    def _chk(s, txt, cw, room, where):
        if len(txt) * cw > room:
            WARN.append(f"{s.id}: '{txt}' ~{len(txt)*cw:.0f}px > {room:.0f}px ({where})")

    def box(s, x, y, w, h, title, subs=(), kind="cmb", tag=None, align="middle", hl=False):
        cls = {"cmb": "bx", "reg": "bx", "mem": "bx mem", "grp": "bx grp",
               "pin": "bx pin", "nz": "bx nzb"}[kind]
        if hl:
            cls += " hl"
        s.p.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" class="{cls}"/>')
        if kind == "grp":
            s.p.append(f'<text x="{x+8}" y="{y+15}" class="t ttl">{E(title)}</text>')
            for i, sub in enumerate(subs):
                s.p.append(f'<text x="{x+8}" y="{y+29+13*i}" class="t sub">{E(sub)}</text>')
            return
        total = 14 + 13 * len(subs)
        y0 = y + (h - total) / 2 + 11
        cx = x + w / 2 if align == "middle" else x + 8
        anc = "middle" if align == "middle" else "start"
        s._chk(title, CH_T, w - 10, "title")
        s.p.append(f'<text x="{cx:.1f}" y="{y0:.1f}" class="t ttl" text-anchor="{anc}">{E(title)}</text>')
        for i, sub in enumerate(subs):
            if sub:
                s._chk(sub, CH_S, w - 10, "sub")
            s.p.append(f'<text x="{cx:.1f}" y="{y0+14+13*i:.1f}" class="t sub" text-anchor="{anc}">{E(sub)}</text>')
        if kind == "reg":
            s.p.append(f'<path d="M{x},{y+h-13} L{x+8},{y+h-7} L{x},{y+h-1}" class="clk"/>')
        if tag:
            s.p.append(f'<text x="{x+w-5}" y="{y+10}" class="t tag" text-anchor="end">{E(tag)}</text>')

    def wire(s, pts, kind="d", arrow="end", label=None, at=None, anchor="start", width=None, wat=None):
        d = "M" + " L".join(f"{x},{y}" for x, y in pts)
        cls = {"d": "w", "c": "w ctl", "n": "w nz", "nc": "w nz ctl", "b": "w bus", "a": "w acc"}[kind]
        mk = {"d": "a", "c": "m", "b": "a", "n": "n", "nc": "n", "a": "c"}[kind]
        m = ""
        if arrow in ("end", "both"):
            m += f' marker-end="url(#{s.id}-{mk})"'
        if arrow in ("start", "both"):
            m += f' marker-start="url(#{s.id}-{mk})"'
        s.p.append(f'<path d="{d}" class="{cls}"{m}/>')
        if label:
            lx, ly = at
            c = "t lbl lbln" if kind in ("n", "nc") else "t lbl"
            if anchor == "start" and lx + len(label) * CH_S > s.w:
                WARN.append(f"{s.id}: label '{label}' runs past the right edge")
            s.p.append(f'<text x="{lx}" y="{ly}" class="{c}" text-anchor="{anchor}">{E(label)}</text>')
        if width is not None:
            wx, wy = wat
            horiz = any(y1 == y2 == wy and min(x1, x2) <= wx <= max(x1, x2)
                        for (x1, y1), (x2, y2) in zip(pts, pts[1:]))
            s.p.append(f'<path d="M{wx-4},{wy+5} L{wx+4},{wy-5}" class="wslash"/>')
            if horiz:
                s.p.append(f'<text x="{wx+3}" y="{wy+15}" class="t wnum">{E(str(width))}</text>')
            else:
                s.p.append(f'<text x="{wx+5}" y="{wy-6}" class="t wnum">{E(str(width))}</text>')

    def dot(s, x, y):
        s.p.append(f'<circle cx="{x}" cy="{y}" r="2.8" class="dot"/>')

    def text(s, x, y, txt, cls="sub", anchor="start"):
        s.p.append(f'<text x="{x}" y="{y}" class="t {cls}" text-anchor="{anchor}">{E(txt)}</text>')

    def raw(s, frag):
        s.p.append(frag)

    def svg(s):
        mk = []
        for k, c in (("a", "ah"), ("m", "ahm"), ("n", "ahn"), ("c", "ahc")):
            mk.append(f'<marker id="{s.id}-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                      f'markerHeight="7" orient="auto-start-reverse"><path d="M0,1 L10,5 L0,9 z" class="{c}"/></marker>')
        return (f'<svg class="dia" viewBox="0 0 {s.w} {s.h}" role="img" aria-label="{E(s.aria)}" '
                f'xmlns="http://www.w3.org/2000/svg"><defs>{"".join(mk)}</defs>{"".join(s.p)}</svg>')


def figure(fig, caption, extra_cls=""):
    return (f'<figure class="fig {extra_cls}"><div class="scroll">{fig.svg()}</div>'
            f'<figcaption>{caption}</figcaption></figure>')


def inv_symbol(f, x, y, left=False, cls="w"):
    """Logic inverter symbol, input at x (or x+32 when pointing left)."""
    if not left:
        f.raw(f'<path d="M{x},{y-13} L{x+24},{y} L{x},{y+13} Z" class="gate"/>'
              f'<circle cx="{x+28}" cy="{y}" r="4" class="gate"/>')
    else:
        f.raw(f'<path d="M{x+32},{y-13} L{x+8},{y} L{x+32},{y+13} Z" class="gate"/>'
              f'<circle cx="{x+4}" cy="{y}" r="4" class="gate"/>')


def mos(f, x, y, p=False, rot=0):
    """MOSFET. Unrotated: gate input at (x-20,y), terminals (x+16,y-24) and (x+16,y+24)."""
    g = [f'<path d="M-20,0 L{-8 if p else 0},0" class="w"/>',
         '<path d="M0,-12 L0,12" class="gate2"/>',
         '<path d="M5,-14 L5,14" class="gate2"/>',
         '<path d="M5,-10 L16,-10 L16,-24 M5,10 L16,10 L16,24" class="w"/>']
    if p:
        g.append('<circle cx="-4" cy="0" r="3.6" class="gate"/>')
    f.raw(f'<g transform="translate({x},{y}) rotate({rot})">{"".join(g)}</g>')


def tg(f, x, y, top="clk", bot="clk̄"):
    """Transmission gate symbol centred at (x, y), 24 px wide."""
    f.raw(f'<path d="M{x-12},{y-9} L{x+12},{y} L{x-12},{y+9} Z" class="gate"/>'
          f'<path d="M{x+12},{y-9} L{x-12},{y} L{x+12},{y+9} Z" class="gate"/>'
          f'<circle cx="{x}" cy="{y+13}" r="3" class="gate"/>')
    f.text(x, y - 13, top, "sub", "middle")
    f.text(x, y + 27, bot, "sub", "middle")


def timeline(fid, segs, aria, total_label, hl=(), heat=()):
    """Cycle bar: segs = [(label, cycles)]."""
    W, x0, y0, H = 1000, 20, 46, 30
    tot = sum(c for _, c in segs)
    px = (W - 40) / tot
    f = Fig(fid, W, 116, aria)
    x = x0
    narrow_i = 0
    for i, (lab, c) in enumerate(segs):
        w = c * px
        cls = "seg"
        if lab in hl:
            cls += " seghl"
        if lab in heat:
            cls += " segheat"
        f.raw(f'<rect x="{x:.1f}" y="{y0}" width="{max(w-2,1):.1f}" height="{H}" class="{cls}"/>')
        txt = f"{lab} {c}" if c > 1 else lab
        if len(txt) * CH_S + 8 < w:
            f.text(x + w / 2, y0 + 19, txt, "seglbl", "middle")
        else:
            ly = 34 if narrow_i % 2 == 0 else 20
            f.raw(f'<path d="M{x+w/2:.1f},{y0} L{x+w/2:.1f},{ly+4}" class="tick"/>')
            f.text(x + w / 2, ly, txt, "seglbl", "middle")
            narrow_i += 1
        x += w
    # cycle axis
    f.raw(f'<path d="M{x0},{y0+H+8} L{x0+tot*px:.1f},{y0+H+8}" class="axis"/>')
    step = 1 if tot <= 14 else 2 if tot <= 40 else 10 if tot <= 120 else 500
    if tot > 1000:
        step = 500
    k = 0
    while k <= tot:
        xx = x0 + k * px
        f.raw(f'<path d="M{xx:.1f},{y0+H+5} L{xx:.1f},{y0+H+11}" class="axis"/>')
        f.text(xx, y0 + H + 24, str(k), "axn", "middle")
        k += step
    f.text(W - 20, 14, total_label, "axn", "end")
    return f


# ----------------------------------------------------------------- figures
def fig_legend():
    f = Fig("lg", 1000, 70, "Legend for the schematics: register, memory, combinational block, data wire, control wire, noise wire, bus width")
    f.box(10, 12, 120, 44, "register", ["clocked"], "reg")
    f.box(150, 12, 120, 44, "memory", ["BSRAM / SSRAM"], "mem")
    f.box(290, 12, 120, 44, "logic", ["combinational"], "cmb")
    f.wire([(440, 26), (520, 26)], "d", label="data", at=(530, 30))
    f.wire([(440, 46), (520, 46)], "c", label="control / strobe", at=(530, 50))
    f.wire([(660, 26), (740, 26)], "n", label="noise (entropy path)", at=(750, 30))
    f.wire([(660, 46), (740, 46)], "d", label="bus, 16 bits wide", at=(750, 50), width=16, wat=(700, 46))
    f.dot(940, 36)
    f.text(950, 40, "joint")
    return f


def fig_system():
    f = Fig("sys", 1100, 590, "Top-level wiring of top.v: pins, reset and button logic, UART, command FSM, schedule, sampling engine, raw noise lane, readout counters, LEDs")
    # pins
    f.box(10, 30, 100, 36, "clk", ["pin 52, 27 MHz"], "pin")
    f.box(10, 86, 100, 36, "btn_rst_n", ["S1, pin 4"], "pin")
    f.box(10, 136, 100, 36, "btn_run_n", ["S2, pin 3"], "pin")
    f.box(10, 236, 100, 36, "uart_rxd", ["pin 18"], "pin")
    f.box(10, 316, 100, 36, "uart_txd", ["pin 17"], "pin")
    f.box(1000, 232, 90, 36, "led[5:0]", ["pins 10-16"], "pin")
    # blocks
    f.box(150, 80, 160, 100, "reset + buttons", ["POR 4-bit counter", "2-FF sync S1, S2", "S2 debounce 2¹⁸"], "reg")
    f.box(150, 226, 160, 56, "uart_rx", ["2-FF sync, mid-bit"], "reg")
    f.box(150, 306, 160, 56, "uart_tx", ["10-bit shift out"], "reg")
    f.box(420, 40, 220, 520, "command FSM", [
        "23 states, S_IDLE ... S_NOISE", "",
        "args_reg 48b ← argument bytes", "word_reg → reply, LSB first",
        "ret_reg = return of S_SEND", "cnt_reg, tot_reg (32b)", "",
        "mode b0 b1 dl sps → schedule", "nzf nzq → sparse noise", "hold_reg → pause per sweep"],
        "reg", align="start")
    f.box(740, 40, 210, 64, "schedule_ctrl", ["β: const, linear, table"], "reg")
    f.box(740, 160, 210, 170, "sampling engine", [
        "SPARSE=0: pbit_array", "  + temper_ctrl if TEMPER_K>1",
        "SPARSE=1: sparse_array", "", "see L4 below"], "cmb", align="start", hl=True)
    f.box(740, 380, 170, 52, "raw noise_bank", ["lane-0 LFSR, cmd 'N'"], "nz")
    f.box(740, 470, 210, 90, "readout_counter", ["hist 2¹² × 32 (BSRAM)", "up[i]: N × 32-bit", "n_samples"], "reg")
    # wires: pins -> blocks
    f.wire([(110, 48), (140, 48)], label="clk → every DFF", at=(146, 52))
    f.wire([(110, 104), (150, 104)])
    f.wire([(110, 154), (150, 154)])
    f.wire([(310, 100), (345, 100)], "c", label="rst → all", at=(316, 94))
    f.wire([(310, 150), (420, 150)], "c", label="btn_run_evt", at=(316, 144))
    f.wire([(110, 254), (150, 254)])
    f.wire([(310, 254), (420, 254)], label="rx_data, valid", at=(316, 248), width=8, wat=(395, 254))
    f.wire([(420, 326), (310, 326)], label="tx_data, valid", at=(316, 320), width=8, wat=(395, 326))
    f.wire([(310, 346), (420, 346)], "c", label="tx_ready", at=(316, 360))
    f.wire([(150, 334), (110, 334)])
    # FSM <-> right side
    f.wire([(640, 72), (740, 72)], "c", label="restart, cfg", at=(646, 66))
    f.wire([(810, 104), (810, 160)], label="β", at=(816, 140), width=8, wat=(810, 122))
    f.wire([(890, 160), (890, 104)], "c", label="swd", at=(896, 140))
    f.wire([(640, 190), (740, 190)], "c", label="prog. buses", at=(646, 184))
    f.wire([(640, 216), (740, 216)], "c", label="run, n_sweeps", at=(646, 210))
    f.wire([(640, 242), (740, 242)], "c", label="seed, spin_we", at=(646, 236))
    f.wire([(640, 268), (740, 268)], "c", label="hold, stop", at=(646, 262))
    f.wire([(740, 296), (640, 296)], "c", label="swd, busy", at=(646, 290))
    f.wire([(740, 320), (640, 320)], label="spins", at=(646, 314), width="N", wat=(720, 320))
    f.wire([(925, 330), (925, 470)], label="spins, swd", at=(931, 420))
    f.wire([(640, 396), (740, 396)], "c", label="raw_en", at=(646, 390))
    f.wire([(740, 418), (640, 418)], "n", label="raw_rnd", at=(646, 412), width=16, wat=(720, 418))
    f.wire([(640, 492), (740, 492)], "c", label="clear, addr", at=(646, 486))
    f.wire([(740, 530), (640, 530)], label="counts", at=(646, 524), width=32, wat=(720, 530))
    f.wire([(950, 250), (1000, 250)])
    f.text(1090, 222, "~{busy, spins[4:0]}", "lbl", "end")
    return f


def fig_dense():
    f = Fig("dn", 1100, 410, "pbit_array datapath: order RAM, bias RAM, coupling_row streaming the weight row through weight_mac, the p-bit, the noise lane and the spin register that feeds back")
    f.box(20, 52, 100, 52, "k_reg", ["order pos"], "reg")
    f.box(150, 46, 130, 64, "omem", ["visit order"], "mem", tag="SSRAM")
    f.box(330, 46, 130, 64, "hmem", ["h[i], Q3.4"], "mem", tag="SSRAM")
    f.box(330, 140, 130, 44, "node_reg", [], "reg")
    f.box(490, 24, 360, 290, "coupling_row: N+2 cycles", [], "grp")
    f.box(510, 56, 110, 44, "j_reg", ["column 0..N"], "reg")
    f.box(670, 46, 160, 64, "wmem", ["J[i][j], 2^2IW × 8"], "mem", tag="BSRAM")
    f.box(510, 196, 110, 44, "s mux", ["spins[jd]"])
    f.box(670, 190, 160, 76, "weight_mac", ["acc ← h (load)", "acc ± J (s = ±1)"], "reg")
    f.box(890, 180, 150, 100, "pbit", ["σ(2β·field) = p", "rnd < p → bit", "12 cycles"], hl=True)
    f.box(890, 340, 150, 52, "noise_bank", ["lfsr lane 0"], "nz")
    f.box(650, 340, 190, 52, "spin write", ["spins[node] ← bit"])
    f.box(430, 340, 170, 52, "spins_reg", ["N bits, Gibbs state"], "reg")
    f.box(20, 220, 260, 100, "control FSM", ["IDLE ORD BIAS MAC", "MACW PBW WR", "N+18 cycles / update"], "reg")
    f.wire([(120, 78), (150, 78)], width="IW", wat=(134, 78))
    f.wire([(280, 78), (330, 78)], label="o_q", at=(286, 72))
    f.dot(305, 78)
    f.wire([(305, 78), (305, 162), (330, 162)])
    f.wire([(460, 78), (475, 78), (475, 250), (670, 250)], label="h_q", at=(470, 72), width=8, wat=(560, 250))
    f.wire([(460, 162), (645, 162), (645, 78)], arrow=None, label="row", at=(500, 156))
    f.dot(645, 78)
    f.wire([(620, 78), (670, 78)], label="{row,j}", at=(622, 72))
    f.wire([(750, 110), (750, 190)], label="w_q", at=(756, 152), width=8, wat=(750, 172))
    f.wire([(620, 218), (670, 218)], label="s_j", at=(628, 212))
    f.wire([(830, 228), (890, 228)], label="field", at=(836, 222), width=14, wat=(872, 228))
    f.wire([(965, 140), (965, 180)], label="β (Q4.4)", at=(971, 160))
    f.wire([(930, 340), (930, 280)], "n", label="rnd", at=(936, 318), width=16, wat=(930, 300))
    f.wire([(1005, 280), (1005, 340)], "nc", label="take", at=(1011, 318))
    f.wire([(890, 262), (865, 262), (865, 366), (840, 366)], label="bit", at=(870, 320))
    f.wire([(650, 366), (600, 366)])
    f.wire([(565, 340), (565, 240)], label="spins", at=(571, 300), width="N", wat=(565, 320))
    f.wire([(395, 184), (395, 296), (745, 296), (745, 340)], "c", label="node", at=(402, 290))
    f.wire([(430, 356), (300, 356)], label="spins → out", at=(304, 350))
    f.wire([(300, 378), (430, 378)], "c", label="spin_we, data", at=(304, 392))
    f.wire([(1040, 230), (1085, 230)], "a", label="upd_*", at=(1044, 224))
    return f


def fig_pbit():
    f = Fig("pb", 1100, 320, "pbit datapath: serial shift-add multiplier, rounding and saturation to a 10-bit logit, sigmoid ROM, 16-bit comparator against the noise draw, output bit register")
    f.wire([(10, 78), (70, 78)], label="field", at=(12, 72), width=14, wat=(50, 78))
    f.wire([(10, 198), (70, 198)], label="β", at=(12, 192), width=8, wat=(50, 198))
    f.box(70, 52, 150, 52, "mcand_reg", ["AW+9 bits, <<1"], "reg")
    f.box(70, 172, 150, 52, "bsh_reg", ["β, >>1"], "reg")
    f.box(270, 62, 90, 60, "+", ["prod+mcand"])
    f.box(410, 62, 140, 60, "prod_reg", ["β × field"], "reg")
    f.box(600, 62, 140, 60, "round, sat", ["(prod+2)>>>2", "clamp to 10 bit"])
    f.box(780, 62, 110, 60, "logit_reg", ["Q5.5 = 2βh"], "reg")
    f.box(940, 56, 150, 72, "sigmoid_lut", ["BSRAM 1024 × 16", "addr = logit+512"], "mem", tag="BSRAM")
    f.box(940, 200, 150, 60, "rnd < p", ["16-bit compare"], hl=True)
    f.box(760, 204, 120, 52, "bit_reg", ["1 = spin +1"], "reg")
    f.box(270, 232, 380, 64, "pbit FSM", ["IDLE, MUL ×8, LOGIT, LUT, CMP", "done + rnd_take in cycle 12"], "reg")
    f.wire([(220, 78), (270, 78)])
    f.wire([(360, 92), (410, 92)])
    f.wire([(550, 92), (600, 92)])
    f.dot(575, 92)
    f.wire([(575, 92), (575, 150), (290, 150), (290, 122)], label="accumulate", at=(440, 166))
    f.wire([(220, 198), (340, 198), (340, 122)], "c", label="bsh[0] gates the add", at=(226, 216))
    f.wire([(740, 92), (780, 92)])
    f.wire([(890, 92), (940, 92)], label="logit", at=(894, 86), width=10, wat=(912, 92))
    f.wire([(1015, 128), (1015, 200)], label="p", at=(1021, 168), width=16, wat=(1015, 150))
    f.text(1008, 186, "USE_PWL=1: sigmoid_pwl", "lbl", "end")
    f.wire([(1015, 306), (1015, 260)], "n", label="rnd", at=(1021, 304), width=16, wat=(1015, 282))
    f.text(1009, 304, "noise lane", "lbl lbln", "end")
    f.wire([(940, 230), (880, 230)])
    f.wire([(760, 230), (700, 230)], label="bit_out", at=(704, 224))
    return f


def fig_lfsr():
    f = Fig("lf", 1100, 360, "Galois LFSR: 32-bit state, lane-0 feedback taps, the 16-step leap network, output draw and next-state mux, plus the optional ring-oscillator jitter source")
    mask = 0xB4BCD35C
    x0, cw, yc = 200, 22, 96
    f.text(20, 104, "state_reg", "ttl")
    f.text(20, 119, "32 DFF, reset = seed")
    for k in range(32):
        x = x0 + (31 - k) * cw
        f.raw(f'<rect x="{x}" y="{yc}" width="{cw-2}" height="26" class="bx cell"/>')
        if k % 4 == 3 or k == 0:
            f.text(x + (cw - 2) / 2, yc + 40, f"b{k}", "axn", "middle")
    # feedback line from b0 output
    xb0 = x0 + 31 * cw + cw - 2
    f.wire([(xb0, 109), (930, 109), (930, 58)], arrow=None)
    taps = [k for k in range(32) if (mask >> k) & 1]
    xl = x0 + (31 - max(taps)) * cw + (cw - 2) / 2
    f.wire([(930, 58), (xl, 58)], arrow=None)
    for k in taps:
        x = x0 + (31 - k) * cw + (cw - 2) / 2
        f.dot(x, 58)
        f.raw(f'<circle cx="{x}" cy="76" r="6" class="xor"/><path d="M{x-6},76 L{x+6},76 M{x},70 L{x},82" class="w"/>')
        f.wire([(x, 58), (x, 70)], arrow=None)
        f.wire([(x, 82), (x, 96)])
    f.text(200, 30, "one Galois step: s ← (s >> 1) ⊕ (s₀ ? MASK : 0)    MASK = 0xB4BCD35C, 18 taps (lane 0)", "lbl")
    f.text(936, 86, "b0 out", "lbl")
    f.box(200, 160, 704, 52, "leap-16: 16 Galois steps unrolled into one XOR network",
          ["new state = A¹⁶·s ;  rnd[k] = s₀ after k steps (linear over GF(2))"])
    f.wire([(552, 122), (552, 160)], width=32, wat=(552, 150))
    f.box(940, 150, 145, 80, "next-state mux", ["seed_load: seed", "en: leap", "else hold"])
    f.wire([(904, 186), (940, 186)], width=32, wat=(920, 186))
    f.wire([(1060, 150), (1060, 40), (180, 40), (180, 109), (200, 109)], label="state_next", at=(960, 52))
    f.wire([(552, 212), (552, 262), (700, 262)], "n", label="rnd → pbit / gauss LUT", at=(560, 256), width=16, wat=(552, 238))
    f.wire([(985, 290), (985, 230)], "c", label="en", at=(990, 284))
    f.wire([(1040, 290), (1040, 230)], "c", label="seed", at=(1045, 284))
    f.text(985, 304, "en = rnd_take of the consumer", "lbl", "middle")
    # ring oscillator inset
    f.box(20, 250, 470, 100, "RO_JITTER = 1 (hardware only, not in these runs)", [], "grp")
    for i, x in enumerate((44, 104, 164)):
        inv_symbol(f, x, 300)
        if i < 2:
            f.wire([(x + 32, 300), (x + 60, 300)], arrow=None)
    f.wire([(196, 300), (214, 300), (214, 330), (30, 330), (30, 300), (44, 300)], arrow=None)
    f.dot(214, 300)
    f.wire([(214, 300), (250, 300)])
    f.box(250, 282, 50, 36, "s1", [], "reg")
    f.box(320, 282, 50, 36, "s2", [], "reg")
    f.wire([(300, 300), (320, 300)])
    f.raw('<circle cx="410" cy="300" r="8" class="xor"/><path d="M402,300 L418,300 M410,292 L410,308" class="w"/>')
    f.wire([(370, 300), (402, 300)])
    f.text(424, 304, "rnd[0]", "lbl")
    f.text(44, 344, "3 inverters in a loop", "lbl")
    return f


def fig_sparse():
    f = Fig("sp", 1090, 530, "sparse_array: four lanes in lock step, each with node RAM, neighbour RAM, weight_mac, p-bit and its own LFSR lane, a shared commit mux and spin register")
    f.box(20, 46, 130, 40, "ptr_reg", ["round"], "reg")
    f.box(20, 104, 130, 40, "slot_reg", ["0..D"], "reg")
    f.box(20, 200, 130, 130, "control FSM", ["NODE MAC", "MACW × D", "PB PBW × 12", "CMT × P", "D+P+15 cycles"], "reg")
    f.wire([(150, 66), (160, 66), (160, 36), (980, 36)], "b", arrow=None, label="ptr", at=(988, 40))
    f.dot(160, 66)
    f.wire([(160, 66), (160, 108)], arrow=None)
    f.wire([(150, 124), (160, 124), (160, 108), (1000, 108)], "b", arrow=None, label="{ptr,slot}", at=(1006, 112))
    f.dot(160, 108)
    lanes = [170 + l * 215 for l in range(4)]
    for l, X in enumerate(lanes):
        f.box(X + 30, 50, 160, 46, f"nmem[{l}]", ["round → {v, node, h}"], "mem")
        f.box(X + 30, 120, 160, 46, f"wmem[{l}]", ["{round,slot} → {j, J}"], "mem")
        f.box(X + 30, 190, 160, 46, "weight_mac", ["acc ± J·s_j"], "reg")
        f.box(X + 30, 260, 160, 46, "pbit", [f"lane {l}, 12 cycles"], hl=(l == 0))
        f.box(X + 90, 330, 100, 36, f"lfsr {l}", [], "nz")
        f.dot(X + 150, 36)
        f.wire([(X + 150, 36), (X + 150, 50)])
        f.dot(X + 170, 108)
        f.wire([(X + 170, 108), (X + 170, 120)])
        f.wire([(X + 30, 73), (X + 12, 73), (X + 12, 400)])
        f.dot(X + 12, 213)
        f.wire([(X + 12, 213), (X + 30, 213)])
        f.text(X + 16, 207, "h", "lbl")
        f.text(X + 16, 392, "v,node", "lbl")
        f.wire([(X + 110, 166), (X + 110, 190)])
        f.wire([(X + 110, 236), (X + 110, 260)])
        f.wire([(X + 140, 330), (X + 140, 306)], "n")
        f.wire([(X + 60, 306), (X + 60, 400)], label="bit", at=(X + 64, 350))
        f.wire([(X + 180, 178), (X + 180, 190)])
        f.dot(X + 180, 178)
    # shared noise line (nz_share)
    f.wire([(lanes[0] + 140, 366), (lanes[0] + 140, 380), (lanes[3] + 140, 380)], "nc", arrow=None)
    for X in lanes[1:]:
        f.dot(X + 140, 380)
        f.wire([(X + 140, 380), (X + 140, 366)], "nc")
    f.text(lanes[0] + 152, 393, "nz_share", "lbl lbln")
    f.box(170, 400, 860, 46, "commit: cidx = 0 .. P-1, one lane per cycle",
          ["spins_next[node] ← bit  if valid ∧ ¬keep (nz_tau)"])
    f.box(430, 476, 340, 40, "spins_reg (N bits)", [], "reg")
    f.wire([(600, 446), (600, 476)])
    f.wire([(770, 496), (1050, 496), (1050, 178), (lanes[0] + 180, 178)], "b", arrow=None)
    f.text(1056, 300, "spins", "lbl")
    f.wire([(430, 496), (200, 496)], label="spins → readout, LEDs", at=(204, 490))
    return f


def fig_temper():
    f = Fig("pt", 1100, 460, "temper_ctrl: replica banks and energies, replica mux into the shared pbit_array, energy update from the update tap, ladder and label tables, swap decision through a second p-bit with its own LFSR lane")
    f.text(85, 40, "e[k] = E/2, 24 b", "lbl", "middle")
    f.text(170, 40, "replica spins, N b", "lbl")
    for k in range(4):
        y = 50 + k * 46
        f.box(30, y, 110, 36, f"e[{k}]", [], "reg")
        f.box(170, y, 140, 36, f"bank[{k}]", [], "reg")
        f.wire([(310, y + 18), (350, y + 18)])
    f.box(350, 50, 60, 174, "mux", ["k_reg"])
    f.box(470, 60, 210, 130, "pbit_array", ["1 sweep per load", "shared engine (L4 dense)"], hl=True)
    f.wire([(410, 110), (470, 110)], label="spins", at=(416, 104), width="N", wat=(452, 110))
    f.wire([(470, 170), (440, 170), (440, 240), (240, 240), (240, 224)], label="store bank[k] at a_swd", at=(250, 256))
    f.box(30, 270, 200, 52, "e[k] ± field", ["on every flip (upd_we)"])
    f.wire([(85, 270), (85, 224)])
    f.wire([(575, 190), (575, 296), (230, 296)], "a", label="upd_old, upd_field", at=(300, 290))
    f.box(740, 50, 160, 56, "ladder[0..3]", ["β per rung, Q4.4"], "reg")
    f.box(740, 130, 160, 56, "rung[], rep_at[]", ["replica ↔ rung"], "reg")
    f.wire([(740, 78), (680, 78)], label="a_beta", at=(686, 72))
    f.wire([(300, 50), (300, 26), (480, 26)], label="spins_out = bank[rep_at[0]] → readout", at=(310, 20))
    f.box(470, 330, 170, 52, "Δe = e[rx] − e[ry]", ["24-bit subtract"])
    f.box(740, 230, 160, 52, "Δβ = β[a] − β[a+1]", ["ladder gap"])
    f.box(700, 330, 140, 52, "sign fold, sat", ["→ field, |Δβ|"])
    f.box(900, 310, 170, 64, "swap pbit", ["σ(2|Δβ|·Δe′)", "1 = accept"], hl=True)
    f.box(910, 404, 150, 40, "lfsr lane 1", ["mask A3000000"], "nz")
    f.wire([(30, 110), (15, 110), (15, 356), (470, 356)], label="e[rx], e[ry]", at=(250, 350))
    f.wire([(900, 78), (920, 78), (920, 256), (900, 256)])
    f.wire([(640, 356), (700, 356)])
    f.wire([(820, 282), (820, 330)])
    f.wire([(840, 342), (900, 342)])
    f.wire([(985, 404), (985, 374)], "n")
    f.wire([(985, 310), (985, 158), (900, 158)], label="accept", at=(991, 240))
    return f


def fig_pmode():
    f = Fig("pm", 1100, 560, "pmode_array: tanh phase through the sigmoid ROM, coupling multiply-accumulate, five-stage gradient pipeline with Gaussian noise from an inverse-CDF ROM, and the Jacobi commit back to the state memory")
    f.text(20, 28, "Phase 1 · tanh: 2 cycles per site", "lbl")
    f.box(20, 40, 130, 60, "xmem", ["x, Q4.12"], "mem", tag="FF")
    f.box(190, 40, 150, 60, "round, sat", ["(x+32)>>>6", "logit of 2x"])
    f.box(380, 40, 150, 60, "sigmoid_lut", ["BSRAM 1024×16", "p = σ(2x)"], "mem")
    f.box(570, 40, 130, 60, "p − 32768", ["= tanh x, Q.15"])
    f.box(740, 40, 130, 60, "smem", ["s, M × 17"], "mem", tag="FF")
    for a, b in ((150, 190), (340, 380), (530, 570), (700, 740)):
        f.wire([(a, 70), (b, 70)])
    f.text(20, 140, "Phase 2 · coupling MAC: 2 cycles per j", "lbl")
    f.box(20, 150, 150, 60, "hmem", ["h[i], Q3.4"], "mem")
    f.box(190, 150, 150, 60, "amem", ["A[i][j], Q3.4"], "mem")
    f.box(380, 150, 150, 60, "aq × sq", ["8 × 17 multiply"], tag="DSP")
    f.box(570, 150, 170, 60, "acc_reg", ["+= aq·sq per j", "Q.19, 40 bits"], "reg")
    f.wire([(340, 180), (380, 180)])
    f.wire([(530, 180), (570, 180)])
    f.wire([(805, 100), (805, 125), (455, 125), (455, 150)], label="s_j", at=(811, 118))
    f.wire([(95, 210), (95, 236), (610, 236), (610, 210)], label="init: h_i << 15", at=(104, 230))
    f.wire([(700, 210), (700, 262), (180, 262), (180, 300)])
    for x, g in ((20, "G1"), (230, "G2"), (440, "G3"), (650, "G4"), (860, "G5")):
        f.text(x, 292, g, "ttl")
    f.text(46, 292, "Phase 3 · 1 cycle per stage", "lbl")
    f.box(20, 300, 190, 44, "f = acc >>> 7", ["field, Q.12"], "reg")
    f.box(20, 360, 190, 44, "g = 1 − s²", ["s = smem[i], Q.15"], "reg", tag="DSP")
    f.box(20, 420, 190, 44, "x² >>> 12", ["x = xmem[i], Q.12"], "reg", tag="DSP")
    f.box(230, 300, 190, 44, "cpl = f·g >>> 15", ["(1−s²)(A·s + h)"], "reg", tag="DSP")
    f.box(230, 420, 190, 44, "q = x(x² − 1)", ["double-well slope"], "reg", tag="DSP")
    f.box(440, 420, 190, 44, "qt = λ·q >>> 4", ["λ, Q4.4"], "reg", tag="DSP")
    f.box(20, 480, 190, 44, "lfsr lane 0", ["16-bit draw"], "nz")
    f.box(230, 480, 190, 44, "gauss_lut (inv-CDF)", ["addr = rnd[15:6]"], "mem")
    f.box(440, 480, 190, 44, "xi_reg", ["Gaussian, Q4.12"], "reg")
    f.box(650, 480, 190, 44, "nz = σ·ξ >>> 12", ["σ = √(2Tε)"], "reg", tag="DSP")
    f.box(860, 340, 190, 70, "xnew[i] ← sat16", ["x − (qt − cpl)>>>k", "  + nz"], "mem", hl=True)
    f.wire([(210, 316), (230, 316)])
    f.wire([(210, 382), (220, 382), (220, 330), (230, 330)])
    f.wire([(210, 442), (230, 442)])
    f.wire([(420, 442), (440, 442)])
    f.wire([(420, 322), (845, 322), (845, 356), (860, 356)], label="cpl", at=(700, 316))
    f.wire([(630, 442), (835, 442), (835, 394), (860, 394)], label="qt", at=(700, 436))
    f.wire([(210, 502), (230, 502)], "n")
    f.wire([(420, 502), (440, 502)], "n")
    f.wire([(630, 502), (650, 502)], "n")
    f.wire([(840, 502), (955, 502), (955, 410)], "n", label="nz", at=(961, 470))
    f.wire([(1050, 375), (1085, 375), (1085, 548), (8, 548), (8, 70), (20, 70)], "a",
           label="CMT: xmem ← xnew for every site at once (Jacobi), 1 cycle", at=(300, 542))
    return f


def fig_uart():
    f = Fig("ua", 1100, 150, "UART frame for the byte 0x57 ('W'): start bit, eight data bits LSB first, stop bit, 234 clock cycles per bit, receiver samples in the middle of each bit")
    bits = [1, 0] + [(0x57 >> i) & 1 for i in range(8)] + [1, 1]
    names = ["idle", "start"] + [f"d{i}" for i in range(8)] + ["stop", "idle"]
    bw, x0, yh, yl = 88, 22, 40, 80
    pts = []
    for i, b in enumerate(bits):
        y = yh if b else yl
        pts += [(x0 + i * bw, y), (x0 + (i + 1) * bw, y)]
    f.raw('<path d="M' + " L".join(f"{x},{y}" for x, y in pts) + '" class="w wave"/>')
    for i, n in enumerate(names):
        cx = x0 + i * bw + bw / 2
        f.text(cx, 104, n, "axn", "middle")
        f.text(cx, 30, str(bits[i]), "axn", "middle")
        if 1 <= i <= 10:
            f.raw(f'<path d="M{cx},{yh-4} L{cx},{yl+4}" class="w ctl"/>')
        if i <= 11:
            f.raw(f'<path d="M{x0+i*bw},{yl+8} L{x0+i*bw},{yl+14}" class="axis"/>')
    f.text(x0 + 2 * bw, 128, "each bit = 234 clocks at 27 MHz (115,385 baud); dashed = uart_rx sample point (mid-bit)", "lbl")
    return f


# transistor panels ------------------------------------------------------------
def tp_inverter():
    f = Fig("t1", 520, 230, "CMOS inverter at transistor level and a three-inverter ring oscillator")
    f.wire([(40, 30), (150, 30)], arrow=None)
    f.text(20, 34, "VDD", "lbl")
    f.wire([(40, 200), (150, 200)], arrow=None)
    f.text(20, 204, "GND", "lbl")
    mos(f, 84, 75, p=True)
    mos(f, 84, 155, p=False)
    f.wire([(100, 51), (100, 30)], arrow=None)
    f.wire([(100, 99), (100, 131)], arrow=None)
    f.wire([(100, 179), (100, 200)], arrow=None)
    f.wire([(64, 75), (64, 155)], arrow=None)
    f.dot(64, 115)
    f.wire([(30, 115), (64, 115)], arrow=None)
    f.text(30, 108, "in", "lbl")
    f.dot(100, 115)
    f.wire([(100, 115), (150, 115)])
    f.text(132, 108, "out", "lbl")
    f.text(110, 72, "PMOS", "lbl")
    f.text(110, 160, "NMOS", "lbl")
    for i, x in enumerate((240, 320, 400)):
        inv_symbol(f, x, 115)
        if i < 2:
            f.wire([(x + 32, 115), (x + 80, 115)], arrow=None)
    f.wire([(432, 115), (452, 115), (452, 165), (222, 165), (222, 115), (240, 115)], arrow=None)
    f.dot(452, 115)
    f.wire([(452, 115), (500, 115)])
    f.text(250, 84, "ring oscillator (RO_JITTER)", "lbl")
    f.text(468, 106, "s1", "lbl")
    return f


def tp_sram():
    f = Fig("t2", 520, 230, "Six-transistor SRAM cell: two cross-coupled inverters and two access transistors on the word line")
    f.wire([(60, 20), (60, 214)], arrow=None)
    f.wire([(460, 20), (460, 214)], arrow=None)
    f.text(66, 210, "BL", "lbl")
    f.text(454, 210, "BL_n", "lbl", "end")
    f.wire([(40, 44), (480, 44)], arrow=None)
    f.text(250, 38, "WL (word line)", "lbl", "middle")
    mos(f, 110, 100, p=False, rot=90)
    mos(f, 410, 100, p=False, rot=90)
    f.wire([(110, 80), (110, 44)], arrow=None)
    f.wire([(410, 80), (410, 44)], arrow=None)
    f.dot(110, 44)
    f.dot(410, 44)
    f.wire([(60, 116), (86, 116)], arrow=None)
    f.wire([(134, 116), (170, 116)], arrow=None)
    f.wire([(386, 116), (350, 116)], arrow=None)
    f.wire([(434, 116), (460, 116)], arrow=None)
    f.dot(60, 116)
    f.dot(460, 116)
    f.wire([(170, 86), (170, 156)], arrow=None)
    f.wire([(350, 86), (350, 156)], arrow=None)
    f.dot(170, 116)
    f.dot(350, 116)
    inv_symbol(f, 230, 86)
    f.wire([(170, 86), (230, 86)], arrow=None)
    f.wire([(262, 86), (350, 86)], arrow=None)
    inv_symbol(f, 258, 156, left=True)
    f.wire([(350, 156), (290, 156)], arrow=None)
    f.wire([(258, 156), (170, 156)], arrow=None)
    f.text(176, 134, "Q", "ttl")
    f.text(344, 134, "Q_n", "ttl", "end")
    f.text(260, 196, "2 inverters = 4 T, + 2 access NMOS", "lbl", "middle")
    return f


def tp_lut():
    f = Fig("t3", 520, 250, "LUT4: sixteen configuration SRAM bits selected by a four-level tree of 2-to-1 pass-transistor multiplexers")
    f.text(24, 14, "16 config bits", "lbl")
    ys = [25 + 13 * i for i in range(16)]
    for i, y in enumerate(ys):
        f.raw(f'<rect x="30" y="{y-5}" width="22" height="10" class="bx mem"/>')
        f.wire([(52, y), (90, y)], arrow=None)
    levels = []
    cur = ys
    xs = [90, 160, 230, 300]
    for li, x in enumerate(xs):
        nxt = []
        for j in range(0, len(cur), 2):
            a, b = cur[j], cur[j + 1]
            c = (a + b) / 2
            hh = (b - a) / 2 + 6
            f.raw(f'<path d="M{x},{c-hh:.1f} L{x+18},{c-hh+5:.1f} L{x+18},{c+hh-5:.1f} L{x},{c+hh:.1f} Z" class="gate"/>')
            f.raw(f'<path d="M{x+9},{c+hh-2.5:.1f} L{x+9},{c+hh+5:.1f}" class="w ctl"/>')
            nxt.append(c)
            if li < 3:
                f.wire([(x + 18, c), (xs[li + 1], c)], arrow=None)
        levels.append(nxt)
        f.text(x + 9, 246, "abcd"[li], "ttl", "middle")
        cur = nxt
    f.wire([(318, cur[0]), (370, cur[0])])
    f.text(376, cur[0] + 4, "F(a,b,c,d)", "ttl")
    f.text(376, cur[0] + 22, "15 muxes, 4 levels", "lbl")
    f.text(376, cur[0] + 36, "picks 1 of 16 bits", "lbl")
    return f


def tp_dff():
    f = Fig("t4", 520, 200, "Master-slave D flip-flop built from transmission gates and inverters")
    f.box(40, 30, 200, 150, "master latch", ["open while clk = 0"], "grp")
    f.box(270, 30, 200, 150, "slave latch", ["open while clk = 1"], "grp")
    f.wire([(10, 100), (54, 100)], arrow=None)
    f.text(12, 94, "D", "ttl")
    tg(f, 66, 100, "clk_n", "clk")
    f.wire([(78, 100), (110, 100)], arrow=None)
    inv_symbol(f, 110, 100)
    f.dot(96, 100)
    f.wire([(142, 100), (284, 100)], arrow=None)
    inv_symbol(f, 130, 150, left=True)
    tg(f, 196, 150, "clk", "clk_n")
    f.wire([(208, 150), (226, 150), (226, 100)], arrow=None)
    f.dot(226, 100)
    f.wire([(184, 150), (162, 150)], arrow=None)
    f.wire([(130, 150), (96, 150), (96, 100)], arrow=None)
    tg(f, 296, 100, "clk", "clk_n")
    f.wire([(308, 100), (340, 100)], arrow=None)
    inv_symbol(f, 340, 100)
    f.dot(326, 100)
    f.wire([(372, 100), (500, 100)])
    f.text(484, 94, "Q", "ttl")
    inv_symbol(f, 360, 150, left=True)
    tg(f, 426, 150, "clk_n", "clk")
    f.wire([(438, 150), (456, 150), (456, 100)], arrow=None)
    f.dot(456, 100)
    f.wire([(414, 150), (392, 150)], arrow=None)
    f.wire([(360, 150), (326, 150), (326, 100)], arrow=None)
    return f


def tp_carry():
    f = Fig("t5", 520, 290, "Carry-chain cells: each LUT computes propagate P, a dedicated mux passes the carry up the column, and an XOR forms the sum bit")
    for k, yc in ((2, 60), (1, 140), (0, 220)):
        f.box(40, yc - 20, 110, 40, f"LUT {k}", [f"P{k} = A{k} ⊕ B{k}"])
        f.raw(f'<rect x="285" y="{yc-16}" width="30" height="32" rx="3" class="gate"/>')
        f.text(300, yc + 4, "mux", "axn", "middle")
        f.wire([(150, yc - 6), (285, yc - 6)], "c", label=f"P{k} (select)", at=(160, yc - 10))
        f.wire([(150, yc + 8), (285, yc + 8)], label=f"A{k}", at=(160, yc + 22))
        f.dot(262, yc - 6)
        f.wire([(262, yc - 6), (262, yc - 28), (391, yc - 28)], arrow=None)
        f.raw(f'<circle cx="400" cy="{yc-28}" r="9" class="xor"/><path d="M391,{yc-28} L409,{yc-28} M400,{yc-37} L400,{yc-19}" class="w"/>')
        f.dot(300, yc + 32)
        f.wire([(300, yc + 32), (400, yc + 32), (400, yc - 19)], arrow=None)
        f.wire([(409, yc - 28), (440, yc - 28)])
        f.text(446, yc - 24, f"S{k}", "ttl")
        top = yc - 16
        nxt = yc - 80 + 16 if k < 2 else 22
        f.wire([(300, top), (300, nxt)], label=("Cout" if k == 2 else f"C{k+1}"), at=(306, top - 18) if k < 2 else (306, 30))
    f.wire([(300, 276), (300, 236)], label="Cin", at=(306, 272))
    f.text(150, 282, "Cout = P ? Cin : A    S = P ⊕ Cin", "lbl", "middle")
    return f


def bar_chart():
    rows = [("readout_counter", 635, 589, False), ("lfsr lane 0 x2", 598, 0, False),
            ("top: command FSM", 529, 63, False), ("pbit_array control", 184, 41, True),
            ("schedule_ctrl", 82, 54, False), ("uart_rx + uart_tx", 75, 39, False),
            ("pbit", 61, 78, True), ("noise_bank x2", 40, 0, False),
            ("weight_mac", 28, 28, True), ("coupling_row", 26, 5, True), ("sigmoid_lut", 1, 0, True)]
    rows.sort(key=lambda r: -(r[1] + r[2]))
    W, lx, x0, bh, gap = 1000, 10, 190, 16, 9
    mx = 1300
    scale = (W - x0 - 90) / mx
    H = 40 + len(rows) * (bh + gap) + 30
    f = Fig("bc", W, H, "Logic cells by block in the dense N=16 build: readout counters and the two lane-0 LFSRs use more logic than the whole sampling core")
    y = 30
    for t in (0, 400, 800, 1200):
        xx = x0 + t * scale
        f.raw(f'<path d="M{xx:.1f},22 L{xx:.1f},{H-26}" class="grid"/>')
        f.text(xx, H - 10, f"{t:,}", "axn", "middle")
    for name, lut, alu, core in rows:
        tot = lut + alu
        w = max(tot * scale, 2)
        cls = "bar core" if core else "bar"
        f.text(x0 - 10, y + 12, name, "axn", "end")
        r = min(4, w / 2)
        f.raw(f'<path d="M{x0},{y} L{x0+w-r:.1f},{y} Q{x0+w:.1f},{y} {x0+w:.1f},{y+r} L{x0+w:.1f},{y+bh-r} '
              f'Q{x0+w:.1f},{y+bh} {x0+w-r:.1f},{y+bh} L{x0},{y+bh} Z" class="{cls}">'
              f'<title>{E(name)}: {lut} LUT4 + {alu} ALU = {tot} logic cells</title></path>')
        f.text(x0 + w + 6, y + 12, f"{tot:,}", "axn")
        y += bh + gap
    f.text(x0, 14, "logic cells = LUT4 + ALU (Yosys synth_gowin, hierarchy kept)", "lbl")
    return f


# ----------------------------------------------------------------- content
CSS = r"""
:root{
  /* layout: sticky layer index left, one reading column; schematics span the column */
  --paper:#f1f4f7; --ink:#141c25; --muted:#556170; --rule:#cfd6df; --signal:#1b5fc9; --heat:#a85f00;
  --panel:color-mix(in srgb,#ffffff 72%,var(--paper));
  --mem:color-mix(in srgb,var(--signal) 9%,var(--panel));
  --font-display:"IBM Plex Sans Condensed","Arial Narrow",sans-serif;
  --font-body:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;
  --font-mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
  color-scheme:light;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#0e1318; --ink:#e1e7ee; --muted:#95a2b1; --rule:#2a3441; --signal:#7eabff; --heat:#f0a640;
  --panel:color-mix(in srgb,#ffffff 4%,var(--paper)); color-scheme:dark;}}
:root[data-theme="dark"]{
  --paper:#0e1318; --ink:#e1e7ee; --muted:#95a2b1; --rule:#2a3441; --signal:#7eabff; --heat:#f0a640;
  --panel:color-mix(in srgb,#ffffff 4%,var(--paper)); color-scheme:dark;}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 var(--font-body)}
a{color:var(--signal)}
a:focus-visible,input:focus-visible{outline:2px solid var(--signal);outline-offset:2px}
code,.mono{font-family:var(--font-mono);font-size:.88em}
.wrap{display:grid;grid-template-columns:210px minmax(0,1fr);gap:44px;max-width:1380px;margin:0 auto;padding:0 24px}
nav.idx{position:sticky;top:0;align-self:start;max-height:100vh;overflow:auto;padding:30px 0 30px;font-size:.88rem}
nav.idx ol{list-style:none;margin:0;padding:0;display:grid;gap:2px}
nav.idx a{display:grid;grid-template-columns:30px 1fr;gap:6px;text-decoration:none;color:var(--ink);padding:3px 6px;border-radius:4px}
nav.idx a:hover{background:var(--panel)}
nav.idx .lv{font-family:var(--font-mono);font-size:.78rem;color:var(--muted);padding-top:2px}
nav.idx .sub a{padding-left:42px;grid-template-columns:1fr;color:var(--muted)}
nav.idx h2{font:600 .78rem var(--font-mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 10px 6px}
main{min-width:0;padding-bottom:80px}
header.top{padding:44px 0 28px}
.eyebrow{font:500 .82rem var(--font-mono);color:var(--muted);letter-spacing:.04em;margin:0 0 8px}
h1{font:600 2.5rem/1.1 var(--font-display);margin:0 0 14px;text-wrap:balance}
h2{font:600 1.65rem/1.2 var(--font-display);margin:0 0 12px;text-wrap:balance;display:flex;gap:12px;align-items:baseline}
h2 .lv{font:500 .85rem var(--font-mono);color:var(--paper);background:var(--ink);padding:2px 7px;border-radius:3px}
h3{font:600 1.15rem/1.3 var(--font-display);margin:26px 0 8px}
p,ul.prose,ol.prose{max-width:72ch}
.lede{font-size:1.08rem;max-width:74ch}
section{padding:40px 0 18px;border-top:1px solid var(--rule)}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:1px;background:var(--rule);border:1px solid var(--rule);border-radius:6px;overflow:hidden;margin:22px 0 0}
.facts div{background:var(--panel);padding:12px 14px}
.facts dt{font-size:.78rem;color:var(--muted)}
.facts dd{margin:2px 0 0;font:600 1.05rem var(--font-display);font-variant-numeric:tabular-nums}
.eq{font-family:var(--font-mono);font-size:.92rem;background:var(--panel);border-left:3px solid var(--signal);padding:10px 14px;max-width:72ch;margin:14px 0}
.ladder{list-style:none;padding:0;margin:14px 0;display:grid;gap:6px;max-width:980px}
.ladder li{display:grid;grid-template-columns:52px 200px 1fr;gap:14px;align-items:baseline;background:var(--panel);border:1px solid var(--rule);border-radius:5px;padding:9px 12px}
.ladder .lv{font:600 .9rem var(--font-mono)}
.ladder a{font-weight:600;text-decoration:none}
.ladder span:last-child{color:var(--muted);font-size:.92rem}
.fig{margin:18px 0;background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:14px 14px 10px}
.fig .scroll{overflow-x:auto}
.fig.small .dia{min-width:0}
figcaption{font-size:.88rem;color:var(--muted);max-width:86ch;margin-top:8px}
.dia{display:block;width:100%;min-width:900px;height:auto;color:var(--ink);font-family:var(--font-mono)}
.dia .bx{fill:var(--panel);stroke:currentColor;stroke-width:1.1}
.dia .bx.mem{fill:var(--mem)}
.dia .bx.grp{fill:none;stroke:var(--muted);stroke-dasharray:5 4}
.dia .bx.pin{fill:var(--paper);stroke:var(--muted)}
.dia .bx.nzb{stroke:var(--heat);stroke-width:1.6}
.dia .bx.hl{stroke:var(--signal);stroke-width:2}
.dia .bx.cell{fill:var(--panel);stroke-width:1}
.dia .t{fill:currentColor;font-size:11px}
.dia .t.ttl{font-size:12.5px;font-weight:600}
.dia .t.sub{font-size:10.5px;fill:var(--muted)}
.dia .t.tag{font-size:8.5px;letter-spacing:.06em;fill:var(--muted)}
.dia .t.lbl{font-size:10.5px;fill:var(--muted)}
.dia .t.lbln{fill:var(--heat)}
.dia .t.wnum{font-size:9.5px}
.dia .t.axn{font-size:10.5px;fill:var(--muted)}
.dia .t.seglbl{font-size:10.5px}
.dia .w{fill:none;stroke:currentColor;stroke-width:1.3;stroke-linejoin:round}
.dia .w.bus{stroke-width:2.6}
.dia .w.ctl{stroke:var(--muted);stroke-dasharray:4 3;stroke-width:1.1}
.dia .w.nz{stroke:var(--heat);stroke-width:1.9}
.dia .w.nz.ctl{stroke:var(--heat)}
.dia .w.acc{stroke:var(--signal);stroke-width:1.8}
.dia .w.wave{stroke-width:2}
.dia .ah{fill:currentColor}.dia .ahm{fill:var(--muted)}.dia .ahn{fill:var(--heat)}.dia .ahc{fill:var(--signal)}
.dia .dot{fill:currentColor}
.dia .clk,.dia .wslash{fill:none;stroke:currentColor;stroke-width:1.1}
.dia .gate{fill:var(--panel);stroke:currentColor;stroke-width:1.3}
.dia .gate2{stroke:currentColor;stroke-width:2.2}
.dia .xor{fill:var(--panel);stroke:currentColor;stroke-width:1.2}
.dia .seg{fill:var(--mem);stroke:none}
.dia .seghl{fill:color-mix(in srgb,var(--signal) 30%,var(--panel))}
.dia .segheat{fill:color-mix(in srgb,var(--heat) 32%,var(--panel))}
.dia .tick,.dia .axis{stroke:var(--muted);stroke-width:1;fill:none}
.dia .grid{stroke:var(--rule);stroke-width:1}
.dia .bar{fill:color-mix(in srgb,var(--muted) 55%,var(--panel))}
.dia .bar.core{fill:var(--signal)}
.dia .bar:hover{opacity:.8}
.tgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,470px),1fr));gap:14px}
.tgrid .fig{margin:0}
.tscroll{overflow-x:auto;margin:14px 0;border:1px solid var(--rule);border-radius:6px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:.86rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--rule);vertical-align:top}
th{font-weight:600;font-size:.78rem;color:var(--muted);background:var(--paper);position:sticky;top:0}
td.n,th.n{text-align:right}
td code{white-space:nowrap}
tr.grp td{background:var(--paper);font:600 .8rem var(--font-mono);color:var(--muted);letter-spacing:.04em}
.meter{display:inline-block;width:70px;height:7px;background:color-mix(in srgb,var(--signal) 14%,var(--panel));border-radius:4px;vertical-align:middle;margin-left:6px;overflow:hidden}
.meter i{display:block;height:100%;background:var(--signal);border-radius:4px}
.tree,.tree ul{list-style:none;margin:0;padding-left:18px;font-family:var(--font-mono);font-size:.84rem}
.tree{padding-left:0}
.tree li{position:relative;padding:2px 0 2px 14px;border-left:1px solid var(--rule)}
.tree li:last-child{border-left-color:transparent}
.tree li::before{content:"";position:absolute;left:0;top:0;width:10px;height:13px;border-left:1px solid var(--rule);border-bottom:1px solid var(--rule)}
.tree li:last-child::before{border-left-color:var(--rule)}
.tree .m{font-weight:600}
.tree .p{color:var(--muted)}
.chip{display:inline-block;font:500 .7rem var(--font-mono);padding:0 5px;border:1px solid var(--rule);border-radius:3px;color:var(--muted);margin-left:4px}
.trees{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:18px 34px;margin:14px 0}
.trees > div{background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:14px}
.trees h3{margin:0 0 8px}
.finding{border-left:3px solid var(--heat);background:var(--panel);padding:10px 14px;margin:12px 0;max-width:86ch}
.finding strong{display:block;font-family:var(--font-display);font-size:1.02rem;margin-bottom:2px}
.filter{display:flex;gap:10px;align-items:center;margin:12px 0}
.filter input{font:inherit;padding:7px 10px;border:1px solid var(--rule);border-radius:5px;background:var(--panel);color:var(--ink);width:min(100%,380px)}
.filter span{color:var(--muted);font-size:.85rem}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:14px}
@media (max-width:980px){.wrap{grid-template-columns:1fr;gap:0}nav.idx{position:static;max-height:none;padding:20px 0 0}
  nav.idx ol{grid-template-columns:repeat(auto-fill,minmax(180px,1fr))}nav.idx .sub a{padding-left:6px}}
@media (max-width:560px){.wrap{padding:0 16px}h1{font-size:1.9rem}.ladder li{grid-template-columns:40px 1fr}.ladder span:last-child{grid-column:1/-1}}
"""

NAV = [
    ("", "overview", "Overview"), ("", "layers", "Layer ladder"), ("", "hierarchy", "Module tree"),
    ("L6", "host", "Host link"), ("L5", "system", "Board top"),
    ("L4", "dense", "Dense engine"), ("", "sparse", "Sparse lanes"), ("", "tempering", "Tempering"),
    ("", "pmode", "Langevin p-modes"), ("", "wrappers", "pgate, denoise"),
    ("L3", "pbit", "p-bit decision"), ("", "noise", "Noise lanes"), ("", "anneal-readout", "Anneal, readout"),
    ("L2", "formats", "Number formats"), ("", "timing", "Cycle budgets"),
    ("L1", "fabric", "FPGA fabric"), ("L0", "transistors", "Transistors"),
    ("", "signals", "Signal dictionary"), ("", "findings", "Findings"), ("", "provenance", "Provenance"),
]


def nav_html():
    out = ['<nav class="idx" aria-label="Sections"><h2>Map index</h2><ol>']
    for lv, hid, name in NAV:
        sub = ' class="sub"' if (lv == "" and hid not in ("overview", "layers", "hierarchy", "signals", "findings", "provenance")) else ""
        if sub:
            out.append(f'<li{sub}><a href="#{hid}">{E(name)}</a></li>')
        else:
            out.append(f'<li><a href="#{hid}"><span class="lv">{lv}</span><span>{E(name)}</span></a></li>')
    out.append("</ol></nav>")
    return "".join(out)


def table(headers, rows, num_cols=()):
    h = "".join(f'<th class="n">{E(x)}</th>' if i in num_cols else f"<th>{E(x)}</th>" for i, x in enumerate(headers))
    body = []
    for r in rows:
        if isinstance(r, str):
            body.append(f'<tr class="grp"><td colspan="{len(headers)}">{E(r)}</td></tr>')
            continue
        body.append("<tr>" + "".join(f'<td class="n">{c}</td>' if i in num_cols else f"<td>{c}</td>" for i, c in enumerate(r)) + "</tr>")
    return f'<div class="tscroll"><table><thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def c(x):
    return f"<code>{E(x)}</code>"


# host protocol: byte -> wire
HOST_ROWS = [
    "Dense and both builds",
    [c("'P'"), "", "reply N, HIST_BITS, 'K' (sparse: N[7:0], HIST_BITS|0x80, N[15:8], P, D, LW)", "S_PING2, S_PING3 send engine"],
    [c("'W' a_hi a_lo d"), "3", c("args[2IW+7:8]") + " → w_addr, " + c("args[7:0]") + " → w_data, w_we pulse", "pbit_array wmem (BSRAM)"],
    [c("'H' i d"), "2", c("args[IW+7:8]") + " → h_addr, " + c("args[7:0]") + " → h_data, h_we", "pbit_array hmem"],
    [c("'O' k i"), "2", c("args[IW+7:8]") + " → o_addr, " + c("args[IW-1:0]") + " → o_data, o_we", "pbit_array omem"],
    [c("'T' k b"), "2", c("args[11:8]") + " → tbl_addr, " + c("args[7:0]") + " → tbl_data, t_we", "schedule_ctrl table"],
    [c("'B' mode b0 b1 dl s_hi s_lo"), "6", c("[41:40]") + " mode, " + c("[39:32]") + " β0, " + c("[31:24]") + " β1, " + c("[23:16]") + " Δ, " + c("[15:0]") + " sweeps/step; restart one cycle later (S_BRST)", "mode/b0/b1/dl/sps_reg → schedule_ctrl"],
    [c("'Z'"), "", "sch_restart, pt_resync", "schedule_ctrl, temper_ctrl"],
    [c("'X' s3 s2 s1 s0"), "4", c("args[31:0]") + " → seed_in; seed_load + spin_we (spins ← all −1); tempering reset_all", "every LFSR (lane k: seed ⊕ 0x9E3779B9·(k+1)), engine"],
    [c("'R' n3..n0"), "4", "tot_reg ← n; arr_run pulse; S_RUNW waits for !busy and !ro_busy", "engine n_sweeps; each swd recorded"],
    [c("'S' n3..n0"), "4", "as 'R' with hold_reg = 1: engine pauses after each sweep while ⌈N/8⌉ bytes go out, hold drops for one cycle", "engine hold, uart_tx"],
    [c("'N' n3..n0"), "4", "raw_en per draw, 2 bytes per draw (LE)", "raw noise_bank"],
    [c("'C'"), "", "ro_clear (hist clear = 2^HIST_BITS cycles)", "readout_counter"],
    [c("'D'"), "", "hist_addr = cnt 0 … 2^HB−1, 4 bytes each", "readout_counter hist_q"],
    [c("'U'"), "", "n_samples, then up_sel = 0 … N−1, 4 bytes each", "readout_counter up_q"],
    [c("'Q'"), "", "spins, ⌈N/8⌉ bytes", "engine spins"],
    "SPARSE = 1",
    [c("'E' l r s j d"), "5", c("[33:32]") + " lane, {" + c("[LW+23:24]") + ", " + c("[DW+15:16]") + "} → n_addr {round, slot}, {" + c("[IW+7:8]") + ", " + c("[7:0]") + "} → {j, J}", "sparse_array wmem[lane]"],
    [c("'M' l r f i d"), "5", c("[33:32]") + " lane, " + c("[LW+23:24]") + " → m_addr, {" + c("[16]") + ", " + c("[IW+7:8]") + ", " + c("[7:0]") + "} → {valid, node, h}", "sparse_array nmem[lane]"],
    [c("'G' r_hi r_lo"), "2", c("args[LW:0]") + " → r_data, g_we", "rounds_reg"],
    [c("'Y' f q_hi q_lo"), "3", c("[17:16]") + " → {nz_share, nz_tau}, " + c("[15:0]") + " → nz_q", "sparse noise options (NOISE_CTL = 1)"],
    "TEMPER_K > 1",
    [c("'L' k b"), "2", c("args[9:8]") + " → ladder_addr, " + c("args[7:0]") + " → ladder_data, l_we", "temper_ctrl ladder"],
]

FORMAT_ROWS = [
    [c("J_ij, h_i"), "8 signed", "Q3.4 (WFRAC = 4)", "−8 … +7.9375, step 1/16", "wmem, hmem, sparse nbr/node words"],
    [c("field"), "AW = 14 signed", "Q9.4", "|h + ΣJ| ≤ (N+1)·128 LSB", "weight_mac acc → pbit"],
    [c("β"), "8 unsigned", "Q4.4", "0 … 15.94", "schedule_ctrl, ladder"],
    [c("β·field"), "AW+9 = 23 signed", "8 fraction bits", "", "pbit prod_reg"],
    [c("logit"), "10 signed", "Q5.5", "2β·field clamped to −16 … +15.97", "pbit logit_reg"],
    [c("p"), "16 unsigned", "p / 65536", "σ(logit/32) rounded, ≤ 65535", "sigmoid ROM output"],
    [c("rnd"), "16 unsigned", "uniform integer", "0 … 65535", "LFSR draw"],
    [c("e[k]"), "24 signed", "E/2 in weight units (Q.4)", "relative to the last resync", "temper_ctrl"],
    [c("clamp bias"), "8", "0x7F / 0x80", "+7.94 / −8.0", "pgate bias rewrite"],
    [c("Γ_t / 2"), "8", "Q3.4", "per step t", "denoise_demo gamma table"],
    [c("x (p-mode)"), "16 signed", "Q4.12", "±8, saturated", "pmode xmem"],
    [c("s = tanh x"), "17 signed", "Q.15", "±1 (±32768)", "pmode smem"],
    [c("A, h (p-mode)"), "8 signed", "Q3.4", "", "pmode amem, hmem"],
    [c("acc (p-mode)"), "40 signed", "Q.19", "h·2^15 + ΣA·s", "pmode acc_reg"],
    [c("λ"), "8 unsigned", "Q4.4", "default 32 = 2.0", "pmode lam"],
    [c("σ = √(2Tε)"), "16 unsigned", "Q4.12", "default 1448 = 0.3535 (T = 1, ε = 1/16)", "pmode sig"],
    [c("ε"), "4", "2^−kshift", "default 1/16", "pmode kshift"],
]

TIMING_ROWS = [
    ["p-bit decision (pbit)", "12", "444 ns", "start → done, one engine"],
    ["Dense update (pbit_array)", "N + 18", "34 at N = 16", "ORD 1, BIAS 1, MAC 1, MACW N+2, PBW 12, WR 1"],
    ["Dense sweep", "N·(N + 18)", "544 at N = 16 → 49.6 k sweeps/s; 18,688 at N = 128 → 1.44 k/s", "one shared engine"],
    ["Sparse round (P lanes)", "D + P + 15", "23 at P = 4, D = 4", "updates P nodes of one colour block"],
    ["Sparse sweep", "ROUNDS·(D + P + 15)", "92 at N = 16 → 293 k/s; 1,472 at N = 256 → 18.3 k/s", "4.7 M p-bit updates/s at P = 4, D = 4"],
    ["Tempering round, K = 4, N = 16", "≈ 4·547 + 16 or 30", "≈ 2,210 → ≈ 12 k rounds/s", "traced from the FSMs, not measured"],
    ["p-mode step", "2M + M(2M + 5) + 1", "61 at M = 4 → 443 k steps/s", "pmode_top paces one step per 2^20 clocks = 25.7 steps/s"],
    ["UART byte", "10 × 234", "2,340 → 11.5 k bytes/s", "'S' at N = 16 sends 2 bytes per sweep, so the link caps it near 5.7 k sweeps/s"],
    ["Histogram sample", "3", "", "read, add, write"],
    ["Histogram clear", "2^HIST_BITS", "4,096", ""],
    ["UP_RAM counter scan", "N + 1", "17 at N = 16", "per sample"],
    ["Bias rewrite (pgate, denoise)", "N", "", "per clamp change / per step"],
]

PNR = {  # nextpnr-himbaechel GW1NR-LV9QN88PC6/I5, 27 MHz target, seed 1
    "dense16": dict(lut=2699, alu=978, ff=1326, bsram=10, ssram=7, mult=0, fmax=72.52, crit="sps_reg → schedule cnt+1 < sps compare → beta_reg clock enable", logic=5.34, route=8.45),
    "temper4": dict(lut=3624, alu=1228, ff=1611, bsram=11, ssram=9, mult=0, fmax=54.88, crit="a_reg → rep_at → e[rx] − e[ry] → saturate → sf_reg", logic=10.22, route=8.00),
    "sparse16": dict(lut=3897, alu=1444, ff=1901, bsram=12, ssram=32, mult=0, fmax=70.36, crit="sps_reg → schedule compare → beta_reg clock enable", logic=5.21, route=9.00),
    "sparse256": dict(lut=7009, alu=870, ff=2463, bsram=17, ssram=84, mult=0, fmax=69.61, crit="args_reg → command decode (args == 0 test) → state_reg", logic=4.90, route=9.46),
    "pmode4": dict(lut=854, alu=364, ff=548, bsram=2, ssram=10, mult=5, fmax=64.96, crit="i_reg → xmem read mux → x·(x²−1) multiply → q_reg", logic=8.62, route=6.78),
}
BUILD_DESC = {
    "dense16": "top, N = 16, HIST_BITS = 12",
    "temper4": "top, N = 16, TEMPER_K = 4",
    "sparse16": "top, SPARSE, N = 16, P = 4, D = 4, NOISE_CTL",
    "sparse256": "top, SPARSE, N = 256, P = 4, D = 4, UP_RAM",
    "pmode4": "pmode_top, M = 4",
}


def meter(v, cap):
    pct = 100 * v / cap
    return f'{v:,} <span class="mono" style="color:var(--muted)">{pct:.0f}%</span><span class="meter" aria-hidden="true"><i style="width:{min(pct,100):.1f}%"></i></span>'


def pnr_table():
    hdr = ["Build", "Configuration", "LUT4 (of 8,640)", "ALU", "Logic sites", "DFF (of 6,480)", "BSRAM (of 26)", "SSRAM", "MULT18 (of 20)", "Fmax"]
    rows = []
    for k, d in PNR.items():
        sites = d["lut"] + d["alu"]
        rows.append([c(k), E(BUILD_DESC[k]), meter(d["lut"], 8640), f'{d["alu"]:,}', meter(sites, 8640),
                     meter(d["ff"], 6480), meter(d["bsram"], 26), str(d["ssram"]), str(d["mult"]), f'<span style="white-space:nowrap">{d["fmax"]:.1f} MHz</span>'])
    return table(hdr, rows, num_cols=(3, 7, 8, 9))


def crit_table():
    rows = []
    for k, d in PNR.items():
        rows.append([c(k), f'{d["fmax"]:.1f} MHz', f'{1000/d["fmax"]:.2f} ns', f'{d["logic"]:.2f} / {d["route"]:.2f}', E(d["crit"])])
    return table(["Build", "Fmax", "Period", "Logic / routing ns", "Critical path"], rows, num_cols=(1, 2, 3))


YOSYS_ROWS = [
    "dense16 build, per instance",
    [c("top"), "command FSM, buttons, reply shifter", "529", "63", "199", "249", "", ""],
    [c("readout_counter"), "16 × 32-bit up counters in DFF, hist RMW", "635", "589", "183", "681", "8", ""],
    [c("lfsr") + " lane 0 × 2", "array lane + raw 'N' lane", "598", "0", "372", "64", "", ""],
    [c("schedule_ctrl"), "β modes, 16-bit sweep counter", "82", "54", "27", "28", "", "4"],
    [c("uart_rx") + " + " + c("uart_tx"), "", "75", "39", "8", "62", "", ""],
    [c("pbit_array"), "control, spin register, RAM ports", "184", "41", "48", "98", "1", "3"],
    [c("coupling_row"), "column counter, valid pipe", "26", "5", "11", "16", "", ""],
    [c("weight_mac"), "14-bit add/sub accumulator", "28", "28", "13", "14", "", ""],
    [c("pbit"), "shift-add ×, round, sat, compare", "61", "78", "5", "72", "", ""],
    [c("sigmoid_lut"), "1024 × 16 ROM", "1", "0", "0", "0", "1", ""],
    [c("noise_bank") + " × 2", "seed XOR glue", "40", "0", "0", "0", "", ""],
    "Variants",
    [c("sigmoid_pwl"), "USE_PWL = 1 instead of the ROM", "49", "43", "11", "16", "0", ""],
    [c("lfsr") + " lane 1 (A3000000)", "4 taps, all in the top byte", "62", "0", "9", "32", "", ""],
    [c("lfsr") + " lane 2 (80200003)", "4 taps", "154", "0", "90", "32", "", ""],
    [c("lfsr") + " lane 3 (80000057)", "6 taps", "135", "0", "61", "32", "", ""],
    [c("temper_ctrl") + " K = 4", "4 banks, 4 × 24-bit energies, swap logic", "754", "183", "137", "249", "", "6"],
    [c("sparse_array") + " N = 16, P = 4", "lanes, commit mux", "321", "104", "84", "167", "", "28"],
    [c("sparse_array") + " N = 256, P = 4", "256:1 spin read mux per lane, 256-way write", "3,686", "56", "2,246", "427", "4", "80"],
    [c("readout_counter") + " UP_RAM = 1, N = 16", "counters in one BSRAM, N+1 scan", "195", "114", "", "237", "9", ""],
    [c("pmode_array") + " M = 4", "MAC, gradient pipeline", "1,166", "403", "640", "429", "", "16"],
]


def sig_rows():
    S = [
        ("top.v", [
            ("clk", "1", "pin 52, 27 MHz crystal", "every DFF and RAM", "Single clock domain, no PLL"),
            ("por_reg", "4", "top", "rst", "Power-on reset: counts to 15, rst held meanwhile"),
            ("rst", "1", "POR or S1 (2-FF synced)", "all modules", "Synchronous reset"),
            ("run_stable, deb_reg", "1, 18", "S2 after 2-FF sync", "btn_run_evt", "2^18-cycle (9.7 ms) debounce"),
            ("btn_run_evt", "1", "debounce", "command FSM", "Press: silent 'R' of DEMO_SWEEPS (128) after a schedule restart"),
            ("rx_data, rx_valid", "8, 1", "uart_rx", "command FSM", "Received byte and one-cycle strobe, no backpressure"),
            ("tx_data_reg, tx_valid_reg", "8, 1", "command FSM", "uart_tx", "Byte to send"),
            ("tx_ready", "1", "uart_tx", "command FSM", "Transmitter idle"),
            ("cmd_reg, nargs_reg, aidx_reg", "8, 3, 3", "command FSM", "command FSM", "Command byte, argument count, bytes received"),
            ("args_reg", "48", "command FSM", "every programming port", "Argument shift register; last byte in [7:0]"),
            ("word_reg, nby_reg", "WW, 6", "command FSM", "tx_data", "Reply word, shifted out LSB first; WW = ⌈N/32⌉·32"),
            ("ret_reg", "5", "command FSM", "command FSM", "Return state after S_SEND"),
            ("cnt_reg, tot_reg", "32, 32", "command FSM", "hist_addr, up_sel, arr_nsw", "Loop counter and total"),
            ("w_we, h_we, o_we", "1", "command FSM", "pbit_array", "Weight, bias, order RAM writes"),
            ("e_we, m_we, g_we", "1", "command FSM", "sparse_array", "Neighbour word, node word, rounds per sweep"),
            ("t_we / l_we", "1", "command FSM", "schedule_ctrl / temper_ctrl", "β table / ladder writes"),
            ("sch_restart, pt_resync", "1", "command FSM", "schedule_ctrl, temper_ctrl", "Reload β schedule; resync replicas"),
            ("seed_load", "1", "command FSM", "every LFSR, temper reset_all", "'X': reseed all lanes"),
            ("spin_we", "1", "command FSM", "engine", "'X': spins ← 0 (all −1)"),
            ("arr_run, arr_nsw", "1, 32", "command FSM", "engine run, n_sweeps", "Start pulse and sweep count"),
            ("hold_reg, arr_stop", "1, 1", "command FSM, constant 0", "engine hold, stop", "Pause at sweep boundary ('S'); stop unused"),
            ("mode/b0/b1/dl/sps_reg", "2/8/8/8/16", "'B'", "schedule_ctrl cfg_*", "Annealing configuration"),
            ("nzf_reg, nzq_reg", "2, 16", "'Y'", "sparse nz_tau, nz_share, nz_q", "Noise options"),
            ("beta", "8", "schedule_ctrl", "engine", "Inverse temperature, Q4.4"),
            ("arr_busy", "1", "engine", "command FSM, led[5]", "Engine running"),
            ("swd", "1", "engine", "schedule_ctrl, readout, FSM", "One pulse per sweep or round; spins already final"),
            ("spins", "N", "engine", "readout, FSM, led[4:0]", "Bit i = node i, 1 = +1"),
            ("raw_en, raw_rnd", "1, 16", "command FSM / raw noise_bank", "raw noise_bank / FSM", "'N' stream"),
            ("ro_clear, hist_addr, up_sel", "1, HB, IW", "command FSM", "readout_counter", "Clear and read addresses"),
            ("hist_q, up_q, n_samples, ro_busy", "32, 32, 32, 1", "readout_counter", "command FSM", "Counts and busy"),
            ("led[5:0]", "6", "top", "pins 10, 11, 13, 14, 15, 16", "Active low ~{busy, spins[4:0]}"),
        ]),
        ("pbit_array.v", [
            ("k_reg", "IW", "control FSM", "omem address", "Position in the visit (colour) order"),
            ("o_q", "IW", "omem", "hmem address, node_reg", "Node at position k"),
            ("node_reg", "IW", "o_q", "coupling_row row, spin write index", "Node being updated"),
            ("h_q", "WB", "hmem", "coupling_row h_in", "Bias, loaded as accumulator start"),
            ("mac_addr", "2·IW", "coupling_row", "wmem address", "{row, j}"),
            ("w_q", "WB", "wmem", "weight_mac w", "J[row][j], one-cycle BSRAM latency"),
            ("field", "AW", "weight_mac", "pbit field, upd_field", "h_i + Σ_j ±J_ij"),
            ("mac_start, mac_done", "1", "FSM / coupling_row", "", "N+2-cycle handshake"),
            ("pb_start, pb_done, pb_bit", "1", "FSM / pbit", "", "12-cycle handshake and new spin"),
            ("pb_take", "1", "pbit", "noise_bank en", "One draw consumed per update"),
            ("rnd", "16", "noise_bank lane 0", "pbit", "Uniform draw"),
            ("spins_reg", "N", "FSM", "spins, s mux", "Gibbs chain state"),
            ("swd_reg", "1", "FSM", "sweep_done", "Registered with the last spin write"),
            ("upd_we, upd_old, upd_new, upd_field", "1, 1, 1, AW", "pbit_array", "temper_ctrl", "Update tap for incremental energies"),
        ]),
        ("coupling_row.v, weight_mac.v", [
            ("j_reg", "IW+1", "coupling_row", "w_addr low bits", "Next column to issue"),
            ("vld_reg, jd_reg", "1, IW", "coupling_row", "add_en, s = spins[jd]", "Column whose word is on w_data"),
            ("row_reg", "IW", "row", "w_addr high bits", "Latched at start"),
            ("acc_reg", "AW", "weight_mac", "field", "load: acc ← h; add_en: acc ← acc ± w"),
        ]),
        ("pbit.v", [
            ("mcand_reg", "AW+9", "sign-extended field", "adder", "Shifted left once per MUL cycle"),
            ("bsh_reg", "8", "beta", "adder enable", "Shifted right; bit 0 selects add"),
            ("cnt_reg", "3", "pbit", "FSM", "Counts the 8 MUL cycles"),
            ("prod_reg", "AW+9", "adder", "rounded", "β·field, WFRAC + 4 fraction bits"),
            ("rounded", "AW+9", "prod_reg", "saturation", "(prod + 2^(S−1)) >>> S, S = WFRAC − 2"),
            ("logit_reg", "10", "saturation", "sigmoid ROM / PWL", "2β·field, Q5.5, clamped to [−512, 511]"),
            ("lut_en", "1", "S_LUT", "sigmoid en", "One-cycle ROM read"),
            ("p_w", "16", "sigmoid", "comparator", "P(+1) × 65536"),
            ("bit_reg, done_reg, take_reg", "1", "S_CMP", "bit_out, done, rnd_take", "bit = rnd < p; done and take pulse together"),
        ]),
        ("sigmoid_lut.v, sigmoid_pwl.v", [
            ("addr", "10", "{~logit[9], logit[8:0]}", "ROM", "logit + 512 (offset binary)"),
            ("p_reg", "16", "ROM", "p", "Registered read while en"),
            ("a, y", "10, 17", "|logit|, segment sum", "p_reg", "PWL: 4 shift-and-add segments, p(−x) = 1 − p(x)"),
        ]),
        ("lfsr.v, noise_bank.v", [
            ("state_reg", "32", "next-state mux", "leap network", "Galois LFSR state, reset = lane seed"),
            ("leap_w", "48", "leap network", "rnd, state_next", "16 output bits and the state 16 steps ahead"),
            ("en", "LANES", "consumer rnd_take", "lfsr en", "Advance a lane by 16 steps"),
            ("jitter_bit", "1", "ring oscillator, 2-FF sync", "rnd[0] XOR", "RO_JITTER = 1 only"),
        ]),
        ("sparse_array.v", [
            ("ptr_reg", "LW", "FSM", "nmem, wmem address", "Round in the sweep"),
            ("slot_reg", "DW+1", "FSM", "wmem address low bits", "Neighbour slot issued"),
            ("vld_reg", "1", "FSM", "add_en of every lane", "Neighbour word valid"),
            ("nq", "IW+WB+1 per lane", "nmem", "h, valid, node", "Node word of the round"),
            ("wq", "IW+WB per lane", "wmem", "j, J", "Neighbour word"),
            ("lane_valid, lane_node, lane_bit", "P, P·IW, P", "lanes", "commit mux", ""),
            ("cidx_reg", "2", "FSM", "commit mux select", "Lane committed this cycle"),
            ("keep_reg", "P", "S_PB: draw ≥ nz_q", "write mask, noise en", "nz_tau: lane keeps its spin"),
            ("nz_en", "P", "lanes", "noise_bank en", "Lanes that advance"),
            ("rnd, lane_rnd", "P·16", "noise_bank, share mux", "pbit rnd", "Own lane, or lane 0 with nz_share"),
            ("rounds_reg", "LW+1", "'G'", "last_round", "Rounds per sweep"),
        ]),
        ("temper_ctrl.v", [
            ("bank[k]", "N each", "a_spins at a_swd", "a_spin_data, spins_out", "Replica spin states"),
            ("e[k]", "24 each", "update tap", "Δe subtractor", "E/2 per replica, incremental"),
            ("rep_at[r], rung[k]", "2 each", "swap accept", "spins_out, β select", "Rung → replica, replica → rung"),
            ("ladder[r]", "8 each", "'L', LADDER", "beta_reg, Δβ", "β per rung, rung 0 = target"),
            ("k_reg, a_reg, par_reg", "2, 3, 1", "FSM", "", "Replica swept, lower rung of pair, round parity"),
            ("beta_reg", "8", "ladder[rung[k]]", "a_beta", "β of the replica being swept"),
            ("sf_reg, sb_reg", "AW, 8", "sign fold, saturate", "swap pbit field, beta", "±Δe and |Δβ|"),
            ("sw_go, sw_done, sw_bit", "1", "FSM / swap pbit", "", "Swap decision, 1 = accept"),
            ("sw_rnd, sw_take", "16, 1", "swap LFSR (lane-1 mask, seed)", "swap pbit", "Separate noise stream"),
            ("a_spin_we, a_run, a_swd, a_busy", "1", "temper_ctrl ↔ pbit_array", "", "Load replica, run one sweep, wait"),
        ]),
        ("schedule_ctrl.v", [
            ("beta_reg", "8", "mode logic", "beta", "Q4.4"),
            ("cnt_reg", "16", "sweep_done", "step test", "Sweeps since the last β step"),
            ("k_reg", "4", "mode 2", "table address", "Holds at 15"),
            ("table_mem", "16 × 8", "'T', TABLE_FILE", "beta in mode 2", "β table"),
        ]),
        ("readout_counter.v", [
            ("idx_reg", "HB", "spins[HB−1:0] at sample", "hist address", "State being incremented"),
            ("hist_mem", "2^HB × 32", "read-modify-write", "hist_q", "Full-state histogram, BSRAM"),
            ("up_reg[i] / up_mem", "N × 32", "sample", "up_q", "Magnetisation counters (DFF or BSRAM)"),
            ("ns_reg", "32", "sample", "n_samples", "Samples recorded"),
            ("snap_reg, ui_reg, scan_reg, pend_reg", "N, IW+1, 1, 1", "UP_RAM scan", "up_mem", "Snapshot, scan index, pending clear"),
        ]),
        ("uart_rx.v, uart_tx.v", [
            ("s1_reg, s2_reg", "1", "rxd", "receiver FSM", "2-FF synchronizer"),
            ("cnt_reg, bit_reg, data_reg", "16, 3, 8", "receiver FSM", "out_data", "Bit timer, bit index, shift-in"),
            ("shift_reg, bits_reg, cnt_reg", "10, 4, 16", "in_data", "txd", "{stop, data, start} out LSB first"),
        ]),
        ("pmode_array.v, pmode_top.v", [
            ("xmem, xnew", "M × 16", "x_we / G5", "x_flat / CMT", "States x, Q4.12, Jacobi double buffer"),
            ("smem", "M × 17", "S_T2", "MAC, g", "s = tanh x, Q.15"),
            ("amem, hmem", "2^2MW × 8, 2^MW × 8", "a_we, h_we", "MAC", "A and h, Q3.4"),
            ("acc_reg", "40", "MAC", "f_reg", "h·2^15 + ΣA·s, Q.19"),
            ("f, g1, x2, cpl, q, qt, nz _reg", "40 each", "G1 … G4", "xn", "Gradient pipeline registers"),
            ("xi_reg", "16", "gauss ROM", "nz", "Gaussian sample ξ"),
            ("lut_idx", "10", "sat((x+32) >>> 6)", "tanh ROM", "Logit of 2x"),
            ("rnd", "16", "noise_bank", "gauss ROM address rnd[15:6]", "Uniform draw → inverse CDF"),
            ("on_reg, pace_reg, tick", "1, 20, 1", "S2 toggle, counter", "run", "One step every 2^20 clocks"),
        ]),
        ("pgates.v, denoise_demo.v", [
            ("mask_reg, val_reg", "N", "start", "hval", "Latched clamps"),
            ("hval (pgate)", "8", "clamp ? (val ? 0x7F : 0x80) : h0", "pbit_array h_data", "Bias rewrite, N cycles"),
            ("t_reg", "8", "T … 1", "gamma table", "Denoising step"),
            ("xt_reg", "N", "x_t, then spins", "spin_data, clamp sign", "Clamped image x^t"),
            ("hval (denoise)", "8", "sat8(h0 ± g_t)", "pbit_array h_data", "Clamp field"),
        ]),
    ]
    rows = []
    for mod, items in S:
        rows.append(mod)
        for name, w, frm, to, mean in items:
            rows.append([c(name), E(w), E(frm), E(to), E(mean)])
    return rows


def build():
    figs = {
        "legend": figure(fig_legend(), "Encodings used in every schematic below. A crossing without a dot is not a connection.", "small"),
        "system": figure(fig_system(), "top.v as built for the Tang Nano 9K. Every argument byte from the host lands in <code>args_reg</code>; one decoded strobe per command then writes a slice of it into a RAM port. The engine pulses <code>swd</code> once per sweep, which drives the schedule, the readout and the reply logic at the same time. Pin numbers are the board's standard ones, as used in my place-and-route run."),
        "dense": figure(fig_dense(), "One time-multiplexed engine visits every node in the order stored in <code>omem</code> (colour block 0 first). The loop from <code>spins_reg</code> back into the <code>s mux</code> is the Markov chain: each new spin is visible to the next node's field immediately."),
        "pbit": figure(fig_pbit(), "The only multiply in the sampler is serial: 8 shift-add cycles, no DSP. The comparator is where the noise enters: P(bit = 1) = p / 65536 = σ(2β·field)."),
        "lfsr": figure(fig_lfsr(), "Each draw advances the LFSR 16 steps in one clock. The 16 steps are linear over GF(2), so synthesis flattens them into a fixed XOR network; how big it gets depends on where the taps sit (see the lane table)."),
        "sparse": figure(fig_sparse(), "All lanes share one round pointer and one slot counter, so they run in lock step. Same-round nodes belong to one colour block and are never coupled, so committing them one per cycle gives the same result as a sequential sweep."),
        "temper": figure(fig_temper(), "The array is unchanged; temper_ctrl drives its spin-load and run ports from the side. Energies are tracked incrementally from the update tap (flipping s_i changes E/2 by s_old·field). A swap only exchanges the rung labels, never the spin banks."),
        "pmode": figure(fig_pmode(), "One Langevin step updates every site from the old state. The tanh comes from the p-bit sigmoid ROM via tanh x = 2σ(2x) − 1; the Gaussian comes from a second ROM addressed by the top 10 bits of a uniform draw. The multiplies marked DSP land in hard 18 × 18 multipliers (5 after place-and-route)."),
        "uart": figure(fig_uart(), "Host link timing. The receiver confirms the start bit at its middle, then samples each data bit one bit-time later."),
        "bars": figure(bar_chart(), "Blue bars are the sampling core (pbit_array control, coupling_row, weight_mac, pbit, sigmoid ROM): 452 of 3,156 logic cells, 14%. Hover a bar for its LUT4 / ALU split; the table below has every number."),
    }
    tl = {
        "pbit": figure(timeline("tl1", [("IDLE", 1), ("MUL", 8), ("LOGIT", 1), ("LUT", 1), ("CMP", 1)], "pbit cycle budget: 12 cycles", "done pulses at cycle 12", hl=("MUL",), heat=("CMP",)), "One p-bit decision. Amber = the cycle the noise draw is compared and consumed."),
        "dense": figure(timeline("tl2", [("ORD", 1), ("BIAS", 1), ("MAC", 1), ("MACW", 18), ("PBW", 12), ("WR", 1)], "dense update cycle budget for N = 16: 34 cycles", "N + 18 = 34 cycles at N = 16", hl=("MACW",), heat=("PBW",)), "One dense update at N = 16. MACW is the row stream (N + 2 cycles); PBW is the p-bit."),
        "sparse": figure(timeline("tl3", [("NODE", 1), ("MAC", 1), ("MACW", 4), ("PB", 1), ("PBW", 12), ("CMT", 4)], "sparse round cycle budget: 23 cycles", "D + P + 15 = 23 cycles, P = 4 updates", hl=("MACW",), heat=("PBW",)), "One sparse round, P = 4 lanes, degree D = 4. The p-bit dominates; adding lanes costs one commit cycle each."),
        "pmode": figure(timeline("tl4", [("tanh x4", 8), ("site 0", 13), ("site 1", 13), ("site 2", 13), ("site 3", 13), ("CMT", 1)], "p-mode step cycle budget for M = 4: 61 cycles", "2M + M(2M+5) + 1 = 61 cycles", hl=("tanh x4",)), "One Langevin step, M = 4: per site 2M MAC cycles plus 5 pipeline cycles."),
    }
    lane_rows = [
        ["0", c("0xB4BCD35C"), "18", c("0x32D689B4"), "pbit_array, sparse lane 0, raw 'N' lane, pmode", "299 + 186"],
        ["1", c("0xA3000000"), "4", c("0x908F037F"), "sparse lane 1, tempering swap LFSR", "62 + 9"],
        ["2", c("0x80200003"), "4", c("0x76479D26"), "sparse lane 2", "154 + 90"],
        ["3", c("0x80000057"), "6", c("0xD43C16E9"), "sparse lane 3", "135 + 61"],
    ]
    physics_rows = [
        ["Noise source: a digitizing comparator fed by Gaussian noise, with a programmable threshold (DTM paper App. K, Fig. 15a)", "32-bit Galois LFSR gives a 16-bit uniform draw; " + c("rnd < p") + " is the threshold compare (inverse-CDF form)", "lfsr, pbit S_CMP", "32 DFF + XOR network in LUT4s; 16-cell carry chain"],
        ["Sigmoidal RNG bias versus control voltage (Fig. 4a, Eq. 11)", "1024 × 16 ROM of σ(logit/32), or 4 shift-and-add segments", "sigmoid_lut, sigmoid_pwl", "1 BSRAM (6T SRAM array), or 49 LUT4 + 43 ALU"],
        ["Resistor network summing sign-corrected neighbour states y = x XOR s (Fig. 10, Eq. E7)", "Conditional add or subtract of J, one neighbour per cycle", "weight_mac, coupling_row", "14-cell ALU chain + 14 DFF"],
        ["One sampling cell per node, two-colour block updates (Sec. III, Fig. 8)", "One shared engine (dense) or P lanes (sparse), visit order by colour", "pbit_array, sparse_array", "FSM + order RAM; P copies of the p-bit"],
        ["Inverse temperature β scales every coupling", "8-cycle shift-add product β·field", "pbit S_MUL", "23-cell ALU chain, shift registers"],
        ["RNG correlation time τ0 (Fig. 4b)", "nz_tau: a lane keeps its spin when its draw ≥ nz_q", "sparse_array NOISE_CTL", "16-bit compare per lane, keep_reg"],
        ["Physical randomness", "RO_JITTER: three inverters in a loop, 2-FF sync, XOR into bit 0", "noise_bank", "3 LUT1 in a loop + 2 DFF"],
        ["Host ↔ FPGA ↔ sampler split, host writes h and J (Framework paper Figs. 6-7)", "UART command set", "top, uart_rx, uart_tx", "~600 logic cells"],
    ]
    est_rows = [
        ["Logic sites (LUT4 + ALU)", "3,677", "≈ 140: 16 config bits × 6T, a 4-level pass-transistor tree, restoring buffers", "≈ 0.5 M"],
        ["DFF", "1,326", "≈ 24-30: two latches, set/reset, clock-enable mux", "≈ 0.04 M"],
        ["BSRAM (18 Kbit each)", "10", "18,432 cells × 6T ≈ 110 k, plus decoders and sense amplifiers", "≈ 1.1 M+"],
        ["MUX2_LUT5…8", "280", "≈ 12", "≈ 3 k"],
        ["Routing switches and their configuration SRAM", "", "not attributable per design", "the largest share on any SRAM FPGA"],
    ]

    findings = [
        ("The lane-0 LFSR mask is the most expensive small block",
         "With MASK = 0xB4BCD35C (18 taps) the 16-step leap synthesizes to 299 LUT4 + 186 MUX2 cells; lane 1's mask (4 taps, all in the top byte) needs 62 + 9. The dense build carries two lane-0 LFSRs (array and raw 'N' lane), about 26% of its LUT4s. A sparser maximal polynomial for lane 0 would save roughly 470 LUT4 per build, but model/fixed.LFSR_MASKS must change with it to keep the bit-exact mirror."),
        ("The readout counters are bigger than the sampler",
         "In the dense N = 16 build, readout_counter uses 635 LUT4, 589 ALU and 681 DFF, mostly the 16 × 32-bit flip-flop counters. UP_RAM = 1 at N = 16 drops it to 195 LUT4, 114 ALU and 237 DFF for one extra BSRAM; its 17-cycle scan is far shorter than a 544-cycle sweep."),
        ("Every build meets 27 MHz with at least 2× margin",
         "Post-route Fmax ranges from 54.9 MHz (tempering) to 72.5 MHz (dense). The slowest path is the tempering swap setup: replica label lookup, 24-bit energy subtract and saturation into sf_reg. In the dense and sparse-16 builds the worst path is the schedule's 16-bit sweep-per-step compare driving the β register enable."),
        ("Sparse N = 256 nearly fills the chip",
         "With P = 4 and D = 4 it places 7,009 LUT4 + 870 ALU, about 91% of the 8,640 logic sites. The 256:1 spin read mux in each lane and the 256-way spin write decoder dominate (sparse_array alone is 3,686 LUT4 + 2,246 MUX2 in Yosys). More lanes or a wider degree will not fit without restructuring the spin register."),
        ("The raw 'N' lane replays lane 0",
         "raw_noise_inst uses lane 0's polynomial and seed, and 'X' reseeds it together with the array. Its stream is the same sequence the dense array's lane 0 consumes, which is useful for checking the generator offline, not as an independent source."),
    ]

    sigs = sig_rows()

    H = []
    H.append(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Thermodynamic Computer Wiring Map</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:wght@400;600&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body><div class="wrap">{nav_html()}<main>""")

    H.append("""<header class="top" id="overview">
<p class="eyebrow">Tang Nano 9K · Gowin GW1NR-9 · 19 Verilog modules · 27 MHz</p>
<h1>Thermodynamic Computer Wiring Map</h1>
<p class="lede">This design is a digital stand-in for a probabilistic computer. It stores a model (couplings J<sub>ij</sub> and biases h<sub>i</sub>) and resamples one spin s<sub>i</sub> ∈ {−1, +1} at a time, so that over many sweeps each spin state appears with its Boltzmann probability. Where the source papers make each decision with noisy transistors, this design uses an LFSR for the noise, a 1024-entry ROM for the sigmoid and a 16-bit comparator for the threshold. The map below follows one signal from a UART byte down to the transistors that hold and switch it.</p>
<div class="eq">E(s) = −½ Σ<sub>ij</sub> J<sub>ij</sub> s<sub>i</sub> s<sub>j</sub> − Σ<sub>i</sub> h<sub>i</sub> s<sub>i</sub> &nbsp;&nbsp; P(s) ∝ exp(−β E(s))<br>
p-bit rule: P(s<sub>i</sub> = +1 | rest) = σ(2β (Σ<sub>j</sub> J<sub>ij</sub> s<sub>j</sub> + h<sub>i</sub>)) &nbsp;→&nbsp; hardware: bit = (rnd &lt; ROM[2β·field])</div>
<dl class="facts">
<div><dt>Clock domains</dt><dd>1 · 27 MHz</dd></div>
<div><dt>p-bit decision</dt><dd>12 cycles · 444 ns</dd></div>
<div><dt>Dense update</dt><dd>N + 18 cycles</dd></div>
<div><dt>Sparse, P = 4, D = 4</dt><dd>4.7 M updates/s</dd></div>
<div><dt>Dense N = 16 build</dt><dd>43% logic · 10/26 BSRAM</dd></div>
<div><dt>Post-route Fmax</dt><dd>54.9 – 72.5 MHz</dd></div>
</dl>
<p>Three things are built from these files: <code>top</code>, the UART-hosted Ising / Max-Cut sampler (dense, sparse or with parallel tempering); <code>pmode_top</code>, a stand-alone continuous Langevin machine; and two library cores, <code>pgate</code> (invertible logic) and <code>denoise_demo</code> (a denoising chain), which neither board top instantiates.</p>
</header>""")

    H.append("""<section id="layers"><h2>Layer ladder</h2>
<p>Each layer is built only from the one below it. Read top to bottom to follow a command into silicon, or bottom to top to see what a transistor ends up doing.</p>
<ol class="ladder">
<li><span class="lv">L6</span><a href="#host">Host link</a><span>UART bytes become one-cycle write strobes and argument slices ('W' → w_we, w_addr, w_data)</span></li>
<li><span class="lv">L5</span><a href="#system">Board top</a><span>Pins, reset, command FSM and the buses between the big blocks</span></li>
<li><span class="lv">L4</span><a href="#dense">Sampling engines</a><span>Dense Gibbs engine, sparse lanes, parallel tempering, Langevin p-modes, clamped wrappers</span></li>
<li><span class="lv">L3</span><a href="#pbit">Mechanisms</a><span>The p-bit decision, LFSR noise lanes, β annealing, on-chip counting</span></li>
<li><span class="lv">L2</span><a href="#formats">Numbers and time</a><span>Fixed-point formats and cycle budgets</span></li>
<li><span class="lv">L1</span><a href="#fabric">FPGA fabric</a><span>LUT4, flip-flops, carry chains, block RAM, multipliers, routing; measured counts and Fmax</span></li>
<li><span class="lv">L0</span><a href="#transistors">Transistors</a><span>SRAM cells, pass-transistor muxes, latches, carry muxes, the ring oscillator</span></li>
</ol>""" + figs["legend"] + "</section>")

    H.append("""<section id="hierarchy"><h2>Module tree</h2>
<p>Instance names as they appear in the netlist. Chips mark which build includes a branch.</p>
<div class="trees">
<div><h3>top (UART sampler)</h3><ul class="tree">
<li><span class="m">uart_rx_inst</span> <span class="p">uart_rx, CLKS_PER_BIT = 234</span></li>
<li><span class="m">uart_tx_inst</span> <span class="p">uart_tx</span></li>
<li><span class="m">schedule_ctrl_inst</span> <span class="p">schedule_ctrl, β table</span></li>
<li><span class="m">gen_dense.pbit_array_inst</span> <span class="p">pbit_array</span><span class="chip">dense</span>
 <ul><li><span class="m">coupling_row_inst</span> <span class="p">coupling_row</span><ul><li><span class="m">weight_mac_inst</span> <span class="p">weight_mac</span></li></ul></li>
 <li><span class="m">pbit_inst</span> <span class="p">pbit</span><ul><li><span class="m">gen_lut.sigmoid_lut_inst</span> <span class="p">or gen_pwl.sigmoid_pwl_inst</span></li></ul></li>
 <li><span class="m">noise_bank_inst</span> <span class="p">noise_bank, 1 lane</span><ul><li><span class="m">gen_lane[0].lfsr_inst</span> <span class="p">lfsr, mask lane 0</span></li></ul></li></ul></li>
<li><span class="m">gen_dense.gen_pt.temper_ctrl_inst</span> <span class="p">temper_ctrl</span><span class="chip">TEMPER_K &gt; 1</span>
 <ul><li><span class="m">swap_pbit_inst</span> <span class="p">pbit → sigmoid_lut</span></li><li><span class="m">swap_lfsr_inst</span> <span class="p">lfsr, mask A3000000</span></li></ul></li>
<li><span class="m">gen_sparse.sparse_array_inst</span> <span class="p">sparse_array</span><span class="chip">SPARSE</span>
 <ul><li><span class="m">gen_lane[0..P−1].weight_mac_inst</span> <span class="p">weight_mac</span></li>
 <li><span class="m">gen_lane[0..P−1].pbit_inst</span> <span class="p">pbit → sigmoid_lut</span></li>
 <li><span class="m">noise_bank_inst</span> <span class="p">noise_bank, P lanes → P × lfsr</span></li></ul></li>
<li><span class="m">raw_noise_inst</span> <span class="p">noise_bank, 1 lane, RO_JITTER</span></li>
<li><span class="m">readout_counter_inst</span> <span class="p">readout_counter, HIST_BITS, UP_RAM</span></li>
</ul></div>
<div><h3>pmode_top (Langevin)</h3><ul class="tree">
<li><span class="m">pmode_array_inst</span> <span class="p">pmode_array</span>
 <ul><li><span class="m">tanh_lut_inst</span> <span class="p">sigmoid_lut, sigmoid table</span></li>
 <li><span class="m">gauss_lut_inst</span> <span class="p">sigmoid_lut, Gaussian inverse CDF</span></li>
 <li><span class="m">noise_bank_inst</span> <span class="p">noise_bank, 1 lane → lfsr</span></li></ul></li></ul>
<h3>pgate (library)</h3><ul class="tree">
<li><span class="m">pbit_array_inst</span> <span class="p">pbit_array, N = 3 (AND gadget)</span></li>
<li><span class="m">gen_pt.temper_ctrl_inst</span> <span class="p">temper_ctrl</span><span class="chip">K &gt; 1</span></li></ul>
<h3>denoise_demo (library)</h3><ul class="tree">
<li><span class="m">pbit_array_inst</span> <span class="p">pbit_array, N = 16, reprogrammed every step</span></li></ul></div>
</div></section>""")

    H.append(f"""<section id="host"><h2><span class="lv">L6</span>Host link</h2>
<p>Every command is one byte followed by 0 to 6 argument bytes (big-endian). <code>nargs_of()</code> gives the count; bytes shift into <code>args_reg</code>; in <code>S_EXEC</code> one strobe fires and the port wires read fixed slices of <code>args_reg</code>. Replies are little-endian and every command ends with 'K'. Unknown bytes return '?'.</p>
{table(["Command", "Arg bytes", "args_reg slice → wire", "Lands in"], HOST_ROWS)}
{figs["uart"]}</section>""")

    H.append(f"""<section id="system"><h2><span class="lv">L5</span>Board top</h2>
<p>One clock, one synchronous reset. Reset is held for the first 15 cycles by a 4-bit power-on counter and whenever S1 is pressed; both buttons pass through a two-flop synchronizer, and S2 also through an 18-bit debounce counter. The command FSM is the only master: nothing in the engine starts unless the FSM pulses <code>arr_run</code>.</p>
{figs["system"]}</section>""")

    H.append(f"""<section id="dense"><h2><span class="lv">L4</span>Dense engine: pbit_array</h2>
<p>A sweep visits all N nodes in the stored order. For each node the engine reads its index and bias, streams row i of J through a conditional adder (spins are ±1, so the multiply is a sign flip), runs the p-bit, and writes the new spin straight back. Because colour block 0 is visited before block 1, sequential updates inside a block equal the parallel block update of the source architecture.</p>
{figs["dense"]}{tl["dense"]}</section>""")

    H.append(f"""<section id="sparse"><h2>Sparse lanes: sparse_array</h2>
<p>For graphs of degree ≤ D, each node needs only D neighbour words. The host pre-arranges every colour block into rounds of P nodes (one per lane) and pads short rounds with valid = 0. Each lane has its own node RAM, neighbour RAM, accumulator, p-bit and LFSR lane; they share the round pointer, the slot counter and the spin register.</p>
{figs["sparse"]}{tl["sparse"]}
<p>Noise options (NOISE_CTL = 1): with <code>nz_tau</code> each valid lane draws once in S_PB; a draw ≥ nz_q means "not decorrelated yet", so the lane keeps its spin and its p-bit draw is not consumed. With <code>nz_share</code> every lane reads lane 0's draw and only lane 0 advances, giving fully correlated noise inside a round.</p></section>""")

    H.append(f"""<section id="tempering"><h2>Parallel tempering: temper_ctrl</h2>
<p>K ≤ 4 replicas share the one dense array. A round sweeps each replica once at the β of its current rung, then tries swaps between neighbouring rungs: pairs (0,1) and (2,3) on even rounds, (1,2) on odd rounds. Acceptance is Barker's rule, σ((β<sub>a</sub> − β<sub>b</sub>)(E<sub>x</sub> − E<sub>y</sub>)), computed by a second p-bit with field = Δe and beta = |Δβ|. Default ladder (Q4.4): β<sub>0</sub> = 3.0, β<sub>1</sub> = 2.0, β<sub>2</sub> = 1.25, β<sub>3</sub> = 0.75. <code>spins_out</code>, the readout and the LEDs always show the replica sitting on rung 0.</p>
{figs["temper"]}</section>""")

    H.append(f"""<section id="pmode"><h2>Langevin p-modes: pmode_array</h2>
<p>Continuous soft spins x<sub>i</sub> in a double well, driven by overdamped Langevin dynamics. One step: x′ = x − 2<sup>−k</sup> dV/dx + σξ with dV/dx<sub>i</sub> = −(1 − s<sub>i</sub>²)(Σ<sub>j</sub> A<sub>ij</sub> s<sub>j</sub> + h<sub>i</sub>) + λ x<sub>i</sub>(x<sub>i</sub>² − 1) and s = tanh x. Defaults: λ = 2.0, ε = 1/16, σ = 0.3535 (T = 1). On the board, S2 toggles running, one step every 2<sup>20</sup> clocks, and LED k is lit while x<sub>k</sub> ≥ 0.</p>
{figs["pmode"]}{tl["pmode"]}</section>""")

    H.append("""<section id="wrappers"><h2>Clamped wrappers: pgate and denoise_demo</h2>
<div class="two"><div>
<h3>pgate: invertible logic</h3>
<p>The gate's J and h make its truth-table rows the lowest-energy states. On <code>start</code> the wrapper latches clamp_mask / clamp_val and rewrites the bias RAM in N cycles: a clamped node gets h = +127 (0x7F) or −128 (0x80), any other node gets its stored h0. Clamp the inputs and the gate runs forward; clamp the output and it runs backward. Sequence: IDLE → BIAS (N) → RUN → WAIT (n_sweeps) → done. With K &gt; 1 the run goes through temper_ctrl.</p></div>
<div><h3>denoise_demo: denoising chain</h3>
<p>Starts from x<sup>T</sup> and runs T = 4 steps on one array. Each step: load spins ← x<sup>t</sup>, rewrite every bias to sat8(h<sub>i</sub> ± Γ<sub>t</sub>/2) with the sign of x<sup>t</sup><sub>i</sub> (N cycles), run K = 16 sweeps at β = 1.0, take the spins as x<sup>t−1</sup>. step_done pulses after each step, done after t = 1. The same hardware is reprogrammed at every step.</p></div></div></section>""")

    H.append(f"""<section id="pbit"><h2><span class="lv">L3</span>The p-bit decision</h2>
<p>Every sample in every engine passes through this block (dense, each sparse lane, and the tempering swap). The product β·field arrives with 8 fraction bits; doubling it and keeping 5 fraction bits is a right shift by WFRAC − 2 = 2 with round-half-up. The 10-bit logit is offset by 512 to address the ROM.</p>
{figs["pbit"]}{tl["pbit"]}
<h3>One dense update, every mechanism it touches</h3>
<ul class="prose">
<li>2 SSRAM reads (order, bias) and N BSRAM reads (one row of J)</li>
<li>N add-or-subtract operations on a 14-bit carry chain, each steered by one spin through an N:1 mux</li>
<li>8 conditional 23-bit adds (β·field), one rounding add, two saturation compares</li>
<li>1 BSRAM read (σ), one 16-bit compare against the noise draw</li>
<li>One 32-bit LFSR leap through the XOR network (16 Galois steps in one clock)</li>
<li>One indexed spin write; at the end of a sweep, one pulse to the schedule and the counters</li>
</ul></section>""")

    H.append(f"""<section id="noise"><h2>Noise lanes: lfsr and noise_bank</h2>
<p>Lane k uses its own maximal-length polynomial (different m-sequences, not shifted copies) and seed = SEED ⊕ (0x9E3779B9 · (k+1)). Asserting <code>en</code> consumes the current draw. Default SEED = 0xACE1F00D.</p>
{figs["lfsr"]}
{table(["Lane", "Mask", "Taps", "Reset seed", "Used by", "Leap cost LUT4 + MUX2"], lane_rows, num_cols=(2, 5))}</section>""")

    H.append("""<section id="anneal-readout"><h2>Annealing and readout</h2>
<div class="two"><div><h3>schedule_ctrl</h3>
<p>Holds β in an 8-bit register. Mode 0 keeps β0. Mode 1 adds Δ every SPS sweeps and saturates at β1. Mode 2 (the default) steps through a 16-entry table every SPS sweeps and holds at entry 15. The power-on default is mode 2, SPS = 8, so 16 × 8 = 128 sweeps (DEMO_SWEEPS) cover the whole table. With parallel tempering the ladder replaces this β.</p></div>
<div><h3>readout_counter</h3>
<p>On each <code>swd</code> pulse it counts one sample. The histogram is a 2<sup>12</sup> × 32 BSRAM indexed by the low 12 spins, updated by a 3-cycle read-modify-write. The up counters count how often each spin was +1: N parallel 32-bit incrementers in flip-flops, or with UP_RAM = 1 one BSRAM and an N + 1-cycle pipelined scan. A clear that arrives during a scan waits and runs right after it.</p></div></div></section>""")

    H.append(f"""<section id="formats"><h2><span class="lv">L2</span>Number formats</h2>
<p>All arithmetic is fixed point. The weight fraction (WFRAC = 4) and β's 4 fraction bits fix every shift in the pipeline.</p>
{table(["Quantity", "Bits", "Format", "Range", "Where"], FORMAT_ROWS)}</section>""")

    H.append(f"""<section id="timing"><h2>Cycle budgets</h2>
<p>Counts from the RTL headers (measured by the testbenches there) and checked against the FSMs. Rates assume the 27 MHz board clock.</p>
{table(["Operation", "Cycles", "Value", "Notes"], TIMING_ROWS)}</section>""")

    H.append(f"""<section id="fabric"><h2><span class="lv">L1</span>FPGA fabric</h2>
<p>I synthesized each build with Yosys (synth_gowin) and placed and routed it with nextpnr for the Tang Nano 9K part (GW1NR-LV9QN88PC6/I5) at 27 MHz. ALU cells are LUT4 sites switched into carry mode, so the logic actually occupied is LUT4 + ALU.</p>
{pnr_table()}
<h3>Where the logic goes (dense N = 16)</h3>
{figs["bars"]}
{table(["Instance", "What it holds", "LUT4", "ALU", "MUX2", "DFF", "BSRAM", "SSRAM"], YOSYS_ROWS, num_cols=(2, 3, 4, 5, 6, 7))}
<h3>Critical paths (post-route)</h3>
{crit_table()}
<p>On every critical path, routing takes more of the delay than logic, except in the tempering build, where the long subtract-and-saturate chain dominates.</p></section>""")

    T = [figure(tp_inverter(), "<strong>CMOS inverter.</strong> PMOS pulls the output to VDD when the input is low, NMOS pulls it to ground when it is high. Three in a loop (RO_JITTER) never settle: they oscillate at about 1/(6·t<sub>pd</sub>), and the 2-FF synchronizer samples that phase.", "small"),
         figure(tp_sram(), "<strong>6T SRAM cell.</strong> Every BSRAM bit (weights, sigmoid and Gaussian tables, histogram) and every LUT and routing configuration bit is one of these.", "small"),
         figure(tp_lut(), "<strong>LUT4.</strong> Any 4-input function is a 16-bit truth table read through a pass-transistor tree. The LFSR's XOR network, the FSMs and every mux in the map are built from these.", "small"),
         figure(tp_dff(), "<strong>D flip-flop.</strong> Every register with a clock notch in the schematics. The master follows D while clk is low; on the rising edge the slave takes over and holds Q.", "small"),
         figure(tp_carry(), "<strong>Carry chain.</strong> Every +, − and &lt; in the map: the β·field adder, weight_mac's ±J (B inverted, Cin = 1), the counters, and the comparator rnd &lt; p, which is the borrow out of a 16-cell chain.", "small")]
    H.append(f"""<section id="transistors"><h2><span class="lv">L0</span>Transistors</h2>
<p>An SRAM FPGA has no gates of its own: it is a large array of SRAM bits, pass transistors and fixed cells, and the bitstream decides which paths conduct. These are textbook structures; Gowin does not publish its cell layouts, so treat the drawings as the mechanism, not the exact circuit.</p>
<div class="tgrid">{"".join(T)}</div>
<h3>What the FPGA emulates</h3>
<p>Physical mechanisms named in the RTL headers (from the cited DTM paper, arXiv 2510.23972, and the Framework paper) and what replaces each one here.</p>
{table(["Physical mechanism in the source hardware", "Digital replacement", "Module", "Primitives"], physics_rows)}
<h3>Rough transistor budget of the dense N = 16 build</h3>
{table(["Primitive", "Used (P&R)", "Transistors each, textbook estimate", "Total"], est_rows, num_cols=(1, 3))}
<p>Order of magnitude: about 1.7 million transistors' worth of cells excluding routing, two thirds of them SRAM bits inside the 10 block RAMs. The 256 × 8 weight table alone occupies a full 18 Kbit block.</p></section>""")

    H.append(f"""<section id="signals"><h2>Signal dictionary</h2>
<p>Registers and inter-module wires, grouped by file. Widths use the module parameters (N, IW, AW, WB, HB = HIST_BITS, P, D, LW, M).</p>
<div class="filter"><label for="sigq">Filter</label><input id="sigq" type="search" placeholder="e.g. beta, rnd, spins, temper" autocomplete="off"><span id="sigc" aria-live="polite"></span></div>
<div id="sigtbl">{table(["Signal", "Width", "Driven by", "Goes to", "Meaning"], sigs)}</div></section>""")

    H.append('<section id="findings"><h2>Findings from synthesis</h2><p>Things the numbers show that the RTL alone does not.</p>' +
             "".join(f'<div class="finding"><strong>{E(t)}</strong>{E(b)}</div>' for t, b in findings) + "</section>")

    H.append("""<section id="provenance"><h2>Provenance</h2>
<ul class="prose">
<li>RTL: the 19 uploaded Verilog files, unmodified.</li>
<li>Synthesis: Yosys 0.70, <code>synth_gowin -noflatten -noiopads</code> for per-module counts. Place and route: nextpnr-himbaechel (Gowin), device GW1NR-LV9QN88PC6/I5, family GW1N-9C, 27 MHz target, seed 1, standard Tang Nano 9K pins.</li>
<li>The weight, bias, order and β table images were placeholders (zeros, identity order). They are RAM initial values, so they do not change any count above. The sigmoid table was generated exactly as sigmoid_lut.v specifies; the Gaussian table as a standard inverse CDF in Q4.12.</li>
<li>The tempering round length is traced from the FSMs; the other cycle counts match the values the RTL headers report as measured.</li>
<li>Transistor counts per primitive are textbook estimates, not Gowin data.</li>
<li>Device totals: <a href="https://wiki.sipeed.com/hardware/en/tang/Tang-Nano-9K/Nano-9K">Sipeed Tang Nano 9K wiki</a> (8,640 LUT4, 6,480 FF, 26 BSRAM, 20 18 × 18 multipliers).</li>
</ul></section>""")

    H.append("""</main></div><script>
(function(){
  var q=document.getElementById('sigq'), cnt=document.getElementById('sigc');
  var rows=[].slice.call(document.querySelectorAll('#sigtbl tbody tr'));
  function run(){
    var t=q.value.trim().toLowerCase(), shown=0, lastGrp=null, grpHas=false;
    rows.forEach(function(r){
      if(r.classList.contains('grp')){ if(lastGrp){lastGrp.hidden=!grpHas;} lastGrp=r; grpHas=false; return; }
      var ok=!t || r.textContent.toLowerCase().indexOf(t)>=0 || (lastGrp && lastGrp.textContent.toLowerCase().indexOf(t)>=0);
      r.hidden=!ok; if(ok){shown++; grpHas=true;}
    });
    if(lastGrp){lastGrp.hidden=!grpHas;}
    cnt.textContent=t? (shown+' of '+rows.filter(function(r){return !r.classList.contains('grp');}).length+' signals'):'';
  }
  q.addEventListener('input',run);
})();
</script></body></html>""")
    return "".join(H)


if __name__ == "__main__":
    page = build()
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(page)
    print(f"wrote {OUT}: {len(page):,} bytes")
    for w in WARN:
        print("WARN", w)
