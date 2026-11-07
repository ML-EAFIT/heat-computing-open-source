"""Every zoomable view of the thermodynamic computer map: drawing, portals (child views) and info text."""
import math
import re
import os
import sys
from statistics import NormalDist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_map as bm                      # noqa: E402
from build_map import Fig, E, inv_symbol, mos, tg, c, table   # noqa: E402

HDR = 44
LAYERS = {6: "L6 · host + board", 5: "L5 · bitstream top", 4: "L4 · block / engine",
          3: "L3 · mechanism", 2: "L2 · datapath element", 1: "L1 · FPGA primitive", 0: "L0 · transistor"}
TYPES = {}


def nf(fid, w, h, aria):
    f = Fig(fid, w, h, aria)
    f.portals = []
    return f


def P(f, x, y, w, h, t, n):
    f.portals.append(dict(t=t, n=n, x=x, y=y, w=w, h=h))


def B(f, x, y, w, h, title, subs=(), kind="cmb", z=None, n=None, **kw):
    f.box(x, y, w, h, title, subs, kind, **kw)
    if z:
        P(f, x, y, w, h, z, n or title)


def add(tid, f, title, layer, info):
    body = re.sub(r"url\(#[A-Za-z0-9_]+-([amnc])\)", r"url(#z-\1)", "".join(f.p))
    svg = (f'<rect x="0.5" y="0.5" width="{f.w-1}" height="{f.h+HDR-1}" rx="10" class="sheet"/>'
           f'<text x="20" y="29" class="t htl">{E(title)}</text>'
           f'<text x="{f.w-20}" y="29" class="t hly" text-anchor="end">{E(LAYERS[layer])}</text>'
           f'<g transform="translate(0,{HDR})">{body}</g>')
    TYPES[tid] = dict(title=title, layer=layer, w=f.w, h=f.h + HDR, svg=svg,
                      portals=[dict(p, y=p["y"] + HDR) for p in f.portals], info=info)


# ---------------------------------------------------------------- info helpers
SIG = {}
_cur = None
for r in bm.sig_rows():
    if isinstance(r, str):
        _cur = r
        SIG[_cur] = []
    else:
        SIG[_cur].append(r)


def sigtab(*mods):
    rows = []
    for m in mods:
        rows += SIG[m]
    return "<h4>Signals</h4>" + table(["Signal", "Width", "From", "To", "Meaning"], rows)


def facts(pairs):
    return '<dl class="kv">' + "".join(f"<div><dt>{E(k)}</dt><dd>{v}</dd></div>" for k, v in pairs) + "</dl>"


def para(*ps):
    return "".join(f"<p>{x}</p>" for x in ps)


def tl(fid, segs, aria, total, hl=(), heat=()):
    f = bm.timeline(fid, segs, aria, total, hl=hl, heat=heat)
    return f'<div class="mini">{f.svg()}</div>'


def fmt_rows(names):
    return table(["Quantity", "Bits", "Format", "Range", "Where"],
                 [r for r in bm.FORMAT_ROWS if any(n in r[0] for n in names)])


def finding(i):
    t, b = FINDINGS[i]
    return f'<div class="finding"><strong>{E(t)}</strong>{E(b)}</div>'


FINDINGS = [
    ("The lane-0 LFSR mask is the most expensive small block",
     "With MASK = 0xB4BCD35C (18 taps) the 16-step leap synthesizes to 299 LUT4 + 186 MUX2; lane 1's mask (4 taps, all in the top byte) needs 62 + 9. The dense build carries two lane-0 LFSRs, about 26% of its LUT4s. A sparser maximal polynomial would save roughly 470 LUT4, but model/fixed.LFSR_MASKS must change with it."),
    ("The readout counters are bigger than the sampler",
     "In the dense N = 16 build readout_counter uses 635 LUT4, 589 ALU and 681 DFF, mostly the 16 × 32-bit flip-flop counters. UP_RAM = 1 drops it to 195 LUT4, 114 ALU and 237 DFF for one extra BSRAM."),
    ("Every build meets 27 MHz with at least 2× margin",
     "Post-route Fmax ranges from 54.9 MHz (tempering) to 72.5 MHz (dense). The slowest path is the tempering swap setup: label lookup, 24-bit energy subtract and saturation into sf_reg."),
    ("Sparse N = 256 nearly fills the chip",
     "With P = 4 and D = 4 it places 7,009 LUT4 + 870 ALU, about 91% of the 8,640 logic sites. The 256:1 spin read mux per lane and the 256-way spin write dominate."),
    ("The raw 'N' lane replays lane 0",
     "raw_noise_inst uses lane 0's polynomial and seed and is reseeded with the array, so its stream is the sequence the dense array's lane 0 consumes, not an independent source."),
]


# ================================================================= L6 world
def v_world():
    f = nf("w", 1300, 600, "Host PC connected over USB to the Tang Nano 9K board; the FPGA chip holds the top bitstream; other bitstreams and library cores on the right")
    B(f, 30, 215, 220, 170, "Host PC", ["Python host scripts", "UART 115200 8N1", "W H O T B R S D U Q"])
    f.box(300, 30, 680, 540, "Sipeed Tang Nano 9K", [], "grp")
    B(f, 312, 270, 40, 60, "USB", [], "pin")
    f.wire([(250, 300), (312, 300)], arrow="both", label="USB-C", at=(258, 292))
    B(f, 395, 245, 160, 110, "BL702", ["USB ↔ UART", "USB ↔ JTAG"])
    f.wire([(352, 300), (395, 300)], arrow="both")
    cx, cy, cs = 615, 120, 300
    for i in range(12):
        o = 24 + i * 23
        f.raw(f'<path d="M{cx+o},{cy-10} L{cx+o},{cy} M{cx+o},{cy+cs} L{cx+o},{cy+cs+10} '
              f'M{cx-10},{cy+o} L{cx},{cy+o} M{cx+cs},{cy+o} L{cx+cs+10},{cy+o}" class="w"/>')
    f.raw(f'<rect x="{cx}" y="{cy}" width="{cs}" height="{cs}" rx="4" class="bx hl"/>')
    f.text(cx + 14, cy + 24, "GW1NR-9 FPGA", "ttl")
    f.text(cx + 14, cy + 40, "8,640 LUT4 · 6,480 DFF · 26 BSRAM")
    P(f, cx + 12, cy + 52, cs - 24, cs - 64, "top", "FPGA · top bitstream")
    f.text(cx + cs - 14, cy + 24, "click to open", "lbl", "end")
    f.wire([(555, 268), (605, 268)], label="rx", at=(562, 262))
    f.wire([(605, 296), (555, 296)], label="tx", at=(562, 290))
    f.wire([(555, 330), (605, 330)], "c", label="JTAG", at=(560, 346))
    B(f, 630, 470, 110, 46, "27 MHz", ["crystal, pin 52"])
    f.wire([(685, 470), (685, 430)])
    B(f, 760, 470, 90, 46, "S1", ["reset, pin 4"])
    B(f, 870, 470, 90, 46, "S2", ["run, pin 3"])
    f.wire([(805, 470), (805, 430)])
    f.wire([(915, 470), (915, 430)])
    for i in range(6):
        y = 160 + i * 42
        f.wire([(925, y), (950, y)])
        f.raw(f'<circle cx="960" cy="{y}" r="9" class="led"/>')
        f.text(940, y - 8, f"led{i}", "axn", "end")
    f.box(1005, 30, 275, 540, "Other bitstreams and cores", [], "grp")
    B(f, 1025, 75, 235, 140, "pmode_top", ["stand-alone bitstream", "Langevin p-modes"], z="pmodetop")
    B(f, 1025, 245, 235, 140, "pgate", ["invertible logic core", "library, testbench only"], z="pgate")
    B(f, 1025, 415, 235, 135, "denoise_demo", ["denoising chain core", "library, testbench only"], z="denoise")
    info = (para("A digital stand-in for a probabilistic (thermodynamic) computer. It stores couplings J<sub>ij</sub> and biases h<sub>i</sub> and resamples one spin at a time so that, over many sweeps, each state appears with its Boltzmann probability.",
                 "Where the source papers decide each spin with noisy transistors, this design uses an LFSR for the noise, a 1024-entry ROM for the sigmoid and a 16-bit comparator for the threshold.")
            + '<div class="eq">E(s) = −½ Σ J<sub>ij</sub> s<sub>i</sub> s<sub>j</sub> − Σ h<sub>i</sub> s<sub>i</sub><br>P(s<sub>i</sub> = +1) = σ(2β (Σ<sub>j</sub> J<sub>ij</sub> s<sub>j</sub> + h<sub>i</sub>))<br>hardware: bit = rnd &lt; ROM[2β·field]</div>'
            + facts([("Clock", "27 MHz, one domain"), ("p-bit decision", "12 cycles"), ("Dense N=16 build", "43% logic sites, 10/26 BSRAM"), ("Fmax range", "54.9 – 72.5 MHz")])
            + "<h4>Place-and-route results</h4>" + bm.pnr_table() + bm.crit_table()
            + "<h4>What the FPGA emulates</h4>" + table(["Physical mechanism", "Digital replacement", "Module", "Primitives"], PHYS)
            + "<h4>Findings</h4>" + "".join(finding(i) for i in range(5)))
    add("world", f, "Host + Tang Nano 9K", 6, info)


PHYS = [
    ["Digitizing comparator fed by Gaussian noise, programmable threshold (DTM App. K, Fig. 15a)", "LFSR draw, compare " + c("rnd < p"), "lfsr, pbit", "32 DFF + XOR net; 16-cell carry chain"],
    ["Sigmoidal RNG bias vs control voltage (Fig. 4a, Eq. 11)", "1024 × 16 ROM of σ, or PWL", "sigmoid_lut", "1 BSRAM"],
    ["Resistor network over sign-corrected neighbours (Fig. 10, Eq. E7)", "Conditional add/sub of J", "weight_mac", "14-cell carry chain"],
    ["One sampling cell per node, two-colour blocks (Sec. III)", "Shared engine or P lanes", "pbit_array, sparse_array", "FSM + RAM"],
    ["RNG correlation time τ0 (Fig. 4b)", "nz_tau keep-spin option", "sparse_array", "compare + keep_reg"],
    ["Physical randomness", "3-inverter ring oscillator", "noise_bank RO_JITTER", "3 LUT1 + 2 DFF"],
]


# ================================================================= L5 top
def v_top():
    f = bm.fig_system()
    f.portals = []
    P(f, 150, 80, 160, 100, "resetbtn", "reset + buttons")
    P(f, 150, 226, 160, 56, "uartrx", "uart_rx")
    P(f, 150, 306, 160, 56, "uarttx", "uart_tx")
    P(f, 420, 40, 220, 520, "cmdfsm", "command FSM")
    P(f, 740, 40, 210, 64, "sched", "schedule_ctrl")
    P(f, 740, 160, 210, 170, "engine", "sampling engine")
    P(f, 740, 380, 170, 52, "lfsr0", "raw noise_bank")
    P(f, 740, 470, 210, 90, "readout", "readout_counter")
    info = (para("The UART sampler bitstream. One clock, one synchronous reset; the command FSM is the only master. Every argument byte lands in <code>args_reg</code>, one strobe per command writes a slice of it into a RAM port, and the engine's <code>swd</code> pulse drives the schedule, the counters and the reply logic at once.")
            + "<h4>Where the logic goes (dense N = 16)</h4>" + f'<div class="mini">{bm.bar_chart().svg()}</div>'
            + table(["Instance", "What it holds", "LUT4", "ALU", "MUX2", "DFF", "BSRAM", "SSRAM"], bm.YOSYS_ROWS, num_cols=(2, 3, 4, 5, 6, 7))
            + sigtab("top.v"))
    add("top", f, "top.v · UART sampler", 5, info)


# ================================================================= L4 blocks
def v_resetbtn():
    f = nf("rb", 980, 330, "Reset and button logic")
    B(f, 20, 40, 110, 40, "btn_rst_n", ["S1"], "pin")
    B(f, 170, 40, 80, 40, "rst_s1", [], "reg", z="dff", n="rst_s1 (DFF)")
    B(f, 290, 40, 80, 40, "rst_s2", [], "reg", z="dff", n="rst_s2 (DFF)")
    B(f, 170, 110, 120, 50, "por_reg", ["4 bit, +1 to 15"], "reg", z="register", n="por_reg")
    B(f, 330, 110, 110, 50, "≠ 15", ["POR active"], z="lut4", n="POR compare (LUT4)")
    B(f, 500, 60, 90, 60, "OR", ["!s2 or POR"], z="lut4", n="reset OR (LUT4)")
    f.wire([(130, 60), (170, 60)])
    f.wire([(250, 60), (290, 60)])
    f.wire([(370, 60), (420, 60), (420, 78), (500, 78)], label="!rst_s2", at=(426, 72))
    f.wire([(290, 135), (330, 135)])
    f.wire([(440, 135), (470, 135), (470, 102), (500, 102)])
    f.wire([(590, 90), (680, 90)], label="rst → every block", at=(598, 84))
    B(f, 20, 210, 110, 40, "btn_run_n", ["S2"], "pin")
    B(f, 170, 210, 80, 40, "run_s1", [], "reg", z="dff", n="run_s1 (DFF)")
    B(f, 290, 210, 80, 40, "run_s2", [], "reg", z="dff", n="run_s2 (DFF)")
    B(f, 410, 200, 120, 60, "deb_reg", ["18 bit, +1", "while s2 ≠ stable"], "reg", z="carry", n="debounce counter (carry chain)")
    B(f, 570, 200, 120, 60, "run_stable", ["takes s2 when", "deb_reg all ones"], "reg", z="dff", n="run_stable (DFF)")
    B(f, 730, 200, 100, 60, "run_prev", ["1 cycle later"], "reg", z="dff", n="run_prev (DFF)")
    B(f, 730, 280, 230, 40, "btn_run_evt = prev ∧ ¬stable", [], z="lut4", n="edge detect (LUT4)")
    f.wire([(130, 230), (170, 230)])
    f.wire([(250, 230), (290, 230)])
    f.wire([(370, 230), (410, 230)])
    f.wire([(530, 230), (570, 230)])
    f.wire([(690, 230), (730, 230)])
    f.wire([(780, 260), (780, 280)])
    f.wire([(630, 260), (630, 300), (730, 300)])
    info = para("Reset is held for the first 15 cycles by a 4-bit power-on counter and whenever S1 is pressed. Both buttons pass through a two-flop synchronizer (they are asynchronous to the 27 MHz clock); S2 also needs to be stable for 2<sup>18</sup> cycles (9.7 ms) before <code>run_stable</code> follows it. A press is the falling edge of <code>run_stable</code>, which starts a silent run of DEMO_SWEEPS = 128 sweeps.")
    add("resetbtn", f, "Reset and buttons", 4, info)


def v_uartrx():
    f = nf("ur", 980, 300, "uart_rx: synchronizer, bit timer, FSM, shift register")
    B(f, 20, 40, 100, 40, "rxd", ["pin 18"], "pin")
    B(f, 160, 40, 70, 40, "s1", [], "reg", z="dff", n="s1 (DFF)")
    B(f, 270, 40, 70, 40, "s2", [], "reg", z="dff", n="s2 (DFF)")
    B(f, 390, 20, 250, 110, "receiver FSM", ["IDLE: wait for low", "START: confirm at mid-bit", "DATA: 8 samples, 1 per bit", "STOP: valid if line high"], "reg")
    B(f, 390, 170, 170, 56, "cnt_reg", ["16 bit, counts down"], "reg", z="carry", n="bit timer (carry chain)")
    B(f, 600, 170, 110, 56, "bit_reg", ["3 bit"], "reg", z="register", n="bit_reg")
    B(f, 700, 40, 200, 56, "data_reg", ["{s2, data[7:1]}"], "reg", z="register", n="data_reg (8 DFF)")
    B(f, 750, 170, 150, 56, "vld_reg", ["1-cycle pulse"], "reg", z="dff", n="vld_reg (DFF)")
    f.wire([(120, 60), (160, 60)])
    f.wire([(230, 60), (270, 60)])
    f.wire([(340, 60), (390, 60)])
    f.wire([(640, 68), (700, 68)], label="shift", at=(648, 62))
    f.wire([(515, 130), (515, 170)], "c")
    f.wire([(900, 68), (950, 68)], label="out_data", at=(904, 62))
    f.wire([(900, 198), (950, 198)], label="valid", at=(904, 192))
    f.text(390, 270, "CLKS_PER_BIT = 234 → 115,385 baud at 27 MHz; a frame with a bad stop bit is dropped", "lbl")
    info = para("8N1 receiver with mid-bit sampling. After the line falls, the FSM waits half a bit and checks the start bit is still low, then samples every 234 cycles, shifting each bit in from the top so the byte ends LSB-first in <code>data_reg</code>.") + f'<div class="mini">{bm.fig_uart().svg()}</div>' + sigtab("uart_rx.v, uart_tx.v")
    add("uartrx", f, "uart_rx", 4, info)


def v_uarttx():
    f = nf("ut", 980, 260, "uart_tx: 10-bit shift register, bit timer, busy flag")
    B(f, 20, 40, 120, 50, "in_data", ["8 bit, in_valid"], "pin")
    B(f, 200, 30, 330, 70, "shift_reg", ["{1, data, 0} loaded on accept", "shifts right every 234 clocks"], "reg", z="register", n="shift_reg (10 DFF)")
    B(f, 600, 30, 120, 70, "txd mux", ["busy ? shift[0]", ": 1 (idle)"], z="lut4", n="txd mux (LUT4)")
    B(f, 200, 150, 170, 56, "cnt_reg", ["16 bit bit timer"], "reg", z="carry", n="bit timer (carry chain)")
    B(f, 410, 150, 120, 56, "bits_reg", ["4 bit, 10 → 0"], "reg", z="register", n="bits_reg")
    B(f, 600, 150, 120, 56, "busy_reg", ["in_ready = ¬busy"], "reg", z="dff", n="busy_reg (DFF)")
    f.wire([(140, 65), (200, 65)])
    f.wire([(530, 65), (600, 65)])
    f.wire([(720, 65), (820, 65)], label="txd → pin 17", at=(728, 59))
    f.wire([(720, 178), (820, 178)], "c", label="in_ready", at=(728, 172))
    add("uarttx", f, "uart_tx", 4, para("Accepts a byte when idle, loads {stop, data, start} into a 10-bit shift register and shifts one bit out every 234 cycles. The line idles high.") + sigtab("uart_rx.v, uart_tx.v"))


def state(f, x, y, name, w=74, hl=False):
    f.raw(f'<rect x="{x}" y="{y}" width="{w}" height="30" rx="15" class="bx st{" hl" if hl else ""}"/>')
    f.text(x + w / 2, y + 19, name, "stt", "middle")


def v_cmdfsm():
    f = nf("cf", 1100, 600, "Command FSM state graph and its argument and reply shift registers")
    state(f, 30, 250, "IDLE", hl=True)
    state(f, 30, 150, "ARGS")
    state(f, 170, 250, "EXEC", hl=True)
    f.wire([(67, 250), (67, 180)], label="byte, nargs > 0", at=(74, 220))
    f.wire([(104, 165), (150, 165), (150, 258), (170, 258)], label="last arg", at=(110, 158))
    f.wire([(104, 272), (170, 272)], label="nargs = 0", at=(108, 290))
    groups = [("R", ["RUN1", "RUNW"]), ("S", ["STRW", "STRS"]), ("N", ["NOISE"]), ("C", ["CLR1", "CLRW"]),
              ("D", ["DADDR", "DWAIT", "DLAT"]), ("U", ["UADDR", "UWAIT", "ULAT"]), ("P", ["PING2", "PING3"]), ("B", ["BRST"])]
    f.wire([(244, 265), (300, 265)], arrow=None)
    f.wire([(300, 40), (300, 500)], arrow=None)
    for i, (cmd, sts) in enumerate(groups):
        y = 30 + i * 60
        f.dot(300, y + 15)
        f.wire([(300, y + 15), (340, y + 15)], label=f"'{cmd}'", at=(306, y + 9))
        for j, sname in enumerate(sts):
            state(f, 340 + j * 110, y, sname)
            if j:
                f.wire([(340 + j * 110 - 36, y + 15), (340 + j * 110, y + 15)])
        xe = 340 + len(sts) * 110 - 36
        f.wire([(xe, y + 15), (720, y + 15)], "c", arrow=None)
    f.wire([(720, 30), (720, 500)], "c", arrow=None)
    f.wire([(720, 265), (780, 265)], "c", label="ret_reg", at=(726, 259))
    state(f, 780, 250, "SEND", hl=True)
    state(f, 900, 190, "SENDA")
    state(f, 900, 310, "SENDB")
    f.wire([(854, 258), (900, 205)], label="tx_ready", at=(830, 222))
    f.wire([(937, 220), (937, 310)])
    f.wire([(900, 325), (854, 272)])
    state(f, 1000, 250, "ACK")
    f.wire([(1037, 280), (1037, 505), (16, 505), (16, 265), (30, 265)], "c", label="'K' sent → IDLE", at=(560, 499))
    f.wire([(854, 265), (1000, 265)], "c", label="ret", at=(950, 259))
    B(f, 30, 520, 420, 56, "args_reg", ["48 bit shift register ← rx_data, last byte in [7:0]"], "reg", z="register", n="args_reg (48 DFF)")
    B(f, 480, 520, 300, 56, "word_reg", ["WW bits → tx_data, LSB first"], "reg", z="register", n="word_reg")
    B(f, 810, 520, 260, 56, "state_reg", ["5 bit, 23 states"], "reg", z="register", n="state_reg (5 DFF)")
    info = (para("Every command is one byte plus 0 to 6 argument bytes. <code>nargs_of()</code> gives the count, bytes shift into <code>args_reg</code>, and in EXEC one strobe fires while the port wires read fixed slices of it. Replies go through a one-level subroutine: S_SEND shifts <code>word_reg</code> out byte by byte, then jumps to <code>ret_reg</code>. Every command ends with 'K'; unknown bytes return '?'.")
            + "<h4>Byte → wire</h4>" + table(["Command", "Args", "args_reg slice → wire", "Lands in"], bm.HOST_ROWS))
    add("cmdfsm", f, "Command FSM", 4, info)


def v_sched():
    f = nf("sc", 1000, 320, "schedule_ctrl: sweep counter, step test, three beta sources and the beta register")
    B(f, 20, 40, 130, 50, "sweep_done", ["from engine"], "pin")
    B(f, 190, 30, 150, 70, "cnt_reg", ["16 bit, +1 per sweep"], "reg", z="register", n="cnt_reg")
    B(f, 380, 30, 170, 70, "cnt+1 < sps ?", ["16-bit compare", "critical path"], z="carry", n="step compare (carry chain)", hl=True)
    B(f, 190, 140, 150, 56, "β + Δ", ["sat at β1"], z="carry", n="β + Δ adder (carry chain)")
    B(f, 190, 220, 150, 56, "table_mem", ["16 × 8"], "mem", z="ssram", n="β table (SSRAM)")
    B(f, 20, 220, 130, 56, "k_reg", ["4 bit, holds at 15"], "reg", z="register", n="k_reg")
    B(f, 420, 150, 150, 110, "mode mux", ["0: β0", "1: linear", "2: table[k]"], z="lut4", n="mode mux (LUT4)")
    B(f, 620, 170, 140, 70, "beta_reg", ["8 bit, Q4.4"], "reg", z="register", n="beta_reg")
    f.wire([(150, 65), (190, 65)])
    f.wire([(340, 65), (380, 65)])
    f.wire([(550, 65), (690, 65), (690, 170)], "c", label="step (CE)", at=(560, 59))
    f.wire([(340, 168), (420, 168)])
    f.wire([(340, 248), (420, 248)])
    f.wire([(150, 248), (190, 248)])
    f.wire([(570, 205), (620, 205)])
    f.wire([(760, 205), (860, 205)], label="beta → engine", at=(766, 199), width=8, wat=(820, 205))
    f.wire([(690, 240), (690, 300), (265, 300), (265, 276)], "c", arrow=None)
    f.wire([(720, 240), (720, 290), (130, 290), (130, 168), (190, 168)], label="β (feedback)", at=(300, 286))
    info = para("Holds β in an 8-bit register. Mode 0 keeps β0; mode 1 adds Δ every SPS sweeps and saturates at β1; mode 2 (default) steps through a 16-entry table every SPS sweeps and holds at entry 15. Power-on: mode 2, SPS = 8, so DEMO_SWEEPS = 128 covers the table.",
                "In the dense and sparse-16 builds the worst timing path in the whole design runs from <code>sps_reg</code> through this 16-bit compare to the clock enable of <code>beta_reg</code> (13.8 ns, 72.5 MHz).") + sigtab("schedule_ctrl.v")
    add("sched", f, "schedule_ctrl", 4, info)


def v_engine():
    f = nf("en", 1000, 400, "The three sampling engines selected by generate parameters")
    f.text(20, 24, "one of these is built, chosen by SPARSE and TEMPER_K", "lbl")
    B(f, 20, 50, 300, 320, "pbit_array", ["SPARSE = 0", "one shared engine", "N + 18 cycles per update"], z="dense", n="dense engine · pbit_array", hl=True)
    B(f, 350, 50, 300, 320, "temper_ctrl", ["TEMPER_K > 1", "K replicas share pbit_array", "Barker swaps"], z="temper", n="parallel tempering · temper_ctrl")
    B(f, 680, 50, 300, 320, "sparse_array", ["SPARSE = 1", "P lanes in lock step", "D + P + 15 cycles per round"], z="sparse", n="sparse lanes · sparse_array")
    info = para("The engine slot of top.v is filled at synthesis time. All three use the same p-bit block; they differ in how fields are gathered and how many updates run at once.") + table(["Operation", "Cycles", "Value", "Notes"], bm.TIMING_ROWS[:6])
    add("engine", f, "Sampling engine", 4, info)


def v_dense():
    f = bm.fig_dense()
    f.portals = []
    P(f, 20, 52, 100, 52, "register", "k_reg")
    P(f, 150, 46, 130, 64, "ssram", "omem (SSRAM)")
    P(f, 330, 46, 130, 64, "ssram", "hmem (SSRAM)")
    P(f, 330, 140, 130, 44, "register", "node_reg")
    P(f, 510, 56, 110, 44, "register", "j_reg")
    P(f, 670, 46, 160, 64, "bsram", "wmem (BSRAM)")
    P(f, 510, 196, 110, 44, "lut4", "N:1 spin mux (LUT4 tree)")
    P(f, 670, 190, 160, 76, "wmac", "weight_mac")
    P(f, 890, 180, 150, 100, "pbit", "pbit")
    P(f, 890, 340, 150, 52, "lfsr0", "noise_bank lane 0")
    P(f, 650, 340, 190, 52, "lut4", "spin write decoder (LUT4)")
    P(f, 430, 340, 170, 52, "register", "spins_reg")
    P(f, 20, 220, 260, 100, "fsm_dense", "control FSM")
    info = (para("A sweep visits all N nodes in the order stored in <code>omem</code> (colour block 0 first). For each node: read index and bias, stream row i of J through the conditional adder (spins are ±1, so the multiply is a sign flip), run the p-bit, write the spin straight back. The loop from <code>spins_reg</code> to the spin mux is the Markov chain.")
            + tl("tl2", [("ORD", 1), ("BIAS", 1), ("MAC", 1), ("MACW", 18), ("PBW", 12), ("WR", 1)], "dense update, 34 cycles", "N + 18 = 34 at N = 16", hl=("MACW",), heat=("PBW",))
            + facts([("Update", "N + 18 cycles"), ("Sweep at N = 16", "544 cycles · 49.6 k/s"), ("Sweep at N = 128", "18,688 cycles · 1.44 k/s")])
            + sigtab("pbit_array.v"))
    add("dense", f, "pbit_array · dense engine", 4, info)


def v_sparse():
    f = bm.fig_sparse()
    f.portals = []
    P(f, 20, 46, 130, 40, "register", "ptr_reg")
    P(f, 20, 104, 130, 40, "register", "slot_reg")
    for l in range(4):
        X = 170 + l * 215
        P(f, X + 30, 50, 160, 46, "ssram", f"nmem[{l}] (SSRAM)")
        P(f, X + 30, 120, 160, 46, "ssram", f"wmem[{l}] (SSRAM)")
        P(f, X + 30, 190, 160, 46, "wmac", f"weight_mac lane {l}")
        P(f, X + 30, 260, 160, 46, "pbit", f"pbit lane {l}")
        P(f, X + 90, 330, 100, 36, f"lfsr{l}", f"lfsr lane {l}")
    P(f, 170, 400, 860, 46, "lut4", "commit mux (LUT4)")
    P(f, 430, 476, 340, 40, "register", "spins_reg")
    info = (para("For graphs of degree ≤ D. The host arranges each colour block into rounds of P nodes (one per lane) and pads short rounds with valid = 0. Lanes share the round pointer, the slot counter and the spin register; same-round nodes are never coupled, so a one-per-cycle commit equals a sequential sweep.",
                 "NOISE_CTL = 1: with <code>nz_tau</code> a lane whose draw ≥ nz_q keeps its spin; with <code>nz_share</code> every lane reads lane 0's draw.")
            + tl("tl3", [("NODE", 1), ("MAC", 1), ("MACW", 4), ("PB", 1), ("PBW", 12), ("CMT", 4)], "sparse round, 23 cycles", "D + P + 15 = 23", hl=("MACW",), heat=("PBW",))
            + finding(3) + sigtab("sparse_array.v"))
    add("sparse", f, "sparse_array · P lanes", 4, info)


def v_temper():
    f = bm.fig_temper()
    f.portals = []
    for k in range(4):
        y = 50 + k * 46
        P(f, 30, y, 110, 36, "register", f"e[{k}] (24 DFF)")
        P(f, 170, y, 140, 36, "register", f"bank[{k}]")
    P(f, 350, 50, 60, 174, "lut4", "replica mux (LUT4)")
    P(f, 470, 60, 210, 130, "dense", "shared pbit_array")
    P(f, 30, 270, 200, 52, "carry", "energy add/sub (carry chain)")
    P(f, 740, 50, 160, 56, "register", "ladder")
    P(f, 740, 130, 160, 56, "register", "rung / rep_at labels")
    P(f, 470, 330, 170, 52, "carry", "Δe subtract (carry chain)")
    P(f, 740, 230, 160, 52, "carry", "Δβ subtract (carry chain)")
    P(f, 700, 330, 140, 52, "lut4", "sign fold + saturate (LUT4)")
    P(f, 900, 310, 170, 64, "pbit", "swap pbit")
    P(f, 910, 404, 150, 40, "lfsr1", "swap LFSR (lane 1)")
    info = (para("K ≤ 4 replicas share one dense array. A round sweeps each replica once at its rung's β, then tries swaps between neighbouring rungs (pairs 0-1 and 2-3 on even rounds, 1-2 on odd). Acceptance is Barker's rule σ((β<sub>a</sub> − β<sub>b</sub>)(E<sub>x</sub> − E<sub>y</sub>)) decided by a second p-bit. Energies are tracked incrementally from the update tap; a swap exchanges labels, never spin banks.")
            + facts([("Default ladder", "β = 3.0, 2.0, 1.25, 0.75"), ("Round, K=4, N=16", "≈ 2,210 cycles"), ("Fmax", "54.9 MHz (slowest build)")])
            + finding(2) + fmt_rows(["e[k]"]) + sigtab("temper_ctrl.v"))
    add("temper", f, "temper_ctrl · parallel tempering", 4, info)


def v_readout():
    f = nf("ro", 1000, 330, "readout_counter: histogram read-modify-write, magnetisation counters, sample counter")
    B(f, 20, 30, 130, 50, "sample", ["= swd, spins"], "pin")
    B(f, 190, 30, 120, 50, "idx_reg", ["spins[11:0]"], "reg", z="register", n="idx_reg")
    B(f, 350, 20, 200, 90, "hist_mem", ["4096 × 32", "8 BSRAM"], "mem", z="bsram", n="hist_mem (BSRAM)")
    B(f, 590, 30, 120, 50, "hq_reg", ["read data"], "reg", z="register", n="hq_reg")
    B(f, 750, 30, 110, 50, "+1", ["32 bit"], z="carry", n="histogram incrementer (carry chain)")
    f.wire([(150, 55), (190, 55)])
    f.wire([(310, 55), (350, 55)], label="addr", at=(316, 49))
    f.wire([(550, 55), (590, 55)])
    f.wire([(710, 55), (750, 55)])
    f.wire([(860, 55), (900, 55), (900, 130), (450, 130), (450, 110)], label="write back (S_WR)", at=(600, 146))
    B(f, 20, 180, 230, 60, "RMW FSM", ["IDLE → RD → WR (3 cycles)", "CLEAR: 4096 cycles"], "reg")
    B(f, 290, 170, 270, 80, "up[0 .. N−1]", ["N × 32-bit counters (DFF)", "+1 when spins[i] = 1"], "reg", z="register", n="up counters")
    B(f, 600, 180, 150, 60, "N incrementers", ["32 bit each"], z="carry", n="up incrementer (carry chain)")
    B(f, 790, 180, 170, 60, "ns_reg", ["32 bit n_samples"], "reg", z="register", n="ns_reg")
    f.text(290, 280, "UP_RAM = 1: counters move into one BSRAM, updated by an N + 1-cycle pipelined scan", "lbl")
    info = para("Counts samples on chip so the host need not stream every sweep. The histogram is indexed by the low 12 spins; the up counters record how often each spin was +1.") + finding(1) + sigtab("readout_counter.v")
    add("readout", f, "readout_counter", 4, info)


def v_pmodetop():
    f = nf("pt0", 1000, 300, "pmode_top: button toggle, step pacer, pmode_array and LED sign display")
    B(f, 20, 40, 120, 50, "btn_run_n", ["S2"], "pin")
    B(f, 180, 40, 160, 50, "sync + debounce", ["toggles on_reg"], "reg", z="register", n="on_reg / debounce")
    B(f, 180, 150, 160, 56, "pace_reg", ["20 bit, +1 per clock"], "reg", z="carry", n="pace counter (carry chain)")
    B(f, 390, 150, 140, 56, "tick", ["pace = 0"], z="lut4", n="tick (LUT4)")
    B(f, 590, 30, 240, 190, "pmode_array", ["M = 4 soft spins", "Langevin step: 61 cycles"], z="pmode", n="pmode_array", hl=True)
    f.wire([(140, 65), (180, 65)])
    f.wire([(340, 178), (390, 178)])
    f.wire([(340, 65), (560, 65), (560, 100), (590, 100)], "c", label="run = on ∧ tick ∧ ¬busy", at=(350, 59))
    f.wire([(530, 178), (560, 178), (560, 110)], "c", arrow=None)
    f.wire([(830, 125), (930, 125)], label="sign of x_k", at=(836, 119))
    for i in range(4):
        f.raw(f'<circle cx="955" cy="{75 + i * 34}" r="9" class="led"/>')
    f.text(590, 260, "one Langevin step every 2^20 clocks = 25.7 steps/s, so the LEDs show the dynamics", "lbl")
    add("pmodetop", f, "pmode_top bitstream", 5, para("Stand-alone board top for continuous Langevin p-modes. Couplings, fields and initial states come from files; there is no host link. S2 toggles running; LED k is lit while x<sub>k</sub> ≥ 0.") + sigtab("pmode_array.v, pmode_top.v"))


def v_pmode():
    f = bm.fig_pmode()
    f.portals = []
    for args in [(20, 40, 130, 60, "register", "xmem (FF memory)"), (190, 40, 150, 60, "carry", "round + saturate (carry chain)"),
                 (380, 40, 150, 60, "sigrom", "tanh via sigmoid ROM"), (570, 40, 130, 60, "carry", "p − 32768 (carry chain)"),
                 (740, 40, 130, 60, "register", "smem (FF memory)"), (20, 150, 150, 60, "ssram", "hmem (SSRAM)"),
                 (190, 150, 150, 60, "ssram", "amem (SSRAM)"), (380, 150, 150, 60, "mult", "aq × sq (MULT18X18)"),
                 (570, 150, 170, 60, "register", "acc_reg (40 DFF)"), (20, 300, 190, 44, "register", "f_reg"),
                 (20, 360, 190, 44, "mult", "1 − s² (MULT18X18)"), (20, 420, 190, 44, "mult", "x² (MULT18X18)"),
                 (230, 300, 190, 44, "mult", "f·g (MULT18X18)"), (230, 420, 190, 44, "mult", "x(x² − 1) (MULT18X18)"),
                 (440, 420, 190, 44, "mult", "λ·q (MULT18X18)"), (20, 480, 190, 44, "lfsr0", "lfsr lane 0"),
                 (230, 480, 190, 44, "gaussrom", "Gaussian ROM"), (440, 480, 190, 44, "register", "xi_reg"),
                 (650, 480, 190, 44, "mult", "σ·ξ (MULT18X18)"), (860, 340, 190, 70, "register", "xnew")]:
        P(f, *args)
    info = (para("Continuous soft spins in a double well under overdamped Langevin dynamics: x′ = x − 2<sup>−k</sup> dV/dx + σξ, with dV/dx<sub>i</sub> = −(1 − s<sub>i</sub>²)(Σ A<sub>ij</sub> s<sub>j</sub> + h<sub>i</sub>) + λ x<sub>i</sub>(x<sub>i</sub>² − 1) and s = tanh x. Every site updates from the old state (Jacobi). tanh comes from the p-bit sigmoid ROM; ξ from an inverse-CDF ROM.")
            + tl("tl4", [("tanh x4", 8), ("site 0", 13), ("site 1", 13), ("site 2", 13), ("site 3", 13), ("CMT", 1)], "p-mode step, 61 cycles", "2M + M(2M+5) + 1 = 61", hl=("tanh x4",))
            + fmt_rows(["p-mode", "tanh", "λ", "σ", "ε"]) + sigtab("pmode_array.v, pmode_top.v"))
    add("pmode", f, "pmode_array · Langevin", 4, info)


def v_pgate():
    f = nf("pg", 1000, 340, "pgate: clamps rewrite the bias RAM, then the shared array runs")
    B(f, 20, 40, 150, 56, "start", ["clamp_mask, clamp_val"], "pin")
    B(f, 210, 40, 150, 56, "mask_reg, val_reg", ["N bits each"], "reg", z="register", n="clamp registers")
    B(f, 210, 140, 150, 56, "hbase[i]", ["gate biases h0"], "mem", z="ssram", n="hbase ROM")
    B(f, 400, 70, 190, 100, "hval mux", ["mask ? (val ? 0x7F", ": 0x80) : h0"], z="lut4", n="clamp mux (LUT4)")
    B(f, 630, 30, 340, 200, "pbit_array", ["gate J and h, N = 3 for AND", "runs n_sweeps after the rewrite"], z="dense", n="pbit_array (gate)", hl=True)
    B(f, 20, 250, 340, 56, "FSM", ["IDLE → BIAS (N cycles) → RUN → WAIT"], "reg")
    B(f, 630, 260, 340, 56, "temper_ctrl", ["only when K > 1"], z="temper", n="temper_ctrl (K > 1)")
    f.wire([(170, 68), (210, 68)])
    f.wire([(360, 68), (380, 68), (380, 100), (400, 100)])
    f.wire([(360, 168), (380, 168), (380, 140), (400, 140)])
    f.wire([(590, 120), (630, 120)], label="h_data", at=(594, 114))
    add("pgate", f, "pgate · invertible logic", 5, para("A Boolean gate run as a Boltzmann sampler: J and h make the truth-table rows the lowest-energy states. Clamp the inputs and it runs forward; clamp the output and it runs backward. Clamping is a bias offset: +127 or −128 written into the bias RAM.") + sigtab("pgates.v, denoise_demo.v"))


def v_denoise():
    f = nf("dd", 1000, 340, "denoise_demo: T steps, each rewriting the biases and running K sweeps")
    B(f, 20, 40, 140, 56, "x_t", ["noisy image"], "pin")
    B(f, 200, 40, 150, 56, "xt_reg", ["N bits, x^t"], "reg", z="register", n="xt_reg")
    B(f, 200, 140, 150, 56, "hbase, gtab", ["h0 and Γ_t/2"], "mem", z="ssram", n="bias and gamma tables")
    B(f, 390, 80, 190, 90, "h ± g_t, sat8", ["sign from x^t_i"], z="carry", n="clamp adder (carry chain)")
    B(f, 620, 30, 350, 200, "pbit_array", ["N = 16, K = 16 sweeps per step", "β = 1.0"], z="dense", n="pbit_array (denoise)", hl=True)
    B(f, 20, 250, 560, 56, "step loop", ["LOAD → BIAS (N) → RUN → WAIT → STEP, t = T … 1"], "reg")
    f.wire([(160, 68), (200, 68)])
    f.wire([(350, 68), (370, 68), (370, 110), (390, 110)])
    f.wire([(350, 168), (370, 168), (370, 140), (390, 140)])
    f.wire([(580, 125), (620, 125)])
    f.wire([(795, 230), (795, 278), (580, 278)], label="spins → x^(t−1)", at=(600, 272))
    add("denoise", f, "denoise_demo · denoising chain", 5, para("A toy Denoising Thermodynamic Model chain: for t = T … 1, clamp x<sup>t</sup> as a bias offset ±Γ<sub>t</sub>/2, load it as the start state, run K Gibbs sweeps and take the result as x<sup>t−1</sup>. One array is reprogrammed at every step.") + sigtab("pgates.v, denoise_demo.v"))


# ================================================================= L3 mechanisms
def v_wmac():
    f = nf("wm", 980, 300, "weight_mac: conditional negate and add, load mux, accumulator register")
    B(f, 20, 30, 140, 50, "h", ["8 bit bias"], "pin")
    B(f, 20, 140, 140, 50, "w", ["8 bit J_ij"], "pin")
    B(f, 20, 220, 140, 50, "s", ["spin s_j"], "pin")
    B(f, 210, 140, 150, 50, "w ⊕ ¬s", ["invert when s = 0"], z="lut4", n="conditional invert (LUT4)")
    B(f, 410, 110, 170, 80, "14-bit add", ["Cin = ¬s", "acc ± J"], z="carry", n="14-bit adder (carry chain)", hl=True)
    B(f, 630, 50, 130, 80, "load mux", ["load ? h : sum"], z="lut4", n="load mux (LUT4)")
    B(f, 800, 60, 150, 70, "acc_reg", ["14 DFF = field"], "reg", z="register", n="acc_reg")
    f.wire([(160, 55), (600, 55), (600, 70), (630, 70)], label="sign-extended h", at=(200, 49))
    f.wire([(160, 165), (210, 165)])
    f.wire([(160, 245), (285, 245), (285, 190)], "c")
    f.wire([(285, 245), (495, 245), (495, 190)], "c", label="¬s → Cin", at=(380, 260))
    f.wire([(360, 165), (410, 165)])
    f.wire([(580, 150), (605, 150), (605, 110), (630, 110)])
    f.wire([(760, 95), (800, 95)])
    f.wire([(950, 95), (970, 95), (970, 22), (440, 22), (440, 110)], label="acc", at=(700, 16))
    info = para("Spins are ±1, so the multiply J·s is a sign flip: s = 0 inverts J and sets carry-in to 1 (two's-complement negate), s = 1 passes it. One neighbour per cycle; <code>load</code> starts the sum at h. This is the digital version of the source chip's resistor network over sign-corrected inputs (y = x XOR s).") + sigtab("coupling_row.v, weight_mac.v")
    add("wmac", f, "weight_mac", 3, info)


def v_pbit():
    f = bm.fig_pbit()
    f.portals = []
    P(f, 70, 52, 150, 52, "register", "mcand_reg (23 DFF)")
    P(f, 70, 172, 150, 52, "register", "bsh_reg (8 DFF)")
    P(f, 270, 62, 90, 60, "carry", "23-bit adder (carry chain)")
    P(f, 410, 62, 140, 60, "register", "prod_reg (23 DFF)")
    P(f, 600, 62, 140, 60, "carry", "rounding adder (carry chain)")
    P(f, 780, 62, 110, 60, "register", "logit_reg (10 DFF)")
    P(f, 940, 56, 150, 72, "sigrom", "sigmoid ROM")
    P(f, 940, 200, 150, 60, "carry", "rnd < p comparator (carry chain)")
    P(f, 760, 204, 120, 52, "dff", "bit_reg (DFF)")
    P(f, 270, 232, 380, 64, "fsm_pbit", "pbit FSM")
    info = (para("Every sample in every engine passes through here. β·field arrives with 8 fraction bits; doubling and keeping 5 is a right shift by WFRAC − 2 = 2 with round-half-up. The 10-bit logit, offset by 512, addresses the ROM. The comparator is where the noise enters: P(bit = 1) = p / 65536 = σ(2β·field).")
            + tl("tl1", [("IDLE", 1), ("MUL", 8), ("LOGIT", 1), ("LUT", 1), ("CMP", 1)], "pbit, 12 cycles", "done in cycle 12", hl=("MUL",), heat=("CMP",))
            + fmt_rows(["field", "β", "logit", "<code>p</code>", "rnd"]) + sigtab("pbit.v"))
    add("pbit", f, "pbit · one p-bit decision", 3, info)


def curve_view(fid, title, xs, ys, xlab, ylab, xt, yt, portal_name, aria):
    f = nf(fid, 980, 380, aria)
    x0, y0, w, h = 70, 20, 560, 300
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    X = lambda v: x0 + (v - xmin) / (xmax - xmin) * w
    Y = lambda v: y0 + h - (v - ymin) / (ymax - ymin) * h
    for v, lab in xt:
        f.raw(f'<path d="M{X(v):.1f},{y0} L{X(v):.1f},{y0+h}" class="grid"/>')
        f.text(X(v), y0 + h + 16, lab, "axn", "middle")
    for v, lab in yt:
        f.raw(f'<path d="M{x0},{Y(v):.1f} L{x0+w},{Y(v):.1f}" class="grid"/>')
        f.text(x0 - 6, Y(v) + 4, lab, "axn", "end")
    pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(xs[::4], ys[::4]))
    f.raw(f'<polyline points="{pts}" class="curve"/>')
    f.text(x0 + w / 2, y0 + h + 34, xlab, "lbl", "middle")
    f.text(x0, y0 - 6, ylab, "lbl")
    B(f, 690, 40, 260, 220, "BSRAM", ["1024 × 16 words", "one 18 Kbit block", "registered read"], "mem", z="bsram", n=portal_name)
    f.wire([(630, 150), (690, 150)])
    return f


def v_sigrom():
    xs = list(range(1024))
    ys = [min(65535, round(65536 / (1 + math.exp(-(a - 512) / 32)))) for a in xs]
    f = curve_view("sg", "", xs, ys, "address = logit + 512 (logit/32 = 2β·field)", "p (of 65536)",
                   [(0, "0 (−16)"), (256, "256 (−8)"), (512, "512 (0)"), (768, "768 (+8)"), (1023, "1023")],
                   [(0, "0"), (32768, "32768"), (65535, "65535")], "sigmoid table (BSRAM)", "Sigmoid table contents: p against address")
    info = para("p = round(65536 · σ(logit/32)), clipped to 65535, stored at address logit + 512 (offset binary: just invert the sign bit). One-cycle registered read, so it maps onto one Gowin block RAM. USE_PWL = 1 replaces it with four shift-and-add segments (49 LUT4 + 43 ALU, max error 0.019).", "The same ROM gives tanh in pmode_array: tanh x = 2σ(2x) − 1.") + sigtab("sigmoid_lut.v, sigmoid_pwl.v")
    add("sigrom", f, "sigmoid ROM", 3, info)


def v_gaussrom():
    nd = NormalDist()
    xs = list(range(1024))
    ys = [nd.inv_cdf((a + 0.5) / 1024) for a in xs]
    f = curve_view("gr", "", xs, ys, "address = rnd[15:6] (uniform)", "ξ (Q4.12 → value)",
                   [(0, "0"), (256, "256"), (512, "512"), (768, "768"), (1023, "1023")],
                   [(-3, "−3"), (0, "0"), (3, "+3")], "Gaussian table (BSRAM)", "Inverse-CDF table: Gaussian value against address")
    add("gaussrom", f, "Gaussian inverse-CDF ROM", 3, para("A uniform 10-bit address picks a standard-normal quantile: inverse-transform sampling. It reuses the sigmoid_lut module with a different table file. (The curve here is a standard inverse CDF; the real table comes from model/export_vectors.py.)"))


LANES = [(0xB4BCD35C, "0x32D689B4"), (0xA3000000, "0x908F037F"), (0x80200003, "0x76479D26"), (0x80000057, "0xD43C16E9")]


def v_lfsr(lane):
    mask, seed = LANES[lane]
    f = nf(f"lf{lane}", 1100, 360, f"Galois LFSR lane {lane}")
    x0, cw, yc = 200, 22, 96
    f.text(20, 104, "state_reg", "ttl")
    f.text(20, 119, "32 DFF")
    for k in range(32):
        x = x0 + (31 - k) * cw
        f.raw(f'<rect x="{x}" y="{yc}" width="{cw-2}" height="26" class="bx cell"/>')
        P(f, x, yc, cw - 2, 26, "dff", f"state bit b{k} (DFF)")
        if k % 4 == 3 or k == 0:
            f.text(x + (cw - 2) / 2, yc + 40, f"b{k}", "axn", "middle")
    xb0 = x0 + 31 * cw + cw - 2
    taps = [k for k in range(32) if (mask >> k) & 1]
    xl = x0 + (31 - max(taps)) * cw + (cw - 2) / 2
    f.wire([(xb0, 109), (930, 109), (930, 58), (xl, 58)], arrow=None)
    for k in taps:
        x = x0 + (31 - k) * cw + (cw - 2) / 2
        f.dot(x, 58)
        f.raw(f'<circle cx="{x}" cy="76" r="6" class="xor"/><path d="M{x-6},76 L{x+6},76 M{x},70 L{x},82" class="w"/>')
        f.wire([(x, 58), (x, 70)], arrow=None)
        f.wire([(x, 82), (x, 96)])
    f.text(200, 30, f"one Galois step: s ← (s >> 1) ⊕ (s₀ ? MASK : 0)    MASK = 0x{mask:08X}, {len(taps)} taps", "lbl")
    f.text(936, 86, "b0 out", "lbl")
    B(f, 200, 160, 704, 52, "leap-16: 16 Galois steps unrolled into one XOR network",
      ["new state = A¹⁶·s ;  rnd[k] = s₀ after k steps (linear over GF(2))"], z="lut4", n="XOR network (LUT4s)")
    f.wire([(552, 122), (552, 160)], width=32, wat=(552, 150))
    B(f, 940, 150, 145, 80, "next-state mux", ["seed_load: seed", "en: leap", "else hold"], z="lut4", n="next-state mux (LUT4)")
    f.wire([(904, 186), (940, 186)], width=32, wat=(920, 186))
    f.wire([(1060, 150), (1060, 40), (180, 40), (180, 109), (200, 109)], label="state_next", at=(960, 52))
    f.wire([(552, 212), (552, 262), (700, 262)], "n", label="rnd → p-bit", at=(560, 256), width=16, wat=(552, 238))
    f.wire([(985, 290), (985, 230)], "c", label="en", at=(990, 284))
    f.wire([(1040, 290), (1040, 230)], "c", label="seed", at=(1045, 284))
    f.text(1012, 304, f"reset seed {seed}", "lbl", "middle")
    f.box(20, 250, 470, 100, "RO_JITTER = 1 (noise_bank option, hardware only)", [], "grp")
    for i, x in enumerate((44, 104, 164)):
        inv_symbol(f, x, 300)
        P(f, x - 2, 286, 36, 28, "inverter", f"ring stage {i} (inverter)")
        if i < 2:
            f.wire([(x + 32, 300), (x + 60, 300)], arrow=None)
    f.wire([(196, 300), (214, 300), (214, 330), (30, 330), (30, 300), (44, 300)], arrow=None)
    f.dot(214, 300)
    f.wire([(214, 300), (250, 300)])
    B(f, 250, 282, 50, 36, "s1", [], "reg", z="dff", n="sync s1 (DFF)")
    B(f, 320, 282, 50, 36, "s2", [], "reg", z="dff", n="sync s2 (DFF)")
    f.wire([(300, 300), (320, 300)])
    f.raw('<circle cx="410" cy="300" r="8" class="xor"/><path d="M402,300 L418,300 M410,292 L410,308" class="w"/>')
    f.wire([(370, 300), (402, 300)])
    f.text(424, 304, "rnd[0]", "lbl")
    cost = {0: "299 LUT4 + 186 MUX2", 1: "62 LUT4 + 9 MUX2", 2: "154 LUT4 + 90 MUX2", 3: "135 LUT4 + 61 MUX2"}[lane]
    used = {0: "pbit_array, sparse lane 0, raw 'N' lane, pmode", 1: "sparse lane 1, tempering swap", 2: "sparse lane 2", 3: "sparse lane 3"}[lane]
    info = (para("Each draw advances the LFSR 16 steps in one clock. Those 16 steps are linear over GF(2), so synthesis flattens them into a fixed XOR network whose size depends on where the taps sit. Lane k is seeded with SEED ⊕ 0x9E3779B9·(k+1).")
            + facts([("Mask", f"0x{mask:08X} ({len(taps)} taps)"), ("Reset seed", seed), ("Leap cost (Yosys)", cost), ("Used by", used)])
            + (finding(0) + finding(4) if lane == 0 else "") + sigtab("lfsr.v, noise_bank.v"))
    add(f"lfsr{lane}", f, f"LFSR noise lane {lane}", 3, info)


def v_fsm(tid, title, states, edges, w, h, note, info, sbits):
    f = nf(tid, w, h, title)
    for name, x, y in states:
        state(f, x, y, name, w=84, hl=(name == "IDLE"))
    for pts, lab, at in edges:
        f.wire(pts, label=lab, at=at)
    B(f, w - 230, h - 70, 210, 50, "state_reg", [f"{sbits} DFF, binary"], "reg", z="register", n="state_reg")
    f.text(20, h - 20, note, "lbl")
    add(tid, f, title, 3, info)


def v_fsms():
    v_fsm("fsm_dense", "pbit_array control FSM",
          [("IDLE", 20, 120), ("ORD", 160, 40), ("BIAS", 300, 40), ("MAC", 440, 40), ("MACW", 580, 40), ("PBW", 720, 120), ("WR", 440, 200), ("HOLD", 160, 200)],
          [([(104, 128), (160, 62)], "run", (110, 88)), ([(244, 55), (300, 55)], None, None), ([(384, 55), (440, 55)], None, None),
           ([(524, 55), (580, 55)], "start", (528, 49)), ([(664, 62), (740, 120)], "mac_done", (706, 84)),
           ([(740, 150), (524, 210)], "pb_done", (640, 168)), ([(440, 215), (104, 140)], "last sweep / stop", (230, 160)),
           ([(482, 200), (482, 140), (202, 140), (202, 70)], "next node", (300, 134)), ([(440, 222), (244, 222)], "hold", (330, 240)),
           ([(202, 200), (202, 172), (230, 172), (230, 70)], "¬hold", (236, 190))],
          900, 330, "N+18 cycles per update: ORD 1, BIAS 1, MAC 1, MACW N+2, PBW 12, WR 1",
          para("Eight states, binary-encoded in 3 flip-flops. The FSM issues <code>mac_start</code> and <code>pb_start</code> and writes the spin in the PBW → WR transition, registering <code>sweep_done</code> with the last write of a sweep. <code>hold</code> is sampled only at sweep boundaries."), 3)
    v_fsm("fsm_pbit", "pbit FSM",
          [("IDLE", 20, 60), ("MUL ×8", 180, 60), ("LOGIT", 340, 60), ("LUT", 500, 60), ("CMP", 660, 60)],
          [([(104, 75), (180, 75)], "start", (110, 69)), ([(264, 75), (340, 75)], "cnt = 7", (268, 69)), ([(424, 75), (500, 75)], None, None),
           ([(584, 75), (660, 75)], None, None), ([(702, 90), (702, 140), (62, 140), (62, 90)], "done, rnd_take", (300, 134))],
          900, 240, "12 cycles from start to done",
          para("Five states in 3 flip-flops. MUL runs the 8-cycle shift-add; LOGIT rounds and saturates; LUT reads the ROM; CMP compares with the noise draw and pulses <code>done</code> and <code>rnd_take</code> together so the noise lane advances exactly once per update."), 3)


# ================================================================= L2 datapath elements
def v_register():
    f = nf("rg", 920, 300, "An N-bit register: one D flip-flop per bit with shared clock, enable and reset")
    f.wire([(140, 40), (900, 40)], "b", arrow=None)
    f.text(20, 44, "D[7:0]", "ttl")
    for i in range(8):
        x = 150 + i * 95
        B(f, x, 80, 70, 80, f"DFF {7-i}", ["D → Q"], "reg", z="dff", n=f"bit {7-i} (DFF)")
        f.dot(x + 35, 40)
        f.wire([(x + 35, 40), (x + 35, 80)])
        f.wire([(x + 35, 160), (x + 35, 250)])
        f.dot(x + 35, 250)
        f.wire([(x + 10, 186), (x + 10, 160)], "c")
        f.wire([(x + 24, 202), (x + 24, 160)], "c")
        f.wire([(x + 56, 216), (x + 56, 160)])
    f.wire([(140, 186), (900, 186)], "c", arrow=None)
    f.wire([(140, 202), (900, 202)], "c", arrow=None)
    f.wire([(140, 216), (900, 216)], arrow=None)
    f.text(20, 190, "CE (enable)", "lbl")
    f.text(20, 206, "RST", "lbl")
    f.text(20, 220, "clk", "lbl")
    f.wire([(140, 250), (900, 250)], "b", arrow=None)
    f.text(20, 254, "Q[7:0]", "ttl")
    add("register", f, "Register (N × DFF)", 2, para("Every <code>*_reg</code> in the map is a row of D flip-flops, one per bit, sharing the clock, a clock enable and a synchronous reset (drawn here for 8 bits). The dense N = 16 build uses 1,326 flip-flops; the sparse N = 256 build 2,463 of the chip's 6,480."))


def v_carry():
    f = nf("cy", 520, 290, "Carry-chain cells")
    for k, yc in ((2, 60), (1, 140), (0, 220)):
        B(f, 40, yc - 20, 110, 40, f"LUT {k}", [f"P{k} = A{k} ⊕ B{k}"], z="lut4", n=f"LUT {k} (propagate)")
        f.raw(f'<rect x="285" y="{yc-16}" width="30" height="32" rx="3" class="gate"/>')
        P(f, 285, yc - 16, 30, 32, "passmux", f"carry mux {k}")
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
    add("carry", f, "Carry chain (ALU mode)", 2, para("Every +, −, &lt; and counter in the map. Each cell's LUT computes propagate P = A ⊕ B; a dedicated mux passes the carry straight up the column without general routing, which is why adders are fast. Subtraction inverts B and sets Cin = 1; the comparator rnd &lt; p is the borrow out of a 16-cell chain. Generic FPGA carry logic; Gowin's ALU mode is functionally equivalent."))


def v_bsram():
    f = nf("bs", 940, 420, "Block SRAM: row decoder, cell array, sense amplifiers and output register")
    B(f, 20, 40, 120, 260, "row decoder", ["address", "→ one word line"])
    rows, cols, cw, ch, x0, y0 = 10, 24, 26, 22, 180, 40
    for r in range(rows):
        y = y0 + r * (ch + 4)
        f.raw(f'<path d="M140,{y+ch/2} L{x0+cols*(cw+4)},{y+ch/2}" class="wl"/>')
        for cc in range(cols):
            x = x0 + cc * (cw + 4)
            f.raw(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="2" class="bx mem"/>')
            P(f, x, y, cw, ch, "sram6t", f"bit cell r{r} c{cc} (6T)")
    for cc in range(0, cols, 1):
        x = x0 + cc * (cw + 4) + cw / 2
        f.raw(f'<path d="M{x},{y0-6} L{x},{y0+rows*(ch+4)+6}" class="bl"/>')
    yb = y0 + rows * (ch + 4) + 10
    B(f, 180, yb, cols * (cw + 4) - 4, 40, "column mux + sense amplifiers", [])
    B(f, 180, yb + 60, 300, 44, "output register", ["p_reg / hq_reg"], "reg", z="register", n="output register")
    f.text(520, yb + 86, "18 Kbit per block: configured 1024 × 16 for the sigmoid ROM", "lbl")
    add("bsram", f, "Block SRAM (B-SRAM)", 2, para("An 18 Kbit block of 6T SRAM cells (18,432 bits ≈ 110,000 transistors in the cells alone) with a row decoder, bit lines, sense amplifiers and an output register. The address picks one word line; the selected cells drive their bit lines; the sense amplifiers resolve the small swing into bits. Drawn as a reduced 10 × 24 array; each square is one cell.", "Used here for the weight RAM (one 256 × 8 table fills a whole block), the sigmoid and Gaussian ROMs and the 4096 × 32 histogram (8 blocks)."))


def v_ssram():
    f = nf("ss", 860, 300, "Shadow SRAM: four LUT4s whose configuration bits are written at run time form a 16 × 4 RAM")
    B(f, 20, 40, 130, 50, "WAD, DI, WRE", ["write port"], "pin")
    B(f, 20, 180, 130, 50, "RAD", ["4-bit read address"], "pin")
    for i in range(4):
        x = 210 + i * 150
        B(f, x, 70, 120, 120, f"LUT4 bit {i}", ["16 × 1 RAM"], z="lut4", n=f"LUT4 bit {i}")
        f.wire([(x + 60, 40), (x + 60, 70)])
        f.wire([(x + 60, 190), (x + 60, 250)], label=f"DO{i}", at=(x + 66, 236))
    f.wire([(150, 40), (780, 40)], "c", arrow=None)
    f.wire([(150, 205), (190, 205), (190, 130), (210, 130)])
    add("ssram", f, "Shadow SRAM (RAM16SDP4)", 2, para("Small memories (order RAM, bias RAM, β table, sparse node and neighbour words) use LUTs as RAM: the 16 configuration bits of each LUT4 become a writable 16 × 1 memory, so four LUTs give 16 × 4. The read address drives the LUT inputs; a separate write port updates the bits."))


def v_mult():
    f = nf("mu", 940, 420, "Hard 18 × 18 multiplier: partial-product array, carry-save adder rows, final adder")
    B(f, 20, 40, 120, 50, "A[17:0]", [], "pin")
    B(f, 20, 120, 120, 50, "B[17:0]", [], "pin")
    for r in range(6):
        for cc in range(8):
            x = 200 + cc * 70 + r * 20
            y = 30 + r * 50
            f.raw(f'<circle cx="{x}" cy="{y+8}" r="4" class="dot"/>')
            if r > 0:
                f.raw(f'<rect x="{x-14}" y="{y+16}" width="40" height="24" rx="3" class="gate"/>')
                P(f, x - 14, y + 16, 40, 24, "fulladder", f"full adder r{r} c{cc}")
                f.text(x + 6, y + 32, "FA", "axn", "middle")
    B(f, 200, 340, 640, 44, "final carry-propagate adder", ["36-bit product"], z="carry", n="final adder")
    f.text(20, 230, "dots: AND gates a_i·b_j", "lbl")
    f.text(20, 246, "boxes: full adders", "lbl")
    f.text(20, 262, "(reduced drawing)", "lbl")
    add("mult", f, "MULT18X18 (DSP)", 2, para("A hard multiplier: every bit pair a<sub>i</sub>·b<sub>j</sub> is an AND (a partial product); rows of full adders reduce them without propagating carries; one final adder produces the 36-bit result. pmode_array's 40-bit multiplies map onto 5 of the chip's 20 MULT18X18 blocks after place and route. The p-bit sampler itself uses none: its only multiply is the serial shift-add."))


def v_fulladder():
    f = nf("fa", 700, 300, "Full adder at gate level")
    for lab, y in (("A", 50), ("B", 100), ("Cin", 200)):
        f.text(20, y + 4, lab, "ttl")
    B(f, 120, 60, 90, 50, "XOR", ["P = A⊕B"], z="lut4", n="XOR (as a LUT)")
    B(f, 300, 150, 90, 50, "XOR", ["S = P⊕Cin"])
    B(f, 120, 220, 90, 44, "AND", ["A·B"])
    B(f, 300, 60, 90, 44, "AND", ["P·Cin"])
    B(f, 480, 120, 90, 50, "OR", ["Cout"])
    f.wire([(50, 50), (120, 75)])
    f.wire([(50, 100), (120, 95)])
    f.wire([(210, 85), (260, 85), (260, 72), (300, 72)])
    f.wire([(260, 85), (260, 165), (300, 165)])
    f.wire([(60, 200), (280, 200), (280, 185), (300, 185)])
    f.wire([(280, 200), (280, 92), (300, 92)])
    f.wire([(50, 50), (90, 50), (90, 235), (120, 235)], arrow=None)
    f.wire([(390, 82), (440, 82), (440, 135), (480, 135)])
    f.wire([(210, 242), (440, 242), (440, 155), (480, 155)])
    f.wire([(390, 175), (420, 175), (420, 280), (600, 280)], label="S", at=(606, 284))
    f.wire([(570, 145), (640, 145)], label="Cout", at=(600, 139))
    add("fulladder", f, "Full adder", 1, para("S = A ⊕ B ⊕ Cin, Cout = A·B + Cin·(A ⊕ B). In a hard multiplier these are fixed CMOS cells (a mirror adder is 28 transistors); in the fabric the same function sits in LUTs plus the carry mux."))


def v_passmux():
    f = nf("pm2", 560, 260, "2:1 pass-transistor multiplexer with output buffer")
    f.wire([(30, 80), (196, 80)], arrow=None, label="I0", at=(30, 72))
    f.wire([(30, 180), (196, 180)], arrow=None, label="I1", at=(30, 172))
    mos(f, 220, 64, rot=90)
    mos(f, 220, 164, rot=90)
    P(f, 190, 36, 60, 52, "nmos_x", "pass NMOS (S_n)")
    P(f, 190, 136, 60, 52, "nmos_x", "pass NMOS (S)")
    f.text(220, 30, "S_n", "ttl", "middle")
    f.text(220, 130, "S", "ttl", "middle")
    f.wire([(244, 80), (300, 80), (300, 180), (244, 180)], arrow=None)
    f.dot(300, 130)
    f.wire([(300, 130), (340, 130)], arrow=None)
    inv_symbol(f, 340, 130)
    P(f, 336, 115, 40, 30, "inverter", "restoring buffer (inverter)")
    f.wire([(372, 130), (470, 130)], label="F", at=(480, 134))
    f.text(30, 240, "S selects which input conducts; the inverter restores the full logic level", "lbl")
    add("passmux", f, "Pass-transistor mux", 1, para("The basic switch of an SRAM FPGA: an NMOS transistor conducts when its gate is high, so two of them with complementary selects form a 2:1 mux. LUT trees, carry muxes and the programmable routing are all built from these, with inverters restoring the level an NMOS cannot pass fully."))


def v_dff():
    f = nf("df", 520, 200, "Master-slave D flip-flop")
    src = bm.tp_dff()
    f.p = src.p
    for (x, y) in ((66, 100), (196, 150), (296, 100), (426, 150)):
        P(f, x - 14, y - 11, 28, 22, "tg", "transmission gate")
    for (x, y) in ((108, 86), (128, 136), (338, 86), (358, 136)):
        P(f, x, y, 36, 28, "inverter", "inverter")
    add("dff", f, "D flip-flop", 1, para("Two latches in series. The master follows D while clk is low; on the rising edge it closes and the slave opens, so Q takes the value D had at the edge and holds it for the rest of the cycle. Each latch is a transmission gate feeding an inverter, with a second inverter and gate closing the loop. About 24–30 transistors with reset and clock enable."))


def v_inverter():
    f = nf("iv", 520, 230, "CMOS inverter")
    src = bm.tp_inverter()
    f.p = src.p
    P(f, 60, 49, 46, 54, "pmos_x", "PMOS")
    P(f, 60, 129, 46, 54, "nmos_x", "NMOS")
    add("inverter", f, "CMOS inverter", 0, para("Two transistors: the PMOS conducts when the input is low and pulls the output to VDD; the NMOS conducts when it is high and pulls it to ground. One of them is always off, so no current flows at rest. Three in a loop never settle: that ring oscillates at about 1/(6·t<sub>pd</sub>), the noise_bank's only physical entropy source."))


def v_tg():
    f = nf("tgv", 520, 250, "Transmission gate: NMOS and PMOS in parallel")
    f.wire([(30, 130), (200, 130)], arrow=None, label="A", at=(30, 122))
    f.wire([(200, 106), (200, 154)], arrow=None)
    f.dot(200, 130)
    mos(f, 260, 90, rot=90)
    mos(f, 260, 170, p=True, rot=-90)
    f.wire([(200, 106), (236, 106)], arrow=None)
    f.wire([(200, 154), (236, 154)], arrow=None)
    f.wire([(284, 106), (320, 106), (320, 154), (284, 154)], arrow=None)
    f.dot(320, 130)
    f.wire([(320, 130), (490, 130)], arrow=None, label="B", at=(480, 122))
    f.text(260, 62, "clk", "ttl", "middle")
    f.text(260, 206, "clk_n", "ttl", "middle")
    P(f, 228, 66, 64, 46, "nmos_x", "NMOS")
    P(f, 228, 150, 64, 46, "pmos_x", "PMOS")
    add("tg", f, "Transmission gate", 0, para("An NMOS and a PMOS in parallel with complementary gates. Together they pass both a full 0 (NMOS) and a full 1 (PMOS), which a single pass transistor cannot. Latches open and close with these."))


def v_lut4():
    f = nf("lt", 520, 250, "LUT4")
    f.text(24, 14, "16 config bits", "lbl")
    ys = [25 + 13 * i for i in range(16)]
    for i, y in enumerate(ys):
        f.raw(f'<rect x="30" y="{y-5}" width="22" height="10" class="bx mem"/>')
        P(f, 30, y - 5, 22, 10, "sram6t", f"config bit {i} (6T SRAM)")
        f.wire([(52, y), (90, y)], arrow=None)
    cur = ys
    xs = [90, 160, 230, 300]
    for li, x in enumerate(xs):
        nxt = []
        for j in range(0, len(cur), 2):
            a, b = cur[j], cur[j + 1]
            cc = (a + b) / 2
            hh = (b - a) / 2 + 6
            f.raw(f'<path d="M{x},{cc-hh:.1f} L{x+18},{cc-hh+5:.1f} L{x+18},{cc+hh-5:.1f} L{x},{cc+hh:.1f} Z" class="gate"/>')
            P(f, x, cc - hh, 18, 2 * hh, "passmux", f"2:1 mux level {li+1}")
            f.raw(f'<path d="M{x+9},{cc+hh-2.5:.1f} L{x+9},{cc+hh+5:.1f}" class="w ctl"/>')
            nxt.append(cc)
            if li < 3:
                f.wire([(x + 18, cc), (xs[li + 1], cc)], arrow=None)
        f.text(x + 9, 246, "abcd"[li], "ttl", "middle")
        cur = nxt
    f.wire([(318, cur[0]), (370, cur[0])])
    f.text(376, cur[0] + 4, "F(a,b,c,d)", "ttl")
    f.text(376, cur[0] + 22, "15 muxes, 4 levels", "lbl")
    f.text(376, cur[0] + 36, "picks 1 of 16 bits", "lbl")
    add("lut4", f, "LUT4", 1, para("Any 4-input function is a 16-bit truth table read through a tree of 2:1 pass-transistor muxes; the inputs a b c d choose the path. The bitstream writes the 16 bits. Wider functions chain LUTs through the MUX2_LUT5…8 cells. The dense N = 16 build uses 2,699 LUT4 plus 978 in carry mode, 43% of 8,640."))


def v_sram6t():
    f = nf("s6", 520, 230, "6T SRAM cell")
    src = bm.tp_sram()
    f.p = src.p
    P(f, 84, 76, 52, 46, "nmos_x", "access NMOS (left)")
    P(f, 384, 76, 52, 46, "nmos_x", "access NMOS (right)")
    P(f, 228, 72, 38, 28, "inverter", "storage inverter (top)")
    P(f, 256, 142, 38, 28, "inverter", "storage inverter (bottom)")
    add("sram6t", f, "6T SRAM cell", 0, para("Two cross-coupled inverters hold a bit as Q and Q_n; two access transistors connect them to the bit lines when the word line is high, for reading (sense the small difference) or writing (overpower the loop). Every block-RAM bit and every LUT and routing configuration bit is one of these."))


def mosfet_x(fid, ptype):
    f = nf(fid, 760, 400, ("PMOS" if ptype else "NMOS") + " cross-section")
    sub, well, sd, car = ("p-substrate", None, "n+", "electrons") if not ptype else ("p-substrate", "n-well", "p+", "holes")
    f.raw('<rect x="40" y="180" width="680" height="190" class="si"/>')
    f.text(60, 360, sub, "lbl")
    if well:
        f.raw('<rect x="120" y="180" width="520" height="150" rx="10" class="well"/>')
        f.text(140, 320, well, "lbl")
    f.raw('<rect x="170" y="180" width="140" height="60" rx="18" class="dopes"/>')
    f.raw('<rect x="450" y="180" width="140" height="60" rx="18" class="dopes"/>')
    f.text(240, 216, f"{sd} source", "ttl", "middle")
    f.text(520, 216, f"{sd} drain", "ttl", "middle")
    f.raw('<rect x="300" y="168" width="160" height="12" class="ox"/>')
    f.raw('<rect x="300" y="110" width="160" height="58" rx="3" class="poly"/>')
    f.text(380, 144, "gate (poly)", "ttl", "middle")
    f.text(470, 176, "gate oxide", "lbl")
    for x, lab in ((200, "S"), (380, "G"), (560, "D")):
        y0 = 110 if lab == "G" else 180
        f.raw(f'<rect x="{x-16}" y="60" width="32" height="20" class="metal"/>')
        f.wire([(x, 80), (x, y0)], arrow=None)
        f.text(x, 52, lab, "ttl", "middle")
    for i in range(14):
        x = 312 + i * 10.5
        f.raw(f'<circle cx="{x:.1f}" cy="{192 + (i % 2) * 5}" r="2.6" class="carrier"/>')
    f.text(380, 274, f"channel: {car} when |V_GS| > V_T", "lbl", "middle")
    f.text(40, 30, ("PMOS conducts when the gate is low" if ptype else "NMOS conducts when the gate is high"), "lbl")
    info = para(("A PMOS sits in an n-well: p+ source and drain, and a gate voltage below the source by more than the threshold pulls holes under the oxide to form the channel." if ptype else
                 "The gate is a conductor separated from the silicon by a thin oxide. Raise it above the threshold voltage and electrons gather under the oxide, forming a channel between the n+ source and drain: the switch closes. Every bit of state and every decision in this design ends here."))
    add(fid, f, "PMOS transistor" if ptype else "NMOS transistor", 0, info)


def build_all():
    v_world(); v_top(); v_resetbtn(); v_uartrx(); v_uarttx(); v_cmdfsm(); v_sched(); v_engine()
    v_dense(); v_sparse(); v_temper(); v_readout(); v_pmodetop(); v_pmode(); v_pgate(); v_denoise()
    v_wmac(); v_pbit(); v_sigrom(); v_gaussrom()
    for l in range(4):
        v_lfsr(l)
    v_fsms(); v_register(); v_carry(); v_bsram(); v_ssram(); v_mult(); v_fulladder(); v_passmux()
    v_dff(); v_inverter(); v_tg(); v_lut4(); v_sram6t(); mosfet_x("nmos_x", False); mosfet_x("pmos_x", True)
    missing = {p["t"] for t in TYPES.values() for p in t["portals"]} - set(TYPES)
    assert not missing, missing
    return TYPES
