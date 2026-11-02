#!/usr/bin/env python3
"""The chip builds the web viewer knows: top.v parameters, power-on program, P&R numbers.

Each chip is one bitstream of fpga/Makefile. `params` are the top.v parameters
of that bitstream; they configure both the Python model (uart_script.TopModel,
whose file paths are relative to model/vectors/) and the Verilator build
(web/build_rtl.py, file paths relative to web/). CLK_HZ / BAUD are changed for
simulation only (8 clocks per UART bit), as in tb/tb_top_script.v.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "model"))
import cases as cs           # noqa: E402
import export_vectors as ev  # noqa: E402

FILE_KEYS = ("W_FILE", "H_FILE", "O_FILE", "T_FILE")


def _case_files(name):
    return {k: f"{name}/{f}" for k, f in zip(FILE_KEYS, ("weights.hex", "bias.hex", "order.hex", "beta_table.hex"))}


def _sched_params(case):
    sc = ev.case_schedule(case)
    return dict(DEF_MODE=sc["mode"], DEF_BETA0=sc["beta_start"], DEF_BETA1=sc["beta_end"],
                DEF_DELTA=sc["delta"], DEF_SPS=sc["sps"])


def _factor6(pt):
    name = "factor6_pt" if pt else "factor6"
    c = {x["name"]: x for x in cs.all_cases()}[name]
    n = c["n"]
    p = dict(N=n, IW=max(1, (n - 1).bit_length()), AW=8 + n.bit_length() + 1, HIST_BITS=min(n, 12),
             WFRAC=c["wfrac"], **_case_files(name), **_sched_params(c))
    if pt:
        p.update(TEMPER_K=len(c["ladder"]), TEMPER_LADDER=sum(b << (8 * k) for k, b in enumerate(c["ladder"])))
    return p


CHIPS = [
    dict(id="dense16", title="Dense p-bit array, N = 16",
         summary="16 fully connected p-bits, one update engine, beta annealing, on-chip histogram. "
                 "Power-on program: the 16-node Max-Cut instance (run button anneals it).",
         params=dict(N=16, IW=4, AW=14, HIST_BITS=12, **_case_files("maxcut16"),
                     DEF_MODE=2, DEF_BETA0=8, DEF_BETA1=64, DEF_DELTA=0, DEF_SPS=8),
         program="maxcut16", log="nextpnr.log", fs="top.fs", flash="make flash",
         blocks=["uart", "fsm", "wram", "mac", "pbit", "lut", "lfsr", "sched", "readout"],
         backends=["model", "rtl", "board"]),
    dict(id="factor6", title="2x2 multiplier run backwards (factors 6)",
         summary="The dense array programmed with the p-gate multiplier (14 p-bits) and the product "
                 "clamped to 6. Annealing finds a, b with a*b = 6. LEDs 0-3 = a0 a1 b0 b1.",
         params=_factor6(False), program="factor6", log="case_factor6.log", fs="case_factor6.fs",
         flash="make flash-case CASE=factor6",
         blocks=["uart", "fsm", "wram", "mac", "pbit", "lut", "lfsr", "sched", "readout"],
         backends=["model", "rtl", "board"]),
    dict(id="factor6_pt", title="Multiplier + parallel tempering (K = 4)",
         summary="Same circuit with 4 replicas at beta 2, 1.25, 0.75, 0.375 and replica-exchange "
                 "swaps decided by a p-bit on the energy gap, so both factorizations are sampled.",
         params=_factor6(True), program="factor6_pt", log="case_factor6_pt.log", fs="case_factor6_pt.fs",
         flash="make flash-case CASE=factor6_pt",
         blocks=["uart", "fsm", "wram", "mac", "pbit", "lut", "lfsr", "sched", "readout", "temper"],
         backends=["model", "rtl", "board"]),
    dict(id="sparse256", title="Sparse array, N = 256, 4 parallel lanes",
         summary="256 p-bits on a degree <= 12 graph (neighbour lists), 4 lanes updating in lock step, "
                 "each with its own noise polynomial; RNG persistence / shared-noise options ('Y').",
         params=dict(N=256, IW=8, AW=13, HIST_BITS=12, UP_RAM=1, SPARSE=1, P=4, D=12, DW=4, LW=6, ROUNDS=1,
                     NOISE_CTL=1, W_FILE="", H_FILE="", O_FILE=""),
         program="g12_ferro16", log="sparse_256_p4_g12_ferro16_noise.log",
         fs="sparse_256_p4_g12_ferro16_noise.fs", flash="make flash-sparse SNOISE=1",
         blocks=["uart", "fsm", "nbr", "lanes", "mac", "pbit", "lut", "lfsr", "sched", "readout", "noisectl"],
         backends=["model", "rtl", "board"]),
    dict(id="pmode_chain4", title="Continuous p-modes (Langevin), M = 4",
         summary="Soft spins x_i in Q4.12 following discretised Langevin dynamics in a double-well "
                 "potential with tanh couplings. Uses 7 hardware multipliers. No host link on the "
                 "board: run button toggles stepping, LEDs show sign(x_i).",
         params={}, program="chain4", log="pmode_chain4.log", fs="pmode_chain4.fs",
         flash="make flash-pmode PMCASE=chain4",
         blocks=["pmode"], backends=["model"],
         note="pmode_top.v has no UART, so the page cannot drive its RTL or the board; "
              "its RTL is bit-exact with this model (tb: make pmode)."),
    dict(id="pysim", title="Ideal Python simulator (float, any size)",
         summary="Not a bitstream: model/pysim.py, the same chromatic Gibbs algorithm with float weights, "
                 "exact sigmoid, numpy noise, hard pins and many parallel chains. No 16 / 256 p-bit, Q3.4 "
                 "or degree limits, so it runs models the FPGA cannot hold (a 174-p-bit digits RBM, "
                 "64x64 MRF) and shows what the fixed-point chips lose.",
         params={}, program=None, log="", fs="", flash="(software only)",
         blocks=["pysim"], backends=["model"],
         note="Ideal simulator: the Learn tab only (Program / Run / Results / Wire drive the UART chips)."),
]
BY_ID = {c["id"]: c for c in CHIPS}


def model_cfg(chip):
    """TopModel configuration (file paths relative to model/vectors/)."""
    return dict(chip["params"])


def verilator_params(chip):
    """-G flags for the Verilator build, run from web/ (paths relative to web/)."""
    out = dict(chip["params"])
    for k in FILE_KEYS:
        if k in out and out[k]:
            out[k] = "../model/vectors/" + out[k]
    out.update(CLK_HZ=8000000, BAUD=1000000, DEB_BITS=4)
    return out


def utilisation(chip):
    path = os.path.join(ROOT, "fpga", "build", chip["log"])
    if not chip["log"] or not os.path.isfile(path):
        return None
    txt = open(path).read()
    rows = {}
    for res in ("LUT4", "ALU", "DFF", "BSRAM", "MULT18X18"):
        m = re.findall(rf"\b{res}:\s+(\d+)/\s*(\d+)", txt)
        if m:
            rows[res] = [int(m[-1][0]), int(m[-1][1])]
    fmax = re.findall(r"Max frequency for clock '[^']*': ([\d.]+) MHz", txt)
    fs = os.path.join(ROOT, "fpga", "build", chip["fs"])
    return dict(resources=rows, fmax_mhz=float(fmax[-1]) if fmax else None,
                bitstream=f"fpga/build/{chip['fs']}", bitstream_built=os.path.exists(fs))


def public(chip):
    d = {k: v for k, v in chip.items() if k != "params"}
    d["params"] = {k: v for k, v in chip["params"].items() if k not in FILE_KEYS}
    d["fpga"] = utilisation(chip)
    return d
