#!/usr/bin/env python3
"""Checks of the web viewer's engines, no browser needed.

  /usr/bin/python3 web/selftest.py

1. For every UART chip: the same session (program, reseed, sweeps, histogram,
   counters, streamed samples) on the Python model and on the Verilator build
   of rtl/top.v; every reply byte must be equal.
2. The board code path without a board: a pseudo-terminal whose far end is
   answered by uart_script.TopModel; ping / program / sample over pyserial must
   give the same results as the model backend.
3. Each chip connects, streams and (for n <= 16) solves through the session API.
4. Live edits: group clamps factor other products, pins hold, an edited program
   gets a new revision, and model == RTL still holds after pins / J / h edits.
5. Machine learning (Learn tab): training on model and RTL gives identical KL
   histories; the trained glyph classifier, Hopfield recall and MRF denoising work.
6. Ideal simulator (model/pysim.py) and p-mode learning: the sampler matches exact
   enumeration, the chip demos run on it, the digits RBM learns, MRF scales to 64x64,
   the p-mode student recovers the teacher's words, and every demo names the chips
   that can run it (the Learn tab connects them).
"""
import json
import os
import pty
import sys
import threading
import tty

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import chips          # noqa: E402
import server as sv   # noqa: E402
import uart_script as us  # noqa: E402

fails = 0


def check(ok, msg):
    global fails
    fails += not ok
    print(("PASS " if ok else "FAIL ") + msg)


def fake_board(chip):
    """Serial device whose other end answers like top.v (command framing = uart_script.NARGS)."""
    master, slave = pty.openpty()
    tty.setraw(slave)
    m = us.TopModel(**chips.model_cfg(chip))

    def serve():
        buf = b""
        while True:
            try:
                buf += os.read(master, 4096)
            except OSError:
                return
            while buf:
                k = us.NARGS.get(chr(buf[0]), 0)
                if len(buf) < 1 + k:
                    break
                c, args, buf = chr(buf[0]), list(buf[1:1 + k]), buf[1 + k:]
                os.write(master, bytes(m.command(c, args)))
    threading.Thread(target=serve, daemon=True).start()
    return os.ttyname(slave)


def main():
    uart_chips = [c["id"] for c in chips.CHIPS if "rtl" in c["backends"]]
    for cid in uart_chips:
        r = sv.compare(cid)
        check(r["equal"], f"model == RTL, {cid}: {r['commands']} commands, every reply byte "
                          f"(model {r['seconds']['model']} s, RTL {r['seconds']['rtl']} s)")

    S = sv.S
    port = fake_board(chips.BY_ID["dense16"])
    a = S.connect("dense16", "board", port)
    ra = S.sample({"sweeps": 3000})
    S.connect("dense16", "model")
    rb = S.sample({"sweeps": 3000})
    check(a["ping"]["N"] == 16 and ra["top"] == rb["top"] and ra["mag"] == rb["mag"],
          f"board path over a pty ({port}) == model backend")

    for cid in uart_chips:
        S.connect(cid, "model")
        st = S.stream({"sweeps": 5})
        ok = len(st["samples"]) == 5
        if S.case["n"] <= 16:
            so = S.solve({"runs": 3})
            ok = ok and so["runs"] == 3
        check(ok, f"session API, {cid}: program {S.case['name']}, stream" + (", solve" if S.case["n"] <= 16 else ""))
    S.connect("pmode_chain4", "model")
    r = S.pm_step({"steps": 500})
    check(len(r["traj"]) > 0 and abs(sum(r["exact"][0]) - 1) < 1e-9, "p-mode model steps, exact marginals normalised")
    S.connect("factor6", "model")
    S.program({"example": "factor6"})
    so = S.solve({"runs": 10})
    check(any("a = 2" in l or "a = 3" in l for l in so["best"]) and "p = 6" in so["best"],
          f"factor6 solve finds a*b = 6 ({so['hits']}/10 runs at the lowest energy)")
    # ---- live edits
    S.connect("factor6", "model")
    for p, want in ((9, {(3, 3)}), (4, {(2, 2)}), (6, {(2, 3), (3, 2)})):
        S.pin_group({"group": "p", "value": p})
        so = S.solve({"runs": 10})
        a = int(next(l for l in so["best"] if l.startswith("a = "))[4:])
        b = int(next(l for l in so["best"] if l.startswith("b = "))[4:])
        check((a, b) in want, f"live group clamp p = {p}: solve finds a = {a}, b = {b} ({so['hits']}/10 runs)")
    r = S.pin_group({"group": "p", "value": None})
    check(not r["pins"] and r["ground"]["count"] == 16, "freeing p restores the free multiplier (16 valid rows)")
    spec = json.load(open(os.path.join(HERE, "..", "examples", "factor6.json")))
    r1 = S.program({"problem": spec, "name": "factor6"})
    spec["clamp"] = {"p0": 1, "p1": 0, "p2": 0, "p3": 1}
    r2 = S.program({"problem": spec, "name": "factor6"})
    check(r2["rev"] > r1["rev"] and r2["clamps"] != r1["clamps"], "re-loading an edited circuit with the same name gets a new revision")
    S.connect("dense16", "model")
    S.pin({"node": 0, "value": 1})
    S.restart({"mode": "const", "beta": 4})
    st = S.stream({"sweeps": 200})
    check(all(int(x["s"], 16) & 1 for x in st["samples"][2:]), "pinned p-bit 0 stays 1 for 200 sweeps at beta 4")

    def edit_session(chip, be, new_edge):
        S.connect(chip, be)
        n = S.case["n"]
        S.pin({"node": 1, "value": 0})
        if new_edge:
            S.coupling({"i": 0, "j": n - 1, "J": 1.25})
        S.coupling({"i": 0, "j": 1, "J": -1.5})
        S.bias({"i": 2, "h": 0.75})
        S.restart({"mode": "const", "beta": 2})
        return [x["s"] for x in S.stream({"sweeps": 50})["samples"]]
    for chip, new_edge in (("dense16", True), ("factor6_pt", True), ("sparse256", False)):
        check(edit_session(chip, "model", new_edge) == edit_session(chip, "rtl", new_edge),
              f"model == RTL after live edits on {chip} (pin, J, h{', new edge' if new_edge else ''})")
    S.connect("sparse256", "model")
    try:
        S.coupling({"i": 0, "j": 255, "J": 1.0})
        ok = False
    except ValueError as e:
        ok = "does not fit the sparse chip" in str(e) and S.case["Jf"][0][255] == 0
    check(ok, "an edit the sparse chip cannot hold is refused and the program is left unchanged")

    # ---- machine learning
    import ml
    hist = {}
    for be in ("model", "rtl"):
        S.connect("dense16", be)
        S.ml_start({"demo": "glyphs"})
        hist[be] = S.ml_train({"epochs": 5})["kl"]
    check(hist["model"] == hist["rtl"], f"glyph training, 5 epochs: model == RTL KL history ({hist['rtl'][-1]:.3f} nats)")
    S.connect("dense16", "model")
    S.ml_start({"demo": "glyphs"})
    while S.ml_train({"epochs": 30})["epoch"] < 120:
        pass
    st = S.ml_status()
    acc = st["metrics"]["classification accuracy (exact, 40 test images)"]
    check(acc >= 0.9, f"glyph classifier trained on the chip: {100 * acc:.0f} % exact accuracy, KL {st['kl'][-1]:.3f}")
    S.ml_use({})
    img = list(ml.bits(ml.GLYPHS["X"]))
    img[0] ^= 1
    r = S.ml_infer({"pins": {i: b for i, b in enumerate(img)}, "sweeps": 1000, "beta": 1})
    lab = r["marg"][9:13]
    check(lab.index(max(lab)) == ml.LABELS.index("X") and abs(max(lab) - max(r["marg_exact"][9:13])) < 0.1,
          f"chip classifies a noisy X as X (P = {max(lab):.2f}, exact {max(r['marg_exact'][9:13]):.2f})")
    S.ml_start({"demo": "hopfield"})
    ok = sum(S.ml_hopfield({"letter": l, "noise": 0.1, "trials": 8, "seed": 2})["successes"] for l in "HCJ")
    check(ok >= 20, f"Hopfield recall at 10 % noise: {ok}/24 letters recalled exactly")
    den = {}
    for be in ("model", "rtl"):
        S.connect("sparse256", be)
        den[be] = S.ml_denoise({"image": "disk", "noise": 0.15, "sweeps": 200, "seed": 4})
    d = den["rtl"]
    check(den["model"]["denoised"] == d["denoised"] and d["error_denoised"] < d["error_noisy"] / 1.5,
          f"MRF denoising on sparse256: model == RTL, pixel error {100 * d['error_noisy']:.1f} % -> "
          f"{100 * d['error_denoised']:.1f} %")

    # ---- ideal simulator, p-mode learning, demo / chip table
    import numpy as np
    import pysim
    import pmode_learn as pml
    rng = np.random.default_rng(3)
    J = np.triu(rng.normal(0, 0.6, (10, 10)), 1)
    J = J + J.T
    hb = rng.normal(0, 0.4, 10)
    p, _ = sv.gd.exact_boltzmann(J, hb, 1.0)
    smp = pysim.Gibbs(J, hb, chains=400, seed=1).run(100, burn=20)
    tv = 0.5 * np.abs(p - np.bincount(pysim.states(smp), minlength=1 << 10) / len(smp)).sum()
    check(tv < 0.05, f"pysim.Gibbs vs exact enumeration (n = 10, {len(smp)} samples): TV {tv:.4f}")
    S.connect("pysim", "model")
    S.ml_start({"demo": "glyphs"})
    while S.ml_train({"epochs": 40})["epoch"] < 120:
        pass
    acc = S.ml_status()["metrics"]["classification accuracy (exact, 40 test images)"]
    check(acc >= 0.9, f"glyph classifier on the ideal simulator: {100 * acc:.0f} % exact accuracy")
    S.ml_start({"demo": "hopfield"})
    ok = sum(S.ml_hopfield({"letter": l, "noise": 0.1, "trials": 8, "seed": 2})["successes"] for l in "HCJ")
    check(ok >= 20, f"Hopfield on the ideal simulator, 10 % noise: {ok}/24")
    d = S.ml_denoise({"image": "ring", "side": 64, "noise": 0.15, "sweeps": 200, "seed": 4})
    check(d["error_denoised"] < d["error_noisy"] / 3, f"MRF 64x64 (4096 p-bits) on the ideal simulator: "
          f"{100 * d['error_noisy']:.1f} % -> {100 * d['error_denoised']:.1f} %")
    S.ml_start({"demo": "digits"})
    st = S.ml_train({"epochs": 15})
    check(st["metrics"]["test accuracy"] >= 0.75, f"digits RBM (74 + 100 p-bits), 15 epochs: "
          f"{100 * st['metrics']['test accuracy']:.1f} % on 357 held-out images")
    try:
        S.restart({})
        ok = False
    except ValueError as e:
        ok = "no UART" in str(e)
    check(ok, "the ideal simulator refuses Run-tab commands with a clear message")
    check(pml.bank_check(), "p-mode PmodeBank == pmode.PmodeArray bit for bit (3 seeds, 300 steps)")
    S.connect("pmode_chain4", "model")
    S.ml_start({"demo": "pmode"})
    st = S.ml_train({"epochs": 50})
    while st["epoch"] < 120:
        st = S.ml_train({"epochs": 50})
    lsb = st["metrics_raw"]["worst parameter error (LSB)"]
    check(lsb <= 1, f"p-mode student recovers the teacher's A, h within {lsb} LSB (120 epochs)")
    bad = [k for k, d in sv.ML_DEMOS.items() if not all(c in chips.BY_ID for c in d["chips"])
           or d["home"] != d["chips"][0]]
    S.connect("dense16", "model")
    try:
        S.ml_start({"demo": "digits"})
        ok = False
    except ValueError as e:
        ok = "ideal simulator" in str(e)
    check(not bad and ok, "every demo lists real chips; a demo on the wrong chip names the one it needs")
    S.close()
    print("all checks passed" if not fails else f"{fails} check(s) FAILED")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
