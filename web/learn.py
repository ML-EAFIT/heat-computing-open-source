#!/usr/bin/env python3
"""Run the Learn-tab demos without the web page: same Session code as web/server.py.

  /usr/bin/python3 web/learn.py                         every demo on its home chip, Python model
  /usr/bin/python3 web/learn.py --demo glyphs hopfield --chip pysim     ideal simulator instead
  /usr/bin/python3 web/learn.py --demo mrf --backend rtl                the Verilog sparse chip
  /usr/bin/python3 web/learn.py --demo digits --epochs 10 --json out.json

From Python / a notebook (run with web/ on sys.path):
  import learn
  r = learn.run("bars3", chip="pysim")          # dict of metrics (+ the trained Session in r["session"])

Each demo trains for its suggested epochs (or --epochs), then evaluates:
  glyphs    exact accuracy on the 40 test glyphs; one noisy X classified by sampling
  bars3     model mass on valid images; share of valid images among 2000 free samples
  hopfield  recalls of each stored letter from 10 % and 20 % noise (8 trials each)
  mrf       pixel error before / after on every test image (--side for the simulator)
  digits    held-out test accuracy; one generated digit per label, classified by the model itself
  pmode     worst |learned - true| parameter in LSB, moment error
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import server as sv          # noqa: E402


def run(demo, chip=None, backend="model", epochs=None, side=16, seed=1, port=None, quiet=True):
    d = sv.ML_DEMOS[demo]
    chip = chip or d["home"]
    S = sv.Session()
    S.connect(chip, backend, port)
    t0 = time.time()
    st = S.ml_start(dict(demo=demo, seed=seed, side=side))
    out = dict(demo=demo, chip=chip, backend=backend, engine=S.engine())
    if d["train"]:
        target = epochs if epochs is not None else st["target"]
        while st["epoch"] < target:
            st = S.ml_train(dict(epochs=min(10, target - st["epoch"])))
            if not quiet:
                print(f"  {demo} epoch {st['epoch']}/{target}", file=sys.stderr)
        out["epochs"] = st["epoch"]
    if demo in ("glyphs", "bars3"):
        out.update(kl_start=st["kl"][0], kl_end=st["kl"][-1], **st["metrics"])
        S.ml_use({})
        if demo == "glyphs":
            noisy_x = [1, 0, 1, 0, 1, 0, 1, 0, 0]                 # X with the bottom-right pixel off
            r = S.ml_infer(dict(pins={i: b for i, b in enumerate(noisy_x)}, sweeps=1500, beta=1))
            lab = r["marg"][9:13]
            out["noisy X labelled"] = sv.ml.LABELS[lab.index(max(lab))]
            out["P(L_X = 1), sampled / exact"] = [round(lab[2], 3), round(r["marg_exact"][11], 3)]
        else:
            r = S.ml_infer(dict(pins={}, sweeps=2000, beta=1, top=1 << 15))
            valid = {sum(b << k for k, b in enumerate(v)) for v in sv.ml.bars_stripes(3)}
            tot = sum(t["count"] for t in r["top"])
            ok = sum(t["count"] for t in r["top"] if int(t["state"], 16) & 0x1FF in valid)
            out["valid among free samples"] = ok / max(tot, 1)
    elif demo == "hopfield":
        for p in (0.1, 0.2):
            out[f"recalled at {int(100 * p)} % noise"] = {
                l: f"{S.ml_hopfield(dict(letter=l, noise=p, trials=8, seed=seed))['successes']}/8" for l in sv.ML_LETTERS}
    elif demo == "mrf":
        for im in sv.ml.IMAGES16:
            r = S.ml_denoise(dict(image=im, side=side, sweeps=300, seed=seed))
            out[f"{im} {r['side']}x{r['side']}"] = f"{100 * r['error_noisy']:.1f} % -> {100 * r['error_denoised']:.1f} %"
    elif demo == "digits":
        out.update(st["metrics"])
        dg = S.ml["digits"]
        ok = 0
        for c in range(10):
            g = S.ml_digits(dict(action="generate", label=c, seed=seed + c))
            ok += int(dg.label_probs([g["samples"][0]])[0].argmax() == c)
        out["generated digits read back as their label"] = f"{ok}/10"
    elif demo == "pmode":
        out.update(st["metrics_raw"])
        out["A learned"], out["A true"] = st["A"], st["A_true"]
    out["seconds"] = round(time.time() - t0, 1)
    out["session"] = S
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", nargs="*", default=list(sv.ML_DEMOS), choices=list(sv.ML_DEMOS))
    ap.add_argument("--chip", default=None, help="default: each demo's home chip (pysim = ideal simulator)")
    ap.add_argument("--backend", default="model", choices=["model", "rtl", "board"])
    ap.add_argument("--port", default=None, help="serial port for --backend board")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--side", type=int, default=16, help="MRF image side on the simulator (16, 32, 64, 128)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    res = []
    for demo in a.demo:
        chip = a.chip if a.chip in (sv.ML_DEMOS[demo]["chips"] if a.chip else []) else None
        if a.chip and chip is None:
            print(f"{demo}: skipped, does not run on {a.chip} (runs on {', '.join(sv.ML_DEMOS[demo]['chips'])})")
            continue
        r = run(demo, chip, a.backend if (chip or sv.ML_DEMOS[demo]["home"]) not in ("pysim", "pmode_chain4") else "model",
                a.epochs, a.side, a.seed, a.port, quiet=False)
        r.pop("session")
        res.append(r)
        print(f"{demo} on {r['engine']}  ({r['seconds']} s)")
        for k, v in r.items():
            if k not in ("demo", "chip", "backend", "engine", "seconds"):
                print(f"    {k}: {round(v, 4) if isinstance(v, float) else v}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
