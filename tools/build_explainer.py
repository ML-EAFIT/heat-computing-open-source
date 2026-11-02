#!/usr/bin/env python3
"""Build docs/explainer.html from tools/explainer_template.html.

Inlines (1) the p-gate circuits, the 2x2 multiplier and the maxcut16 graph
from model/ as JSON, (2) the basic-unit SVG drawings of tools/make_diagrams.py
(without their <style>, the page CSS styles them), (3) the measured p-gate
statistics from tb/build/pgates.json when present.

  python3 tools/build_explainer.py                 -> docs/explainer.html (full page, open locally)
  python3 tools/build_explainer.py --body OUT.html -> page body only (for publishing as an Artifact)
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "model"))
sys.path.insert(0, HERE)
import cases as cs           # noqa: E402
import make_diagrams as md   # noqa: E402
import pgates as pg          # noqa: E402

# node positions (same layout as the SVG drawings), in a 300 x 220 box
POS = {
    "NOT": {"A": (60, 110), "Y": (240, 110)},
    "AND": {"A": (60, 60), "B": (60, 160), "Y": (240, 110)},
    "OR":  {"A": (60, 60), "B": (60, 160), "Y": (240, 110)},
    "XOR": {"A": (60, 50), "B": (60, 170), "Y": (240, 50), "X": (240, 170)},
}
MULT_POS = {"a0": (40, 60), "a1": (40, 130), "b0": (40, 220), "b1": (40, 290),
            "p0": (200, 40), "q01": (200, 125), "q10": (200, 210), "q11": (200, 300),
            "p1": (360, 150), "p1_aux": (360, 70), "c1": (360, 240),
            "p2": (520, 250), "p2_aux": (520, 170), "p3": (520, 320)}


def circuit_json(c, pos, ins, outs, valid):
    Jm, hv = c.dense()
    return dict(nodes=c.nodes, J=Jm, h=hv, pos=[pos[w] for w in c.nodes], ins=ins, outs=outs,
                valid=sorted("".join(map(str, r)) for r in valid))


def data():
    gates = {}
    for kind in ("NOT", "AND", "OR", "XOR"):
        t = pg.GADGETS[kind]["terms"]
        nin = pg.N_IN[kind]
        gates[kind] = circuit_json(pg.gate(kind), POS[kind], list(t[:nin]), [t[nin]], pg.truth_rows(kind))
    m = pg.mult2x2()
    mult = circuit_json(m, MULT_POS, ["a0", "a1", "b0", "b1"], ["p0", "p1", "p2", "p3"], set())
    mc = {c["name"]: c for c in cs.all_cases()}["maxcut16"]
    pj = os.path.join(ROOT, "tb", "build", "pgates.json")
    meas = None
    if os.path.exists(pj):
        meas = {r["case"]: [dict(label=s["label"], p_ok=round(s["p_ok"], 4), p_ok_exact=round(s["p_ok_exact"], 4))
                            for s in r["scenarios"]] for r in json.load(open(pj))}
    return dict(gates=gates, mult=mult, maxcut=dict(n=16, edges=mc["edges"], optimum=20), measured=meas)


def svg_fragments():
    out = {}
    pgj_path = os.path.join(ROOT, "tb", "build", "pgates.json")
    pgj = json.load(open(pgj_path)) if os.path.exists(pgj_path) else None
    for name, fn in md.DRAWINGS:
        s = md.d_compose(pgj) if name == "04_composition" else fn()
        out[name] = s.render(style=False)
    return out


def build():
    tpl = open(os.path.join(HERE, "explainer_template.html")).read()
    tpl = tpl.replace("/*DATA*/null", json.dumps(data(), separators=(",", ":")))
    tpl = tpl.replace("/*SVGCSS*/", md.CSS_CLASSES)
    frags = svg_fragments()
    tpl = re.sub(r"<!--SVG:(\w+)-->", lambda m: frags[m.group(1)], tpl)
    return tpl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default=None)
    a = ap.parse_args()
    body = build()
    if a.body:
        open(a.body, "w").write(body)
        print(f"wrote {a.body}")
    out = os.path.join(ROOT, "docs", "explainer.html")
    open(out, "w").write("<!doctype html>\n<html lang=\"en\">\n<head><meta charset=\"utf-8\">"
                         "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">"
                         "</head>\n<body>\n" + body + "\n</body>\n</html>\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
