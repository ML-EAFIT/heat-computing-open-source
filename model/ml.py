#!/usr/bin/env python3
"""Machine-learning examples for the p-bit chip: data sets, model builders, exact evaluation.

Four demos, sized for the bitstreams in fpga/build/ (see web/README.md):
  bars3    Boltzmann machine, 9 visible + 6 hidden, trained on the 14 3x3
           bars-and-stripes images (train.DEMOS["bars3"]); generate, complete
  glyphs   joint model of 3x3 glyphs T, L, X, O and a one-hot label (9 + 4 visible),
           trained on clean + 1-pixel-noise images; classify = pin the image and
           read the label units, generate = pin a label (train.DEMOS["glyphs"])
  hopfield 4x4 associative memory, one-shot Hebbian couplings J = 1/8 sum xi xi^T of
           3 letters; recall from a noisy image given as soft evidence h = 0.5 y,
           sampled at beta 3 after a warm-up ramp. Couplings and evidence must be
           on the same scale: with J = 1/2 sum xi xi^T the chain froze in whichever
           stored letter it met first (chip recall 75 % at 10 % noise even though
           the exact posterior recalled 98 %); with 1/8 it recalls 96 % / 92 % at
           10 % / 20 % noise (web/server.py ml_hopfield, 24 trials each)
  mrf      16x16 image denoising on the sparse chip: Ising prior J on the 4-grid,
           evidence h_i = L * y_i with L = 1/2 ln((1-p)/p) for flip noise p (J = 0.5:
           above the 2-D critical coupling ~0.44 the prior over-smooths thin shapes), so
           the chip samples the exact posterior P(x | y) [Geman & Geman MRF;
           FPGA-APPROX: the posterior-marginal estimate from the 'U' counters]
Too big for the chips, on the ideal float simulator (model/pysim.py) only:
  digits   joint RBM over 8x8 handwritten digits (scikit-learn's 1797 images, binarised)
           + a one-hot label: 74 visible + 100 hidden p-bits, PCD training; classify by
           the exact free energy, generate / complete by block Gibbs (class Digits)
  mrf      at any size (pysim.grid_denoise: 32x32 .. 256x256 pixels)
Learning rule, chip interface and quantization: model/train.py.

  python3 model/ml.py            train / evaluate all demos on the Python chip model
  python3 model/ml.py --export   write examples/hopfield_letters.json, examples/mrf_denoise_16x16.json
"""
import argparse
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cases as cs          # noqa: E402
import fixed as fx          # noqa: E402

# ------------------------------------------------------------------ data sets
GLYPHS = {                  # 3x3, row-major, 1 = ink
    "T": "111" "010" "010",
    "L": "100" "100" "111",
    "X": "101" "010" "101",
    "O": "111" "101" "111",
}
LABELS = list(GLYPHS)
HOPFIELD = {                # 4x4 letters; H, C, J are mutually orthogonal (overlap 0 of 16)
    "H": "X..X" "XXXX" "X..X" "X..X",
    "C": "XXXX" "X..." "X..." "XXXX",
    "J": "...X" "...X" "...X" "XXX.",
}
# X, O, Z were tried first: O is nearly -X (overlap -8) and noisy Z fell into a mixture
# state 6 pixels away (exact recall 78 % at 10 % noise); H, C, J recall 93 % at 20 % noise.


def bits(s):
    return tuple(1 if ch in "1X" else 0 for ch in s)


def bars_stripes(k=3):
    """The 2^k + 2^k - 2 distinct k x k bars-and-stripes images (row-major bits)."""
    imgs = set()
    for rows in itertools.product((0, 1), repeat=k):
        imgs.add(tuple(rows[r] for r in range(k) for _ in range(k)))       # horizontal bars
        imgs.add(tuple(rows[c] for _ in range(k) for c in range(k)))       # vertical stripes
    return sorted(imgs)


def bars_data():
    imgs = bars_stripes(3)
    return {v: 1 / len(imgs) for v in imgs}


def flips(img, k=1):
    """Every image at Hamming distance k."""
    out = []
    for idx in itertools.combinations(range(len(img)), k):
        v = list(img)
        for i in idx:
            v[i] ^= 1
        out.append(tuple(v))
    return out


def glyph_data(p_clean=0.5):
    """(9 pixels + one-hot label) -> weight: each class 1/4, of which p_clean on the clean
    glyph and the rest spread over its 9 one-pixel-noise variants."""
    d = {}
    for c, (name, g) in enumerate(GLYPHS.items()):
        lab = tuple(int(k == c) for k in range(len(LABELS)))
        img = bits(g)
        d[img + lab] = d.get(img + lab, 0) + p_clean / 4
        for v in flips(img):
            d[v + lab] = d.get(v + lab, 0) + (1 - p_clean) / 4 / 9
    return d


def glyph_tests():
    """(image, class) test pairs: every clean glyph and every 1-pixel-noise variant."""
    out = []
    for c, g in enumerate(GLYPHS.values()):
        img = bits(g)
        out += [(img, c)] + [(v, c) for v in flips(img)]
    return out


def image16(name):
    """16x16 test images, rows of '.'/'X'."""
    ys, xs = np.mgrid[0:16, 0:16]
    if name == "disk":
        m = (ys - 7.5) ** 2 + (xs - 7.5) ** 2 <= 6.2 ** 2
    elif name == "ring":
        r2 = (ys - 7.5) ** 2 + (xs - 7.5) ** 2
        m = (r2 <= 7.0 ** 2) & (r2 >= 3.8 ** 2)
    elif name == "letterA":
        m = np.zeros((16, 16), bool)
        for y in range(2, 15):
            half = (y - 2) * 5 // 12
            m[y, max(0, 7 - half):max(0, 7 - half) + 3] = True
            m[y, min(13, 6 + half):min(13, 6 + half) + 3] = True
        m[9:11, 4:12] = True
    elif name == "stripes":
        m = (xs // 4) % 2 == 0
    elif name == "checker":
        m = ((xs // 4) + (ys // 4)) % 2 == 0
    else:
        raise KeyError(name)
    return m.astype(int).ravel().tolist()


IMAGES16 = ["disk", "ring", "letterA", "stripes", "checker"]


def image(name, side=16):
    """A 16x16 test image at side x side (nearest-neighbour upsampling; side = 16 is image16)."""
    m = np.array(image16(name)).reshape(16, 16)
    if side == 16:
        return m.ravel().tolist()
    r = np.arange(side) * 16 // side
    return m[np.ix_(r, r)].ravel().tolist()


# ------------------------------------------------------------------ exact evaluation (n <= 16)
def cond_dist(J, h, clamp, beta=1.0):
    """Exact P(free nodes | clamped nodes) for H = -1/2 s.J.s - h.s, s = +-1.
    clamp: {node: bit}. Returns (free node list, probabilities over their 2^k states, marginals P(bit=1))."""
    J, h = np.asarray(J, float), np.asarray(h, float)
    n = len(h)
    free = [i for i in range(n) if i not in clamp]
    s_c = np.zeros(n)
    for i, b in clamp.items():
        s_c[i] = 2 * b - 1
    k = len(free)
    idx = np.arange(1 << k)
    sf = 2 * ((idx[:, None] >> np.arange(k)) & 1) - 1
    Jff = J[np.ix_(free, free)]
    field = h[free] + J[free] @ s_c                   # clamped nodes act as extra bias
    e = -0.5 * np.einsum("si,ij,sj->s", sf, Jff, sf) - sf @ field
    w = np.exp(-beta * (e - e.min()))
    p = w / w.sum()
    marg = ((sf + 1) / 2 * p[:, None]).sum(0)
    return free, p, marg


def quantized(J, h, wfrac=fx.WFRAC):
    q = lambda x: fx.to_q(float(x), wfrac, fx.WB) / (1 << wfrac)      # noqa: E731
    return np.vectorize(q)(np.asarray(J, float)), np.vectorize(q)(np.asarray(h, float))


def glyph_label_probs(J, h, img, nh=0):
    """P(label = c | image) for c in LABELS (one-hot rows only, renormalised), exact."""
    free, p, _ = cond_dist(J, h, {i: b for i, b in enumerate(img)})
    nl = len(LABELS)
    lab_of = (np.arange(1 << len(free)) & ((1 << nl) - 1))          # label nodes are the first free nodes
    out = np.array([p[lab_of == (1 << c)].sum() for c in range(nl)])
    return out / max(out.sum(), 1e-300), float(out.sum())


def glyph_accuracy(J, h, nh=0):
    tests = glyph_tests()
    ok = [int(np.argmax(glyph_label_probs(J, h, img, nh)[0]) == c) for img, c in tests]
    return sum(ok) / len(ok), len(ok)


def bars_quality(J, h, nv=9):
    """Exact model mass on the 14 valid images, and how often the most likely completion of
    an image whose bottom row is hidden is a valid bars/stripes image (for horizontal bars
    the bottom row is genuinely free, so 'valid' is the right test, not 'equal')."""
    from golden import exact_boltzmann
    p, _ = exact_boltzmann(J, h, 1.0)
    idx = np.arange(len(p))
    pv = np.bincount(idx & ((1 << nv) - 1), weights=p, minlength=1 << nv)
    imgs = bars_stripes(3)
    mass = sum(pv[sum(b << k for k, b in enumerate(v))] for v in imgs)
    valid = set(imgs)
    ok = 0
    for v in imgs:
        clamp = {i: v[i] for i in range(6)}                          # rows 0-1 known, row 2 hidden
        free, pf, _ = cond_dist(J, h, clamp)
        # marginalise hidden units: free = [6, 7, 8, hidden...]
        st = np.arange(len(pf)) & 0b111
        best = int(np.argmax(np.bincount(st, weights=pf, minlength=8)))
        ok += (v[:6] + tuple((best >> k) & 1 for k in range(3))) in valid
    return float(mass), ok / len(imgs)


# ------------------------------------------------------------------ model builders (case dicts)
def _grid_pos(names, side):
    return [((i % side) / (side - 1), (i // side) / (side - 1)) for i in range(len(names))]


def hopfield_case(letters=("H", "C", "J"), scale=0.125, beta=3.0, name="hopfield_letters"):
    """One-shot Hebbian couplings J_ij = scale * sum_mu xi_i xi_j (no biases)."""
    pats = np.array([[2 * b - 1 for b in bits(HOPFIELD[l])] for l in letters])
    J = scale * (pats.T @ pats).astype(float)
    np.fill_diagonal(J, 0)
    c = cs._finish(name, J.tolist(), [0.0] * 16, beta, _colors_dense(J))
    c.update(nodes=[f"p{r}{q}" for r in range(4) for q in range(4)], kind="ising", letters=list(letters),
             grid=4, image_nodes=list(range(16)))
    return c


def _colors_dense(J):
    from golden import greedy_coloring
    return greedy_coloring(np.asarray(J).tolist())


def hopfield_recall_bias(noisy_bits, lam=0.5):
    """Soft evidence: h_i = lam * y_i (y = +-1 from the noisy image)."""
    return [lam * (2 * b - 1) for b in noisy_bits]


def noisy(img, p, rng):
    return [b ^ int(rng.random() < p) for b in img]


def mrf_case(truth, noisy_img, J=0.5, p=0.15, lam=None, name="mrf_denoise_16x16"):
    """Ising posterior for flip noise p: grid coupling J, evidence h = L * y, L = 1/2 ln((1-p)/p)."""
    side = 16
    n = side * side
    L = 0.5 * math.log((1 - p) / p) if lam is None else lam
    Jf = [[0.0] * n for _ in range(n)]
    for y in range(side):
        for x in range(side):
            i = y * side + x
            for j in ((i + 1) if x + 1 < side else None, (i + side) if y + 1 < side else None):
                if j is not None:
                    Jf[i][j] = Jf[j][i] = J
    hf = [L * (2 * b - 1) for b in noisy_img]
    colors = [[i for i in range(n) if ((i // side) + (i % side)) % 2 == c] for c in (0, 1)]   # checkerboard
    c = cs._finish(name, Jf, hf, 1.0, colors)
    c.update(nodes=[str(i) for i in range(n)], kind="ising", grid=16, truth=list(truth), noisy=list(noisy_img),
             mrf=dict(J=J, p=p, lam=L), image_nodes=list(range(n)))
    return c


def trained_case(tr, kind, name):
    """A trained train.Trainer as an array case: visible / label / hidden node names, grid-ish layout."""
    n, nv = tr.n, tr.nv
    if kind == "glyphs":
        names = [f"x{k}" for k in range(9)] + [f"L_{l}" for l in LABELS] + [f"h{k}" for k in range(n - nv)]
    else:
        names = [f"x{k}" for k in range(nv)] + [f"h{k}" for k in range(n - nv)]
    J, h = tr.J.tolist(), tr.h.tolist()
    c = cs._finish(name, J, h, 1.0, _colors_dense(J))
    pos = []
    for i in range(n):
        if i < 9:
            pos.append((0.05 + 0.15 * (i % 3), 0.15 + 0.3 * (i // 3)))
        elif kind == "glyphs" and i < 13:
            pos.append((0.95, 0.1 + 0.27 * (i - 9)))
        else:
            k = i - nv
            pos.append((0.55 + 0.14 * (k % 2), 0.1 + 0.8 * (k // 2) / max(1, (n - nv - 1) // 2)))
    c.update(nodes=names, kind="ising", grid=3, image_nodes=list(range(9)), demo=kind, _pos=pos, _names=names,
             label_nodes=list(range(9, 13)) if kind == "glyphs" else [])
    return c


# ------------------------------------------------------------------ digits (pysim only)
DIGITS_NTRAIN = 1440


def digits_data():
    """scikit-learn's 8x8 digits, pixel = 1 if intensity > 7 of 16. First 1440 train, last 357 test."""
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    B = (X > 7).astype(int)
    return B[:DIGITS_NTRAIN], y[:DIGITS_NTRAIN], B[DIGITS_NTRAIN:], y[DIGITS_NTRAIN:]


class Digits:
    """Joint RBM over 64 pixels + 10 one-hot label units (+-1), trained with PCD-1 on pysim.Gibbs."""

    def __init__(self, nh=100, lr=0.02, batch=20, chains=100, epochs=40, seed=0):
        import pysim
        self.Xtr, self.ytr, self.Xte, self.yte = digits_data()
        self.Vtr = 2.0 * np.hstack([self.Xtr, np.eye(10)[self.ytr]]) - 1
        self.rbm = pysim.RBM(74, nh, chains=chains, lr=lr, seed=seed, init=self.Vtr.mean(0))
        self.lr, self.batch, self.target = lr, batch, epochs
        self.rng = np.random.default_rng(seed)
        self.epoch = 0
        self.hist = [self.accuracy()]

    def train_epoch(self):
        perm = self.rng.permutation(len(self.Vtr))
        lr = self.lr / (1 + self.epoch / 20)
        for b in range(0, len(perm), self.batch):
            self.rbm.fit_batch(self.Vtr[perm[b:b + self.batch]], lr=lr)
        self.epoch += 1
        self.hist.append(self.accuracy())

    def label_probs(self, imgs):
        """Exact P(label = c | image) over one-hot labels, from the free energy. imgs: (B, 64) bits."""
        V = 2.0 * np.asarray(imgs, float).reshape(-1, 64) - 1
        F = np.stack([self.rbm.free_energy(np.hstack([V, np.tile(2 * np.eye(10)[c] - 1, (len(V), 1))]))
                      for c in range(10)], 1)
        w = np.exp(-(F - F.min(1, keepdims=True)))
        return w / w.sum(1, keepdims=True)

    def accuracy(self):
        return dict(train=float((self.label_probs(self.Xtr).argmax(1) == self.ytr).mean()),
                    test=float((self.label_probs(self.Xte).argmax(1) == self.yte).mean()))

    def sample(self, pins, sweeps=200, chains=32, seed=0):
        """Block Gibbs with pins {visible index: bit} held: (P(v_i = 1) for the 74 visibles, last states)."""
        g = self.rbm.sampler(chains, seed)
        idx = sorted(pins)
        g.clamp(idx, [pins[i] for i in idx])
        g.sweep(max(20, sweeps // 4))
        m = g.marginals(sweeps)
        return m[:74], (g.s[:, :74] > 0).astype(int)

    def filters(self, k=32):
        """Pixel weights of the k hidden units with the largest weight norm (8x8 each)."""
        W = self.rbm.W[:64]
        top = np.argsort(-np.linalg.norm(W, axis=0))[:k]
        return [W[:, j].tolist() for j in top]


# ------------------------------------------------------------------ CLI
def export():
    rng = np.random.default_rng(5)
    hc = hopfield_case()
    edges = [[i, j, hc["Jf"][i][j]] for i in range(16) for j in range(i + 1, 16) if hc["Jf"][i][j]]
    noisy_x = noisy(bits(HOPFIELD["H"]), 0.15, rng)
    spec = dict(type="ising", comment="4x4 Hopfield memory of H, C, J (J = 1/8 sum xi xi^T); h = 0.5 * noisy H "
                "as evidence. Sample at beta 3: the chip recalls the stored letter.",
                n=16, edges=edges, h=hopfield_recall_bias(noisy_x), beta=3.0)
    path = os.path.join(HERE, "..", "examples", "hopfield_letters.json")
    json.dump(spec, open(path, "w"), indent=1)
    truth = image16("disk")
    y = noisy(truth, 0.15, rng)
    mc = mrf_case(truth, y)
    edges = [[i, j, mc["Jf"][i][j]] for i in range(256) for j in range(i + 1, 256) if mc["Jf"][i][j]]
    spec = dict(type="ising", comment="16x16 image denoising (sparse chip): Ising prior J = 0.5 on the grid, "
                "h = 0.867 * y for a disk with 15 % flipped pixels. Sample at beta 1; sign(<s_i>) is the "
                "denoised pixel.", n=256, edges=edges, h=[round(v, 4) for v in mc["hf"]], beta=1.0,
                schedule=dict(beta_start=1.0, beta_end=1.0, sweeps_per_step=8))
    path2 = os.path.join(HERE, "..", "examples", "mrf_denoise_16x16.json")
    json.dump(spec, open(path2, "w"))
    print(f"wrote {os.path.relpath(path)}, {os.path.relpath(path2)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args()
    if a.export:
        return export()
    import train
    d = Digits()
    for _ in range(d.target):
        d.train_epoch()
    print(f"digits (ideal simulator, {d.rbm.nv} + {d.rbm.nh} p-bits): test accuracy {100 * d.hist[0]['test']:.1f} % "
          f"-> {100 * d.hist[-1]['test']:.1f} % after {d.epoch} epochs ({len(d.yte)} held-out images)")
    for demo in ("glyphs", "bars3"):
        tr, hist = train.run_demo(demo, a.epochs, quiet=True)
        Jq, hq = tr.quantized()
        if demo == "glyphs":
            acc, nt = glyph_accuracy(Jq, hq)
            print(f"glyphs: KL {hist[0]:.3f} -> {hist[-1]:.3f} nats; classification accuracy (exact, "
                  f"{nt} clean + 1-pixel-noise images) {100 * acc:.1f} %")
        else:
            mass, comp = bars_quality(Jq, hq)
            print(f"bars3: KL {hist[0]:.3f} -> {hist[-1]:.3f} nats; model mass on the 14 valid images "
                  f"{100 * mass:.1f} %; bottom-row completion {100 * comp:.0f} %")


if __name__ == "__main__":
    main()
