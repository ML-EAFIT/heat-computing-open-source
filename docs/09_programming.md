# 09 — Programming the chip

## There is no instruction set
A CPU runs a list of instructions. A p-bit array has none. **A program is an energy function**:

    H(s) = -Σ_{i<j} J_ij s_i s_j - Σ_i h_i s_i,      s_i ∈ {-1, +1}

It comes with an update order (a graph colouring) and a β schedule. The chip samples states with probability ∝ e^(−βH). Raising β (annealing) concentrates the samples on the lowest-H states, and those states are the answer. So programming means writing your problem so that its best answers have the lowest energy. Bit k of a state read back from the board is 1 ⇔ s_k = +1.

| Classical | Here |
|---|---|
| program | J, h (+ order, β schedule) |
| compiler | `model/problem.py`: problem file → quantized Q3.4 J/h, colouring |
| loader / OS | `fpga/host.py`: UART commands `W H O B X R D S Q` (`rtl/top.v` header) |
| run | Gibbs sweeps (`R n`) |
| output | final state (`Q`), histogram (`D`), streamed samples (`S`) |

## Problem files
JSON with a `"type"` (examples in `examples/`, full format in the `model/problem.py` docstring):

| type | you give | chip energy |
|---|---|---|
| `ising` | `n`, `edges [[i,j,J]]`, `h` | as written |
| `maxcut` | `n`, `edges [[i,j(,w)]]` | J_ij = −w, so low energy = big cut |
| `qubo` | `n`, `Q [[i,j,q]]` over x ∈ {0,1} (i = j: linear term) | x = (1+s)/2, scaled to fit ±8 |
| `circuit` | `gates [["AND","a","b","y"],…]`, `inputs`, `outputs`, `clamp`, `show` | sum of the p-gate gadgets (docs/05) |

All types take `beta` and optionally `schedule: {beta_start, beta_end, sweeps_per_step}`: 16 geometric steps, then hold. The default is β/4 → β with 8 sweeps per step.

Clamping a circuit's outputs runs it backwards. `examples/factor6.json` clamps the product of a 2×2 multiplier to 6, and the chip answers with the factors.

**Bigger multipliers:** `tools/make_multiplier.py N [--bits NA NB] [--sps S] -o examples/factorN.json` writes an NA×NB-bit array multiplier (AND partial products, half and full adders) with the product clamped to N. It lists the expected answers, or warns when N has no factorization at those widths (for example, a prime). Sizes: 2×2 = 14 p-bits (identical to `factor6.json`), 3×3 ≈ 47, 4×4 = 92 (max degree 8), 4×5 ≈ 120. Anything over 16 p-bits needs the SPARSE build.

Measured on a Tang Nano 20K (2026-11-05, `make BOARD=20k flash-sparse`, then `host.py solve --runs 20`):
- `factor143.json` (4×4, sps 1024): 10/20 anneals give 11×13 or 13×11 with all gates satisfied. The `--sim` sparse model also gives 10/20.
- `factor91.json` (4×4): 2/20 give 7×13. `factor91_3x4.json` (3×4, 66 p-bits, a single answer): 7/20.
- `factor77.json` (4×4): 5/20 give 7×11 or 11×7. `factor77_3x4.json` (3×4): 8/20.
- `factor103.json`: 103 is prime. No anneal satisfies every gate, so the best states always violate at least one adder.
- Shorter anneals fail more often: at sps 32, 2/10 reach the lowest energy found, and those states still violate gates. At sps 256, 5/20 satisfy all gates (sparse model).

Limits:
- **Size:** n ≤ 16 on the default dense bitstream; n ≤ 256 on the SPARSE build (`make sparse`).
- **Precision:** coefficients are stored in 1/16 steps within ±8. `problem.py` warns when rounding or saturation changes your numbers.

## Workflow
```bash
python3 model/problem.py examples/factor6.json                      # inspect: quantized J/h, exact ground states (n <= 16)
python3 fpga/host.py --sim solve --problem examples/factor6.json    # no board: byte-exact Python model of top.v
# with the board:
source ~/tools/oss-cad-suite/environment && cd fpga && make flash
python3 host.py --port /dev/ttyUSB1 ping
python3 host.py --port /dev/ttyUSB1 solve  --problem ../examples/factor6.json --runs 20
python3 host.py --port /dev/ttyUSB1 sample --problem ../examples/ring8.json --sweeps 20000   # distribution, TV vs exact
```
Actions:
- `solve`: anneals `--runs` times and prints the lowest-energy state in your own variables (cut, x vector, wire values / integers, gate check).
- `sample`, `stream`, `anneal` and `pgate` also accept `--problem`.

Measured with `--sim` (2026-10-31):

| example | result |
|---|---|
| factor6 | 9/10 anneals give 2×3 or 3×2 |
| qubo_small | 95/100 reach the brute-force optimum |
| maxcut_pentagon | 19/20 reach the optimum cut 5 |
| ring8 (`sample`) | TV 0.0354 vs exact, bit-exact vs `model/fixed.py`; identical to `--case ring8` |

**No board run yet.**

## Machine learning (web viewer, Learn tab; `model/ml.py`, `model/train.py`)
The same energy function is a machine-learning model: P(s) ∝ e^(−βH(s)) is a Boltzmann machine.
- **Training:**
  - Move J toward ⟨s_i s_j⟩_data − ⟨s_i s_j⟩_model. The second term comes from samples the chip draws (the free phase).
  - The first term needs the chip only for hidden units: their visibles are pinned to each training example (the clamped phase).
- **Inference** is the same pinning: pin what you know (an image, a label, noisy evidence as a bias) and sample the rest.

Demos:
- a glyph classifier (train, then classify or generate);
- bars & stripes with hidden units (generate, complete);
- a Hopfield memory (one-shot Hebbian);
- 16×16 MRF image denoising on the 256-p-bit sparse chip.

Two more run where they fit:
- handwritten digits, an RBM with 174 p-bits, on the **ideal Python simulator** (`model/pysim.py`: the same Gibbs algorithm with float weights and no size limit; it also runs the four demos above for comparison);
- learning a p-mode circuit's couplings from its samples, on the continuous **p-mode chip** (`model/pmode_learn.py`).

In the Learn tab, a demo card connects the chip it needs (on the Python model). `python3 web/learn.py` runs every demo from the command line.

Run `python3 model/ml.py` for the CLI version. `python3 model/train.py --demos glyphs bars3` trains them on the chip model. Measured results are in `web/README.md`.
