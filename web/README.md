# Web viewer: every chip, three engines

```bash
/usr/bin/python3 web/server.py          # then open http://localhost:8000   (--port / --host to change)
/usr/bin/python3 web/selftest.py        # model == RTL on every chip (also after live edits and training), board path over a pty, live edits, ML demos, ideal simulator (~2 min)
/usr/bin/python3 web/learn.py           # every Learn demo without the page, on its home chip (Python model); --chip pysim, --backend rtl, --json
```

It needs only the repo's existing dependencies: numpy, networkx and pyserial. The Verilog backend also needs the OSS CAD Suite in `~/tools/oss-cad-suite`, which `web/build_rtl.py` finds by itself.

## What it is
One page shows each bitstream in `fpga/build/`. It can drive that chip through any of three engines. All three speak the same UART byte protocol (`rtl/top.v` header), and all go through the same `fpga/host.Board` driver.

| engine | what answers the bytes | speed (dense N=16) |
|---|---|---|
| **Python model** | `model/uart_script.TopModel`, bit-exact mirror of `top.v` | ~8 k sweeps/s |
| **Verilog RTL** | the real `rtl/top.v` compiled by Verilator. `web/rtl/harness.cpp` bit-bangs its UART pins at 8 clocks/bit | ~4-6 k sweeps/s |
| **Tang Nano 9K** | the board on a serial port (115200 8N1), after `make flash…` | ~50 k sweeps/s on chip, UART-limited |

| chip | bitstream | engines |
|---|---|---|
| Dense N = 16 (Max-Cut power-on) | `top.fs` (`make flash`) | model, RTL, board |
| 2×2 multiplier run backwards (factor 6) | `case_factor6.fs` (`make flash-case CASE=factor6`) | model, RTL, board |
| Multiplier + parallel tempering K = 4 | `case_factor6_pt.fs` | model, RTL, board |
| Sparse N = 256, 4 lanes, noise options | `sparse_256_p4_g12_ferro16_noise.fs` (`make flash-sparse SNOISE=1`) | model, RTL, board |
| Continuous p-modes M = 4 | `pmode_chain4.fs` (`make flash-pmode`) | model only (`pmode_top.v` has no UART) |

The page does not flash the board. It shows the `make` command for the selected chip. On the board, pick the bitstream that matches the chip card; the page warns if the ping does not match.

## Tabs
- **Architecture:**
  - The block diagram. Blocks present in the build are highlighted; click one to read its Verilog.
  - The real utilisation and Fmax from the P&R log.
- **Program:** any `examples/*.json` file, any built-in case, or JSON you edit (format: `docs/09_programming.md`). Shows the quantized couplings and the exact ground states.
- **Program → Circuit editor:** builds a circuit without writing JSON.
  - **Add a gate:** pick a gate type, then click its input wires. The gate gets a new output wire.
  - **Clamp a wire:** right-click it (free → 1 → 0).
  - **Rename a wire:** double-click it.
  - **Load:** "Load circuit into chip" programs it. "Open loaded program" brings the running circuit back into the editor.
- **Run:**
  - The live coupling graph: blue = +1, red = −1, dashed edges for J < 0, a ring and badge on pinned p-bits.
  - The board's 6 LEDs.
  - Energy and β per sweep, plus the decoded current state.
  - Restart the anneal, or hold a β.
  - **Pin mode:** click a p-bit to cycle it free → 1 → 0. For circuits, a whole group can be clamped as an integer, e.g. `p = 9` turns factor6 into "factor 9".
  - **Edit mode:** click a p-bit to change its bias h, its couplings, or connect it to another p-bit. Click an edge to change its J.
  - Edits go to the running chip right away. The spins, seed and β schedule keep running; only the changed words are re-sent (`H` alone for pins on dense chips, a full W/H/O or lane reload otherwise). **Reset program** goes back to the loaded program.
  - A pin is the largest bias the chip can store (±7.94 for 1, −8 for 0), not a forced bit. It holds with probability σ(2β(|h| − Σ|J|)) per update, which the node tooltip shows. At low β early in an anneal, a pinned p-bit can still flip now and then.
  - **Sparse chip limit:** edits must keep the graph within the build's degree (≤ 12) and its 64 rounds. An edge that would need a third colour is refused, and the program is left unchanged.
- **Results:**
  - `Sample` runs `R` + `D` + `U`, then shows the histogram vs exact Boltzmann (TV / KL) and the per-spin magnetisation.
  - `Solve` anneals from many seeds.
  - For the p-mode chip, this tab instead shows the marginals vs exact e^(−V/T).
- **Learn:** machine learning with the chip as the sampler. Each card lists the chips that can run it. When the connected chip can't, a click connects one on the Python model (nothing connected yet: the selected chip if it fits, with the selected backend). On the RTL or the board the card is disabled instead, with the reason.

  | demo | chip | what happens |
  |---|---|---|
  | **Glyph classifier** | dense, ≥ 13 p-bits | Learns 3×3 glyphs T, L, X, O plus a one-hot label (`model/train.py` on the chip, live KL curve and J heatmap). **Classify:** draw an image, pin it, read the label p-bits. **Generate:** pin a label, sample the image |
  | **Bars & stripes** | Dense N = 16 | 9 visible + 6 hidden p-bits. **Generate:** samples. **Complete:** dashed pixels are unknown, the chip fills them in (chip vs exact marginals) |
  | **Hopfield memory** | Dense N = 16 | H, C, J stored one-shot (J = ⅛ Σξξᵀ). A noisy letter enters as bias h = 0.5·y; the chip recalls the letter (trials, success count). Click the input image to edit it |
  | **Image denoising** | Sparse N = 256 | 16×16 MRF: grid prior J + evidence h = ½ ln((1−p)/p)·y. The chip samples the posterior; the majority of the up-counters (`U`) is the denoised image |
  | **Handwritten digits** | ideal simulator | RBM: 64 pixels + 10 one-hot labels visible, 100 hidden (174 p-bits, too many dense couplings for any chip). scikit-learn's 8×8 digits, 1440 to train, 357 held out. Live accuracy curve and hidden-unit filters. **Classify** a test digit or a drawing; **generate** from a label; **complete** a half-hidden digit |
  | **p-mode learning** | p-mode chip | A teacher p-mode circuit (chain4) emits soft spins. A student chip with A = h = 0 matches ⟨tanh x_i tanh x_j⟩ by Langevin sampling and recovers the teacher's Q3.4 words. Shows A student vs teacher, h, and the marginals of each x_i (`model/pmode_learn.py`) |

  - **Ideal Python simulator** (`pysim` chip, `model/pysim.py`): the same chromatic Gibbs algorithm with float weights, exact sigmoid, numpy noise, hard pins and many parallel chains, with no 16 / 256 p-bit, Q3.4 or degree limits. It runs glyphs, bars & stripes, Hopfield and MRF (up to 128×128 = 16 384 p-bits) for comparison with the fixed-point chips, plus the digits RBM. Only the Learn and Architecture tabs work on it.

  - **Use trained model** makes the trained machine the session program, so the Run tab, pins and edits work on it.
  - Every inference warms up first: β ramps from β/8 to β in 16 steps, so the chain does not stay in the all −1 reset state.
- **Wire:**
  - The UART log: each command and its reply bytes.
  - **Compare model vs RTL** runs one session on both engines and checks every reply byte (length + CRC-32 per command).

Deep links work too, e.g. `/?chip=factor6&backend=rtl&tab=run&play=1`. Other parameters: `example=`, `sample=1`, `solve=1`, `compare=1`, `pin=0:1,5:0`, `group=p:9`, `mode=edit`, `select=2`, `editor=example:factor6` or `editor=current`.

## Files
| file | role |
|---|---|
| `server.py` | stdlib HTTP server + JSON API. One session. Transports: `SimSerial`, `RtlSerial` (pipe to Verilator), pyserial |
| `chips.py` | chip registry: `top.v` parameters (shared by the model and the Verilator build), P&R log parsing |
| `build_rtl.py` | `verilator --cc --exe --build` per chip into `/tmp/pbit_vl_$USER/web_<chip>` (Verilator cannot build in a path with spaces). Rebuilds when the RTL or the parameters change |
| `rtl/harness.cpp` | Verilator main: stdin bytes → `uart_rxd`, `uart_txd` → stdout; sleeps only when the array is idle (`led[5]`) |
| `static/` | `index.html`, `app.js`, `style.css`: vanilla JS + SVG, no build step |
| `selftest.py` | the checks above |
| `learn.py` | the Learn demos from the command line or a notebook (`learn.run("bars3", chip="pysim")`) |

## Machine learning: measured (2026-10-31, Python model; the RTL gives the same bytes)
- **Glyph classifier:** 120 epochs, about 14 s. KL 5.83 → 0.35 nats. Exact accuracy 100 % on the 40 clean and 1-pixel-noise glyphs. On the chip, a noisy X is labelled X with P = 1.00, the same as exact.
- **Bars & stripes:** 150 epochs, about 50 s. KL 16.4 → 0.29 nats; 93 % of the model mass on the 14 valid images. Every bottom-row completion (from the exact conditional) is a valid image.
- **Hopfield:** recall from noisy letters on the chip is 23/24 at 10 % noise and 22/24 at 20 %.
  - The first version stored X, O, Z with J = ½ Σξξᵀ. It recalled only 75 % at 10 % noise, although the exact posterior recalled 98 %.
  - Two causes: O is nearly −X, and couplings 16× stronger than the evidence froze the chain in whichever memory it met first.
  - Orthogonal letters (H, C, J) and J = ⅛ Σξξᵀ fixed both.
- **MRF denoising**, J = 0.5, 15 % flip noise, 300 sweeps: pixel error falls to 5–7 % on the disk, the letter A and the stripes, and to 10 % on the checkerboard.
  - Above the 2-D Ising critical coupling (≈ 0.44 at β = 1) the prior washes out thin shapes: with J = 0.75 the ring and the checkerboard got worse than the noisy input.
- **Model == RTL:** the training KL histories and the denoised images match byte for byte (`selftest.py`).

## Ideal simulator and p-mode learning: measured (2026-11-01, `web/learn.py`)
| demo | fixed-point chip (Python model) | ideal simulator |
|---|---|---|
| glyphs, 120 epochs | KL 0.348, accuracy 100 %, 19.6 s | KL 0.318, accuracy 100 %, 1.3 s |
| bars & stripes, 150 epochs | KL 0.290; 93 % mass on valid images; 92.7 % of free samples valid; 60 s | KL 0.516; 97 % mass; 96.4 % of samples valid; 11.7 s |
| Hopfield, 8 trials per letter | 24/24 at 10 % noise, 21/24 at 20 % | 24/24 at 10 %, 20/24 at 20 % |
| MRF 16×16, 15 % noise | 13.3 % → 3.5–9.4 % | 13.3 % → 2.7–9.0 % |

- **Lower KL is not always better samples.** The simulator's KL is lower on glyphs and higher on bars & stripes, yet its bars & stripes model puts more mass on the valid images. KL also charges for how the mass is split among the 14 images. Neither engine is uniformly better on these small models.
- **MRF at larger sizes** (simulator only): the 64×64 ring goes from 15.7 % to 2.4 % wrong pixels; 128×128 takes about 1.2 s for 300 sweeps. The larger images are upsampled 16×16 shapes, so their edges are a smaller share of the pixels. That makes them easier, not a fair comparison with 16×16.
- **Digits RBM:** 88.2 % test accuracy after 40 epochs (96.6 % on the training set); 9 of 10 generated digits are classified back as their label by the model. About 14 s.
- **p-mode learning:** after 120 epochs every A and h word is within 1 LSB of the teacher's. Two couplings (A₀₃, A₁₂) end 1 LSB short and one zero coupling (A₁₃) reads −1/16. 96 000 data samples, about 18 s.
  - The student samples with 64 copies of the bit-exact p-mode model (`PmodeBank`, a numpy version checked bit for bit against `pmode.PmodeArray`). The board has one copy: there it would take about 64× longer, or more boards.
  - A single pure-Python chain was too noisy: 3 LSB off after 150 epochs.

## RTL bug found by the Learn tab (fixed 2026-10-31)
- **Symptom:** under CPU load, the Verilog sparse chip sometimes returned different up-counters (`U`) than the model, which made the MRF demo differ.
- **Cause, in `readout_counter.v` with `UP_RAM = 1`:** a `clear` arriving while the up-counter scan of the last sample was still running was silently dropped. Stale counts then leaked into the next run. Whether the `C` landed inside that 257-cycle window depended on how many clock cycles the host's bytes took to arrive.
- **Fix:** the clear is now held pending and runs right after the scan (`busy` stays high). The dense build (`UP_RAM = 0`) never had the problem.
- **On the board:** at 115200 baud a `C` takes over 2000 clocks to arrive, so the window is almost impossible to hit, but the bitstream was rebuilt with the fix anyway.

## Measured (2026-10-31, this machine)
- **Model == RTL:** every reply byte is identical on all four UART chips.

  | chip | commands |
  |---|---|
  | dense16 | 313 |
  | factor6 | 249 |
  | factor6_pt | 249 |
  | sparse256 | 3354 |

- **Board path:** pyserial against a pty answered by the model gives the same histogram and counters as the model backend. **No real board has been connected.**
- **RTL build:** about 11 s per chip the first time; cached afterwards.
- **Sampling bias at high β.** Sampling the QUBO example at its β = 3 gives TV ≈ 0.1-0.17 vs exact. The fixed-point chain itself has that bias at high β, so it is not a viewer bug (see `docs/08` B.8). Use β ≲ 2 for distribution checks.
