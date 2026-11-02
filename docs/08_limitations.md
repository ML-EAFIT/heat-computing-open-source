# 08 — Limitations and next experiments

Short paper names as in `docs/01_evidence.md`.

## A. What the papers claim (kept separate from the emulation)

| Claim | Tag | Citation |
|---|---|---|
| An all-transistor DTCA device "could achieve performance parity with GPUs on a simple image benchmark using approximately 10,000 times less energy". This is a system-level estimate on binarized Fashion-MNIST, not a measured chip. | [DOCUMENTED] | [DTM, p.1 abstract; p.2; Fig. 1] |
| Energy model E = T·K_mix·L²·E_cell, E_cell = E_rng + E_bias + E_clock + E_comm ≈ 2 fJ (main text). The Fig. 12b breakdown totals ≈1.6 fJ and is dominated by neighbour communication. DTM total ≈ 1.6·T nJ, with E_init + E_read ≈ 0.01·T nJ. | [DOCUMENTED] | [DTM, p.6 Eqs. 12-13; p.29 Fig. 12b and text] |
| RNG: τ0 ≈ 100 ns, ~10 MHz, ~350 aJ per bit, ~10 transistors, 3 × 3 µm. A 1000 × 1000 cell grid would fit in 6 mm × 6 mm. | [DOCUMENTED] | [DTM, p.6 Fig. 4; p.25 App. E] |
| The model "almost certainly underestimates the actual energy consumption", with agreement "within an order of magnitude" vs detailed designs. | [DOCUMENTED] | [DTM, p.30 App. E.5] |
| X0 pbits: hundreds of aJ to single fJ per sample at tens of MHz. Calibration error ≈5 %. | [DOCUMENTED] | [SDP, p.9 Sec. IV B; p.20 App. D] |
| Superconducting thermodynamic neurons operate "many orders of magnitude closer to the Landauer limit" and use ambient thermal noise. They are measured at 50–200 mK. | [DOCUMENTED] | [BP, p.2; p.18-21 Sec. VIII] |
| The diffusion-factorization monograph asserts "no hardware speedup or manufacturing feasibility … without synthesis and measurements". | [DOCUMENTED] | [DF, p.1 abstract; p.75] |

**None of these numbers apply to this FPGA.** The emulation makes no energy claim.

## B. What the FPGA emulation does NOT capture

1. **True physical noise.** The randomness is a deterministic 32-bit LFSR (period 2³²−1) [FPGA-APPROX]. The sources' RNG is shot/thermal noise digitized by a comparator [DOCUMENTED, DTM p.37]. Consequences:
   - The noise is reproducible. That is what makes the bit-exact tests possible.
   - It is not cryptographically or statistically "true". In the dense array one lane serves all p-bits; the sparse array has up to 4 lanes with distinct polynomials. Successive 16-bit draws are consecutive bits of one m-sequence, so every bit is a GF(2) linear combination of the previous 32 (the LFSR recurrence). All statistical tests here pass at TV ≈ iid floor (docs/06), but **any linear structure in the problem could alias with LFSR linearity**.
   - How much correlated noise matters is now measured (docs/06 §8). Cells updated in the same round that share one draw bias the law strongly (ring8, P = 4: TV 0.34, 27× the iid floor). A slow RNG that keeps its last value only slows mixing (τ_int ∝ 1/q) and leaves the law unchanged.
   - `RO_JITTER = 1` adds a ring-oscillator bit as an experiment. It is uncharacterized and unsimulatable. The tooling for a min-entropy estimate now exists (`'N'` raw stream + `fpga/entropy.py`, docs/06 §8), but **it has only been run on the LFSR in simulation**; the ring-oscillator measurement needs the board. DF p.76 asks for exactly this estimate before claiming physical randomness [DOCUMENTED, DF p.76 Sec. 22.4].
2. **Analog dynamics.** The real cell is a continuous-time resistor network relaxing with τ_bias = C/G_Σ, feeding an RNG whose output wanders with autocorrelation e^{−t/τ0} [DOCUMENTED, DTM p.26 Eq. E3; p.6 Fig. 4b]. By default our cell is a discrete-time exact Gibbs update: every draw is fresh and independent. `sparse_array.v` NOISE_CTL now emulates τ0 as a lazy update (keep the old value with probability 1 − q) and correlated noise as shared lanes (docs/06 §8). Both are discrete-time stand-ins [FPGA-APPROX]. There is still no partial relaxation of the bias node and no continuous-time telegraph process.
3. **Device variation and calibration.** X0 needs per-site calibration (≈5 % error) [DOCUMENTED, SDP p.9], and DTM studies process corners [DOCUMENTED, DTM p.6 Fig. 4c]. Our p-bits are identical and exact to 1/65536.
4. **Parallelism.** Hardware updates a whole colour block in ≈τ0 (one sweep ≈ 2τ0) [DOCUMENTED, DTM p.5].
   - The dense array time-multiplexes **one** engine: N + 18 cycles per update, about 20 µs per 16-p-bit sweep at 27 MHz (docs/06).
   - The sparse array runs P ≤ 4 lanes in lock step: N = 256 on G12 at 13.6 k sweeps/s, i.e. 3.5 M p-bit updates/s (docs/07). That is still far from one cell per node: about 74 µs per sweep vs the ~200 ns implied for DTCA.
5. **Energy efficiency.** None. One p-bit update takes N + 18 clock cycles of a full digital datapath (MAC, multiplier, BSRAM reads). The energy per update was not measured and is certainly far above the fJ budget in A [FPGA-APPROX, unmeasured].
6. **Model size and training.** DTM uses 70×70 grids with G12 connectivity, latent variables, and on-GPU training with total-correlation penalties [DOCUMENTED, DTM p.5; p.7 Eqs. 14-15].
   - The dense array fits N = 64 on the board; the sparse array fits N = 256 on G8/G12-style grids (docs/07). That is still about 19× fewer nodes than 70×70.
   - Training now exists as a host loop with the chip as the sampler (`model/train.py`, `host.py train`, docs/06 §10). It learns AND, XOR with a hidden unit, and the ring8 couplings.
   - It is small: N ≤ 16, statistics streamed over UART, plain moment matching without DTM's total-correlation penalty. **Learning is not on-chip**: the gradient is computed on the host.
7. **Continuous variables.** Langevin soft-spin p-modes (the torx `LangevinGate`) are now in RTL (`pmode_array.v`, docs/06 §11), bit-exact with `model/pmode.py`, and they reproduce float Langevin statistics. Limits:
   - ULA without the Metropolis step, so there is an O(ε) bias (TV ≈ 0.05 at ε = 1/32), and explicit Euler is unstable for ε·3λx² > 2.
   - M ≤ 16, with no coupling to the p-bit arrays (no hybrid discrete/continuous circuits).
   - No UART interface: the board image is a stand-alone LED demo.
   - Gaussian and affine-Gaussian gates and superconducting double-well neurons [DOCUMENTED, BP p.3-6; SDP p.21] are not modelled.
8. **Precision.** Q3.4 weights shift the torx/06 ring's law by TV 0.053. Q1.6 brings that to 0.013 (docs/06). β has 1/16 resolution, so 0.45 becomes 0.4375. Logits saturate at |2βℓ| ≥ 16.
   - **Found with the web viewer (2026-10-31):** at high β the chain's own stationary law, from `fixed.fixed_stationary` with no sampling noise, drifts away from Boltzmann. With `examples/qubo_small.json` the TV is ≤ 0.0002 for β ≤ 1.5, 0.003 at β = 2 and 0.114 at β = 3. At β = 3, P(ground) is 0.88 instead of 0.997.
   - Likely cause [INFERRED]: rare flips should have probability far below 2^−16, but the 16-bit σ output and comparator cannot represent such small probabilities. So the chip accepts unlikely moves too often.
   - Use β ≲ 2 for sampling with `|field|` ≈ 7. Annealing to find minima (`solve`) is unaffected.
9. **Denoising demo.** T = 4 steps, no latents, Hebbian marginal weights. It reproduces only the *mechanism*: a clamped x^t enters as a bias, and EBM steps are chained. It says nothing about DTM's FID or energy claims. The D1 sign convention is taken as ferromagnetic [INFERRED] (docs/01).

10. **Readout clear hazard (found and fixed 2026-10-31).** With `UP_RAM = 1` (sparse build), a `C` that arrived while the up-counter scan of the last sample was still running was dropped, and stale counts leaked into the next run. It showed up in the web viewer's Verilator backend under CPU load. Now the clear is held pending (`readout_counter.v` v1.2), and the sparse bitstream was rebuilt. Details: `web/README.md`.

## C. Next experiments

Status 2026-10-30: items 3 and 4 were done on 2026-10-29 (sparse array). Items 2, 5, 6 and 7 are done in simulation (docs/06 §8-11). Item 1 and the board half of item 2 still need hardware.

1. **Hardware bring-up.** *(open)* Run `fpga/host.py sample --case maxcut12` on the board. The histogram must be bit-exact with `model/fixed.py` for ≤ 20,000 sweeps, which the script checks. This is the first test that is not simulation.
2. **Entropy-source study.** *(tooling done: `'N'` + `fpga/entropy.py`, `host.py entropy`; board run open)* Enable `RO_JITTER`, then measure the bit bias, the autocorrelation, and the TV of sampled Ising laws against the LFSR baseline. This mirrors SDP's X0-vs-PRNG comparison [DOCUMENTED, SDP p.14 Fig. 17].
3. **Parallel cells.** *(done: `sparse_array.v`, P ≤ 4)* Instantiate P engines, one per colour-block slice, with independent noise lanes (`noise_bank` already has 4). This gives a speed-up of up to the colour-block size, and gets closer to the physical block-parallel update.
4. **Sparse fixed-degree graph.** *(done: `sparse_array.v`, N ≤ 256)* Store neighbours per node instead of a dense N×N RAM, for example G8/G12 patterns [DOCUMENTED, DTM p.24 Table II]. That lifts the 128-node BSRAM limit and makes the MAC O(degree) instead of O(N).
5. **Hybrid training loop.** *(done in simulation: `model/train.py`, docs/06 §10)* Run host-side contrastive divergence (clamped/free phases, as in thrml/02 and torx/06 PCD) with the FPGA as the sampler. A complete gradient pass (clamped plus free phase) is a direct use of `R`/`U`.
6. **Emulate τ0.** *(done as a lazy update with refresh probability q, docs/06 §8; not a continuous-time telegraph process)* Replace the independent draw with a two-state telegraph process of programmable switching rate. This would study how RNG autocorrelation affects mixing, as DTM Fig. 4b motivates.
7. **Langevin pmode cell.** *(done: `pmode_array.v`, docs/06 §11; with an inverse-CDF Gaussian LUT)* Euler–Maruyama update x ← x − ε∇U + √(2Tε)ξ in fixed point, with ξ from a sum of uniforms or a Gaussian LUT [DOCUMENTED rule, torx/helpers/_langevin.py:L46-143].

## D. p-gates (added 2026-10-25, docs/05)

1. The AND/OR/NOT/XOR coefficients are standard Ising gadgets, not taken from the sources [NOT IN SOURCES]. The sources only claim that such composition is possible [DOCUMENTED, BP p.3].
2. Backward runs of composed circuits mix slowly between solutions behind an energy barrier. With factor 3 the RTL stayed on 1×3 (TV 0.47 vs the exact 50/50 law). **Parallel tempering fixes this** (`temper_ctrl.v`, docs/06 §9): with K = 4 replicas and the same sweep budget, TV drops to 0.065. Swap acceptance is 0.31-0.43. Larger circuits will need more rungs or a tuned ladder.
3. The factor6 power-on bitstream and `host.py pgate` were built but not run on a board.

## E. New next experiments (2026-10-30)

1. **Metropolis-adjusted p-modes (MALA).** Add the accept/reject step of torx `metropolis=True`, so the O(ε) bias goes away. It needs V(x') − V(x) (one more pass) and a log-ratio → LUT, much like the swap p-bit.
2. **Hybrid discrete/continuous circuits.** Feed p-mode signs into p-bit fields and the reverse (torx `HybridPCircuit`).
3. **Tempering on the sparse array.** Banks of 256 bits per replica fit in BSRAM. The energy tap would come from each lane.
4. **On-chip correlation counters.** Count ⟨s_i s_j⟩ per edge during the run, so training does not have to stream every sample over UART (the bottleneck on the board).
5. **Board bring-up of everything above.** Nothing in this document has run on a Tang Nano 9K yet.
