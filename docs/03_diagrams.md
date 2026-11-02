# 03 — Diagrams

Paper short names as in `docs/01_evidence.md`. These Mermaid diagrams show the **FPGA modules**. For pictures of the **basic units themselves** (classical ↔ thermodynamic map, p-bit anatomy, p-gates, composition, stack, timing, board), see `docs/04_basic_units.md` (SVGs in `docs/img/units/`) and the interactive `docs/explainer.html`. Each diagram here has two parts. The **physical** side reproduces what the papers describe. The **FPGA** side shows our digital emulation.

## D1. Physical sampling cell (DTM Fig. 10)

```mermaid
flowchart LR
  subgraph NEIGH[neighbour states]
    x1[x_1] --> X1((XOR s_1))
    xn[x_n] --> Xn((XOR s_n))
  end
  X1 -- y_1·Vdd --> G1[R: G_1]
  Xn -- y_n·Vdd --> Gn[R: G_n]
  ONE[fixed 1] --> Gb1[R: G_n+1]
  ZERO[fixed 0] --> Gb2[R: G_n+2]
  G1 --> VB((V_b node + C))
  Gn --> VB
  Gb1 --> VB
  Gb2 --> VB
  VB --> RNG[RNG: Gaussian noise + shift → comparator]
  CLK[clock] --> RNG
  SET[set state] --> RNG
  RNG --> LATCH[latched x_i]
  LATCH --> OUT[to neighbours 1..n]
  LATCH --> GET[get state]
```
*Caption:* [DOCUMENTED] resistor-network bias, XOR sign correction, RNG latched by the clock, set/get state [DTM, p.25, Fig. 10; p.26, Eqs. E1-E7]. RNG internals: Gaussian noise + control voltage → comparator [DTM, p.38, Fig. 15a; p.37, App. K].

## D2. Noise source — `lfsr.v` / `noise_bank.v`

```mermaid
flowchart LR
  SEED[seed 32b] --> L0
  subgraph L0[lane k: Galois LFSR32]
    S[state reg 32b] --> LEAP[16× Galois step, combinational]
    LEAP --> S
    LEAP --> R[16 output bits]
  end
  R --> XO((XOR))
  RO[ring-osc jitter bit<br/>RO_JITTER=1 only, not simulatable] -.-> XO
  XO --> RAND[rand16 lane k]
  EN[draw enable] --> S
```
*Caption:* [FPGA-APPROX] replaces the subthreshold shot-noise RNG [DTM, p.6; p.37]. Each lane uses a distinct primitive polynomial, verified in `model/fixed.py`. The physical RNG's τ0 corresponds to "one draw per update" here.

## D3. Sigmoid unit — `sigmoid_lut.v` / `sigmoid_pwl.v`

```mermaid
flowchart LR
  LOG[logit Q5.5, 10b signed] --> OFF[+512 offset]
  OFF --> ROM[(1024 × 16b ROM, BSRAM<br/>p = round 65536·σ)]
  ROM --> P[p16]
  LOG --> PWL[|x| segments, shifts + adds]
  PWL --> P2[p16 PWL]
```
*Caption:* [DOCUMENTED] sigmoidal bias response [DTM, p.6, Fig. 4a; p.5, Eq. 11]. [FPGA-APPROX] the LUT and PWL quantization.

## D4. Weight MAC — `weight_mac.v` / `coupling_row.v`

```mermaid
flowchart LR
  H[h_i from bias RAM] --> ACC
  WRAM[(weight RAM row i: J_i0..J_iN-1)] --> J[J_ij]
  SJ[spin s_j] --> SEL{s_j = +1?}
  J --> SEL
  SEL -- +J --> ADD[(+)]
  SEL -- −J --> ADD
  ADD --> ACC[acc reg, AW bits]
  ACC --> ADD
  CNT[j counter 0..N−1] --> WRAM
  CNT --> SJ
  ACC --> FIELD[field ℓ_i]
```
*Caption:* [DOCUMENTED] MAC of weights and neighbour states, with sign set by XOR [DTM, p.26, Eq. E7]. [FPGA-APPROX] serial accumulation, one term per clock, no DSP.

## D5. p-bit — `pbit.v`

```mermaid
flowchart LR
  FIELD[field ℓ_i] --> MUL[serial shift-add × β<br/>8 cycles]
  BETA[β Q4.4] --> MUL
  MUL --> SH[>>> 2, saturate → logit Q5.5]
  SH --> SIG[sigmoid LUT]
  SIG --> P[p16]
  RAND[rand16] --> CMP{rand16 < p16}
  P --> CMP
  CMP --> BIT[new s_i]
```
*Caption:* [DOCUMENTED] P(x_i=+1) = σ(2β(ΣJx + h)) [DTM, p.5, Eq. 11]. [FPGA-APPROX] `rand < p` is the inverse-CDF form of the "noise vs threshold" comparator [DTM, p.37].

## D6. Annealing controller — `schedule_ctrl.v`

```mermaid
stateDiagram-v2
  [*] --> HOLD
  HOLD --> STEP: sweep_done && count==SWEEPS_PER_STEP
  STEP --> HOLD: β ← mode(const / β+Δ / table[k]), saturate at β_end
  HOLD --> HOLD: sweep_done (count++)
```
*Caption:* [DOCUMENTED] increasing-β schedule with fixed sweeps per β [thrml/03_codon_optimization.py:L322-344]. [FPGA-APPROX] Q4.4 β and linear/table modes.

## D7. p-bit array + scheduler — `pbit_array.v`

```mermaid
flowchart TB
  ORD[(order ROM: colour 0 nodes, colour 1 nodes…)] --> FSM
  FSM[update FSM: IDLE→BIAS→MAC→MUL→LUT→CMP→WRITE] --> MAC[weight MAC]
  WRAM[(J RAM N×N)] --> MAC
  HRAM[(h RAM N)] --> MAC
  SREG[spin register s N bits] --> MAC
  MAC --> PB[pbit: ×β, sigmoid, compare]
  NB[noise_bank lane 0] --> PB
  BETA[β from schedule_ctrl] --> PB
  PB --> SREG
  FSM --> SWD[sweep_done pulse]
  HOST[host write port] --> WRAM
  HOST --> HRAM
```
*Caption:* [DOCUMENTED] chromatic Gibbs over colour blocks [DTM, p.5; p.21, Fig. 8; torx/06_ising_sampling_contrastive_divergence.py:L198-208]. [FPGA-APPROX] one time-multiplexed engine instead of one circuit per node. [INFERRED] sequential updates within a colour equal the block update (docs/02 §2.2).

## D8. Readout — `readout_counter.v`

```mermaid
flowchart LR
  S[spins after sweep] --> UP[per-spin up-counters N × CW]
  S --> IDX[state index HIST_BITS] --> HIST[(histogram RAM 2^HIST_BITS × CW)]
  SMP[sample strobe] --> UP
  SMP --> HIST
  RD[host read addr] --> HIST
  RD --> UP
  NS[sample counter] --> OUT[read data]
```
*Caption:* [DOCUMENTED] readout = drawing samples, then histograms/expectations [torx/04_execution_interface_readouts.py:L98-118, L202]. [FPGA-APPROX] on-chip counters avoid streaming every sample over UART.

## D9. Denoising chain — `denoise_demo.v`

```mermaid
flowchart LR
  XT[x^T random] --> S1
  subgraph S1[step t = T..1]
    CL[clamp x^t] --> BIAS[h'_i = h_i + Γ_t/2 · x^t_i]
    BIAS --> GIBBS[K Gibbs sweeps on J_marg]
    GIBBS --> XN[x^t−1]
  end
  XN -- next t --> CL
  XN --> X0[x^0 output]
```
*Caption:* [DOCUMENTED] chained EBMs with forward coupling between x^t and x^{t−1} [DTM, p.4, Eqs. 7-8, Fig. 3; p.23-24, Eq. D1, Fig. 9b]. [INFERRED] clamping = bias offset. [FPGA-APPROX] hand-set Hebbian J_marg and no latent nodes.

## D10. System level — `top.v` on Tang Nano 9K

```mermaid
flowchart TB
  PC[Host PC: fpga/host.py] <-- USB --> BL702[BL702 USB-UART on board]
  BL702 -- RX pin --> URX[uart_rx]
  UTX[uart_tx] -- TX pin --> BL702
  URX --> CMD[command FSM: W H B S R C D Q X + E M G Y N L]
  CMD --> UTX
  CMD -- write J,h --> ARR
  CMD -- config --> SCH[schedule_ctrl]
  CMD -- run n sweeps --> ARR
  subgraph ARR[pbit_array]
    WR[(J/h RAM)] --> ENG[MAC → ×β → sigmoid → compare]
    NBK[noise_bank] --> ENG
    ENG --> SP[spin reg]
  end
  SCH -- β --> ARR
  ARR -- sweep_done, spins --> RO[readout_counter]
  RO -- dump --> CMD
  SP --> LED[6 LEDs: spins / status]
  CLK[27 MHz osc] --> ARR
  BTN[S1 button = reset] --> CMD
```
*Caption:* [DOCUMENTED] the host-plus-FPGA-plus-sampler architecture mirrors XTR-0 [SDP, p.9, Fig. 6] and the Z1 host↔chip h, J protocol [SDP, p.9, Fig. 7]. [FPGA-APPROX] every block on the FPGA side.

## D11. p-gate — `pgates.v` (docs/05)

```mermaid
flowchart LR
  CL[clamp_mask, clamp_val] --> FSM[pgate FSM: IDLE → BIAS N cycles → RUN → WAIT]
  H0[(h0 from gate file)] --> MUX{clamped?}
  CL --> MUX
  MUX -- "+127 / −128" --> HW[h RAM write port]
  MUX -- h0_i --> HW
  FSM --> HW
  HW --> ARR[pbit_array with the gate's J]
  FSM -- run n_sweeps --> ARR
  ARR --> SP[spins = gate wires incl. aux]
```
*Caption:* [NOT IN SOURCES] gadget coefficients (model/pgates.py). [INFERRED] clamping = bias offset. Forward = clamp inputs, backward = clamp outputs. Picture: `docs/img/units/03_pgates.svg`, `04_composition.svg`.

## D12. Sparse parallel array — `sparse_array.v` (added 2026-10-29)

```mermaid
flowchart TB
  CTL[round FSM: NODE → MAC D slots → PB → COMMIT P bits] --> PTR[round pointer r, slot s]
  subgraph L0[lane 0 … lane P−1, lock step]
    NR[(node RAM: r → valid, node i, h_i)] --> MAC[weight_mac]
    WR[(nbr RAM: r,s → j, J_ij)] --> MAC
    MUX{{spins j}} --> MAC
    MAC --> PB[pbit: ×β, sigmoid, compare]
    NB[noise lane e: own polynomial] --> PB
  end
  PTR --> NR
  PTR --> WR
  SREG[spin register N bits] --> MUX
  PB -- bit, node i --> CMT[commit: one lane per cycle]
  CMT --> SREG
  CTL --> SWD[sweep_done after round R−1]
```
*Caption:* [DOCUMENTED] sparse local connectivity, G8/G12 grid patterns and block-parallel Gibbs updates of a two-colourable graph [DTM, p.5; p.24, Table II]. [INFERRED] nodes of one round share a colour block, so they are uncoupled and the parallel update equals the sequential one; this is what makes the RTL bit-exact with `model/fixed.SparseParallelSampler`. [FPGA-APPROX] P lanes (≤ 4) instead of one cell per node, colour blocks padded to whole rounds (idle lanes draw no noise), commit serialised over P cycles to keep the spin write port single.

## D13. Parallel tempering — `temper_ctrl.v` next to `pbit_array.v` (added 2026-10-30)

```mermaid
flowchart TB
  subgraph PT[temper_ctrl]
    BANK[(K replica spin banks)] --> LD[load bank k]
    LAD[(ladder β_0 > β_1 > … , rung ↔ replica labels)] --> BSEL[β of the rung of k]
    EN[(energies e_k = E/2)]
    SW[swap p-bit: field = e_x − e_y, β = β_a − β_b] --> LBL[accepted: exchange labels]
    LF2[LFSR lane 1] --> SW
    EN --> SW
    LAD --> SW
  end
  LD -- spin_we, spin_data --> ARR[pbit_array, unchanged core]
  BSEL -- β --> ARR
  PT -- run 1 sweep --> ARR
  ARR -- "upd tap: node, old, new, field" --> EUP[e_k += s_old · field on a flip]
  EUP --> EN
  ARR -- sweep_done, spins --> BANK
  LBL --> OUT[spins_out = replica on rung 0, round_done]
```
*Caption:* one round = K sweeps (replica k at the β of its rung), then swaps of rung pairs (0,1),(2,3) on even rounds and (1,2) on odd rounds. [FPGA-APPROX] the swap is a Barker test, P = σ(2Δβ·Δe), evaluated by an ordinary `pbit` instance; energies are relative to the common start state, so `resync` is needed after h/J change. Mirror: `model/tempering.py`.

## D14. Continuous p-mode array — `pmode_array.v` (added 2026-10-30)

```mermaid
flowchart LR
  X[(x_i Q4.12)] --> T[tanh via sigmoid LUT: s = 2σ(2x) − 1]
  T --> S[(s_j Q.15)]
  A[(A_ij, h_i Q3.4)] --> MAC[F_i = h_i + Σ_j A_ij s_j]
  S --> MAC
  MAC --> CPL["(1 − s_i²) · F_i"]
  X --> QRT["λ x_i (x_i² − 1)"]
  CPL --> G[grad = quartic − coupling]
  QRT --> G
  NB[noise lane 0] --> GL[Gaussian inverse-CDF LUT] --> NZ["√(2Tε) · ξ"]
  G --> UPD["x' = sat(x − grad >> k + noise)"]
  NZ --> UPD
  UPD --> SH[(shadow x')] -- commit after all sites --> X
```
*Caption:* [DOCUMENTED] the soft-spin energy and the ULA update [torx/helpers/_langevin.py:L46-143]; continuous pmodes [SDP p.21], double-well neurons [BP p.3-6]. [FPGA-APPROX] fixed point, LUT tanh, discrete Gaussian, ε = 2^−k, no Metropolis step. Mirror: `model/pmode.py`.

## D15. Noise path options — `sparse_array.v` NOISE_CTL, raw lane in `top.v` (added 2026-10-30)

```mermaid
flowchart LR
  L0[lane 0 LFSR] --> MX{nz_share?}
  Le[lane e LFSR] --> MX
  MX -- d --> KP{nz_tau: d ≥ q?}
  KP -- yes --> KEEP[keep spin, no p-bit draw]
  KP -- no --> PB[p-bit compare r < p]
  MX -- r --> PB
  PB --> CMT[commit]
  RAW[separate lane-0 LFSR, optional ring-osc bit] -- "'N' command" --> HOST[host: fpga/entropy.py]
```
*Caption:* [DOCUMENTED motivation] RNG decorrelation time τ0 [DTM p.6 Fig. 4b]; min-entropy estimates before claiming physical randomness [DF p.76]. [FPGA-APPROX] q = probability that the RNG has decorrelated per update; shared mode = fully correlated lanes. Results: docs/06 §8.
