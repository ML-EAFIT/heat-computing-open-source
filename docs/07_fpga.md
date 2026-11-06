# 07 — Tang Nano 9K port

Target: Sipeed Tang Nano 9K, Gowin **GW1NR-LV9QN88PC6/I5** (family GW1N-9C), 27 MHz oscillator. Everything in this file is [FPGA-APPROX] unless cited.

## Files
- `fpga/tangnano9k.cst`: pin constraints.
- `fpga/Makefile`: `make` (bitstream), `make report`, `make sweep` (fit study), `make flash` (SRAM), `make flash-fl` (flash).
- `fpga/sweep_wrapper.py`: generates `top` variants with N p-bits for the sweep.
- `fpga/host.py`: UART host driver (pyserial). It has **not been run on hardware**; no board was attached.

## Pins (`tangnano9k.cst`)
| Signal | Pin | Verified against |
|---|---|---|
| clk (27 MHz) | 52 | apicula `examples/tangnano9k.cst`; Sipeed `uart/src/top.cst`, `led/src/9K_LED_project.cst` |
| uart_txd (FPGA → BL702 USB-UART) | 17 | apicula ("Use buildin USB-serial", TXD 17); Sipeed `uart_tx` 17 |
| uart_rxd | 18 | apicula RXD 18; Sipeed `uart_rx` 18 |
| led[0..5] (active low) | 10, 11, 13, 14, 15, 16 | apicula and Sipeed led example |
| btn_rst_n | 4 | apicula `rst_i` 4; Sipeed led example `sys_rst_n` 4 |
| btn_run_n | 3 | apicula `key_i` 3; **but** Sipeed's uart example uses pin 3 as `rst_n` |

**Not verified:** which silkscreen button (S1/S2) maps to pin 3 vs pin 4. The two Sipeed examples disagree about which pin is "reset". Swap the two `IO_LOC` lines if the buttons behave the other way round. Bank I/O standards (LVCMOS18 for LEDs and keys, LVCMOS33 for clk and UART) are copied from apicula's file.

## Tool flow (actually run)
```
yosys  : read_verilog -defer <rtl>; hierarchy -top top -chparam N 16 -chparam HIST_BITS 12; synth_gowin -top top -json build/top.json
nextpnr-himbaechel --json build/top.json --write build/pnr.json --device GW1NR-LV9QN88PC6/I5 \
                   --vopt family=GW1N-9C --vopt cst=tangnano9k.cst --freq 27
gowin_pack -d GW1N-9C -o build/top.fs build/pnr.json
openFPGALoader -b tangnano9k build/top.fs          (make flash; not run, no board)
```
Deviation from the requested command: **`nextpnr-gowin` is no longer shipped** in OSS CAD Suite 2026-10-23 (`nextpnr-gowin: command not found`). Its replacement is `nextpnr-himbaechel` with `--vopt family=GW1N-9C`. `read_verilog -defer` is required so that `$readmemh` paths resolve with the final parameters.

## Real utilisation — default build (N = 16, 12-bit histogram, LUT sigmoid)
`make report` output (nextpnr-himbaechel, rows with non-zero use plus the DSP rows):
```
Info: 	                 IOB:      11/    276     3%
Info: 	                LUT4:    2534/   8640    29%
Info: 	           MUX2_LUT5:     223/   4320     5%
Info: 	           MUX2_LUT6:      35/   2160     1%
Info: 	           MUX2_LUT7:      12/   1080     1%
Info: 	           MUX2_LUT8:       1/   1080     0%
Info: 	                 ALU:     998/   6480    15%
Info: 	                 DFF:    1293/   6480    19%
Info: 	           RAM16SDP4:       7/    270     2%
Info: 	               BSRAM:      10/     26    38%
Info: 	           MULT18X18:       0/     20     0%
Info: 	             MULT9X9:       0/     40     0%
Info: Max frequency for clock 'pbit_array_inst.clk': 62.67 MHz (PASS at 27.00 MHz)
```
The bitstream `fpga/build/top.fs` (3,556,722 bytes) was produced by `gowin_pack`. **No DSP or multiplier blocks are used**, as required. BSRAM breakdown, inferred from the design: 8 blocks for the 4096 × 32-bit histogram, 1 for the 1024 × 16-bit sigmoid LUT, 1 for the weight RAM.

## Fit study (`make sweep`, real P&R per N, zero-initialised RAMs)
| N | IW | LUT4 | ALU | DFF | BSRAM | Fmax (27 MHz target) |
|---|---|---|---|---|---|---|
| 8 | 3 | 1987 (22%) | 716 (11%) | 1006 (15%) | 3 (11%) | 68.89 MHz PASS |
| 16 | 4 | 2478 (28%) | 976 (15%) | 1293 (19%) | 10 (38%) | 63.63 MHz PASS |
| 32 | 5 | 3786 (43%) | 1554 (23%) | 1862 (28%) | 10 (38%) | 53.91 MHz PASS |
| 64 | 6 | 7191 (83%) | 2650 (40%) | 3009 (46%) | 11 (42%) | 43.04 MHz PASS |
| 128 | 7 | **16674 (192%)** | 4836 (74%) | 5292 (81%) | 17 (65%) | does not fit, P&R stopped |

These lines are copied from `fpga/build/sweep_<N>.log` ("Device utilisation" and "Max frequency for clock").

**Maximum p-bit count with the current top: N = 64** (83 % LUT). The logic that grows with N is:
1. the per-spin 32-bit up-counters in `readout_counter.v`, N × 32 DFF + adders;
2. the N:1 spin mux feeding the MAC, and the N-bit spin register with its write decode;
3. the UART send word (WW = N rounded up to 32).

The weight RAM is *not* the limit until N = 128: a dense 2^(2·IW) × 8-bit RAM is 128 kbit at N = 128, about 8 BSRAM blocks. The way past this limit is the SPARSE build below: neighbour lists, up-counters in BSRAM, and P parallel lanes.

## SPARSE build — `sparse_array.v`, N ≤ 256 (added 2026-10-29)
`top.v` with `SPARSE = 1` replaces the dense array with `sparse_array.v`. That module has P lock-step lanes (MAC + pbit + its own noise polynomial), one neighbour list of ≤ D entries per node, and colour blocks cut into rounds of P nodes (docs/03 D12). `UP_RAM = 1` moves the N up-counters into one BSRAM. Wrapper: `fpga/sparse_wrapper.py`. UART additions: `E` (neighbour word), `M` (node word), `G` (rounds per sweep) and an extended ping. See the `top.v` header. `host.py` detects the build from the ping.

Fit study (`make sweep-sparse`, D = 12, HIST_BITS = 12, empty RAMs, real P&R; copied from `fpga/build/sweep_sparse.txt`):

| N | P | LUT4 | DFF | BSRAM | Fmax (27 MHz target) |
|---|---|---|---|---|---|
| 64 | 1 | 2273 (26%) | 1072 (16%) | 11 (42%) | 54.55 MHz PASS |
| 64 | 2 | 2797 (32%) | 1201 (18%) | 13 (50%) | 50.44 MHz PASS |
| 64 | 4 | 4817 (55%) | 1462 (22%) | 17 (65%) | 41.13 MHz PASS |
| 128 | 1 | 2425 (28%) | 1164 (17%) | 13 (50%) | 44.17 MHz PASS |
| 128 | 2 | 3398 (39%) | 1499 (23%) | 13 (50%) | 45.05 MHz PASS |
| 128 | 4 | 4486 (51%) | 1762 (27%) | 17 (65%) | 45.46 MHz PASS |
| 256 | 1 | 3188 (36%) | 1556 (24%) | 15 (57%) | 41.97 MHz PASS |
| 256 | 2 | 4520 (52%) | 1672 (25%) | 17 (65%) | 40.67 MHz PASS |
| **256** | **4** | **6321 (73%)** | **2446 (37%)** | **17 (65%)** | **39.47 MHz PASS** |

- **Every point fits.** N = 128 dense needed 192 % LUT; sparse N = 256 with 4 lanes needs 73 %. The LUT cost now grows with P (one N:1 spin mux, MAC, multiplier and sigmoid per lane), not with N².
- BSRAM at N = 256, P = 4: 8 for the histogram, 4 for the neighbour RAMs (1024 × 16 bit each), 4 for the sigmoid LUTs (one per lane), and 1 for the up-counters. The node RAMs map to distributed RAM.
- 256 nodes is the limit of the UART protocol (1-byte node and round indices), not of the fabric. N = 512 would need 2-byte indices.
- **Throughput.** A round costs D + P + 15 cycles (measured in `tb_sparse_array`), and a sweep has R = N/P rounds for a two-colourable grid. On the board (D = 12, P = 4, N = 256): 64 × 31 = 1984 cycles per sweep, i.e. **13.6 k sweeps/s, 3.5 M p-bit updates/s** at 27 MHz. Dense N = 64 gives 5.1 k sweeps/s, 0.33 M updates/s, so this is about 10× more updates per second with 4× the nodes. P = 1 vs P = 4 on the same 256-node grid: 6400 vs 1792 cycles per sweep (D = 9 case), i.e. 3.6×. The serial commit (P cycles) and the fixed 15-cycle overhead (8-cycle shift-add multiplier, LUT, compare) keep it below 4×.
- `make sparse` builds `build/sparse_256_p4_g12_ferro16.fs`, whose power-on program is the 16 × 16 G12 ferromagnet (run button = 128 sweeps, LEDs = spins 0–4). It has not been run on a board.

## New builds (added 2026-10-30, real P&R, nextpnr-himbaechel, 27 MHz target)

| build | command | LUT4 | ALU | DFF | BSRAM | MULT18X18 | Fmax |
|---|---|---|---|---|---|---|---|
| default dense top (now with raw `N` lane, `Y`/`L` decoding) | `make` | 2801 (32 %) | 1000 | 1326 | 10 | 0 | 63.41 MHz |
| factor6 multiplier + parallel tempering, K = 4 | `make case CASE=factor6_pt` | 3674 (42 %) | 1136 | 1531 | 11 | 0 | 43.26 MHz |
| SPARSE N = 256, P = 4, empty RAMs, NOISE_CTL = 0 | (fit check) | 6303 (72 %) | 924 | 2478 | 17 | 0 | 36.89 MHz |
| SPARSE N = 256, P = 4, G12 power-on, NOISE_CTL = 1 | `make sparse SNOISE=1` | 7023 (81 %) | 996 | 2500 | 17 | 0 | 37.05 MHz |
| p-mode chain4 (M = 4) board demo | `make pmode PMCASE=chain4` | 1203 (13 %) | 420 | 548 | 2 | 7 | 52.26 MHz |
| p-mode fit M = 8 / 16 | `make pmode-fit` | 1232 (14 %) / 1682 (19 %) | | 678 / 936 | | 7 / 7 | 44.53 / 41.23 MHz |

All builds pass timing at 27 MHz, and the bitstreams were produced by `gowin_pack`. **None has run on a board.**

- **Default top.** The dense top grew from 2534 to 2801 LUTs: the raw noise lane and its `N` state, the `Y`/`L` decoding, and the `upd_*` tap. With TEMPER_K = 1, `temper_ctrl` is not instantiated.
- **Tempering** costs about 1270 LUTs for N = 14, K = 4: replica banks, energy registers, label permutation, and a second p-bit with its sigmoid ROM (the 11th BSRAM).
- **NOISE_CTL** costs about 720 LUTs at P = 4. A first version gated the spin commit with `keep_reg[cidx]` next to `c_valid`. That doubled the 256-bit spin-write logic (1.1 k → 2.5 k LUTs, 102 % total, no placement), and even NOISE_CTL = 0 grew to 84 % because the constant keep register was not removed. Folding keep into a per-lane write-valid vector (`lane_wr`), and generating no keep logic at all when NOISE_CTL = 0, fixed both. The plain sparse build is back at 72 % (73 % before today).
- **p-modes** are the first design here that uses the DSP blocks: 7 MULT18X18 of 20, for the MAC, (1 − s²)·F, x², x(x² − 1), λ·q and σ·ξ. The site count costs only registers and muxes. The run button toggles stepping at 26 steps/s and LEDs 0-3 show sign(x_i). Full speed would be 27 MHz / 61 cycles = 443 k steps/s for M = 4.

## Timing notes
- The critical path at N = 16 is the reset/button logic: `run_stable → run_prev → rst` fan-out, 62.7 MHz. The datapath (MAC, shift-add multiplier, LUT) is not critical. Fmax falls with N because of the spin mux and the counters, and is still 43 MHz at N = 64.
- Throughput: N + 18 cycles per p-bit update (measured, docs/06). At 27 MHz, N = 16 gives 544 cycles = 20.1 µs per sweep, about 49.6 k sweeps/s. N = 64 gives 64 × 82 = 5248 cycles, about 5.1 k sweeps/s.

## Fixed-point overflow
- The accumulator width satisfies AW ≥ WB + clog2(N+1) + 1, so |h + Σ ±J| ≤ (N+1)·128 **cannot overflow**. `pbit_array.v` stops elaboration with `$error` if AW is too small. Default AW = 14 for N = 16.
- The product β·field is computed at full width (AW + 9 bits). The logit is then rounded and **saturated** to [−512, 511] (|2βℓ| < 16), where σ is within 1.1·10⁻⁷ of 0 or 1.
- The denoise bias h + Γ/2·x is saturated to signed 8 bits.
- β is Q4.4 (0 … 15.94). Weights are Q3.4 by default (±8, 1/16 steps), or Q1.6 (`WFRAC = 6`, ±2, 1/64 steps) for small couplings. Quantization error is quantified in docs/06.

## LFSR correlation pitfalls
- **One lane, used sequentially.** Each update consumes 16 fresh bits of one maximal 32-bit LFSR. Never reuse the same draw for two p-bits. Never take two overlapping windows (for example `state[15:0]` after a single-step shift): consecutive windows would share 15 bits and be strongly correlated. The leap-16 design avoids this.
- **Parallel lanes.** If several p-bits draw in the same cycle (future parallel engines), use different primitive polynomials (`noise_bank.v` lanes 0–3), not seed-shifted copies of the same polynomial. Shifted copies are the *same* sequence at an unknown offset, and can overlap for long runs. The lane-0/lane-1 bit-0 agreement measured by `tb_noise_bank` was 120/256, consistent with independence (expected 128 ± 8).
- **Linearity.** Any bit of the stream is a GF(2) linear function of the previous 32. Problems with XOR-like (parity) structure, such as spin-glass gauge symmetries or the factorization relations of DF, can in principle alias with it. For such problems, combine lanes with different polynomials, or use a non-linear generator.
- **Seeding.** `X` reseeds. Seed 0 is mapped to 1 (an all-zero LFSR state is a fixed point). Identical seeds give identical runs. This is useful for bit-exact host checks (`host.py sample` verifies this for ≤ 20,000 sweeps).

## Reading results over UART
The board enumerates two serial devices. The BL702 exposes JTAG + UART, usually `/dev/ttyUSB0` and `/dev/ttyUSB1`, and the UART is normally the second one. 115200 8N1. Protocol summary (full table in the `rtl/top.v` header):

```
python3 fpga/host.py --port /dev/ttyUSB1 ping                      # -> board: N=16 HIST_BITS=12
python3 fpga/host.py --port /dev/ttyUSB1 sample --case maxcut12 --sweeps 20000
python3 fpga/host.py --port /dev/ttyUSB1 anneal --case maxcut16 --runs 10
python3 fpga/host.py --port /dev/ttyUSB1 stream --case grid16 --sweeps 50
```
- SPARSE build: `ping` prints `SPARSE P=… D=… LW=…`. `sample --case grid16` and `mag --case g12_glass16` program E/M/G, run, read the histogram and up-counters, and check bit-exactness against `model/fixed.SparseParallelSampler` for ≤ 20,000 sweeps. Idle nodes are never visited, so there are no padding draws.
- `sample` programs W/H/O, sets a constant β, reseeds, clears, runs n sweeps, then dumps the 4096-bin histogram. It reports TV/KL against exact Boltzmann. For ≤ 20,000 sweeps it also checks bit-exactness against `model/fixed.py` on the padded 16-node array.
- Without a host: press the run button (pin 3). The board restarts the annealing table schedule (power-on model = maxcut16) and runs 128 sweeps. LEDs 0–4 show spins 0–4, and LED 5 lights while busy.
- Dump sizes at 115200 baud: histogram 16 KiB ≈ 1.4 s, up-counters 68 B. `S` streaming is limited to about 5,700 samples/s for N = 16 (2 bytes per sample), so long statistics should use on-chip `R` + `D`.

## Tang Nano 20K port (added 2026-11-05) — first on-hardware runs

The same RTL also builds for the Tang Nano 20K (GW2AR-LV18QN88C8/I7, family GW2A-18C). `openFPGALoader --detect` reports idcode 0x81b, GW2A(R)-18(C).

```bash
cd fpga
make BOARD=20k           # -> build_20k/top.fs   (BOARD=9k is the default and is unchanged)
make BOARD=20k report
make BOARD=20k flash     # SRAM; flash-fl / case / sparse / pmode / flash-* take BOARD=20k too
```

- **Pins (`tangnano20k.cst`).** clk 4, LEDs 15–20 (active low), UART TX 69 / RX 70, keys 88 (reset) and 87 (run). All LVCMOS33. Checked against apicula `examples/tangnano20k.cst` and Sipeed `TangNano-20K-example/uart/src/top.cst`.
- **Keys are active high on the 20K** (Sipeed: `wire rst_n = !rst`). `tn20k_top.v` inverts them around `top`. With `-DINNER=<module>` it wraps the generated `case_top` / `sparse_top` / `pmode_wrap` / `sweep_top` instead. `rtl/` is unchanged.
- **Default build (N = 16):** LUT4 2803/20736 (13 %), ALU 1002, DFF 1326, BSRAM 10/46, IOB 11, 0 DSP, Fmax 105.70 MHz (27 MHz target).

**Measured on the board** (host.py, `/dev/ttyUSB1`):

| test | result |
|---|---|
| `ping` | N = 16, HIST_BITS = 12 |
| `sample --case maxcut12 --sweeps 20000` | TV 0.1102, KL 0.0872, **bit-exact vs model/fixed.py**; identical numbers to `--sim`; 48 k sweeps/s incl. UART |
| `pgate pgate_and Y=1` | 11 → 0.965 |
| `pgate pgate_and Y=0` | 00/01/10 → 0.333 / 0.332 / 0.329, 11 → 0.006 |
| `pgate pgate_xor Y=1` | 10/01 → 0.468 / 0.463 |
| `pgate fulladder` (free) | 8 valid rows 0.097–0.116 each, invalid ≈ 0.02 |
| `entropy --sweeps 100000` | P(1) = 0.50009 (z = +0.22), max \|r\| 0.0019 (0 lags outside band), min-entropy 0.9968 / Markov 0.9993 bit/bit, byte χ² p = 0.683 |
| `train --case xor_h` | KL 0.8377 → 0.0510 nats; P(000, 011, 101, 110) = 0.278 / 0.191 / 0.217 / 0.245 |

**USB-UART bridge (BL616) quirks, fixed in `host.py`:**
- Back-to-back bursts of more than ~32 commands lose data: 140 of 288 `H` acks came back, while the FPGA parser stayed in sync. `Board.cmd_w` / `flush` now keep at most `WINDOW = 16` commands un-acknowledged. This gives 288/288, and programming N = 16 takes about 0.1 s.
- The bridge keeps about 1 kB of an aborted stream after the host has stopped reading. `Board` now drains the line on open.
- `Board.read` times out per stall, not per call. The `N` noise stream (200 kB ≈ 17 s at 115200) used to hit the 5 s timeout.
