# 00 — Source inventory

Every file in `data/`, `thrml/` and `torx/` is listed below, with what it contains and why it matters for the hardware model.

Citation conventions used throughout `docs/`:
- PDF page numbers are **PDF page indices** as reported by `pdftotext` (form-feed split, see `tools/cite.py`). For the arXiv papers these coincide with the printed page numbers. For the Vélez monograph, PDF page = printed page.
- Code line numbers come from `grep -n` / `sed -n` on the files as they are on disk.
- Text was extracted with `pdftotext` (poppler 25.03). `pypdf` is not installed. **All five PDFs were readable.** Equations in two-column layout extract with scrambled symbols, so equation contents were cross-checked against the surrounding prose.

## data/ (papers)

| File | Contents | Why it matters for the hardware model | Cite |
|---|---|---|---|
| `An efficient probabilistic hardware architecture for diffusion-like models - 2510.23972v2.pdf` (42 pp, "DTM paper") | Extropic/MIT. Denoising Thermodynamic Models (DTM) and the Denoising Thermodynamic Computer Architecture (DTCA): grids of Boltzmann-machine sampling cells, an all-transistor RNG with a sigmoidal bias, chromatic Gibbs, and an energy model. | **Primary source** for the p-bit cell, the bias network (resistor MAC), RNG behaviour, 2-colour Gibbs, the denoising chain and the energy accounting. | [data/An efficient…2510.23972v2.pdf, p.1 abstract; p.5 Sec. III; p.25-26 App. E, Fig. 10; p.37 App. K] |
| `A Blueprint for Equilibrium-Based Differentiable Continuous-Variable Thermodynamic Computing - 2607.16183v2.pdf` (42 pp, "Blueprint") | Extropic. Continuous-variable thermodynamic computing via Langevin dynamics in tunable potentials (Gaussian, double-well), PGM construction, superconducting "thermodynamic neuron" experiments. | The continuous-variable (Gaussian/Langevin) variant, and the double-well as a continuous analogue of a p-bit. | [data/A Blueprint…2607.16183v2.pdf, p.1 abstract; p.3 Eq. 7; p.4 Eqs. 8-11; p.6 Eqs. 19-20; p.18 Sec. VIII] |
| `A Framework for Stochastic Differentiable Programming - 2608.01612v2.pdf` (21 pp, "SDP/PSC paper") | Extropic. Parametrized Stochastic Circuits (PSC): typed wires (pbit/pdit/pmode) and probabilistic gates (PNOT, PSWAP, PCNOT, PIsing…). XTR-0 platform (CPU+FPGA+X0 chip), Z1 chip (hardwired degree-16 graph). | The **gate-level vocabulary** (the "AND/OR/NOT" of stochastic computing), the system architecture of real Extropic platforms, and X0 p-bit facts (8-bit DAC controls, τ0≈100 ns, ~5 % calibration error). | [data/A Framework…2608.01612v2.pdf, p.2 Table I; p.6 Eqs. 18-23; p.9-10 Figs. 6-7; p.17 App. A; p.20 App. D] |
| `Heat factorization.pdf` (30 pp) | Cadavid, Hoyos, Jorgenson, Smajlović, Vélez (arXiv:2104.11616): integer factorization using discrete-time heat diffusion on a finite graph. Defines "diffusion steps". | A diffusion-based "probabilistic ALU" gadget (factorization). It is mathematical, not a p-bit design. | [data/Heat factorization.pdf, p.1 abstract; p.5 "Random walks and the discrete time heat kernel"] |
| `Diffusion_Based_Integer_Factorization__…Hardware_Research_Program.pdf` (80 pp) | Vélez monograph (Oct 2026): programmed heat kernels on Cayley graphs, collision networks, an RC-network diffusion primitive, and an explicit FPGA prototype plan with acceptance criteria. | The RC network as the physical diffusion primitive, an integer heat-kernel update suitable for a small FPGA, and honest resource accounting. | [data/Diffusion_Based…pdf, p.63 Sec. 17; p.71 Sec. 19.6 Eq. 19.4; p.75-76 Sec. 22] |

## thrml/ (Extropic THRML examples, notebooks exported as scripts)

| File | Contents | Why it matters | Cite |
|---|---|---|---|
| `00_probabilistic_computing.py` | What a probabilistic computer is: arrays of RNGs, factorized local EBMs, block Gibbs. Potts model on a 20×20 grid, 2-colour blocks. | Defines the computing model the hardware implements (block Gibbs over local factors). | [thrml/00_probabilistic_computing.py:L12-47, L89-103, L160-183] |
| `01_all_of_thrml.py` | Full THRML tour: Gibbs sampling a **Gaussian PGM** on a grid with chromatic blocks, custom factors and samplers. | Continuous-variable Gibbs conditional (Gaussian node update). | [thrml/01_all_of_thrml.py:L6-53, L177] |
| `02_spin_models.py` | Spin EBMs, the Gibbs update `P(s_i=1)=σ(2γ)`, IsingEBM, DSATUR coloring into Blocks, KL-gradient training, GPU throughput, and a pointer to an FPGA sampler (~60 flips/ns). | **Core p-bit update rule** and colouring logic reused in `model/golden.py`. | [thrml/02_spin_models.py:L13-25, L69-94, L129-160] |
| `03_codon_optimization.py` | Optimization by **simulated annealing** with a log-spaced β schedule (Potts and Ising formulations). | Source for `schedule_ctrl.v` (β ramp: "scaling all weights by β"). | [thrml/03_codon_optimization.py:L303-366] |

## torx/ (Extropic Torx examples)

| File | Contents | Why it matters | Cite |
|---|---|---|---|
| `Getting_started.txt` | Install notes and a first PSWAP circuit. | API orientation only. | [torx/Getting_started.txt:L1-30] |
| `basic_usage.py` | Build, simulate and train a small circuit; the PISING energy-based gate; parameter-shift gradients. | PISING = two-spin Glauber kernel. | [torx/basic_usage.py:L66, L162, L192] |
| `01_introduction_to_parametrised_stochastic_circuits.py` | pbit, pdit and pmode gates and the gate inventory. | Gate vocabulary for docs/02. | [torx/01_introduction_to_parametrised_stochastic_circuits.py:L73, L268, L303, L514] |
| `02_random_walks_on_graphs.py` | Continuous-time random walk with one PSWAP per edge; Trotter layers. | Diffusion as composed PSWAPs (stochastic "wire" primitive). | [torx/02_random_walks_on_graphs.py:L85, L136, L292] |
| `03_bunny_graph_diffusion.py` | Edge-local PSWAP diffusion scaled to a mesh, checked against exact heat flow. | Ordered-sweep error compared with exact diffusion. | [torx/03_bunny_graph_diffusion.py:L91, L118, L326] |
| `04_execution_interface_readouts.py` | Readouts: histograms and expectations from samples, with 1/√N concentration. | Basis for `readout_counter.v` and the TV/KL metrics. | [torx/04_execution_interface_readouts.py:L98, L202, L251, L321] |
| `05_chemical_reaction_networks.py` | Mass-action CRNs mapped to single-token Markov chains. | Shows the gate model beyond Ising (not built in RTL). | [torx/05_chemical_reaction_networks.py:L107, L192, L354] |
| `06_ising_sampling_contrastive_divergence.py` | **8-site Ising ring**: chromatic Gibbs via PNOT, exact enumeration of all 256 states, PISING bond-kernel bias, PCD fit. | **Reference test case** (J, h, β=1.5) and exact-enumeration code reused in `model/golden.py`. | [torx/06_ising_sampling_contrastive_divergence.py:L99-121, L194-238, L281-301] |
| `07_discrete_diffusion.py` | Per-pixel posterior denoising on MNIST: offline UNet logits drive PNOT gates. | Discrete diffusion → p-bit draws. The denoiser is a NN, not an EBM. | [torx/07_discrete_diffusion.py:L138, L198, L229] |
| `08_stochastic_convolutional_networks.py` | Shared stochastic kernel classifies 4×4 bars-and-stripes; parameter-shift training. | Weight sharing over tiled gates. | [torx/08_stochastic_convolutional_networks.py:L154, L297, L545] |
| `09_stochastic_graph_networks.py` | MaxCut via sequential PISING edge updates, trained with REINFORCE, brute-force cut check. | **Max-Cut** target for `top.v`. Brute-force cut code. | [torx/09_stochastic_graph_networks.py:L127, L211, L268] |
| `10_pmode_gaussian_gates.py` | Affine Gaussian gates, mixture gate, exact moment propagation, conditioning. | Continuous-variable (Gaussian) gate set. | [torx/10_pmode_gaussian_gates.py:L166, L403, L442] |
| `11_gaussian_hierarchical_ssm.py` | Linear-Gaussian state-space model unrolled into affine gates. | Gaussian composition example. | [torx/11_gaussian_hierarchical_ssm.py:L95, L226, L365] |
| `12_langevin_graph_ising.py` | Overdamped-Langevin (ULA/MALA) sampling of a soft-spin graph Ising energy. | **Langevin variant** (continuous relaxation of p-bits). | [torx/12_langevin_graph_ising.py:L92, L126, L269, L392] |
| `13_regime_switching_diffusion.py` | pdit regime chain selecting Gaussian increments (jump diffusion). | Hybrid discrete/continuous gates. | [torx/13_regime_switching_diffusion.py:L93, L205] |
| `14_gaussian_bernoulli_clustering.py` | Gaussian–categorical Boltzmann machine; softmax conditional; block Gibbs. | Gaussian-Bernoulli EBM (matches DTM App. C.1.b). | [torx/14_gaussian_bernoulli_clustering.py:L79, L151, L377] |
| `15_intro_to_factors.py` | Torx factors and directed factor graphs (DFG), Tiled/Chain factors. | Software abstraction of the "cell" and its tiling. | [torx/15_intro_to_factors.py:L91, L206, L367] |
| `16_gibbs_sampling_factor_graph.py` | **4×4 periodic Ising** Gibbs via a one-spin factor, checkerboard tiling, temperature sweep, exact 65,536-state check. | Second reference case (4×4 torus, J=1, β=0.45) and checkerboard colouring. | [torx/16_gibbs_sampling_factor_graph.py:L117-160, L164-200, L592] |

## torx/helpers/

| File | Contents | Why it matters | Cite |
|---|---|---|---|
| `_plots_sampling.py` | Includes `chromatic_gibbs` (host NumPy 2-colour ring Gibbs), `torx_chromatic_gibbs`, `moments`, temperature sweeps. | **Golden-model algorithm** reused for the float model. | [torx/helpers/_plots_sampling.py:L462-494, L496] |
| `_plots_schematics.py` | Circuit and graph drawing; `lattice_index/neighbors/edges`, `checkerboard_colors`. | Checkerboard colouring code for the 4×4 grid. | [torx/helpers/_plots_schematics.py:L1643-1674] |
| `_langevin.py` | `LangevinGate`: overdamped Langevin step on the soft-spin quartic Ising energy, optional MALA correction. | Continuous-variable update rule (docs only, no RTL). | [torx/helpers/_langevin.py:L46-143] |
| `_nb09_maxcut.py` | Random regular MaxCut graphs, `spin_table`, `brute_force_cuts`, PISING edge matrix. | Max-Cut graph and exact cut evaluation for `top.v` tests. | [torx/helpers/_nb09_maxcut.py:L31-72] |
| `_graph_diffusion.py` | Graph generator, exact heat flow, PSWAP product-formula sampling. | Diffusion via PSWAP. | [torx/helpers/_graph_diffusion.py:L16-165] |
| `_nb07_diffusion.py` | UNet denoiser and Fréchet distance for nb07. The docstring says it does not touch Torx. | Shows nb07's denoiser is a NN, not hardware. | [torx/helpers/_nb07_diffusion.py:L1-7] |
| `_affine_gaussian.py` | Gaussian and mixture density helpers. | Plot support. | [torx/helpers/_affine_gaussian.py:L21-69] |
| `_nb01b.py` | Training-loss plots, PISING relaxation plot, gradient comparison. | Plot support. | [torx/helpers/_nb01b.py:L19-93] |
| `_nb05_crn.py` | CRN species-dynamics plot. | Plot support. | [torx/helpers/_nb05_crn.py:L9] |
| `_plots_fields.py` | Gaussian and cluster field plots. | Plot support. | [torx/helpers/_plots_fields.py:L40-976] |
| `_plots_training.py` | Diffusion loss, reconstruction and flip-probability plots. | Plot support. | [torx/helpers/_plots_training.py:L23-293] |
| `_plot_utils.py` | Covariance ellipses, axis helpers. | Plot support. | [torx/helpers/_plot_utils.py:L9-126] |
| `_notebook_style.py` | Matplotlib style and savefig helpers. | Plot support. | [torx/helpers/_notebook_style.py:L89-201] |
| `_notebook_paths.py` | Figure and asset directories. | Plot support. | [torx/helpers/_notebook_paths.py:L6-20] |
| `_stanford_bunny.py` | PLY loader and mesh decimation for nb03. | Plot/mesh support. | [torx/helpers/_stanford_bunny.py:L21-473] |

Also present: `task.md` (the user's original Spanish request) and `.claude/skills/{thrml,torx,verilog}` (assistant skills; not used as technical sources).
