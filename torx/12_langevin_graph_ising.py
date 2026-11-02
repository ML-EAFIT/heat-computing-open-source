#!/usr/bin/env python
# coding: utf-8

# # Sampling Langevin dynamics on a graph Ising model

# <div class="note"><span class="note-t">In brief</span>
# <p>We use a custom <a href="#ref-parisi-1981">overdamped-Langevin</a> <code>LangevinGate</code> to sample the continuous Boltzmann law of a soft-spin graph Ising energy. A JAX scan repeatedly calls <code>HybridPCircuit.sample</code>; on a tractable one-dimensional target, exact quadrature lets us validate the <a href="#ref-roberts-tweedie-1996">Metropolis-adjusted (MALA)</a> chain and measure the step-size bias of the <a href="#ref-roberts-tweedie-1996">unadjusted (ULA)</a> chain.</p>
# </div>

# Each node in the graph carries a real-valued soft spin. Neighboring spins favor alignment, while an external field tilts the pattern, giving us a continuous-variable counterpart of an Ising sampler. We want samples from the resulting Boltzmann law, which assigns more probability to low-energy states while retaining temperature-dependent fluctuations.
# 
# Overdamped Langevin dynamics provides such a sampling procedure by combining gradient descent on the energy with Gaussian noise. Repeating the step $x \leftarrow x - \varepsilon\,\nabla V(x) + \sqrt{2T\varepsilon}\,\xi$ with $\xi\sim\mathcal N(0,I)$ drives the state toward the Boltzmann law $\pi(x)\propto e^{-V(x)/T}$: the drift moves toward lower $V$, and the noise produces a spread controlled by the temperature $T$.
# 
# <details class="tx-disclosure"><summary>Term: Langevin sampler vocabulary</summary><p>A pmode is a Torx continuous-valued site, and ULA takes every Langevin proposal while MALA applies a Metropolis accept/reject correction. A stationary law is a distribution unchanged by another update, and quadrature here means numerical integration on a grid.</p></details>
# 
# We implement the update with `LangevinGate`, notebook code that satisfies Torx's `AbstractContinuousGate` interface and lives in `examples/helpers/_langevin.py`. Its `sample` method advances the soft-spin state by one gradient-drift step. We then use `jax.lax.scan` to repeat `HybridPCircuit.sample` in either unadjusted (ULA) or Metropolis-adjusted (MALA) mode.
# 
# The nonlinear drift $\nabla V$ rules out the closed-form affine-Gaussian channel available for a linear gate, so exact Gaussian moments are not available as a reference. Instead, we first apply the same sampler to a one-dimensional instance of the energy, compute its Boltzmann density by fine-grid quadrature, and only then deploy the gate on the full graph.
# 
# The main steps are:
# 
# 1. define the soft-spin Ising energy $V$ (continuous spins instead of hard plus or minus one) on a 10-node graph,
# 2. build the custom `LangevinGate` (ULA and MALA) and sample it by scanning `HybridPCircuit.sample`,
# 3. validate MALA against exact quadrature on a tractable 1-D instance and expose ULA's step-size bias, and
# 4. deploy the validated MALA gate on the full graph and read off its terminal soft-spin field, magnetization, and energy-relaxation diagnostic.
# 
# This example is the continuous-state companion to the discrete graph notebooks earlier in the series.
# 
# 

# ## Setup
# 
# The setup cells wire the import path and plot styling.
# 
# <details class="tx-disclosure"><summary>What runs where?</summary><ul><li>Torx: <code>HybridPCircuit.sample</code> executes every Langevin transition.</li><li>Notebook code: <code>jax.lax.scan</code> repeats that Torx transition, <code>jax.vmap</code> batches independent chains, and cells choose the graph, parameters, and diagnostics.</li><li>Helpers: computation lives in <code>examples/helpers/_langevin.py</code>, while paths, style, and figures live in <code>examples/helpers/_notebook_paths.py</code>, <code>examples/helpers/_notebook_style.py</code>, <code>examples/helpers/_plots_sampling.py</code>, and <code>examples/helpers/_plots_schematics.py</code>.</li></ul></details>

# In[1]:


from pathlib import Path
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np

from torx.psc import HybridPCircuit


# In[2]:


ROOT = Path.cwd()
if not (ROOT / "helpers").exists() and (ROOT.parent / "helpers").exists():
    ROOT = ROOT.parent
HELPER_DIR = ROOT / "helpers"
sys.path.insert(0, str(HELPER_DIR))

from _notebook_paths import figure_dir
from _notebook_style import (
    apply_notebook_style,
    display_and_close,
    make_savefig,
)
from _langevin import (
    LangevinGate,
    boltzmann_1d_reference,
    ks_distance,
    langevin_theta,
    plot_ks_vs_step,
    plot_mala_reference,
    plot_site_mean_parity,
    plot_ula_bias,
)
import _plots_sampling as P_samp
import _plots_schematics as P_sch


# In[3]:


FIGURE_DIR = figure_dir(ROOT)
apply_notebook_style()

SEED = 31

savefig = make_savefig(FIGURE_DIR)


# ## The graph and soft-spin Ising energy
# 
# Before we can sample anything we need the energy the gate will descend, and that energy is fixed by the geometry we choose. The target is a 10-node graph: a cycle plus three cross-chords, which give the ring a few nonlocal couplings.
# 
# Each node carries a soft spin $s_i = \tanh(x_i)$, $A_{\mathrm{adj}}$ is the adjacency matrix, and
# 
# $$V(x) = \underbrace{-\frac{\beta}{2}\, s^\top A_{\mathrm{adj}}\, s}_{\vphantom{\big|}\text{bond energy}} \;-\; h^\top s \;+\; \underbrace{\frac{\lambda_q}{4}\sum_i (x_i^2 - 1)^2}_{\vphantom{\big|}\text{spin well}}, \qquad s_i = \tanh(x_i).$$
# 
# The bond term rewards aligned neighbors, the field term $-h^\top s$ tilts the pattern, and the quartic well softly confines each coordinate near $x_i = \pm 1$.
# 
# This $V$ is the potential the gate descends, so its Boltzmann law $\pi(x)\propto e^{-V(x)/T}$ is the distribution we sample. The cell below builds the `graph` and `field` that define it.

# In[4]:


graph = nx.cycle_graph(10)
# Add three long-range chords so the cycle has nonlocal couplings.
graph.add_edges_from([(0, 5), (2, 7), (3, 8)])
pos = nx.spring_layout(graph, seed=SEED)

n = graph.number_of_nodes()
adj = nx.to_numpy_array(graph, dtype=float)
# Antisymmetric field: a per-site tilt from negative to positive that sums to ~0.
field = np.linspace(-0.30, 0.30, n)


# In[5]:


assert graph.number_of_nodes() == 10
assert graph.number_of_edges() == 13  # 10 cycle edges + 3 cross-chords
print(f"graph: {n} nodes, {graph.number_of_edges()} edges")


# ## The Langevin gate
# 
# We begin with the transition implemented by `LangevinGate`. The class is supplied with these notebooks rather than by the Torx API; to participate in a circuit, it only needs to satisfy the `AbstractContinuousGate` interface. At run time, the circuit passes the gate's `theta` to `sample` as `params`, leaving the proposal and any correction step under the gate's control. The following excerpt comes from `examples/helpers/_langevin.py`:
# 
# ```python
# class LangevinGate(AbstractContinuousGate[dict[str, Array], tuple[int, ...]]):
#     ...
#     metropolis: bool = eqx.field(static=True, default=False)
# 
#     def energy(
#         self, x: Float[Array, " d"], theta: dict[str, Array]
#     ) -> Float[Array, ""]:
#         """Soft-spin quartic Ising energy $V(x)$ that the gate descends."""
#         s = jnp.tanh(x)
#         coupling = -0.5 * theta["beta"] * (s @ (theta["adj"] @ s))
#         bias = -(theta["field"] @ s)
#         quartic = 0.25 * theta["lam_quartic"] * jnp.sum((x * x - 1.0) ** 2)
#         return coupling + bias + quartic
# 
#     def sample(
#         self,
#         key: Key[Array, ""],
#         inputs: Mapping[str, PyTree[Array]],
#         params: dict[str, Array],
#         ...
#     ):
#         """Advance the continuous state by one (optionally adjusted) Langevin step."""
#         x = inputs["continuous"]
#         eps = params["step"]
#         temperature = params["temperature"]
#         grad_x = jax.grad(self.energy)(x, params)
# 
#         noise_key, accept_key = jax.random.split(key)
#         ...
#         noise = jax.random.normal(noise_key, x.shape, dtype=x.dtype)
#         ...
# 
#         proposal = x - eps * grad_x + jnp.sqrt(2.0 * temperature * eps) * noise
# 
#         if not self.metropolis:
#             output = proposal
#         else:
#             grad_prop = jax.grad(self.energy)(proposal, params)
#             # log target ratio: -(V(x') - V(x)) / T
#             log_target = (
#                 -(self.energy(proposal, params) - self.energy(x, params)) / temperature
#             )
#             # log proposal ratio log q(x | x') - log q(x' | x); the Gaussian
#             # proposal has mean y - eps * grad V(y) and covariance 2 T eps I.
#             fwd = proposal - (x - eps * grad_x)
#             bwd = x - (proposal - eps * grad_prop)
#             log_proposal = (jnp.sum(fwd**2) - jnp.sum(bwd**2)) / (
#                 4.0 * temperature * eps
#             )
#             accept_prob = jnp.exp(jnp.minimum(0.0, log_target + log_proposal))
#             ...
#             accepted = jax.random.bernoulli(accept_key, accept_prob)
#             ...
#             output = jnp.where(accepted, proposal, x)
# ```
# 
# The proposal is one overdamped-Langevin step on the soft-spin energy:
# 
# $$x' = x \underbrace{-\,\varepsilon\,\nabla V(x)}_{\vphantom{\big|}\text{gradient drift}} + \underbrace{\sqrt{2T\varepsilon}\,\xi}_{\vphantom{\big|}\text{Gaussian noise}},\qquad \xi\sim\mathcal N(0, I).$$
# 
# Here `jax.grad` differentiates the gate's own `energy` method, so the drift and the target $\pi(x)\propto e^{-V(x)/T}$ use the same potential. The static `metropolis` flag determines how the proposal becomes the next state:
# 
# - ULA (`metropolis=False`) always returns the proposal. Its discrete-time transition approximates the continuous Langevin dynamics and introduces an $O(\varepsilon)$ bias in the stationary distribution.
# - MALA (`metropolis=True`) evaluates the Metropolis-Hastings probability for $\pi$ and either accepts the proposal or leaves the state unchanged. This correction removes the discretization bias, making the chain asymptotically exact for $e^{-V/T}$.
# 
# The energy coefficients and integrator constants travel in `theta`, just as `AffineGaussianGate` carries its $(A, b, \log\mathrm{var})$. The entries are `adj`, `field`, `beta`, `lam_quartic`, `step` ($\varepsilon$), and `temperature` ($T$). We next set the deployment values and package them with `langevin_theta`.

# In[6]:


BETA = 0.35
LAM_QUARTIC = 0.6
STEP_SIZE = 0.04
TEMPERATURE = 0.35

DEPLOY_REPS = 1200     # Metropolis-adjusted steps per deployed chain
LONGRUN_REPS = 6000    # a much longer MALA run, used as a convergence reference
NUM_SAMPLES = 1024


# In[7]:


theta = langevin_theta(
    adj=adj,
    field=field,
    beta=BETA,
    lam_quartic=LAM_QUARTIC,
    step=STEP_SIZE,
    temperature=TEMPERATURE,
)


# `LangevinGate` itself carries the structural pieces (its sites, dims, and the ULA/MALA switch) and reads every physical coefficient from `theta`, so one helper can run either mode without touching the gate's logic. The helper below builds a one-step Torx circuit in either mode, repeats it with `jax.lax.scan`, and batches independent chains with `jax.vmap`.

# In[8]:


def terminal_samples(theta, *, metropolis, reps, num_samples, dim, key):
    """Run scanned Torx transitions for independent Langevin chains."""
    gate = LangevinGate(sites=0, dims=(dim,), metropolis=metropolis)
    circuit = HybridPCircuit([gate])
    initial_state = {
        "discrete": jnp.zeros((0,), dtype=jnp.int32),
        "continuous": jnp.zeros(dim),
    }

    def run_chain(chain_key):
        def step(carry, _):
            state, step_key = carry
            step_key, sample_key = jax.random.split(step_key)
            state = circuit.sample(sample_key, state, [theta])
            return (state, step_key), None

        (state, _), _ = jax.lax.scan(
            step, (initial_state, chain_key), None, length=reps
        )
        return state

    keys = jax.random.split(key, num_samples)
    sampled = eqx.filter_jit(jax.vmap(run_chain))(keys)
    return np.asarray(sampled["continuous"])


# The schematic makes the repeated computation explicit. Each step applies one custom gate to the entire vector pmode $x\in\mathbb{R}^{10}$; the compact bus therefore represents all ten coupled coordinates rather than ten independent transitions. The scan repeats this one-step Torx circuit `reps` times.

# In[9]:


fig = P_sch.draw_pcircuit(
    [("MALA", [0])],
    wire_labels=[rf"$x\in\mathbb{{R}}^{{{n}}}$"],
    title="One MALA step on the full spin vector",
    reps=DEPLOY_REPS,
)
display_and_close(fig)


# ## Validating against an exact reference
# 
# The nonlinear gradient $\nabla V$ gives the gate neither an affine-Gaussian channel nor closed-form Gaussian moments. We therefore validate it on a tractable restriction of the same model: one soft spin in the quartic double well, with a field bias and the couplings switched off. On this one-dimensional problem, fine-grid quadrature gives the Boltzmann density $\pi(x)\propto e^{-V(x)/T}$ directly.
# 
# This comparison exercises the same `sample` implementation used for the graph: both modes use the gradient-drift proposal, and MALA adds the same Metropolis accept/reject calculation. Only the energy coefficients change. Agreement with the one-dimensional reference can therefore test the proposal and acceptance machinery against an exact target, but it cannot establish convergence of the deployed 10-D chain. We address that separate question later by comparing its per-site means with a much longer MALA run.
# 
# To separate discretization error from the amount of simulated dynamics, we fix $T_{\mathrm{sim}} = \texttt{reps}\times\varepsilon$ throughout the step-size sweep. Every chain then covers the same simulated time while $\varepsilon$ controls the discretization.

# In[10]:


LAM_1D = 1.5      # a stiffer well than the graph run, to make ULA's bias vivid
T_1D = 0.30
H_1D = 0.25
T_SIM = 60.0      # fixed simulated time; reps = T_sim / step isolates step-size bias
N_1D = 8000

EPS_SWEEP = [0.02, 0.06, 0.11, 0.16]
EPS_MATCH = 0.06  # step for the MALA-matches-reference panel
EPS_BIAS = 0.16   # step for the ULA-bias panel (the coarsest in the sweep)

grid = np.linspace(-3.0, 3.0, 1401)


# The `boltzmann_1d_reference` helper evaluates the gate's own `energy` method on the grid and normalizes $e^{-V(x)/T}$ by quadrature, so the target we compare against is the same $V$ the gate descends.

# In[11]:


ref_gate = LangevinGate(sites=0, dims=(1,), metropolis=True)
theta_1d = langevin_theta(
    adj=[[0.0]],
    field=[H_1D],
    beta=0.0,
    lam_quartic=LAM_1D,
    step=EPS_MATCH,
    temperature=T_1D,
)
ref_density, ref_cdf = boltzmann_1d_reference(ref_gate, theta_1d, grid)


# We sweep the step size for both ULA and MALA, sampling `N_1D` independent chains at each $\varepsilon$ and scoring each empirical marginal against the exact reference with the `ks_distance` helper, a one-sample Kolmogorov-Smirnov statistic that reports the largest gap between the sampled and exact cumulative distributions, so smaller means closer.

# In[12]:


ula_1d, mala_1d = {}, {}
ks_ula, ks_mala = [], []
for eps in EPS_SWEEP:
    reps = round(T_SIM / eps)  # fixed simulated time T_sim = reps * eps
    theta_eps = langevin_theta(
        adj=[[0.0]],
        field=[H_1D],
        beta=0.0,
        lam_quartic=LAM_1D,
        step=eps,
        temperature=T_1D,
    )
    ula_1d[eps] = terminal_samples(
        theta_eps, metropolis=False, reps=reps, num_samples=N_1D, dim=1,
        key=jax.random.key(SEED + 100),
    )
    mala_1d[eps] = terminal_samples(
        theta_eps, metropolis=True, reps=reps, num_samples=N_1D, dim=1,
        key=jax.random.key(SEED + 200),
    )
    ks_ula.append(ks_distance(ula_1d[eps], grid, ref_cdf))
    ks_mala.append(ks_distance(mala_1d[eps], grid, ref_cdf))


# In[13]:


for eps, ku, km in zip(EPS_SWEEP, ks_ula, ks_mala):
    print(f"eps={eps:.2f}  reps={round(T_SIM / eps):5d}  ULA KS={ku:.4f}   MALA KS={km:.4f}")

ks_match = ks_mala[EPS_SWEEP.index(EPS_MATCH)]
ks_bias_ula = ks_ula[EPS_SWEEP.index(EPS_BIAS)]
ks_bias_mala = ks_mala[EPS_SWEEP.index(EPS_BIAS)]

assert ks_match < 0.03, f"MALA does not match the reference: KS {ks_match:.4f}"
assert ks_bias_ula > 0.05, f"ULA bias not visible at eps={EPS_BIAS}: KS {ks_bias_ula:.4f}"
assert ks_bias_ula > 4 * ks_bias_mala, "MALA did not remove ULA's step-size bias"
assert ks_ula[-1] > ks_ula[0], "ULA KS should grow with the step size" 


# The first panel compares the adjusted transition with the exact target at $\varepsilon=\texttt{EPS_MATCH}$. It overlays the MALA histogram with the quadrature Boltzmann density and reports their KS distance. The close overlap and small KS distance show no visible mismatch at this finite-sample resolution; they do not remove the Monte Carlo uncertainty in the histogram.

# In[14]:


fig = plot_mala_reference(
    grid, ref_density, mala_1d[EPS_MATCH], ks=ks_match, temperature=T_1D
)
savefig(fig, "12_langevin_mala_reference")


# The coarse-step comparison isolates the effect of the Metropolis correction. ULA and MALA use the same Langevin proposal, but ULA accepts every proposal whereas MALA may reject one. In the panel, ULA underweights the main mode and overfills the trough relative to the exact density; MALA remains aligned with the reference. The difference is the discretization bias removed by the accept/reject step.

# In[15]:


fig = plot_ula_bias(
    grid,
    ref_density,
    ula_1d[EPS_BIAS],
    mala_1d[EPS_BIAS],
    step=EPS_BIAS,
    ks_ula=ks_bias_ula,
    ks_mala=ks_bias_mala,
)
savefig(fig, "12_langevin_ula_bias")


# Across the step-size sweep, ULA's KS distance to the exact marginal grows with $\varepsilon$ (its $O(\varepsilon)$ discretization bias), while MALA stays flat near the Monte Carlo floor: the accept/reject correction holds at every step size.

# In[16]:


fig = plot_ks_vs_step(EPS_SWEEP, ks_ula, ks_mala)
savefig(fig, "12_langevin_ks_vs_step")


# ## Sampling the full graph
# 
# The one-dimensional experiment validates the transition machinery where an exact target is available. We now use the same MALA gate for the full 10-node energy. The computation advances `NUM_SAMPLES` independent chains from the zero state for `DEPLOY_REPS` Metropolis-adjusted steps and returns their terminal states in one JIT-compiled pass.

# In[17]:


samples = terminal_samples(
    theta, metropolis=True, reps=DEPLOY_REPS, num_samples=NUM_SAMPLES, dim=n,
    key=jax.random.key(SEED),
)
assert samples.shape == (NUM_SAMPLES, n), f"expected ({NUM_SAMPLES}, {n}), got {samples.shape}"
assert np.all(np.isfinite(samples)), "non-finite samples"
print(f"samples shape: {samples.shape}  min={samples.min():.3f}  max={samples.max():.3f}")


# The ten-dimensional target has no tractable quadrature reference, so `DEPLOY_REPS` cannot be assessed by the same exact-density comparison. Instead, we compare the deployed per-site mean soft spins $\langle\tanh x_i\rangle$ with estimates from a much longer MALA run. Site-specific Monte Carlo error bars account for the uncertainty in each pair of estimates, while the standardized residual at a site is its absolute difference divided by its own standard error. This is a finite-sample convergence sanity check: agreement with the longer run is useful evidence, but it is not a proof of stationarity.

# In[18]:


long_samples = terminal_samples(
    theta, metropolis=True, reps=LONGRUN_REPS, num_samples=NUM_SAMPLES, dim=n,
    key=jax.random.key(SEED + 1),
)
mu_deploy = np.mean(np.tanh(samples), axis=0)
mu_long = np.mean(np.tanh(long_samples), axis=0)

# Monte Carlo standard error for each site's two independent mean estimates.
se_deploy = np.std(np.tanh(samples), axis=0, ddof=1) / np.sqrt(NUM_SAMPLES)
se_long = np.std(np.tanh(long_samples), axis=0, ddof=1) / np.sqrt(NUM_SAMPLES)
se_difference = np.sqrt(se_deploy**2 + se_long**2)
site_differences = np.abs(mu_deploy - mu_long)
standardized_residuals = site_differences / se_difference
worst_site = int(np.argmax(standardized_residuals))
conv_err = float(site_differences.max())
max_standardized_residual = float(standardized_residuals[worst_site])
print(
    f"largest |Δ_i|/SE_i = {max_standardized_residual:.2f} at site {worst_site}, "
    f"|Δ_i|={site_differences[worst_site]:.4f}, SE_i={se_difference[worst_site]:.4f}"
)
assert conv_err < 0.06, f"deployed MALA differs from long run: max abs diff {conv_err:.4f}"


# In[19]:


fig = plot_site_mean_parity(
    mu_deploy,
    mu_long,
    se_short=se_deploy,
    se_long=se_long,
    reps_short=DEPLOY_REPS,
    reps_long=LONGRUN_REPS,
)
savefig(fig, "12_langevin_convergence_parity")


# The parity plot labels each site and gives both estimates site-specific error bars, allowing the deployed and long-run means to be compared on the scale of their Monte Carlo uncertainty. The printed maximum $|\Delta_i|/\mathrm{SE}_i$ identifies the worst standardized residual and therefore the site at which the two runs disagree most strongly.

# ## Terminal field and magnetization
# 
# We summarize the deployed terminal samples at two levels. The graph view reports the mean soft spin $\tanh(x_i)$ at each site, while the magnetization distribution retains the variation across sampled paths.
# 
# In the graph view, a diverging colormap encodes the sign and magnitude of the mean soft spin over `NUM_SAMPLES` terminal states. This makes the response to the signed external field visible site by site. The next cell computes both the site means and the per-path magnetizations from the same terminal samples.

# In[20]:


mean_soft_spin = np.mean(np.tanh(samples), axis=0)
terminal_magnetization = np.mean(np.tanh(samples), axis=1)
mean_mag = float(terminal_magnetization.mean())


# To make a color traceable back to a site, each terminal-field node is labeled with its graph-site ID on the first line and its mean soft spin on the second.

# In[21]:


fig = P_sch.plot_terminal_field(graph, pos, mean_soft_spin)
savefig(fig, "12_langevin_terminal_field")


# The field above averages over paths, which hides how much the paths disagree, so we also look at the spread. The magnetization histogram shows the per-path magnetization $m = \frac{1}{n}\sum_i \tanh(x_i)$ with the sample mean marked.

# In[22]:


fig = P_samp.plot_magnetization_histogram_langevin(
    terminal_magnetization,
    title_hist=f"Magnetization distribution  ({NUM_SAMPLES} paths)",
)
savefig(fig, "12_langevin_magnetization")


# For these parameters, the sampled per-path magnetization is broad and has an empirical mean near zero (-0.001). This value should not be interpreted as a symmetry requirement: a zero-sum external field does not by itself force the net magnetization to vanish. The histogram and graph colors report an empirical finite-sample outcome, with the graph exposing the site-specific responses hidden by the aggregate mean.

# ## Energy relaxation diagnostic
# 
# Terminal-state summaries do not show how a chain approached its endpoint. We therefore record the energy $V(x_t)$ along one MALA path. The path starts at $x = 0$, where every term except the quartic well vanishes, so its initial energy is known exactly: $V(x_0) = \tfrac{\lambda_q}{4}\,n$. A decline from this level demonstrates relaxation from the chosen initial state; a 500-step trace is not sufficient to establish stationarity.
# 
# For this diagnostic, we call the custom `LangevinGate` directly rather than using the terminal-state circuit, because the direct call exposes every intermediate state. One compiled scan records $V(x_t)$ together with the acceptance rate.

# In[23]:


def run_mala_path(theta, *, dim, steps, key):
    """Record one directly sampled MALA path and its acceptance rate."""
    gate = LangevinGate(sites=0, dims=(dim,), metropolis=True)

    def body(carry, _):
        x, k = carry
        k, subkey = jax.random.split(k)
        substate = {"discrete": jnp.zeros((0,), dtype=jnp.int32), "continuous": x}
        x_next = gate.sample(subkey, substate, theta)
        moved = jnp.any(jnp.abs(x_next - x) > 0.0)  # a rejection leaves x unchanged
        return (x_next, k), (x, moved)

    # One compiled scan records intermediate states instead of only the terminal state.
    _, (path, moved) = eqx.filter_jit(
        lambda key: jax.lax.scan(body, (jnp.zeros(dim), key), None, length=steps)
    )(key)
    energies = jax.vmap(lambda x: gate.energy(x, theta))(path)
    return np.asarray(energies), float(np.mean(np.asarray(moved)))


# In[24]:


TRACE_STEPS = 500
energies_arr, accept_rate = run_mala_path(
    theta, dim=n, steps=TRACE_STEPS, key=jax.random.key(SEED + 2)
)
assert np.all(np.isfinite(energies_arr)), "non-finite energy trace"

mean_energy = float(energies_arr.mean())
initial_energy = float(energies_arr[0])
tail_energy = float(energies_arr[-100:].mean())  # late-trace level shown in the figure and summary


# In[25]:


fig = P_samp.plot_energy_trace(energies_arr, tail_energy)
savefig(fig, "12_langevin_energy_trace")


# The trace falls below its initial energy but continues to drift late in the run. Accordingly, the dashed line denotes only the mean over the last 100 steps. It summarizes the late part of this path as a relaxation diagnostic and is not an equilibrium-energy estimate.

# ## Verification
# 
# We close with the checks the notebook actually asserts: the MALA acceptance rate has to stay above 0.5, and the recorded path has to relax below its initial energy, which leaves the terminal magnetization outside the assertions entirely and keeps it an empirical summary rather than a symmetry claim.

# In[26]:


assert accept_rate > 0.5, f"MALA acceptance too low: {accept_rate:.3f}"
assert np.isfinite(mean_energy), "mean energy is non-finite"
assert energies_arr.min() < initial_energy, "energy never descended below initial value"
assert tail_energy < initial_energy - 0.1 * abs(initial_energy), (
    f"energy did not relax below the start: last-window mean {tail_energy:.3f} vs "
    f"initial {initial_energy:.3f}"
)


# In[27]:


print(f"-- Langevin graph Ising  (n={n}, deploy reps={DEPLOY_REPS}, {NUM_SAMPLES} paths) --")
print(f"  MALA acceptance rate  : {accept_rate:.3f}")
print(
    f"  deployed vs long-run  : max |Δ_i|/SE_i = {max_standardized_residual:.2f} "
    f"(site {worst_site})"
)
print(f"  mean magnetization    : {mean_mag:+.4f}   (empirical, range [-1, 1])")
print(
    f"  magnetization range   : [{terminal_magnetization.min():+.3f}, {terminal_magnetization.max():+.3f}]"
)
print(f"  initial energy V(x_0) : {initial_energy:+.4f}")
print(f"  last-100-step mean    : {tail_energy:+.4f}  (relaxation diagnostic)")
print(f"  1-D MALA KS (eps={EPS_MATCH}): {ks_match:.4f}")
print(f"  1-D ULA  KS (eps={EPS_BIAS}): {ks_bias_ula:.4f}   (step-size bias MALA removes)")
print("--------------------------------------------------------")


# ## Conclusion
# 
# We used a custom `LangevinGate` to sample the continuous Boltzmann law of a soft-spin graph Ising energy. The gate computes the gradient of $V$ with `jax.grad`, advances the state by one Langevin proposal, and optionally applies the Metropolis correction. A JAX scan repeats the Torx `HybridPCircuit.sample` transition, while `jax.vmap` batches independent chains in one JIT-compiled pass.
# 
# The validation separates questions that require different evidence. On the tractable one-dimensional model, the finite MALA sample was consistent with the exact quadrature density, and the step-size sweep exposed the $O(\varepsilon)$ bias of ULA. On the full graph, where exact quadrature is unavailable, per-site means from the deployed chains were compared with a much longer run using site-specific Monte Carlo errors. The single-path energy trace established relaxation from the zero state but was not treated as evidence of equilibrium.
# 
# Because the nonlinear drift does not admit a closed-form affine-Gaussian moment simulator, these checks cover distinct limitations rather than replacing one another: exact low-dimensional validation tests the transition, and finite-sample deployment diagnostics characterize the observed run.
# 
# Next, [`13_regime_switching_diffusion.ipynb`](13_regime_switching_diffusion.ipynb) drives a Gaussian process with a discrete control pdit.
# 
# ## References
# 
# 1. <span id="ref-roberts-tweedie-1996"></span>Roberts, G.O., Tweedie, R.L. 1996. [*Exponential convergence of Langevin distributions and their discrete approximations*](https://projecteuclid.org/journals/bernoulli/volume-2/issue-4/Exponential-convergence-of-Langevin-distributions-and-their-discrete-approximations/bj/1178291835.full). Bernoulli 2(4), 341-363. Establishes the target-invariance role of the Metropolis adjustment and the possible bias of the unadjusted discretization.
# 2. <span id="ref-parisi-1981"></span>Parisi, G. 1981. [*Correlation functions and computer simulations*](https://doi.org/10.1016/0550-3213(81)90056-0). Nuclear Physics B 180(3), 378-384. Develops the Langevin-dynamics view of sampling a continuous field from its energy.
