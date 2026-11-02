#!/usr/bin/env python
# coding: utf-8

# # Gibbs sampling on a directed factor graph
# 

# <div class="note"><span class="note-t">In brief</span>
# <p>We build a 4×4 Ising Gibbs sampler from a one-spin factor, checkerboard tiling, and a repeated directed factor graph. Because a 4×4 lattice has only 65,536 states, we can check zero-field sampling, temperature sweeps, and field-conditioned pattern completion against exact enumeration rather than against another approximation.</p>
# </div>

# <div class="note"><span class="note-t">Prerequisite</span>
# <p>For background on factors, <a href="15_intro_to_factors.ipynb">15 - Introduction to factors and directed factor graphs</a> introduces the core abstraction before this full Ising sampler.</p>
# </div>

# A **factor** in Torx is a directed conditional $P(\text{output} \mid \text{inputs})$. A custom factor specifies its inputs and output and implements `sample` and `init_params`. The one-spin factor below also implements an exact `log_probability`, giving us an analytic reference for its conditional distribution.
# 
# A parametrised stochastic circuit (PSC) is an ordered list of stochastic gates applied to an initial state. We do not construct a PSC, compile a circuit, or use a simulator here. Instead, we wire hand-written factors into a graph so that each part of the sampling update remains visible.
# 
# [Gibbs sampling](#ref-geman1984) updates variables from their conditional distributions, but it is not Torx's main use case; [thrml](https://github.com/extropic-ai/thrml) is the dedicated library for Gibbs sampling. Why use Gibbs here, then? One sweep brings together the complete `DFG` workflow: a custom factor with an analytic conditional, weight tying, tiling, deterministic reassembly, and an exact finite-state reference against which we can compare the result.
# 
# <details class="tx-disclosure"><summary>Term: Gibbs and chromatic Gibbs sampling</summary><p>Gibbs sampling resamples variables from their conditional target distributions. In <a href="#ref-gonzalez2011">chromatic Gibbs sampling</a>, same-color spins have no edges between them, so they are conditionally independent given the other color and can update together before the colors swap.</p></details>
# 
# The coloring follows [notebook 06](06_ising_sampling_contrastive_divergence.ipynb), where non-adjacent spins update together. That notebook uses `PNOT` gates on an 8-spin ring; here we express the same chromatic-Gibbs conditional with hand-built factors on a $4\times4$ torus. Thus, the update rule and coloring strategy agree, while the graphs and Ising distributions differ.
# 
# One design choice will let us reuse the graph throughout the notebook: the per-site `field` is an input rather than a baked-in weight. At zero field the graph represents the symmetric Ising magnet. Supplying a finite field at selected sites turns the same graph into a field-conditioned pattern-completion model, without changing its wiring.
# 
# The $4\times4$ torus contains $2^{16} = 65{,}536$ configurations, all of which fit in memory. We can therefore compute the exact distribution by brute force for any fixed field and compare each sampling result with that distribution rather than with another approximation.
# 
# <details class="tx-disclosure"><summary>What runs where?</summary><ul>
# <li>Torx runs the stochastic factor, tiling, directed factor graph, and weight-tied chain.</li>
# <li>Notebook code builds the exact distribution and quantitative comparisons.</li>
# <li><code>examples/helpers/_plots_schematics.py</code> owns lattice geometry and its diagram.</li>
# <li><code>examples/helpers/_plots_sampling.py</code>, <code>examples/helpers/_notebook_paths.py</code>, and <code>examples/helpers/_notebook_style.py</code> own batching, statistics, plots, paths, and styling.</li>
# </ul></details>
# 
# We proceed in seven steps:
# 
# 1. set up the $4\times4$ Boltzmann machine and draw its `DFG`,
# 2. write the one-spin update as a custom factor and check its draws against the conditional it reports analytically,
# 3. tile the update into two color blocks and wire them into an explicit `DFG` for one Gibbs sweep,
# 4. repeat that sweep as a weight-tied chain, build the exact distribution by brute-force enumeration, and check the sampler against it,
# 5. sweep the temperature and watch the magnet order as it cools,
# 6. drive the model with a site-varying field and compare the sampler against that same exact distribution spin by spin and connection by connection, and
# 7. drive the graph with a strong finite field for field-conditioned pattern completion.

# ## Setup
# 
# The setup cells put the local helper directory on the path, then fix the plotting style and the seed so every figure and draw below reproduces.

# In[1]:


from pathlib import Path
import sys

SEARCH_ROOTS = (Path.cwd().resolve(), *Path.cwd().resolve().parents)
HELPER_DIR = next(
    (
        root / "examples" / "helpers"
        for root in SEARCH_ROOTS
        if (root / "examples" / "helpers").is_dir()
    ),
    None,
)
if HELPER_DIR is None:
    raise FileNotFoundError("could not locate examples/helpers from the current directory")
ROOT = HELPER_DIR.parent
sys.path.insert(0, str(HELPER_DIR))


# In[2]:


import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np



# In[3]:


from _notebook_paths import figure_dir
from _notebook_style import apply_notebook_style, make_savefig
import _plots_sampling as P_samp
import _plots_schematics as P_sch



# In[4]:


from torx import (
    AbstractHasLogProbability,
    AbstractReferenceFactor,
    ChainFactor,
    DeterministicFactor,
    DFG,
    Site,
    TiledFactor,
)


# In[5]:


apply_notebook_style()
FIGURE_DIR = figure_dir(ROOT)
SEED = 15
rng = np.random.default_rng(SEED)


savefig = make_savefig(FIGURE_DIR)


# ## The model as a Boltzmann machine
# 
# The model is a $4\times4$ grid of spins with periodic boundaries, so every spin has exactly four neighbors. This is the finite Ising model used as a Boltzmann machine by [Ackley, Hinton, and Sejnowski (1985)](#ref-ackley1985).
# 
# The grid holds 16 spins and 32 connections, and the state is stored as spins $s \in \{-1, +1\}$. Internally each update draws a Bernoulli **pbit** in $\{0, 1\}$ and maps it to the stored spin via $s = 2\,\text{pbit} - 1$, so the two representations describe the same variable in different coordinates. The neighbor lists, connection list, and checkerboard coloring come from `examples/helpers/_plots_schematics.py`.
# 
# The energy says what the model prefers. Neighboring spins prefer to agree, and a per-site field $b_i$ biases each spin.
# 
# $$
# H(s) = -J\,\sum_{(i,j)\in E} s_i\, s_j \;-\; \sum_i b_i\, s_i,
# \qquad \pi(s) \propto e^{-\beta H(s)}.
# $$
# 
# Lower energy means higher probability, and $\beta$ is the inverse temperature.
# 
# With $b = 0$ the model is the symmetric Ising magnet. A finite nonzero $b_i$ biases spin $i$ but doesn't hold it fixed, which is what makes the pattern completion at the end of the notebook a sampling problem rather than a clamping trick.

# In[6]:


GRID = 4
N = GRID * GRID
J = 1.0  # ferromagnetic coupling: neighbors prefer to agree
beta = 0.45  # baseline inverse temperature for the low-temperature comparison

EDGES = P_sch.lattice_edges()
COLOR_A, COLOR_B = P_sch.checkerboard_colors()
NEIGHBORS_OF = [P_sch.lattice_neighbors(i) for i in range(N)]

print(
    f"{N} spins, {len(EDGES)} connections, two colors of size "
    f"{len(COLOR_A)} and {len(COLOR_B)}"
)
print(f"every spin has {len(NEIGHBORS_OF[0])} neighbors (periodic)")


# The figure below is the lattice the `DFG` will sweep, drawn so that the update order is visible.
# 
# The 16 spins split into two color blocks. Solid links are the neighbor dependencies inside the grid, and the dashed stubs are the periodic wrap-around connections that close the lattice into a torus. The two colors are the two checkerboard update groups, and every spin reads four neighbors that all lie in the other block, which is what makes a whole block safe to update at once. Color `A` reads the incoming state, and color `B` then reads the freshly updated color `A`.

# In[7]:


fig = P_sch.plot_factor_graph()
savefig(fig, "16_factor_graph")


# ## The per-spin update as a factor
# 
# We begin with the smallest stochastic part of the model: updating one spin given its four neighbors and local `drive`. Once this conditional is defined, tiling and graph assembly will extend it to the full lattice.
# 
# The update is a directed factor that supplies its input and output specifications and implements `sample` and `init_params`.
# 
# It first draws a Bernoulli pbit in $\{0, 1\}$ with probability $\sigma(2\gamma_i)$, where $\sigma$ is the logistic sigmoid and $\gamma_i$ is the local field. The pbit is then mapped to the stored spin $s_i = 2\,\text{pbit} - 1 \in \{-1, +1\}$.
# 
# The draw uses `jax.random.bernoulli`. Both inputs to the local field already include the inverse temperature: we provide $\text{drive}_i = \beta\,b_i$ and set the factor parameter to $j = \beta J$. By contrast, the brute-force `exact_distribution` defined later accepts the physical field $b_i$ and applies $\beta$ itself. These two interfaces compute the same conditional:
# 
# $$
# \gamma_i = \beta\,b_i + \beta J\,\sum_{j\in\mathcal{N}(i)} s_j,
# \qquad
# P(s_i = +1 \mid \mathcal{N}(i),\, b_i) = \sigma(2\gamma_i).
# $$

# In[8]:


SPIN = jax.ShapeDtypeStruct((), jnp.int32)  # one spin, -1 or +1
NEIGHBORS = jax.ShapeDtypeStruct((4,), jnp.int32)  # its four neighbor spins
DRIVE = jax.ShapeDtypeStruct((), jnp.float32)  # this spin's local drive (beta * b_i)




# In[9]:


class SpinUpdate(AbstractReferenceFactor, AbstractHasLogProbability):
    r"""$P(s_i \mid \mathcal{N}(i))$: stored as a spin in $\{-1, +1\}$, drawn via one
    Bernoulli pbit with probability $\sigma(2\gamma_i)$ then mapped to a spin.

    The `drive` input is the temperature-scaled bias `beta * b_i`; the raw
    field `b_i` is scaled before it reaches the kernel. Besides the
    specs, a custom factor implements `sample` and `init_params`, and
    `AbstractReferenceFactor` supplies the generic `sample_with_references`.
    `log_probability` exactly evaluates this one-spin conditional only, not the
    joint 16-spin distribution or the full-chain distribution.
    """

    input_ports: dict[str, jax.ShapeDtypeStruct] = eqx.field(static=True)
    output_spec: jax.ShapeDtypeStruct = eqx.field(static=True)

    def __init__(self):
        self.input_ports = {"neighbors": NEIGHBORS, "drive": DRIVE}
        self.output_spec = SPIN

    def init_params(self, key):
        return {"j": jnp.zeros((), jnp.float32)}

    def _local_field(self, inputs, params):
        return inputs["drive"] + params["j"] * inputs["neighbors"].sum()

    def sample(self, key, inputs, params, info=None, site_info=None, return_aux=False):
        # draw a Bernoulli pbit in {0, 1}, then map to the stored spin {-1, +1}
        draw = jax.random.bernoulli(
            key, jax.nn.sigmoid(2 * self._local_field(inputs, params))
        ).astype(jnp.int32)
        spin = 2 * draw - 1
        return (spin, None) if return_aux else spin

    def log_probability(
        self, inputs, outputs, params, info=None, site_info=None, return_aux=False
    ):
        # Exact one-spin conditional only: P(s_i | neighbors, drive).
        valid = (outputs == 1) | (outputs == -1)
        log_prob = jnp.where(
            valid,
            jax.nn.log_sigmoid(2 * self._local_field(inputs, params) * outputs),
            -jnp.inf,
        )
        return (log_prob, None) if return_aux else log_prob


# Before assembling the lattice, we check the factor in isolation. A small error in the conditional can still produce plausible-looking configurations, so visual inspection is not enough. We fix the neighbors and `drive`, compute the resulting local field $\gamma$, draw many updates, and compare the observed fraction of $+1$ spins with $\sigma(2\gamma)$.
# 
# This check establishes that `sample` follows the intended single-spin conditional. We next ask whether `log_probability` reports the same distribution.

# In[10]:


probe = SpinUpdate()

probe_neighbors = jnp.array([1, 1, 1, -1], dtype=jnp.int32)  # sums to +2
probe_drive = 0.30  # scaled local bias the kernel consumes
probe_coupling = beta * J
probe_params = {"j": probe_coupling}
gamma = probe_drive + probe_coupling * probe_neighbors.sum()
target_p = float(jax.nn.sigmoid(2 * gamma))

n_probe = 20000
keys = jax.random.split(jax.random.key(SEED), n_probe)
inputs = {"neighbors": probe_neighbors, "drive": probe_drive}
draws = jax.vmap(lambda k: probe.sample(k, inputs, probe_params))(keys)
probe_p = float((np.asarray(draws) == 1).mean())

print(f"gamma = {float(gamma):.4f}  ->  target sigmoid(2*gamma) = {target_p:.4f}")
print(f"sampled fraction of +1 spins                     = {probe_p:.4f}")


# With the empirical fraction matched, we read the same conditional back analytically to confirm that `log_probability` describes the distribution `sample` actually draws from. The analytic value should agree with the sampled one, and the two outcomes should sum to one.

# In[11]:


logp_p_up = float(np.exp(probe.log_probability(inputs, 1, probe_params)))
logp_mass = float(
    np.exp(probe.log_probability(inputs, 1, probe_params))
    + np.exp(probe.log_probability(inputs, -1, probe_params))
)

print(
    f"log_probability gives P(+1) = {logp_p_up:.4f},  outcomes sum to {logp_mass:.6f}"
)
np.testing.assert_allclose(probe_p, target_p, atol=0.01)
np.testing.assert_allclose(logp_p_up, target_p, atol=1e-5)
np.testing.assert_allclose(logp_mass, 1.0, atol=1e-6)


# ## Wiring the lattice into a directed factor graph
# 
# The two checkerboard colors are the two Gibbs blocks. Block Gibbs means we resample a whole group of variables in one move instead of one variable at a time, which is valid here because no two spins of the same color are neighbors, so inside a block the conditionals don't depend on each other. Every spin in a block updates from the same conditional, so each block is the single `SpinUpdate` factor wrapped in a `TiledFactor`: one tile per spin, eight tiles per block.
# 
# The tiles share one coupling, and each tile receives its own neighbors and `drive` as data. The geometry never changes during sampling, so we precompute each tile's neighbor indices once.

# In[12]:


SPINS = jax.ShapeDtypeStruct((N,), jnp.int32)  # the full state
DRIVES = jax.ShapeDtypeStruct((N,), jnp.float32)  # per-site drive (beta * field)
BLOCK = len(COLOR_A)  # 8 spins per color
BLOCK_SPINS = jax.ShapeDtypeStruct((BLOCK,), jnp.int32)



# In[13]:


COLOR_A_IDX = jnp.array(COLOR_A, dtype=jnp.int32)
COLOR_B_IDX = jnp.array(COLOR_B, dtype=jnp.int32)
position_in_a = {site: tile for tile, site in enumerate(COLOR_A)}



# In[14]:


# Per-tile neighbor lookups, fixed by the geometry. Color A reads its four
# neighbors out of the incoming state; color B reads them out of the freshly
# updated color A block (every neighbor of a color B site is a color A site).
A_NEIGHBORS = jnp.array([NEIGHBORS_OF[i] for i in COLOR_A], dtype=jnp.int32)
B_NEIGHBORS = jnp.array(
    [[position_in_a[j] for j in NEIGHBORS_OF[i]] for i in COLOR_B], dtype=jnp.int32
)


# Wiring a `Site` into a `DFG` means saying where its inputs come from, which is the job of a porting function. Each block needs one that slices its tiles' neighbors and `drive` out of the parent outputs, and a third function then scatters the two blocks back into one state vector.

# In[15]:


def gather_block_a(parent_outputs):
    spins, drives = parent_outputs
    return {"neighbors": spins[A_NEIGHBORS], "drive": drives[COLOR_A_IDX]}




# In[16]:


def gather_block_b(parent_outputs):
    """Route block-A spins and B-site drives into block B's kernel inputs."""
    block_a, drives = parent_outputs
    return {"neighbors": block_a[B_NEIGHBORS], "drive": drives[COLOR_B_IDX]}




# In[17]:


def scatter_blocks(inputs, site_info):
    """Place the two color blocks back into one lattice-ordered state vector."""
    state = jnp.zeros(N, dtype=jnp.int32)
    state = state.at[COLOR_A_IDX].set(inputs["block_a"])
    state = state.at[COLOR_B_IDX].set(inputs["block_b"])
    return state


# In[18]:


block = TiledFactor(SpinUpdate(), BLOCK, weight_tied=True)  # 8 spins in parallel
reassemble = DeterministicFactor(
    scatter_blocks, {"block_a": BLOCK_SPINS, "block_b": BLOCK_SPINS}, SPINS
)


# We now assemble one complete Gibbs sweep from three `Site` objects. The first site updates color `A`, the second updates color `B` using the new color-`A` values, and the final deterministic site reassembles both blocks into a state vector.
# 
# Because the eight conditionally independent spins in a block update in parallel, each color appears as one tiled factor rather than as eight separate `Site` objects.

# In[19]:


sites = (
    Site(
        "block_a",
        block,
        parents=("spins", "drive"),
        porting_fn=gather_block_a,
        param_key="coupling",
        info_key=None,
        site_info=None,
    ),
    Site(
        "block_b",
        block,
        parents=("block_a", "drive"),
        porting_fn=gather_block_b,
        param_key="coupling",
        info_key=None,
        site_info=None,
    ),
    Site(
        "state",
        reassemble,
        parents=("block_a", "block_b"),
        porting_fn=("block_a", "block_b"),
        param_key=None,
        info_key=None,
        site_info=None,
    ),
)



# In[20]:


sweep = DFG(
    sites=sites, input_ports={"spins": SPINS, "drive": DRIVES}, output_name="state"
)
print(
    f"directed factor graph: {len(sites)} sites "
    f"(2 tiled blocks of {BLOCK} + 1 reassembly)"
)


# ## Running the chain
# 
# The `DFG` above performs one Gibbs sweep. To obtain a Markov chain, we repeat that graph with a weight-tied `ChainFactor`, so every step uses the same sweep and the same coupling parameters.
# 
# Only the spin state feeds back from one sweep to the next. The `drive`, which contains the $\beta$-scaled field, is a non-feedback input and therefore remains fixed throughout a chain.
# 
# We run 200 sweeps for each of 4,000 independently initialized chains and retain one final state from each chain. The resulting 4,000 draws come from separate chains rather than from successive, correlated states of one trajectory.

# In[21]:


n_steps = 200  # Gibbs sweeps per chain
settling_mark = 80  # heuristic marker shown on the settling trace
n_chains = 4000

chain = ChainFactor(sweep, n_steps=n_steps, feedback_porting_fn="spins", weight_tied=True)




# In[22]:


def run(key, beta, init, field):
    # field is the raw physical field; scale it and the coupling by beta into the
    # drive the kernel consumes for this temperature.
    """Run the full `n_steps`-sweep chain at inverse temperature `beta`."""
    inputs = {"spins": init, "drive": beta * field}
    return chain.sample(key, inputs, params={"coupling": {"j": beta * J}})


@eqx.filter_jit
def run_many(keys, beta, inits, field):
    # reusable batched sampler; beta stays a traced scalar so changing the
    # temperature reuses this compile instead of recompiling per beta.
    """Batched `run` over chains."""
    return jax.vmap(lambda k, x: run(k, beta, x, field), in_axes=(0, 0))(keys, inits)




# In[23]:


zero_field = jnp.zeros(N, dtype=jnp.float32)
inits = (
    jax.random.bernoulli(jax.random.key(SEED), 0.5, (n_chains, N)).astype(jnp.int32) * 2
    - 1
)
keys = jax.random.split(jax.random.key(SEED + 1), n_chains)
# Run independent chains from independent starts and keep one final state per chain.
samples = np.asarray(
    run_many(keys, jnp.asarray(beta, jnp.float32), inits, zero_field)
)
print(f"drew {samples.shape[0]} final-chain samples of {samples.shape[1]} spins each")


# Before comparing final-sample distributions, we examine whether 200 sweeps gives the chains time to settle from a disordered start. At each sweep we compute the **order parameter**, the magnitude of the mean spin for each chain, and then summarize it across chains. The order parameter is 0 for a lattice with no net direction and 1 for a fully aligned lattice.
# 
# The bold curve shows the mean order parameter, and the band spans the 16th to 84th percentiles across chains. The dashed line marks sweep 80. This hand-chosen reference lets us compare the apparent leveling of the trace with the full 200-sweep runtime; it is not an estimated mixing time.

# In[24]:


n_trace = 1000  # chains tracked for the heuristic settling trace
trace_init = (
    jax.random.bernoulli(jax.random.key(SEED + 2), 0.5, (n_trace, N)).astype(jnp.int32)
    * 2
    - 1
)




# In[25]:


@eqx.filter_jit
def settling_trace(key, init_state):
    # one lax.scan over sweeps, vmapped across chains: records the per-step order
    # parameter for every chain inside a single dispatch, no per-step host sync.
    """Record the per-sweep order parameter for chains from a disordered start."""
    def sweep_all(state, k):
        def one_sweep(kk, x):
            return sweep.sample(
                kk,
                {"spins": x, "drive": beta * zero_field},
                params={"coupling": {"j": beta * J}},
            )

        new = jax.vmap(one_sweep, in_axes=(0, 0))(
            jax.random.split(k, init_state.shape[0]), state
        )
        return new, jnp.abs(new.mean(axis=1))

    keys = jax.random.split(key, n_steps)
    _, orders = jax.lax.scan(sweep_all, init_state, keys)
    init_order = jnp.abs(init_state.mean(axis=1))
    return jnp.concatenate([init_order[None, :], orders], axis=0)




# In[26]:


# (n_trace, n_steps + 1) for plotting
order_traces = np.asarray(settling_trace(jax.random.key(SEED + 4), trace_init)).T


# The order parameter rises from its disordered initial value and becomes approximately flat before the marker. This is evidence that this observable has settled, but it does not establish that the chains have mixed: the magnitude cannot reveal whether a chain moves between the two mirror-image ordered states. We therefore treat the trace as a diagnostic and use exact-distribution comparisons to evaluate the final samples after 200 sweeps.

# In[27]:


fig = P_samp.plot_settling_trace(order_traces, settling_step=settling_mark)
savefig(fig, "16_settling_trace")


# The first comparison asks whether the sampler reproduces the right distribution of total magnetization, the sum of all spins in each sample. We enumerate the exact distribution for the zero-field case and put the two histograms side by side.
# 
# At this inverse temperature the magnet is ordered, so the exact histogram should have two peaks, one for each of the two mirror-image ordered states. We summarise the difference between the histograms with the [total variation distance](https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures), which is half the summed absolute probability difference across magnetization bins. It ranges from 0 for identical histograms to 1 for disjoint histograms.

# In[28]:


# Enumerate all 2**16 configurations once: bits and spins.
ALL_BITS = ((np.arange(2**N)[:, None] >> np.arange(N - 1, -1, -1)) & 1).astype(np.int8)
ALL_SPINS = (2 * ALL_BITS - 1).astype(np.int8)
edge_i = np.array([i for i, _ in EDGES])
edge_j = np.array([j for _, j in EDGES])




# In[29]:


def exact_distribution(field):
    """Exact Boltzmann probabilities over all 2**16 configs for a fixed field."""
    agreement = (ALL_SPINS[:, edge_i] * ALL_SPINS[:, edge_j]).sum(axis=1)
    energy = -J * agreement - ALL_SPINS @ np.asarray(field, dtype=float)
    logp = -beta * energy
    logp -= logp.max()
    p = np.exp(logp)
    return p / p.sum()




# In[30]:


exact_p = exact_distribution(np.zeros(N))
exact_mag = ALL_SPINS.sum(axis=1)

levels = np.arange(-N, N + 1, 2)
exact_level_probs = np.array([exact_p[exact_mag == m].sum() for m in levels])
sampler_mag = P_samp.total_magnetization(samples)
sampler_level_probs = np.array([(sampler_mag == m).mean() for m in levels])
mag_gap = (
    0.5 * np.abs(sampler_level_probs - exact_level_probs).sum()
)  # total variation on magnetization

print(f"exact distribution sums to {exact_p.sum():.6f}")
print(f"gap to exact (magnetization total variation) = {mag_gap:.4f}")


# Overlaid on the exact line, the sampled counts reproduce both peaks and the trough between them, at a total variation distance of 0.0222 from the exact distribution.

# In[31]:


fig = P_samp.plot_magnetization_histogram_lattice(sampler_mag, levels, exact_level_probs, gap=mag_gap)
savefig(fig, "16_magnetization_vs_exact")


# ## Cooling the model
# 
# Cooling increases magnetic order. We measure that change with the same order parameter: 0 indicates no net direction, while 1 indicates complete alignment.
# 
# We evaluate six inverse temperatures from 0.10 to 0.60 and compare three curves. Exact enumeration gives the order parameter for this $4\times4$ torus. The sampler curve tests whether the chain follows that reference across the full range, rather than only at the baseline temperature considered above. Finally, the [mean-field approximation](https://www.scholarpedia.org/article/Mean_field_theory) replaces neighboring spins with their expected average. Because this replacement removes the fluctuations that are important near a transition, we expect mean field to differ most in the middle of the range.
# 
# Two reference points help interpret the comparison. Mean field predicts zero order below its critical inverse temperature $\beta = 1/(4J) = 0.25$, whereas the infinite square lattice has the [exact critical point](#ref-onsager1944) $\beta_c J = \tfrac{1}{2}\log(1+\sqrt{2}) \approx 0.4407$. A $4\times4$ torus is too small to have a true phase transition at either value. Instead, it exhibits a [finite-size crossover](#ref-fisher1972): a smooth change in order near the infinite-lattice critical point.
# 
# <details class="tx-disclosure"><summary>Where the sampler curve comes from</summary><p>The sampler curve comes from <code>sweep_temperatures</code> in <code>examples/helpers/_plots_sampling.py</code>, which runs fresh chains at each inverse temperature and reports their order parameter.</p></details>

# In[32]:


def exact_order(beta_val):
    """Exact order parameter E|mean spin| at a given inverse temperature."""
    agreement = (ALL_SPINS[:, edge_i] * ALL_SPINS[:, edge_j]).sum(axis=1)
    logp = beta_val * (J * agreement)  # zero field
    logp -= logp.max()
    p = np.exp(logp)
    p /= p.sum()
    abs_mean_spin = np.abs(ALL_SPINS.mean(axis=1))
    return float((p * abs_mean_spin).sum())




# In[33]:


def mean_field_order(beta_val, iters=500):
    """Self-consistent mean-field magnetization m = tanh(4*beta*J*m)."""
    m = 0.9
    for _ in range(iters):
        m = np.tanh(4.0 * beta_val * J * m)
    return abs(m)


# In[34]:


betas = np.linspace(0.10, 0.60, 6)
exact_curve = np.array([exact_order(b) for b in betas])
mf_curve = np.array([mean_field_order(b) for b in betas])
sampler_curve = P_samp.sweep_temperatures(
    run,
    betas,
    n_chains=2000,
    seed=SEED + 10,
)

print("beta      exact   sampler  mean-field")
for b, e, s, m in zip(betas, exact_curve, sampler_curve, mf_curve, strict=True):
    print(f"{b:.2f}     {e:.3f}    {s:.3f}    {m:.3f}")


# Plotted together, the sampler stays within 0.012 of the exact curve at every inverse temperature, while mean field runs high through the crossover region, reading 0.659 against the exact 0.520 at $\beta = 0.30$.

# In[35]:


fig = P_samp.plot_temperature_sweep(betas, exact_curve, sampler_curve, mf_curve)
savefig(fig, "16_temperature_sweep")


# ## Checking against the exact answer
# 
# The magnetization histogram is a useful aggregate, but it is not sufficient: different joint distributions can have the same total-magnetization distribution. Because all 16-spin states can be enumerated, `exact_distribution` gives us a stronger reference for any fixed field. We can compare the sampler with it separately for each spin and each connection.

# For this comparison, we apply a field that varies from site to site. The field breaks the equivalence among spins, giving each site a distinct exact marginal.
# 
# The left panel compares the sampled and exact mean of every spin. The right panel makes the corresponding comparison for mean agreement across every connection. In both panels, the diagonal represents exact agreement.

# In[36]:


parity_field = jnp.asarray(rng.normal(0.0, 0.6, size=N), dtype=jnp.float32)

parity_keys = jax.random.split(jax.random.key(SEED + 20), n_chains)
parity_inits = (
    jax.random.bernoulli(jax.random.key(SEED + 21), 0.5, (n_chains, N)).astype(
        jnp.int32
    )
    * 2
    - 1
)
parity_samples = np.asarray(
    run_many(parity_keys, jnp.asarray(beta, jnp.float32), parity_inits, parity_field)
)



# In[37]:


# exact per-spin magnetization and per-connection agreement at this field
exact_pf = exact_distribution(np.asarray(parity_field))
exact_spin = (exact_pf[:, None] * ALL_SPINS).sum(axis=0)
exact_conn = np.array(
    [float((exact_pf * ALL_SPINS[:, i] * ALL_SPINS[:, j]).sum()) for i, j in EDGES]
)



# In[38]:


# sampler per-spin magnetization and per-connection agreement
sampler_spin = parity_samples.mean(axis=0)
sampler_conn = np.array(
    [(parity_samples[:, i] * parity_samples[:, j]).mean() for i, j in EDGES]
)



# In[39]:


spin_err = float(np.max(np.abs(sampler_spin - exact_spin)))
conn_err = float(np.max(np.abs(sampler_conn - exact_conn)))
print(f"max per-spin error       = {spin_err:.4f}")
print(f"max per-connection error = {conn_err:.4f}")


# Each spin and each connection becomes one point, sampler against exact, so a point on the dashed diagonal means the sampler recovered the exact value. Every point sits on the diagonal to within 0.0293 per spin and 0.0232 per connection, and the spread along the diagonal comes from the per-site field, which pulls different spins to different magnetizations.

# In[40]:


fig = P_samp.plot_parity(exact_spin, sampler_spin, exact_conn, sampler_conn)
savefig(fig, "16_parity")


# ## Driving it with a field
# 
# Because `field` is an input, we can now use the same graph for field-conditioned pattern completion.
# 
# We apply a field of magnitude 6 at a small set of sites, leave the remaining sites at zero, and run the chain. The selected spins are still stochastic: the finite field strongly favors its sign but does not pin them. The zero-field sites then sample around those biases. For this reason, we compare the biased-site marginals with the exact field-conditioned distribution instead of checking only whether their signs agree.
# 
# The result is the field-conditioned Boltzmann machine $P(\text{spins} \mid \text{field})$. Relative to the previous experiments, only the `field` input has changed.

# In[41]:


# strongly bias a few top-half sites toward +1 and a few bottom-half toward -1
pattern = np.zeros(N)
bias_up = [0, 3, 5]
bias_down = [10, 12, 15]
pattern[bias_up] = +1.0
pattern[bias_down] = -1.0

field_magnitude = 6.0
# the finite field biases selected sites without holding their spins fixed
field = field_magnitude * jnp.asarray(pattern, dtype=jnp.float32)



# In[42]:


n_complete = 3000
complete_keys = jax.random.split(jax.random.key(SEED + 20), n_complete)
complete_init = (
    jax.random.bernoulli(jax.random.key(SEED + 21), 0.5, (n_complete, N)).astype(
        jnp.int32
    )
    * 2
    - 1
)
completed = np.asarray(
    run_many(complete_keys, jnp.asarray(beta, jnp.float32), complete_init, field)
)

one_sample = completed[0]
average = completed.mean(axis=0)



# In[43]:


# exact conditional at this field: run() feeds the kernel beta*field as the drive
# and exact_distribution applies beta to the same raw field, so they match.
exact_complete = exact_distribution(np.asarray(field))
exact_avg = (exact_complete[:, None] * ALL_SPINS).sum(axis=0)



# In[44]:


biased_mask = np.asarray(pattern) != 0.0
# check biased-site marginals and alignment against the exact field-conditioned result
biased_align = float(
    np.mean(np.sign(average[biased_mask]) == np.sign(np.asarray(field)[biased_mask]))
)
biased_marginal_err = float(np.max(np.abs(average[biased_mask] - exact_avg[biased_mask])))
complete_err = float(np.max(np.abs(average - exact_avg)))
print(f"biased sites aligned with their field sign: {biased_align:.0%}")
print(f"biased-site marginal max error vs exact:   {biased_marginal_err:.4f}")
print(f"all-site marginal max error vs exact:      {complete_err:.4f}")


# The figure has three panels. The first is the field we imposed, where colored sites carry positive or negative bias with $|b_i|=6$ and pale sites have $b_i=0$. The other two show what the chain did with it: one field-conditioned sample, and the mean spin across all 3,000 chains.

# In[45]:


fig = P_samp.plot_pattern_completion(np.asarray(field), one_sample, average)
savefig(fig, "16_pattern_completion")


# ## Verification
# 
# We gather the quantitative checks for this tutorial in one place, and we assert them rather than only printing them, so a drift in any number fails the notebook. The single-spin probe is checked against its own analytic sigmoid conditional, which tests the factor in isolation from the graph. The magnetization, per-spin, per-connection, temperature, and field-completion checks are all measured against the brute-force exact distribution, which is the one reference here that carries no sampling error of its own.

# In[46]:


# enforce the printed claims instead of only printing them
assert abs(probe_p - target_p) < 0.02, (
    f"single-spin probe {probe_p:.4f} off target {target_p:.4f}"
)
assert mag_gap < 0.10, f"magnetization gap {mag_gap:.4f} >= 0.10"
assert spin_err < 0.10, f"max per-spin error {spin_err:.4f} >= 0.10"
assert conn_err < 0.10, f"max per-connection error {conn_err:.4f} >= 0.10"
# cooling raises the order parameter monotonically (warm -> cold)
np.testing.assert_array_less(-np.diff(exact_curve), 1e-9)
assert exact_curve[-1] > exact_curve[0], "order parameter did not rise with cooling"
# temperature sweep: the sampler curve tracks the exact enumeration
temp_err = float(np.max(np.abs(sampler_curve - exact_curve)))
assert temp_err < 0.08, f"temperature sweep off exact by {temp_err:.4f}"
# field-conditioned completion: sampler marginals match exact enumeration and the
# biased sites align with their field sign
assert complete_err < 0.10, f"completion marginal error {complete_err:.4f} >= 0.10"
assert biased_align > 0.99, f"biased sites did not align: {biased_align:.2f}"



# In[47]:


print("all checks passed")
print(f"  single-spin probe        = {probe_p:.4f}  (target {target_p:.4f})")
print(f"  magnetization gap        = {mag_gap:.4f}  (< 0.10)")
print(f"  max per-spin error       = {spin_err:.4f}  (< 0.10)")
print(f"  max per-connection error = {conn_err:.4f}  (< 0.10)")
print(f"  temperature sweep error  = {temp_err:.4f}  (< 0.08)")
print(f"  completion marginal err  = {complete_err:.4f}  (< 0.10)")
print(f"  order parameter warm->cold = {exact_curve[0]:.3f} -> {exact_curve[-1]:.3f}")


# ## Conclusion
# 
# We constructed a Boltzmann machine from hand-written factors, sampled it with Gibbs updates, and compared its outputs with exact enumeration.
# 
# - A single spin update is one directed factor. It draws a pbit with value $1$ with probability $\sigma(2\gamma_i)$ and reports the same one-spin conditional through `log_probability`; the probe showed agreement between the sampled fraction and analytic probability.
# - The checkerboard colors form two Gibbs blocks. Each block tiles the same factor over eight spins with one shared coupling, and a `DeterministicFactor` reassembles the blocks. Together, these three sites define the `DFG` for one sweep.
# - A weight-tied `ChainFactor` repeats the sweep to form an MCMC chain. After 200 sweeps, final states from independent chains match the exact distribution within the reported per-spin and per-connection errors.
# - As the model cools, the order parameter rises through the finite-size crossover. The sampler follows the exact curve, while mean field overestimates the order in the crossover region.
# - Since `field` is an input, the same graph also performs finite-field pattern completion without any change to its wiring, representing $P(\text{spins} \mid \text{field})$.
# 
# [Notebook 06](06_ising_sampling_contrastive_divergence.ipynb) applies the same chromatic-Gibbs conditional with `PNOT` gates on an 8-spin ring. Here, the conditional is expressed as factors on a $4\times4$ torus; the update rule and coloring strategy are shared, while the graph and Ising distribution differ.
# 
# Gibbs sampling provides a concrete way to exercise the `DFG` workflow: an analytic custom conditional is tiled, weight-tied, and combined with deterministic reassembly, and the resulting graph can be checked exactly. For Gibbs sampling as a working tool rather than as a demonstration of these Torx components, use [thrml](https://github.com/extropic-ai/thrml).

# ## References
# 
# 1. Extropic. 2026. [thrml](https://github.com/extropic-ai/thrml). This is the dedicated Gibbs sampling library mentioned in the motivation.
# 2. Extropic. 2026. [Notebook 06: Ising sampling and contrastive divergence](06_ising_sampling_contrastive_divergence.ipynb). This companion notebook implements the same chromatic-Gibbs update idea on a different 8-spin ring distribution.
# 3. Extropic. 2026. [Notebook 15: Introduction to factors and directed factor graphs](15_intro_to_factors.ipynb). This companion notebook introduces the factor and `DFG` APIs used here.
# 4. <span id="ref-ackley1985"></span>Ackley, D.H., Hinton, G.E., Sejnowski, T.J. 1985. [*A learning algorithm for Boltzmann machines*](https://www.cs.toronto.edu/~hinton/absps/cogscibm.pdf). Cognitive Science 9(1), 147-169.
# 5. <span id="ref-geman1984"></span>Geman, S., Geman, D. 1984. [*Stochastic relaxation, Gibbs distributions, and the Bayesian restoration of images*](https://doi.org/10.1109/TPAMI.1984.4767596). IEEE Transactions on Pattern Analysis and Machine Intelligence 6(6), 721-741.
# 6. <span id="ref-gonzalez2011"></span>Gonzalez, J.E., Low, Y., Gretton, A., Guestrin, C. 2011. [*Parallel Gibbs sampling: From colored fields to thin junction trees*](https://proceedings.mlr.press/v15/gonzalez11a.html). Proceedings of AISTATS 15, 324-332.
# 7. <span id="ref-fisher1972"></span>Fisher, M.E., Barber, M.N. 1972. [*Scaling theory for finite-size effects in the critical region*](https://doi.org/10.1103/PhysRevLett.28.1516). Physical Review Letters 28, 1516-1519.
# 8. <span id="ref-onsager1944"></span>Onsager, L. 1944. [*Crystal statistics. I. A two-dimensional model with an order-disorder transition*](https://doi.org/10.1103/PhysRev.65.117). Physical Review 65, 117-149.
# 
# 
