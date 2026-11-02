#!/usr/bin/env python
# coding: utf-8

# # The execution interface: sampling and readouts
# 

# <div class="note"><span class="note-t">In brief</span>
# <p>We build a Torx circuit with two <a href="https://doi.org/10.1063/1.5055860">probabilistic bits (pbits)</a>, compute its exact readouts, and estimate the same quantities from raw samples. Ordinary array reductions produce histograms and expectations, while block estimates show both sampled readouts concentrating at the expected $1/\sqrt{N}$ rate.</p>
# </div>

# Executing a stochastic circuit returns bitstrings rather than a preselected statistic. How do we turn those bitstrings into a density or an expectation, and how can we tell whether the result is close to the quantity we intended to measure?
# 
# A readout is a function $f$ averaged against the circuit's output distribution. In this tutorial, the readouts reduce terminal bitstrings either to a [probability mass function](https://en.wikipedia.org/wiki/Probability_mass_function), which assigns a probability to each finite basis state, or to a mean. We then use Monte Carlo error bars to quantify the uncertainty of these sampled estimates.
# 
# The code uses Torx, JAX, NumPy, and Matplotlib. It assumes basic familiarity with probability distributions and expectations.
# 
# By the end, you'll be able to:
# 
# - build a two-pbit readout circuit,
# - compare an exact `StateVectorSimulator` density with reductions of `samples`,
# - plot histogram concentration toward the exact density, and
# - estimate the Monte Carlo rate $1/\sqrt{N}$ from a block analysis.
# 
# All of these sampled readouts begin with the same object: `BranchingSimulator.sample` returns the stochastic output as a `(num_samples, num_pbits)` integer array. Histograms, sample means, and confidence summaries are standard NumPy reductions over its rows.
# 
# We begin by building the circuit and computing an exact reference. We then sample the circuit, compare two ways to obtain an expectation, and measure how the sampling error changes with $N$.
# 
# 

# ## Setup
# 
# We import JAX, NumPy, and the Torx circuit, gate, and simulator classes.

# In[1]:


from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from torx.psc import (
    DiscretePCircuit,
    PCNOT,
    PNOT,
    BranchingSimulator,
    StateVectorSimulator,
)


# We put the repository helper directory on the import path before importing the notebook-local helpers.

# In[2]:


ROOT = Path.cwd()
if not (ROOT / "helpers").exists() and (ROOT.parent / "helpers").exists():
    ROOT = ROOT.parent
HELPER_DIR = ROOT / "helpers"
sys.path.insert(0, str(HELPER_DIR))


# Next we import the notebook-local helpers and apply the notebook plotting style.
# 
# <details class="tx-disclosure"><summary>What runs where?</summary><ul><li>Torx builds and executes the circuit.</li><li>Notebook code computes the readouts and diagnostics.</li><li>Plotting lives in <code>examples/helpers/_plots_sampling.py</code> and <code>examples/helpers/_plots_schematics.py</code>.</li></ul></details>
# 

# In[3]:


import _plots_sampling as P_samp
import _plots_schematics as P_sch
from jax.scipy.special import logit
from _notebook_paths import figure_dir
from _notebook_style import apply_notebook_style, make_savefig

FIGURE_DIR = figure_dir(ROOT)
SEED = 17

# Keep generated figures consistent with the rest of the gallery.
apply_notebook_style()
savefig = make_savefig(FIGURE_DIR)


# The histogram readout and the convergence panels later on need the same reduction, so we write it once: the next cell defines `empirical_two_pbit_density`, which reduces a batch of samples to a four-state density.

# In[4]:


def empirical_two_pbit_density(samples_2d):
    # Encode two bits as 00, 01, 10, 11 before counting frequencies.
    ids = samples_2d[:, 0] * 2 + samples_2d[:, 1]
    return np.bincount(ids, minlength=4) / len(samples_2d)


# ## The readout as an expectation
# 
# A Torx circuit defines a **stochastic kernel** $K$, a map from one probability distribution to another. Applied to an input distribution, it produces
# 
# $$\rho_{\text{out}} = K\,\rho_{\text{in}}.$$
# 
# We define a readout by choosing a function $f$ on the output configurations and averaging it against $\rho_{\text{out}}$:
# 
# $$\langle f, \rho \rangle = \sum_{a} f(a)\,\rho_a.$$
# 
# Here a ket such as $|a)$, or $|00)$ below, labels one basis configuration of the sites, and $\rho_a$ is its probability. Thus $\langle f, \rho \rangle$ is the expectation of $f$ under the distribution $\rho$.
# 
# There are two ways to obtain this expectation in the notebook. The `StateVectorSimulator` enumerates the output distribution and returns the exact $\rho_{\text{out}}$. In contrast, `BranchingSimulator.sample` draws $N$ bitstrings $s_n \sim \rho_{\text{out}}$, from which we estimate the expectation by
# 
# $$\widehat{\langle f \rangle}_N = \underbrace{\frac{1}{N} \sum_{n=1}^{N}}_{\vphantom{\big|}\text{sample average}} f(s_n).$$
# 
# The sum runs over the $N$ drawn bitstrings, with $f(s_n)$ evaluated on each one. If $f$ is the indicator of a basis state, the estimator gives the empirical density $\hat\rho_N$. If $f(s) = s_i$, it gives the per-pbit expectation returned by `expval_all`, which we introduce below.
# 
# Both estimators fluctuate because they use a finite sample. To compare an empirical density with the exact density using one number, we use the [**total-variation distance**](https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures): half the absolute probability gap, summed over the basis states. For the empirical and exact densities, this distance scales as $\mathrm{TV}(\hat\rho_N, \rho) = O(N^{-1/2})$. With two pbits, enumerating the exact $\rho$ is inexpensive, so every sampled estimate can be compared with this reference.

# ## The smallest readout
# 
# A two-pbit circuit is large enough to show all four basis states and small enough to enumerate exactly.
# 
# The circuit contains a `PNOT` on pbit 0 with flip probability 0.38, a `PCNOT` with conditional flip probability 0.70 from pbit 0 to pbit 1, and a `PNOT` on pbit 1 with flip probability 0.25. Those probabilities spread the distribution across all four basis states $\{|00), |01), |10), |11)\}$, so the histogram has something to show.
# 
# We convert the probabilities to logits (log-odds, the log of a probability against its complement) before constructing the Torx gates.

# In[5]:


circuit = DiscretePCircuit(
    [
        PNOT(0),
        PCNOT([0, 1]),
        PNOT(1),
    ]
)

# Parameters are kept separate from the circuit structure. The thetas list is
# aligned with `circuit.gates`; each per-gate theta has shape (1,).
thetas = [
    jnp.array([logit(0.38)]),
    jnp.array([logit(0.70)]),
    jnp.array([logit(0.25)]),
]


# In[6]:


fig = P_sch.draw_pcircuit(
    circuit,
    wire_labels=[r"$p_0$", r"$p_1$"],
    title="Two-pbit readout circuit",
)
savefig(fig, "04_readout_circuit")


# The diagram encodes the execution order of the three gates along the two pbit wires. Slate marks the single-pbit `PNOT` gates, while orange marks the two-pbit `PCNOT` coupling gate. It shows the circuit structure, but not the probability assigned to each output state; we compute that distribution next.

# The `StateVectorSimulator` provides a reference distribution for the sampling estimates.

# In[7]:


# Both inputs encode the |00) start. Derive the one-hot state-vector input from
# the bit pattern so the two simulator encodings cannot drift apart.
initial_bits = jnp.array([0, 0], dtype=jnp.int32)
start_index = int(initial_bits[0] * 2 + initial_bits[1])
initial_distribution = jnp.zeros(4).at[start_index].set(1.0)

exact_sim = StateVectorSimulator()
exact_compiled = exact_sim.build_circuit(circuit, thetas)
exact_density = np.asarray(exact_sim.density(exact_compiled, initial_distribution))
exact_expval = np.asarray(exact_sim.expval_all(exact_compiled, initial_distribution))

states = ["00", "01", "10", "11"]
print("exact density   p(s):", dict(zip(states, np.round(exact_density, 3).tolist())))
print("exact expval     <s>:", np.round(exact_expval, 3).tolist())


# The printed values give `exact_density` over basis states and the exact per-pbit expectation `exact_expval`. We score every sampled estimator below against these ground-truth references.

# The sampling interface returns the underlying observations rather than a finished statistic. One call to `BranchingSimulator.sample` draws 5,000 terminal bitstrings and stores them in a `(num_samples, num_pbits)` integer array, leaving the choice of readout to the subsequent reduction.

# In[8]:


NUM_SAMPLES = 5_000

sim = BranchingSimulator(num_samples=NUM_SAMPLES)
sample_compiled = sim.build_circuit(circuit, thetas)
# Use a fixed PRNG key so the sampled readout is reproducible.
samples = np.asarray(sim.sample(sample_compiled, initial_bits, jax.random.key(SEED)))

assert samples.shape == (NUM_SAMPLES, 2)

print(f"samples shape: {samples.shape}  (num_samples, num_pbits)")
print(f"first five:    {samples[:5].tolist()}")


# The shape confirms that each row of `samples` is one sampled bitstring and each column is one pbit.

# ## Histogram and expectation
# 
# We now compute two readouts from the same sample array.
# 
# For the density, we encode each row as a basis-state index, count the indices with `np.bincount`, and divide by $N$. For the per-pbit expectation, averaging each column with `samples.mean(axis=0)` directly estimates $\langle s_i \rangle$.
# 
# Torx also provides the latter readout through `BranchingSimulator.expval_all`. This method draws a fresh sample internally and returns its mean, so its result need not match `samples.mean(axis=0)` exactly. At $N = 5{,}000$, both sampled estimators agree with the `StateVectorSimulator` reference within the `atol = 0.04` tolerance asserted in the next cell.

# In[9]:


# Histogram readout: reuse empirical_two_pbit_density to encode and count states.
sample_probs = empirical_two_pbit_density(samples)

sample_mean = samples.mean(axis=0)
# SEED + 1 is just a distinct PRNG key, so expval_all draws an independent
# sample rather than reusing the one above.
api_expval = np.asarray(
    sim.expval_all(sample_compiled, initial_bits, jax.random.key(SEED + 1))
)

print(
    f"sample probs    p_hat(s): {dict(zip(states, np.round(sample_probs, 3).tolist()))}"
)
print(f"sample mean   samples.mean: {np.round(sample_mean, 3).tolist()}")
print(f"expval_all (independent draw): {np.round(api_expval, 3).tolist()}")


# In[10]:


np.testing.assert_allclose(sample_probs.sum(), 1.0, atol=1e-6)
np.testing.assert_allclose(sample_probs, exact_density, atol=0.04)
np.testing.assert_allclose(sample_mean, exact_expval, atol=0.04)
np.testing.assert_allclose(api_expval, exact_expval, atol=0.04)

fig = P_samp.readout_histogram_expectation(
    sample_probs,
    exact_density,
    exact_expval,
    sample_mean,
    api_expval,
)
savefig(fig, "04_readout_histogram_expectation")


# The empirical-density bars can be compared state by state with `exact_density`. In the expectation panel, the `expval_all` cross lies near both the sample-mean and exact-expectation bars, although it comes from the separate `SEED + 1` draw. Thus `sample_mean` and `expval_all` provide two sampled estimates of the same expectation in this run. The figure shows their agreement at $N = 5{,}000$; by itself, it does not show how either error changes with sample size.
# 

# ## Histogram concentration
# 
# To examine the dependence on sample size, we redraw the empirical histogram for $N \in \{100, 500, 2000, 5000\}$. Across the panels, the exact density remains fixed while the sampled bars change with $N$.
# 
# Each panel also reports the total-variation distance to `exact_density`, averaged over 16 non-overlapping blocks:
# 
# $$\overline{\mathrm{TV}}(\hat\rho, \rho) = \underbrace{\frac{1}{16} \sum_{k=1}^{16}}_{\vphantom{\big|}\text{block average}} \underbrace{\tfrac{1}{2} \sum_{s} |\hat\rho_k(s) - \rho(s)|}_{\vphantom{\big|}\text{TV per block}}.$$
# 
# The inner sum compares the empirical density $\hat\rho_k$ from one block with the exact density $\rho$. The outer average combines the 16 block distances, reducing the influence of any one noisy block at small $N$. We use the resulting values to examine the expected $1/\sqrt{N}$ decrease.

# In[11]:


PANEL_NS = [100, 500, 2_000, 5_000]
TV_REPLICATES = 16
LARGEST_N = max(PANEL_NS)


# In[12]:


conv_sim = BranchingSimulator(num_samples=LARGEST_N * TV_REPLICATES)
conv_compiled = conv_sim.build_circuit(circuit, thetas)
# Any key distinct from the draws above gives an independent convergence run.
conv_samples = np.asarray(
    conv_sim.sample(conv_compiled, initial_bits, jax.random.key(SEED + 100))
)


# For each $N$, we use a prefix of the run for the displayed empirical density and 16 equal-size blocks for the mean total-variation distance. Because every sample size is derived from the same run, estimates at different $N$ share observations and are correlated. The block analysis therefore describes concentration in this run; it does not provide independent experiments at each sample size.

# In[13]:


np.testing.assert_array_equal(conv_samples.shape, (LARGEST_N * TV_REPLICATES, 2))

panel_probs = []
tv_means = []
for n in PANEL_NS:
    # Prefixes give the histogram panels; equal-size blocks estimate TV variation.
    panel_probs.append(empirical_two_pbit_density(conv_samples[:n]))
    blocks = conv_samples[: TV_REPLICATES * n].reshape(TV_REPLICATES, n, 2)
    block_tvs = [
        0.5 * np.abs(empirical_two_pbit_density(b) - exact_density).sum()
        for b in blocks
    ]
    tv_means.append(float(np.mean(block_tvs)))



# In[14]:


tv_means = np.asarray(tv_means)
# One finite run, so check the trend rather than pinning the exact rate:
# TV shrinks with N and the log-log slope sits in a band around -1/2.
tv_slope = float(np.polyfit(np.log(PANEL_NS), np.log(tv_means), 1)[0])
assert tv_means[-1] < tv_means[0]
assert -0.85 < tv_slope < -0.2
print(f"TV log-log slope: {tv_slope:.3f}  (illustrates -0.5)")

fig = P_samp.histogram_convergence(
    PANEL_NS, panel_probs, exact_density, tv_means, TV_REPLICATES
)
savefig(fig, "04_histogram_convergence")


# From left to right, the empirical bars approach `exact_density`. The annotation in each panel is the mean total-variation distance from 16 blocks, not the distance between the particular bars shown in that panel and the exact density. These block means decrease with $N$, and their printed log-log least-squares slope is near $-1/2$, consistent with the $1/\sqrt{N}$ Monte Carlo rate. Since the four sample sizes share one finite run, the figure illustrates this scaling rather than establishing it from independent datasets.
# 

# ## Sample-mean concentration
# 
# We next ask whether the expectation readout exhibits the same dependence on sample size. A single sample mean provides one error value, but not the distribution of that error. We therefore divide one long run into blocks and compute the pbit-1 mean separately in each block:
# 
# 1. draw one long $32{,}000$-sample run,
# 2. split it into non-overlapping blocks of size $N \in \{25, 50, 100, 200, 500, 1000\}$,
# 3. average pbit 1 within each block to form one estimator of $\langle s_1 \rangle$.
# 
# For any fixed $N$, the blocks are disjoint. Across different values of $N$, however, the construction reuses prefixes of the same long run, so those collections of estimators are correlated. This is sufficient for describing the slope of the curve, but it does not make the points independent tests of the rate.
# 
# Using 32 blocks at each $N$, we report the mean absolute error relative to the exact pbit-1 expectation. The shaded band is the **Monte Carlo standard error** of that mean across the 32 blocks: the typical variation of the 32-block average under a new run. On the log-log plot, the [central limit theorem (CLT)](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3661.htm) supplies the $N^{-1/2}$ reference slope.

# In[15]:


BLOCK_SAMPLES = 32_000
NS = np.array([25, 50, 100, 200, 500, 1000])
REPLICATES = 32


# The next cell draws the long `big_samples` run used for every block size.

# In[16]:


big_sim = BranchingSimulator(num_samples=BLOCK_SAMPLES)
big_compiled = big_sim.build_circuit(circuit, thetas)
# SEED + 2 is another distinct key, giving the long run its own draw.
big_samples = np.asarray(
    big_sim.sample(big_compiled, initial_bits, jax.random.key(SEED + 2))
)


# In[17]:


readout = big_samples[:, 1]

block_means = np.stack(
    [readout[: REPLICATES * n].reshape(REPLICATES, n).mean(axis=1) for n in NS]
)
# Compare each block estimate with the exact pbit-1 expectation.
errors = np.abs(block_means - exact_expval[1])
err_mean = errors.mean(axis=1)
err_std = errors.std(axis=1, ddof=1)



# In[18]:


# One finite run with correlated prefixes, so check the trend, not the exact
# rate: the error shrinks with N and the log-log slope sits around -1/2.
err_slope = float(np.polyfit(np.log(NS), np.log(err_mean), 1)[0])
assert err_mean[-1] < err_mean[0]
assert -0.85 < err_slope < -0.2
print(f"sample-mean error log-log slope: {err_slope:.3f}  (illustrates -0.5)")

fig = P_samp.sample_mean_error(NS, err_mean, err_std, REPLICATES)
savefig(fig, "04_readout_clt_error")


# The pbit-1 error curve lies close to the $N^{-1/2}$ reference, and the printed least-squares slope is near $-1/2$, consistent with the Monte Carlo rate. The reference line is anchored at the smallest-$N$ point, so agreement there holds by construction; the remaining points provide the visual comparison. Because they reuse prefixes from one finite run, the curve and fitted slope are descriptive diagnostics rather than independent estimates at each $N$.

# ## Conclusion
# 
# The exact calculation and the sampled calculation answer different questions. `StateVectorSimulator` enumerates the output distribution, while `BranchingSimulator.sample` provides finite observations from which we estimate readouts. For the two-pbit circuit, their comparison gives the following results:
# 
# - `BranchingSimulator.sample` returns a `(num_samples, num_pbits)` integer array containing the full stochastic output.
# - Applying `np.bincount` to the sampled bitstrings approximates the exact `StateVectorSimulator` density within sampling error (`atol = 0.04` at $N = 5000$).
# - `samples.mean(axis=0)` and the independent `expval_all` draw estimate the same expectation, and both agree with the exact value within the same tolerance.
# - In the block analyses, the histogram and pbit-1 expectation errors decrease consistently with $1/\sqrt{N}$, with log-log slopes near the $N^{-1/2}$ reference. The shared finite runs make these figures diagnostics of concentration rather than independent measurements at each $N$.
# - These terminal-bitstring readouts are ordinary NumPy reductions on `samples`. Differentiable or `jit`-transformable readouts instead use `jnp` or the Torx expectation APIs.
# 
# See also:
# 
# - [`05_chemical_reaction_networks.ipynb`](05_chemical_reaction_networks.ipynb), which uses `BranchingSimulator` to recover a full trajectory against an exact reference.
