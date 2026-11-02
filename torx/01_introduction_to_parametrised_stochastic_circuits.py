#!/usr/bin/env python
# coding: utf-8

# # Introduction to parametrised stochastic circuits
# 

# <div class="note"><span class="note-t">In brief</span>
# <p>We introduce parametrised stochastic circuits through the three Torx data primitives: <a href="#Pbit-gates">pbits</a>, <a href="#Pdit-gates">pdits</a>, and <a href="#Pmode-gates">pmodes</a>. For each primitive, we connect the gate constructor to its mathematical action—a transition kernel or a Gaussian map. We then inspect gates in two complementary ways: sampling a composed circuit shows the distribution it produces, while evaluating a transition matrix exposes the kernel directly.</p>
# </div>

# Probability models often combine binary, finite-state, and continuous variables. How can a single programming model transform distributions over all three? Torx addresses this question with parametrised stochastic circuits (PSCs): JAX programs whose gates reshape probability distributions. This notebook introduces the gates and sampling workflow used throughout the example gallery. It assumes basic familiarity with probability distributions and JAX, and uses Torx, JAX, NumPy, and Matplotlib.
# 
# By the end, you'll be able to:
# 
# - build a PSC from gates,
# - read each gate as a transition kernel, and
# - sample pbit and pmode circuits and read the pdit kernel from its transition matrix.
# 
# Torx provides three data primitives:
# 
# - pbits, binary sites, $\{0,1\}$,
# - pdits, $d$-state sites, $\{0,\dots,d-1\}$, and
# - pmodes, continuous sites, $\mathbb{R}^N$.
# 
# We begin with pbits, whose small transition matrices make every branch explicit. We then extend the same kernel view to a cyclic pdit and finish with continuous Gaussian maps:
# 
# 1. pbit gates `PSWAP`, `PNOT`, and `PISING`, reading each transition matrix and sampling the branch gates,
# 2. the pdit gate that takes a stay/forward/backward random walk on its $d$ states, and
# 3. pmode gates `AffineGaussianGate` and `MixtureGaussianGate`.
# 
# 

# ## Setup
# 
# Nothing in this section is specific to stochastic circuits. We point Python at the shared helper modules, apply the plotting style the whole gallery uses, and create the `savefig` utility so every figure below is written to the same output directory.
# 
# <details class="tx-disclosure"><summary>What runs where?</summary><ul><li>Torx gates and simulators produce transition matrices and samples.</li><li>Notebook code assembles circuits and computes the NumPy analytic mixture contour and checks.</li><li><code>examples/helpers/_plots_fields.py</code> and <code>examples/helpers/_plots_schematics.py</code> render figures.</li><li><code>examples/helpers/_notebook_style.py</code> and <code>examples/helpers/_notebook_paths.py</code> apply styling and choose output paths.</li></ul></details>

# In[1]:


from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logit

# Resolve examples/helpers whether run from repo root or examples.
ROOT = Path.cwd()
if (ROOT / "examples" / "helpers").exists():
    ROOT = ROOT / "examples"
elif not (ROOT / "helpers").exists() and (ROOT.parent / "helpers").exists():
    ROOT = ROOT.parent
HELPER_DIR = ROOT / "helpers"
sys.path.insert(0, str(HELPER_DIR))


# In[2]:


from _notebook_paths import figure_dir
from _notebook_style import apply_notebook_style, make_savefig
import _plots_fields as P_fld
import _plots_schematics as P_sch

apply_notebook_style()
FIGURE_DIR = figure_dir(ROOT)
savefig = make_savefig(FIGURE_DIR)


# ## Pbit gates
# 
# We start with pbits because their gates are small enough to write out in full as a matrix, which means we can check every sampled result against something we derived by hand.
# 
# A **pbit** (probabilistic bit) is a binary stochastic site: at any instant it's 0 or 1 with some probability. A joint configuration is written as a ket, $|ab)$ with $a,b\in\{0,1\}$, a label for one of the $2^2=4$ basis configurations of two pbits ($n$ pbits have $2^n$).
# 
# A gate over pbits is a column-stochastic kernel $K(y\mid x)$ on these configurations, with $\sum_y K(y\mid x)=1$. Read it by column: column $x$ holds the distribution over outputs $y$ given the input $|x)$.
# 
# The branch-table pbit gates [`PSWAP`](https://docs.torx.ai/en/latest/api-gates.html#PSWAP) and [`PNOT`](https://docs.torx.ai/en/latest/api-gates.html#PNOT) have two branches: leave the pbit unchanged, or apply a deterministic operation $B$. The operation branch is selected with probability $p$:
# 
# $$G(\theta)=\underbrace{(1-\sigma(\theta))\,I}_{\vphantom{\big|}\text{stay branch}}+\underbrace{\sigma(\theta)\,B}_{\vphantom{\big|}\text{op branch}},\qquad p=\sigma(\theta)=\frac{1}{1+e^{-\theta}}.$$
# 
# What Torx actually stores is the logit $\theta=\log\frac{p}{1-p}$ rather than $p$ itself, because $\theta$ ranges over all of $\mathbb{R}$ and therefore works cleanly with gradients, and applying the sigmoid recovers the physical switching probability whenever we need it.
# 
# `PISING`, introduced below, is the energy-based exception: its parameter vector defines an Ising bond update rather than a single switching logit.
# 
# Throughout this section we fix the switching probability at $p=0.30$, so the sampled bars from different gates are directly comparable.

# We import the circuit and simulator classes together with the two binary gates, then fix the values every pbit demo shares: the switching logit, the sample count, and one random number generator key per sampled figure, so re-running a single cell reproduces its own plot rather than shifting the ones after it.

# In[3]:


from torx.psc import DiscretePCircuit, BranchingSimulator
from torx.psc import PSWAP, PNOT

# Keep one probability across the simple gate demos so their plots are comparable.
P_DEMO = 0.30
DEMO_SAMPLES = 20000
# Gates are structure only; their parameters live in a separate `thetas` list,
# one entry per gate aligned with `circuit.gates`. The two-branch gates here
# take a length-1 logit vector, matching `circuit.init_params`.
THETA_DEMO = jnp.array([logit(P_DEMO)])
# explicit per-draw keys keep each sampling cell idempotent under re-runs
DEMO_KEYS = jax.random.split(jax.random.key(11), 4)


# Sampling looks the same for every gate below, so we set it up once. We create the shared `BranchingSimulator` and define `sample_distribution`, which takes a compiled circuit, calls `sample`, and returns the empirical probability of each output state. Each gate demo compiles its circuit once and reuses that compiled object across the inputs it feeds in.

# In[4]:


demo_sim = BranchingSimulator(num_samples=DEMO_SAMPLES)


def sample_distribution(compiled, initial, num_states, key):
    """Run the compiled circuit on `initial`, return the empirical distribution."""
    samples = np.asarray(
        demo_sim.sample(compiled, jnp.asarray(initial, dtype=jnp.int32), key)
    )
    states = np.ravel_multi_index(samples.T, (num_states,) * samples.shape[1])
    return np.bincount(states, minlength=num_states ** samples.shape[1]) / len(states)


# ### PSWAP
# 
# `PSWAP` is the smallest gate that couples two pbits, which makes it a good place to establish the pattern the rest of the gallery follows: write the kernel down, build a one-gate circuit, then sample it.
# 
# With probability $p$, `PSWAP` swaps two pbits, and otherwise it leaves them alone. Only $|01)$ and $|10)$ change, and $|00)$ and $|11)$ are fixed points:
# 
# $$\mathsf{PSWAP}(p)\,|ab)=(1-p)\,|ab)+p\,|ba).$$
# 
# In the basis $(|00), |01), |10), |11))$ that is the column-stochastic matrix:
# 
# $$\mathsf{PSWAP}(p)=\begin{pmatrix}1 & 0 & 0 & 0\\ 0 & 1-p & p & 0\\ 0 & p & 1-p & 0\\ 0 & 0 & 0 & 1\end{pmatrix}.$$
# 
# Below we build the one-gate circuit and compile it once for both inputs, then draw it so the wiring is explicit: a single `PSWAP` gate acting on two pbits.

# In[5]:


circuit = DiscretePCircuit([PSWAP([0, 1])])
# One parameter vector per gate, in gate order.
thetas = [THETA_DEMO]
# compile once and reuse across both input states
pswap_compiled = demo_sim.build_circuit(circuit, thetas)


# In[6]:


fig = P_sch.draw_pcircuit(
    circuit, wire_labels=[r"$p_0$", r"$p_1$"], title="PSWAP on two pbits"
)
savefig(fig, "01_circuit_pswap")


# The matrix says that only $|01)$ and $|10)$ can move, so those are the two inputs worth sampling. We draw 20,000 samples from each of them and plot the empirical distribution, where the stay and swap branches should appear as bars at $1-p$ and $p$.

# In[7]:


labels = ["00", "01", "10", "11"]
# Only |01) and |10) can move under PSWAP; the other basis states are fixed.
dists = [
    (
        "input |01)",
        labels,
        sample_distribution(pswap_compiled, [0, 1], num_states=2, key=DEMO_KEYS[0]),
    ),
    (
        "input |10)",
        labels,
        sample_distribution(pswap_compiled, [1, 0], num_states=2, key=DEMO_KEYS[1]),
    ),
]


# In[8]:


fig = P_sch.transition_bars(dists, gate="PSWAP", p=P_DEMO)
savefig(fig, "01_gate_pswap")


# ### PNOT
# 
# `PNOT` is the single-site version of the same branch construction, so its kernel is the smallest one in the notebook and takes only two columns to read.
# 
# With probability $p$, `PNOT` flips one pbit, and otherwise the bit is unchanged. Its kernel is the convex combination (a probability-weighted blend) of $I$ and the NOT operation:
# 
# $$\mathsf{PNOT}(p)=\begin{pmatrix}1-p & p\\ p & 1-p\end{pmatrix}.$$
# 
# A single pbit has only two possible inputs, so we sample both below and compare the flipped and unchanged outcomes against the two columns above.

# In[9]:


circuit = DiscretePCircuit([PNOT(0)])
thetas = [THETA_DEMO]
pnot_compiled = demo_sim.build_circuit(circuit, thetas)
labels = ["0", "1"]
dists = [
    (
        "input |0)",
        labels,
        sample_distribution(pnot_compiled, [0], num_states=2, key=DEMO_KEYS[2]),
    ),
    (
        "input |1)",
        labels,
        sample_distribution(pnot_compiled, [1], num_states=2, key=DEMO_KEYS[3]),
    ),
]


# In[10]:


fig = P_sch.transition_bars(dists, gate="PNOT", p=P_DEMO)
savefig(fig, "01_gate_pnot")


# ### PISING
# 
# The preceding gates choose between the identity and one fixed operation. `PISING` instead derives its kernel from an energy model, so its finite-time transition matrix can include paths with more than one flip.
# 
# The [`PISING`](https://docs.torx.ai/en/latest/api-gates.html#PISING) gate is a finite-time kernel for [Glauber dynamics](https://doi.org/10.1063/1.1703954) on an Ising bond. It integrates a single-spin-flip generator for a time $\Delta t$; exponentiating that generator sums the possible jump sequences over the interval.
# 
# <details class="tx-disclosure"><summary>Term: Glauber dynamics</summary><p>The generator $Q$ gives instantaneous rates for one-spin jumps. Its matrix exponential sums every allowed jump sequence over $\Delta t$, so a finite-time transition can contain more than one flip.</p></details>
# 
# Let $s_i=2b_i-1$ for pbit values $b_i\in\{0,1\}$. The bond energy and the single-flip generator are shown below. [Hamming distance](https://doi.org/10.1002/j.1538-7305.1950.tb00463.x) counts the pbits on which two configurations differ.
# 
# $$E(\mathbf{s})=-J\,s_1 s_2-h_1 s_1-h_2 s_2,\qquad
# Q_{ab}=\mathbb{1}[\mathrm{Hamming}(a,b)=1]\;\sigma\left[-\beta\big(E_a-E_b\big)\right]\ (a\neq b),\qquad
# Q_{bb}=-\!\!\sum_{a\neq b}Q_{ab},$$
# 
# Only single-spin-flip neighbors receive off-diagonal rates, and the diagonal term makes each column sum to zero. Here $J$ is the bond coupling, while $h_1$ and $h_2$ are local fields. The inverse temperature $\beta$ scales the energy change of a proposed flip, so increasing $\beta$ produces a colder, more selective update.
# 
# The finite-time gate is the matrix exponential $\mathsf{PISING}(\theta)=\exp(\Delta t\,Q)$. Unlike the branch-table gates above, it is a $4\times 4$ column-stochastic kernel. Its parameter vector $\theta=[J,h_1,h_2,\beta,\Delta t]$ contains the physical bond parameters.
# 
# We evaluate this kernel directly with `get_matrix` rather than estimating it from samples. The heatmap below shows the transition matrix for the selected bond parameters. In particular, finite-time multi-flip entries are positive, although at $\Delta t=0.35$ they are small enough to display as 0.00 after rounding.

# In[11]:


from torx.psc import PISING


# In[12]:


# Parameters are [J, h1, h2, beta, dt] for this two-site Ising update.
gate = PISING([0, 1])
theta = jnp.array([1.0, 0.0, 0.0, 1.5, 0.35])
M = np.asarray(gate.get_matrix(theta))


# In[13]:


fig = P_sch.ising_matrix(M, J=1.0, beta=1.5, dt=0.35)
savefig(fig, "01_gate_pising")


# ## Pdit gates
# 
# The same column-stochastic construction applies when a site has more than two states.
# 
# A **pdit** is a discrete stochastic site with $d$ states, generalizing a pbit beyond $\{0,1\}$. We use [`PditCycle`](https://docs.torx.ai/en/latest/api-gates.html#PditCycle), which defines a random walk on a cyclic $d$-state pdit with stay, forward, and backward branches. Because all three branches appear in each input column, we inspect the transition matrix directly, as we did for `PISING`. The heatmap below uses $d=3$.
# 
# <details class="tx-disclosure"><summary>Term: three-way Pdit softmax</summary><p><code>PditCycle</code> applies softmax to $[0,\theta_0,\theta_1]$ for the stay, forward, and backward probabilities. Thus $\theta=\log([0.30,0.20]/0.50)$ reproduces $[0.50,0.30,0.20]$.</p></details>
# 
# Other pdit permutation gates include `PditShift` and `PditSWAP`. Later, [notebook 13](13_regime_switching_diffusion.ipynb) uses `PditCycle` to drive a regime chain.

# In[14]:


from torx.psc import PditCycle


# In[15]:


DIMS = 3
cycle_theta = jnp.log(
    jnp.array([0.30 / 0.50, 0.20 / 0.50])
)  # stay=0.50, forward=0.30, backward=0.20
matrices = [
    ("PditCycle", np.asarray(PditCycle(sites=0, dims=DIMS).get_matrix(cycle_theta))),
]


# In[16]:


fig = P_sch.pdit_matrices(matrices, dims=DIMS)
savefig(fig, "01_gate_pdit")


# ## Pmode gates
# 
# A continuous site requires a different representation: when the state space is $\mathbb{R}^N$, there is no finite transition matrix to enumerate.
# 
# A **pmode** is a continuous stochastic site valued in $\mathbb{R}^N$. The [`AffineGaussianGate`](https://docs.torx.ai/en/latest/api-gates.html#AffineGaussianGate) applies a linear map, adds a bias, and then adds diagonal Gaussian noise:
# 
# $$X \mapsto \underbrace{A X}_{\vphantom{\big|}\text{linear map}} + \mathbf{b} + \varepsilon,\qquad \varepsilon \sim \mathcal{N}(0, \Delta),$$
# 
# where $A$ is the linear map, $\mathbf{b}$ is the bias shift, and $\varepsilon$ is the diagonal Gaussian noise. Gaussian inputs remain Gaussian: $\mathcal{N}(\mu,\Sigma)\mapsto\mathcal{N}(A\mu+\mathbf{b},\,A\Sigma A^\top+\Delta)$.
# 
# We first use `AffineGaussianGate` directly. We then introduce [`MixtureGaussianGate`](https://docs.torx.ai/en/latest/api-gates.html#MixtureGaussianGate), which uses a discrete control to select one of several diagonal Gaussian components, each defined by a mean shift and diagonal noise.
# 
# The affine form includes the common special cases of shift, scale, rotation, and diffusion.

# ### AffineGaussianGate
# 
# One gate is enough to see what a pmode gate does to a distribution, because each of its three parts moves a cloud of points in a visibly different way. We apply a single `AffineGaussianGate` to an $\mathcal{N}(0, I)$ input cloud and plot the input and the output on the same axes, where the linear map stretches and tilts, the bias shifts, and the diagonal noise sets the spread. The next cell defines `sample_continuous`, which compiles a circuit and samples its cloud starting from the origin.
# 
# See [`10_pmode_gaussian_gates.ipynb`](10_pmode_gaussian_gates.ipynb) for the full pmode-gate tour: the specialized `Displace`, `Scale`, `Mix`, and `Diffuse` gates, the exact moment and composition laws, closed-form conditioning, and the analytic-versus-sampled checks.

# In[17]:


from torx.psc import (
    AffineGaussianGate,
    MixtureGaussianGate,
    HybridPCircuit,
)

# This identity-plus-unit-noise AffineGaussianGate is used as an N(0, I) source.
PMODE_SAMPLES = 1500


def sample_continuous(circuit, thetas, key, num_continuous=2):
    """Compile `circuit` with `thetas`, run it from the origin, return the cloud."""
    origin = {
        "discrete": jnp.zeros(0, dtype=jnp.int32),
        "continuous": jnp.zeros(num_continuous),
    }
    out = circuit.sample_multiple(key, origin, thetas, n_samples=PMODE_SAMPLES)
    return np.asarray(out["continuous"])


# In[18]:


# A simple affine map: a linear stretch-and-tilt, a bias shift, and a little
# diagonal Gaussian noise. The point here is only that a pmode gate reshapes a
# distribution; the detailed gate tour lives in `10_pmode_gaussian_gates.ipynb`.
A_demo = np.array([[1.3, 0.5], [0.0, 0.7]], dtype=np.float32)
b_demo = np.array([0.6, -0.4], dtype=np.float32)
log_var_demo = np.array([np.log(0.05), np.log(0.05)], dtype=np.float32)


# In[19]:


affine_gate = AffineGaussianGate(
    sites=[0, 1],
    dims=(1, 1),
)
# Parameters are separate from the gate structure.
affine_params = {
    "A": jnp.asarray(A_demo),
    "b": jnp.asarray(b_demo),
    "log_var": jnp.asarray(log_var_demo),
}

# A second affine-Gaussian gate at its default identity params (A = I, b = 0,
# log_var = 0) adds unit Gaussian noise, so it is used as an N(0, I) source.
# The input and output clouds therefore both come out of Torx circuits.
source_gate = AffineGaussianGate(sites=[0, 1], dims=(1, 1))
source_params = source_gate.init_params(jax.random.key(0))

input_circuit = HybridPCircuit([source_gate])
affine_circuit = HybridPCircuit([source_gate, affine_gate])


# In[20]:


# The source circuit emits N(0, I); the source + affine circuit transforms it.
in_cloud = sample_continuous(input_circuit, [source_params], jax.random.key(1000))
out_cloud = sample_continuous(
    affine_circuit, [source_params, affine_params], jax.random.key(1001)
)


# In[21]:


fig = P_fld.affine_gaussian_clouds(in_cloud, out_cloud)
savefig(fig, "01_gate_affine_gaussian")


# The output cloud is stretched, tilted, and shifted relative to the input, as predicted by the affine map and its diagonal noise. The figure does not pair individual input and output points: the input cloud comes from a separate $\mathcal{N}(0, I)$ source circuit with its own key, so only the two distributions—not pointwise trajectories—can be compared.

# ### MixtureGaussianGate
# 
# We now combine a discrete control with a continuous state.
# 
# `MixtureGaussianGate` uses the control pdit value to choose a diagonal Gaussian component. The gate adds the selected component mean and diagonal Gaussian noise to the input continuous state. Starting the pmode at the origin and sampling the control with known weights $\pi$ produces the marginal mixture
# 
# $$p_{X'}(x)=\sum_{k=0}^{K-1}\pi_k\,\mathcal{N}(x;\,\mu_k,\,\Sigma_k),$$
# 
# where $\pi_k$ is the probability that branch $k$ fires, $\mu_k$ is its mean, and $\Sigma_k=\mathrm{diag}(\sigma_k^2)$ is its diagonal covariance.
# 
# Because $\pi_k$, $\mu_k$, and $\Sigma_k$ are specified parameters, they also provide direct numerical checks. For each branch, we compare the sample mean with $\mu_k$ using a tolerance that decreases as $1/\sqrt{n_k}$, where $n_k$ is the number of samples assigned to that branch. We then compare the empirical branch frequencies with $\pi$ using the tolerance $3/\sqrt{N}$. The first check tests the component means; the second tests whether the control selects components with the stated weights.
# 
# In `sites=(0, 0)`, the first `0` addresses discrete control site 0, and the second addresses continuous site 0 in a separate namespace. Torx generates both the continuous samples and their control labels. The notebook's NumPy code evaluates the analytic mixture density and performs the assertions against those samples.

# In[22]:


def sample_mixture(circuit, thetas, key):
    """Sample the marginal mixture by drawing the control inside the circuit."""
    initial = {
        "discrete": jnp.array([0], dtype=jnp.int32),
        "continuous": jnp.zeros(2),
    }
    out = circuit.sample_multiple(key, initial, thetas, n_samples=MIXTURE_SAMPLES)
    samples = np.asarray(out["continuous"])
    labels = np.asarray(out["discrete"])[:, 0]
    return samples, labels


def mixture_density_grid(means, sigmas, weights, samples, pad=0.45, n=120):
    """Evaluate the analytic diagonal-Gaussian mixture on a 2D grid."""
    lo = samples.min(axis=0) - pad
    hi = samples.max(axis=0) + pad
    x0 = np.linspace(lo[0], hi[0], n)
    x1 = np.linspace(lo[1], hi[1], n)
    xx, yy = np.meshgrid(x0, x1)
    pts = np.stack([xx, yy], axis=-1)
    density = np.zeros(xx.shape)
    for weight, mean, sigma in zip(weights, means, sigmas):
        z = (pts - np.asarray(mean)) / np.asarray(sigma)
        norm = 1.0 / (2.0 * np.pi * np.prod(sigma))
        density += float(weight) * norm * np.exp(-0.5 * np.sum(z**2, axis=-1))
    return xx, yy, density


# In[23]:


K = 3
mix_means = jnp.array([[-1.2, 0.0], [0.4, 0.8], [1.0, -0.6]], dtype=jnp.float32)
mix_sigmas = np.array([[0.18, 0.18], [0.20, 0.15], [0.16, 0.22]], dtype=np.float32)
mix_probs = jnp.array([0.50, 0.30, 0.20], dtype=jnp.float32)


# In[24]:


mixture_gate = MixtureGaussianGate(
    sites=(0, 0),
    dims=(2,),
    num_components=K,
)
mixture_params = {
    "means": mix_means,
    "log_vars": jnp.log(jnp.asarray(mix_sigmas**2)),
}
# Starting at control 0, PditCycle maps stay/forward/backward to components 0/1/2.
control_gate = PditCycle(sites=0, dims=K)
control_theta = jnp.log(mix_probs[1:] / mix_probs[0])

mixture_circuit = HybridPCircuit([control_gate, mixture_gate])
MIXTURE_SAMPLES = 800


# In[25]:


samples, labels = sample_mixture(
    mixture_circuit, [control_theta, mixture_params], jax.random.key(99)
)
branch_counts = np.bincount(labels, minlength=K)
assert branch_counts.min() > 0
for k in range(K):
    # finite-sample
    tol = 5.0 * mix_sigmas[k].max() / np.sqrt(branch_counts[k])
    err = np.max(np.abs(samples[labels == k].mean(0) - np.asarray(mix_means[k])))
    assert err < tol, (k, err, tol)
empirical_weights = branch_counts / branch_counts.sum()
assert np.max(np.abs(empirical_weights - np.asarray(mix_probs))) < 3.0 / np.sqrt(
    MIXTURE_SAMPLES
)


# In[26]:


fig = P_fld.mixture_clouds(samples, labels)
xx, yy, zz = mixture_density_grid(
    mix_means, mix_sigmas, mix_probs, samples
)
fig.axes[0].contour(
    xx, yy, zz, levels=5, colors="0.20", linewidths=0.8, alpha=0.75
)
count_text = ", ".join(str(int(count)) for count in branch_counts)
weight_text = ", ".join(f"{float(weight):.2f}" for weight in mix_probs)
fig.axes[0].set_title(
    f"Torx samples by control (N = {MIXTURE_SAMPLES:,}, counts = {count_text})\n"
    f"NumPy analytic density contours, weights = {weight_text}"
)
savefig(fig, "01_gate_mixture_gaussian")


# The plot shows one sample from the marginal mixture, with color encoding the branch label produced by the control gate rather than a cluster assigned afterward. The contour lines encode the analytic density computed from the same $\pi_k$, $\mu_k$, and $\Sigma_k$. Each colored lobe lies beneath the corresponding density peak, consistent with the branch-mean and branch-weight checks above; the plot itself remains a finite-sample comparison.

# ## Gate inventory
# 
# The table below collects the gates introduced in this tutorial, groups them by primitive, and links to later notebooks that use them.
# 
# Each gate is parametrised by a logit or a small parameter set and then composed through `DiscretePCircuit` or `HybridPCircuit`. The gate object specifies structure, while its parameters occupy the corresponding entry in a separate list. The Used in column identifies notebooks that compose each gate into a larger model.
# 
# The Torx gate library extends beyond this introductory set. For example, `PJUMP` is a two-pbit branch gate that moves probability from $|10)$ to $|01)$.
# 
# | Primitive | Gate | Role | Used in |
# |---|---|---|---|
# | Pbit | `PSWAP` | Two-pbit swap with probability $p$ | [02](02_random_walks_on_graphs.ipynb), [03](03_bunny_graph_diffusion.ipynb) |
# | Pbit | `PNOT` | Single-bit flip | [04](04_execution_interface_readouts.ipynb), [06](06_ising_sampling_contrastive_divergence.ipynb), [07](07_discrete_diffusion.ipynb), [08](08_stochastic_convolutional_networks.ipynb) |
# | Pbit | `PISING` | $4 \times 4$ Glauber bond update | [06](06_ising_sampling_contrastive_divergence.ipynb), [09](09_stochastic_graph_networks.ipynb) |
# | Pdit | `PditCycle` | Stay, move forward, or move backward on a cyclic $d$-state pdit | [13](13_regime_switching_diffusion.ipynb) |
# | Pmode | `AffineGaussianGate` | Affine transform and Gaussian noise | [10](10_pmode_gaussian_gates.ipynb), [11](11_gaussian_hierarchical_ssm.ipynb) |
# | Pmode | `MixtureGaussianGate` | Control-conditional Gaussian mixture | [10](10_pmode_gaussian_gates.ipynb), [13](13_regime_switching_diffusion.ipynb), [14](14_gaussian_bernoulli_clustering.ipynb) |

# ## Conclusion
# 
# We have now described the three Torx data primitives and connected each introductory gate to the kernel it implements:
# 
# - Torx programs act on three data primitives: pbits (binary), pdits ($d$-state), and pmodes (continuous, valued in $\mathbb{R}^N$).
# - Discrete gates are column-stochastic kernels, while pmode gates are continuous Markov kernels (Gaussian maps). Many discrete branch-table gates combine the identity with a deterministic operation selected with probability $p$, parametrised by the logit $\theta$ so they train with ordinary gradients. `PISING` uses a physical parameter vector instead.
# - `PSWAP`, `PNOT`, and `PISING` are the core pbit gates, and `PditCycle` walks a cyclic $d$-state pdit (stay/forward/backward).
# - `AffineGaussianGate` is an affine-Gaussian map, and `MixtureGaussianGate` selects one of several diagonal Gaussian components under a discrete control.
# 
# The following notebooks develop these constructions in larger circuits:
# 
# - [`02_random_walks_on_graphs.ipynb`](02_random_walks_on_graphs.ipynb), the `PSWAP` gate tiled over graph edges,
# - [`06_ising_sampling_contrastive_divergence.ipynb`](06_ising_sampling_contrastive_divergence.ipynb), the `PISING` gate as a Gibbs sampler, and
# - [`13_regime_switching_diffusion.ipynb`](13_regime_switching_diffusion.ipynb), the `PditCycle` gate and `MixtureGaussianGate` composed into one process.
