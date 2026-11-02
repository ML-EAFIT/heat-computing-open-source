#!/usr/bin/env python
# coding: utf-8

# # Basic usage

# <div class="note"><span class="note-t">In brief</span>
# <p>We build a small stochastic circuit, simulate it, and train its parameters with <code>optax</code>. We also run the energy-based <code>PISING</code> gate and check that sampled parameter-shift gradients match exact ones.</p>
# </div>

# In this tutorial, we build a parametrised stochastic circuit and train it end to end, following the Torx workflow from circuit construction through simulation and parameter updates. This compact notebook is the quickstart companion to [`01_introduction_to_parametrised_stochastic_circuits.ipynb`](01_introduction_to_parametrised_stochastic_circuits.ipynb), which introduces each gate and data primitive in more detail.
# 
# By the end, you'll be able to:
# 
# - build a `DiscretePCircuit` from gates and inspect it with `draw_circuit`,
# - train its parameters against a target distribution on the `StateVectorSimulator`,
# - run the energy-based `PISING` gate and watch it relax toward equilibrium, and
# - compare sampled parameter-shift gradients against exact ones with the `BranchingSimulator`.

# ## Setup
# 
# We start by importing the gates and simulators, resolving the helper path, and applying the shared plotting style and `savefig` utility used throughout the notebook.

# In[1]:


from pathlib import Path
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from torx.psc import (
    DiscretePCircuit,
    draw_circuit,
    PCNOT,
    PISING,
    PNOT,
    PSWAP,
    BranchingSimulator,
    StateVectorSimulator,
)

# Resolve examples/helpers whether run from the repo root or from examples/.
ROOT = Path.cwd()
if not (ROOT / "helpers").exists() and (ROOT.parent / "helpers").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "helpers"))


# In[2]:


from _notebook_paths import figure_dir
from _notebook_style import apply_notebook_style, make_savefig
import _nb01b as P

apply_notebook_style()
FIGURE_DIR = figure_dir(ROOT)
savefig = make_savefig(FIGURE_DIR)
SEED = 0


# ## Build a circuit and simulate it
# 
# A circuit is a list of gates that fixes the structure. Here we chain three Boolean pbit gates: a `PNOT` on site 0, a `PSWAP` across sites 0 and 1, and a `PCNOT` controlled on site 1. The `StateVectorSimulator` evolves the full probability vector over the four two-pbit configurations, which is exact and cheap at this size.

# In[3]:


circuit = DiscretePCircuit(
    [
        PNOT(0),
        PSWAP([0, 1]),
        PCNOT([1, 0]),
    ]
)
sim = StateVectorSimulator()


# `draw_circuit` prints the wiring, so you can confirm the structure before training.

# In[4]:


draw_circuit(circuit)


# The parameters live in a separate pytree, one entry per gate. We draw random starting values, put the initial distribution on |00), and set up an Adam optimizer from `optax`.

# In[5]:


key = jax.random.key(SEED)
init_keys = jax.random.split(key, len(circuit))
thetas = [
    jax.random.uniform(k, t.shape, minval=-1.0, maxval=1.0)
    for k, t in zip(init_keys, circuit.init_params(key))
]

optimizer = optax.adam(learning_rate=1e-2)
opt_state = optimizer.init(thetas)

init_state = jnp.zeros(4, dtype=jnp.float32).at[0].set(1.0)


# Inside the loss, we rebuild the compiled circuit from `thetas`; this keeps the computation differentiable with respect to those parameters. The squared-error objective compares the output distribution with an even split between |00) and |11), and the jitted step applies one Adam update.

# In[6]:


@eqx.filter_value_and_grad
def loss_fn(thetas):
    compiled = sim.build_circuit(circuit, thetas)
    state = sim.density(compiled, init_state)
    target = jnp.array([0.5, 0.0, 0.0, 0.5])
    return jnp.sum((target - state) ** 2)


@eqx.filter_jit
def do_step(thetas, opt_state):
    loss, grads = loss_fn(thetas)
    updates, opt_state = optimizer.update(grads, opt_state)
    thetas = eqx.apply_updates(thetas, updates)
    return thetas, opt_state, loss


# Run a thousand steps.

# In[7]:


miniters = 100
train_loop = tqdm(range(1000), miniters=miniters)
losses = []
for step in train_loop:
    thetas, opt_state, loss = do_step(thetas, opt_state)
    losses.append(loss)
    if step % miniters == 0:
        train_loop.set_description(f"Step {step + 1}, Loss {loss:.5f}")


# The trained distribution sits close to the target, with most of its mass on |00) and |11).

# In[8]:


sim.density(sim.build_circuit(circuit, thetas), init_state)


# The loss falls smoothly on a log scale.

# In[9]:


fig = P.training_loss(losses)
savefig(fig, "01b_training_loss")


# ## An energy-based gate: PISING
# 
# Unlike the Boolean gates above, which apply a fixed stochastic map, `PISING` thermalizes two pbits toward the Boltzmann distribution of a two-spin Ising model using Glauber dynamics. Its parameter is the five-vector `[J, h1, h2, beta, dt]`, supplied as the gate's `theta` at build time. `J` is the coupling, `h1` and `h2` are the local fields, `beta` is the inverse temperature, and `dt` sets how much relaxation is applied.

# Starting from a uniform distribution, we sweep the step size `dt` at fixed ferromagnetic coupling. As `dt` grows, the two pbits align and the mass collects on the |00) and |11) states.

# In[10]:


sim = StateVectorSimulator()
init_state = jnp.ones(4) / 4  # uniform over |00), |01), |10), |11)

coupling, beta = 1.0, 2.0
dts = [0.1, 1.0, 5.0]
labels = ["|00)", "|01)", "|10)", "|11)"]

dists = []
for dt in dts:
    gate_circuit = DiscretePCircuit([PISING([0, 1])])
    theta = jnp.array([coupling, 0.0, 0.0, beta, dt])  # [J, h1, h2, beta, dt]
    dists.append(sim.density(sim.build_circuit(gate_circuit, [theta]), init_state))


# In[11]:


fig = P.pising_relaxation(dists, dts, labels=labels, coupling=coupling, beta=beta)
savefig(fig, "01b_pising_relaxation")


# ## Sampled vs exact gradients
# 
# The `BranchingSimulator` estimates gradients by parameter-shift sampling, whereas the `StateVectorSimulator` computes them exactly. To compare the two, we train the same circuit from the same initial parameters with each simulator and compare the loss curves.
# 
# For this comparison, we compile each circuit once before entering the jitted step and optimize the compiled circuit directly, since its parameters are inexact-array leaves. The sampled simulator encodes the circuit structure as arrays during compilation. Because that compilation cannot run under `jit`, it must happen before the training loop.

# Build the circuit and a shared set of random initial parameters.

# In[12]:


circuit = DiscretePCircuit(
    [
        PNOT(0),
        PSWAP([0, 1]),
        PCNOT([1, 0]),
    ]
)

key = jax.random.key(SEED)
init_keys = jax.random.split(key, len(circuit))
base_thetas = [
    jax.random.uniform(k, t.shape, minval=-1.0, maxval=1.0)
    for k, t in zip(init_keys, circuit.init_params(key))
]


# Compile the circuit once for each simulator, then give each path its own Adam optimizer and starting state.

# In[13]:


sim = BranchingSimulator(num_samples=1000, diff_method="param_shift_inf")
statevec_sim = StateVectorSimulator()

sampled_circuit = sim.build_circuit(circuit, base_thetas)
statevec_circuit = statevec_sim.build_circuit(circuit, base_thetas)

optimizer = optax.adam(learning_rate=1e-2)
opt_state = optimizer.init(eqx.filter(sampled_circuit, eqx.is_inexact_array))

optimizer_vec = optax.adam(learning_rate=1e-2)
opt_statevec = optimizer_vec.init(eqx.filter(statevec_circuit, eqx.is_inexact_array))

init_state = jnp.array([0, 0])
init_statevec = jnp.zeros(4, dtype=jnp.float32).at[0].set(1.0)


# For each path, the loss is the negative expectation of pbit 1, and the step applies one Adam update. The sampled loss also takes a PRNG key for its Monte Carlo estimate.

# In[14]:


@eqx.filter_value_and_grad
def loss_fn_exp(circuit, key):
    exp = sim.expval(circuit, init_state, 1, key)
    return -exp


@eqx.filter_jit
def do_step(circuit, opt_state, key):
    loss, grads = loss_fn_exp(circuit, key)
    updates, opt_state = optimizer.update(grads, opt_state)
    circuit = eqx.apply_updates(circuit, updates)
    return circuit, opt_state, loss


# In[15]:


@eqx.filter_value_and_grad
def loss_fn_vec(circuit):
    """Negative expectation of pbit 1 under the exact statevector."""
    exp = statevec_sim.expval(circuit, init_statevec, 1)
    return -exp


@eqx.filter_jit
def do_step_vec(circuit, opt_state):
    """One optimizer step on the circuit parameters."""
    loss, grads = loss_fn_vec(circuit)
    updates, opt_state = optimizer_vec.update(grads, opt_state)
    circuit = eqx.apply_updates(circuit, updates)
    return circuit, opt_state, loss


# Train both for two thousand steps.

# In[16]:


miniters = 100
key = jax.random.key(SEED)
losses = []
losses_vec = []

train_loop = tqdm(range(2000), miniters=miniters)
for step in train_loop:
    key, subkey = jax.random.split(key)
    sampled_circuit, opt_state, loss = do_step(sampled_circuit, opt_state, subkey)
    losses.append(loss)
    if step % miniters == 0:
        train_loop.set_description(f"Step {step + 1}, Loss {loss:.5f}")

train_loop = tqdm(range(2000), miniters=miniters)
for step in train_loop:
    statevec_circuit, opt_statevec, loss_vec = do_step_vec(statevec_circuit, opt_statevec)
    losses_vec.append(loss_vec)
    if step % miniters == 0:
        train_loop.set_description(f"Step {step + 1}, Statevector Loss {loss_vec:.5f}")


# Both paths reach an expectation close to 1 on the target pbit.

# In[17]:


print(init_state)
print(sim.expval(sampled_circuit, init_state, 1, key))
print(init_statevec)
print(statevec_sim.expval(statevec_circuit, init_statevec, 1))


# The sampled loss stays noisy but follows the exact curve down to the same optimum.

# In[18]:


fig = P.gradient_comparison(losses, losses_vec)
savefig(fig, "01b_gradient_comparison")


# ## Where to go next
# 
# - [`01_introduction_to_parametrised_stochastic_circuits.ipynb`](01_introduction_to_parametrised_stochastic_circuits.ipynb) for the full tour of gates over pbits, pdits, and pmodes.
# - [`04_execution_interface_readouts.ipynb`](04_execution_interface_readouts.ipynb) for the expectation readout used in the gradient comparison.
# - [`06_ising_sampling_contrastive_divergence.ipynb`](06_ising_sampling_contrastive_divergence.ipynb) for `PISING` as a Gibbs sampler.
