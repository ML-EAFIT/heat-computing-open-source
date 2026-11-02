#!/usr/bin/env python
# coding: utf-8

# # Spin Models in THRML

# Probabilistic computers that sample from graphical models defined over binary random variables are most natural to build using transistors, and therefore are of elevated interest to Extropic. As such, we've built some tooling into THRML that is dedicated to sampling from these binary PGMs and training machine learning models based on them. This notebook will walk through this functionality and show you how to use it.

# We specifically consider spin-valued EBMs with polynomial interactions. These models implement the probability distribution,
# 
# 
# $$ P(x) \propto e^{-\mathcal{E}(x)}$$
# 
# $$ \mathcal{E}(x) = -\beta \left( \sum_{i \in S_1} W^{(1)}_i s_i + \sum_{(i, j) \in S_2} W^{(2)}_{i, j} s_i s_j +  \sum_{(i, j, k) \in S_3} W^{(3)}_{i, j, k} s_i s_j s_k  + \dots \right) $$
# 
# Here, the $s_i \in \{-1, 1\}$ are spin variables that couple with each other via the $W^{(k)}$, which are scalars that represent the strengths of $k^{th}$ order interactions. $S_k$ is the set of all interactions of order $k$.
# 
# A model of this type that contains at most second-order interactions is called an Ising model or Boltzmann machine. Boltzmann machines are one of the original machine learning models, and their significance was recognized in 2024 with a Nobel prize in physics for John Hopfield and Geoffrey Hinton. 
# 
# Gibbs sampling defines a simple procedure for sampling from this type of model that is very hardware friendly. In particular, the Gibbs sampling update rule corresponding to the above energy function is,
# 
# $$ P(s_i = 1 | s_{nb(i)}) = \sigma[2 \gamma]$$
# 
# $$ \gamma = W^{(1)}_i +  \sum_{j \in S_2[i]} W^{(2)}_{i, j} s_j + \sum_{(j, k) \in S_3[i]} W^{(3)}_{i, j, k} s_j s_k + \dots$$
# 
# where $s_{nb(i)}$ are the spins that are neighbours of $s_i$, and  $S_k[i]$ is the members of $S_k$ that contain $i$.
# 
# From the above equation, we see that we can implement the Gibbs sampling update rule for a spin-valued model by computing simple functions of the neighbour states, multiply-accumulating the results, and then using them to generate an appropriately biased random bit. This can be done very efficiently using mixed signal (analog + digital) hardware; we flesh out a way to do this using only transistors on a modern process [in our recent paper](https://arxiv.org/abs/2510.23972).
# 
# Now that we understand the significance of this type of model, let's see how they can be sampled from using some of the tools built in to THRML.
# 

# First, some imports,

# In[ ]:


import time
import jax

import dwave_networkx
import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# In[2]:


from thrml.block_management import Block
from thrml.block_sampling import sample_states, SamplingSchedule
from thrml.models.discrete_ebm import SpinEBMFactor
from thrml.models.ising import (
    estimate_kl_grad,
    hinton_init,
    IsingEBM,
    IsingSamplingProgram,
    IsingTrainingSpec,
)
from thrml.pgm import SpinNode


# In this example, we will implement a quadratic binary model (Ising model). We will use DWave's "Pegasus" graph topology to allow us to directly compare the speed of our GPU-based sampler to results obtained using other hardware accelerators,

# In[3]:


# make the graph using DWave's code
graph = dwave_networkx.pegasus_graph(14)
coord_to_node = {coord: SpinNode() for coord in graph.nodes}
nx.relabel_nodes(graph, coord_to_node, copy=False)


# Now we can define our model using the functionality exposed by `thrml.models.ising`. For the sake of this example, we will choose random values for the biases and weights $W^{(1)}$ and $W^{(2)}$,

# In[4]:


nodes = list(graph.nodes)
edges = list(graph.edges)

seed = 4242
key = jax.random.key(seed)

key, subkey = jax.random.split(key, 2)
biases = jax.random.normal(subkey, (len(nodes),))

key, subkey = jax.random.split(key, 2)
weights = jax.random.normal(subkey, (len(edges),))

beta = jnp.array(1.0)

model = IsingEBM(nodes, edges, biases, weights, beta)


# The `IsingEBM` class is simply a thin frontend that takes in your weights and biases and produces an appropriate set of `SpinEBMFactor`s,

# In[5]:


[x.__class__ for x in model.factors]


# Now let's do some computation using our `IsingEBM`. Specifically, we are going to look at the tools THRML exposes for *training* this type of model in the context of machine learning. In machine learning, the variables in an EBM are often segmented into "visible" variables (x) and "latent" variables (z). The visible variables represent the data, and the latent variables serve to increase the expressivity of the model. Given these latent variables, our EBMs model of the data is,
# 
# $$ P(x) \propto \sum_z e^{-\mathcal{E}(x, z)}$$
#  
# When training EBMs, one is often interested in minimizing the distributional distance between the EBM and some dataset. This can be done by iteratively updating the model parameters according to the gradient,
# 
# $$ \nabla_{\theta} D(Q(x)|| P(x)) = \mathbb{E}_Q \left[ \mathbb{E}_{P(z|x)} \left[ \nabla_{\theta} \mathcal{E}\right] - \mathbb{E}_{P(z, \: x)} \left[ \nabla_{\theta} \mathcal{E}\right] \right]$$
# 
# 
# Where $D(Q||P)$ indicates the *KL-divergence* between Q and P, which is a common measure of distributional distance in machine learning. Each of the two terms in this gradient can be estimated by sampling from the EBM. The first term is estimated by clamping the data nodes to a member of the dataset and sampling the latents. The second is estimated by sampling both the data and latent variables. We can leverage THRML for both of these computations.

# First, lets set up our block specifications for both the free and clamped sampling. First, lets choose some random subset of our nodes to represent the data,

# In[6]:


n_data = 500

np.random.seed(seed)

data_inds = np.random.choice(len(graph.nodes), n_data, replace=False)
data_nodes = [nodes[x] for x in data_inds]


# Now, lets compute the minimum coloring for the unclamped term in our gradient estimator,

# In[7]:


coloring = nx.coloring.greedy_color(graph, strategy="DSATUR")
n_colors = max(coloring.values()) + 1
free_coloring = [[] for _ in range(n_colors)]
# form color groups
for node in graph.nodes:
    free_coloring[coloring[node]].append(node)

free_blocks = [Block(x) for x in free_coloring]


# and the same for the clamped term,

# In[8]:


# in this case we will just re-use the free coloring
# you can always do this, but it might not be optimal

# a graph without the data nodes
graph_copy = graph.copy()
graph_copy.remove_nodes_from(data_nodes)

clamped_coloring = [[] for _ in range(n_colors)]
for node in graph_copy.nodes:
    clamped_coloring[coloring[node]].append(node)

clamped_blocks = [Block(x) for x in clamped_coloring]


# We have now defined everything we need to calculate some gradients! We can set up a few more details and get to it,
# 
# `IsingTrainingSpec` creates logical programs for the positive and negative phases. The calls below use the default JAX backend and explicit initial states. Rebuild the training specification after updating the model so both phase programs expose current biases, weights, and temperature. This rebuilding can happen inside a JAX training loop.

# In[9]:


# lets define some random "data" to use for our example
# in real life this could be encoded images, text, video etc
data_batch_size = 50

key, subkey = jax.random.split(key, 2)
data = jax.random.bernoulli(subkey, 0.5, (data_batch_size, len(data_nodes))).astype(jnp.bool)

# we will use the same sampling schedule for both cases
schedule = SamplingSchedule(5, 100, 5)

# convenient wrapper for everything you need for training
training_spec = IsingTrainingSpec(model, [Block(data_nodes)], [], clamped_blocks, free_blocks, schedule, schedule)

# how many parallel sampling chains to run for each term
n_chains_free = data_batch_size
n_chains_clamped = 1

# initial states for each sampling chain
# THRML comes with simple code for implementing the hinton initialization, which is commonly used with boltzmann machines
key, subkey = jax.random.split(key, 2)
init_state_free = hinton_init(subkey, model, free_blocks, (n_chains_free,))
key, subkey = jax.random.split(key, 2)
init_state_clamped = hinton_init(subkey, model, clamped_blocks, (n_chains_clamped, data_batch_size))


# In[10]:


# now for gradient estimation!
# this function returns the gradient estimators for the weights and edges of our model, along with the moment data that was used to estimate them
# the moment data is also returned in case you want to use it for something else in your training loop
key, subkey = jax.random.split(key, 2)
weight_grads, bias_grads, clamped_moments, free_moments = estimate_kl_grad(
    subkey,
    training_spec,
    nodes,  # the nodes for which to compute bias gradients
    edges,  # the edges for which to compute weight gradients
    [data],
    [],
    init_state_clamped,
    init_state_free,
)


# This function simply returns vectors for the weight and bias grads,

# In[11]:


print(weight_grads)
print(bias_grads)


# The returned gradients can be used to train the model. With an external backend, pass `backend=your_backend` to `estimate_kl_grad` and keep the outer loop in host-side Python. The backend owns request batching and continuation: `None` requests one retained-chain replica for a phase, while JAX requires explicit initialization for sampled phases. A fully clamped positive phase accepts `[]` or `None` and computes moments directly from data and conditioning values without backend execution.

# The default JAX backend runs sampling programs very efficiently on GPUs and is competitive with the state of the art for sampling from sparse Ising models. Let's demonstrate that with a simple benchmark,

# !!! warning
# 
#     The following requires 8x GPUs.

# In[12]:


from jax.sharding import PartitionSpec as P


# In[17]:


mesh = jax.make_mesh((8,), ("x",))
sharding = jax.sharding.NamedSharding(mesh, P("x"))

timing_program = IsingSamplingProgram(model, free_blocks, [])

timing_chain_len = 100

batch_sizes = [8, 80, 800, 8000, 64_000, 160_000, 320_000]
times = []
flips = []
dofs = []

schedule = SamplingSchedule(timing_chain_len, 1, 1)

call_f = jax.jit(
    jax.vmap(lambda k: sample_states(k, timing_program, schedule, [x[0] for x in init_state_free], [], [Block(nodes)]))
)

for batch_size in batch_sizes:
    key, subkey = jax.random.split(key, 2)
    keys = jax.random.split(key, batch_size)
    keys = jax.device_put(keys, sharding)
    _ = jax.block_until_ready(call_f(keys))

    start_time = time.time()
    _ = jax.block_until_ready(call_f(keys))
    stop_time = time.time()

    times.append(stop_time - start_time)
    flips.append(timing_chain_len * len(nodes) * batch_size)
    dofs.append(batch_size * len(nodes))


# In[18]:


flips_per_ns = [x / (y * 1e9) for x, y in zip(flips, times)]


# In[ ]:


fig, axs = plt.subplots()
plt.title("Performance on 8xB200")
axs.plot(dofs, flips_per_ns)
axs.set_xscale("log")
axs.set_xlabel("Parallel Degrees of Freedom")
axs.set_ylabel("Flips/ns")
plt.savefig("fps.png", dpi=300)
plt.show()


# <center>
# <img src="../fps.png" width="400" style="background:white; padding:10px; border-radius:8px;">
# </center>

# You can compare your results to an FPGA implementation that bakes the sampling problem directly into hardware [here](https://arxiv.org/abs/2303.10728) (they get ~60 flips/ns).

# Note that despite our focus on quadratic models here, THRML comes with the ability to support spin interactions of arbitrary order out of the box. This ability can be accessed via `thrml.models.discrete_ebm.SpinEBMFactor`,

# In[16]:


# this creates a cubic interaction s_1 * s_2 * s_3 between a subset of our nodes
SpinEBMFactor([Block(nodes[:10]), Block(nodes[10:20]), Block(nodes[20:30])], jax.random.normal(key, (10,)))


# That's about everything there is to know about binary EBMs in THRML! We hope you use these tools to help us gain a better understanding of how to most effectively use these powerful primitives in more advanced machine learning architectures.
