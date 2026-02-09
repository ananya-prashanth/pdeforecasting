---
layout: post
title: "Equivariant Neural Fields for PDE Forecasting: Learning Physics-Informed Continuous Dynamics"
date: 2026-02-03
categories: [deep-learning, pde, physics, symmetry]
author: Ananya Prashanth
description: Solving partial differential equations on complex geometries by respecting physical symmetries through equivariant neural fields
math: true
---

## Introduction

<details class="callout tldr">
<summary><span class="label">TLDR</span></summary>

Equivariant Neural Fields enable continuous PDE forecasting on complex geometries by encoding symmetry-aware latent representations and decoding them equivariantly, avoiding grids and improving generalization.


</details>

Partial differential equations (PDEs) are fundamental to understanding and modeling spatiotemporal dynamics across virtually all scientific domains: from fluid mechanics and weather forecasting to biological systems and materials science. 

At their core, PDEs describe how things change over space and time. They are used to model phenomena like heat diffusion, fluid flow, and wave propagation, ie essentially whenever both space and time matter.

Given an initial state, a PDE specifies how a physical quantity evolves everywhere in space as time moves forward.

Traditionally, PDEs are solved using numerical methods such as finite element or spectral methods. These approaches discretize the continuous domain onto computational grids and approximate the solution numerically. While highly accurate, such solvers often require dense grids and can be computationally expensive, especially for complex geometries or long time horizons.


<img src="{{ site.baseurl }}/assets/images/pde.png"
     alt="Training procedure"
     style="width: 80%; display: block; margin: auto;">
<p style="font-style: italic; font-size: 0.9em;">
		Figure 1: Illustration of a physical field governed by a partial differential equation evolving over space and time.
</p>


Recent advances in deep learning have opened new possibilities for data-driven PDE solving[^3]. However, most existing neural approaches face critical limitations: they rely on regular grids, struggle with complex geometries, and fail to incorporate known physical symmetries, properties that are intrinsic to the systems they model. **What if we could build neural models that naturally respect the geometry and symmetries of physical laws?**

In this blog post, we explore the work of Knigge et al. (2024)[^1], which introduces a framework for **space-time continuous PDE forecasting using Equivariant Neural Fields (ENFs)**. This approach combines the flexibility of neural fields[^4] with the structured inductive biases of equivariant neural networks[^5], enabling accurate and efficient prediction of PDE dynamics on challenging geometries like spheres, tori, and 3D balls.

## The Challenge of PDE Forecasting


Consider the task of forecasting weather patterns, ocean currents, or heat diffusion through materials. In each case, we observe a physical field $$\nu(x, t)$$ that evolves over space $$x$$ and time $$t$$ according to some governing PDE. The challenges are multifold:

1. **Irregular Sampling**: Observations may be sparse and irregularly distributed in space and time
2. **Complex Geometries**: Physical domains may be non-planar (e.g., spherical Earth, periodic boundaries)
3. **Symmetries**: Physical laws remain invariant under certain transformations (rotations, translations)
4. **Continuous Queries**: We need predictions at arbitrary spatial and temporal locations, not just grid points

Traditional grid-based methods like CNNs excel when data is regular and planar, but struggle with irregular sampling and complex geometries. On the other hand, methods that ignore physical symmetries waste modeling capacity learning patterns that should be known *a priori*.

We are thus, in need of a more reliable model that can deal with real-world data, that is irregularly and sparsely sampled, have complex geometries involved, while also respecting the underlying symmetry properties.

## Background and Motivation

<details class="callout tldr" markdown="1">

<summary><span class="label">TLDR</span></summary>

Neural fields offer continuous, grid-free representations of physical systems. By conditioning them on latent codes and enforcing equivariance to physical symmetries, models generalize better, require less data, and produce physically consistent predictions.


</details>

### Neural Fields for Continuous Representations


Neural fields are a way to represent physical quantities, like temperature, pressure, or velocity, not as values on a fixed grid, but as a **continuous function**. Instead of storing values at predefined points, we train a neural network that can answer the question:

<p style="margin-left: 1em; font-style: italic; color: #030d66;">
<b> What is the value of the field at this coordinate?</b>
</p>

You give the network a coordinate, for example a position in space and time, and it outputs the corresponding field value.

Formally, this idea is captured by a neural field, also called a **coordinate-based neural network**, which represents a function


$$f_\theta: \mathbb{R}^n \rightarrow \mathbb{R}^d$$

where the input is a coordinate $x \in \mathbb{R}^n$, the output is a field value in $\mathbb{R}^d$, and $\theta$ denotes the neural network parameters [^4].


To represent different physical states, such as different initial conditions of a PDE, we extend this idea using a conditional neural field. In this setting, the network is additionally conditioned on a latent code $z$:

$$f_\theta(x; z): \mathbb{R}^n \times \mathbb{R}^c \rightarrow \mathbb{R}^d$$

You can think of the latent code $z$ as a compact summary of an entire physical system. By changing $z$, the same neural network can represent many different fields. In other words, a single model can encode a whole family of solutions, each indexed by a different latent code.

![Alt text]({{ site.baseurl }}/assets/images/cnef.png)
<p style="font-style: italic; font-size: 0.9em;">
Figure 2: Conditional neural fields map continuous coordinates to field values, with different latent codes representing different physical states or PDE solutions.
</p>

This makes neural fields especially attractive for scientific problems:

- They work on **irregular or sparse data**, not just grids  
- They naturally interpolate between observed points  
- They store complex, high-resolution fields in a **compact and memory-efficient** way  


Neural fields have been successfully applied to 3D shape representation, novel view synthesis, and various other tasks[^3]. Recent work by Yin et al. (2022)[^4] proposed using conditional neural fields for PDE forecasting by learning dynamics in latent space. However, this approach treats the latent space as an unstructured vector space, ignoring the geometric properties of the underlying physical system.




### The Role of Symmetry in Physics


Many physical laws exhibit **symmetries**, meaning their behavior does not depend on arbitrary choices such as orientation or position. Intuitively, if we transform a physical system in a way that respects its geometry, the underlying law governing the system should remain unchanged.

For example:

- **Heat diffusion** is rotationally symmetric: rotating the initial temperature distribution yields a rotated solution 

- **Navier–Stokes equations** on a periodic domain exhibit translational symmetry  

- **Atmospheric flow** on a sphere has rotational symmetry around Earth’s axis  




<img src="{{ site.baseurl }}/assets/images/heat.png"
     alt="Training procedure"
     style="width: 80%; display: block; margin: auto;">
<p style="font-style: italic; font-size: 0.9em;">
Figure 3: Heat diffusion is rotationally symmetric. Rotating the initial temperature distribution produces a rotated solution.
</p>

<img src="{{ site.baseurl }}/assets/images/navier_stokes_periodic.png"
     alt="Training procedure"
     style="width: 80%; display: block; margin: auto;">
<p style="font-style: italic; font-size: 0.9em;">
Figure 4: Navier–Stokes equations on a periodic domain are equivariant to translations, reflecting the topology of the 2-torus.
</p>


These properties are not accidental; they reflect fundamental invariances of the physical systems being modeled.

Mathematically, this idea is formalized through **equivariance**. If a PDE operator $$\mathcal{N}$$ is equivariant to a group $$G$$ of transformations, then for any group element $$g \in G$$:

$$\mathcal{N}[L_g \nu] = L_g \mathcal{N}[\nu]$$

where $$L_g$$ denotes the action of $$g$$ on the field $$\nu$$. This equation expresses a simple but powerful principle: applying a transformation to a solution of the PDE produces another valid solution of the same equation. In other words, **transforming a solution gives another valid solution**.

Respecting this structure is crucial for learning-based models of physical systems. If symmetry is ignored, a model must rediscover these invariances from data, which can lead to poor generalization and physically inconsistent predictions. Encoding equivariance directly into the model allows us to align learning with the known structure of the underlying physics.

**Equivariant neural networks**[^5] enforce this property by construction, providing a powerful inductive bias that:
- Improves generalization to unseen transformations
- Increases data efficiency
- Ensures physical consistency

Group equivariant CNNs, steerable CNNs, and E(n)-equivariant graph neural networks[^6] have demonstrated significant benefits in various domains including molecular property prediction and physical simulation.

## Method Overview

<details class="callout tldr" markdown="1">
<summary><span class="label">TLDR</span></summary>

- PDE states are represented as structured latent point clouds of poses and context features.
- Group actions transform poses while leaving context invariant, ensuring equivariance by design.
- Decoding uses bi-invariant attributes and cross-attention to guarantee symmetry-respecting predictions.
- Latent dynamics are modeled with an equivariant neural ODE using message passing on group manifolds.
- Meta-learning enables fast inference and a well-structured latent space.


</details>


The proposed framework models PDE solutions as **flows in a structured latent space** that explicitly preserves geometric information.  
At a high level, instead of representing a physical state as a single vector, the model represents it as a **set of interacting elements**, each with a clear geometric meaning.

Formally, each field state $$\nu_t$$ is encoded as a latent set

$$z^\nu_t = \{(p_i, c_i)\}_{i=1}^N$$

where:
- $$p_i$$ is a **pose** that encodes *where* the element lives geometrically (for example, a position and orientation in groups like SE(2) or SO(3))
- $$c_i$$ is a **context vector** that encodes *what* the element represents, such as local physical features

Intuitively, you can think of this latent representation as a **point cloud in a geometric space**. Each point carries both:
- a location that transforms correctly under rotations or translations, and  
- a feature vector that describes local physical information.

This separation is crucial. When a symmetry transformation (such as a rotation) is applied:
- the poses $$p_i$$ transform according to the group action,  
- while the context vectors $$c_i$$ remain invariant.

Because geometry is explicitly represented in the latent space, **equivariance is preserved by construction**. The model never needs to relearn how physical symmetries behave; they are built directly into the representation and the dynamics.

![Alt text]({{ site.baseurl }}/assets/images/framework.png)  
<p style="font-style: italic; font-size: 0.9em;">
Figure 5: Overview of the Equivariant Neural Fields framework. PDE states are encoded as structured latent point clouds and evolved using equivariant neural ODEs before being decoded into continuous fields.
</p>

### Structured Latent Representation

Unlike prior approaches that represent PDE states using a single **unstructured latent vector**, this framework represents each state as a **set of elements with explicit geometric meaning**. Concretely, each state is encoded as a **point cloud in a symmetry group space**, where every point carries both geometry and features.

Each latent element consists of:

1. **Geometric information** ($$p_i$$):  
   This represents *where* the element lives in space and orientation.  
   Importantly, $$p_i$$ lies on a **group manifold** corresponding to the symmetry of the problem (for example SE(2) for planar rotations and translations, or SO(3) for rotations on the sphere).

2. **Feature information** ($$c_i$$):  
   This is a standard vector in Euclidean space that stores *what* the element represents, such as local physical properties or learned features.

A symmetry transformation is modeled as a **group action** $$g \in G$$ acting on the latent set as:

$$g \cdot z^\nu = \{(g \cdot p_i, c_i)\}_{i=1}^N$$

Crucially, the transformation affects **only the geometric part** $$p_i$$, while the feature vectors $$c_i$$ remain unchanged.

Intuitively, this mirrors how physical systems behave: when we rotate or translate a system, the *positions* change, but the underlying physical quantities associated with those positions do not. By separating geometry from features in this way, the model guarantees **equivariance by construction**, rather than learning it implicitly from data.

This structured latent design is the foundation that allows equivariant decoding and equivariant latent dynamics in later stages of the model.


### Equivariant Neural Fields as Decoders

The role of the decoder is to reconstruct the physical field (for example temperature or velocity) at any queried coordinate $$x$$, given the structured latent representation. This decoder is implemented as an **equivariant neural field**, meaning that if we transform both the input coordinate and the latent representation in the same way, the output transforms consistently.

At a high level, the decoder works by letting each latent element “contribute” to the value at a query point $$x$$, based on how *geometrically related* that latent element is to the query location.

To achieve this, the decoder $$f_\theta$$ reconstructs field values from latents via **cross-attention**[^8] over **bi-invariant attributes**. A bi-invariant attribute $$a(p_i, x)$$ satisfies:

$$a(p_i, x) = a(g \cdot p_i, g \cdot x) \quad \forall g \in G$$

This property is crucial: it ensures that the relative relationship between a latent pose $$p_i$$ and a query coordinate $$x$$ does not change under symmetry transformations. In other words, the decoder only depends on *relative geometry*, not absolute position or orientation.



#### Example: SE(2) (Planar Translations and Rotations)

For planar problems with rotational and translational symmetry, such as diffusion on the plane, the bi-invariant attribute is defined as:

$$a^{SE(2)}_i(x) = R_i^T(x - p_i)$$

where $$p_i = (t_i, R_i)$$ consists of a translation and a rotation. This expression represents the query point $$x$$ expressed in the local coordinate frame of the latent pose $$p_i$$.

If both $$x$$ and $$p_i$$ are rotated or translated together, this relative coordinate remains unchanged. As a result, attention weights computed from this attribute are shared across all equivalent geometric configurations.


#### Example: 2-Torus (Periodic Boundary Conditions)

For PDEs defined on a flat torus, such as Navier–Stokes with periodic boundaries, the symmetry is periodic translation rather than rotation. In this case, the bi-invariant attribute is:

$$a^{T^2}_i(x) = \cos(2\pi(x_0 - p_0)) \oplus \cos(2\pi(x_1 - p_1))$$

This construction respects periodic translations while intentionally ignoring rotations, matching the physical symmetry of the domain.



#### Cross-Attention Decoder

Using these bi-invariant attributes, the decoder computes the field value as:

$$f_\theta(x; z^\nu) = \text{CrossAttention}(Q(a_{:,x}), K(c), V(c; a))$$

Here:
- **Queries** depend on the relative geometry between latents and the query point
- **Keys and values** depend on the context vectors $$c_i$$
- Attention weights are therefore shared across geometrically equivalent configurations

This design guarantees equivariance by construction:

$$f_\theta(g \cdot x; g \cdot z^\nu) = f_\theta(x; z^\nu)$$

In simple terms: transforming the input and latent state together produces the same output, up to the same transformation.



<!-- Side-by-side pseudo-code comparison -->
<div class="two-col">
	<div class="col">
	#### Decoder (pseudo)
	<pre><code>def decode(x, latents):
		attrs = [a(pi, x) for pi in latents.poses]
		q = proj_query(attrs)
		k,v = proj_key_value(latents.contexts)
		out = cross_attention(q,k,v)
		return readout(out)</code></pre>
	</div>
	<div class="col">
	#### Latent ODE (pseudo)
	<pre><code>def latent_dynamics(z):
		for i in nodes(z):
			msgs = [phi(a(i,j))*c_j for j in neighbors]
			c_i' = aggregate(msgs)
			p_i' = exp_map(p_i, avg_log_maps(msgs))
		return z'
	</code></pre>
	</div>
</div>

Overall, the decoder can be understood as a **symmetry-aware interpolation mechanism**: each latent element contributes to the output based on its relative position to the query point, ensuring that the reconstructed field respects the known physical symmetries of the PDE.

### Equivariant Latent Dynamics

Once a PDE state has been encoded into a structured latent representation, the next step is to **predict how this latent state evolves over time**. Instead of directly evolving the physical field, the method learns dynamics in latent space using a **neural ordinary differential equation (neural ODE)**[^7]:

$$\frac{dz^\nu}{dt} = F_\psi(z^\nu)$$

Here, $$F_\psi$$ defines a continuous-time vector field over the latent space. Integrating this ODE forward in time produces future latent states, which can then be decoded back into physical fields.

Crucially, because the underlying PDE respects geometric symmetries, the latent dynamics must do so as well. This is enforced by requiring the vector field to be **equivariant**:

$$F_\psi(g \cdot z) = g \cdot F_\psi(z)$$

Intuitively, this means that if we rotate or translate the latent representation, the predicted time derivative rotates or translates in exactly the same way.


#### Message Passing on the Latent Point Cloud

To construct such an equivariant vector field, the model applies **message passing** over the latent point cloud. Each latent element exchanges information with others based on their relative geometric relationships.

The updates are split into two parts:

**Context update:**

$$c_i^{l+1} = \sum_{j} \phi_{\text{context}}(a^l_{i,j}) c_j^l$$

This update aggregates information from neighboring elements to update the feature vectors $$c_i$$. The weights depend only on bi-invariant attributes $$a_{i,j}$$, ensuring symmetry-respecting interactions.

**Pose update (on the manifold):**

$$p_i^{l+1} = \exp_{p_i^l}\left(\frac{1}{N}\sum_{j} \phi_{\text{pose}}(a^l_{i,j}) \log_{p_i^l}(p_j^l)\right)$$

Here, pose updates are performed directly on the group manifold. The logarithmic map $$\log$$ converts relative poses into vectors in the tangent space, where weighted averaging is well-defined. The exponential map $$\exp$$ then projects the update back onto the manifold.

In Euclidean space, this reduces to standard coordinate updates. On curved spaces such as spheres or rotation groups, it naturally generalizes to respect the geometry.



#### Equivariance of the Flow

A key consequence of this construction is that the **entire ODE solution is equivariant**. Denoting the flow of the ODE by $$\phi_t$$, we obtain:

$$\phi_t(g \cdot z_0) = g \cdot \phi_t(z_0)$$

In simple terms: transforming the initial condition and then solving the latent dynamics gives the same result as solving first and transforming afterward.

This property guarantees that symmetry is preserved not only locally in time, but throughout the entire rollout of the latent trajectory.


<div style="text-align: center;">
  <img src="{{ site.baseurl }}//assets/images/transolve.png"
       alt="Training procedure"
       style="width: 55%;">
  <p style="font-style: italic; font-size: 0.9em;">
    Figure 6: Solving then transforming = transforming then solving
	
  </p>
</div>


As a result, long-term predictions remain physically consistent, and the model generalizes more reliably across different geometric configurations.

### Meta-Learning for Efficient Inference

To obtain the initial latent $$z_0^\nu$$ from the initial condition $$\nu_0$$, the authors use **meta-learning**[^9] instead of slow autodecoding.

In standard autodecoding, the latent code is optimized from scratch for each new initial condition using many gradient steps, which is computationally expensive and can lead to poorly structured latent spaces. Meta-learning addresses this by learning how to infer latents efficiently.

The procedure consists of two nested optimization loops:

1. **Outer loop**  
   Learns shared parameters, including the decoder parameters $$\theta$$ and a common latent initialization.

2. **Inner loop**  
   For each PDE instance, adapts the initial latent $$z_0^\nu$$ using only **3–4 gradient steps** to minimize the reconstruction loss on $$\nu_0$$.

This design provides two key benefits:

- **Fast inference**  
  At test time, only a handful of optimization steps are required to infer the latent representation of a new initial condition.

- **Structured latent space**  
  Because all latents must be reachable from a shared initialization in very few steps, the latent space becomes smoothly organized. Temporally adjacent states are naturally encouraged to lie close together.

![Alt text]({{ site.baseurl }}/assets/images/tsne.png)

*Figure 7: T-SNE embeddings of ENF latent states under different training setups.  
(a) Autodecoding without equivariance produces a disordered latent space.  
(b) Meta-learning alone improves structure and temporal coherence.  
(c) Combining meta-learning with equivariant weight sharing yields the most coherent and temporally organized latent space, highlighting the complementary benefits of both design choices.*

Visualizing the latent space via t-SNE[^10] confirms the authors’ hypothesis: **meta-learning and equivariance jointly impose strong structure on the latent representations**, simplifying the task of the neural ODE and improving stability and generalization.


## Experiments and Results

<details class="callout tldr" markdown="1">
<summary><span class="label">TLDR</span></summary>

- Equivariant Neural Fields generalize to unseen geometric transformations where non-equivariant baselines fail.
- The model remains robust under sparse and irregular observations of initial conditions.
- Equivariance significantly improves data efficiency and stability on spheres, tori, and 3D volumes.
- The approach supports continuous queries and zero-shot super-resolution on complex geometries.



</details>

### Experimental Setup

The framework is evaluated on a diverse set of PDEs defined over domains with different geometries and symmetry groups:

| **PDE** | **Domain** | **Symmetry** |
|---------|------------|--------------|
| Heat equation | Plane $$\mathbb{R}^2$$ | SE(2) |
| Heat equation | Sphere $$\mathbb{S}^2$$ | SO(3) |
| Navier–Stokes | 2-Torus $$\mathbb{T}^2$$ | Periodic translations |
| Shallow-water | Sphere $$\mathbb{S}^2$$ | Axial rotation |
| Convection | Ball $$\mathbb{B}^3$$ | SO(3) |

**Training protocol**
- Datasets generated using numerical solvers (py-pde, Dedalus)
- Disjoint train and test initial conditions
- Evaluation on both seen ($$t_{\text{in}}$$) and unseen ($$t_{\text{out}}$$) time horizons
- Mean squared error (MSE) measured on rollouts conditioned only on the initial state



### Generalization Under Geometric Transformations

**Experiment**  
Heat equation on $$\mathbb{R}^2$$ with spatially separated training and test regions.

- **Training data**: Gaussian spikes in the upper half-plane ($$0 < x_2 < 1$$)
- **Test data**: Gaussian spikes in the lower half-plane ($$-1 < x_2 < 0$$)

This setup forces the model to generalize across **unseen translations and rotations** of the initial condition.

<div style="text-align: center;">
  <img src="{{ site.baseurl }}/assets/images/traint.png"
       alt="Heat equation generalization experiment"
       style="width: 55%;">
  <p style="font-style: italic; font-size: 0.9em;">
    Figure 8: Training and test regions for the planar heat equation experiment, designed to evaluate generalization under unseen geometric transformations.
  </p>
</div>

| **Model** | $$t_{\text{in}}$$ Train | $$t_{\text{out}}$$ Train | $$t_{\text{in}}$$ Test | $$t_{\text{out}}$$ Test |
|-----------|---------------------|----------------------|-------------------|---------------------|
| DINo (baseline) | 5.92e-04 | 2.40e-04 | 3.85e-03 | 5.12e-03 |
| Ours (no equivariance) | 6.23e-06 | 4.90e-06 | 2.19e-03 | 5.08e-04 |
| **Ours (SE(2) equivariant)** | **1.18e-05** | **2.53e-05** | **1.50e-05** | **2.53e-05** |

The equivariant model maintains low error on test initial conditions that differ geometrically from training data, while both the baseline and non-equivariant variant fail. This demonstrates that **explicitly encoding symmetry enables zero-shot generalization to unseen transformations**.



### Robustness and Data Efficiency

#### Sparse Initial Conditions: Navier–Stokes on $$\mathbb{T}^2$$

The model is evaluated under increasingly sparse observations of the initial condition.

<div style="text-align: center;">
  <img src="{{ site.baseurl }}/assets/images/nst.png"
       alt="Navier–Stokes robustness experiment"
       style="width: 60%;">
<p style="font-style: italic; font-size: 0.9em;">
	  Figure 9: Robustness of Navier–Stokes predictions on the 2-torus under increasingly sparse observations of the initial condition.
</p>
</div>

| **Model** | 100% observed | 50% observed | 5% observed |
|-----------|---------------|--------------|-------------|
| FNO[^11] | 8.44e-05 | 3.20e-02 | 3.84e-01 |
| G-FNO[^12] | **3.15e-05** | 2.32e-02 | 3.40e-01 |
| DINo | 1.11e-02 | 3.74e-02 | 3.94e-02 |
| **Ours (equivariant)** | 1.57e-03 | **5.75e-03** | **3.44e-02** |

Grid-based methods perform best when the field is fully observed on a regular grid. However, under severe sparsity, equivariant neural fields significantly outperform all baselines. This regime is particularly relevant for real-world scientific data, where dense measurements are rarely available.


#### Data Efficiency: Heat Diffusion on $$\mathbb{S}^2$$

Data efficiency is evaluated by varying the number of training trajectories.

<div style="text-align: center;">
  <img src="{{ site.baseurl }}/assets/images/spheret.png"
       alt="Data efficiency on the sphere"
       style="width: 55%;">
	<p style="font-style: italic; font-size: 0.9em;">
	   Figure 10: Data efficiency for heat diffusion on the sphere. Equivariant models generalize reliably with significantly fewer training trajectories.
</p>
</div>

Key observations:
- Non-equivariant models overfit when trained on small datasets
- **Equivariant models generalize reliably with as few as 16 trajectories**
- This corresponds to approximately a **4× reduction in data requirements**



### Complex Geometries

#### Shallow-Water Equations on $$\mathbb{S}^2$$

<div style="text-align: center;">
  <img src="{{ site.baseurl }}/assets/images/shalwater.png"
       alt="Shallow-water equations on the sphere"
       style="width: 55%;">
	<p style="font-style: italic; font-size: 0.9em;">
	   Figure 11: Predictions for the shallow-water equations on the sphere, demonstrating symmetry-aware modeling and zero-shot super-resolution.
	   </p>
</div>

This experiment demonstrates the ability to:
1. Handle spherical geometry without coordinate singularities
2. Respect axial symmetry induced by Coriolis forces
3. Perform zero-shot super-resolution with 2× spatial upsampling



#### Internally Heated Convection in $$\mathbb{B}^3$$

<div style="text-align: center;">
  <img src="{{ site.baseurl }}/assets/images/ball.png"
       alt="Convection in a 3D ball"
       style="width: 55%;">
	<p style="font-style: italic; font-size: 0.9em;">
	   Figure 12: Internally heated convection inside a 3D ball. Enforcing rotational equivariance leads to substantially improved prediction accuracy.
	</p>
</div>

| **Model** | $$t_{\text{in}}$$ Test | $$t_{\text{out}}$$ Test |
|-----------|-------------------|---------------------|
| DINo | 3.06e-03 | 7.78e-02 |
| **Ours** | **5.99e-04** | **7.97e-03** |

By respecting rotational symmetry in three dimensions, the equivariant model achieves nearly a **10× reduction in error** compared to the baseline. This highlights the value of symmetry-aware inductive biases when modeling PDEs on challenging geometries.


## Discussion

<details class="callout tldr">
<summary><span class="label">TLDR</span></summary>

Equivariant Neural Fields offer strong physical consistency and data efficiency, but require prior knowledge of symmetries and introduce additional computational and implementation complexity.

</details>

### Strengths

1. **Physical consistency**  
   The model provably respects known symmetries of the underlying PDEs, ensuring physically meaningful predictions.

2. **Data efficiency**  
   By encoding strong inductive biases, the method requires significantly fewer training trajectories than non-equivariant baselines.

3. **Geometric flexibility**  
   The framework naturally extends to a wide range of domains, including planes, spheres, tori, and volumetric geometries.

4. **Continuous representation**  
   Neural fields enable evaluation at arbitrary space–time coordinates, allowing the model to handle sparse, irregularly sampled observations.



### Limitations

1. **Domain knowledge required**  
   The symmetry group must be specified in advance, which assumes prior knowledge of the physical system.

2. **Computational cost**  
   Global attention in neural field decoders scales quadratically with the number of latent elements, increasing runtime and memory usage.

3. **Error accumulation**  
   Long-term autoregressive rollouts can accumulate errors, a limitation shared by most learned PDE solvers.

4. **Implementation complexity**  
   Correctly handling group actions and manifold-valued operations requires careful and non-trivial implementation.



### Comparison to Alternatives

- **Grid-based methods (CNNs, FNOs)**  
  Efficient and effective on regular grids, but struggle with irregular sampling and complex geometries.

- **Neural operators**  
  Offer strong theoretical foundations, but are typically tied to fixed discretizations in practice.

- **Other neural field approaches (e.g., DINo)**  
  Support continuous representations, but lack explicit geometric structure and symmetry preservation.

Overall, the strength of this approach lies in applications where physical symmetries are known, data is sparse or irregular, and the geometry of the domain is complex. These are precisely the regimes where many traditional numerical and learning-based methods face significant challenges.

## Conclusion



- Equivariant Neural Fields provide a principled framework for learning continuous spatiotemporal dynamics while explicitly respecting the physical symmetries that govern many scientific systems. By representing PDE states as structured latent point clouds and enforcing equivariance throughout the architecture, the model aligns learning with known geometric and physical structure rather than relying on data alone.

- Across a diverse set of experiments, this design leads to clear benefits. The method shows improved generalization to unseen geometric transformations, robustness to sparse and irregular observations, greater data efficiency when training samples are limited, and the ability to handle complex non-planar geometries such as tori, spheres, and volumetric domains. In particular, the results demonstrate that symmetry-preserving models consistently outperform non-equivariant baselines in regimes where traditional grid-based or unstructured latent approaches struggle.

- More broadly, this work highlights the importance of inductive biases in scientific machine learning. Neural fields provide the flexibility required to model continuous phenomena from sparse data, while equivariant architectures ensure that learned representations remain physically meaningful and stable over long time horizons. As learning-based methods increasingly rely on real-world observational data, which is inherently noisy, incomplete, and irregular, physics-informed modeling approaches will become increasingly essential.

- Overall, this study reinforces a central lesson. Incorporating symmetry and geometric structure into deep learning models is not only beneficial, but often necessary for building reliable, scalable, and scientifically grounded methods for PDE forecasting and beyond.


### Future Directions



There are several promising directions for future work.
- First, the current approach assumes that the relevant symmetries of the system are known in advance. An important extension would be to automatically discover or learn these symmetries directly from data.
- Second, while equivariance improves stability, long-term rollouts still suffer from error accumulation. Addressing this remains an open challenge for autoregressive PDE solvers.
- From a practical perspective, improving scalability is also important, particularly by reducing the computational overhead of global attention in neural field architectures.
- Finally, extending this framework to more complex multi-physics systems and real-world observational data would further demonstrate its applicability in scientific settings


---


## References

[^1]: Knigge, D. M., Wessels, D. R., Valperga, R., Papa, S., Sonke, J. J., Gavves, E., & Bekkers, E. J. (2024). Space-Time Continuous PDE Forecasting using Equivariant Neural Fields. arXiv:2406.06660.

[^3]: Xie, Y., Takikawa, T., Saito, S., Litany, O., Yan, S., et al. (2022). Neural fields in visual computing and beyond. Computer Graphics Forum, 41(2), 641–676.

[^4]: Yin, Y., Kirchmeyer, M., Franceschi, J. Y., Rakotomamonjy, A., & Gallinari, P. (2022). Continuous PDE dynamics forecasting with implicit neural representations. arXiv:2209.14855.

[^5]: Cohen, T., & Welling, M. (2016). Group equivariant convolutional networks. ICML.

[^6]: Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021). Geometric deep learning. arXiv:2104.13478.

[^7]: Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K. (2018). Neural ordinary differential equations. NeurIPS.

[^8]: Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention is all you need. NeurIPS.

[^9]: Finn, C., Abbeel, P., & Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. ICML.

[^10]: van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research, 9, 2579–2605.

[^11]: Li, Z., Kovachki, N., Azizzadenesheli, K., et al. (2020). Fourier neural operator for parametric partial differential equations. arXiv:2010.08895.

[^12]: Helwig, J., Zhang, X., Fu, C., et al. (2023). Group equivariant Fourier neural operators for partial differential equations. ICML.

[^13]: Galewsky, J., Scott, R. K., & Polvani, L. M. (2004). An initial-value problem for testing numerical models of the global shallow-water equations. Tellus A.