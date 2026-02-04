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

Partial differential equations (PDEs) are fundamental to understanding and modeling spatiotemporal dynamics across virtually all scientific domains—from fluid mechanics and weather forecasting to biological systems and materials science. Traditionally, these equations are solved using numerical methods like finite element or spectral methods, which require discretization onto computational grids and can be computationally expensive. 

Recent advances in deep learning have opened new possibilities for data-driven PDE solving. However, most existing neural approaches face critical limitations: they rely on regular grids, struggle with complex geometries, and fail to incorporate known physical symmetries—properties that are intrinsic to the systems they model. **What if we could build neural models that naturally respect the geometry and symmetries of physical laws?**

In this blog post, we explore the work of Knigge et al. (2024)[^1], which introduces a framework for **space-time continuous PDE forecasting using Equivariant Neural Fields (ENFs)**. This approach combines the flexibility of neural fields with the structured inductive biases of equivariant neural networks, enabling accurate and efficient prediction of PDE dynamics on challenging geometries like spheres, tori, and 3D balls.

## The Challenge of PDE Forecasting

Consider the task of forecasting weather patterns, ocean currents, or heat diffusion through materials. In each case, we observe a physical field $$\nu(x, t)$$ that evolves over space $$x$$ and time $$t$$ according to some governing PDE. The challenges are multifold:

1. **Irregular Sampling**: Observations may be sparse and irregularly distributed in space and time
2. **Complex Geometries**: Physical domains may be non-planar (e.g., spherical Earth, periodic boundaries)
3. **Symmetries**: Physical laws remain invariant under certain transformations (rotations, translations)
4. **Continuous Queries**: We need predictions at arbitrary spatial and temporal locations, not just grid points

Traditional grid-based methods like CNNs excel when data is regular and planar, but struggle with irregular sampling and complex geometries. On the other hand, methods that ignore physical symmetries waste modeling capacity learning patterns that should be known *a priori*.

## Background and Motivation

### Neural Fields for Continuous Representations

**Neural Fields** (also called coordinate-based neural networks) are continuous representations that map coordinates to field values:

$$f_\theta: \mathbb{R}^n \rightarrow \mathbb{R}^d$$

where $$\theta$$ represents neural network parameters. A **conditional neural field** extends this by conditioning on a latent code $$z$$:

$$f_\theta(x; z): \mathbb{R}^n \times \mathbb{R}^c \rightarrow \mathbb{R}^d$$

This allows a single network to represent multiple signals by associating different latent codes with different fields.

The key advantages of neural fields include:
- **Grid-agnostic**: Can evaluate at any coordinate, not limited to training grid
- **Continuous**: Natural interpolation between sampled points
- **Memory-efficient**: Compress high-resolution data into compact latent codes

Recent work by Yin et al. (2022)[^2] proposed using conditional neural fields for PDE forecasting by learning dynamics in latent space. However, this approach treats the latent space as an unstructured vector space, ignoring the geometric properties of the underlying physical system.

### The Role of Symmetry in Physics

Many physical laws exhibit **symmetries**—they remain unchanged under certain transformations. For example:

- **Heat diffusion** is rotationally symmetric: rotating the initial temperature distribution yields a rotated solution
- **Navier-Stokes equations** on a periodic domain exhibit translational symmetry
- **Atmospheric flow** on a sphere has rotational symmetry around Earth's axis

Mathematically, if a PDE operator $$\mathcal{N}$$ is equivariant to a group $$G$$ of transformations, then for any group element $$g \in G$$:

$$\mathcal{N}[L_g \nu] = L_g \mathcal{N}[\nu]$$

where $$L_g$$ denotes the action of $$g$$ on the field $$\nu$$. This means: **transforming a solution gives another valid solution**.

**Equivariant neural networks** enforce this property by construction, providing a powerful inductive bias that:
- Improves generalization to unseen transformations
- Increases data efficiency
- Ensures physical consistency

## Method Overview

The proposed framework models PDE solutions as flows in a structured latent space that preserves geometric information. The key insight is to represent each field state $$\nu_t$$ using a latent set:

$$z^\nu_t = \{(p_i, c_i)\}_{i=1}^N$$

where:
- $$p_i$$ is a **pose** encoding geometric position (e.g., in SE(2), SO(3), etc.)
- $$c_i$$ is a **context vector** encoding features

This explicit geometric structure enables equivariance throughout the pipeline.

![Framework Overview]({{ site.baseurl }}/assets/images/framework_overview.png)
*Figure 1: The proposed framework represents PDE states as point clouds in group space and models dynamics via equivariant neural ODEs.*

### Structured Latent Representation

Unlike prior work that uses unstructured latent vectors, this approach represents states with a **point cloud** in the group space. Each element contains:

1. **Geometric information** ($$p_i$$): Lives on the symmetry group manifold
2. **Feature information** ($$c_i$$): Lives in Euclidean space

A group action $$g \in G$$ transforms the latent by acting only on poses:

$$g \cdot z^\nu = \{(g \cdot p_i, c_i)\}_{i=1}^N$$

This structure is crucial for maintaining equivariance.

### Equivariant Neural Fields as Decoders

The decoder $$f_\theta$$ reconstructs field values from latents via **cross-attention** over **bi-invariant attributes**. A bi-invariant attribute $$a(p_i, x)$$ satisfies:

$$a(p_i, x) = a(g \cdot p_i, g \cdot x) \quad \forall g \in G$$

**Example for SE(2) (planar translations and rotations):**

$$a^{SE(2)}_i(x) = R_i^T(x - p_i)$$

where $$p_i = (t_i, R_i)$$ consists of translation and rotation. This relative position is unchanged when both $$p_i$$ and $$x$$ are transformed by the same group element.

**Example for the 2-Torus (periodic boundary conditions):**

$$a^{T^2}_i(x) = \cos(2\pi(x_0 - p_0)) \oplus \cos(2\pi(x_1 - p_1))$$

This respects periodic translations but not rotations.

The decoder then computes:

$$f_\theta(x; z^\nu) = \text{CrossAttention}(Q(a_{:,x}), K(c), V(c; a))$$

where queries depend on bi-invariant attributes, and keys/values depend on contexts. This weight-sharing over geometric equivalence classes guarantees equivariance:

$$f_\theta(g \cdot x; g \cdot z^\nu) = f_\theta(x; z^\nu)$$

### Equivariant Latent Dynamics

To forecast future states, the method learns a **neural ODE** that evolves latents:

$$\frac{dz^\nu}{dt} = F_\psi(z^\nu)$$

The vector field $$F_\psi$$ must be equivariant: $$F_\psi(g \cdot z) = g \cdot F_\psi(z)$$.

This is achieved using **message passing** on the latent point cloud:

**Context update:**

$$c_i^{l+1} = \sum_{j} \phi_{\text{context}}(a^l_{i,j}) c_j^l$$

**Pose update (on the manifold):**

$$p_i^{l+1} = \exp_{p_i^l}\left(\frac{1}{N}\sum_{j} \phi_{\text{pose}}(a^l_{i,j}) \log_{p_i^l}(p_j^l)\right)$$

where $$\log$$ and $$\exp$$ are the logarithmic and exponential maps on the group manifold. This generalizes coordinate-based updates to curved spaces.

**Key property:** The ODE solution commutes with group actions:

$$\phi_t(g \cdot z_0) = g \cdot \phi_t(z_0)$$

meaning: transforming the initial condition then solving gives the same result as solving then transforming.

### Meta-Learning for Efficient Inference

To obtain the initial latent $$z_0^\nu$$ from initial condition $$\nu_0$$, the authors use **meta-learning** instead of slow autodecoding:

1. **Outer loop**: Train shared initialization and decoder parameters $$\theta$$
2. **Inner loop**: For each sample, adapt $$z_0^\nu$$ with just 3-4 gradient steps on reconstruction loss

This provides two benefits:
- **Fast inference**: Only a few optimization steps needed at test time
- **Structured latent space**: Forces latents to organize smoothly around the shared initialization

Visualizing the latent space (via t-SNE) reveals that meta-learning + equivariance produce highly structured representations where temporally contiguous states cluster together—simplifying the job of the neural ODE.

## Experiments and Results

### Experimental Setup

The framework was evaluated on multiple PDEs across diverse geometries:

| **PDE** | **Domain** | **Symmetry** |
|---------|------------|--------------|
| Heat Equation | Plane $$\mathbb{R}^2$$ | SE(2) |
| Heat Equation | Sphere $$\mathbb{S}^2$$ | SO(3) |
| Navier-Stokes | 2-Torus $$\mathbb{T}^2$$ | Periodic translations |
| Shallow-Water | Sphere $$\mathbb{S}^2$$ | Rotation around axis |
| Convection | Ball $$\mathbb{B}^3$$ | SO(3) |

**Training protocol:**
- Datasets generated via numerical solvers (py-pde, Dedalus)
- Disjoint train/test initial conditions
- Evaluation on both seen ($$t_{\text{in}}$$) and unseen ($$t_{\text{out}}$$) time horizons
- Metrics: Mean Squared Error (MSE) on predictions from initial condition only

### Generalization Under Geometric Transformations

**Experiment:** Heat equation on $$\mathbb{R}^2$$ with spatially separated train/test regions.

- **Training data**: Gaussian spikes in left half-plane ($$0 < x_2 < 1$$)
- **Test data**: Gaussian spikes in right half-plane ($$-1 < x_2 < 0$$)

This forces the model to generalize to unseen rotations/translations of initial conditions.

**Results:**

| **Model** | $$t_{\text{in}}$$ Train | $$t_{\text{out}}$$ Train | $$t_{\text{in}}$$ Test | $$t_{\text{out}}$$ Test |
|-----------|---------------------|----------------------|-------------------|---------------------|
| DINo (baseline) | 5.92e-04 | 2.40e-04 | **3.85e-03** | **5.12e-03** |
| Ours (no equivariance) | 6.23e-06 | 4.90e-06 | **2.19e-03** | **5.08e-04** |
| **Ours (SE(2) equivariant)** | **1.18e-05** | **2.53e-05** | **1.50e-05** | **2.53e-05** |

The equivariant model maintains low error on test conditions (different geometric transformations), while baselines fail completely. This demonstrates that **encoding symmetries enables zero-shot generalization to unseen transformations**.

### Robustness and Data Efficiency

**Navier-Stokes on $$\mathbb{T}^2$$** with varying observation rates of initial conditions:

| **Model** | 100% observed | 50% observed | 5% observed |
|-----------|---------------|--------------|-------------|
| FNO | 8.44e-05 | 3.20e-02 | 3.84e-01 |
| G-FNO | **3.15e-05** | 2.32e-02 | 3.40e-01 |
| DINo | 1.11e-02 | 3.74e-02 | 3.94e-02 |
| **Ours (equivariant)** | 1.57e-03 | **5.75e-03** | **3.44e-02** |

While grid-based methods (FNO) excel on fully observed regular grids, **equivariant neural fields maintain performance under severe sparsity** (5% observations). This is critical for real-world applications where sensors provide limited coverage.

**Data efficiency** was evaluated by varying training set size for heat diffusion on $$\mathbb{S}^2$$:

- Non-equivariant model overfits with small datasets
- **Equivariant model generalizes well with only 16 trajectories**
- 4× reduction in data requirements compared to baseline

### Complex Geometries

**Shallow-Water Equations on $$\mathbb{S}^2$$** (global atmospheric flow):

The model handles:
1. **Spherical geometry** (avoiding polar singularities)
2. **Axial symmetry only** (due to Coriolis forces)
3. **Zero-shot super-resolution** (2× spatial upsampling)

**3D Ball** (internally-heated convection in $$\mathbb{B}^3$$):

| **Model** | $$t_{\text{in}}$$ Test | $$t_{\text{out}}$$ Test |
|-----------|-------------------|---------------------|
| DINo | 3.06e-03 | 7.78e-02 |
| **Ours** | **5.99e-04** | **7.97e-03** |

The equivariant approach achieves **10× error reduction** by respecting spherical symmetries in 3D—a challenging geometry where coordinate singularities often cause issues.

## Discussion

### Strengths

1. **Physical consistency**: Models provably respect known symmetries
2. **Data efficiency**: Requires fewer training trajectories than baselines
3. **Geometric flexibility**: Works on planes, spheres, tori, balls
4. **Continuous representation**: Handles irregular sampling and arbitrary queries

### Limitations

1. **Domain knowledge required**: Must specify symmetry group *a priori*
2. **Computational cost**: Global attention in neural fields scales quadratically
3. **Error accumulation**: Long-term autoregressive predictions degrade (common to all learned PDE solvers)
4. **Complexity**: Implementation requires careful handling of group actions and manifold operations

### Comparison to Alternatives

- **Grid-based methods (FNO, CNN)**: Faster on regular grids, but inflexible to geometry/sampling
- **Neural operators**: Theoretically infinite-dimensional, but practically limited to fixed discretizations
- **Other neural fields (DINo)**: Continuous but lack geometric structure

The sweet spot for this approach is **scientific applications with known symmetries, irregular sampling, and complex geometries**—exactly where traditional methods struggle.

## Conclusion

Equivariant Neural Fields offer a principled framework for learning continuous spatiotemporal dynamics while respecting physical symmetries. By explicitly encoding geometry in the latent space and ensuring equivariance throughout the architecture, the method achieves:

- Superior generalization to unseen geometric transformations
- Robustness to sparse observations  
- Data efficiency with limited training samples
- Capability to handle complex non-planar geometries

This work demonstrates that **combining the flexibility of neural fields with the inductive biases of equivariant architectures** creates powerful models for scientific machine learning. As we push toward learning from real-world observational data—which is invariably sparse, irregular, and governed by physical laws—such physics-informed architectures will become increasingly essential.

### Future Directions

- Learning symmetries automatically from data
- Mitigating error accumulation in long rollouts
- Scaling to higher-dimensional systems
- Applications to real-world climate and materials modeling

---

## References

[^1]: Knigge, D. M., Wessels, D. R., Valperga, R., Papa, S., Sonke, J. J., Gavves, E., & Bekkers, E. J. (2024). Space-Time Continuous PDE Forecasting using Equivariant Neural Fields. *arXiv preprint arXiv:2406.06660*.

[^2]: Yin, Y., Kirchmeyer, M., Franceschi, J. Y., Rakotomamonjy, A., & Gallinari, P. (2022). Continuous PDE dynamics forecasting with implicit neural representations. *arXiv preprint arXiv:2209.14855*.
