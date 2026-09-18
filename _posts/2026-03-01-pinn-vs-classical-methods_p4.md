---
layout: post
title: "Physics-Informed Neural Networks versus Classical Numerical Methods: A Comparative Assessment of Accuracy and Computational Cost"
date: 2026-03-01
tags: [scientific-ml, pinns, numerical-methods, method-comparison]
giscus_comments: true
published: true
---

## Motivation

The first three posts in this series worked entirely within classical numerical analysis: fast linear algebra, stability-aware time integration, and polynomial-based surrogate modeling. Any treatment of the subject written in 2026 would be incomplete without addressing physics-informed neural networks (PINNs) {% cite raissi2019physics %}, which are widely represented in the scientific machine learning literature and are frequently presented as a general-purpose replacement for classical PDE and ODE solvers.

This post examines that claim quantitatively, rather than either dismissing PINNs outright or accepting the prevailing enthusiasm uncritically. As with most comparisons of tools in applied mathematics, the conclusion is conditional: performance **depends on the problem**, and specifying *what it depends on* is more useful than a general verdict in either direction.

## What a PINN is

A PINN approximates the solution of a differential equation with a neural network $\hat{y}(t; \theta)$, trained not on labeled solution data but on the **residual of the governing equation itself**. For an ODE $\dot{y} = g(t, y)$, the training loss is built from

$$
\mathcal{L}(\theta) = \underbrace{\frac{1}{N}\sum_{i=1}^N \left(\frac{d\hat{y}}{dt}(t_i;\theta) - g(t_i, \hat{y}(t_i;\theta))\right)^2}_{\text{ODE residual loss}}
\;+\;
\lambda \underbrace{\left(\hat{y}(0;\theta) - y_0\right)^2}_{\text{initial-condition loss}},
$$

evaluated at a set of **collocation points** $t_i$ sampled across the domain. Training minimizes this loss over the network's parameters $\theta$, using automatic differentiation, or closed-form derivatives for sufficiently small networks, to compute $d\hat{y}/dt$ exactly as a function of $\theta$.

A PINN therefore embodies a genuinely different computational paradigm from advancing a solution forward in time. Rather than marching through the domain, a PINN solves a **global nonlinear optimization problem** over the entire domain at once, exchanging the guaranteed local error control of a time-stepping method for the flexibility of a mesh-free function approximator.

## A test problem with a known exact solution

To ensure the comparison is well posed, a problem simple enough to admit a closed-form solution was chosen, so that both methods can be scored against ground truth rather than against each other: a forced, damped linear ODE representing Newton cooling with an oscillating ambient temperature,

$$
\frac{dy}{dt} = -k\big(y - A\sin(\omega t)\big), \qquad y(0) = y_0,
$$

with $k=2$, $A=1$, $\omega=3$, $y_0 = 0.3$, solved over $t \in [0, 3]$. This is a first-order linear ODE, solvable exactly via an integrating factor: multiplying through by $e^{kt}$ turns the left-hand side into an exact derivative,

$$
\frac{d}{dt}\left(e^{kt} y\right) = kA\, e^{kt}\sin(\omega t),
$$

which integrates in closed form to

$$
y(t) = \left(y_0 - \frac{kA\omega}{k^2+\omega^2}\right)e^{-kt}
+ \frac{kA}{k^2+\omega^2}\Big(k\sin(\omega t) - \omega\cos(\omega t)\Big).
$$

Because the exact solution is available, any discrepancy between a numerical method and the truth is unambiguous rather than an artifact of an unknown reference.

## Building a minimal PINN without a deep learning framework

Because this is a first-order ODE, only a *first* derivative of the network output is required, which permits the network and its exact analytic time-derivative to be written in closed form for a single-hidden-layer tanh network,

$$
\hat{y}(t;\theta) = \sum_{i=1}^{H} w_i^{(2)} \tanh\!\big(w_i^{(1)} t + b_i^{(1)}\big) + b^{(2)},
\qquad
\frac{d\hat{y}}{dt} = \sum_{i=1}^{H} w_i^{(2)} w_i^{(1)} \Big(1 - \tanh^2\!\big(w_i^{(1)} t + b_i^{(1)}\big)\Big),
$$

and trained with a standard gradient-based optimizer, with no automatic differentiation framework required:

```python
import numpy as np
from scipy.optimize import minimize

n_hidden = 12

def unpack(theta):
    w1 = theta[0:n_hidden]
    b1 = theta[n_hidden:2*n_hidden]
    w2 = theta[2*n_hidden:3*n_hidden]
    b2 = theta[3*n_hidden]
    return w1, b1, w2, b2

def y_hat(t, theta):
    w1, b1, w2, b2 = unpack(theta)
    z = np.outer(t, w1) + b1
    return np.tanh(z) @ w2 + b2

def dy_hat_dt(t, theta):
    w1, b1, w2, b2 = unpack(theta)
    z = np.outer(t, w1) + b1
    dh = (1 - np.tanh(z)**2) * w1     # d/dt[tanh(w1 t + b1)]
    return dh @ w2

def loss(theta, t_colloc):
    y = y_hat(t_colloc, theta)
    dy = dy_hat_dt(t_colloc, theta)
    residual = dy - rhs(t_colloc, y)               # ODE residual
    ic_pred = y_hat(np.array([0.0]), theta)[0]
    return np.mean(residual**2) + 50.0 * (ic_pred - y0)**2

t_colloc = np.linspace(0, 3.0, 60)
theta0 = 0.5 * np.random.default_rng(42).standard_normal(3*n_hidden + 1)
result = minimize(loss, theta0, args=(t_colloc,), method="L-BFGS-B",
                   options=dict(maxiter=4000, ftol=1e-14, gtol=1e-12))
```

The classical baseline is `scipy.integrate.solve_ivp` with `RK45`, called
once, at tight tolerances.

## Results

```
Classical RK45:  wall = 15.9 ms,   max error vs. exact = 3.33e-10
PINN training:   wall = 1022.4 ms, max error vs. exact = 1.94e-02, iters=332
```

<p align="center">
  <img src="/assets/img/posts/solution_comparison_p4.svg" alt="Solution comparison: exact vs classical RK45 vs PINN" style="width: 100%; max-width: 70%; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 1: Exact analytic solution vs. classical RK45 vs. PINN prediction. The classical solver is visually exact; the PINN captures the qualitative shape but has visible, non-trivial error.*

<p align="center">
  <img src="/assets/img/posts/cost_accuracy_p4.svg" alt="Cost vs accuracy tradeoff scatter plot" style="width: 100%; max-width: 70%; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 2: Accuracy vs. compute cost, both axes log-scaled. For this problem, the classical solver dominates on both axes simultaneously, being both faster and more accurate. No trade-off arises here; one method is strictly preferable for this specific problem class.*

For this problem, the classical solver is not merely faster: it is **more than seven orders of magnitude more accurate**, while requiring roughly **65 times less wall-clock time**. This outcome is expected. A well-posed, smooth, low-dimensional forward ODE problem is precisely the setting for which classical numerical analysis has been refined over a century. There was no basis for expecting a general-purpose function approximator, trained from scratch by nonlinear optimization, to compete with a purpose-built, convergence-guaranteed adaptive-step solver in that regime.

## When PINNs are the appropriate tool

PINNs are a reasonable tool for a different *class* of problem from the
one above. The settings in which the trade-off genuinely shifts include:

- **Inverse and parameter-inference problems.** If the objective is to infer an unknown coefficient in the governing equation from sparse, noisy observations, a PINN can combine the physics residual and the data-fit term into a single unified loss, which is considerably more cumbersome to construct with a classical forward solver plus a separate optimization loop.
- **High-dimensional PDEs.** Classical mesh-based methods, including finite difference and finite element schemes, suffer their own curse of dimensionality, since cost grows exponentially with spatial dimension. A mesh-free neural-network parameterization avoids discretizing a high-dimensional grid, which is genuinely valuable beyond roughly four to six spatial dimensions, where classical mesh generation becomes impractical {% cite karniadakis2021physics %}.
- **Irregular geometries and mesh generation cost.** For complex geometries in which generating a good-quality mesh is itself expensive or unreliable, a mesh-free method removes that requirement entirely.
- **Amortized or parametric solves.** If the solution is required for *many* different parameter settings or boundary conditions, a network trained once and conditioned on those parameters can amortize cost across queries in a manner unavailable to a classical solver re-run from scratch.

None of these characterizations describe the test problem above, which is precisely the point. The comparison is informative only because a problem *type* was selected in which classical methods should be expected to prevail, so as to avoid constructing a straw comparison in either direction.

## Sources of the residual PINN error

The reasons the PINN underperformed here merit explicit treatment, rather than being left as an unexplained figure, since they are themselves informative about how these networks behave.

**Training is a non-convex optimization problem.** Unlike a classical solver, which advances with a mathematically guaranteed local error per step, the training loss of a PINN is a highly non-convex function of the network weights. `L-BFGS-B` converged to a local minimum with residual loss of approximately $4\times 10^{-3}$ rather than driving it to zero. A different random initialization, a different optimizer (Adam followed by a second-order refinement is a common procedure in the literature), or more training iterations may perform better, but no guarantee of convergence to the global optimum exists of the kind available for a well-posed linear solve.

**Spectral bias.** A well-documented phenomenon in the neural-network literature, sometimes termed the "F-principle", is that networks trained by gradient descent tend to fit low-frequency components of a target function considerably faster than high-frequency ones {% cite rahaman2019spectral %}{% cite xu2020frequency %}. For an oscillatory right-hand side such as $\sin(\omega t)$ with $\omega = 3$, this bias works directly against the PINN, and is part of the reason the residual loss plateaus rather than continuing to decrease with further iterations {% cite wang2022when %}.

**Collocation density and network capacity.** With only 60 collocation points and 12 hidden units, the network operates within a small budget by the standards of scientific machine learning. Production PINN implementations typically use hundreds to thousands of collocation points, deeper networks, and considerably longer training runs, all of which would plausibly narrow the accuracy gap, though at a proportionally larger compute cost. That additional cost reinforces rather than undermines the cost-accuracy conclusion above for a problem of this simplicity.

None of this constitutes a defense of the specific figures: the classical solver remains decisively preferable on this problem, and would remain so under any reasonable amount of additional PINN tuning. The narrower point is that the gap is not a fundamental limitation but a direct consequence of *how* PINNs are trained, and understanding that mechanism is what permits prediction of which harder problems might shift the balance.

## Summary Notes and Highlights

- Benchmarking against a **known exact solution**, rather than against the competing method alone, is what makes a comparison of this kind trustworthy rather than an arbitrary choice between two opaque procedures.
- For smooth, low-dimensional, well-posed forward problems, classical adaptive solvers remain both faster and substantially more accurate. This should not be surprising, and treating it as a surprising result would itself indicate a problem with how the comparison was constructed.
- PINNs justify their computational cost in a different regime: inverse problems, high-dimensional PDEs, irregular geometries, and amortized multi-query settings, rather than as a direct substitute for `solve_ivp` {% cite lagaris1998artificial %}.
- The productive question is not whether method A is superior to method B in the abstract, but what the structure of *this* problem makes expensive, and which tool is designed to avoid that particular cost.

That question, namely what the structure of a specific problem makes expensive and how to avoid paying for it, has been the unifying theme of this series: exploiting low-rank structure to avoid redundant factorizations, exploiting a solver's stability region to avoid unnecessary small steps, exploiting the smoothness of a QoI to avoid unnecessary sampling, and, in this post, relying on an assessment of a method's demonstrated strengths rather than on its current prominence.

## References

The PINN framework discussed throughout this post follows {% cite raissi2019physics %}, with earlier precedent in {% cite lagaris1998artificial %}. {% cite karniadakis2021physics %} surveys physics-informed machine learning more broadly, including the inverse-problem and high-dimensional-PDE use cases discussed above. The spectral-bias phenomenon used to explain the PINN's training behavior is documented independently in {% cite rahaman2019spectral %} and {% cite xu2020frequency %}, and {% cite wang2022when %} gives a more technical account of the training pathologies that affect PINN convergence in practice.

{% bibliography --cited --file blog_references %}

---

*Full code for this post is available in [`pinn_demo_p4.py`]({{ '/assets/code/pinn_demo_p4.py' | relative_url }}).*
