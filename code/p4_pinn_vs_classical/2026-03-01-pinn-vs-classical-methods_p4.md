---
layout: post
title: "Physics-Informed Neural Networks versus Adaptive Runge-Kutta: Accuracy and Cost on a Forced Linear ODE"
date: 2026-03-01
tags: [scientific-ml, pinns, numerical-methods, method-comparison]
giscus_comments: true
published: true
description: "A PINN and an adaptive RK45 solver are compared on a forced linear ODE with a known exact solution, measuring accuracy and wall-clock cost."
# edited: true
---

## Motivation

The first three posts in this series worked entirely within classical numerical analysis: fast linear algebra in [the Sherman-Morrison post]({{ '/blog/2026/sherman-morrison-ode_p1/' | relative_url }}), stability-aware time integration in [the stiff-ODE post]({{ '/blog/2026/stiff-odes-solver-stability_p2/' | relative_url }}), and polynomial-based surrogate modeling in [the uncertainty-quantification post]({{ '/blog/2026/uncertainty-quantification-surrogate-modeling_p3/' | relative_url }}). A natural next step is to examine physics-informed neural networks (PINNs) {% cite raissi2019physics %}, which are widely represented in the scientific machine learning literature and are frequently presented as a general-purpose replacement for classical PDE and ODE solvers.

This post examines that claim quantitatively, without either dismissing PINNs outright or accepting the prevailing enthusiasm uncritically. As with most comparisons of tools in applied mathematics, the conclusion is conditional: performance **depends on the problem**, and specifying _what it depends on_ is more useful than a general verdict in either direction.

## What a PINN is

A PINN approximates the solution of a differential equation with a neural network $\hat{y}(t; \theta)$, trained not on labeled solution data but on the **residual of the governing equation itself**. The idea of training a network to satisfy a differential equation predates the current terminology; an early formulation appears in {% cite lagaris1998artificial %}. For an ODE $\dot{y} = g(t, y)$, the training loss is built from

$$
\mathcal{L}(\theta) = \underbrace{\frac{1}{N}\sum_{i=1}^N \left(\frac{d\hat{y}}{dt}(t_i;\theta) - g(t_i, \hat{y}(t_i;\theta))\right)^2}_{\text{ODE residual loss}}
\;+\;
\lambda \underbrace{\left(\hat{y}(0;\theta) - y_0\right)^2}_{\text{initial-condition loss}},
$$

evaluated at a set of **collocation points** $t_i$ sampled across the domain, with a penalty weight $\lambda > 0$ on the initial condition. Training minimizes this loss over the network's parameters $\theta$, using automatic differentiation, or closed-form derivatives for sufficiently small networks, to compute $d\hat{y}/dt$ exactly as a function of $\theta$.

A PINN therefore embodies a different computational paradigm from advancing a solution forward in time. Instead of marching through the domain, a PINN solves a **global nonlinear optimization problem** over the entire domain at once, exchanging the guaranteed local error control of a time-stepping method for the flexibility of a mesh-free function approximator.

## A test problem with a known exact solution

To ensure the comparison is well posed, I chose a problem simple enough to admit a closed-form solution, so that both methods are scored against ground truth and not merely against each other: a forced, damped linear ODE representing Newton cooling with an oscillating ambient temperature,

$$
\frac{dy}{dt} = -k\big(y - A\sin(\omega t)\big), \qquad y(0) = y_0,
$$

with $k=2$, $A=1$, $\omega=3$, $y_0 = 0.3$, solved over $t \in [0, 3]$. This is a first-order linear ODE, solvable exactly via an integrating factor: multiplying through by $e^{kt}$ turns the left-hand side into an exact derivative,

$$
\frac{d}{dt}\left(e^{kt} y\right) = kA\, e^{kt}\sin(\omega t),
$$

which integrates in closed form to

$$
y(t) = \left(y_0 + \frac{kA\omega}{k^2+\omega^2}\right)e^{-kt}
+ \frac{kA}{k^2+\omega^2}\Big(k\sin(\omega t) - \omega\cos(\omega t)\Big).
$$

The sign of the homogeneous coefficient follows from the initial condition: the particular solution satisfies $y_p(0) = -kA\omega/(k^2+\omega^2)$, so the coefficient of $e^{-kt}$ must be $y_0 - y_p(0) = y_0 + kA\omega/(k^2+\omega^2)$. Because the exact solution is available, any discrepancy between a numerical method and the truth is unambiguous and cannot be an artifact of an unknown reference.

## Building a minimal PINN without a deep learning framework

Because this is a first-order ODE, only a _first_ derivative of the network output is required, which permits the network and its exact analytic time-derivative to be written in closed form for a single-hidden-layer tanh network,

$$
\hat{y}(t;\theta) = \sum_{i=1}^{H} w_i^{(2)} \tanh\!\big(w_i^{(1)} t + b_i^{(1)}\big) + b^{(2)},
\qquad
\frac{d\hat{y}}{dt} = \sum_{i=1}^{H} w_i^{(2)} w_i^{(1)} \Big(1 - \tanh^2\!\big(w_i^{(1)} t + b_i^{(1)}\big)\Big),
$$

and trained with a standard gradient-based optimizer, with no automatic differentiation framework required. The penalty weight $\lambda$ in the loss above is the constant `50.0` in the code:

```python
import numpy as np
from scipy.optimize import minimize

# Problem definition (as in the full script)
k, A, omega, y0 = 2.0, 1.0, 3.0, 0.3

def rhs(t, y):
    return -k * (y - A * np.sin(omega * t))

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

The classical baseline is `scipy.integrate.solve_ivp` with `RK45`, called once, at tight tolerances (`rtol=1e-10`, `atol=1e-12`).

## Results

The script was run on an AMD Ryzen 7 6800H laptop under WSL2 (Ubuntu 26.04) with Python 3.12.14, NumPy 2.5.3, and SciPy 1.18.1, using single-threaded BLAS. The timings below come from a single run and should be read as indicative of order of magnitude, not as precise benchmarks.

```
Classical RK45: wall=62.658 ms, max err vs exact = 3.33e-10
PINN training: wall=2960.5 ms, final loss=1.425e-03, iters=338, converged=False
PINN: max err vs exact = 1.48e-02
```

<p align="center">
  <img src="/assets/img/posts/solution_comparison_p4.svg" alt="Solution comparison: exact vs classical RK45 vs PINN" style="width: 100%; max-width: 70%; height: auto; display: block; margin: 0 auto;">
</p>

_Figure 1: Exact analytic solution, classical RK45 solution, and PINN prediction. The classical solver is visually exact; the PINN captures the qualitative shape but has visible, non-trivial error._

<p align="center">
  <img src="/assets/img/posts/cost_accuracy_p4.svg" alt="Cost vs accuracy tradeoff scatter plot" style="width: 100%; max-width: 70%; height: auto; display: block; margin: 0 auto;">
</p>

_Figure 2: Accuracy against compute cost, both axes log-scaled. For this problem, the classical solver dominates on both axes simultaneously, being both faster and more accurate. No trade-off arises here; one method is strictly preferable for this specific problem class._

For this problem, the classical solver is not merely faster: its maximum error is smaller by a factor of about $4.5 \times 10^{7}$, **more than seven orders of magnitude**, while its wall-clock time is roughly **47 times shorter**. This outcome is expected. A well-posed, smooth, low-dimensional forward ODE problem is precisely the setting for which classical numerical analysis has been refined over a century. There was no basis for expecting a general-purpose function approximator, trained from scratch by nonlinear optimization, to compete with a purpose-built, convergence-guaranteed adaptive-step solver in that regime.

## When PINNs are the appropriate tool

PINNs are a reasonable tool for a different _class_ of problem from the one above. The settings in which the trade-off shifts include:

- **Inverse and parameter-inference problems.** If the objective is to infer an unknown coefficient in the governing equation from sparse, noisy observations, a PINN can combine the physics residual and the data-fit term into a single loss. Classical approaches, such as a forward solver coupled to an optimizer with adjoint-based gradients, are well established and often very efficient, but they require more problem-specific machinery to set up.
- **High-dimensional PDEs.** Classical mesh-based methods, including finite difference and finite element schemes, suffer their own curse of dimensionality, since cost grows exponentially with spatial dimension. A mesh-free neural-network parameterization avoids discretizing a high-dimensional grid, which is valuable in high dimensions, where classical mesh generation becomes impractical {% cite karniadakis2021physics %}.
- **Irregular geometries and mesh generation cost.** For complex geometries in which generating a good-quality mesh is itself expensive or unreliable, a mesh-free method removes that requirement entirely.
- **Amortized or parametric solves.** If the solution is required for _many_ different parameter settings or boundary conditions, a network trained once and conditioned on those parameters can amortize cost across queries in a manner unavailable to a classical solver re-run from scratch.

None of these characterizations describe the test problem above, and that is deliberate. The comparison is informative because I selected a problem _type_ in which classical methods should be expected to prevail, which avoids constructing a straw comparison in either direction.

## Sources of the residual PINN error

The reasons the PINN underperformed here merit explicit treatment, since they are informative about how these networks behave.

**Training is a non-convex optimization problem.** Unlike a classical solver, which advances with a mathematically guaranteed local error per step, the training loss of a PINN is a highly non-convex function of the network weights. `L-BFGS-B` stopped after 338 iterations with a final loss of approximately $1.4\times 10^{-3}$, well short of zero, and reported that its convergence criteria were not met (`converged=False`). A different random initialization, a different optimizer (Adam followed by a second-order refinement is a common procedure in the literature), or more training iterations may perform better, but no guarantee of convergence to the global optimum exists of the kind available for a well-posed linear solve.

**Spectral bias.** A well-documented phenomenon in the neural-network literature, sometimes termed the "F-principle", is that networks trained by gradient descent tend to fit low-frequency components of a target function considerably faster than high-frequency ones {% cite rahaman2019spectral %}{% cite xu2020frequency %}. For an oscillatory right-hand side such as $\sin(\omega t)$ with $\omega = 3$, this bias works directly against the PINN, and is part of the reason the residual loss plateaus instead of continuing to decrease with further iterations {% cite wang2022when %}.

**Collocation density and network capacity.** With only 60 collocation points and 12 hidden units, the network operates within a small budget by the standards of scientific machine learning. Production PINN implementations typically use hundreds to thousands of collocation points, deeper networks, and considerably longer training runs, all of which would plausibly narrow the accuracy gap, though at a proportionally larger compute cost. For a problem of this simplicity, that additional cost reinforces the cost-accuracy conclusion above.

None of this constitutes a defense of the specific figures. Two gaps should be distinguished. The accuracy gap is largely a training artifact: it reflects non-convex optimization, spectral bias, and a small budget, and more careful training could reduce it. The cost gap is structural: any PINN must solve a global optimization problem with many residual evaluations, whereas the adaptive solver needs only a modest number of right-hand-side evaluations, so the classical solver would remain preferable on this problem under any reasonable amount of additional PINN tuning. Understanding _how_ PINNs are trained is what permits prediction of which harder problems might shift the balance.

## Summary Notes and Highlights

- Benchmarking against a **known exact solution**, and not only against the competing method, is what makes a comparison of this kind trustworthy.
- For smooth, low-dimensional, well-posed forward problems, classical adaptive solvers remain both faster and substantially more accurate, as should be expected.
- PINNs justify their computational cost in a different regime: inverse problems, high-dimensional PDEs, irregular geometries, and amortized multi-query settings. They are not a direct substitute for `solve_ivp`.
- The productive question is not whether method A is superior to method B in the abstract, but what the structure of _this_ problem makes expensive, and which tool is designed to avoid that particular cost.

That question, namely what the structure of a specific problem makes expensive and how to avoid paying for it, has been the unifying theme of this series: exploiting low-rank structure to avoid redundant factorizations, exploiting a solver's stability region to avoid unnecessary small steps, exploiting the smoothness of a QoI to avoid unnecessary sampling, and, in this post, assessing a method by its demonstrated strengths instead of its current prominence.

## References

The PINN framework discussed throughout this post follows {% cite raissi2019physics %}, with earlier precedent in {% cite lagaris1998artificial %}. {% cite karniadakis2021physics %} surveys physics-informed machine learning more broadly, including the inverse-problem and high-dimensional-PDE use cases discussed above. The spectral-bias phenomenon used to explain the PINN's training behavior is documented independently in {% cite rahaman2019spectral %} and {% cite xu2020frequency %}, and {% cite wang2022when %} gives a more technical account of the training pathologies that affect PINN convergence in practice.

{% bibliography --cited --file blog_references %}

---

*Full code for this post is available in [`pinn_demo_p4.py`]({{ '/assets/code/pinn_demo_p4.py' | relative_url }}).*
