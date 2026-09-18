---
layout: post
title: "Accelerating ODE Solvers with the Sherman-Morrison Formula: A Case Study in Sustainable Aviation Modeling"
date: 2026-02-01
tags: [numerical-methods, linear-algebra, dynamical-systems, aerospace]
giscus_comments: true
published: true
---


## Motivation

Sustainable Aviation Fuel (SAF) blending studies, fuel-burn optimization, and mission-performance models share a common computational bottleneck: each entails integrating a system of ordinary differential equations (ODEs) describing the aircraft’s state (mass, velocity, altitude, and various internal dynamic modes) over the course of a flight. Because aircraft mass decreases continuously as fuel burns, the matrix governing this dynamics is not fixed; it drifts over time.

For _stiff_ or tightly coupled formulations, **implicit** time-stepping schemes are the standard choice, since explicit schemes would demand impractically small time steps for stability (a topic treated in more depth in [the next post in this series]({{ '/blog/2026/stiff-odes-solver-stability_p2/' | relative_url }})). Implicit schemes, however, require solving a linear system at every time step. Implemented directly, this means refactoring an $n \times n$ matrix from scratch at each step: an $O(n^3)$ operation repeated hundreds or thousands of times over a full flight-mission simulation.

This post examines a case in which the cost is avoidable. When the time-varying part of the system matrix is **low rank**, as occurs when a single scalar quantity, such as remaining fuel mass, couples inside the dynamics, the **Sherman-Morrison formula** permits the system matrix to be factorized once and that factorization to be reused for the entire simulation, reducing the per-step cost from $O(n^3)$ to $O(n^2)$.

This post presents a generalized reconstruction of a numerical approach used during a graduate research internship at Lawrence Livermore National Laboratory, in which the technique accelerated an ODE solver for a sustainable aviation modeling application. The system, data, and code below constitute a self-contained synthetic reconstruction prepared for this post; none of the material reproduces proprietary code or data.

## Setting up the problem

Consider a linear time-varying ODE system:

$$
\frac{dy}{dt} = A(t), y + b, \qquad A(t) = A_0 + c(t), u v^\top .
$$

Here $y(t) \in \mathbb{R}^n$ is the state vector (velocity perturbations, structural modes, and thermal states, depending on the quantities the model tracks), $A_0$ is a fixed baseline dynamics matrix, and $c(t)uv^\top$ is a rank-one correction whose direction ($u, v \in \mathbb{R}^n$) remains fixed while its strength $c(t)$ evolves smoothly over the flight. Physically, $c(t)$ acts as a fuel-mass-dependent coupling value: as fuel is consumed, the aircraft’s mass distribution moves, and that shift enters the dynamics matrix along a fixed structural direction, scaled by a coefficient that decays over the flight horizon.

Discretizing with **backward (implicit) Euler** at step size $h$,

$$
y_{k+1} = y_k + h\left(A(t_{k+1}), y_{k+1} + b\right)
\quad\Longrightarrow\quad
\big(I - hA_{k+1}\big), y_{k+1} = y_k + hb,
$$

and substituting $A_{k+1} = A_0 + c_{k+1}, uv^\top$ with the **fixed** matrix $B_0 := I - hA_0$, gives

$$
M_{k+1}y_{k+1} = y_k + hb,
\qquad
M_{k+1} = B_0 - hc_{k+1}uv^\top.
$$

The key structural observation is that $M_{k+1}$ is always a rank-one perturbation of the same fixed matrix $B_0$: the perturbation strength changes at every step, whereas $B_0$ does not.

## The Sherman-Morrison formula

For an invertible matrix $B \in \mathbb{R}^{n\times n}$ and vectors $u, v \in \mathbb{R}^n$ such that $B + \alpha uv^\top$ remains invertible, the Sherman-Morrison formula gives the inverse of the rank-one update directly, without a fresh factorization:

$$
\left(B + \alpha, uv^\top\right)^{-1}
= B^{-1} - \frac{\alpha, B^{-1}u, v^\top B^{-1}}{1 + \alpha, v^\top B^{-1}u}.
$$

**Derivation sketch.** Write $B^{-1}u = p$ and $v^\top B^{-1} = q^\top$, and let $\sigma = 1 + \alpha v^\top B^{-1}u = 1 + \alpha q^\top u$. Multiplying the right-hand side above by $\left(B + \alpha uv^\top\right)$ and expanding term by term,

$$
\left(B^{-1} - \frac{\alpha, p q^\top}{\sigma}\right)\left(B + \alpha uv^\top\right)
= I + \alpha B^{-1} u v^\top - \frac{\alpha}{\sigma}, p q^\top B

* \frac{\alpha^2}{\sigma}, p q^\top u v^\top .
$$

Using $q^\top B = v^\top$ and $q^\top u = v^\top B^{-1}u = (\sigma - 1)/\alpha$, the last two terms combine to $-\dfrac{\alpha}{\sigma}pv^\top\!\left(1 + \alpha \dfrac{\sigma-1}{\alpha}\right) = -\alpha pv^\top$, which exactly cancels the second term $\alpha B^{-1}uv^\top = \alpha pv^\top$. What remains is the identity matrix, confirming the formula.

**Why this computationally important.** Once $B_0^{-1}$ has been computed, at a cost of one $O(n^3)$ factorization, applying the formula for a new value of $\alpha = -hc_{k+1}$ costs only

* two matrix-vector products, $B_0^{-1}u$ and $v^\top B_0^{-1}$: $O(n^2)$;
* one rank-one outer-product update: $O(n^2)$.

No refactorization is required. The $O(n^3)$ cost is paid exactly once, at the start of the simulation, rather than at every one of the $T$ time steps. For a simulation with $T$ steps, this reduces an $O(Tn^3)$ algorithm to an $O(n^3 + Tn^2)$ one, a substantial saving whenever $T$ is large relative to $n$.

## Implementation

```python
import numpy as np

def solve_naive(A0, u, v, b, y0, h, steps, T):
    """Refactor the system matrix from scratch at every step: O(n^3) per step."""
    n = len(y0)
    y = y0.copy()
    traj = [y.copy()]
    for k in range(steps):
        t = k * h
        c = fuel_burn_coeff(t, T)
        Ak = A0 + c * np.outer(u, v)
        M = np.eye(n) - h * Ak
        y = np.linalg.solve(M, y + h * b)
        traj.append(y.copy())
    return np.array(traj)

def solve_sherman_morrison(A0, u, v, b, y0, h, steps, T):
    """Factor B0 once; apply a rank-1 Sherman-Morrison update every step: O(n^2) per step."""
    n = len(y0)
    B0 = np.eye(n) - h * A0
    B0_inv = np.linalg.inv(B0)            # ONE O(n^3) factorization
    y = y0.copy()
    traj = [y.copy()]
    for k in range(steps):
        t = k * h
        c = fuel_burn_coeff(t, T)
        alpha = -h * c
        Binv_u = B0_inv @ u
        v_Binv = v @ B0_inv
        denom = 1.0 + alpha * (v @ Binv_u)
        M_inv = B0_inv - (alpha / denom) * np.outer(Binv_u, v_Binv)
        y = M_inv @ (y + h * b)           # O(n^2) per step
        traj.append(y.copy())
    return np.array(traj)
```

The `fuel_burn_coeff` function models the decaying bond strength as fuel is consumed over the flight horizon $T$:

$$
c(t) = c_0\left(1 - \frac{t}{T}\right), \qquad c_0 = 2.
$$

```python
def fuel_burn_coeff(t, T, c0=2.0):
    return c0 * (1.0 - t / T)
```

## Correctness check

Before the speedup can be relied upon, both methods must be confirmed to produce the same trajectory up to floating-point round-off. Running both solvers on an identical $30$-dimensional system over $400$ steps gives a maximum absolute deviation between the two trajectories of $2.277 \times 10^{-11}$, a magnitude consistent with floating-point round-off rather than with a numerical discrepancy. The two methods are algebraically equivalent, as expected, since Sherman-Morrison is an exact identity rather than an approximation.

<p align="center">

  <img src="/assets/img/posts/trajectory_p1.svg" alt="Implicit Euler trajectory using the Sherman-Morrison solver" style="width: 90%; max-width: 700px; height: auto; display: block; margin: 0 auto;">

</p>

Figure 1: Trajectories of the first three state components under the Sherman-Morrison-accelerated implicit Euler scheme.

## Benchmark: measured performance

Wall-clock time for both approaches was measured across system sizes $n \in {20, 40, 80, 160, 320}$, running $150$ implicit Euler steps for each:

```
n=  20  naive=0.0041s  sherman-morrison=0.0021s  speedup=1.97x
n=  40  naive=0.0070s  sherman-morrison=0.0027s  speedup=2.64x
n=  80  naive=0.0158s  sherman-morrison=0.0052s  speedup=3.05x
n= 160  naive=0.0621s  sherman-morrison=0.0121s  speedup=5.12x
n= 320  naive=0.3402s  sherman-morrison=0.0752s  speedup=4.52x
```

<p align="center">
  <img src="/assets/img/posts/timing_p1.svg" alt="Runtime comparison: naive refactoring vs Sherman-Morrison" style="width: 90%; max-width: 700px; height: auto; display: block; margin: 0 auto;">
</p>

Figure 2: Wall-clock time for 150 implicit Euler steps, naive re-solve vs.
Sherman-Morrison rank-one update, across system dimension $n$.

The measured speedup increases with $n$, consistent with the $O(n^3)$ against $O(n^2)$ per-step scaling. The exact multiplier depends on implementation details, including the BLAS backend, cache behavior, and the number of steps run; at small $n$, the fixed overhead of Python-level bookkeeping partially masks the asymptotic advantage. Larger gains appear either at greater $n$ or over longer time horizons, where the one-time $O(n^3)$ factorization is amortized over more steps.

## Conditions for applicability

The Sherman-Morrison structure is common in practice, but it is not
universal. It applies cleanly when:

- The time-varying part of the system matrix is genuinely **low rank**  (rank one, or rank $k$ via the generalized Woodbury identity below), and 
- The **direction** of the perturbation is fixed, with only its **magnitude** changing over time, as with a single scalar physical quantity, such as fuel mass or temperature-dependent stiffness, or a single damaged structural element.

The formula offers no advantage if the entire matrix changes at every step with no shared low-rank structure. In that regime, the available options are per-step refactorization or the exploitation of a different structure entirely, such as sparsity, banded matrices, or iterative Krylov methods.

## Asymptotic complexity

The asymptotic comparison is set out explicitly below, since the argument for this technique rests on it: for each step, the naive approach requires a fresh $O(n^3)$ factorization, while the Sherman-Morrison approach uses one initial $O(n^3)$ factorization and then $O(n^2)$ work per step.

| Operation | Naive re-solve | Sherman-Morrison |
|---|---|---|
| One-time setup | - | $O(n^3)$ (factor $B_0$) |
| Cost per time step | $O(n^3)$ (refactor $M_k$) | $O(n^2)$ (rank-one update) |
| Total over $T$ steps | $O(Tn^3)$ | $O(n^3 + Tn^2)$ |
| Break-even point | - | favorable once $T \gtrsim n$ |

The break-even condition follows directly: the one-time $O(n^3)$ factorization is worth paying only if it is amortized over a sufficient number of steps. For a small number of time steps on a large system, the naive approach stays competitive; for the long time horizons typical of flight-mission simulations, which involve thousands of steps, the Sherman-Morrison approach is decisively faster.

## A note on the computational stability.

The Sherman-Morrison formula has a denominator, $\sigma := 1 + \alpha, v^\top B^{-1} u$, and, as with any expression of this form, it loses accuracy when that term approaches zero. This happens mostly when the rank-one update pushes the perturbed matrix toward singularity: the classical matrix-determinant lemma gives $\det!\left(B + \alpha uv^\top\right) = \sigma \det(B)$, so $\sigma \to 0$ is equivalent to the perturbed system becoming singular. Physically, this corresponds to the fuel-mass-driven coupling term growing large enough to push part of the system toward an unstable or degenerate configuration.

This condition is worth guarding against explicitly:

```python
denom = 1.0 + alpha * (v @ Binv_u)
if abs(denom) < 1e-10:
    # Fall back to a direct solve for this step, or flag for inspection --
    # the rank-1 update is close to singular here.
    ...
```

For well-behaved physical systems in which the interaction coefficient stays bounded well away from the singular states of the matrix, as holds for the smoothly decaying fuel-burning coefficient used here, the condition rarely arises in practice. It is nonetheless an edge case that is inexpensive to test for and costly to diagnose if left unguarded.

## Extending to rank-$k$ updates: the Woodbury identity

Real systems sometimes involve more than one time-varying scalar coupling within the dynamics: for instance, both a fuel-mass term and a temperature-dependent stiffness term, each having its own direction. The Sherman-Morrison formula generalizes directly to this case via the **Woodbury matrix identity**:

$$
\left(B + UCV^\top\right)^{-1}
= B^{-1} - B^{-1}U\left(C^{-1} + V^\top B^{-1} U\right)^{-1} V^\top B^{-1},
$$

where $U, V \in \mathbb{R}^{n\times k}$ collect one column per rank-one term and $C \in \mathbb{R}^{k \times k}$ is a small coefficient matrix. As long as $k \ll n$, the same core argument holds: the expensive $n \times n$ factorization is still required only once, and the per-step update cost is dominated by inverting the small $k \times k$ matrix $C^{-1} + V^\top B^{-1}U$, which is inexpensive whenever the number of independently changing coupling terms remains small relative to the size of the full system.

## Summary Notes and Highlights

- Implicit ODE solvers incur a per-step linear-algebra cost; identifying low-rank structure in the evolution of the system matrix allows that cost to be paid once rather than at every step.
- The Sherman-Morrison formula, together with its rank-$k$ generalization in the Woodbury identity, applies whenever a simulation involves a fixed baseline matrix with a small time-varying correction. The same pattern occurs well beyond aviation: recursive least squares, Kalman filtering, and quasi-Newton optimization all rely on this identity {% cite hager1989updating %}.
- Performance optimization should always be validated against a naive reference implementation before relying on the speed-up, since a result obtained quickly but incorrectly is of less value than a correct result obtained slowly.

In the [next post]({{ '/blog/2026/stiff-odes-solver-stability_p2/' | relative_url }}), I turn to the obstacle that implicit solvers exist to address in the first place: numerical stiffness, and the reasons explicit methods break down outright on certain dynamical systems regardless of how much rank structure is available for exploitation.

## References

The rank-one and rank-$k$ inverse-update identities used throughout this post trace back to {% cite sherman1950adjustment %} and {% cite woodbury1950inverting %}; {% cite hager1989updating %} surveys the wider family of matrix-inverse update formulas and their use in optimization, while {% cite golub2013matrix %} and {% cite datta2010numerical %} give standard treatments of rank-one updates in the context of numerical linear algebra for time-stepping schemes.

{% bibliography --cited --file blog_references %}

---

*Full code for this post is available in [`sherman_morrison_demo_p1.py`]({{ '/assets/code/sherman_morrison_demo_p1.py' | relative_url }}).*
