---
layout: post
title: "An Introduction to Uncertainty Quantification Using Surrogate Modeling"
date: 2026-02-22
tags: [uncertainty-quantification, polynomial-chaos, surrogate-modeling]
giscus_comments: true
published: true
---

## Motivation

I have previously made posts about making a *single* simulation run fast: accelerating the linear algebra ([Sherman-Morrison]({{ '/blog/2026/sherman-morrison-ode_p1/' | relative_url }})) and choosing a solver that is not constrained by numerical stiffness ([stiff ODEs]({{ '/blog/2026/stiff-odes-solver-stability_p2/' | relative_url }})). However, in practice, a single does not adequately capture the user's needs. Real inputs, including material properties, boundary conditions, and flight parameters, are uncertain. As a result, propagating that uncertainty via a simulation to determine its effect on the output (the *quantity of interest* (QoI)) typically requires running the simulation many times.

If a single high-fidelity run takes hours on a cluster, evaluating it tens of thousands of times for a Monte Carlo uncertainty study is infeasible. **Uncertainty quantification (UQ)** exists to close this gap in situations like this. For this reason, we will discuss one of the central tools of uncertainty evaluation/quantification, namely; **surrogate modeling**: which replacing the expensive simulation with an inexpensive and accurate substitute constructed from a modest number of well-chosen runs. This post introduces one of the standard techniques for building such surrogates, **polynomial chaos expansion (PCE)** {% cite wiener1938homogeneous %}{% cite ghanem1991stochastic %}, from first principles using a worked example.

## The setup

Suppose an uncertain input parameter $x$, for example, a normalized material property or geometric tolerance, is distributed $x \sim \mathrm{Uniform}(-1, 1)$, and the simulation output, the quantity of interest, is some function $f(x)$. In real applications, $f$ is an expensive black box, such as a CFD solve or a finite-element structural analysis. Here, to keep the mathematics inspectable, a smooth substitute with a genuinely sharp response feature is used, so that the convergence dynamics reported below are not trivially favorable:

$$
f(x) = \frac{1}{1 + 25x^2}.
$$

The goal of UQ is typically to estimate statistics of the output induced by the uncertainty in the input, most commonly the mean and variance,

$$
\mathbb{E}[f(x)], \qquad \mathrm{Var}[f(x)].
$$

The direct approach is Monte Carlo: draw many samples of $x$, evaluate $f$ at each, and compute sample statistics. It is simple and general, but its error decays only as $O(N^{-1/2})$ in the number of simulation calls $N$, which is prohibitively slow when each call is expensive.

## Polynomial chaos expansion

The principle behind PCE is to expand the *output* as a series in polynomials that are orthogonal with respect to the probability distribution of the *input*,

$$
f(x) \approx \sum_{j=0}^{p} c_j\, \Phi_j(x).
$$

For $x \sim \mathrm{Uniform}(-1,1)$, the correct orthogonal set is the **Legendre polynomials** $\Phi_j = P_j$, satisfying

$$
\int_{-1}^{1} P_i(x) P_j(x)\, dx = \frac{2}{2j+1}\, \delta_{ij}.
$$

Different input distributions call for different polynomial families, a correspondence known as the **Askey scheme** {% cite xiu2002wiener %}: Hermite polynomials for the Gaussian inputs, Laguerre for exponential, and so on. Using an ill-matched polynomial series for a given distribution still converges in principle, but forfeits the fast convergence that makes PCE worthwhile.

Because the basis is orthogonal, the expansion coefficients are simply projections,

$$
c_j = \frac{1}{\|P_j\|^2}\int_{-1}^{1} f(x)\, P_j(x)\, \frac{dx}{2},
$$

where the factor $dx/2$ normalizes the uniform density on $[-1,1]$ to integrate to $1$. In practice, this integral is evaluated numerically by **Gauss-Legendre quadrature**: a small number of carefully chosen evaluation points that integrate polynomials of a given degree *exactly*. For a degree-$p$ polynomial integrand, $\lceil (p+1)/2 \rceil$ quadrature nodes suffice, which is why the coefficient budget below scales only linearly with polynomial order rather than with a separately chosen sample count.

The consequence is that, once the coefficients $c_j$ are known, the mean and variance of the output follow algebraically, with no further sampling:

$$
\mathbb{E}[f] = c_0,
\qquad
\mathrm{Var}[f] = \sum_{j=1}^{p} c_j^2 \, \frac{\|P_j\|^2}{2}.
$$

This is the key structural advantage over Monte Carlo: the *statistics* are a direct algebraic readout of the expansion coefficients rather than a separate sampling process layered on top of the surrogate.

## Implementation

```python
import numpy as np
from numpy.polynomial import legendre as L

def legendre_basis_eval(order, x):
    """Phi[i, j] = P_j(x_i) for orders 0..order."""
    n_terms = order + 1
    Phi = np.zeros((len(x), n_terms))
    for j in range(n_terms):
        c = np.zeros(n_terms)
        c[j] = 1.0
        Phi[:, j] = L.legval(x, c)
    return Phi

def pce_coefficients(order, f):
    n_quad = order + 5
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    fvals = f(nodes)
    Phi = legendre_basis_eval(order, nodes)
    coeffs = np.zeros(order + 1)
    for j in range(order + 1):
        norm_j = 2.0 / (2 * j + 1)
        coeffs[j] = np.sum(weights * fvals * Phi[:, j]) / norm_j
    return coeffs

def pce_eval(coeffs, x):
    return L.legval(x, coeffs)
```

## Surrogate accuracy

Fitting PCE surrogates of increasing polynomial order against the true response function gives the following:

<p align="center">
  <img src="/assets/img/posts/surrogate_fit_p3.svg" alt="PCE surrogate fit vs true function, multiple polynomial orders" style="width: 100%; max-width: 640px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 1: True response $f(x)$ against PCE surrogates of order 2, 4, 8, and 14. Low-order expansions miss the sharp peak near $x=0$; by order 14, the surrogate is visually indistinguishable from the truth.*

The low-order surrogates perform least well precisely where $f$ is sharpest, which shows a deliberate choice of test function. A smoother QoI is likely to converge faster still; a more demanding one was selected here to ensure that the convergence results shown below are not artificially favorable.

## PCE vs. Monte Carlo: convergence comparison

The decisive test is efficiency: for a *fixed budget* of simulation calls, the relevant question is the accuracy each method obtains in estimating the mean and variance.

```python
# "Exact" reference via a very high-order PCE
coeffs_ref = pce_coefficients(40, f)
mean_ref = coeffs_ref[0]
var_ref = np.sum((coeffs_ref[1:]**2) * (2.0/(2*np.arange(1,len(coeffs_ref))+1))/2.0)
# mean_ref = 0.274680, var_ref = 0.081122
```

<p align="center">
  <img src="/assets/img/posts/convergence_p3.svg" alt="Convergence comparison: PCE vs Monte Carlo for mean and variance" style="width: 100%; max-width: 720px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 2: Absolute error in the estimated mean (left) and variance (right) of $f(x)$, as a function of the number of simulation evaluations used, comparing PCE (quadrature-based) against standard Monte Carlo. Both axes are log-scaled.*

The contrast is pronounced. PCE reaches accuracy at the level of machine precision with approximately **15 to 20 simulation evaluations**. Monte Carlo, plotted with its characteristic noisy $O(N^{-1/2})$ decay, remains several orders of magnitude less accurate even after **30,000** evaluations. For a QoI of this smoothness, the advantage of PCE derives from **spectral convergence**: the approximation error declines geometrically, or faster, with polynomial order, in contrast to the fixed $1/\sqrt{N}$ rate of Monte Carlo, which holds regardless of the smoothness of the underlying function.

## Limitations

The efficiency of PCE is not unconditional, and the principal failure modes merit explicit statement:

- **Curse of dimensionality.** A full tensor-product PCE basis grows as $O(p^d)$ in the number of uncertain inputs $d$, which rapidly becomes less favorable than Monte Carlo for problems with many uncertain parameters. Sparse-grid and adaptive basis-selection methods exist specifically to reduce this, at the cost of additional implementation difficulty.
- **Non-smooth QoIs.** If the response has a genuine discontinuity, such as a bifurcation, a phase change, or a contact event, global polynomial formulations converge slowly or exhibit Gibbs-type oscillation; the appropriate remedy is often a localized or piecewise surrogate instead.
- **Correlated or non-standard input distributions** require either a different orthogonal polynomial function set, in accordance with the Askey scheme, or a transform to a standard distribution beforehand. Using an ill-matched basis degrades the convergence rate without producing an outright incorrect answer, which makes the error easy to overlook.

## Beyond mean and variance: sensitivity analysis

A further advantage of constructing a PCE surrogate is that, once it exists, higher-order UQ analyses require additional computation on the surrogate rather than new simulation runs. **Sobol sensitivity indices** {% cite sobol2001global %}, which measures how much of the output fluctuation is attributable to each individual input and to interactions among inputs, can be read directly from the PCE coefficients in the multi-input case, with no additional model evaluations required {% cite sudret2008global %}. For a single-input problem such as the one above, the distinction is not visible, since the single input trivially explains $100\%$ of the variance. In a realistic multi-parameter engineering study, however, this is often a more actionable output than the mean and variance alone: identifying which one or two input parameters dominate the output uncertainty indicates where further experimentation or tighter manufacturing tolerances should be directed.

## Two routes to constructing a PCE surrogate

The projection approach shown above, which uses Gauss quadrature to compute each coefficient via numerical integration, is the classical route and is exact up to quadrature error when the evaluation points may be chosen freely. In some engineering settings, however, a **fixed** set of simulation runs is given, such as a legacy design-of-experiments campaign, and new evaluation points cannot be selected. In that case, PCE coefficients are instead estimated by **least-squares regression**: fitting the coefficients $c_j$ that best match the existing $(x_i, f(x_i))$ data in a least-squares sense, using the same Legendre basis functions. Regression-based PCE recovers the same spectral convergence benefits when the sample points are well-distributed and outnumber the basis terms by a sufficient margin, and it degrades gradually rather than exactly when the data are sparse or unevenly distributed. This makes it a useful, practical alternative when quadrature is unavailable {% cite xiu2010numerical %}.

## Applied implementation notes

Several considerations apply before adopting a PCE library, of which
`chaospy` and `UQpy` are common choices in Python, on a real project:

- **Validate against a held-out sample.** Fit the surrogate on one set of points, then assess its predictions on points excluded from fitting. This is the same discipline applied to any regression or machine-learning model, and for the same reason: a surrogate that reproduces only its training quadrature points is of no practical use.
- **Consider the trade-off betwixt polynomial order and evaluation budget.** Higher order yields speedier convergence *per basis function*, but the number of basis functions itself grows combinatorially with the number of uncertain inputs, so for high-dimensional cases the order must be chosen more conservatively than the single-input example here would suggest.
- **Non-uniform, correlated, or bounded-but-non-uniform inputs** are common in practice and require either selecting the matching Askey-scheme polynomial class or applying an isoprobabilistic transform, such as the Rosenblatt transform, to map the actual input distribution onto a standard one before building the expansion.

## Summary Notes and Highlights

- Surrogate modeling exists to make uncertainty propagation tractable when the underlying simulation is too expensive to evaluate thousands of times.
- Polynomial chaos expansion constructs that surrogate by projecting the QoI onto polynomials orthogonal to the probability distribution of the input. For smooth QoIs, this yields **spectral** convergence in place of the fixed $1/\sqrt{N}$ rate of Monte Carlo.
- The performance improvement is conditional: assumptions regarding dimensionality as well as smoothness both require checking before PCE is preferred over more robust, if slower, sampling-based alternatives.

This progression, comprising fast linear algebra, stability-aware time integration, and efficient uncertainty propagation, constitutes much of the standard toolkit of applied computational science: make each simulation call inexpensive, ensure the solver does not silently demand more calls than necessary, and exercise judgment regarding how many calls are required to answer the question posed.

## References

The polynomial-chaos construction used throughout this post traces back to {% cite wiener1938homogeneous %} for Gaussian-type inputs via Hermite polynomials, with {% cite ghanem1991stochastic %} giving the first systematic engineering application. {% cite xiu2002wiener %} establishes the generalized Askey-scheme correspondence between input distributions and orthogonal algebraic polynomial families used above, and {% cite xiu2010numerical %} gives a thorough textbook treatment of both quadrature- and regression-based parameter calculation. The sensitivity-analysis discussion draws on {% cite sobol2001global %} for the classic Sobol indices and {% cite sudret2008global %}, which shows how those indices can be read directly off PCE coefficients without additional simulation runs.

{% bibliography --cited --file blog_references %}

---

*Full code for this post is available in [`pce_demo_p3.py`]({{ '/assets/code/pce_demo_p3.py' | relative_url }}).*
