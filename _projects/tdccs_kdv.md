---
layout: page
title: Central Compact Schemes for the KdV Equation
description: An open-source Python reproduction of the TDCCS third-derivative scheme (Salian, Samala & Ghosh, 2026)
img: assets/img/projects/figure15_ito_gaussian.png
importance: 1
category: research
giscus_comments: false
---

Third-order spatial derivatives appear in every dispersive wave model, and the Korteweg-de Vries (KdV) equation is the canonical example. Salian, Samala and Ghosh (_Numerical Methods for Partial Differential Equations_, 2026, 42:e70060, [doi:10.1002/num.70060](https://doi.org/10.1002/num.70060)) proposed the **third-derivative central compact scheme (TDCCS)**, which evolves node values and cell-center values as independent variables and computes derivatives at each set of points from _both_ grids, with no interpolation at any stage.

The original work uses MATLAB and its code is not publicly available. This project is an independent, open-source Python implementation: it derives the scheme's coefficients from the order conditions rather than transcribing the published tables, verifies them against the paper, and reproduces every figure and table of the paper's numerical section at the paper's own parameters.

## Highlights

- **Coefficients derived symbolically.** 25 of the 26 published coefficient rows satisfy every order condition they claim; the derivation also identified discrepancies in the printed Table 1 and Eqs. (3.5) to (3.7).
- **Truncation-error constants** agree with the paper in every published digit.
- **Spectral and stability analysis** (modified wavenumber, resolving efficiency, CFL bounds) agrees to within $10^{-4}$ or to three or four significant figures.
- **Two-dimensional test (Example 7.5):** TDCCS matches the published errors in every digit at $N = 10$, $15$ and $20$.
- **Nonlinear problems** (soliton collisions and splitting, the zero-dispersion limit, the coupled Ito system) are run at the paper's grids and integration windows and reproduce the structures the paper describes.

## Selected results

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/projects/figure9_double_soliton.png" title="Double soliton collision" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">Example 7.3: a double-soliton collision computed with TDCNCS (top) and TDCCS (bottom), with the space-time surface up to t = 4.</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/projects/figure13_example75.png" title="Two-dimensional linear dispersion" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">Example 7.5: two-dimensional linear dispersion. Solution surfaces (top) and pointwise errors (bottom); TDCCS reproduces the published errors exactly.</div>

<div class="row">
  <div class="col-sm mt-3 mt-md-0">
    {% include figure.liquid loading="lazy" path="assets/img/projects/figure15_ito_gaussian.png" title="Ito coupled system" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">Example 7.6: the coupled Ito system with a Gaussian initial condition. Only the u equation carries a third derivative, so u stays smooth while v steepens into shock-like fronts.</div>

## Implementation

The code is self-contained Python (`numpy`, `scipy`, `sympy`, `matplotlib`) and is written to be read as a tutorial: each example script opens with an explanation of the physical problem and how it maps onto the solver. Computation and plotting are separate. The example scripts solve once and save their results, and a single plotting script redraws every figure from the saved data, because the stable time step scales like $\Delta x^3$ and the finest runs need hundreds of thousands of Runge-Kutta steps. The figures follow the paper's MATLAB conventions, including the turbo colormap and camera angles measured from the published panels.

## Code

The complete code, together with the saved data behind every figure, is on GitHub: [abelokoj.github.io/\_projects/tdccs_project](https://github.com/abelokoj/abelokoj.github.io/tree/main/_projects/tdccs_project). The folder's README gives a suggested reading order and the script-to-figure map.

## Read more

The full derivation, validation and discussion of where the reproduction is exact and where it necessarily diverges are in the blog post: [Reproducing a Central Compact Finite-Difference Scheme for the Korteweg-de Vries Equation]({% post_url 2026-08-17-compact_finite_difference_kdv_p11 %}).
