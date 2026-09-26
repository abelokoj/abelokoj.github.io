---
layout: post
title: "Reproducing a Central Compact Finite-Difference Scheme for the Korteweg-de Vries Equation"
date: 2026-08-17
tags: [numerical-methods, compact-finite-difference, dispersive-waves, KdV, reproducibility]
giscus_comments: true
---

## Motivation

Third-order spatial derivatives appear in every dispersive wave model, of which the Korteweg-de Vries (KdV) equation is the canonical example. Their accurate approximation matters well beyond KdV itself: the same third-derivative operators arise in Airy-type dispersion terms for internal waves, in certain regularized shock-capturing schemes, and in the dispersive-error budgets that govern long-time wave propagation generally.

Classical compact (Padé-type) schemes, following Lele {% cite lele1992compact %}, obtain spectral-like resolution from narrow stencils at the cost of a matrix inversion at each evaluation. Node-based compact schemes are accurate but lose resolution at short wavelengths. Cell-centered compact schemes improve resolution but require an interpolation step onto the staggered grid, and that interpolation introduces transfer errors which offset part of the resolution gain.

This post reproduces, derives, and validates the scheme proposed by Salian, Samala, and Ghosh {% cite salian2026central %}, published in *Numerical Methods for Partial Differential Equations* (2026, 42:e70060), whose contribution is a third option: retain **both** the node values and the cell-center values as independently evolved variables, and compute derivatives at each set of points using values from *both* grids, with no interpolation at any stage. The authors designate this the **third-derivative central compact scheme (TDCCS)**.

The original work uses MATLAB, and its code is not publicly available. The implementation presented here is therefore an open-source alternative, intended additionally as an entry point for readers approaching compact finite differences for the first time.

Three objectives structure what follows: to derive the coefficients of the scheme rather than transcribe the published tables, to implement the scheme and verify it against the paper's own reported values, and to state explicitly where the reproduction is exact and where it necessarily diverges. The third objective is not incidental. Four of the paper's six numerical examples use first-derivative operators cited from other sources but not written out numerically in the paper itself, so any independent reproduction encounters the same difficulty documented in Section 4 below.

## 1. The three schemes and the basis of comparison

All three schemes approximate $f'''$ at the same order of accuracy, from a pentadiagonal (or tridiagonal, or explicit) implicit relation of the form

$$
\beta f'''_{j-2} + \alpha f'''_{j-1} + f'''_j + \alpha f'''_{j+1} + \beta f'''_{j+2} = \text{(explicit combination of } f \text{ values)}.
$$

They differ only in **the grid from which the explicit right-hand side draws its values**:

| Scheme | RHS uses values at | Requires interpolation |
|---|---|---|
| **TDCNCS** (node-based, Eq. 2.4) | nodes only | no |
| **TDCCCS** (cell-centered, Eq. 2.5) | cell centers only | yes, to obtain the centers |
| **TDCCS** (the paper's new scheme, Eq. 3.1-3.2) | both nodes and centers, evolved together | **no** |

The principle underlying TDCCS is straightforward once stated. Rather than interpolating $f$ onto the half-grid points and then differentiating, which is the step at which TDCCCS forfeits accuracy, the same form of compact-derivative formula is applied to obtain $f'''$ directly at the half-grid points, treating the half-grid values as their own evolved unknowns coupled to the node values. Memory cost approximately doubles. The paper's efficiency argument is that an interpolation solve is replaced by a derivative solve of comparable cost, so the additional expense is in memory rather than computation.

## 2. Deriving the coefficients

Each of these schemes reduces to a Taylor-series matching problem, following Lele's approach {% cite lele1992compact %}. Represent $f$ as a formal Taylor series about the node $x_j$:

$$
f_{j+m} = \sum_{k} D_k \frac{(mh)^k}{k!}, \qquad D_k \equiv f^{(k)}(x_j),
$$

substitute into the difference of the left- and right-hand sides of the scheme, and collect powers of $h$ against each symbolic derivative $D_3, D_4, D_5, \dots$. The coefficient of $D_3$ yields the order-2 condition, which requires only that the scheme approximate $f'''$ consistently. The coefficient of $D_5$ yields the order-4 condition, $D_4$ vanishing automatically by antisymmetry, and so on for higher orders. The derivation was carried out symbolically in `sympy` rather than by hand; see `derive_order_conditions.py`.

```python
h, m = sp.symbols('h m', positive=True)
D = sp.symbols('D0:15')

def taylor_value(offset):
    s = offset * h
    return sum(D[k] * s**k / sp.factorial(k) for k in range(K + 1))

def taylor_deriv3(offset):
    s = offset * h
    return sum(D[k + 3] * s**k / sp.factorial(k) for k in range(K - 2))
```

Applying this procedure to TDCCS (Eq. 3.1) reproduces the paper's order conditions (its Eqs. 3.3-3.4) exactly:

```
coefficient of D[3] : -a + 2*alpha - b + 2*beta - c + 1        # matches Eq. 3.3
coefficient of D[5] : -a/16 + alpha - 13*b/80 + 4*beta - 29*c/80  # matches Eq. 3.4
```

### A discrepancy in the published order conditions above fourth order

Continuing the derivation to sixth order and beyond produces conditions that do not match the paper's printed Eqs. (3.5), (3.6), and (3.7). The derived conditions are

$$
\begin{aligned}
\text{order } 6:\quad & \alpha + 2^{2}\beta \;=\; \frac{3a}{160} + \frac{19b}{160} + \frac{741c}{1120}, \\[4pt]
\text{order } 8:\quad & \alpha + 2^{4}\beta \;=\; \frac{85a}{10752} + \frac{1261b}{10752} + \frac{18589c}{10752}, \\[4pt]
\text{order } 10:\quad & \alpha + 2^{6}\beta \;=\; \frac{31a}{7680} + \frac{211b}{1536} + \frac{42271c}{7680},
\end{aligned}
$$

against printed right-hand sides of $13a/160 + 93b/160 + 2451c/1120$, $205a/2688 + 4069b/2688 + 30025c/2688$, and $671a/7680 + 36991b/7680 + 534991c/7680$ respectively. The order-2 and order-4 conditions, Eqs. (3.3) and (3.4), agree exactly.

Two independent checks indicate that the derivation above is correct and the printed equations are not. First, substituting the paper's own Table 4 coefficients into its own printed Eqs. (3.5) to (3.7) leaves non-zero residuals for every row of order six or higher: TDCCS-E6 gives $1181/7680$, TDCCS-T6 gives $2$, TDCCS-T8 gives $831841/564800$ at order six, and so on. Second, substituting the same coefficients into the derived conditions gives exactly zero for all eight rows of Table 4, at every order each row claims. The coefficients the paper actually uses are therefore consistent with the derivation presented here and inconsistent with the equations printed alongside them.

The most probable explanation is a transcription error affecting the three higher-order equations as printed, rather than an error in the coefficients themselves. Since Table 4 supplies the values used in every figure and reported result, the error does not propagate into any of the paper's numerical output. It is recorded here because a reader attempting to reconstruct the coefficients from Eqs. (3.5) to (3.7) will not obtain Table 4.

### Verification against the published coefficient tables

As a stronger check, all 26 published coefficient rows (Tables 1, 2, and 4) were verified against the exact symbolic order conditions, and the leading truncation-error constant $Q$ in $Q f^{(11)}(x)\, \Delta x^8 + \mathcal{O}(\Delta x^{10})$ was extracted for the eighth-order tridiagonal variant of each scheme, which is the variant used throughout the paper's numerical section:

| Scheme | Derived $Q$ | Paper's stated $Q$ |
|---|---|---|
| TDCNCS (Eq. 2.4) | $3.121917\times10^{-5}$ | $3.12192\times10^{-5}$ |
| TDCCCS (Eq. 2.5) | $6.572523\times10^{-5}$ | $6.57252\times10^{-5}$ |
| TDCCS (Eq. 3.1) | $-2.188201\times10^{-6}$ | $2.1882\times10^{-6}$ (magnitude only) |

The magnitudes agree to every published digit. This constitutes the strongest available evidence that both the derivation and the interpretation of the coefficient tables are correct. The sign of the TDCCS constant is negative in the derivation; the paper reports magnitude alone, so the two are not in conflict.

Of the 26 rows, 25 satisfy every order condition they claim, exactly. The single exception is **TDCNCS-P8** (Table 1), which fails all four of its conditions rather than only the leading one. Substituting $a=160/83$, $b=-5/166$, $c=0$, $\alpha=147/332$, and $\beta=-1/166$ into the order-2 condition gives

$$
1 + 2\alpha + 2\beta - (a+b+c) = \frac{311}{166} - \frac{315}{166} = -\frac{2}{83} \neq 0 ,
$$

with residuals of $-4/83$, $-4/249$, and $-8/3735$ at orders four, six, and eight respectively. A row failing its consistency condition at every order is more consistent with a misprinted coefficient than with a scheme of reduced accuracy, though which coefficient is at fault cannot be determined from the published values alone. The row is not used in any of the paper's computations, which employ the T8 tridiagonal variant throughout Section 7, so the consequence is limited.

## 3. Implementing the operators

On a periodic domain, both the implicit (pentadiagonal) left-hand side and the explicit finite-difference right-hand side of every scheme are circulant matrices. They are therefore constructed as sparse circulant operators and inverted once per grid size, with the factorization cached:

```python
def third_derivative_TDCNCS(f, dx, coeffs):
    a, b, c, alpha, beta, _ = coeffs
    rhs = (a * (roll(f,-2) - 2*roll(f,-1) + 2*roll(f,1) - roll(f,2)) / (2*dx**3)
         + b * (roll(f,-3) - 3*roll(f,-1) + 3*roll(f,1) - roll(f,3)) / (8*dx**3)
         + c * (roll(f,-4) - 4*roll(f,-1) + 4*roll(f,1) - roll(f,4)) / (20*dx**3))
    return pentadiagonal_solve(rhs, alpha, beta)
```

TDCCS is the more demanding case, because Eq. (3.1), which gives derivatives at nodes, and Eq. (3.2), which gives derivatives at centers, are coupled: neither can be solved independently of the other. Establishing which node and center indices correspond in Eq. (3.2) requires care. The equation as printed is centered at the half-grid point $x_{j-1/2}$, so relabelling with $k = j-1$ is necessary before it aligns with a clean circulant pentadiagonal pattern in the array of center values. An initial implementation used the wrong relabelling, with the diagnostic symptom that the third-derivative operator diverged under grid refinement rather than converging, which is characteristic of an index-shift error. After correcting the relabelling, the operator was validated on $f = \sin(kx)$ over a periodic domain:

```
N     max error (TDCCS, 8th order)
20    5.80e-05
40    3.87e-07
80    2.05e-09
160   2.05e-09
```

The error reduction factors between successive refinements are approximately 150 and 189, against the factor of $2^8 = 256$ expected for a formally eighth-order scheme. The observed rates are therefore consistent with eighth-order convergence in the sense that they are of the correct order of magnitude, though they fall somewhat short of the asymptotic factor; the discrepancy is not accounted for here. The error does not decrease between $N=80$ and $N=160$, indicating that some floor has been reached. That floor is not machine precision, which for double-precision arithmetic is of order $10^{-16}$, roughly seven orders of magnitude below the observed value. The more likely explanation is the conditioning of the coupled circulant solve, but this was not investigated further, and the identical values at the two finest grids leave open the possibility of a tabulation error. **Flagged for verification.**

## 4. Reproducibility of the paper's numerical examples

The paper's Section 7 presents six numerical examples. Two of them, **Example 7.1** (one-dimensional linear KdV, $u_t + c^{-2}u_{xxx}=0$) and **Example 7.5** (two-dimensional linear dispersion, $u_t+u_{xxx}+u_{yyy}=0$), involve only the third-derivative operator; no convective first-derivative term is required. These two are reproducible without ambiguity, because every coefficient involved is tabulated in the paper.

The remaining four examples (7.2, 7.3, 7.4, and 7.6) are nonlinear and require a first-derivative compact scheme for the flux term $g(u)_x$. One point of navigation is worth recording for anyone working from the paper directly: several of its captions carry example numbers that do not match the surrounding text. Table 11 and Figure 13 are labelled as belonging to Example 7.6 although they present the two-dimensional linear results of Example 7.5, and Figures 11 and 12 are labelled Example 7.3 although they present the zero-dispersion and top-hat results of Example 7.4. The text itself is consistent; only the captions are affected. The paper states that it uses the existing eighth-order cell-node compact scheme of Lele {% cite lele1992compact %} for TDCNCS, and the existing eighth-order central compact scheme of Liu et al. {% cite liu2013central %} for TDCCS. Neither operator's numerical coefficients are tabulated in the paper. An earlier version of this post substituted a spectral (FFT) first derivative at this point and flagged the substitution explicitly. The coefficients have since been taken directly from the two cited sources and implemented:

- **Lele (1992).** The classical node-only compact first-derivative family. The eighth-order tridiagonal member, used for the TDCNCS convective term, has $\alpha=3/8$, $a=25/16$, $b=1/5$, and $c=-1/80$. These values were verified against Lele's own ladder of Taylor order conditions before use; all four conditions hold exactly, including the order-2 condition $a+b+c = 1+2\alpha = 7/4$.
- **Liu, Zhang, Zhang, and Shu (2013).** Their central compact scheme couples node and cell-center values in the same manner as the present paper's TDCCS couples them for the third derivative, with no interpolation and both grids evolved together. Row CCS-T8 of their Table 2.2, used for the TDCCS convective term, gives $\alpha=-3/20$, $a=2$, $b=-61/50$, $c=-2/25$, and $d=e=0$. These were verified against their Eq. (2.8) order-2 condition before use.

Both operators are implemented in `tdccs_lib.py` as `first_derivative_Lele_CNCS8` and `first_derivative_Liu_CCS8`, and were validated on $f=\sin(kx)$:

```
N     Lele CNCS8 error    Liu CCS8 error (coupled node+center)
20    1.14e-04            3.39e-06
40    4.21e-07            1.53e-08
80    1.62e-09            6.22e-11
160   6.35e-12            3.42e-13
```

The Lele operator reduces error by factors of 271, 260, and 255 across successive refinements, in close agreement with the expected factor of 256. The Liu operator gives factors of 222, 246, and 182, which are of the right order but noticeably more variable; the value at the finest grid in particular is well below $2^8$, plausibly because the error has begun to approach a conditioning-limited floor of the kind observed in Section 3. This variability is reported rather than averaged away.

With both operators in place, every nonlinear example below uses the first-derivative operator the paper cites, on the TDCNCS side using node values only and on the TDCCS side using node and center values evolved together, matching the paper's internal consistency between its third- and first-derivative treatments. What remains reduced relative to the paper in several figures is the grid size $N$, the integration window $T$, or both, for reasons of wall-clock budget: the restriction $\Delta t \sim \mathrm{CFL}\cdot\Delta x^3$ is the binding constraint throughout. Each such reduction is identified where it occurs, and none involves substituting a different equation or a different scheme.

### Fourier and spectral-resolution results: exact

Figures 2 and 3 and Tables 5 and 6 depend only on the modified-wavenumber formulas (Eqs. 2.6, 2.7, and 4.1), which are fully specified in the paper.

<p align="center">
  <img src="/assets/img/posts/figure2_TDCNCS_TDCCCS.png" alt="Modified wavenumber against wavenumber for TDCNCS, TDCCCS, and TDCCCS-CI" style="width: 100%; max-width: 700px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 2: Modified wavenumber against wavenumber for (a) TDCNCS, (b) TDCCCS, and (c) TDCCCS-CI, the last using tenth-order compact interpolation onto the half grid. The divergence of the T4 curve near $\omega \approx 2$ in panel (a) is a genuine property of the scheme rather than an implementation error: its implicit denominator $1+\cos\omega$ vanishes at $\omega=\pi$.*

<p align="center">
  <img src="/assets/img/posts/figure3_TDCCS.png" alt="Modified wavenumber for TDCCS and TDCCS-CI" style="width: 100%; max-width: 700px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 3: (a) The paper's new scheme tracks the exact cubic considerably further in wavenumber than TDCNCS or TDCCCS at the same formal order. (b) The degradation incurred by interpolating onto the half grid instead of evolving it directly, which is the transfer error the paper's design avoids.*

The computed bandwidth-resolving efficiencies match the paper's Tables 5 and 6 to four decimal places:

| Scheme (8th-order tridiagonal) | Paper's $e$ ($\epsilon_t=10^{-3}$) | Computed |
|---|---|---|
| TDCNCS | 0.5018 | 0.5018 |
| TDCCCS | 0.4672 | 0.4673 |
| TDCCS | 0.7828 | 0.7828 |

### Stability analysis: agreement within grid-resolution noise

The maximum eigenvalue magnitude of each spatial operator was computed directly. For TDCCS this was done by constructing the full $2N\times2N$ coupled operator numerically and calling `numpy.linalg.eigvals`, rather than deriving its $2\times2$ Fourier symbol by hand. The result was then combined with the TVDRK3 stability polynomial $R(z) = 1+z+z^2/2+z^3/6$.

<p align="center">
  <img src="/assets/img/posts/figure4_stability.png" alt="Eigenvalues of the spatial operators and the TVDRK3 stability region" style="width: 100%; max-width: 700px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 4: (a) Eigenvalues of the exact, TDCNCS, and TDCCS spatial operators, all purely imaginary as required for a non-dissipative central scheme. (b) The TVDRK3 stability region with each operator's eigenvalues scaled to its own maximum stable CFL number.*

| Quantity | Computed | Paper's value |
|---|---|---|
| max &#124;eigenvalue&#124;, TDCNCS-T8 | 15.19 | 15.157 |
| max &#124;eigenvalue&#124;, TDCCS-T8 | 147.28 | 147.168 |
| TVDRK3 imaginary-axis intercept | 1.7321 | 1.732 |
| CFL bound $\Delta t/\Delta x^3$, TDCNCS | 0.114 | 0.11 |
| CFL bound $\Delta t/\Delta x^3$, TDCCS | 0.0118 | 0.011 |

### Example 7.1, linear KdV: agreement to three or four significant figures

The problem is $u_t + c^{-2}u_{xxx}=0$ on $[0,2\pi]$, with exact solution $u(x,t)=\sin(c(x+t))$.

<p align="center">
  <img src="/assets/img/posts/figure5_example71_c1.png" alt="Example 7.1 with c equal to 1" style="width: 100%; max-width: 700px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 5: Low-wavenumber case ($c=1$, $N=40$). Top: numerical solution (markers) against the exact traveling wave (lines) at five times. Bottom: pointwise error, on a scale of $10^{-12}$, which is near machine precision. TDCCS (right) is approximately twice as accurate as TDCNCS (left) at this resolution, in agreement with the paper's comparison.*

<p align="center">
  <img src="/assets/img/posts/figure6_example71_c8.png" alt="Example 7.1 with c equal to 8" style="width: 100%; max-width: 700px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 6: Higher-wavenumber case ($c=8$, $N=40$). The error scale is now $10^{-3}$ to $10^{-4}$, since the grid barely resolves this wavenumber. The superior resolving power of TDCCS evident in Figure 3 appears here directly as an error roughly ten times smaller.*

*Table 8 ($c=1$), $L^\infty$ error:*

| $N$ | TDCNCS (computed) | TDCNCS (paper) | TDCCS (computed) | TDCCS (paper) |
|---|---|---|---|---|
| 10 | 4.1920e-07 | 4.1920e-07 | 1.1729e-07 | 1.1729e-07 |
| 20 | 1.6088e-09 | 1.6089e-09 | 6.4031e-10 | 6.4028e-10 |

*Table 9 ($c=8$), $L^\infty$ error:*

| $N$ | TDCNCS (computed) | TDCNCS (paper) | TDCCS (computed) | TDCCS (paper) |
|---|---|---|---|---|
| 40 | 1.0796e-03 | 1.1000e-03 | 1.1749e-04 | 1.1796e-04 |
| 80 | 3.4196e-06 | 3.4187e-06 | 9.5506e-07 | 9.5592e-07 |

The computation was run to $N=80$ rather than the paper's $N=160$. Because the CFL restriction scales as $\Delta x^3$, $N=160$ requires approximately $1.6\times10^6$ RK3 steps, which exceeds what a single run of this pure-Python implementation accommodates. The code accepts arbitrary $N$; the limit is a runtime budget rather than a property of the scheme.

### Example 7.5, two-dimensional linear dispersion: agreement to four significant figures

The problem is $u_t+u_{xxx}+u_{yyy}=0$, with exact solution $u(x,y,t)=\sin(x+y+2t)$.

<p align="center">
  <img src="/assets/img/posts/figure13_example75.png" alt="Two-dimensional linear dispersion solution and error surfaces" style="width: 100%; max-width: 700px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 13: TDCNCS solution surface at $t=1$, $N=40$ (top), and the corresponding pointwise-error surface (bottom). The diagonal-stripe pattern in the error follows from sampling a separable solution of the form $\sin(x+y+2t)$ on a tensor-product grid.*

| $N$ | TDCNCS (computed) | TDCNCS (paper) |
|---|---|---|
| 10 | 8.6251e-07 | 8.6153e-07 |
| 15 | 3.2436e-08 | 3.2458e-08 |
| 20 | 3.2038e-09 | 3.2016e-09 |

## 5. The nonlinear examples

Every example in this section uses the operators described in Section 4: Lele's eighth-order cell-node scheme for the TDCNCS convective term, and Liu et al.'s eighth-order central compact scheme for the TDCCS convective term. No spectral substitute remains in any figure. What is still reduced relative to the paper is grid size, integration window, or both, and each reduction is identified in the caption of the figure concerned.

### Single soliton propagation ($\epsilon=1$)

With the dispersion coefficient normalized to unity rather than set to the small $\epsilon$ used elsewhere, this case is computationally inexpensive and permits a full grid-convergence study.

<p align="center">
  <img src="/assets/img/posts/figure7_soliton_example72.png" alt="Single soliton propagation and grid convergence" style="width: 100%; max-width: 700px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 7: (a) and (b) show the TDCNCS solution and pointwise error at three times for $u_t - 3(u^2)_x + u_{xxx}=0$, with exact solution $u(x,t)=-2\,\mathrm{sech}^2(x-4t)$, using Lele's eighth-order cell-node scheme for the convective term. (c) Grid convergence at $t=0.5$; the steep log-log slope reflects the high formal order of both operators involved.*

### Single soliton at small $\epsilon$ (Example 7.3 scaling)

<p align="center">
  <img src="/assets/img/posts/figure8_soliton_illustrative.png" alt="Single soliton at small epsilon, TDCNCS compared with TDCCS" style="width: 100%; max-width: 700px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 8: Results at $\epsilon=5\times10^{-4}$, the scaling used in the paper's Example 7.3. TDCNCS (left) uses Lele's node-only convective scheme; TDCCS (right) uses Liu et al.'s node-and-center convective scheme, evolved alongside the TDCCS node-and-center dispersive scheme, consistent with the paper's stated pairing. The TDCCS pointwise error is approximately half that of TDCNCS at this scale, in the same direction as the paper's comparison. Grid size and integration window remain reduced relative to the paper.*

### Double soliton collision

<p align="center">
  <img src="/assets/img/posts/figure9_double_soliton.png" alt="Double soliton collision" style="width: 100%; max-width: 700px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 9: Two solitons of differing amplitude and speed approaching one another, from the Eq. (7.7) initial condition, computed with TDCNCS and Lele's convective scheme. Panel (b) is a space-time contour in which the faster, taller soliton closes on the slower one. The run uses a reduced grid and time window relative to the paper's $N=100$ and $t$ up to 4, and therefore shows the onset of the approach rather than a complete collision-and-separation cycle.*

### Soliton splitting and the effect of the low-pass filter

<p align="center">
  <img src="/assets/img/posts/figure10_filter_effect.png" alt="Effect of the twelfth-order filter on a splitting soliton" style="width: 100%; max-width: 700px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 10: A single hump, from the Eq. (7.8) initial condition of the paper's triple-soliton-splitting case, beginning to split. (a) Unfiltered TDCNCS develops small-amplitude ripples ahead of the main pulse. (b) The same run with the twelfth-order low-pass filter (Eq. 5.1, Table 7) applied every 20 steps damps them. The effect is present but modest at this reduced scale; a longer integration of the kind the paper reports would be expected to show a starker contrast, consistent with the paper's finding that TDCNCS requires frequent filtering whereas TDCCS requires it only rarely.*

### Zero-dispersion limit and top-hat breakup

<p align="center">
  <img src="/assets/img/posts/figure11_zero_dispersion_illustrative.png" alt="Zero-dispersion limit, framework demonstration only" style="width: 100%; max-width: 700px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 11: **This figure does not demonstrate the phenomenon the paper's Example 7.4 concerns, and should not be read as doing so.** That example resolves a genuine train of fast oscillations as $\epsilon\to0^{+}$, sweeping $\epsilon$ from $10^{-4}$ down to $10^{-7}$ with correspondingly refined grids from $N=100$ to $N=1600$, against a reference solution computed at $N=1000$. That regime requires grid resolution beyond the compute budget available here, since the restriction $\Delta t \sim \Delta x^3$ makes grids of this size prohibitively slow in pure Python. The figure uses $\epsilon=0.05$, larger than the paper's coarsest value by a factor of 500, and shows only the early, still-smooth steepening precursor rather than the oscillatory wave train. It is included so that the code path is complete and documented.*

<p align="center">
  <img src="/assets/img/posts/figure12_tophat_illustrative.png" alt="Top-hat breakup with and without filtering" style="width: 100%; max-width: 700px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 12: Top-hat breakup at reduced scale ($N=200$, $\epsilon=5\times10^{-3}$, against the paper's $N=1000$ and $\epsilon=10^{-4}$). A top-hat initial condition breaks up into a dispersive oscillatory wave train ahead of each discontinuity. (a) Unfiltered. (b) With the twelfth-order filter applied, which visibly reduces both the amplitude and the roughness of the train. Unlike Figure 11, this case does exhibit the phenomenon it is intended to illustrate.*

### The Ito-type coupled system

<p align="center">
  <img src="/assets/img/posts/figure14_ito_trig.png" alt="Ito coupled system with trigonometric initial condition" style="width: 100%; max-width: 700px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 14: The system $u_t - (3u^2+v^2)_x - u_{xxx}=0$, $v_t - 2(uv)_x=0$ with $u(x,0)=v(x,0)=\cos x$. As the paper describes, $u$ in panel (a) remains smooth and dispersive while $v$ in panel (b) develops a considerably steeper, shock-like feature, visible here as the sharp spike at $t=1$.*

<p align="center">
  <img src="/assets/img/posts/figure15_ito_gaussian.png" alt="Ito coupled system with Gaussian initial condition" style="width: 100%; max-width: 700px; height: auto; display: block; margin: 0 auto;">
</p>

*Figure 15: The same system with Gaussian initial data on a wider domain. Both components translate and steepen; the trailing oscillations in $v$ at $t=2$ are consistent with the paper's Figure 15.*

## 6. Code

All code is self-contained Python, requiring `numpy`, `scipy`, `sympy`, and `matplotlib`:

- **`tdccs_lib.py`**: coefficient tables (Tables 1, 2, 3, 4, and 7), modified-wavenumber closures, the periodic circulant solvers for all three third-derivative operators, the first-derivative operators of Lele and of Liu et al., the twelfth-order filter, and TVDRK3.
- **`pub_style.py`**: the shared publication-quality `matplotlib` style, using Computer Modern fonts at 600 dpi with tight bounding boxes, applied by every figure script.
- **`derive_order_conditions.py`**: symbolic Taylor-series derivation of the order conditions for all three schemes.
- **`verify_tables.py`**: verification of every published coefficient row against the derived order conditions, and extraction of the leading truncation-error constants.
- **`fig_fourier_analysis.py`**: Figures 2 and 3, Tables 5, 6, and 7.
- **`fig_stability.py`**: Figure 4 and the Eq. (6.3) CFL bounds.
- **`example_7_1_linear_kdv.py`**: Tables 8 and 9, Figures 5 and 6.
- **`example_7_5_2d_linear.py`**: Table 11 and Figure 13.
- **`example_nonlinear_soliton.py`**: Figures 7 to 12, 14, and 15.

## 7. Summary of reproduction status

| Result | Status |
|---|---|
| Order-condition derivation (Tables 1, 2, 4) | Exact, verified symbolically: 25 of 26 published rows satisfy every condition they claim. Discrepancies identified in printed Table 1 (TDCNCS-P8, fails at all four orders) and in printed Eqs. (3.5) to (3.7) |
| Truncation-error constants | Exact agreement to all published digits: $3.121917\times10^{-5}$, $6.572523\times10^{-5}$, $-2.188201\times10^{-6}$ |
| Figures 2, 3 (modified wavenumber) | Exact |
| Tables 5, 6 (resolving efficiency) | Exact to four decimal places |
| Figure 4, Eq. (6.3) (stability and CFL) | Agreement within grid-resolution noise |
| Tables 8, 9, Figures 5, 6 (linear KdV) | Agreement to three or four significant figures |
| Table 11, Figure 13 (2D linear) | Agreement to four significant figures |
| Figure 7 (soliton, $\epsilon=1$) | Qualitative agreement with grid-convergence check; cited convective operator used |
| Figures 8-10, 14, 15 (soliton scaling, collision, filter, Ito system) | Qualitative agreement at reduced grid size or integration window; cited convective operator used |
| Figure 11 (zero-dispersion limit) | Framework demonstration only; does **not** reach the oscillatory regime of the paper's Example 7.4 |
| Figure 12 (top-hat breakup) | Qualitative agreement at reduced scale |

The nonlinear examples now use the first-derivative operators the paper cites rather than a spectral substitute, so the remaining gap between this reproduction and the paper's Section 7 is one of scale rather than of method. Closing it requires a faster time stepper, whether compiled or FFT-preconditioned, to remove the runtime ceiling that constrained several figures above to smaller $N$ or $T$ than the paper uses. Whether the nonlinear examples would then agree to several significant figures, as Examples 7.1 and 7.5 do here, remains to be established; the qualitative agreement documented above is consistent with that outcome but does not demonstrate it.

## References

The compact finite-difference framework used throughout traces to {% cite lele1992compact %}, which also supplies the eighth-order node-only first-derivative operator used for the TDCNCS convective term. The central compact scheme coupling node and cell-center values is due to {% cite liu2013central %}, whose CCS-T8 row provides the TDCCS convective operator. The scheme reproduced in this post, together with the numerical examples of its Section 7, is that of {% cite salian2026central %}.

{% bibliography --cited --file blog_references %}

---

*Full code for this post is available in [`tdccs_lib.py`]({{ '/assets/code/tdccs_lib.py' | relative_url }}) and the accompanying scripts listed in Section 6.*
