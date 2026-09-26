# TDCCS Reproduction: Central Compact Finite-Difference Schemes for the KdV Equation

Open-source Python reproduction of

> L. V. Salian, R. Samala, and D. Ghosh, "Central Compact Finite-Difference Scheme With High Spectral Resolution for KdV Equation," *Numerical Methods for Partial Differential Equations*, 2026, 42:e70060. doi:10.1002/num.70060

The original work uses MATLAB and its code is not publicly available.

## Reading the code

The six `example_7_*.py` scripts are written to be read as a self-study tutorial, not merely run. Each begins with a long header explaining the physical problem, why that problem is a useful test, and how it maps onto the general form the solvers expect, and the body is commented step by step. No prior exposure to computational fluid dynamics or to compact finite-difference schemes is assumed.

A reasonable reading order:

1. `example_7_1_linear_kdv.py`: The simplest case. One equation, one spatial derivative, an exact solution to compare against.
2. `nonlinear_kdv.py`: The shared machinery behind the nonlinear examples. Explains what each term of the PDE does physically, why the time-step is so restrictive, and how the two schemes differ.
3. `example_7_2_soliton.py`: The first nonlinear problem, and the clearest illustration of what a soliton is.
4. `example_7_3_solitons.py`: The soliton collisions and splitting, plus the role of the low-pass filter.
5. `example_7_5_2d_linear.py`: How a one-dimensional scheme extends to two dimensions.
6. `example_7_4_zero_dispersion.py` and `example_7_6_ito_system.py`: The two hardest cases: a singular limit, and a coupled system.

`tdccs_lib.py` holds the numerical operators themselves. Its docstrings explain what a compact scheme is, what makes the TVDRK3 time stepper suitable for time marching, and why a low-pass filter is needed at all.

## Requirements

```
numpy  scipy  sympy and  matplotlib
```

## Figure output

Every figure is written in three formats by a single shared helper, `savefig_all` in `pub_style.py`:

| Format | Purpose |
|---|---|
| `.png` | 600 dpi raster, for web embedding |
| `.pdf` | vector, for LaTeX inclusion |
| `.svg` | vector, for web embedding and further editing (if needed) |

All figures are written to `figs/`, resolved relative to `pub_style.py` instead of the current working directory. This is done so that the scripts may be run from any location. To change the set of formats globally, edit `FORMATS` in `pub_style.py`; to change it for one call, pass `formats=(...)` to `savefig_all`.

```python
from pub_style import apply_style, savefig_all, FIGDIR, cm_x, cm_y

apply_style()
fig, ax = plt.subplots()
...
savefig_all(fig, "my_figure")            # -> figs/my_figure.{png,pdf,svg}
savefig_all(fig, "my_figure", outdir=OUT, close=True)
```

## Scripts

| Script | Reproduces | Approximate runtime |
|---|---|---|
| `derive_order_conditions.py` | symbolic order conditions for all three schemes | seconds |
| `verify_tables.py` | every published coefficient row; truncation constants | seconds |
| `fig_fourier_analysis.py` | Figures 2, 3; Tables 5, 6, 7 | under a minute |
| `fig_stability.py` | Figure 4; Eq. (6.3) CFL bounds | under a minute |
| `example_7_1_linear_kdv.py` | **Example 7.1**: Figures 5, 6; Tables 8, 9 | long |
| `example_7_2_soliton.py` | **Example 7.2**: Figure 7; Table 10 | moderate to long |
| `example_7_3_solitons.py` | **Example 7.3**: Figures 8, 9, 10 | long |
| `example_7_4_zero_dispersion.py` | **Example 7.4**: Figures 11, 12 | very long |
| `example_7_5_2d_linear.py` | **Example 7.5**: Figure 13; Table 11 | a few minutes |
| `example_7_6_ito_system.py` | **Example 7.6**: Figures 14, 15 | long |
| `nonlinear_kdv.py` | shared solvers for Examples 7.2, 7.3, 7.4, 7.6 | library |

### Example-to-figure map

The paper's captions carry several example numbers that disagree with the surrounding text. The mapping used here follows the text and the cited equation numbers:

| Example | Equations | Figures | Tables | Caption note |
|---|---|---|---|---|
| 7.1 | (7.3) | 5, 6 | 8, 9 | consistent |
| 7.2 | (7.4) | 7 | 10 | consistent |
| 7.3 | (7.5)-(7.8) | 8, 9, 10 | none | consistent |
| 7.4 | (7.9), (7.10) | 11, 12 | none | captions of Figs. 11, 12 read "Example 7.3" |
| 7.5 | (7.11) | 13 | 11 | caption of Fig. 13 and Table 11 read "Example 7.6" |
| 7.6 | (7.12)-(7.14) | 14, 15 | none | captions cite Eq. (7.12) instead of the actual example 7.6 |

The text of Example 7.4 also refers to "the KdV equation (7.3)", but Eq. (7.3) is the *linear* problem of Example 7.1, which has no convective term and therefore no zero-dispersion limit. The equation intended is Eq. (7.5), and that is what `example_7_4_zero_dispersion.py` solves.

The two long scripts are constrained by the time-step restriction $\Delta t \sim \mathrm{CFL}\cdot\Delta x^{3}$, which is the binding cost in every example. Reduce the largest grid size in the convergence sweeps to shorten them.

<!-- ## State of the `figs/` directory

Figures 2, 3, 4, 7, and 13 in this archive were regenerated after the three-format change and are present as `.png`, `.pdf`, and `.svg`. The remaining figures are the `.png` files carried over from the previous run; re-running `example_7_1_linear_kdv.py` and `example_nonlinear_soliton.py` to completion writes them in all three formats. -->

## Verification status

Independent symbolic derivation of the order conditions confirms:

- 25 of the 26 published coefficient rows (Tables 1, 2, and 4) satisfy every order condition they claim, exactly.
- TDCNCS-P8 (Table 1) fails all four of its conditions, with residuals $-2/83$, $-4/83$, $-4/249$, and $-8/3735$ at orders 2, 4, 6, and 8. This row is not used in the paper's computations.
- The printed Eqs. (3.5), (3.6), and (3.7) disagree with both the derivation and the paper's own Table 4. Eqs. (3.3) and (3.4) agree exactly. Table 4 satisfies the derived conditions exactly for all eight rows, so the coefficients used throughout the paper are unaffected.
- Truncation-error constants match the published values: $3.121917\times10^{-5}$ (TDCNCS), $6.572523\times10^{-5}$ (TDCCCS), and $-2.188201\times10^{-6}$ (TDCCS; the paper reports magnitude only).

Run `verify_tables.py` to reproduce these checks.
