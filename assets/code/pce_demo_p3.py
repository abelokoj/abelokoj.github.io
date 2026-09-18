"""
Polynomial chaos expansion (PCE) surrogate modeling demo.

Quantity of interest (QoI): a stand-in for an expensive simulation output
(e.g. peak stress, drag coefficient) as a function of one uncertain input
parameter x ~ Uniform(-1, 1):

    f(x) = 1 / (1 + 25 x^2)      (a sharp, Runge-type response -- chosen
                                   deliberately, since it is hard to
                                   approximate globally with low-order
                                   polynomials, keeping the convergence
                                   story honest rather than flattering)

A PCE surrogate is built using Legendre polynomials (the correct
orthogonal basis for a Uniform(-1,1) input), with coefficients estimated
via Gauss-Legendre quadrature. The script then compares:
  (1) surrogate accuracy vs. polynomial order
  (2) PCE-based mean/variance convergence vs. brute-force Monte Carlo
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from numpy.polynomial import legendre as L

# ---------------------------------------------------------------------
# Publication-style plot settings.
# ---------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Serif"],
    "axes.formatter.use_mathtext": True, "font.size": 11,
    "axes.labelsize": 12, "legend.fontsize": 9,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.linewidth": 0.8, "lines.linewidth": 1.2,
    "figure.dpi": 600, "savefig.dpi": 600, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.6, "grid.linestyle": "--",
})


class CMTickFormatter(ScalarFormatter):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.set_useMathText(False)

    def __call__(self, x, pos=None):
        return f"${super().__call__(x, pos)}$"


def cm_x(ax):
    ax.xaxis.set_major_formatter(CMTickFormatter())


def cm_y(ax):
    ax.yaxis.set_major_formatter(CMTickFormatter())


def savefig_all(fig, basename, **kwargs):
    """Save a figure as both a high-resolution raster (PNG) and a vector
    (SVG) copy, matching the naming convention used across the post."""
    fig.savefig(f"{basename}.png", **kwargs)
    fig.savefig(f"{basename}.svg", **kwargs)


rng = np.random.default_rng(1)


def f(x):
    return 1.0 / (1.0 + 25.0 * x**2)


# ---------------------------------------------------------------------
# Build a PCE surrogate via Gauss-Legendre quadrature projection.
# ---------------------------------------------------------------------
def legendre_basis_eval(order, x):
    """Return Phi[i, j] = P_j(x_i) for Legendre polynomials of degree
    0..order, evaluated at points x."""
    n_terms = order + 1
    Phi = np.zeros((len(x), n_terms))
    for j in range(n_terms):
        c = np.zeros(n_terms)
        c[j] = 1.0
        Phi[:, j] = L.legval(x, c)
    return Phi


def pce_coefficients(order, f):
    # Gauss-Legendre nodes/weights on [-1, 1]; a small buffer of extra
    # nodes keeps the quadrature exact for the polynomial products involved.
    n_quad = order + 5
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    fvals = f(nodes)
    Phi = legendre_basis_eval(order, nodes)
    coeffs = np.zeros(order + 1)
    for j in range(order + 1):
        norm_j = 2.0 / (2 * j + 1)  # <P_j, P_j> normalization on [-1, 1]
        coeffs[j] = np.sum(weights * fvals * Phi[:, j]) / norm_j
    return coeffs


def pce_eval(coeffs, x):
    return L.legval(x, coeffs)


# =======================================================================
# 1) Surrogate accuracy vs. polynomial order
# =======================================================================
x_fine = np.linspace(-1, 1, 500)
f_true = f(x_fine)

orders_to_plot = [2, 4, 8, 14]
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.plot(x_fine, f_true, "k-", label=r"true $f(x)$")
for order in orders_to_plot:
    coeffs = pce_coefficients(order, f)
    f_approx = pce_eval(coeffs, x_fine)
    ax.plot(x_fine, f_approx, lw=1.2, label=rf"PCE order {order}")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$f(x)$")
ax.set_title("Legendre polynomial-chaos surrogate vs. true response")
ax.legend(frameon=False, fontsize=8.5)
cm_x(ax)
cm_y(ax)
fig.tight_layout()
savefig_all(fig, "surrogate_fit_p3")
plt.close(fig)

# =======================================================================
# 2) Mean/variance convergence: PCE vs. Monte Carlo
# =======================================================================
# "Exact" reference via a very high-order quadrature-based PCE.
coeffs_ref = pce_coefficients(40, f)
mean_ref = coeffs_ref[0]  # E[f] = c_0 for the Legendre basis on Uniform(-1,1)
var_ref = np.sum((coeffs_ref[1:]**2) * (2.0 / (2 * np.arange(1, len(coeffs_ref)) + 1)) / 2.0)
print(f"Reference mean = {mean_ref:.6f}, reference variance = {var_ref:.6f}")

# PCE convergence: mean/variance estimate vs. polynomial order; cost =
# (order + 5) quadrature evaluations.
pce_orders = list(range(1, 16))
pce_mean_err, pce_var_err, pce_neval = [], [], []
for order in pce_orders:
    c = pce_coefficients(order, f)
    mean_est = c[0]
    var_est = np.sum((c[1:]**2) * (2.0 / (2 * np.arange(1, len(c)) + 1)) / 2.0)
    pce_mean_err.append(abs(mean_est - mean_ref))
    pce_var_err.append(abs(var_est - var_ref))
    pce_neval.append(order + 5)

# Monte Carlo convergence: mean/variance estimate vs. sample size, averaged
# over repeated draws to smooth out sampling noise in the reported error.
mc_sizes = [10, 30, 100, 300, 1000, 3000, 10000, 30000]
mc_mean_err, mc_var_err = [], []
n_repeats = 30
for N in mc_sizes:
    mean_errs, var_errs = [], []
    for _ in range(n_repeats):
        x_samp = rng.uniform(-1, 1, N)
        fvals = f(x_samp)
        mean_errs.append(abs(fvals.mean() - mean_ref))
        var_errs.append(abs(fvals.var() - var_ref))
    mc_mean_err.append(np.mean(mean_errs))
    mc_var_err.append(np.mean(var_errs))

# Note: both axes are log-scale (loglog) in this figure, so the CM tick
# formatter (built on ScalarFormatter) is not applied here -- it isn't
# appropriate for log axes, which need LogFormatter-style tick labels.
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
axes[0].loglog(pce_neval, pce_mean_err, "s-", ms=5, label="PCE (quadrature evals)")
axes[0].loglog(mc_sizes, mc_mean_err, "o-", ms=5, label="Monte Carlo (samples)")
axes[0].set_xlabel("number of model evaluations")
axes[0].set_ylabel(r"$|$mean error$|$")
axes[0].set_title(r"Convergence of $\mathbb{E}[f]$ estimate")
axes[0].legend(frameon=False, fontsize=8.5)
axes[0].grid(which="both")

axes[1].loglog(pce_neval, pce_var_err, "s-", ms=5, label="PCE (quadrature evals)")
axes[1].loglog(mc_sizes, mc_var_err, "o-", ms=5, label="Monte Carlo (samples)")
axes[1].set_xlabel("number of model evaluations")
axes[1].set_ylabel(r"$|$variance error$|$")
axes[1].set_title(r"Convergence of $\mathrm{Var}[f]$ estimate")
axes[1].legend(frameon=False, fontsize=8.5)
axes[1].grid(which="both")

fig.tight_layout()
savefig_all(fig, "convergence_p3")
plt.close(fig)

with open("results.txt", "w") as fout:
    fout.write(f"mean_ref={mean_ref:.6f} var_ref={var_ref:.6f}\n")
    for o, ne, me, ve in zip(pce_orders, pce_neval, pce_mean_err, pce_var_err):
        fout.write(f"order={o} nevals={ne} mean_err={me:.3e} var_err={ve:.3e}\n")

print("done")