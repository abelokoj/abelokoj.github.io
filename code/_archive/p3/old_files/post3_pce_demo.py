"""
Polynomial Chaos Expansion (PCE) surrogate modeling demo.

QoI: a stand-in for an expensive simulation output (e.g. peak stress, drag
coefficient) as a function of one uncertain input parameter x ~ Uniform(-1, 1):

    f(x) = 1 / (1 + 25 x^2)      (sharp, Runge-type response -- deliberately
                                   hard to approximate globally with low-order
                                   polynomials, to make the convergence story honest)

We build a PCE surrogate using Legendre polynomials (the correct orthogonal
basis for a Uniform(-1,1) input), estimate coefficients via Gauss-Legendre
quadrature, and compare:
  (1) surrogate accuracy vs. polynomial order
  (2) PCE-based mean/variance convergence vs. brute-force Monte Carlo
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import legendre as L

rng = np.random.default_rng(1)

def f(x):
    return 1.0 / (1.0 + 25.0 * x**2)

# ---- Build PCE via Gauss-Legendre quadrature projection ----
def legendre_basis_eval(order, x):
    """Return matrix Phi[i,j] = P_j(x_i) for normalized Legendre polynomials on [-1,1]."""
    n_terms = order + 1
    Phi = np.zeros((len(x), n_terms))
    for j in range(n_terms):
        c = np.zeros(n_terms)
        c[j] = 1.0
        Phi[:, j] = L.legval(x, c)
    return Phi

def pce_coefficients(order, f):
    # Gauss-Legendre quadrature nodes/weights on [-1,1] (need enough nodes for exactness)
    n_quad = order + 5
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    fvals = f(nodes)
    Phi = legendre_basis_eval(order, nodes)
    coeffs = np.zeros(order + 1)
    for j in range(order + 1):
        norm_j = 2.0 / (2 * j + 1)   # <P_j, P_j> weight-orthogonality constant on [-1,1]
        coeffs[j] = np.sum(weights * fvals * Phi[:, j]) / norm_j
    return coeffs

def pce_eval(coeffs, x):
    return L.legval(x, coeffs)

# ---- 1) Surrogate accuracy vs. order ----
x_fine = np.linspace(-1, 1, 500)
f_true = f(x_fine)

orders_to_plot = [2, 4, 8, 14]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(x_fine, f_true, "k-", lw=2.2, label="true $f(x)$")
for order in orders_to_plot:
    coeffs = pce_coefficients(order, f)
    f_approx = pce_eval(coeffs, x_fine)
    ax.plot(x_fine, f_approx, lw=1.3, label=f"PCE order {order}")
ax.set_xlabel("$x$")
ax.set_ylabel("$f(x)$")
ax.set_title("Legendre polynomial chaos surrogate vs. true response")
ax.legend()
fig.tight_layout()
fig.savefig("surrogate_fit.png", dpi=150)
plt.close(fig)

# ---- 2) Mean/variance convergence: PCE vs Monte Carlo ----
# "Exact" reference via very high-order quadrature
coeffs_ref = pce_coefficients(40, f)
mean_ref = coeffs_ref[0]                      # E[f] = c_0 for Legendre basis on Uniform(-1,1)
var_ref = np.sum((coeffs_ref[1:]**2) * (2.0 / (2*np.arange(1, len(coeffs_ref)) + 1)) / 2.0)
print(f"Reference mean = {mean_ref:.6f}, reference variance = {var_ref:.6f}")

# PCE convergence: mean/var estimate vs polynomial order, cost = order+5 quadrature evals
pce_orders = list(range(1, 16))
pce_mean_err, pce_var_err, pce_neval = [], [], []
for order in pce_orders:
    c = pce_coefficients(order, f)
    mean_est = c[0]
    var_est = np.sum((c[1:]**2) * (2.0 / (2*np.arange(1, len(c)) + 1)) / 2.0)
    pce_mean_err.append(abs(mean_est - mean_ref))
    pce_var_err.append(abs(var_est - var_ref))
    pce_neval.append(order + 5)

# Monte Carlo convergence: mean/var estimate vs sample size (averaged over repeats)
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

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].loglog(pce_neval, pce_mean_err, "s-", label="PCE (quadrature evals)")
axes[0].loglog(mc_sizes, mc_mean_err, "o-", label="Monte Carlo (samples)")
axes[0].set_xlabel("number of model evaluations")
axes[0].set_ylabel("$|$mean error$|$")
axes[0].set_title("Convergence of $\\mathbb{E}[f]$ estimate")
axes[0].legend()
axes[0].grid(alpha=0.3, which="both")

axes[1].loglog(pce_neval, pce_var_err, "s-", label="PCE (quadrature evals)")
axes[1].loglog(mc_sizes, mc_var_err, "o-", label="Monte Carlo (samples)")
axes[1].set_xlabel("number of model evaluations")
axes[1].set_ylabel("$|$variance error$|$")
axes[1].set_title("Convergence of $\\mathrm{Var}[f]$ estimate")
axes[1].legend()
axes[1].grid(alpha=0.3, which="both")

fig.tight_layout()
fig.savefig("convergence.png", dpi=150)
plt.close(fig)

with open("results.txt", "w") as fout:
    fout.write(f"mean_ref={mean_ref:.6f} var_ref={var_ref:.6f}\n")
    for o, ne, me, ve in zip(pce_orders, pce_neval, pce_mean_err, pce_var_err):
        fout.write(f"order={o} nevals={ne} mean_err={me:.3e} var_err={ve:.3e}\n")

print("done")
