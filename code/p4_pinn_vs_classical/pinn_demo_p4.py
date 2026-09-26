"""
Physics-informed neural network (PINN) vs. a classical solver, on a
problem simple enough to have a closed-form solution: a forced, damped
first-order linear ODE (Newton cooling with an oscillating ambient
temperature),

    dy/dt = -k (y - A sin(omega t)),   y(0) = y0.

This has an exact analytic solution (via an integrating factor), so both
methods can be scored against ground truth rather than against each
other.

The network is a small single-hidden-layer tanh network. Because it only
needs a first derivative, the network output y_hat(t) and its exact
analytic time-derivative can be written in closed form for this
architecture, and trained with a plain gradient-based optimizer (L-BFGS-B)
minimizing the ODE-residual-plus-initial-condition loss -- no automatic
differentiation library required.
"""
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

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


# ---- Problem definition ----
k = 2.0
A = 1.0
omega = 3.0
y0 = 0.3
T = 3.0


def rhs(t, y):
    return -k * (y - A * np.sin(omega * t))


def exact_solution(t):
    # y' + k y = k A sin(w t); solved via integrating factor e^{kt}.
    part = (k * A / (k**2 + omega**2)) * (k * np.sin(omega * t) - omega * np.cos(omega * t))
    homog_coeff = y0 - (k * A / (k**2 + omega**2)) * (-omega)
    return homog_coeff * np.exp(-k * t) + part


# sanity check exact solution against the initial condition at t=0
assert abs(exact_solution(0.0) - y0) < 1e-10

# ---- Classical solver baseline ----
t_dense = np.linspace(0, T, 400)
t0 = time.perf_counter()
sol_classical = solve_ivp(rhs, (0, T), [y0], t_eval=t_dense, method="RK45",
                           rtol=1e-10, atol=1e-12)
t1 = time.perf_counter()
classical_time = t1 - t0
y_classical = sol_classical.y[0]
y_exact_dense = exact_solution(t_dense)
classical_err = np.max(np.abs(y_classical - y_exact_dense))
print(f"Classical RK45: wall={classical_time * 1000:.3f} ms, "
      f"max err vs exact = {classical_err:.2e}")

# ---- PINN setup ----
n_hidden = 12
n_params = 3 * n_hidden + 1  # w1, b1, w2 (n_hidden each), plus scalar b2


def unpack(theta):
    w1 = theta[0:n_hidden]
    b1 = theta[n_hidden:2 * n_hidden]
    w2 = theta[2 * n_hidden:3 * n_hidden]
    b2 = theta[3 * n_hidden]
    return w1, b1, w2, b2


def y_hat(t, theta):
    w1, b1, w2, b2 = unpack(theta)
    z = np.outer(t, w1) + b1          # (N, H)
    h = np.tanh(z)
    return h @ w2 + b2                # (N,)


def dy_hat_dt(t, theta):
    w1, b1, w2, b2 = unpack(theta)
    z = np.outer(t, w1) + b1
    dh = (1 - np.tanh(z)**2) * w1     # d/dt[tanh(w1 t + b1)]
    return dh @ w2


def loss(theta, t_colloc):
    y = y_hat(t_colloc, theta)
    dy = dy_hat_dt(t_colloc, theta)
    residual = dy - rhs(t_colloc, y)
    ic_pred = y_hat(np.array([0.0]), theta)[0]
    return np.mean(residual**2) + 50.0 * (ic_pred - y0)**2


rng = np.random.default_rng(42)
t_colloc = np.linspace(0, T, 60)
theta0 = 0.5 * rng.standard_normal(n_params)

t0 = time.perf_counter()
res = minimize(loss, theta0, args=(t_colloc,), method="L-BFGS-B",
                options=dict(maxiter=4000, ftol=1e-14, gtol=1e-12))
t1 = time.perf_counter()
pinn_time = t1 - t0
theta_star = res.x
print(f"PINN training: wall={pinn_time * 1000:.1f} ms, final loss={res.fun:.3e}, "
      f"iters={res.nit}, converged={res.success}")

y_pinn_dense = y_hat(t_dense, theta_star)
pinn_err = np.max(np.abs(y_pinn_dense - y_exact_dense))
print(f"PINN: max err vs exact = {pinn_err:.2e}")

# =======================================================================
# Plot 1: solution comparison
# =======================================================================
fig, ax = plt.subplots(figsize=(6.4, 3.9))
ax.plot(t_dense, y_exact_dense, "k-", label="exact analytic solution")
ax.plot(t_dense, y_classical, "--",
        label=rf"classical RK45 (err$={classical_err:.1e}$)")
ax.plot(t_dense, y_pinn_dense, ":",
        label=rf"PINN (err$={pinn_err:.1e}$)")
ax.set_xlabel(r"time $t$")
ax.set_ylabel(r"$y(t)$")
ax.set_title("Forced damped ODE: exact vs. classical solver vs. PINN")
ax.legend(frameon=False)
cm_x(ax)
cm_y(ax)
fig.tight_layout()
savefig_all(fig, "solution_comparison_p4")
plt.close(fig)

# =======================================================================
# Plot 2: cost/accuracy tradeoff summary
# =======================================================================
fig, ax = plt.subplots(figsize=(5.6, 4.3))
ax.scatter([classical_time * 1000], [classical_err], s=130, marker="o",
           label="classical RK45", zorder=3)
ax.scatter([pinn_time * 1000], [pinn_err], s=130, marker="^",
           label="PINN (L-BFGS-B training)", zorder=3)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("wall-clock time (ms, log scale)")
ax.set_ylabel("max error vs. exact solution (log scale)")
ax.set_title("Accuracy vs. compute cost: classical solver dominates\n"
              "for this well-posed 1D forward problem")
ax.legend(frameon=False)
ax.grid(which="both")
fig.tight_layout()
savefig_all(fig, "cost_accuracy_p4")
plt.close(fig)

with open("results.txt", "w") as f:
    f.write(f"classical_time_ms={classical_time * 1000:.4f} classical_err={classical_err:.3e}\n")
    f.write(f"pinn_time_ms={pinn_time * 1000:.4f} pinn_err={pinn_err:.3e} "
            f"pinn_iters={res.nit}\n")

print("done")