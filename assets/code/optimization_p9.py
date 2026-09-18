"""
Part A: Convex optimization in practice -- fitting logistic regression on
the real breast cancer diagnostic dataset by minimizing its (convex)
negative log-likelihood via three different optimizers implemented from
scratch: gradient descent, Newton's method, and comparison against scipy's
L-BFGS-B. Convergence rate is compared, not just final accuracy.

Part B: A non-convex, physically-motivated trajectory optimization problem
-- minimum-fuel control of a simple 1D vehicle (a toy stand-in for the
kind of minimum-fuel aviation trajectory problem referenced elsewhere in
this series) subject to boundary conditions, solved via SLSQP.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import minimize

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


# =====================================================================
# PART A: Logistic regression on real data -- GD vs Newton vs L-BFGS
# =====================================================================
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X_raw, y = data.data, data.target
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.25, stratify=y, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Use only 2 features for the 2D contour-plot visualization; use all 30
# for the "real" convergence comparison below.
feat_idx = [np.where(data.feature_names == "mean radius")[0][0],
            np.where(data.feature_names == "mean texture")[0][0]]

n, d = X_train.shape


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


LAMBDA = 1.0   # L2 regularization strength


def neg_log_likelihood(beta, X, y):
    z = X @ beta
    # numerically stable log(1+exp(z)) via logaddexp
    ll = np.sum(y * z - np.logaddexp(0, z))
    reg = 0.5 * LAMBDA * np.sum(beta ** 2)
    return -ll / len(y) + reg / len(y)


def gradient(beta, X, y):
    p = sigmoid(X @ beta)
    grad = X.T @ (p - y) / len(y) + LAMBDA * beta / len(y)
    return grad


def hessian(beta, X, y):
    p = sigmoid(X @ beta)
    W = p * (1 - p)
    H = (X.T * W) @ X / len(y) + LAMBDA * np.eye(X.shape[1]) / len(y)
    return H


# ---- Gradient descent ----
def gradient_descent(X, y, n_iter=500, lr=0.5):
    beta = np.zeros(X.shape[1])
    losses = [neg_log_likelihood(beta, X, y)]
    for _ in range(n_iter):
        beta = beta - lr * gradient(beta, X, y)
        losses.append(neg_log_likelihood(beta, X, y))
    return beta, losses


# ---- Newton's method ----
def newtons_method(X, y, n_iter=20):
    beta = np.zeros(X.shape[1])
    losses = [neg_log_likelihood(beta, X, y)]
    for _ in range(n_iter):
        g = gradient(beta, X, y)
        H = hessian(beta, X, y)
        beta = beta - np.linalg.solve(H, g)
        losses.append(neg_log_likelihood(beta, X, y))
    return beta, losses


# ---- L-BFGS via scipy, tracking loss at every iteration ----
def lbfgs_method(X, y):
    losses = []

    def callback(beta):
        losses.append(neg_log_likelihood(beta, X, y))

    beta0 = np.zeros(X.shape[1])
    losses.append(neg_log_likelihood(beta0, X, y))
    res = minimize(neg_log_likelihood, beta0, args=(X, y), jac=gradient,
                    method="L-BFGS-B", callback=callback,
                    options=dict(maxiter=200))
    return res.x, losses


print("Running optimizers on full 30-feature logistic regression...")
beta_gd, losses_gd = gradient_descent(X_train, y_train, n_iter=500, lr=0.9)
beta_newton, losses_newton = newtons_method(X_train, y_train, n_iter=20)
beta_lbfgs, losses_lbfgs = lbfgs_method(X_train, y_train)

final_loss_gd = neg_log_likelihood(beta_gd, X_train, y_train)
final_loss_newton = neg_log_likelihood(beta_newton, X_train, y_train)
final_loss_lbfgs = neg_log_likelihood(beta_lbfgs, X_train, y_train)
print(f"Final training loss -- GD: {final_loss_gd:.6f} ({len(losses_gd)-1} iters)")
print(f"Final training loss -- Newton: {final_loss_newton:.6f} ({len(losses_newton)-1} iters)")
print(f"Final training loss -- L-BFGS: {final_loss_lbfgs:.6f} ({len(losses_lbfgs)-1} iters)")


def test_accuracy(beta, X, y):
    pred = (sigmoid(X @ beta) > 0.5).astype(int)
    return (pred == y).mean()


print(f"Test accuracy -- GD: {test_accuracy(beta_gd, X_test, y_test):.4f}")
print(f"Test accuracy -- Newton: {test_accuracy(beta_newton, X_test, y_test):.4f}")
print(f"Test accuracy -- L-BFGS: {test_accuracy(beta_lbfgs, X_test, y_test):.4f}")

# ---- Convergence plot (log scale, aligned to best known loss) ----
# Note: y-axis is log-scale, so only the linear x-axis (iteration) gets
# the CM tick formatter.
best_loss = min(final_loss_gd, final_loss_newton, final_loss_lbfgs)
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.semilogy(np.array(losses_gd) - best_loss + 1e-12,
            label=rf"gradient descent ({len(losses_gd)-1} iters)")
ax.semilogy(np.array(losses_newton) - best_loss + 1e-12, marker="o", ms=4,
            label=rf"Newton's method ({len(losses_newton)-1} iters)")
ax.semilogy(np.array(losses_lbfgs) - best_loss + 1e-12, marker="s", ms=4,
            label=rf"L-BFGS ({len(losses_lbfgs)-1} iters)")
ax.set_xlabel("iteration")
ax.set_ylabel("loss $-$ best loss (log scale)")
ax.set_title("Convergence comparison: logistic regression on breast cancer data")
ax.legend(frameon=False)
ax.set_xlim(0, 60)
ax.grid(which="both")
cm_x(ax)
fig.tight_layout()
savefig_all(fig, "optimizer_convergence_p9")
plt.close(fig)

# ---- 2D loss landscape + optimizer paths (using only 2 features) ----
X2 = X_train[:, feat_idx]


def nll_2d(beta, X, y):
    return neg_log_likelihood(beta, X, y)


def grad_2d(beta, X, y):
    return gradient(beta, X, y)


def hess_2d(beta, X, y):
    return hessian(beta, X, y)


def gd_path(X, y, n_iter=60, lr=0.9):
    beta = np.zeros(2)
    path = [beta.copy()]
    for _ in range(n_iter):
        beta = beta - lr * grad_2d(beta, X, y)
        path.append(beta.copy())
    return np.array(path)


def newton_path(X, y, n_iter=8):
    beta = np.zeros(2)
    path = [beta.copy()]
    for _ in range(n_iter):
        g = grad_2d(beta, X, y)
        H = hess_2d(beta, X, y)
        beta = beta - np.linalg.solve(H, g)
        path.append(beta.copy())
    return np.array(path)


path_gd_2d = gd_path(X2, y_train)
path_newton_2d = newton_path(X2, y_train)

b0_range = np.linspace(-3, 5, 100)
b1_range = np.linspace(-3, 5, 100)
B0, B1 = np.meshgrid(b0_range, b1_range)
Z = np.zeros_like(B0)
for i in range(B0.shape[0]):
    for j in range(B0.shape[1]):
        Z[i, j] = nll_2d(np.array([B0[i, j], B1[i, j]]), X2, y_train)

fig, ax = plt.subplots(figsize=(6.6, 5.8))
cs = ax.contour(B0, B1, Z, levels=30, cmap="viridis", linewidths=0.8)
ax.plot(path_gd_2d[:, 0], path_gd_2d[:, 1], "o-", color="red", ms=3, lw=1.2,
        label=rf"gradient descent ({len(path_gd_2d)-1} steps)")
ax.plot(path_newton_2d[:, 0], path_newton_2d[:, 1], "s-", color="orange", ms=5, lw=1.4,
        label=rf"Newton's method ({len(path_newton_2d)-1} steps)")
ax.set_xlabel(r"$\beta_{\mathrm{mean\ radius}}$")
ax.set_ylabel(r"$\beta_{\mathrm{mean\ texture}}$")
ax.set_title("Convex loss landscape (2-feature logistic regression)\nand optimizer trajectories")
ax.legend(frameon=False, fontsize=8.5)
fig.colorbar(cs, ax=ax, fraction=0.046, label="negative log-likelihood")
cm_x(ax)
cm_y(ax)
fig.tight_layout()
savefig_all(fig, "loss_landscape_paths_p9")
plt.close(fig)

# =====================================================================
# PART B: Minimum-fuel trajectory optimization (non-convex control problem)
# =====================================================================
# A simple 1D vehicle: position q(t), velocity v(t), control u(t) = thrust
#   dq/dt = v,  dv/dt = u - k*v^2*sign(v)   (thrust minus quadratic drag)
# Objective: reach target position q_f at time T with v(T)=0, minimizing
# total fuel (integral of |u|), discretized via direct collocation.
print("\n\nPART B: Minimum-fuel trajectory optimization")

N_NODES = 60
T_FINAL = 10.0
dt_traj = T_FINAL / (N_NODES - 1)
Q0, V0 = 0.0, 0.0
QF, VF = 20.0, 0.0
DRAG_COEF = 0.02


def unpack_traj(z):
    q = z[0:N_NODES]
    v = z[N_NODES:2*N_NODES]
    u = z[2*N_NODES:3*N_NODES]
    return q, v, u


def objective(z):
    _, _, u = unpack_traj(z)
    # trapezoidal approximation of integral of u^2 (fuel proxy; using u^2
    # rather than |u| keeps the objective smooth for gradient-based SLSQP)
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz
    return trapz_fn(u ** 2, dx=dt_traj)


def dynamics_constraints(z):
    q, v, u = unpack_traj(z)
    cons = []
    for k in range(N_NODES - 1):
        # trapezoidal collocation for q and v
        drag_k = DRAG_COEF * v[k] * np.abs(v[k])
        drag_k1 = DRAG_COEF * v[k+1] * np.abs(v[k+1])
        q_next_pred = q[k] + 0.5 * dt_traj * (v[k] + v[k+1])
        v_next_pred = v[k] + 0.5 * dt_traj * ((u[k] - drag_k) + (u[k+1] - drag_k1))
        cons.append(q[k+1] - q_next_pred)
        cons.append(v[k+1] - v_next_pred)
    return np.array(cons)


def boundary_constraints(z):
    q, v, u = unpack_traj(z)
    return np.array([q[0] - Q0, v[0] - V0, q[-1] - QF, v[-1] - VF])


constraints = [
    {"type": "eq", "fun": dynamics_constraints},
    {"type": "eq", "fun": boundary_constraints},
]

# Initial guess: straight-line interpolation, zero control
q_guess = np.linspace(Q0, QF, N_NODES)
v_guess = np.full(N_NODES, (QF - Q0) / T_FINAL)
u_guess = np.zeros(N_NODES)
z0 = np.concatenate([q_guess, v_guess, u_guess])

result = minimize(objective, z0, constraints=constraints, method="SLSQP",
                   options=dict(maxiter=300, ftol=1e-9))
print(f"Optimization success: {result.success}, message: {result.message}")
print(f"Final fuel cost (integral of u^2): {result.fun:.4f}")

q_opt, v_opt, u_opt = unpack_traj(result.x)
t_grid = np.linspace(0, T_FINAL, N_NODES)

fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
axes[0].plot(t_grid, q_opt, color="tab:blue")
axes[0].axhline(QF, color="gray", linestyle="--", lw=1, label="target position")
axes[0].set_xlabel("time"); axes[0].set_ylabel(r"position $q(t)$")
axes[0].set_title("Optimized position trajectory")
axes[0].legend(frameon=False, fontsize=8)

axes[1].plot(t_grid, v_opt, color="tab:orange")
axes[1].set_xlabel("time"); axes[1].set_ylabel(r"velocity $v(t)$")
axes[1].set_title("Optimized velocity trajectory")

axes[2].plot(t_grid, u_opt, color="tab:green")
axes[2].axhline(0, color="gray", linestyle=":", lw=1)
axes[2].set_xlabel("time"); axes[2].set_ylabel(r"control (thrust) $u(t)$")
axes[2].set_title("Optimized control (fuel usage) profile")

for ax in axes:
    cm_x(ax)
    cm_y(ax)

fig.tight_layout()
savefig_all(fig, "trajectory_optimization_p9")
plt.close(fig)

with open("results.txt", "w") as f:
    f.write(f"Part A: GD final loss={final_loss_gd:.6f} iters={len(losses_gd)-1}\n")
    f.write(f"Part A: Newton final loss={final_loss_newton:.6f} iters={len(losses_newton)-1}\n")
    f.write(f"Part A: LBFGS final loss={final_loss_lbfgs:.6f} iters={len(losses_lbfgs)-1}\n")
    f.write(f"Part B: success={result.success} fuel_cost={result.fun:.4f}\n")

print("done")