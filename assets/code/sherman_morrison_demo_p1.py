"""
Sherman-Morrison-accelerated implicit Euler integration for a mass-varying
linear ODE system, motivated by sustainable-aviation-fuel (SAF) burn
dynamics.

Model
-----
    dy/dt = A(t) y + b,      A(t) = A0 + c(t) u v^T

The correction term c(t) u v^T is rank one: its *direction* (u, v) is fixed
over the flight, while its *magnitude* c(t) decays as fuel burns off.

Implicit (backward) Euler
--------------------------
    (I - h A_{k+1}) y_{k+1} = y_k + h b
    M_{k+1} = B0 - (h c_{k+1}) u v^T,      B0 = I - h A0   (fixed matrix)

Because M_{k+1} is always a rank-one perturbation of the same fixed matrix
B0, the Sherman-Morrison formula lets a single O(n^3) factorization of B0
be reused for the entire simulation, replacing an O(n^3) refactorization at
every step with an O(n^2) rank-one update.

This script (1) validates the Sherman-Morrison solve against a naive
refactor-every-step baseline, (2) plots a representative trajectory, and
(3) benchmarks wall-clock cost across system size n.
"""
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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


rng = np.random.default_rng(0)


def build_system(n, rng):
    """Construct a random, stable baseline matrix A0 and rank-one
    perturbation directions (u, v), plus a forcing term b and initial
    condition y0."""
    A0 = -1.5 * np.eye(n) + 0.05 * rng.standard_normal((n, n))
    # Shift A0 so all eigenvalues have negative real part (a stable
    # baseline dynamics matrix).
    A0 = A0 - (np.max(np.real(np.linalg.eigvals(A0))) + 1.0) * np.eye(n)
    u = rng.standard_normal(n)
    v = rng.standard_normal(n)
    b = rng.standard_normal(n)
    y0 = rng.standard_normal(n)
    return A0, u, v, b, y0


def fuel_burn_coeff(t, T, c0=2.0):
    """Coupling strength c(t): decays linearly to zero as fuel burns off
    over the flight horizon [0, T]."""
    return c0 * (1.0 - t / T)


def solve_naive(A0, u, v, b, y0, h, steps, T):
    """Refactor the full system matrix from scratch at every step: O(n^3)
    per step."""
    n = len(y0)
    y = y0.copy()
    traj = [y.copy()]
    for k in range(steps):
        t = k * h
        c = fuel_burn_coeff(t, T)
        Ak = A0 + c * np.outer(u, v)
        M = np.eye(n) - h * Ak
        rhs = y + h * b
        y = np.linalg.solve(M, rhs)
        traj.append(y.copy())
    return np.array(traj)


def solve_sherman_morrison(A0, u, v, b, y0, h, steps, T):
    """Factor B0 once, then apply a rank-one Sherman-Morrison update at
    each step: O(n^2) per step after the initial O(n^3) factorization."""
    n = len(y0)
    B0 = np.eye(n) - h * A0
    B0_inv = np.linalg.inv(B0)  # single O(n^3) factorization, reused below
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
        rhs = y + h * b
        y = M_inv @ rhs
        traj.append(y.copy())
    return np.array(traj)


# =======================================================================
# 1) Correctness check: naive refactorization vs. Sherman-Morrison
# =======================================================================
n_check = 30
A0, u, v, b, y0 = build_system(n_check, rng)
T, steps = 5.0, 400
h = T / steps
traj_naive = solve_naive(A0, u, v, b, y0, h, steps, T)
traj_sm = solve_sherman_morrison(A0, u, v, b, y0, h, steps, T)
max_err = np.max(np.abs(traj_naive - traj_sm))
print(f"Max abs difference between naive and Sherman-Morrison trajectories: {max_err:.3e}")

# =======================================================================
# 2) Trajectory plot (first three state components)
# =======================================================================
t_grid = np.linspace(0, T, steps + 1)
fig, ax = plt.subplots(figsize=(6.0, 3.6))
for i in range(3):
    ax.plot(t_grid, traj_sm[:, i], label=rf"$y_{i+1}(t)$")
ax.set_xlabel(r"time $t$ (normalized flight horizon)")
ax.set_ylabel(r"state value $y_i(t)$")
ax.set_title("Implicit-Euler trajectories via the Sherman-Morrison solve")
ax.legend(frameon=False)
cm_x(ax)
cm_y(ax)
fig.tight_layout()
savefig_all(fig, "trajectory_p1")
plt.close(fig)

# =======================================================================
# 3) Timing benchmark vs. system size n
# =======================================================================
sizes = [20, 40, 80, 160, 320]
steps_bench = 150
T_bench = 3.0
h_bench = T_bench / steps_bench

naive_times, sm_times = [], []
for n in sizes:
    A0n, un, vn, bn, y0n = build_system(n, rng)

    t0 = time.perf_counter()
    solve_naive(A0n, un, vn, bn, y0n, h_bench, steps_bench, T_bench)
    t1 = time.perf_counter()
    naive_times.append(t1 - t0)

    t0 = time.perf_counter()
    solve_sherman_morrison(A0n, un, vn, bn, y0n, h_bench, steps_bench, T_bench)
    t1 = time.perf_counter()
    sm_times.append(t1 - t0)

    print(f"n={n:4d}  naive={naive_times[-1]:.4f}s  "
          f"sherman-morrison={sm_times[-1]:.4f}s  "
          f"speedup={naive_times[-1] / sm_times[-1]:.2f}x")

fig, ax = plt.subplots(figsize=(6.0, 3.9))
ax.plot(sizes, naive_times, "o-", ms=5, label="naive (refactor every step)")
ax.plot(sizes, sm_times, "s-", ms=5, label="Sherman-Morrison (factor once)")
ax.set_xlabel(r"system dimension $n$")
ax.set_ylabel(rf"wall-clock time for {steps_bench} steps (s)")
ax.set_title("Runtime: naive re-solve vs. Sherman-Morrison update")
ax.legend(frameon=False)
cm_x(ax)
cm_y(ax)
fig.tight_layout()
savefig_all(fig, "timing_p1")
plt.close(fig)

with open("results.txt", "w") as f:
    f.write(f"max_err={max_err:.3e}\n")
    for n, tn, ts in zip(sizes, naive_times, sm_times):
        f.write(f"n={n} naive={tn:.5f} sm={ts:.5f} speedup={tn / ts:.2f}\n")

print("done")