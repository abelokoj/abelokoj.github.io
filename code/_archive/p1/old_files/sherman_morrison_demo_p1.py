"""
Sherman-Morrison accelerated implicit Euler for a mass-varying linear ODE system,
motivated by sustainable-aviation fuel-burn dynamics.

Model:
dy/dt = A(t) y + b, A(t) = A0 + c(t) * u v^T (rank-1, fuel-burn-driven term)

Implicit (backward) Euler:
(I - h A_k) y_{k+1} = y_k + h b
M_k = I - h A0 - h c_k u v^T = B0 - h c_k u v^T, B0 = I - h A0 (fixed!)

Since only a rank-1 term changes at each step, Sherman-Morrison lets us reuse a
single O(n^3) factorization of B0 and apply an O(n^2) rank-1 update per step,
instead of refactoring M_k from scratch (O(n^3)) at every step.
"""
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Publication-quality plotting defaults ---
matplotlib.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "axes.formatter.use_mathtext": True,
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "lines.linewidth": 1.4,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "figure.dpi": 150,
})

rng = np.random.default_rng(0)

def build_system(n, rng):
    A0 = -1.5 * np.eye(n) + 0.05 * rng.standard_normal((n, n))
    # keep A0 stable (eigenvalues with negative real part) by construction below
    A0 = A0 - (np.max(np.real(np.linalg.eigvals(A0))) + 1.0) * np.eye(n)
    u = rng.standard_normal(n)
    v = rng.standard_normal(n)
    b = rng.standard_normal(n)
    y0 = rng.standard_normal(n)
    return A0, u, v, b, y0

def fuel_burn_coeff(t, T, c0=2.0):
    # coupling strength decays as fuel burns off over the flight horizon T
    return c0 * (1.0 - t / T)

def solve_naive(A0, u, v, b, y0, h, steps, T):
    n = len(y0)
    y = y0.copy()
    traj = [y.copy()]
    for k in range(steps):
        t = k * h
        c = fuel_burn_coeff(t, T)
        Ak = A0 + c * np.outer(u, v)
        M = np.eye(n) - h * Ak
        rhs = y + h * b
        y = np.linalg.solve(M, rhs)  # full O(n^3) solve every step
        traj.append(y.copy())
    return np.array(traj)

def solve_sherman_morrison(A0, u, v, b, y0, h, steps, T):
    n = len(y0)
    B0 = np.eye(n) - h * A0
    B0_inv = np.linalg.inv(B0)  # ONE O(n^3) factorization, reused every step
    y = y0.copy()
    traj = [y.copy()]
    for k in range(steps):
        t = k * h
        c = fuel_burn_coeff(t, T)
        alpha = -h * c
        # (B0 + alpha * u v^T)^{-1} via Sherman-Morrison
        Binv_u = B0_inv @ u
        v_Binv = v @ B0_inv
        denom = 1.0 + alpha * (v @ Binv_u)
        M_inv = B0_inv - (alpha / denom) * np.outer(Binv_u, v_Binv)
        rhs = y + h * b
        y = M_inv @ rhs  # O(n^2) per step
        traj.append(y.copy())
    return np.array(traj)

# ---- 1) Correctness check ----
n_check = 30
A0, u, v, b, y0 = build_system(n_check, rng)
T, steps = 5.0, 400
h = T / steps
traj_naive = solve_naive(A0, u, v, b, y0, h, steps, T)
traj_sm = solve_sherman_morrison(A0, u, v, b, y0, h, steps, T)
max_err = np.max(np.abs(traj_naive - traj_sm))
print(f"Max abs difference between naive and Sherman-Morrison trajectories: {max_err:.3e}")

# ---- 2) Trajectory plot (first 3 state components) ----
t_grid = np.linspace(0, T, steps + 1)
fig, ax = plt.subplots(figsize=(7, 4.2))
for i in range(3):
    ax.plot(t_grid, traj_sm[:, i], label=rf"state $y_{i+1}(t)$")
ax.set_xlabel(r"time $t$ (flight horizon, normalized)")
ax.set_ylabel("state value")
ax.set_title("Implicit-Euler trajectories (Sherman-Morrison solve)")
ax.legend(frameon=True)
ax.grid(alpha=0.25, linewidth=0.5)
fig.tight_layout()
fig.savefig("trajectory_p1.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---- 3) Timing benchmark vs. system size n ----
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

    print(f"n={n:4d}  naive={naive_times[-1]:.4f}s  sherman-morrison={sm_times[-1]:.4f}s  "
          f"speedup={naive_times[-1]/sm_times[-1]:.2f}x")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(sizes, naive_times, "o-", label="Naive (refactor every step)")
ax.plot(sizes, sm_times, "s-", label="Sherman-Morrison (factor once)")
ax.set_xlabel(r"system dimension $n$")
ax.set_ylabel(f"wall-clock time for {steps_bench} steps (s)")
ax.set_title("Runtime: naive re-solve vs. Sherman-Morrison update")
ax.legend(frameon=True)
ax.grid(alpha=0.25, linewidth=0.5)
fig.tight_layout()
fig.savefig("timing_p1.png", dpi=300, bbox_inches="tight")
plt.close(fig)

with open("results.txt", "w") as f:
    f.write(f"max_err={max_err:.3e}\n")
    for n, tn, ts in zip(sizes, naive_times, sm_times):
        f.write(f"n={n} naive={tn:.5f} sm={ts:.5f} speedup={tn/ts:.2f}\n")

print("done")
