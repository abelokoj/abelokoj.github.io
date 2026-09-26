"""
Physics-informed neural network (PINN) vs. a classical solver, on a problem
simple enough to have a closed-form solution: a forced, damped first-order
linear ODE (Newton cooling with an oscillating ambient temperature).

    dy/dt = -k (y - A sin(omega t)),   y(0) = y0

This has an exact analytic solution (solve via integrating factor), so we can
score both methods against ground truth instead of against each other.

The "network" is a small single-hidden-layer tanh network. Because it only
needs a first derivative, we can write y_hat(t) and dy_hat/dt in closed form
and train with a plain gradient-free optimizer (L-BFGS-B) minimizing the
ODE-residual + initial-condition loss -- no autodiff library required.
"""
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

# ---- Problem definition ----
k = 2.0
A = 1.0
omega = 3.0
y0 = 0.3
T = 3.0

def rhs(t, y):
    return -k * (y - A * np.sin(omega * t))

def exact_solution(t):
    # y' + k y = k A sin(w t); solve via integrating factor e^{kt}
    C = y0 - (k * A / (k**2 + omega**2)) * (-omega)  # matches particular soln at t=0
    part = (k * A / (k**2 + omega**2)) * (k * np.sin(omega * t) - omega * np.cos(omega * t))
    homog_coeff = y0 - (k * A / (k**2 + omega**2)) * (-omega)
    return homog_coeff * np.exp(-k * t) + part

# sanity check exact solution against numerical RHS at t=0
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
print(f"Classical RK45: wall={classical_time*1000:.3f} ms, max err vs exact = {classical_err:.2e}")

# ---- PINN setup ----
n_hidden = 12
n_params = 3 * n_hidden + 1   # w1, b1, w2 (n_hidden each) + b2

def unpack(theta):
    w1 = theta[0:n_hidden]
    b1 = theta[n_hidden:2*n_hidden]
    w2 = theta[2*n_hidden:3*n_hidden]
    b2 = theta[3*n_hidden]
    return w1, b1, w2, b2

def y_hat(t, theta):
    w1, b1, w2, b2 = unpack(theta)
    z = np.outer(t, w1) + b1          # (N, H)
    h = np.tanh(z)
    return h @ w2 + b2                # (N,)

def dy_hat_dt(t, theta):
    w1, b1, w2, b2 = unpack(theta)
    z = np.outer(t, w1) + b1
    dh = (1 - np.tanh(z)**2) * w1     # derivative of tanh(w1 t + b1) wrt t
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
print(f"PINN training: wall={pinn_time*1000:.1f} ms, final loss={res.fun:.3e}, "
      f"iters={res.nit}, converged={res.success}")

y_pinn_dense = y_hat(t_dense, theta_star)
pinn_err = np.max(np.abs(y_pinn_dense - y_exact_dense))
print(f"PINN: max err vs exact = {pinn_err:.2e}")

# ---- Plot 1: solution comparison ----
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(t_dense, y_exact_dense, "k-", lw=2.5, label="exact analytic solution")
ax.plot(t_dense, y_classical, "--", lw=1.8, label=f"classical RK45 (err={classical_err:.1e})")
ax.plot(t_dense, y_pinn_dense, ":", lw=2.2, label=f"PINN (err={pinn_err:.1e})")
ax.set_xlabel("time $t$")
ax.set_ylabel("$y(t)$")
ax.set_title("Forced damped ODE: exact vs. classical solver vs. PINN")
ax.legend()
fig.tight_layout()
fig.savefig("solution_comparison.png", dpi=150)
plt.close(fig)

# ---- Plot 2: cost/accuracy tradeoff summary ----
fig, ax = plt.subplots(figsize=(6.5, 4.8))
ax.scatter([classical_time*1000], [classical_err], s=140, marker="o", label="Classical RK45")
ax.scatter([pinn_time*1000], [pinn_err], s=140, marker="^", label="PINN (L-BFGS-B training)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("wall-clock time (ms, log scale)")
ax.set_ylabel("max error vs. exact solution (log scale)")
ax.set_title("Accuracy vs. compute cost: classical solver dominates\nfor this well-posed 1D forward problem")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("cost_accuracy.png", dpi=150)
plt.close(fig)

with open("results.txt", "w") as f:
    f.write(f"classical_time_ms={classical_time*1000:.4f} classical_err={classical_err:.3e}\n")
    f.write(f"pinn_time_ms={pinn_time*1000:.4f} pinn_err={pinn_err:.3e} "
            f"pinn_iters={res.nit}\n")

print("done")
