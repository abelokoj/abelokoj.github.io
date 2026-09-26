"""
Stiff ODE demo: Van der Pol oscillator with large mu, comparing an explicit
solver (RK45) against an implicit/stiff-aware solver (Radau / BDF-family).
"""
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def vdp(t, y, mu):
    x, v = y
    return [v, mu * ((1 - x**2) * v - x)]

mu = 100.0
y0 = [2.0, 0.0]
t_span = (0.0, 300.0)
t_eval = np.linspace(*t_span, 3000)

results = {}
for name, method in [("Radau (implicit, stiff)", "Radau")]:
    t0 = time.perf_counter()
    sol = solve_ivp(vdp, t_span, y0, args=(mu,), method=method,
                     t_eval=t_eval, rtol=1e-6, atol=1e-9)
    t1 = time.perf_counter()
    results[name] = dict(sol=sol, wall=t1 - t0)
    print(f"{name:28s} success={sol.success} nfev={sol.nfev:7d} "
          f"njev={getattr(sol,'njev',0):5d} nsteps={len(sol.t):6d} wall={t1-t0:.3f}s")

# ---- Solution trajectory plot (only implicit solver, since it's the one that succeeds cleanly) ----
sol_imp = results["Radau (implicit, stiff)"]["sol"]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(sol_imp.t, sol_imp.y[0], lw=1.2)
ax.set_xlabel("time $t$")
ax.set_ylabel("$x(t)$")
ax.set_title(f"Van der Pol oscillator, $\\mu={mu:.0f}$ (relaxation oscillation)")
fig.tight_layout()
fig.savefig("vdp_trajectory.png", dpi=150)
plt.close(fig)

# ---- Phase portrait ----
fig, ax = plt.subplots(figsize=(5.5, 5.2))
ax.plot(sol_imp.y[0], sol_imp.y[1], lw=0.8)
ax.set_xlabel("$x$")
ax.set_ylabel("$v = dx/dt$")
ax.set_title("Phase portrait (limit cycle)")
fig.tight_layout()
fig.savefig("vdp_phase.png", dpi=150)
plt.close(fig)

# ---- Cost comparison across mu (short fixed horizon, no artificial step cap) ----
mus = [1, 10, 50, 100, 300, 600, 1000, 2000]
cost_explicit_nfev = []
cost_implicit_nfev = []
cost_explicit_steps = []
cost_implicit_steps = []
for m in mus:
    t_span_m = (0.0, 4.0)
    t0 = time.perf_counter()
    sol_e = solve_ivp(vdp, t_span_m, y0, args=(m,), method="RK45",
                       rtol=1e-6, atol=1e-9)
    t1 = time.perf_counter()
    cost_explicit_nfev.append(sol_e.nfev if sol_e.success else np.nan)
    cost_explicit_steps.append(len(sol_e.t))

    t0 = time.perf_counter()
    sol_i = solve_ivp(vdp, t_span_m, y0, args=(m,), method="Radau",
                       rtol=1e-6, atol=1e-9)
    t1 = time.perf_counter()
    cost_implicit_nfev.append(sol_i.nfev)
    cost_implicit_steps.append(len(sol_i.t))
    print(f"mu={m:5d}  RK45 nfev={cost_explicit_nfev[-1]:>8} steps={cost_explicit_steps[-1]:>6}  "
          f"Radau nfev={cost_implicit_nfev[-1]:>6} steps={cost_implicit_steps[-1]:>5}")

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.semilogy(mus, cost_explicit_nfev, "o-", label="RK45 (explicit)")
ax.semilogy(mus, cost_implicit_nfev, "s-", label="Radau (implicit)")
ax.set_xlabel(r"stiffness parameter $\mu$")
ax.set_ylabel("number of RHS evaluations (log scale)")
ax.set_title("Solver cost vs. stiffness, fixed accuracy tolerance")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("cost_vs_stiffness.png", dpi=150)
plt.close(fig)

with open("results.txt", "w") as f:
    for m, ne, ni in zip(mus, cost_explicit_nfev, cost_implicit_nfev):
        f.write(f"mu={m} nfev_explicit={ne} nfev_implicit={ni}\n")

print("done")
