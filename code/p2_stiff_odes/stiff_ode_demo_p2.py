"""
Stiff ODE demo: the Van der Pol oscillator at large mu, comparing an
explicit solver (RK45) against an implicit, stiff-aware solver (Radau,
from the BDF/Runge-Kutta-implicit family).

This script produces four figures:
  1. vdp_trajectory_p2 -- x(t) showing relaxation-oscillation behavior
  2. vdp_phase_p2       -- phase portrait / limit cycle
  3. cost_vs_stiffness_p2 -- RHS evaluations needed by each solver vs. mu
  4. stability_regions_p2 -- absolute-stability regions of forward and
                              backward Euler in the complex z = h*lambda plane
"""
import numpy as np
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
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
          f"njev={getattr(sol, 'njev', 0):5d} nsteps={len(sol.t):6d} "
          f"wall={t1 - t0:.3f}s")

# =======================================================================
# 1) Solution trajectory (implicit solver only -- it is the one that
#    integrates the full horizon cleanly at this stiffness)
# =======================================================================
sol_imp = results["Radau (implicit, stiff)"]["sol"]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(sol_imp.t, sol_imp.y[0], color="tab:blue")
ax.set_xlabel(r"time $t$")
ax.set_ylabel(r"$x(t)$")
ax.set_title(rf"Van der Pol oscillator, $\mu = {mu:.0f}$ (relaxation oscillation)")
cm_x(ax)
cm_y(ax)
fig.tight_layout()
savefig_all(fig, "vdp_trajectory_p2")
plt.close(fig)

# =======================================================================
# 2) Phase portrait
# =======================================================================
fig, ax = plt.subplots(figsize=(4.8, 4.6))
ax.plot(sol_imp.y[0], sol_imp.y[1], lw=0.9, color="tab:blue")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$v = dx/dt$")
ax.set_title("Phase portrait (limit cycle)")
cm_x(ax)
cm_y(ax)
fig.tight_layout()
savefig_all(fig, "vdp_phase_p2")
plt.close(fig)

# =======================================================================
# 3) Cost comparison across mu (short fixed horizon, no artificial step cap)
# =======================================================================
mus = [1, 10, 50, 100, 300, 600, 1000, 2000]
cost_explicit_nfev, cost_implicit_nfev = [], []
cost_explicit_steps, cost_implicit_steps = [], []
for m in mus:
    t_span_m = (0.0, 4.0)
    sol_e = solve_ivp(vdp, t_span_m, y0, args=(m,), method="RK45",
                       rtol=1e-6, atol=1e-9)
    cost_explicit_nfev.append(sol_e.nfev if sol_e.success else np.nan)
    cost_explicit_steps.append(len(sol_e.t))

    sol_i = solve_ivp(vdp, t_span_m, y0, args=(m,), method="Radau",
                       rtol=1e-6, atol=1e-9)
    cost_implicit_nfev.append(sol_i.nfev)
    cost_implicit_steps.append(len(sol_i.t))
    print(f"mu={m:5d}  RK45 nfev={cost_explicit_nfev[-1]:>8} "
          f"steps={cost_explicit_steps[-1]:>6}  "
          f"Radau nfev={cost_implicit_nfev[-1]:>6} "
          f"steps={cost_implicit_steps[-1]:>5}")

# Note: y-axis is log-scale (semilogy), so only the linear x-axis gets
# the CM tick formatter -- ScalarFormatter is not appropriate on a log axis.
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.semilogy(mus, cost_explicit_nfev, "o-", ms=5, label="RK45 (explicit)")
ax.semilogy(mus, cost_implicit_nfev, "s-", ms=5, label="Radau (implicit)")
ax.set_xlabel(r"stiffness parameter $\mu$")
ax.set_ylabel("number of RHS evaluations (log scale)")
ax.set_title("Solver cost vs. stiffness, fixed accuracy tolerance")
ax.legend(frameon=False)
ax.grid(which="both")
cm_x(ax)
fig.tight_layout()
savefig_all(fig, "cost_vs_stiffness_p2")
plt.close(fig)

# =======================================================================
# 4) Absolute-stability regions of forward and backward Euler in the
#    complex z = h*lambda plane
# =======================================================================
theta = np.linspace(0, 2 * np.pi, 400)
# Forward Euler: |1 + z| <= 1  -> circle of radius 1 centered at z = -1
fe_circle = -1 + np.exp(1j * theta)
# Backward Euler boundary: |1 - z| = 1 -> circle of radius 1 centered at z = 1;
# the *stable* region is the exterior of this circle.
be_circle = 1 + np.exp(1j * theta)

fig, ax = plt.subplots(figsize=(5.4, 5.0))
ax.axhline(0, color="black", lw=0.6)
ax.axvline(0, color="black", lw=0.6)

ax.fill(fe_circle.real, fe_circle.imag, color="tab:orange", alpha=0.35,
        label="forward Euler: stable region")
ax.plot(fe_circle.real, fe_circle.imag, color="tab:orange", lw=1.3)

# Shade the (bounded rendering of the unbounded) backward-Euler stable region
# as the left half-plane minus the small disk it excludes, restricted to the
# plotted window for a clean figure.
xlim, ylim = (-4, 4), (-4, 4)
xx, yy = np.meshgrid(np.linspace(*xlim, 500), np.linspace(*ylim, 500))
zz = xx + 1j * yy
be_stable_mask = np.abs(1 - zz) >= 1
ax.contourf(xx, yy, be_stable_mask, levels=[0.5, 1.5], colors=["tab:blue"], alpha=0.18)
ax.plot(be_circle.real, be_circle.imag, color="tab:blue", lw=1.3,
        label="backward Euler: excluded disk (stable everywhere outside)")

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_xlabel(r"$\mathrm{Re}(z)$, $z = h\lambda$")
ax.set_ylabel(r"$\mathrm{Im}(z)$")
ax.set_title("Absolute-stability regions: forward vs. backward Euler")
ax.legend(frameon=False, loc="upper right", fontsize=8)
ax.set_aspect("equal")
cm_x(ax)
cm_y(ax)
fig.tight_layout()
savefig_all(fig, "stability_regions_p2")
plt.close(fig)

with open("results.txt", "w") as f:
    for m, ne, ni in zip(mus, cost_explicit_nfev, cost_implicit_nfev):
        f.write(f"mu={m} nfev_explicit={ne} nfev_implicit={ni}\n")

print("done")