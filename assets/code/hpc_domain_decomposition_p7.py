"""
Domain decomposition demo: 2D linear wave equation (a simplified stand-in
for tsunami-wave propagation from a seafloor displacement source), solved
two ways:
  (1) monolithically, on the full grid, in one process
  (2) via 1D domain decomposition into P column-blocks, each with ghost-column
      halo exchange every time step -- simulating, in serial Python, exactly
      the communication pattern a real MPI implementation would use.

The point of the exercise is to verify that decomposition + halo exchange
reproduces the monolithic solution bit-for-bit (up to floating point
round-off), before reasoning about the *performance* of doing it this way
across real distributed hardware.
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


# =====================================================================
# Problem setup: 2D wave equation  u_tt = c^2 (u_xx + u_yy)
# Initial condition: a Gaussian "uplift" pulse, loosely modeling an
# offshore seafloor displacement source.
# =====================================================================
NX, NY = 240, 240
LX, LY = 1.0, 1.0
dx = LX / NX
dy = LY / NY
c = 1.0
CFL = 0.5
dt = CFL * min(dx, dy) / c
N_STEPS = 300

x = np.linspace(0, LX, NX)
y = np.linspace(0, LY, NY)
X, Y = np.meshgrid(x, y, indexing="ij")


def initial_pulse(amplitude=1.0, x0=0.3, y0=0.5, width=0.03):
    return amplitude * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * width**2))


# Sensor locations (analogous to seafloor pressure sensors in a real
# tsunami early-warning network)
sensor_locs = [(150, 60), (180, 120), (200, 180)]

# =====================================================================
# 1) Monolithic solve
# =====================================================================
def solve_monolithic(n_steps=N_STEPS, record_sensors=True):
    u_prev = initial_pulse()
    u_curr = u_prev.copy()   # zero initial velocity -> first step uses u_prev twice
    r2 = (c * dt / dx) ** 2  # assume dx == dy here
    sensor_traces = {loc: [] for loc in sensor_locs} if record_sensors else None

    for step in range(n_steps):
        lap = (
            np.roll(u_curr, 1, axis=0) + np.roll(u_curr, -1, axis=0)
            + np.roll(u_curr, 1, axis=1) + np.roll(u_curr, -1, axis=1)
            - 4 * u_curr
        )
        u_next = 2 * u_curr - u_prev + r2 * lap
        # Dirichlet (fixed) boundary -- absorbing edges would be more
        # physically realistic but add unneeded complexity for this demo
        u_next[0, :] = u_next[-1, :] = u_next[:, 0] = u_next[:, -1] = 0.0
        u_prev, u_curr = u_curr, u_next
        if record_sensors:
            for loc in sensor_locs:
                sensor_traces[loc].append(u_curr[loc])
    return u_curr, sensor_traces


# =====================================================================
# 2) Domain-decomposed solve: split rows (axis 0) into P blocks,
#    each with 1 ghost row on either side, exchanged every step.
# =====================================================================
def solve_decomposed(n_procs, n_steps=N_STEPS):
    r2 = (c * dt / dx) ** 2
    u0_full = initial_pulse()

    # Partition NX rows (axis 0) into n_procs contiguous blocks
    bounds = np.linspace(0, NX, n_procs + 1).astype(int)
    blocks_prev, blocks_curr = [], []
    for p in range(n_procs):
        lo, hi = bounds[p], bounds[p + 1]
        # local array padded with 1 ghost row on each side
        local = np.zeros((hi - lo + 2, NY))
        local[1:-1, :] = u0_full[lo:hi, :]
        blocks_prev.append(local.copy())
        blocks_curr.append(local.copy())

    for step in range(n_steps):
        # ---- Halo exchange: each rank sends its boundary row to neighbors,
        # receives into its own ghost rows. This is exactly the
        # communication pattern of MPI_Sendrecv between neighboring ranks. ----
        for p in range(n_procs):
            if p > 0:
                blocks_curr[p][0, :] = blocks_curr[p - 1][-2, :]   # recv from left neighbor
            if p < n_procs - 1:
                blocks_curr[p][-1, :] = blocks_curr[p + 1][1, :]   # recv from right neighbor

        # ---- Local compute: each rank updates only its own interior + owned rows ----
        new_blocks = []
        for p in range(n_procs):
            local_curr = blocks_curr[p]
            local_prev = blocks_prev[p]
            lap = np.zeros_like(local_curr)
            # interior in y always via roll (y-direction has no decomposition here)
            lap_y = np.roll(local_curr, 1, axis=1) + np.roll(local_curr, -1, axis=1) - 2 * local_curr
            lap_y[:, 0] = lap_y[:, -1] = 0.0   # y-boundary handled below anyway
            lap_x = np.zeros_like(local_curr)
            lap_x[1:-1, :] = local_curr[2:, :] + local_curr[:-2, :] - 2 * local_curr[1:-1, :]
            lap = lap_x + lap_y

            local_next = 2 * local_curr - local_prev + r2 * lap
            # global physical boundaries (only apply at actual domain edges, not ghost rows)
            local_next[:, 0] = local_next[:, -1] = 0.0
            lo, hi = bounds[p], bounds[p + 1]
            if lo == 0:
                local_next[1, :] = 0.0
            if hi == NX:
                local_next[-2, :] = 0.0
            new_blocks.append(local_next)

        blocks_prev = blocks_curr
        blocks_curr = new_blocks

    # Reassemble global field from blocks (owned rows only, excluding ghosts)
    u_full = np.zeros((NX, NY))
    for p in range(n_procs):
        lo, hi = bounds[p], bounds[p + 1]
        u_full[lo:hi, :] = blocks_curr[p][1:-1, :]
    return u_full


# =====================================================================
# Correctness check: does decomposition reproduce the monolithic result?
# =====================================================================
print("Running monolithic solve...")
u_mono, sensor_traces = solve_monolithic()

print("\nCorrectness check: decomposed solve vs monolithic solve")
for n_procs in [1, 2, 4, 8, 16]:
    u_decomp = solve_decomposed(n_procs)
    max_err = np.max(np.abs(u_decomp - u_mono))
    print(f"  P={n_procs:3d} subdomains: max abs difference = {max_err:.3e}")

# =====================================================================
# Figure 1: wave field snapshot + sensor traces
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
im = axes[0].imshow(u_mono.T, origin="lower", cmap="RdBu_r",
                     vmin=-0.15, vmax=0.15, extent=[0, LX, 0, LY])
for loc in sensor_locs:
    axes[0].plot(loc[0] / NX * LX, loc[1] / NY * LY, "k^", markersize=7)
axes[0].set_title(f"Wave field after {N_STEPS} steps (source + sensors)")
axes[0].set_xlabel(r"$x$"); axes[0].set_ylabel(r"$y$")
fig.colorbar(im, ax=axes[0], fraction=0.046)
cm_x(axes[0]); cm_y(axes[0])

for loc in sensor_locs:
    axes[1].plot(np.arange(N_STEPS) * dt, sensor_traces[loc],
                 label=rf"sensor at $({loc[0]/NX:.2f}, {loc[1]/NY:.2f})$")
axes[1].set_xlabel(r"time")
axes[1].set_ylabel(r"amplitude")
axes[1].set_title("Simulated sensor recordings")
axes[1].legend(fontsize=7.5, frameon=False)
cm_x(axes[1]); cm_y(axes[1])
fig.tight_layout()
savefig_all(fig, "wave_field_sensors_p7")
plt.close(fig)

# =====================================================================
# Figure 2: domain decomposition schematic + correctness table
# =====================================================================
n_procs_demo = 4
bounds_demo = np.linspace(0, NX, n_procs_demo + 1).astype(int)
fig, ax = plt.subplots(figsize=(5.0, 5.0))
ax.imshow(u_mono.T, origin="lower", cmap="RdBu_r", vmin=-0.15, vmax=0.15,
          extent=[0, LX, 0, LY])
for p in range(n_procs_demo):
    lo, hi = bounds_demo[p], bounds_demo[p + 1]
    ax.axvline(lo / NX * LX, color="k", lw=1.3)
    ax.text((lo + hi) / 2 / NX * LX, 0.95, rf"rank {p}", ha="center",
            color="black", fontsize=10, fontweight="bold")
ax.set_title(f"Domain decomposition into {n_procs_demo} subdomains\n"
             "(vertical lines = partition boundaries)")
ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
cm_x(ax); cm_y(ax)
fig.tight_layout()
savefig_all(fig, "domain_decomposition_p7")
plt.close(fig)

# =====================================================================
# Amdahl's Law and weak-scaling analysis (analytic -- no need for real
# parallel hardware to reason about these curves correctly)
# =====================================================================
def amdahl_speedup(p_procs, parallel_fraction):
    return 1.0 / ((1 - parallel_fraction) + parallel_fraction / p_procs)


procs = np.arange(1, 4097)
# Note: x-axis is log-scale (base 2), so only the linear y-axis gets the
# CM tick formatter.
fig, ax = plt.subplots(figsize=(6.4, 4.4))
for f in [0.5, 0.9, 0.95, 0.99, 0.999]:
    ax.plot(procs, amdahl_speedup(procs, f), label=rf"$f = {f}$")
ax.set_xscale("log", base=2)
ax.set_xlabel("number of processes (log scale)")
ax.set_ylabel("theoretical speedup")
ax.set_title("Amdahl's Law: speedup ceiling set by the serial fraction")
ax.legend(frameon=False, fontsize=8.5)
ax.grid(which="both")
cm_y(ax)
fig.tight_layout()
savefig_all(fig, "amdahl_law_p7")
plt.close(fig)


# Weak scaling: ideal (flat efficiency) vs. a realistic model with
# per-step communication overhead that grows with the number of neighbors
def weak_scaling_efficiency(p_procs, comm_fraction_per_proc=0.02):
    # crude model: each additional halo-exchange edge costs a fixed
    # overhead fraction of the local compute time
    overhead = comm_fraction_per_proc * np.log2(np.maximum(p_procs, 1))
    return 1.0 / (1.0 + overhead)


# Note: x-axis is log-scale (base 2), so only the linear y-axis gets the
# CM tick formatter.
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.plot(procs, np.ones_like(procs, dtype=float), "k--", lw=1.3,
        label="ideal (no communication cost)")
for cf in [0.01, 0.03, 0.06]:
    ax.plot(procs, weak_scaling_efficiency(procs, cf),
            label=rf"comm. overhead coeff. $= {cf}$")
ax.set_xscale("log", base=2)
ax.set_xlabel("number of processes (log scale)")
ax.set_ylabel("parallel efficiency")
ax.set_title("Weak scaling: efficiency degrades as halo-exchange overhead grows")
ax.legend(frameon=False, fontsize=8.5)
ax.grid(which="both")
cm_y(ax)
fig.tight_layout()
savefig_all(fig, "weak_scaling_p7")
plt.close(fig)

with open("results.txt", "w") as f:
    f.write("Correctness check (decomposed vs monolithic):\n")
    for n_procs in [1, 2, 4, 8, 16]:
        u_decomp = solve_decomposed(n_procs)
        max_err = np.max(np.abs(u_decomp - u_mono))
        f.write(f"P={n_procs}: max_err={max_err:.3e}\n")

print("done")