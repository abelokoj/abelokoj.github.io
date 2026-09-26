"""
plot_bench_p7.py
================
Draws the measured strong- and weak-scaling figures for the post from
bench_p7.json (written by bench_mpi_p7.py). Uses the same plot style as
hpc_domain_decomposition_p7.py and saves PNG and SVG copies.

    python3 plot_bench_p7.py [bench_p7.json]
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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


def savefig_all(fig, basename):
    fig.savefig(f"{basename}.png")
    fig.savefig(f"{basename}.svg")


def amdahl(p, f):
    return 1.0 / ((1 - f) + f / p)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "bench_p7.json")
    d = json.load(open(path))
    phys = d["system"]["physical_cores"]

    # ---------------- strong scaling ----------------
    s = d["strong"]
    P = np.array([r["ranks"] for r in s])
    T = np.array([r["wall_s"] for r in s])
    C = np.array([r["comm_s_max"] for r in s])
    S = T[0] / T
    # Amdahl fit: least squares for the parallel fraction f on the measured speedups
    fs = np.linspace(0.5, 0.99999, 20000)
    f_fit = fs[np.argmin([np.sum((amdahl(P, f) - S) ** 2) for f in fs])]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    ax.plot(P, P, "k--", lw=1.1, label="ideal, $S(P) = P$")
    ax.plot(P, S, "o-", color="tab:blue", label="measured")
    pp = np.linspace(1, P.max(), 200)
    ax.plot(pp, amdahl(pp, f_fit), ":", color="tab:red", lw=1.4,
            label=rf"Amdahl fit, $f = {f_fit:.3f}$")
    if P.max() > phys:
        ax.axvspan(phys, P.max(), color="0.9", zorder=0, label="hardware threads")
    ax.set_xlabel("MPI ranks $P$")
    ax.set_ylabel("speedup $T(1)/T(P)$")
    g = s[0]
    ax.set_title(f"Strong scaling, {g['nx']}$\\times${g['ny']} grid")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.bar(P - 0.2, T - C, width=0.4, label="computation", color="tab:blue")
    ax.bar(P + 0.2, C, width=0.4, label="halo exchange", color="tab:orange")
    ax.set_xlabel("MPI ranks $P$")
    ax.set_ylabel("time on slowest rank (s)")
    ax.set_title("Where the time goes")
    ax.set_xticks(P)
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig_all(fig, "strong_scaling_measured_p7")
    plt.close(fig)

    # ---------------- weak scaling ----------------
    w = d["weak"]
    Pw = np.array([r["ranks"] for r in w])
    Tw = np.array([r["wall_s"] for r in w])
    E = Tw[0] / Tw
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(Pw, np.ones_like(Pw, dtype=float), "k--", lw=1.1, label="ideal")
    ax.plot(Pw, E, "o-", color="tab:blue", label="measured")
    if Pw.max() > phys:
        ax.axvspan(phys, Pw.max(), color="0.9", zorder=0, label="hardware threads")
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("MPI ranks $P$")
    ax.set_ylabel("weak-scaling efficiency $T(1)/T(P)$")
    ax.set_title(f"Weak scaling, {d['settings']['rows_per_rank']} rows per rank")
    ax.legend(frameon=False)
    ax.yaxis.set_major_formatter(CMTickFormatter())
    fig.tight_layout()
    savefig_all(fig, "weak_scaling_measured_p7")
    plt.close(fig)

    print(f"Amdahl fit f = {f_fit:.4f}")
    for p, sp, t in zip(P, S, T):
        print(f"strong P={p:2d}: T={t:.3f}s  S={sp:.2f}  efficiency={sp/p:.2f}")
    for p, e in zip(Pw, E):
        print(f"weak   P={p:2d}: efficiency={e:.2f}")


if __name__ == "__main__":
    main()
