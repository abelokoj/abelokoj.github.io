"""
EXAMPLE 7.6 -- The ITO-TYPE COUPLED nonlinear system
=====================================================
Reproduces, from Salian, Samala & Ghosh (2026):
    Figure 14   trigonometric initial condition, Eq. (7.13),
                80 cells on [0, 2*pi], at t = 0, 0.5, 1
    Figure 15   Gaussian initial condition, Eq. (7.14),
                160 cells on [-15, 15], at t = 0, 1, 2

This example has no associated table, and no filtering is applied in
either case; the paper states that explicitly.

------------------------------------------------------------------------
THE SYSTEM, Eq. (7.12)
------------------------------------------------------------------------
        u_t - (3u^2 + v^2)_x - u_xxx = 0
        v_t - 2(u v)_x               = 0

This is the first example in the paper with TWO unknown functions rather
than one, and they are coupled: the equation for u contains v, and the
equation for v contains u. Neither can be solved on its own; both must be
marched forward together, in lockstep.

Rewriting each line in the general form u_t + g(...)_x + D*u_xxx = 0:

    u-equation:   g_u(u,v) = -(3u^2 + v^2),     D_u = -1
    v-equation:   g_v(u,v) = -2*u*v,            D_v =  0

------------------------------------------------------------------------
THE KEY STRUCTURAL POINT
------------------------------------------------------------------------
Look at D_v: it is ZERO. Only the u-equation has a third derivative at
all. The v-equation is pure nonlinear advection, with no dispersion
whatsoever.

That asymmetry drives everything one sees in the figures. Dispersion is
what prevents a wave from breaking, so:

    u  keeps its dispersion and therefore stays smooth, spreading into
       oscillatory wave structures;
    v  has none, so nothing stops it steepening, and it develops sharp,
       shock-like fronts.

The paper reports exactly this, and adds that TDCCS produces smoother,
more stable results near those steep gradients in v, while TDCNCS tends
to introduce mild spurious oscillations there.

------------------------------------------------------------------------
HOW THE STATE IS ORGANIZED
------------------------------------------------------------------------
TDCNCS, node values only:
        W[0] = u at the nodes
        W[1] = v at the nodes                         shape (2, N)

TDCCS, nodes and centers for BOTH unknowns:
        W[0] = u at the nodes        W[1] = u at the centers
        W[2] = v at the nodes        W[3] = v at the centers
                                                      shape (4, N)

Stacking everything into one array lets the same Runge-Kutta stepper
advance the whole system in a single call, with no special handling.

------------------------------------------------------------------------
RUNTIME
------------------------------------------------------------------------
Figure 14 is the slower of the two: the domain is short and the grid
relatively fine, so dx is small, and dt shrinks like dx^3.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3d projection)

from pub_style import (apply_style, savefig_all, FIGDIR,
                       SURFACE_CMAP, matlab_view, subcaption)
from nonlinear_kdv import periodic_grid, cfl_dt, CFL, SCHEME_KEY
from tdccs_lib import (TDCNCS, TDCCS,
                       third_derivative_TDCNCS, third_derivative_TDCCS,
                       first_derivative_Lele_CNCS8, first_derivative_Liu_CCS8,
                       tvdrk3_step)

apply_style()
OUT = FIGDIR

# Both components steepen considerably during the run, so the convective
# bound estimated from the initial data alone would be too optimistic.
# Inflating it by this factor keeps the fixed time step inside the
# stability region for the whole simulation. Raise it if a run goes
# unstable at a grid size not tested here.
CFL_SAFETY = 2.5


# ======================================================================
# RIGHT-HAND SIDES FOR THE COUPLED SYSTEM
# ======================================================================
def rhs_ito_TDCNCS(dx):
    """Build S(W) for the node-only scheme. W is (2, N): rows u and v."""
    c3 = TDCNCS[SCHEME_KEY]

    def rhs(W):
        u, v = W[0], W[1]

        # Each flux is formed pointwise from BOTH unknowns, then
        # differentiated once in space. This is where the coupling enters:
        # the u-flux contains v, and the v-flux contains u.
        gu_x = first_derivative_Lele_CNCS8(-(3.0 * u**2 + v**2), dx)
        gv_x = first_derivative_Lele_CNCS8(-2.0 * u * v, dx)

        # Only u carries a third derivative.
        uxxx = third_derivative_TDCNCS(u, dx, c3)

        # Move everything to the right-hand side. D_u = -1 for the
        # u-equation, and the v-equation has no dispersive term at all.
        du = -(gu_x + (-1.0) * uxxx)
        dv = -gv_x
        return np.stack([du, dv])

    return rhs


def rhs_ito_TDCCS(dx):
    """Build S(W) for the coupled node+center scheme.

    W is (4, N): [u_node, u_half, v_node, v_half].

    Both derivative operators are coupled across the node and center
    grids, so each returns a pair, and both members of the pair are
    needed because both are genuine evolved unknowns.
    """
    c3 = TDCCS[SCHEME_KEY]

    def rhs(W):
        un, uh, vn, vh = W[0], W[1], W[2], W[3]

        # Fluxes are evaluated on the node grid and the center grid
        # independently, then handed to the coupled derivative operator.
        gu_n, gu_h = first_derivative_Liu_CCS8(-(3.0 * un**2 + vn**2),
                                               -(3.0 * uh**2 + vh**2), dx)
        gv_n, gv_h = first_derivative_Liu_CCS8(-2.0 * un * vn,
                                               -2.0 * uh * vh, dx)
        d3n, d3h = third_derivative_TDCCS(un, uh, dx, c3)

        dun = -(gu_n + (-1.0) * d3n)
        duh = -(gu_h + (-1.0) * d3h)
        dvn = -gv_n            # no dispersion in the v-equation
        dvh = -gv_h
        return np.stack([dun, duh, dvn, dvh])

    return rhs


def solve_ito(scheme, xlo, xhi, N, u0f, v0f, T, sample_times):
    """Integrate the coupled system and return (x, {time: (u, v)})."""
    x, xh, dx = periodic_grid(xlo, xhi, N)
    u0, v0 = u0f(x), v0f(x)

    # ---- time step, Eq. (7.1) ---------------------------------------
    # For a system the convective bound comes from the largest of the
    # partial derivatives of the fluxes with respect to the unknowns:
    #   d/du of (3u^2 + v^2) = 6u,   d/dv of (3u^2 + v^2) = 2v
    #   d/du of (2uv)        = 2v,   d/dv of (2uv)        = 2u
    # We take the largest plausible combination and then apply the safety
    # factor, since both components grow as they steepen.
    max_gp = max(6.0 * np.abs(u0).max() + 2.0 * np.abs(v0).max(),
                 2.0 * np.abs(u0).max() + 2.0 * np.abs(v0).max())
    max_gp = CFL_SAFETY * max(max_gp, 1e-12)   # guard against a zero bound
    dt = cfl_dt(dx, max_gp, 1.0, CFL)          # |f'| = 1 from the u-equation

    # ---- assemble the initial state ---------------------------------
    if scheme == "TDCNCS":
        W = np.stack([u0, v0])
        rhs = rhs_ito_TDCNCS(dx)
        # Extract (u, v) from a state for plotting: rows 0 and 1.
        pick = lambda S: (S[0].copy(), S[1].copy())
    else:
        # Sample the initial condition on the centers as well as the nodes.
        W = np.stack([u0, u0f(xh), v0, v0f(xh)])
        rhs = rhs_ito_TDCCS(dx)
        # Node components are rows 0 (u) and 2 (v); rows 1 and 3 hold the
        # center values, which are evolved but not plotted.
        pick = lambda S: (S[0].copy(), S[2].copy())

    # ---- march, stopping exactly on each requested time --------------
    # Same reasoning as nonlinear_kdv.integrate: sampling only to within
    # half a step would introduce a timing offset that looks like error.
    stops = sorted(sample_times)
    samples = {}
    t_prev = 0.0
    nsteps = 0

    if abs(stops[0]) < 1e-14:
        samples[stops[0]] = pick(W)        # record the initial condition

    for t_stop in stops:
        seg = t_stop - t_prev
        if seg <= 1e-14:
            continue
        nseg = max(1, int(np.ceil(seg / dt)))   # whole steps, rounded up
        dt_seg = seg / nseg                     # shrink to land exactly
        for _ in range(nseg):
            W = tvdrk3_step(W, dt_seg, rhs)
            nsteps += 1
        samples[t_stop] = pick(W)
        t_prev = t_stop

    print(f"    {scheme}: {nsteps} steps, dt = {dt:.3e}", flush=True)
    return x, samples


# ======================================================================
# SHARED FIGURE BUILDER FOR FIGURES 14 AND 15
# ======================================================================
def _ito_figure(basename, xlo, xhi, N, u0f, v0f, snap_times, title):
    """Draw one of the paper's four-by-four Ito panels.

    Paper layout: four rows by four columns. The rows are, from top,
    TDCNCS u, TDCNCS v, TDCCS u, TDCCS v. The first three columns are
    snapshots at the requested times and the fourth is a space-time
    surface at the final time, drawn with the same jet colormap and
    MATLAB camera angle used throughout the paper. Panel labels sit
    below each axis and each line panel carries a small legend box.

    Stacking u directly above v for each scheme is what makes the central
    contrast legible: only the u equation carries a third derivative, so
    the u rows stay smooth and dispersive while the v rows steepen into
    shock-like fronts.
    """
    T = max(snap_times)
    surf_times = list(np.linspace(0.0, T, 61))
    all_times = sorted(set(list(snap_times) + surf_times))

    results = {}
    for scheme in ("TDCNCS", "TDCCS"):
        x, samples = solve_ito(scheme, xlo, xhi, N, u0f, v0f, T, all_times)
        results[scheme] = (x, samples)

    fig = plt.figure(figsize=(16, 13))
    rows = [("TDCNCS", 0, "$u(x,t)$", "k"),
            ("TDCNCS", 1, "$v(x,t)$", "k"),
            ("TDCCS", 0, "$u(x,t)$", "tab:blue"),
            ("TDCCS", 1, "$v(x,t)$", "tab:blue")]
    labels = "abcdefghijklmnop"
    idx = 0

    for r, (scheme, comp, ylab, color) in enumerate(rows):
        x, samples = results[scheme]
        # Common y range per row, so the three snapshots are comparable.
        vals = np.concatenate([samples[t][comp] for t in snap_times])
        lo, hi = float(vals.min()), float(vals.max())
        pad = 0.08 * (hi - lo if hi > lo else 1.0)

        for t in snap_times:
            ax = fig.add_subplot(4, 4, r * 4 + idx % 4 + 1)
            ax.plot(x, samples[t][comp], color=color, lw=1.0, label=scheme)
            ax.set_xlabel("$x$"); ax.set_ylabel(ylab)
            ax.set_xlim(xlo, xhi); ax.set_ylim(lo - pad, hi + pad)
            ax.legend(loc="upper left", fontsize=7, framealpha=1.0)
            subcaption(ax, f"({labels[idx]}) $t = {t:g}$")
            idx += 1

        ax = fig.add_subplot(4, 4, r * 4 + 4, projection="3d")
        Tg, Xg = np.meshgrid(surf_times, x, indexing="ij")
        Z = np.array([samples[t][comp] for t in surf_times])
        surf = ax.plot_surface(Xg, Tg, Z, cmap=SURFACE_CMAP, linewidth=0,
                               antialiased=True, rstride=1, cstride=1)
        matlab_view(ax)
        ax.set_xlabel("$x$"); ax.set_ylabel("$t$"); ax.set_zlabel(ylab)
        ax.set_xlim(xlo, xhi); ax.set_ylim(0, T)
        fig.colorbar(surf, ax=ax, shrink=0.60, pad=0.10)
        subcaption(ax, f"({labels[idx]}) $t = {T:g}$", y=-0.16)
        idx += 1

    fig.tight_layout()
    savefig_all(fig, basename, outdir=OUT)
    plt.close(fig)
    print(f"{basename} written.")


# ======================================================================
# FIGURE 14 : trigonometric initial condition, Eq. (7.13)
# ======================================================================
def figure14(N=80):
    """u(x,0) = v(x,0) = cos(x) on [0, 2*pi].

    Both components start identical, yet they diverge immediately, purely
    because their equations differ. By t = 1 the v-component has grown a
    sharp spike while u remains a smooth dispersive wave.
    """
    _ito_figure("figure14_ito_trig",
                xlo=0.0, xhi=2.0 * np.pi, N=N,
                u0f=lambda x: np.cos(x),
                v0f=lambda x: np.cos(x),
                snap_times=[0.0, 0.5, 1.0],
                title="Ito-type coupled system, trigonometric initial "
                      "condition, Eq. (7.13)")


# ======================================================================
# FIGURE 15 : Gaussian initial condition, Eq. (7.14)
# ======================================================================
def figure15(N=160):
    """u(x,0) = v(x,0) = exp(-x^2) on the wider domain [-15, 15].

    A localized bump rather than a periodic wave. The wide domain gives
    the structures room to travel and separate without wrapping around
    and interfering with themselves.
    """
    _ito_figure("figure15_ito_gaussian",
                xlo=-15.0, xhi=15.0, N=N,
                u0f=lambda x: np.exp(-x**2),
                v0f=lambda x: np.exp(-x**2),
                snap_times=[0.0, 1.0, 2.0],
                title="Ito-type coupled system, Gaussian initial "
                      "condition, Eq. (7.14)")


def main():
    print("Example 7.6: Ito-type coupled nonlinear system\n")
    print("Figure 14 (trigonometric initial condition) ...", flush=True)
    figure14()
    print("\nFigure 15 (Gaussian initial condition) ...", flush=True)
    figure15()


if __name__ == "__main__":
    main()
