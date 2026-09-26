"""
EXAMPLE 7.3 -- Nonlinear KdV with a SMALL dispersion coefficient
=================================================================
Reproduces, from Salian, Samala & Ghosh (2026):
    Figure 8    single soliton propagation,      Eq. (7.6), N = 80
    Figure 9    double soliton collision,        Eq. (7.7), N = 100
    Figure 10   triple soliton splitting,        Eq. (7.8), N = 150,
                with and without the F12 filter

This example has no associated table.

------------------------------------------------------------------------
THE PROBLEM, Eq. (7.5)
------------------------------------------------------------------------
        u_t + (u^2 / 2)_x + eps * u_xxx = 0

matching the general form u_t + g(u)_x + D*u_xxx = 0 with

        g(u) = u^2 / 2   so  g'(u) = u,   max|g'| = max|u|
        D    = eps       so  f(u)  = eps*u,  max|f'| = eps

------------------------------------------------------------------------
WHY "SMALL EPS" IS THE HARD CASE
------------------------------------------------------------------------
In Example 7.2 the dispersion coefficient was 1 and the soliton was a
wide, gentle bump. Here eps is between 1e-4 and 5e-4, which is tiny, and
that changes the character of the problem completely.

Weak dispersion means very little is holding back the nonlinear
steepening, so the structures that form are NARROW and TALL. Look at the
soliton width parameter used below, k = 0.5*sqrt(c/eps): as eps shrinks,
k grows, and the bump becomes a thin spike. Resolving a thin spike takes
many grid points across a small region, which is precisely the regime
where a scheme's short-wavelength resolution decides whether the answer
is right. That is exactly what the paper's Figures 2 and 3 measure, so
this example is where those spectral plots meet a real computation.

------------------------------------------------------------------------
THE THREE SUB-CASES
------------------------------------------------------------------------
Section 7.1 / Figure 8   ONE soliton travelling. There is an exact
                         solution, so error can be measured.

Section 7.2 / Figure 9   TWO solitons of different heights. The taller
                         one moves faster (that is a property of KdV
                         solitons) and catches the shorter one. The
                         remarkable fact is that they pass THROUGH each
                         other and re-emerge unchanged, merely shifted.

Section 7.3 / Figure 10  ONE hump that is not a soliton, which therefore
                         splits into several solitons plus a train of
                         small ripples. This is the case where filtering
                         matters, so it is run twice, filtered and not.

------------------------------------------------------------------------
ABOUT THE FILTER (used only in Figure 10)
------------------------------------------------------------------------
High-order schemes like these add no damping of their own. That is
usually a virtue, but it means small high-frequency wiggles, produced by
discretization error near steep gradients, are never removed and can
grow. The remedy in Section 5 is a low-pass filter: it leaves smooth,
well-resolved parts of the solution essentially untouched while damping
the shortest wavelengths the grid can represent.

The paper applies it every 20 steps for TDCNCS but only every 50 steps
for TDCCS, and says so explicitly. The reason is the point of the paper:
TDCCS resolves short waves better, so it generates fewer spurious wiggles
and needs less help.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (enables 3d projection)

from pub_style import (apply_style, savefig_all, FIGDIR,
                       SURFACE_CMAP, matlab_view, subcaption)
from nonlinear_kdv import solve_scalar

apply_style()
OUT = FIGDIR


def g_of_u(u):
    """Flux function for Eq. (7.5): g(u) = u^2/2, shared by all three cases."""
    return 0.5 * u**2


# ======================================================================
# SECTION 7.1 / FIGURE 8 : single soliton propagation, Eq. (7.6)
# ======================================================================
EPS_1 = 5.0e-4      # dispersion coefficient
C_1 = 0.3           # soliton speed
X0_1 = 0.5          # where the soliton starts
K_1 = 0.5 * np.sqrt(C_1 / EPS_1)    # width parameter; large k = narrow spike


def u0_single(x):
    """Initial condition, Eq. (7.6): one soliton of height 3c centred at x0."""
    return 3.0 * C_1 / np.cosh(K_1 * (x - X0_1))**2


def u_exact_single(x, t):
    """Exact solution: the same soliton, translated right at speed c.

    Only the argument changes, from (x - x0) to (x - x0) - c*t. Height and
    width are untouched, which is what makes it a soliton.
    """
    return 3.0 * C_1 / np.cosh(K_1 * ((x - X0_1) - C_1 * t))**2


def figure8(N=80):
    """Solution and pointwise error at t = 0, 1, 2, 3, for both schemes.

    Expect the TDCCS error to be several times smaller than the TDCNCS
    error at every time. Because the soliton here is narrow (k is about
    12), this is a genuine test of short-wave resolution rather than of
    formal order alone.
    """
    times = [0.0, 1.0, 2.0, 3.0]
    colors = ["k", "b", "g", "r"]
    xlo, xhi = 0.0, 2.0
    T = max(times)

    # Bounds for the time step. The soliton conserves its amplitude, so
    # max|u| stays at its initial value 3c throughout the run.
    max_gp = 3.0 * C_1
    max_fp = EPS_1

    results = {}
    for scheme in ("TDCNCS", "TDCCS"):
        sol = solve_scalar(scheme, xlo, xhi, N, u0_single, g_of_u, EPS_1, T,
                           max_gp, max_fp, sample_times=times)
        results[scheme] = (sol["x"], sol["samples"])
        print(f"  Figure 8: {scheme} done ({sol['nsteps']} steps)", flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, scheme in enumerate(("TDCNCS", "TDCCS")):
        x, samples = results[scheme]
        for t, cc in zip(times, colors):
            u = samples[t]
            ex = u_exact_single(x, t)
            axes[0, col].plot(x, ex, "-", color=cc, lw=1.2)       # exact
            axes[0, col].plot(x, u, "o", ms=3, mfc="none", color=cc)  # computed
            axes[1, col].plot(x, np.abs(ex - u), color=cc, lw=1.0)    # error
        subcaption(axes[0, col],
                   f"({'a' if col == 0 else 'b'}) {scheme} - Numerical solution")
        axes[0, col].set_xlabel("$x$"); axes[0, col].set_ylabel("$u(x,t)$")
        axes[0, col].set_xlim(xlo, xhi)
        axes[1, col].set_xlim(xlo, xhi)
        subcaption(axes[1, col],
                   f"({'c' if col == 0 else 'd'}) {scheme} - Pointwise error")
        axes[1, col].set_xlabel("$x$"); axes[1, col].set_ylabel("Error")
    fig.tight_layout()
    savefig_all(fig, "figure8_soliton_illustrative", outdir=OUT)
    plt.close(fig)
    print("Figure 8 written.")


# ======================================================================
# SECTION 7.2 / FIGURE 9 : double soliton collision, Eq. (7.7)
# ======================================================================
EPS_2 = 4.84e-4
C1_2, C2_2 = 0.3, 0.1       # speeds: the first soliton is three times faster
X1_2, X2_2 = 0.4, 0.8       # starting positions: the fast one starts behind


def u0_double(x):
    """Initial condition, Eq. (7.7): two solitons, simply added together.

    Each has its own speed c_j and its own width k_j = 0.5*sqrt(c_j/eps).
    Taller solitons are both faster AND narrower, so the one at x = 0.4
    with c = 0.3 will overtake the one at x = 0.8 with c = 0.1.

    Adding two solitons is not itself an exact solution of a nonlinear
    equation, but when they start far enough apart the overlap is
    negligible and it serves as a valid initial condition.
    """
    k1 = 0.5 * np.sqrt(C1_2 / EPS_2)
    k2 = 0.5 * np.sqrt(C2_2 / EPS_2)
    return (3.0 * C1_2 / np.cosh(k1 * (x - X1_2))**2
            + 3.0 * C2_2 / np.cosh(k2 * (x - X2_2))**2)


def figure9(N=100):
    """Snapshots at t = 0, 1, 2 plus a space-time surface out to t = 4.

    There is no exact solution to compare against here, so the figure
    shows the solution itself rather than an error. What to look for: the
    two peaks approach, merge into a single lump, then separate again with
    their original heights intact. Surviving a collision unchanged is the
    defining property of solitons.

    The surface panel plots u against both x and t at once, which makes
    the crossing easy to see: two ridges that meet and continue.
    """
    xlo, xhi = 0.0, 2.0
    T = 4.0
    snap_times = [0.0, 1.0, 2.0]

    # During the collision the peak can exceed either soliton alone, so
    # the CFL bound uses the SUM of the two amplitudes rather than the
    # larger of them. Underestimating this bound would risk instability.
    max_gp = 3.0 * (C1_2 + C2_2)
    max_fp = EPS_2

    # For the surface we need many snapshots, closely spaced in time.
    surf_times = list(np.linspace(0.0, T, 81))
    all_times = sorted(set(snap_times + surf_times))

    results = {}
    for scheme in ("TDCNCS", "TDCCS"):
        sol = solve_scalar(scheme, xlo, xhi, N, u0_double, g_of_u, EPS_2, T,
                           max_gp, max_fp, sample_times=all_times)
        results[scheme] = (sol["x"], sol["samples"])
        print(f"  Figure 9: {scheme} done ({sol['nsteps']} steps)", flush=True)

    # Paper layout, Figure 9: two rows (TDCNCS on top, TDCCS below) by
    # four columns. The first three columns are snapshots at t = 0, 1, 2;
    # the fourth is a space-time surface at t = 4. Panel labels sit BELOW
    # each axis, as in the paper, and each 1-D panel carries a small
    # legend box naming the scheme.
    fig = plt.figure(figsize=(15, 6.4))
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    idx = 0
    for row, scheme in enumerate(("TDCNCS", "TDCCS")):
        x, samples = results[scheme]
        color = "k" if scheme == "TDCNCS" else "tab:blue"
        for t in snap_times:
            ax = fig.add_subplot(2, 4, row * 4 + idx % 4 + 1)
            ax.plot(x, samples[t], color=color, lw=1.1, label=scheme)
            ax.set_xlabel("$x$")
            ax.set_ylabel("$u(x,t)$")
            ax.set_xlim(xlo, xhi)
            ax.set_ylim(-0.1, 1.0)
            ax.legend(loc="upper right", fontsize=8, framealpha=1.0)
            subcaption(ax, f"{labels[idx]} $t = {t:g}$")
            idx += 1

        # Space-time surface, MATLAB camera and jet colormap to match.
        ax = fig.add_subplot(2, 4, row * 4 + 4, projection="3d")
        Tg, Xg = np.meshgrid(surf_times, x, indexing="ij")
        Z = np.array([samples[t] for t in surf_times])
        surf = ax.plot_surface(Xg, Tg, Z, cmap=SURFACE_CMAP, linewidth=0,
                               antialiased=True, rstride=1, cstride=1)
        matlab_view(ax)
        ax.set_xlabel("$x$"); ax.set_ylabel("$t$"); ax.set_zlabel("$u(x,t)$")
        ax.set_xlim(xlo, xhi); ax.set_ylim(0, T)
        fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.10)
        subcaption(ax, f"{labels[idx]} $t = {T:g}$", y=-0.16)
        idx += 1

    fig.tight_layout()
    savefig_all(fig, "figure9_double_soliton", outdir=OUT)
    plt.close(fig)
    print("Figure 9 written.")


# ======================================================================
# SECTION 7.3 / FIGURE 10 : triple soliton splitting, Eq. (7.8)
# ======================================================================
EPS_3 = 1.0e-4
ALPHA_F = 0.4                                 # filter free parameter, |a_F| < 0.5
FILTER_EVERY = {"TDCNCS": 20, "TDCCS": 50}    # paper, Section 7.3


def u0_triple(x):
    """Initial condition, Eq. (7.8): a single hump that is NOT a soliton.

    A soliton needs a specific relationship between its height and its
    width. This hump does not satisfy it, so it cannot travel unchanged.
    Instead it breaks up into several genuine solitons, which then
    separate because taller ones move faster, trailed by a train of small
    dispersive ripples.
    """
    return (2.0 / 3.0) / np.cosh((x - 1.0) / np.sqrt(108.0 * EPS_3))**2


def figure10(N=150):
    """Splitting, computed four ways: each scheme with and without the filter.

    Four runs in total. The comparison to look for:

      TDCNCS unfiltered   develops visible ripples ahead of the main pulse
      TDCNCS filtered     those ripples are damped away
      TDCCS unfiltered    already fairly smooth without any help
      TDCCS filtered      only a small further change

    That asymmetry is the paper's argument for the new scheme: it needs
    less artificial help to stay clean, which is why it can be filtered
    less often (every 50 steps rather than every 20).
    """
    xlo, xhi = 0.0, 3.0
    T = 4.0
    snap_times = [0.0, 1.0, 2.0]
    surf_times = list(np.linspace(0.0, T, 81))
    all_times = sorted(set(snap_times + surf_times))

    # The initial hump splits into solitons TALLER than itself, so the CFL
    # bound must allow for that growth rather than using the initial peak
    # of 2/3. The value 1.0 covers the whole run.
    max_gp = 1.0
    max_fp = EPS_3

    results = {}
    for scheme in ("TDCNCS", "TDCCS"):
        for filtered in (False, True):
            fe = FILTER_EVERY[scheme] if filtered else None
            sol = solve_scalar(scheme, xlo, xhi, N, u0_triple, g_of_u, EPS_3,
                               T, max_gp, max_fp, filter_every=fe,
                               alphaF=ALPHA_F, sample_times=all_times)
            results[(scheme, filtered)] = (sol["x"], sol["samples"])
            tag = "F12" if filtered else "no filter"
            print(f"  Figure 10: {scheme} ({tag}) done "
                  f"({sol['nsteps']} steps)", flush=True)

    # Paper layout, Figure 10: three rows. The first row is TDCNCS at
    # t = 0, 1, 2, each panel overlaying the unfiltered run (solid) with
    # the F12-filtered run (dotted red). The second row is the same for
    # TDCCS. The third row holds two large space-time surfaces at t = 4,
    # one per scheme, computed without the filter.
    fig = plt.figure(figsize=(14, 12))
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    idx = 0
    for row, scheme in enumerate(("TDCNCS", "TDCCS")):
        base_color = "k" if scheme == "TDCNCS" else "tab:blue"
        x, s_un = results[(scheme, False)]
        _, s_fl = results[(scheme, True)]
        for j, t in enumerate(snap_times):
            ax = fig.add_subplot(3, 3, row * 3 + j + 1)
            ax.plot(x, s_un[t], color=base_color, lw=1.1, label=scheme)
            ax.plot(x, s_fl[t], ":", color="r", lw=1.1, label=f"{scheme}-F12")
            ax.set_xlabel("$x$"); ax.set_ylabel("$u(x,t)$")
            ax.set_xlim(xlo, xhi); ax.set_ylim(-0.1, 1.0)
            ax.legend(loc="upper left", fontsize=7.5, framealpha=1.0)
            subcaption(ax, f"{labels[idx]} $t = {t:g}$")
            idx += 1

    # Bottom row: one wide surface per scheme, unfiltered.
    for col, scheme in enumerate(("TDCNCS", "TDCCS")):
        x, s_un = results[(scheme, False)]
        ax = fig.add_subplot(3, 2, 5 + col, projection="3d")
        Tg, Xg = np.meshgrid(surf_times, x, indexing="ij")
        Z = np.array([s_un[t] for t in surf_times])
        surf = ax.plot_surface(Xg, Tg, Z, cmap=SURFACE_CMAP, linewidth=0,
                               antialiased=True, rstride=1, cstride=1)
        matlab_view(ax)
        ax.set_xlabel("$x$"); ax.set_ylabel("$t$"); ax.set_zlabel("$u(x,t)$")
        ax.set_xlim(xlo, xhi); ax.set_ylim(0, T)
        fig.colorbar(surf, ax=ax, shrink=0.60, pad=0.10)
        subcaption(ax, f"{labels[6 + col]} $t = {T:g}$ ({scheme})", y=-0.16)

    fig.tight_layout()
    savefig_all(fig, "figure10_filter_effect", outdir=OUT)
    plt.close(fig)
    print("Figure 10 written.")


def main():
    print("Example 7.3: nonlinear KdV with a small dispersion coefficient\n")
    print("Figure 8 (single soliton) ...", flush=True)
    figure8()
    print("\nFigure 9 (double soliton collision) ...", flush=True)
    figure9()
    print("\nFigure 10 (triple soliton splitting, filter comparison) ...",
          flush=True)
    figure10()


if __name__ == "__main__":
    main()
