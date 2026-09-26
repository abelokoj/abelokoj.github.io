"""
EXAMPLE 7.3 -- Nonlinear KdV with a SMALL dispersion coefficient
=================================================================
This script COMPUTES ONLY. It performs no plotting: every figure is drawn
by `make_figures.py` from the data files written here. Separating the two
matters because these solves are expensive (the time step scales like
dx^3), and redrawing a figure to change a colour or a label should never
require recomputing the solution.

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

from figdata import save_data
from nonlinear_kdv import solve_scalar



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


def figure8_data(N=80):
    """Compute and save the single-soliton case (paper Figure 8).

    Both schemes are run to t = 3 with snapshots at t = 0, 1, 2, 3, and
    the exact soliton is evaluated on the same grid at those times.
    """
    times = [0.0, 1.0, 2.0, 3.0]
    xlo, xhi = 0.0, 2.0
    T = max(times)
    max_gp = 3.0 * C_1        # the soliton conserves its amplitude
    max_fp = EPS_1

    out = {}
    for scheme in ("TDCNCS", "TDCCS"):
        sol = solve_scalar(scheme, xlo, xhi, N, u0_single, g_of_u, EPS_1, T,
                           max_gp, max_fp, sample_times=times)
        out[scheme] = sol
        print(f"    figure8: {scheme} done ({sol['nsteps']} steps)", flush=True)

    x = out["TDCNCS"]["x"]
    save_data("figure8_soliton_illustrative",
              x=x,
              u_TDCNCS=out["TDCNCS"]["samples"],
              u_TDCCS=out["TDCCS"]["samples"],
              exact={t: u_exact_single(x, t) for t in times},
              meta={"N": N, "eps": EPS_1, "c": C_1, "k": K_1,
                    "times": times, "xlo": xlo, "xhi": xhi})


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


def figure9_data(N=100):
    """Compute and save the double-soliton collision (paper Figure 9).

    Snapshots at t = 0, 1, 2 feed the three line panels; a dense set of
    times feeds the space-time surface in the fourth column. There is no
    exact solution for this case, so none is stored.
    """
    xlo, xhi = 0.0, 2.0
    T = 4.0
    snap_times = [0.0, 1.0, 2.0]
    # During the collision the peak exceeds either soliton alone, so the
    # convective bound uses the SUM of the two amplitudes.
    max_gp = 3.0 * (C1_2 + C2_2)
    max_fp = EPS_2

    surf_times = list(np.linspace(0.0, T, 81))
    all_times = sorted(set(snap_times + surf_times))

    out = {}
    for scheme in ("TDCNCS", "TDCCS"):
        sol = solve_scalar(scheme, xlo, xhi, N, u0_double, g_of_u, EPS_2, T,
                           max_gp, max_fp, sample_times=all_times)
        out[scheme] = sol
        print(f"    figure9: {scheme} done ({sol['nsteps']} steps)", flush=True)

    save_data("figure9_double_soliton",
              x=out["TDCNCS"]["x"],
              u_TDCNCS=out["TDCNCS"]["samples"],
              u_TDCCS=out["TDCCS"]["samples"],
              surf_times=np.array(surf_times),
              meta={"N": N, "eps": EPS_2, "T": T, "snap_times": snap_times,
                    "xlo": xlo, "xhi": xhi})


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


def figure10_data(N=150):
    """Compute and save the triple-splitting case (paper Figure 10).

    Four runs in total: each scheme, with and without the twelfth-order
    filter. The filter interval differs by scheme (every 20 steps for
    TDCNCS, every 50 for TDCCS), exactly as the paper specifies.
    """
    xlo, xhi = 0.0, 3.0
    T = 4.0
    snap_times = [0.0, 1.0, 2.0]
    surf_times = list(np.linspace(0.0, T, 81))
    all_times = sorted(set(snap_times + surf_times))
    # The hump splits into solitons TALLER than itself, so the convective
    # bound must allow for that growth rather than use the initial peak.
    max_gp = 1.0
    max_fp = EPS_3

    fields = {}
    for scheme in ("TDCNCS", "TDCCS"):
        for filtered in (False, True):
            fe = FILTER_EVERY[scheme] if filtered else None
            sol = solve_scalar(scheme, xlo, xhi, N, u0_triple, g_of_u, EPS_3,
                               T, max_gp, max_fp, filter_every=fe,
                               alphaF=ALPHA_F, sample_times=all_times)
            tag = "f12" if filtered else "raw"
            fields[f"u_{scheme}_{tag}"] = sol["samples"]
            x = sol["x"]
            print(f"    figure10: {scheme} ({tag}) done "
                  f"({sol['nsteps']} steps)", flush=True)

    save_data("figure10_filter_effect",
              x=x, surf_times=np.array(surf_times),
              meta={"N": N, "eps": EPS_3, "T": T, "alphaF": ALPHA_F,
                    "snap_times": snap_times, "filter_every": FILTER_EVERY,
                    "xlo": xlo, "xhi": xhi},
              **fields)


def main():
    print("Example 7.3: nonlinear KdV with a small dispersion coefficient\n")
    print("Figure 8 data (single soliton) ...", flush=True)
    figure8_data()
    print("\nFigure 9 data (double soliton collision) ...", flush=True)
    figure9_data()
    print("\nFigure 10 data (triple splitting, filter comparison) ...",
          flush=True)
    figure10_data()
    print("\nExample 7.3 data written. Run make_figures.py to plot.")


if __name__ == "__main__":
    main()
