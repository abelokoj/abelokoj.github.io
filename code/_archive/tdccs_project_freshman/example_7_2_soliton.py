"""
EXAMPLE 7.2 -- The classical KdV SOLITON
=========================================
Reproduces, from Salian, Samala & Ghosh (2026):
    Figure 7    solutions and pointwise errors, N = 80, t = 0, 0.25, 0.5
    Table 10    errors and convergence rates at t = 0.5, N = 20 ... 160

------------------------------------------------------------------------
THE PROBLEM, Eq. (7.4)
------------------------------------------------------------------------
        u_t - 3(u^2)_x + u_xxx = 0        on x in [-10, 12], periodic
        u(x, 0) = -2 sech^2(x)

with exact solution, for t in [0, 0.5],

        u(x, t) = -2 sech^2(x - 4t)

------------------------------------------------------------------------
WHAT IS A SOLITON, AND WHY IS IT A GOOD TEST?
------------------------------------------------------------------------
Look at the exact solution: it is the initial shape with x replaced by
x - 4t. That is the same bump, moved to the right at constant speed 4,
with its height and width completely unchanged. It never spreads and
never steepens.

That is remarkable, because BOTH of those things should happen. The
nonlinear term -3(u^2)_x makes taller parts travel faster, which should
steepen the front into a shock. The dispersive term u_xxx makes short
wavelengths travel at different speeds, which should smear the bump into
ripples. In a soliton these two effects cancel each other exactly, and
the wave survives intact. This delicate balance is why solitons are the
standard benchmark: a scheme that mishandles either term will visibly
distort the shape, and the damage accumulates as the wave travels.

Because the exact solution is known, we can measure error precisely
rather than merely comparing two numerical answers to each other.

------------------------------------------------------------------------
PUTTING IT IN THE GENERAL FORM
------------------------------------------------------------------------
The shared solver expects u_t + g(u)_x + D*u_xxx = 0. Matching term by
term against Eq. (7.4):

        g(u) = -3u^2      so  g'(u) = -6u
        D    = +1         so  f(u)  = u  and  f'(u) = 1

For the time-step formula we need bounds over the whole run. The soliton
holds its amplitude at exactly 2, so

        max|g'(u)| = 6 * 2 = 12,     max|f'(u)| = 1

Both are exact here, not estimates, because the amplitude is conserved.

------------------------------------------------------------------------
RUNTIME
------------------------------------------------------------------------
The N = 160 row of Table 10 takes roughly 2.4e4 time steps, since the
stable dt shrinks like dx^3. The full table takes a while. To shorten it,
call main() with a shorter list, for example main(table_Ns=(20, 40, 60)).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")           # write figures to files, no GUI window
import matplotlib.pyplot as plt

from pub_style import apply_style, savefig_all, FIGDIR, subcaption
from nonlinear_kdv import solve_scalar, error_norms, print_convergence_table

apply_style()
OUT = FIGDIR


# ----------------------------------------------------------------------
# PROBLEM DEFINITION, Eq. (7.4)
# ----------------------------------------------------------------------
XLO, XHI = -10.0, 12.0     # spatial domain (periodic)
T_FINAL = 0.5              # the paper's final time for this example


def u0(x):
    """Initial condition: a single negative bump of height 2 at x = 0.

    numpy has no sech, but sech = 1/cosh, so sech^2(x) = 1/cosh(x)^2.
    """
    return -2.0 / np.cosh(x)**2


def u_exact(x, t):
    """The exact solution: the same bump, shifted right by 4t."""
    return -2.0 / np.cosh(x - 4.0 * t)**2


def g_of_u(u):
    """The flux function. From Eq. (7.4), g(u) = -3u^2."""
    return -3.0 * u**2


D_DISP = 1.0        # coefficient of u_xxx in Eq. (7.4)
MAX_GPRIME = 12.0   # bound on |g'(u)| = |-6u| = 6*2, exact for this soliton
MAX_FPRIME = 1.0    # bound on |f'(u)|, with f(u) = u


def solve(scheme, N, T, sample_times=None):
    """Thin wrapper so the rest of the file need not repeat the constants."""
    return solve_scalar(scheme, XLO, XHI, N, u0, g_of_u, D_DISP, T,
                        MAX_GPRIME, MAX_FPRIME, sample_times=sample_times)


# ======================================================================
# TABLE 10
# ======================================================================
def table10(Ns=(20, 40, 60, 80, 100, 120, 140, 160)):
    """Solve on a sequence of grids and tabulate errors and rates.

    Two things to look for in the output:

      1. The RATES should sit near 8, confirming both schemes achieve
         their advertised 8th-order accuracy on this problem.
      2. At every grid size the TDCCS errors should be smaller than the
         TDCNCS ones, by roughly an order of magnitude. That gap is the
         paper's central claim.
    """
    rows = []
    for scheme in ("TDCNCS", "TDCCS"):
        for N in Ns:
            sol = solve(scheme, N, T_FINAL)          # run to t = 0.5
            ex = u_exact(sol["x"], T_FINAL)          # true answer there
            Linf, L1, L2 = error_norms(sol["u"], ex)
            rows.append((scheme, N, Linf, L1, L2))
            # Progress line, since the finer grids take a noticeable while.
            print(f"    {scheme:7s} N={N:4d}  Linf={Linf:.4e}  "
                  f"({sol['nsteps']} steps)", flush=True)

    print_convergence_table(
        rows, "TABLE 10 | Errors and spatial orders of convergence for "
              "Example 7.2 at t = 0.5")
    return rows


# ======================================================================
# FIGURE 7
# ======================================================================
def figure7(N=80):
    """Draw the solution and its pointwise error at three times.

    Layout, following the paper:
        (a), (b)   numerical solution (circles) on the exact one (line)
        (c), (d)   the pointwise error at the same three times

    Each run is done ONCE, asking the solver to record snapshots as it
    passes t = 0, 0.25 and 0.5, rather than restarting from scratch for
    each time. Those snapshots are taken at exactly the requested times;
    see the long note in nonlinear_kdv.integrate for why that detail
    matters more than it might appear to.
    """
    times = [0.0, 0.25, 0.5]
    colors = ["k", "b", "r"]        # black, blue, red, as in the paper

    results = {}
    for scheme in ("TDCNCS", "TDCCS"):
        sol = solve(scheme, N, T_FINAL, sample_times=times)
        results[scheme] = (sol["x"], sol["samples"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, scheme in enumerate(("TDCNCS", "TDCCS")):
        x, samples = results[scheme]
        for t, cc in zip(times, colors):
            u = samples[t]              # numerical solution at this time
            ex = u_exact(x, t)          # exact solution at the same time

            # Top row: exact drawn as a line, numerical as open circles.
            # If the scheme is working, the circles sit on the line.
            axes[0, col].plot(x, ex, "-", color=cc, lw=1.2)
            axes[0, col].plot(x, u, "o", ms=3, mfc="none", color=cc)

            # Bottom row: the gap between them, which is where the two
            # schemes actually differ.
            axes[1, col].plot(x, np.abs(ex - u), color=cc, lw=1.0)

        subcaption(axes[0, col],
                   f"({'a' if col == 0 else 'b'}) {scheme} - Numerical solution")
        axes[0, col].set_xlabel("$x$")
        axes[0, col].set_ylabel("$u(x,t)$")
        axes[0, col].set_xlim(XLO, XHI)

        subcaption(axes[1, col],
                   f"({'c' if col == 0 else 'd'}) {scheme} - Pointwise error")
        axes[1, col].set_xlabel("$x$")
        axes[1, col].set_ylabel("Error")
        axes[1, col].set_xlim(XLO, XHI)

    fig.tight_layout()
    savefig_all(fig, "figure7_soliton_example72", outdir=OUT)
    plt.close(fig)
    print("Figure 7 written.")
    # Note the error axes are scaled independently per panel: TDCNCS peaks
    # near 3.5e-5 while TDCCS peaks near 2.4e-6, so read the numbers on
    # the axis rather than comparing the curve heights by eye.


def main(table_Ns=(20, 40, 60, 80, 100, 120, 140, 160)):
    print("Example 7.2: the classical KdV soliton\n")
    print("Computing Table 10 ...", flush=True)
    table10(table_Ns)
    print("\nComputing Figure 7 ...", flush=True)
    figure7()


if __name__ == "__main__":
    main()
