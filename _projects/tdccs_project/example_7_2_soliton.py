"""
EXAMPLE 7.2 -- The classical KdV SOLITON
=========================================
This script COMPUTES ONLY. It performs no plotting: every figure is drawn
by `make_figures.py` from the data files written here. Separating the two
matters because these solves are expensive (the time step scales like
dx^3), and redrawing a figure to change a colour or a label should never
require recomputing the solution.

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

from figdata import save_data
from nonlinear_kdv import (solve_scalar, error_norms,
                           print_convergence_table)



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

    # Persist the table so it can be reformatted without re-solving.
    save_data("table10",
              rows=np.array([r[1:] for r in rows], dtype=float),
              schemes=np.array([r[0] for r in rows]),
              meta={"T": T_FINAL, "Ns": list(Ns),
                    "columns": ["N", "Linf", "L1", "L2"]})
    return rows


# ======================================================================
# FIGURE 7
# ======================================================================
def figure7_data(N=80):
    """Compute and save everything the paper's Figure 7 needs.

    Both schemes are run once each, recording snapshots as they pass
    t = 0, 0.25 and 0.5, and the exact solution is evaluated on the same
    grid at those times. Nothing is plotted.
    """
    times = [0.0, 0.25, 0.5]
    out = {}
    for scheme in ("TDCNCS", "TDCCS"):
        sol = solve(scheme, N, T_FINAL, sample_times=times)
        out[scheme] = sol
        print(f"    figure7: {scheme} done ({sol['nsteps']} steps)", flush=True)

    x = out["TDCNCS"]["x"]
    save_data("figure7_soliton_example72",
              x=x,
              u_TDCNCS=out["TDCNCS"]["samples"],
              u_TDCCS=out["TDCCS"]["samples"],
              exact={t: u_exact(x, t) for t in times},
              meta={"N": N, "T": T_FINAL, "times": times,
                    "xlo": XLO, "xhi": XHI})


def main(table_Ns=(20, 40, 60, 80, 100, 120, 140, 160)):
    print("Example 7.2: the classical KdV soliton\n")
    print("Computing Table 10 ...", flush=True)
    table10(table_Ns)
    print("\nComputing Figure 7 data ...", flush=True)
    figure7_data()
    print("\nExample 7.2 data written. Run make_figures.py to plot.")


if __name__ == "__main__":
    main()
