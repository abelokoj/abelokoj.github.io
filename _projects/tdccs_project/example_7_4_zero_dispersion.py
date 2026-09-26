"""
EXAMPLE 7.4 -- The ZERO-DISPERSION LIMIT
=========================================
This script COMPUTES ONLY. It performs no plotting: every figure is drawn
by `make_figures.py` from the data files written here. Separating the two
matters because these solves are expensive (the time step scales like
dx^3), and redrawing a figure to change a colour or a label should never
require recomputing the solution.

Reproduces, from Salian, Samala & Ghosh (2026):
    Figure 11   continuous initial condition, Eq. (7.9), at t = 0.5,
                for eps = 1e-4, 1e-5, 1e-6, 1e-7
    Figure 12   discontinuous top-hat initial condition, Eq. (7.10),
                N = 1000, eps = 1e-4, with and without the F12 filter

This example has no associated table.

------------------------------------------------------------------------
TWO LABELLING SLIPS IN THE PAPER, WORTH KNOWING BEFORE YOU READ IT
------------------------------------------------------------------------
1. The text introducing this example says it solves "the KdV equation
   (7.3)". But Eq. (7.3) is the LINEAR problem of Example 7.1, which has
   no convective term at all, and without a convective term there is no
   steepening and therefore no zero-dispersion limit to study. The
   equation actually intended is Eq. (7.5), the nonlinear problem of
   Example 7.3, and that is what this script solves.

2. The captions of Figures 11 and 12 both read "Example 7.3", but the
   initial conditions they cite, Eqs. (7.9) and (7.10), and the
   surrounding text place them squarely in Example 7.4.

------------------------------------------------------------------------
THE PROBLEM
------------------------------------------------------------------------
        u_t + (u^2 / 2)_x + eps * u_xxx = 0,       eps -> 0+

------------------------------------------------------------------------
WHAT IS THE ZERO-DISPERSION LIMIT, AND WHY IS IT INTERESTING?
------------------------------------------------------------------------
Set eps exactly to 0 and the equation becomes the inviscid Burgers
equation, u_t + (u^2/2)_x = 0. Its solutions steepen and form SHOCKS:
genuine discontinuities, in finite time.

Now keep eps small but not zero. Dispersion cannot stop the steepening,
but it refuses to allow an actual jump. What appears instead, right where
the shock would have been, is a packet of very rapid oscillations. As eps
shrinks these oscillations get faster and more numerous but never
disappear, and the solution does NOT converge pointwise to the shock. That
non-convergence is the zero-dispersion limit, studied theoretically by
Lax, Levermore and Venakides.

The crucial practical point for a numerical study: those oscillations are
PHYSICAL, part of the true solution. They are not numerical noise, and a
scheme must resolve rather than remove them. This is what makes the
example a demanding test, and it is why the mesh must be refined as eps
shrinks: eps = 1e-4 needs N = 100, but eps = 1e-7 needs N = 1600.

The paper notes that a low-pass filter does NOT affect these results,
precisely because the oscillations are genuine features and not the
grid-scale noise the filter targets.

------------------------------------------------------------------------
RUNTIME WARNING: THIS IS THE MOST EXPENSIVE SCRIPT IN THE PROJECT
------------------------------------------------------------------------
The stable time step falls off like dx^3, so the eps = 1e-7, N = 1600
case needs on the order of 2e5 TVDRK3 steps. Measured costs on a modest laptop core, after the sparse-solver fix:

    reference, eps=1e-4, N=1000   5,125,000 steps   about 1.3 hours
    reference, eps=1e-5, N=1000     625,000 steps   about 9 minutes
    panels (c),(d), N=800           125,600 steps   about 2 minutes each
    panels (e),(f), N=1600          220,481 steps   about 4 minutes each

so the two reference solves alone account for roughly 80 percent of the
total. For a quick exploratory run that finishes in a couple of minutes:

    main(cases=[(1e-4, 100), (1e-5, 200)], n_ref=200)

but read the comment on N_REF below before treating a coarse reference as
authoritative.
"""
import time

import numpy as np

from figdata import save_data
from nonlinear_kdv import solve_scalar



def g_of_u(u):
    """Flux for Eq. (7.5): g(u) = u^2/2, the Burgers flux."""
    return 0.5 * u**2


# ======================================================================
# FIGURE 11 : continuous initial condition, Eq. (7.9)
# ======================================================================
XLO_C, XHI_C = 0.0, 1.0
T_C = 0.5                    # the paper evaluates at t = 0.5

# (eps, N) pairs taken from the paper's text. Note how N must grow as eps
# shrinks: smaller dispersion means finer oscillations to resolve.
CASES_11 = [(1e-4, 100), (1e-5, 200), (1e-6, 800), (1e-7, 1600)]

# A reference solution on a deliberately over-refined mesh. With no exact
# solution available, this is the closest thing to "truth" we have, and
# the coarser runs are judged against it.
#
# COST WARNING. This single number dominates the runtime of the whole
# script, and not for the reason one might guess. The time step is
#
#     dt = CFL / ( max|g'| / dx  +  eps / dx^3 )
#
# and at N = 1000 the dispersive part of that denominator, eps/dx^3, is
# about forty times the convective part, so dt is throttled almost
# entirely by dispersion. Worse, eps/dx^3 is PROPORTIONAL to eps, which
# means the reference for the LARGEST eps is the most expensive one:
#
#     eps = 1e-4, N = 1000  ->  5,125,000 time steps   (about 1.3 hours)
#     eps = 1e-5, N = 1000  ->    625,000 time steps   (about 9 minutes)
#
# Reducing N_REF cuts this sharply, but do so knowingly: in the
# zero-dispersion regime the solution oscillates rapidly and convergence
# with mesh refinement is genuinely slow, so a coarse "reference" is not
# a reference at all. Measured differences between successive meshes at
# eps = 1e-4 were 4.6e-2 (N=100 to 150) and 2.4e-2 (N=150 to 250), still
# far from converged. The paper's choice of 1000 is not gratuitous.
N_REF = 1000


def u0_continuous(x):
    """Initial condition, Eq. (7.9): a smooth sine riding on a constant.

    The constant 2 shifts everything upward so u stays positive, which
    keeps the wave travelling in one direction. The sine part steepens on
    its downslope, and that is where the oscillations eventually form.
    """
    return 2.0 + 0.5 * np.sin(2.0 * np.pi * x)


def _solve_c(scheme, eps, N, T=T_C, label=""):
    """Run one (scheme, eps, N) case of Figure 11.

    max|u| = 2.5 from the initial condition, and because this equation
    conserves the maximum for smooth positive data, that bound holds
    throughout the run.

    Prints an estimate before starting. Several of these solves run for
    many minutes, and without a message first the script gives no output
    at all while it works, which is indistinguishable from a hang.
    """
    from nonlinear_kdv import periodic_grid, cfl_dt
    _, _, dx = periodic_grid(XLO_C, XHI_C, N)
    nsteps = int(np.ceil(T / cfl_dt(dx, 2.5, eps)))
    print(f"    [{label}] {scheme}, eps={eps:.0e}, N={N}: "
          f"~{nsteps:,} steps ...", flush=True)

    t0 = time.perf_counter()
    out = solve_scalar(scheme, XLO_C, XHI_C, N, u0_continuous, g_of_u, eps, T,
                       max_gprime=2.5, max_fprime=eps)
    print(f"    [{label}] done in {time.perf_counter() - t0:.1f} s", flush=True)
    return out


def figure11_data(cases=CASES_11, n_ref=None):
    """Compute and save the zero-dispersion sweep (paper Figure 11).

    For the two mildest cases both schemes are run alongside a refined
    TDCNCS reference, since all three fit legibly on one axis. For the
    two finest cases each scheme is stored separately, because the
    oscillations are far too dense to overlay.

    Every solution is stored on its own grid, since the grids differ
    between cases; the arrays are keyed by eps so the plotting script can
    pair each solution with the right x.
    """
    n_ref = N_REF if n_ref is None else n_ref
    fields = {}
    case_meta = []

    for k, (eps, N) in enumerate(cases[:2]):
        ref = _solve_c("TDCNCS", eps, n_ref, label="reference")
        s_n = _solve_c("TDCNCS", eps, N, label=f"panel {k}")
        s_c = _solve_c("TDCCS", eps, N, label=f"panel {k}")
        tag = f"e{abs(int(np.log10(eps)))}"
        fields[f"x_{tag}"] = s_n["x"]
        fields[f"x_ref_{tag}"] = ref["x"]
        fields[f"u_TDCNCS_{tag}"] = s_n["u"]
        fields[f"u_TDCCS_{tag}"] = s_c["u"]
        fields[f"u_ref_{tag}"] = ref["u"]
        case_meta.append({"eps": eps, "N": N, "tag": tag, "with_ref": True})

    for eps, N in cases[2:]:
        s_n = _solve_c("TDCNCS", eps, N, label="fine")
        s_c = _solve_c("TDCCS", eps, N, label="fine")
        tag = f"e{abs(int(np.log10(eps)))}"
        fields[f"x_{tag}"] = s_n["x"]
        fields[f"u_TDCNCS_{tag}"] = s_n["u"]
        fields[f"u_TDCCS_{tag}"] = s_c["u"]
        case_meta.append({"eps": eps, "N": N, "tag": tag, "with_ref": False})

    save_data("figure11_zero_dispersion_illustrative",
              meta={"cases": case_meta, "n_ref": n_ref, "T": T_C,
                    "xlo": XLO_C, "xhi": XHI_C},
              **fields)


# ======================================================================
# FIGURE 12 : discontinuous top-hat initial condition, Eq. (7.10)
# ======================================================================
XLO_T, XHI_T = 0.0, 5.0
EPS_T = 1.0e-4
N_T = 1000
TIMES_T = [0.01, 0.05]                        # note how EARLY these are
ALPHA_F = 0.4
FILTER_EVERY = {"TDCNCS": 20, "TDCCS": 50}


def u0_tophat(x):
    """Initial condition, Eq. (7.10): a rectangular pulse, 1 inside, 0 out.

    This is DISCONTINUOUS, which is a much harsher test than the smooth
    sine of Figure 11. A jump contains every wavelength at once, including
    ones far shorter than the grid can represent, so the scheme is
    immediately confronted with the very short waves it handles worst.

    Each of the two edges promptly breaks up into a train of dispersive
    waves, which is why the interesting times here are 0.01 and 0.05
    rather than the order-1 times used elsewhere.
    """
    return np.where((x > 0.25) & (x < 4.0), 1.0, 0.0)


def figure12_data(N=N_T):
    """Compute and save the top-hat breakup (paper Figure 12).

    Four runs: each scheme, with and without the twelfth-order filter,
    each recording snapshots at t = 0.01 and t = 0.05.
    """
    fields = {}
    for scheme in ("TDCNCS", "TDCCS"):
        for filtered in (False, True):
            fe = FILTER_EVERY[scheme] if filtered else None
            sol = solve_scalar(scheme, XLO_T, XHI_T, N, u0_tophat, g_of_u,
                               EPS_T, max(TIMES_T),
                               # dispersive overshoot pushes the solution
                               # above 1, so bound the speed at 1.5
                               max_gprime=1.5, max_fprime=EPS_T,
                               filter_every=fe, alphaF=ALPHA_F,
                               sample_times=TIMES_T)
            tag = "f12" if filtered else "raw"
            fields[f"u_{scheme}_{tag}"] = sol["samples"]
            x = sol["x"]
            print(f"    figure12: {scheme} ({tag}) done "
                  f"({sol['nsteps']} steps)", flush=True)

    save_data("figure12_tophat_illustrative",
              x=x,
              meta={"N": N, "eps": EPS_T, "times": TIMES_T,
                    "alphaF": ALPHA_F, "filter_every": FILTER_EVERY,
                    "xlo": XLO_T, "xhi": XHI_T},
              **fields)


def main(cases=CASES_11, n_ref=None):
    print("Example 7.4: the zero-dispersion limit\n")
    print("Figure 11 data (continuous initial condition) ...", flush=True)
    figure11_data(cases, n_ref=n_ref)
    print("\nFigure 12 data (top-hat breakup) ...", flush=True)
    figure12_data()
    print("\nExample 7.4 data written. Run make_figures.py to plot.")


if __name__ == "__main__":
    main()