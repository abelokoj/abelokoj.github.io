"""
EXAMPLE 7.1 -- The LINEAR one-dimensional KdV equation
=======================================================
Reproduces, from Salian, Samala & Ghosh (2026):
    Figure 5   solutions and pointwise errors, c = 1, N = 40
    Figure 6   solutions and pointwise errors, c = 8, N = 40
    Table 8    errors and convergence rates, c = 1
    Table 9    errors and convergence rates, c = 8

------------------------------------------------------------------------
THE PROBLEM, Eq. (7.3)
------------------------------------------------------------------------
        u_t + c^(-2) u_xxx = 0        on x in [0, 2*pi], periodic
        u(x, 0) = sin(c*x)

with the exact solution known in closed form:

        u(x, t) = sin( c*(x + t) )

This is simply the initial sine wave sliding to the LEFT at speed 1,
holding its shape exactly. Because we know the true answer at every
instant, we can measure the error precisely, which is what makes this
the right first test.

------------------------------------------------------------------------
WHY START HERE?
------------------------------------------------------------------------
Notice what is MISSING from the equation: there is no g(u)_x term. The
only spatial derivative is the third one. That matters a great deal for
a reproduction study, because it means this example depends ONLY on the
scheme this paper itself defines and tabulates, Eqs. (2.4) and (3.1)-(3.2).
No operator borrowed from another paper is involved, so there is nothing
ambiguous to guess at. If our numbers match the paper here, the core
third-derivative machinery is right.

The parameter c controls the wavelength. c = 1 gives one smooth hump per
domain, which any decent scheme handles easily. c = 8 packs eight
oscillations into the same 40 grid points, roughly 5 points per
wavelength, which is where a scheme's ability to resolve SHORT waves
starts to reflect. That is exactly the property the paper's Figures 2 and
3 measure, and Figure 6 shows it affecting a real computation.

------------------------------------------------------------------------
RUNTIME WARNING
------------------------------------------------------------------------
The stable time step scales like dx^3 (see nonlinear_kdv.cfl_dt for why),
so refining the grid is expensive: N = 160 needs roughly 1.6 million
TVDRK3 steps. The convergence tables below therefore stop earlier than
what is shown in the paper. The code itself has no such limit; edit the 
grid lists at the bottom to push further if you have the time to spare.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")          # draw to files, never to a screen window
import matplotlib.pyplot as plt

from pub_style import apply_style, savefig_all, FIGDIR
from tdccs_lib import (TDCNCS, TDCCS,
                       third_derivative_TDCNCS, third_derivative_TDCCS,
                       tvdrk3_step)

apply_style()      # publication fonts, sizes and resolution
OUT = FIGDIR       # <project>/figs

L_DOMAIN = 2.0 * np.pi     # the domain is [0, 2*pi]
CFL = 0.01                 # safety factor for the time step, Section 7


# ======================================================================
# SOLVER 1: TDCNCS  (everything lives at the grid nodes)
# ======================================================================
def solve_linear_kdv_TDCNCS(N, c, T, cfl=CFL):
    """March u_t + c^-2 u_xxx = 0 to time T using the node-only scheme."""
    # ---- grid -------------------------------------------------------
    dx = L_DOMAIN / N              # periodic, so N points and no duplicate end
    x = np.arange(N) * dx
    u = np.sin(c * x)              # initial condition u(x,0) = sin(cx)

    # ---- time step, Eq. (7.1) ---------------------------------------
    # There is no convective term here, so max|g'(u)| = 0 and only the
    # dispersive part of the CFL formula survives: dt = cfl * dx^3.
    dt = cfl / (1.0 / dx**3)
    nsteps = int(np.ceil(T / dt))  # round UP so we never overshoot dt...
    dt = T / nsteps                # ...then shrink dt to land exactly on T

    # ---- the semi-discrete right-hand side, Eq. (2.2) ---------------
    def rhs(u):
        # u_t = -c^-2 * u_xxx. The minus sign moves the term across the
        # equals sign; the 1/c^2 is the equation's own coefficient.
        return -(1.0 / c**2) * third_derivative_TDCNCS(u, dx, TDCNCS["T8"])

    # ---- march ------------------------------------------------------
    for _ in range(nsteps):
        u = tvdrk3_step(u, dt, rhs)
    return x, u


# ======================================================================
# SOLVER 2: TDCCS  (values at the nodes AND at the cell centers)
# ======================================================================
def solve_linear_kdv_TDCCS(N, c, T, cfl=CFL):
    """Same problem, solved with the paper's new central compact scheme.

    The difference from TDCNCS is that we now carry TWO arrays of
    unknowns: u at the nodes and u at the cell centers. Both are marched
    forward in time. Neither is obtained by interpolating the other, and
    avoiding that interpolation is the whole idea of the new scheme.

    Here the two arrays are glued into one long vector of length 2N so
    that the generic Runge-Kutta stepper can advance them together.
    """
    # ---- grid: nodes and centers ------------------------------------
    dx = L_DOMAIN / N
    x = np.arange(N) * dx          # nodes      x_j
    xh = x + dx / 2                # centers    x_{j+1/2}
    un = np.sin(c * x)             # initial condition sampled at the nodes
    uh = np.sin(c * xh)            # ...and at the centers

    # ---- time step (identical reasoning to TDCNCS above) ------------
    dt = cfl / (1.0 / dx**3)
    nsteps = int(np.ceil(T / dt))
    dt = T / nsteps

    # ---- right-hand side --------------------------------------------
    def rhs(state):
        # Split the long vector back into its node and center halves.
        un, uh = state[:N], state[N:]
        # This operator is COUPLED: it needs both halves to produce
        # either derivative, and it returns both at once.
        f3n, f3h = third_derivative_TDCCS(un, uh, dx, TDCCS["T8"])
        # Apply the PDE to each half, then re-glue into one vector.
        return np.concatenate([-(1.0 / c**2) * f3n,
                               -(1.0 / c**2) * f3h])

    state = np.concatenate([un, uh])
    for _ in range(nsteps):
        state = tvdrk3_step(state, dt, rhs)

    # Return both halves separately again for convenience.
    return x, state[:N], xh, state[N:]


# ======================================================================
# ERROR MEASUREMENT
# ======================================================================
def errors(u_num, u_exact):
    """Three summaries of the error; see nonlinear_kdv.error_norms for
    what each one means and why the paper divides by (N+1)."""
    e = u_exact - u_num
    N = len(e)
    Linf = np.max(np.abs(e))                       # worst single point
    L1 = np.sum(np.abs(e)) / (N + 1)               # average magnitude
    L2 = np.sqrt(np.sum(np.abs(e)**2) / (N + 1))   # root mean square
    return Linf, L1, L2


def convergence_table(c, T, Ns):
    """Print one block of Table 8 (c=1) or Table 9 (c=8).

    We solve the same problem on a sequence of finer and finer grids and
    watch how fast the error falls. Because both schemes are formally
    8th-order accurate, doubling N should cut the error by about
    2^8 = 256, which shows up as a printed "rate" close to 8.
    """
    print(f"\n--- c = {c} ---")
    print(f"{'Scheme':10s}{'N':>6s}{'Linf':>14s}{'rate':>8s}"
          f"{'L1':>14s}{'rate':>8s}{'L2':>14s}{'rate':>8s}")

    for scheme in ("TDCNCS", "TDCCS"):
        prev = None                       # previous grid's errors
        for N in Ns:
            if scheme == "TDCNCS":
                x, u = solve_linear_kdv_TDCNCS(N, c, T)
            else:
                x, u, _, _ = solve_linear_kdv_TDCCS(N, c, T)

            exact = np.sin(c * (x + T))   # the known solution at time T
            Linf, L1, L2 = errors(u, exact)

            if prev is None:
                # Nothing coarser to compare against on the first grid.
                rI = rL1 = rL2 = float("nan")
            else:
                # These grids double, so log base 2 gives the order directly.
                rI = np.log2(prev[0] / Linf)
                rL1 = np.log2(prev[1] / L1)
                rL2 = np.log2(prev[2] / L2)

            print(f"{scheme:10s}{N:6d}{Linf:14.4e}{rI:8.4f}"
                  f"{L1:14.4e}{rL1:8.4f}{L2:14.4e}{rL2:8.4f}", flush=True)
            prev = (Linf, L1, L2)


# ======================================================================
# FIGURES 5 AND 6
# ======================================================================
def solution_figure(c, N, times, colors, basename, panel_labels):
    """Draw one of the paper's two-by-two comparison figures.

    Layout, matching the paper:
        top row     numerical solution (circles) over the exact solution
                    (solid line), at several times, one column per scheme
        bottom row  the pointwise error |exact - numerical| at those times

    Reading the top row alone, both schemes usually look perfect; it is
    the bottom row that separates them, and note that the two error axes
    are scaled independently, so compare the numbers and not the heights.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for col, scheme in enumerate(("TDCNCS", "TDCCS")):
        for t, cc in zip(times, colors):
            if t == 0:
                # At t = 0 the answer is just the initial condition; no
                # need to run the solver at all.
                x = np.arange(N) * (L_DOMAIN / N)
                u = np.sin(c * x)
            elif scheme == "TDCNCS":
                x, u = solve_linear_kdv_TDCNCS(N, c, t)
            else:
                x, u, _, _ = solve_linear_kdv_TDCCS(N, c, t)

            exact = np.sin(c * (x + t))

            # Top: exact as a line, numerical as open circles on top of it.
            axes[0, col].plot(x, exact, "-", color=cc, lw=1.2)
            axes[0, col].plot(x, u, "o", ms=3, mfc="none", color=cc)
            # Bottom: how far apart they actually are.
            axes[1, col].plot(x, np.abs(exact - u), color=cc, lw=1.0)

        axes[0, col].set_title(f"({panel_labels[col]}) {scheme} "
                               f"- Numerical solution")
        axes[0, col].set_xlabel("$x$")
        axes[0, col].set_ylabel("$u(x,t)$")
        axes[1, col].set_title(f"({panel_labels[col + 2]}) {scheme} "
                               f"- Pointwise error")
        axes[1, col].set_xlabel("$x$")
        axes[1, col].set_ylabel("Error")

    fig.tight_layout()
    savefig_all(fig, basename, outdir=OUT)   # writes .png, .pdf and .svg
    plt.close(fig)                            # free the memory


def main():
    # ---- Tables 8 and 9 ---------------------------------------------
    # c = 1: the paper reports N up to 40, which we match exactly.
    convergence_table(c=1, T=1.0, Ns=[10, 20, 30, 40])

    # c = 8: the paper goes to N = 160. Because dt ~ dx^3, that needs
    # about 1.6 million time steps, far more than is practical here, so
    # we stop at N = 80. The convergence RATE is already clear by then,
    # and the same code reproduces the full table given more time.
    # You are free to edit the line below so that Ns=[..., 160]
    convergence_table(c=8, T=1.0, Ns=[20, 40, 60, 80])

    # ---- Figure 5: low wavenumber, c = 1 ----------------------------
    # Errors here reach the 1e-12 level, essentially machine precision:
    # with only one hump across 40 points the grid is far finer than the
    # solution needs.
    solution_figure(c=1, N=40,
                    times=[0, 0.25, 0.5, 0.75, 1.0],
                    colors=["k", "b", "g", "m", "r"],
                    basename="figure5_example71_c1",
                    panel_labels="abcd")

    # ---- Figure 6: high wavenumber, c = 8 ---------------------------
    # Eight oscillations across the same 40 points, roughly 5 points per
    # wave. Errors jump to the 1e-3 to 1e-4 range, and the difference
    # between the two schemes becomes plainly visible: this is the
    # spectral-resolution advantage of Figure 3 showing up in practice.
    solution_figure(c=8, N=40,
                    times=[0, 0.3, 0.7, 1.0],
                    colors=["k", "b", "g", "r"],
                    basename="figure6_example71_c8",
                    panel_labels="abcd")

    print("\nFigures 5 and 6 written.")


if __name__ == "__main__":
    main()


