"""
EXAMPLE 7.5 -- TWO-DIMENSIONAL linear dispersion
=================================================
This script COMPUTES ONLY. It performs no plotting: every figure is drawn
by `make_figures.py` from the data files written here. Separating the two
matters because these solves are expensive (the time step scales like
dx^3), and redrawing a figure to change a colour or a label should never
require recomputing the solution.

Reproduces, from Salian, Samala & Ghosh (2026):
    Figure 13   solution surface and pointwise error, N = 40, t = 1
    Table 11    errors and convergence rates at t = 1

A note on the paper's labelling: the caption of its Figure 13 and the
caption of its Table 11 both read "Example 7.6", but the text introducing
them, and the equation they cite, place them in Example 7.5. The text is
the reliable guide here, and that is what this script follows.

------------------------------------------------------------------------
THE PROBLEM, Eq. (7.11)
------------------------------------------------------------------------
        u_t + u_xxx + u_yyy = 0     on (x,y) in (0,2pi) x (0,2pi), periodic
        u(x, y, 0) = sin(x + y)

with exact solution

        u(x, y, t) = sin(x + y + 2t)

Again a travelling wave with a known closed form, so the error can be
measured exactly.

------------------------------------------------------------------------
WHY BOTHER WITH TWO DIMENSIONS?
------------------------------------------------------------------------
The point is to show the scheme extends to more than one space dimension
without any new ideas being required. The equation has two third-derivative
terms, one in x and one in y, and each is handled by applying the SAME 1D
operator along the corresponding direction of the grid. Nothing about the
scheme itself changes; only the bookkeeping does.

This "apply the 1D operator along each axis in turn" strategy is called a
dimension-by-dimension or tensor-product approach, and it is the standard
way to lift a 1D finite-difference scheme to 2D on a rectangular grid.

Like Example 7.1, this problem has NO convective term, so it exercises
only the third-derivative machinery this paper defines itself.

------------------------------------------------------------------------
A PRACTICAL NOTE ON SPEED
------------------------------------------------------------------------
Applying a 1D operator to every row and every column of an N-by-N grid
would mean 2N separate small solves per Runge-Kutta stage, and with the
dx^3 time-step restriction there are a great many stages. Doing that with
a Python loop would be painfully slow.

Instead we use `third_derivative_TDCNCS_2d`, which performs the whole
sweep as a single dense matrix multiplication per axis. The mathematics
is identical; only the arrangement of the arithmetic changes.
"""
import numpy as np

from figdata import save_data
from tdccs_lib import (TDCNCS, TDCCS, third_derivative_TDCNCS_2d,
                       third_derivative_TDCCS, tvdrk3_step)


L_DOMAIN = 2.0 * np.pi
CFL = 0.01


def d3_along(U, dx, axis):
    """Third derivative of the 2D array U along one axis.

    axis = 0 differentiates down the columns (the x direction);
    axis = 1 differentiates across the rows (the y direction).

    Because the grid is a rectangle and the scheme is the same in both
    directions, one function with an `axis` switch covers both terms.
    """
    return third_derivative_TDCNCS_2d(U, dx, TDCNCS["T8"], axis)


def solve_2d_linear_TDCNCS(N, T, cfl=CFL):
    """March u_t + u_xxx + u_yyy = 0 to time T on an N-by-N grid."""
    # ---- grid -------------------------------------------------------
    dx = L_DOMAIN / N
    x = np.arange(N) * dx
    # meshgrid with indexing='ij' makes X[i,j] = x[i] and Y[i,j] = x[j],
    # i.e. the first index runs over x and the second over y. That matches
    # the axis numbering used in d3_along above.
    X, Y = np.meshgrid(x, x, indexing="ij")
    U = np.sin(X + Y)               # initial condition on the whole grid

    # ---- time step --------------------------------------------------
    # No convective term, so only the dispersive limit applies: dt ~ dx^3.
    # Eq. (7.2), the two-dimensional time step. Both the x and the y
    # third-derivative terms contribute to the denominator, so the
    # admissible step is HALF what the one-dimensional formula gives on
    # the same grid. Omitting that factor leaves TDCCS unstable here,
    # because Eq. (6.3) already restricts it to dt/dx^3 <= 0.011, an
    # order of magnitude tighter than TDCNCS.
    dt = cfl / (1.0 / dx**3 + 1.0 / dx**3)
    nsteps = max(1, int(np.ceil(T / dt)))
    dt = T / nsteps                 # adjust to land exactly on T

    # ---- right-hand side --------------------------------------------
    def rhs(U):
        # u_t = -(u_xxx + u_yyy): one sweep along each axis, then add.
        return -(d3_along(U, dx, axis=0) + d3_along(U, dx, axis=1))

    for _ in range(nsteps):
        U = tvdrk3_step(U, dt, rhs)
    return X, Y, U


def errors(u_num, u_exact):
    """Error norms for a 2D array.

    Same three summaries as in the 1D examples, except the averages
    divide by the TOTAL number of grid points (N*N), since we are now
    averaging over a surface rather than a line.
    """
    e = u_exact - u_num
    Ntot = u_num.size               # N*N for a square grid
    Linf = np.max(np.abs(e))
    L1 = np.sum(np.abs(e)) / Ntot
    L2 = np.sqrt(np.sum(np.abs(e)**2) / Ntot)
    return Linf, L1, L2


def table11(Ns=(10, 15, 20, 25, 30), T=1.0):
    """Print Table 11: errors and convergence rates for BOTH schemes.

    The grids here are not successive doublings, so the rate uses the
    general formula

        rate = log(error_coarse / error_fine) / log(N_fine / N_coarse)

    which reduces to a log base 2 when N doubles.
    """
    print(f"{'Scheme':9s}{'N':>5s}{'Linf':>14s}{'rate':>8s}{'L1':>14s}"
          f"{'rate':>8s}{'L2':>14s}{'rate':>8s}")
    rows = []   # collected so the table can be persisted
    for scheme in ("TDCNCS", "TDCCS"):
        prev = None
        for N in Ns:
            if scheme == "TDCNCS":
                X, Y, U = solve_2d_linear_TDCNCS(N, T=T)
            else:
                X, Y, U = solve_2d_linear_TDCCS(N, T=T)
            exact = np.sin(X + Y + 2 * T)
            Linf, L1, L2 = errors(U, exact)

            if prev is None:
                rI = rL1 = rL2 = float("nan")
            else:
                pN, pI, p1, p2 = prev
                denom = np.log(N / pN)
                rI = np.log(pI / Linf) / denom
                rL1 = np.log(p1 / L1) / denom
                rL2 = np.log(p2 / L2) / denom

            print(f"{scheme:9s}{N:5d}{Linf:14.4e}{rI:8.4f}{L1:14.4e}{rL1:8.4f}"
                  f"{L2:14.4e}{rL2:8.4f}", flush=True)
            rows.append((scheme, N, Linf, L1, L2, rI, rL1, rL2))
            prev = (N, Linf, L1, L2)

    save_data("table11",
              rows=np.array([r[1:] for r in rows], dtype=float),
              schemes=np.array([r[0] for r in rows]),
              meta={"T": T, "Ns": list(Ns),
                    "columns": ["N", "Linf", "L1", "L2",
                                "rate_Linf", "rate_L1", "rate_L2"]})
    return rows


def solve_2d_linear_TDCCS(N, T=1.0, cfl=CFL):
    """The TDCCS counterpart of the solver above.

    Extending a node-and-center scheme to two dimensions needs more care
    than the third derivative itself does. In one dimension the state is
    a pair: values at the nodes and values at the cell centers. On a
    tensor-product grid each axis has its own node/center split, so the
    state becomes FOUR fields, one for each combination:

        W[0] = u at (x node,   y node  )
        W[1] = u at (x center, y node  )
        W[2] = u at (x node,   y center)
        W[3] = u at (x center, y center)

    The pairing matters. To differentiate in x, the coupled operator
    needs node and center values in x taken at the SAME y, so W[0] pairs
    with W[1] and W[2] pairs with W[3]. To differentiate in y, it needs
    node and center values in y at the same x, so W[0] pairs with W[2]
    and W[1] pairs with W[3]. Pairing W[0] with W[3], or pairing across
    different y lines, silently mismatches the grids: the run stays
    bounded but loses all accuracy, which is easy to mistake for a
    convergence problem rather than an indexing one.
    """
    dx = L_DOMAIN / N
    x = np.arange(N) * dx
    xh = x + dx / 2

    # Eq. (7.2), the two-dimensional time step; see the TDCNCS solver.
    dt = cfl / (1.0 / dx**3 + 1.0 / dx**3)
    nsteps = max(1, int(np.ceil(T / dt)))
    dt = T / nsteps
    c3 = TDCCS["T8"]

    def d3_pair_axis(A, B, axis):
        """Coupled third derivative of the (node, center) pair A, B."""
        outA = np.empty_like(A)
        outB = np.empty_like(B)
        if axis == 0:                      # differentiate down columns
            for j in range(A.shape[1]):
                outA[:, j], outB[:, j] = third_derivative_TDCCS(
                    A[:, j], B[:, j], dx, c3)
        else:                              # differentiate across rows
            for i in range(A.shape[0]):
                outA[i, :], outB[i, :] = third_derivative_TDCCS(
                    A[i, :], B[i, :], dx, c3)
        return outA, outB

    def rhs(W):
        nn, hn, nh, hh = W
        # x-derivatives: pair along x at fixed y.
        dx_nn, dx_hn = d3_pair_axis(nn, hn, axis=0)
        dx_nh, dx_hh = d3_pair_axis(nh, hh, axis=0)
        # y-derivatives: pair along y at fixed x.
        dy_nn, dy_nh = d3_pair_axis(nn, nh, axis=1)
        dy_hn, dy_hh = d3_pair_axis(hn, hh, axis=1)
        return np.stack([-(dx_nn + dy_nn), -(dx_hn + dy_hn),
                         -(dx_nh + dy_nh), -(dx_hh + dy_hh)])

    Xnn, Ynn = np.meshgrid(x, x, indexing="ij")
    Xhn, Yhn = np.meshgrid(xh, x, indexing="ij")
    Xnh, Ynh = np.meshgrid(x, xh, indexing="ij")
    Xhh, Yhh = np.meshgrid(xh, xh, indexing="ij")
    W = np.stack([np.sin(Xnn + Ynn), np.sin(Xhn + Yhn),
                  np.sin(Xnh + Ynh), np.sin(Xhh + Yhh)])

    for _ in range(nsteps):
        W = tvdrk3_step(W, dt, rhs)
    # Return the node-node field, so both schemes plot on the same grid.
    return Xnn, Ynn, W[0]


def figure13_data(N=40, T=1.0):
    """Compute and save the two-dimensional surfaces (paper Figure 13).

    Four fields are stored: the numerical solution from each scheme and
    the corresponding pointwise error, all on the same node grid, so the
    plotting script can draw the paper's two-by-two arrangement of
    surfaces without re-solving.
    """
    X, Y, U_n = solve_2d_linear_TDCNCS(N, T=T)
    print("    figure13: TDCNCS done", flush=True)
    _, _, U_c = solve_2d_linear_TDCCS(N, T=T)
    print("    figure13: TDCCS done", flush=True)
    exact = np.sin(X + Y + 2 * T)

    save_data("figure13_example75",
              X=X, Y=Y,
              u_TDCNCS=U_n, u_TDCCS=U_c,
              err_TDCNCS=np.abs(exact - U_n),
              err_TDCCS=np.abs(exact - U_c),
              meta={"N": N, "T": T, "L_domain": L_DOMAIN})


def main():
    print("Example 7.5: two-dimensional linear dispersion\n")
    print("TABLE 11 | Errors and spatial orders of convergence at t = 1")
    table11()
    figure13_data()
    print("\nExample 7.5 data written. Run make_figures.py to plot.")


if __name__ == "__main__":
    main()
