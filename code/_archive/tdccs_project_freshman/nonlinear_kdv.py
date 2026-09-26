"""
nonlinear_kdv.py
================
Shared machinery for the four NONLINEAR examples of Salian, Samala &
Ghosh (2026): Examples 7.2, 7.3, 7.4 and 7.6.

------------------------------------------------------------------------
WHAT PROBLEM ARE WE SOLVING?
------------------------------------------------------------------------
Every nonlinear example in the paper has the same shape, Eq. (2.1):

        u_t  +  g(u)_x  +  D * u_xxx  =  0

Reading that left to right:

  u_t       how fast u changes in time at a fixed point in space.
            This is what we want to march forward.

  g(u)_x    the CONVECTIVE (or "flux") term. It transports the wave
            along. Because g depends on u itself, this term is
            NONLINEAR: tall parts of the wave travel at a different
            speed from short parts, which makes the wave steepen, the
            same way an ocean wave steepens before it breaks.

  D*u_xxx   the DISPERSIVE term (third derivative in space). Dispersion
            makes waves of different wavelengths travel at different
            speeds, so a lump spreads out into a train of ripples.

The famous behaviour of the KdV equation comes from the balance between
those last two: steepening pushes the wave to break, dispersion pulls it
apart, and when the two exactly cancel you get a SOLITON, a lump that
travels without changing shape at all.

To turn this PDE into something a computer can advance, we discretize
space only, which gives Eq. (2.2), one ordinary differential equation
per grid point:

        d u_j / dt  =  - g'_j  -  f'''_j

Here g'_j is a numerical approximation of the first derivative and f'''_j
of the third derivative, both evaluated at grid point j. Choosing HOW to
approximate those two derivatives is the entire subject of the paper.

------------------------------------------------------------------------
THE TWO SCHEMES BEING COMPARED
------------------------------------------------------------------------
The paper's Section 7 compares two pairings:

  TDCNCS  = Lele (1992) 8th-order cell-NODE compact scheme  [ref 1]
            for the first derivative,
          + Eq. (2.4) 8th-order cell-node scheme for the third derivative.
            Everything lives at the grid nodes x_0, x_1, x_2, ...

  TDCCS   = Liu et al. (2013) 8th-order CENTRAL compact scheme [ref 23]
            for the first derivative,
          + Eqs. (3.1)-(3.2), the paper's new scheme, for the third.
            Values live at the nodes AND at the cell centers
            x_{1/2}, x_{3/2}, ..., and BOTH sets are marched forward in
            time as independent unknowns. Nothing is interpolated.

So a TDCCS "state" is twice as long as a TDCNCS state: it stores u at the
nodes and u at the centers. That extra memory is the price; the payoff is
better resolution of short waves (the paper's Figures 2 and 3).

"Compact" means the derivative at a point is defined by a small linear
SYSTEM coupling neighbouring derivative values, rather than by an
explicit formula. That system is solved once per derivative evaluation.
The library `tdccs_lib.py` handles that; here we just call it.

------------------------------------------------------------------------
MARCHING FORWARD IN TIME
------------------------------------------------------------------------
Once space is discretized we have du/dt = S(u), a big system of ODEs. We
advance it with TVDRK3, a three-stage Runge-Kutta method, Eq. (6.2).

The time step is NOT free. An explicit method is only stable if dt is
small enough, and for a third-derivative operator that limit is brutal,
Eq. (7.1):

        dt = CFL / ( max|g'(u)|/dx  +  max|f'(u)|/dx^3 )

Notice the dx^3 in the denominator. HALVING dx makes the allowed dt EIGHT
times smaller, so it costs 8x more steps to reach the same final time on
a grid twice as fine, on top of each step costing twice as much. That
single fact explains why the fine-grid runs in this project are slow.

All examples use PERIODIC boundary conditions: the domain wraps around,
so whatever leaves the right edge re-enters on the left.
"""
import numpy as np

# Operators supplied by the project library. Each is a self-contained
# numerical approximation of a derivative on a periodic grid.
from tdccs_lib import (
    TDCNCS, TDCCS,                    # coefficient tables (paper's Tables 1 and 4)
    third_derivative_TDCNCS,          # Eq. (2.4)      third derivative, nodes only
    third_derivative_TDCCS,           # Eqs. (3.1-3.2) third derivative, nodes+centers
    first_derivative_Lele_CNCS8,      # Lele  (1992)   first derivative, nodes only
    first_derivative_Liu_CCS8,        # Liu   (2013)   first derivative, nodes+centers
    apply_filter_F12,                 # Eq. (5.1)      12th-order low-pass filter
    tvdrk3_step,                      # Eq. (6.2)      one TVDRK3 time step
)

CFL = 0.01          # the paper's chosen safety factor, Section 7, Eq. (7.1)
SCHEME_KEY = "T8"   # "T8" = Tridiagonal, 8th order: the member used throughout


# ======================================================================
# 1. BUILDING THE GRID
# ======================================================================
def periodic_grid(xlo, xhi, N):
    """Build a periodic grid of N points on the interval [xlo, xhi].

    Because the domain is periodic, the point at xhi is the SAME point as
    the one at xlo. We therefore store only N distinct points and leave
    off the duplicate at the right end, which is why dx = L/N and not
    L/(N-1).

    Returns
    -------
    x  : the N cell NODES,   x[j]  = xlo + j*dx
    xh : the N cell CENTERS, xh[j] = x[j] + dx/2
         In the paper's notation xh[j] is the point written x_{j+1/2}.
         TDCNCS ignores this array; TDCCS evolves values on it.
    dx : the spacing between neighbouring nodes.
    """
    L = xhi - xlo            # total length of the domain
    dx = L / N               # spacing (note: L/N, not L/(N-1), see above)
    x = xlo + np.arange(N) * dx      # [xlo, xlo+dx, xlo+2dx, ...]
    xh = x + 0.5 * dx                # shift each node right by half a cell
    return x, xh, dx


def cfl_dt(dx, max_gprime, max_fprime, cfl=CFL):
    """Largest stable time step, Eq. (7.1).

    max_gprime : a bound on |g'(u)|, i.e. how fast the wave is transported.
    max_fprime : a bound on |f'(u)|, the strength of the dispersive term.

    The two effects each impose their own limit, and we must respect the
    stricter of the two, so their contributions are ADDED in the
    denominator: a larger denominator gives a smaller (safer) dt.

    IMPORTANT for anyone reusing this: the bounds must hold for the WHOLE
    run, not just at t = 0. Several of these solutions grow taller as they
    evolve, and if max_gprime was measured only at the start the time step
    can drift out of the stability region partway through and the run will
    blow up.
    """
    denom = max_gprime / dx + max_fprime / dx**3
    return cfl / denom


# ======================================================================
# 2. THE RIGHT-HAND SIDE, S(u), FOR EACH SCHEME
# ======================================================================
# These functions do NOT compute a number. Each one BUILDS AND RETURNS
# another function. That inner function is the S(u) that the Runge-Kutta
# stepper calls repeatedly.
#
# Why do it this way? S(u) needs to know dx, the flux g, and the
# dispersion coefficient D, but the time stepper only ever wants to hand
# it u. Wrapping those fixed quantities in an enclosing function lets the
# inner function "remember" them. In Python this pattern is a closure.
# ======================================================================

def rhs_TDCNCS_scalar(dx, g_of_u, D):
    """Build S(u) for the node-only (TDCNCS) discretization.

    Given u at the nodes, return du/dt at the nodes.
    """
    coeffs3 = TDCNCS[SCHEME_KEY]   # look up the a,b,c,alpha,beta of Table 1

    def rhs(u):
        # Step 1: the convective term. Form the flux g(u) pointwise, then
        # differentiate that whole array once in space.
        gx = first_derivative_Lele_CNCS8(g_of_u(u), dx)

        # Step 2: the dispersive term, the third derivative of u.
        uxxx = third_derivative_TDCNCS(u, dx, coeffs3)

        # Step 3: assemble. The PDE reads u_t + g(u)_x + D*u_xxx = 0, so
        # solving for u_t moves both terms to the right with a minus sign.
        return -(gx + D * uxxx)

    return rhs


def rhs_TDCCS_scalar(dx, g_of_u, D):
    """Build S(U) for the coupled node+center (TDCCS) discretization.

    The state U is a (2, N) array:
        U[0] = values at the nodes    (u_j)
        U[1] = values at the centers  (u_{j+1/2})
    Both rows are genuine unknowns marched forward in time. Neither is
    interpolated from the other, which is precisely the paper's point:
    interpolation is what introduces the "transfer errors" that the new
    scheme is designed to avoid.
    """
    coeffs3 = TDCCS[SCHEME_KEY]    # coefficients from the paper's Table 4

    def rhs(U):
        un, uh = U[0], U[1]        # unpack nodes and centers

        # Both derivative operators take BOTH arrays and return BOTH
        # derivatives, because the node equation needs center values and
        # the center equation needs node values. They are coupled.
        gn, gh = first_derivative_Liu_CCS8(g_of_u(un), g_of_u(uh), dx)
        d3n, d3h = third_derivative_TDCCS(un, uh, dx, coeffs3)

        # Apply the same PDE to each row and stack them back together.
        return np.stack([-(gn + D * d3n),
                         -(gh + D * d3h)])

    return rhs


# ======================================================================
# 3. MARCHING FORWARD IN TIME
# ======================================================================
def integrate(u0, dt_target, rhs, stops, filter_every=None, alphaF=0.4,
              t0=0.0):
    """Advance the solution with TVDRK3, recording snapshots along the way.

    A subtle but important design choice
    ------------------------------------
    The obvious way to take snapshots is to pick a single dt, step
    repeatedly, and grab the solution whenever the clock happens to pass
    a time we care about. That is WRONG here, and the error it causes is
    easy to mistake for a genuine numerical error.

    Suppose we want the solution at t = 0.25 but our steps land on
    0.2499 and 0.2501. We would compare a solution computed at 0.2499
    against the exact solution at 0.25. For a wave travelling at speed 4,
    that half-step of clock mismatch shifts the wave by 4 * 0.0001 in
    space, and the resulting "error" can be a hundred times larger than
    the real discretization error we are trying to measure.

    The fix used here: walk from one requested time to the next, and on
    each leg choose a whole number of equal steps that lands exactly on
    the target. Every step stays at or below the stable dt, and every
    snapshot is taken at exactly the time requested.

    Parameters
    ----------
    u0         : initial state (1D array, or 2D for the TDCCS pair).
    dt_target  : the stable step from Eq. (7.1). Never exceeded.
    rhs        : the S(u) function built above.
    stops      : sorted list of times at which to record the solution.
    filter_every : apply the F12 low-pass filter every this many steps
                   (None means no filtering, which is the default in most
                   of the paper's examples).

    Returns (final_state, {time: state}, total_number_of_steps).
    """
    u = np.array(u0, dtype=float, copy=True)   # copy, so the caller's array is untouched
    samples = {}
    t_prev = t0
    step_count = 0

    # If t=0 was requested, record the initial condition before stepping.
    if stops and abs(stops[0] - t0) < 1e-14:
        samples[stops[0]] = u.copy()

    for t_stop in stops:
        seg = t_stop - t_prev            # length of this leg in time
        if seg <= 1e-14:
            continue                     # already there; nothing to do

        # How many equal steps fit in this leg without exceeding dt_target?
        # ceil rounds UP, which makes each step slightly SMALLER than the
        # limit, so we stay safely inside the stability region.
        nseg = max(1, int(np.ceil(seg / dt_target)))
        dt = seg / nseg                  # exact division: lands on t_stop

        for _ in range(nseg):
            u = tvdrk3_step(u, dt, rhs)  # one TVDRK3 step, Eq. (6.2)
            step_count += 1

            # Optional low-pass filtering, Section 5. High-order schemes
            # add no damping of their own, so tiny high-frequency wiggles
            # can accumulate. The filter removes those without noticeably
            # touching the smooth part of the solution.
            if filter_every and step_count % filter_every == 0:
                if u.ndim == 1:
                    u = apply_filter_F12(u, alphaF)
                else:
                    # TDCCS: filter the node row and the center row separately.
                    u = np.stack([apply_filter_F12(row, alphaF) for row in u])

        samples[t_stop] = u.copy()
        t_prev = t_stop

    return u, samples, step_count


def solve_scalar(scheme, xlo, xhi, N, u0_func, g_of_u, D, T,
                 max_gprime, max_fprime, filter_every=None, alphaF=0.4,
                 sample_times=None):
    """Solve u_t + g(u)_x + D*u_xxx = 0 from t=0 to t=T. The main entry point.

    Parameters
    ----------
    scheme     : "TDCNCS" or "TDCCS".
    xlo, xhi   : ends of the (periodic) spatial domain.
    N          : number of grid points.
    u0_func    : function of x giving the initial condition.
    g_of_u     : the flux function g. For example, g(u) = u**2/2.
    D          : coefficient multiplying u_xxx (often called epsilon).
    T          : final time.
    max_gprime,
    max_fprime : bounds used for the time step; see cfl_dt above.
    sample_times : times at which to record snapshots (T is always added).

    Returns a dictionary with:
       'x'       node positions
       'u'       final solution at the nodes
       'u_half'  final solution at the centers (TDCCS only; None otherwise)
       'dx','dt' grid spacing and target time step
       'nsteps'  how many TVDRK3 steps were actually taken
       'samples' {time: solution at the nodes}
    """
    # ---- build the grid and pick a stable time step --------------------
    x, xh, dx = periodic_grid(xlo, xhi, N)
    dt_target = cfl_dt(dx, max_gprime, max_fprime)

    # Make sure the final time is always among the recorded snapshots.
    stops = sorted(sample_times) if sample_times else []
    if not stops or abs(stops[-1] - T) > 1e-14:
        stops = stops + [T]

    # ---- run the chosen scheme ----------------------------------------
    if scheme == "TDCNCS":
        # State is a single array of node values.
        rhs = rhs_TDCNCS_scalar(dx, g_of_u, D)
        u_final, samples, nsteps = integrate(
            u0_func(x), dt_target, rhs, stops, filter_every, alphaF)
        return dict(x=x, u=u_final, u_half=None, dx=dx,
                    dt=dt_target, nsteps=nsteps, samples=samples)

    elif scheme == "TDCCS":
        # State stacks node values on top of center values. Both rows get
        # the initial condition, each sampled on its own set of points.
        rhs = rhs_TDCCS_scalar(dx, g_of_u, D)
        U0 = np.stack([u0_func(x), u0_func(xh)])
        U_final, samples, nsteps = integrate(
            U0, dt_target, rhs, stops, filter_every, alphaF)
        # For plotting we keep only the node row, so that both schemes can
        # be drawn on the same grid and compared directly.
        node_samples = {t: S[0].copy() for t, S in samples.items()}
        return dict(x=x, u=U_final[0], u_half=U_final[1], dx=dx,
                    dt=dt_target, nsteps=nsteps, samples=node_samples)

    raise ValueError(f"unknown scheme {scheme!r}")


# ======================================================================
# 4. MEASURING THE ERROR
# ======================================================================
def error_norms(u_num, u_exact):
    """Compare a numerical solution against the exact one, Section 7.

    Three different summaries of the same error array:

      L-infinity : the single WORST point. Answers "how bad does it ever
                   get anywhere?" Most sensitive to isolated spikes.
      L1         : the AVERAGE size of the error over the grid.
      L2         : the root-mean-square error. Sits between the other two;
                   penalizes large errors more than L1 but is less
                   dominated by a single outlier than L-infinity.

    Note the division by (N+1) rather than N. That is the paper's own
    convention, kept here so the printed numbers can be compared with its
    tables directly.
    """
    e = np.abs(u_num - u_exact)
    n = len(e)
    Linf = e.max()
    L1 = e.sum() / (n + 1)
    L2 = np.sqrt((e**2).sum() / (n + 1))
    return Linf, L1, L2


def print_convergence_table(rows, title, scheme_label_width=10):
    """Print an errors-and-rates table laid out like the paper's tables.

    What is a "rate"?
    -----------------
    A scheme is called p-th order accurate if halving dx cuts the error by
    a factor of 2^p. We can measure p from two runs on different grids:

            rate = log(error_coarse / error_fine) / log(N_fine / N_coarse)

    So an 8th-order scheme should print rates near 8.0. Seeing the
    measured rate match the theoretical order is the standard way to
    confirm an implementation is correct: a coding mistake almost always
    shows up as a rate that is too low.

    Rates are undefined for the coarsest grid (nothing to compare against),
    so that row prints "--".
    """
    print(f"\n{title}")
    print(f"{'Scheme':{scheme_label_width}s}{'N':>6s}{'Linf-error':>15s}{'Rate':>8s}"
          f"{'L1-error':>15s}{'Rate':>8s}{'L2-error':>15s}{'Rate':>8s}")
    prev = {}   # remembers the previous grid's results, per scheme
    for scheme, N, Linf, L1, L2 in rows:
        if scheme in prev:
            pN, pI, p1, p2 = prev[scheme]
            ratio = np.log(N / pN)          # denominator of the rate formula
            rI = np.log(pI / Linf) / ratio
            r1 = np.log(p1 / L1) / ratio
            r2 = np.log(p2 / L2) / ratio
            print(f"{scheme:{scheme_label_width}s}{N:6d}{Linf:15.4e}{rI:8.4f}"
                  f"{L1:15.4e}{r1:8.4f}{L2:15.4e}{r2:8.4f}")
        else:
            print(f"{scheme:{scheme_label_width}s}{N:6d}{Linf:15.4e}{'--':>8s}"
                  f"{L1:15.4e}{'--':>8s}{L2:15.4e}{'--':>8s}")
        prev[scheme] = (N, Linf, L1, L2)
