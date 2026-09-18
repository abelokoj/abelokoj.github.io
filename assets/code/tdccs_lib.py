"""
tdccs_lib.py
============
Core numerical library implementing the schemes of

  Salian, Samala & Ghosh, "Central Compact Finite-Difference Scheme With
  High Spectral Resolution for KdV Equation", Numerical Methods for PDEs,
  2026;42:e70060.

Contents
--------
  * Coefficient tables (Tables 1, 2, 3, 4, 7) as plain dict-of-tuples.
  * Modified-wavenumber closures, Eqs. (2.6), (2.7), (4.1).
  * Periodic circulant solvers that build and invert the pentadiagonal /
    tridiagonal operators of Eqs. (2.4), (2.5), (3.1)-(3.2) on a uniform
    periodic grid (this is what actually gets used point-wise in Section 7).
  * The 12th-order low-pass filter, Eq. (5.1)-(5.2), Table 7.
  * TVDRK3 time integrator, Eq. (6.2).

Notation matches the paper as closely as possible; every public function
docstring cites the corresponding equation number.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

# ---------------------------------------------------------------------
# Table 1 : TDCNCS coefficients  (a, b, c, alpha, beta, order)
# ---------------------------------------------------------------------
TDCNCS = {
    "E2":  (1, 0, 0, 0, 0, 2),
    "E4":  (2, -1, 0, 0, 0, 4),
    "E6":  (169/60, -12/5, 7/12, 0, 0, 6),
    "T4":  (2, 0, 0, 1/2, 0, 4),
    "T6":  (2, -1/8, 0, 7/16, 0, 6),
    "T8":  (2367/1180, -167/1180, 1/236, 205/472, 0, 8),
    "P6":  (40/21, 0, 0, 4/9, 1/126, 6),
    "P8":  (160/83, -5/166, 0, 147/332, -1/166, 8),   # NOTE: fails its own
                                                        # order-2 condition
                                                        # when checked exactly
                                                        # -- flagged in blog post,
                                                        # likely an OCR/typo in
                                                        # the published Table 1.
    "P10": (18221/5478, -1846/913, 5/66, 799/2739, -557/5478, 10),
}

# Table 2 : TDCCCS coefficients
TDCCCS = {
    "E2":  (1, 0, 0, 0, 0, 2),
    "E4":  (13/8, -5/8, 0, 0, 0, 4),
    "E6":  (1299/640, -499/384, 259/960, 0, 0, 6),
    "T4":  (4/3, 0, 0, 1/6, 0, 4),
    "T6":  (205/166, 35/166, 0, 37/166, 0, 6),
    "T8":  (1058279/975200, 96627/195040, -24787/487600, 3229/12190, 0, 8),
    "P6":  (320/233, 0, 0, 134/699, -7/1398, 6),
    "P8":  (49720/79903, 91400/79903, 0, 28838/79903, 3541/159806, 8),
    "P10": (55463611/150617762, 677644345/451853286, 6301771/225926643,
            93443398/225926643, 15505921/451853286, 10),
}

# Table 4 : TDCCS coefficients (the paper's new scheme)
TDCCS = {
    "E4":  (13/8, -5/8, 0, 0, 0, 4),
    "E6":  (361/192, -129/128, 49/384, 0, 0, 6),
    "T4":  (8/7, 0, 0, 1/14, 0, 4),
    "T6":  (5, -5, 0, -1/2, 0, 6),
    "T8":  (58021/14120, -109007/28240, 1029/28240, -1261/3530, 0, 8),
    "P6":  (320/273, 0, 0, 74/819, -1/234, 6),
    "P8":  (19640/4621, -353000/87799, 0, -33746/87799, -147/175598, 8),
    "P10": (74390155/19635801, -45752035/13090534, 4684435/39271602,
            -5803114/19635801, 74747/39271602, 10),
}

# Table 3 : compact-interpolation (CI) transfer function coefficients
CI = {
    "E2":  (1, 0, 0, 0, 0, 2),
    "E4":  (9/8, -1/8, 0, 0, 0, 4),
    "E6":  (75/64, -25/128, 3/128, 0, 0, 6),
    "T4":  (4/3, 0, 0, 1/6, 0, 4),
    "T6":  (3/2, 1/10, 0, 3/10, 0, 6),
    "T8":  (25/16, 5/32, -1/224, 5/14, 0, 8),
    "P6":  (64/45, 0, 0, 2/9, -1/90, 6),
    "P8":  (8/5, 8/35, 0, 2/5, 1/70, 8),
    "P10": (5/3, 5/14, 1/126, 10/21, 5/126, 10),
}

# Table 7 : 12th-order filter coefficients (function of free parameter alphaF)
def filter_F12_coeffs(alphaF):
    """Eq. (5.1)-(5.2), Table 7. Returns a0..a6 for the 12th-order filter."""
    a0 = (793 + 462 * alphaF) / 1024
    a1 = (99 + 314 * alphaF) / 256
    a2 = 495 * (-1 + 2 * alphaF) / 2048
    a3 = 55 * (1 - 2 * alphaF) / 512
    a4 = 33 * (-1 + 2 * alphaF) / 1024
    a5 = 3 * (1 - 2 * alphaF) / 512
    a6 = (-1 + 2 * alphaF) / 2048
    return np.array([a0, a1, a2, a3, a4, a5, a6])


# =======================================================================
# Modified wavenumber closures : Eqs. (2.6), (2.7), (4.1)
# =======================================================================
def omega3_TDCNCS(omega, coeffs):
    a, b, c, alpha, beta, _ = coeffs
    num = (a * (2 * np.sin(omega) - np.sin(2 * omega))
           + (b / 4) * (3 * np.sin(omega) - np.sin(3 * omega))
           + (c / 10) * (4 * np.sin(omega) - np.sin(4 * omega)))
    den = 1 + 2 * alpha * np.cos(omega) + 2 * beta * np.cos(2 * omega)
    return num / den


def omega3_TDCCCS(omega, coeffs):
    a, b, c, alpha, beta, _ = coeffs
    num = (2 * a * (3 * np.sin(omega / 2) - np.sin(3 * omega / 2))
           + (2 * b / 5) * (5 * np.sin(omega / 2) - np.sin(5 * omega / 2))
           + (c / 7) * (7 * np.sin(omega / 2) - np.sin(7 * omega / 2)))
    den = 1 + 2 * alpha * np.cos(omega) + 2 * beta * np.cos(2 * omega)
    return num / den


def omega3_TDCCS(omega, coeffs):
    a, b, c, alpha, beta, _ = coeffs
    num = (2 * a * (8 * np.sin(omega / 2) - 4 * np.sin(omega))
           + (2 * b / 5) * (12 * np.sin(omega) - 8 * np.sin(3 * omega / 2))
           + (2 * c / 35) * (20 * np.sin(omega) - 8 * np.sin(5 * omega / 2)))
    den = 1 + 2 * alpha * np.cos(omega) + 2 * beta * np.cos(2 * omega)
    return num / den


def T_CI(omega, coeffs):
    """Eq. (2.9): transfer function of the compact interpolation, Table 3."""
    a, b, c, alpha, beta, _ = coeffs
    num = a * np.cos(omega / 2) + b * np.cos(3 * omega / 2) + c * np.cos(5 * omega / 2)
    den = 1 + 2 * alpha * np.cos(omega) + 2 * beta * np.cos(2 * omega)
    return num / den


# =======================================================================
# Periodic circulant solves for the compact 3rd-derivative operators
# =======================================================================
def _circulant_pentadiag_matrix(N, beta, alpha, center=1.0):
    """Sparse periodic pentadiagonal matrix with (beta,alpha,center,alpha,beta)."""
    offs = [-2, -1, 0, 1, 2]
    vals = [beta, alpha, center, alpha, beta]
    rows = []
    cols = []
    data = []
    idx = np.arange(N)
    for off, v in zip(offs, vals):
        if v == 0:
            continue
        rows.append(idx)
        cols.append((idx + off) % N)
        data.append(np.full(N, v))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.concatenate(data)
    A = sparse.coo_matrix((data, (rows, cols)), shape=(N, N))
    return A.tocsc()


_INV_CACHE = {}


class _DenseSolver:
    """Cheap drop-in for splu when N is small (<~400): precompute the dense
    inverse of the (fixed, circulant) pentadiagonal operator once and reuse
    it as a matrix-vector product on every subsequent RK stage -- this is
    dramatically faster in pure Python than re-factorizing / re-solving a
    sparse system at every one of the (many, since dt is CFL-limited by
    Delta x^3) time steps needed for the higher-N convergence runs."""
    def __init__(self, Ainv):
        self.Ainv = Ainv

    def solve(self, rhs):
        return self.Ainv @ rhs


_DENSE_INVERSE_MAX_N = 256

def _get_pentadiag_lu(N, alpha, beta):
    key = (N, alpha, beta)
    if key not in _INV_CACHE:
        A = _circulant_pentadiag_matrix(N, beta, alpha)
        if N <= _DENSE_INVERSE_MAX_N:
            _INV_CACHE[key] = _DenseSolver(np.linalg.inv(A.toarray()))
        else:
            _INV_CACHE[key] = splu(A.tocsc())
    return _INV_CACHE[key]


def third_derivative_TDCNCS(f, dx, coeffs):
    """
    Node-based 3rd derivative, Eq. (2.4), periodic boundary conditions.

    HOW A COMPACT SCHEME DIFFERS FROM AN ORDINARY ONE
    -------------------------------------------------
    An ordinary ("explicit") finite difference gives each derivative by a
    direct formula: the third derivative at a point equals some weighted
    sum of nearby f values. To reach high accuracy that way you need a
    WIDE stencil, reaching many points to the left and right.

    A compact scheme instead writes a relation mixing the UNKNOWN
    derivatives with the KNOWN function values. Writing DDD_j for the
    third derivative at grid point j:

        beta*DDD_{j-2} + alpha*DDD_{j-1} + DDD_j
                       + alpha*DDD_{j+1} + beta*DDD_{j+2}
            = (a weighted sum of f values near j)

    The left side couples the derivative at five neighbouring points, so
    the derivatives at ALL grid points are determined together, by solving
    one linear system. The payoff is high accuracy from a NARROW stencil
    and much better treatment of short wavelengths (the paper Figures 2
    and 3). The price is that linear solve.

    The grid here is uniform and periodic, so the matrix is identical at
    every point, a so-called circulant matrix, and it never changes during
    a run. We therefore factorize it once and cache the result, which
    makes every later solve cheap.

    f : array of values at the nodes x_0 .. x_{N-1}
    """
    a, b, c, alpha, beta, _ = coeffs
    N = len(f)
    rhs = (a * (np.roll(f, -2) - 2 * np.roll(f, -1) + 2 * np.roll(f, 1) - np.roll(f, 2)) / (2 * dx**3)
           + b * (np.roll(f, -3) - 3 * np.roll(f, -1) + 3 * np.roll(f, 1) - np.roll(f, 3)) / (8 * dx**3)
           + c * (np.roll(f, -4) - 4 * np.roll(f, -1) + 4 * np.roll(f, 1) - np.roll(f, 4)) / (20 * dx**3))
    lu = _get_pentadiag_lu(N, alpha, beta)
    return lu.solve(rhs)


def third_derivative_TDCNCS_2d(F, dx, coeffs, axis):
    """
    Vectorized version of third_derivative_TDCNCS applied along one axis of
    a 2D array F, using a single dense-matrix multiply instead of looping
    row-by-row / column-by-column in Python (needed for the 2D example,
    where many explicit RK3 substeps are required because dt ~ CFL*dx^3).
    """
    a, b, c, alpha, beta, _ = coeffs
    N = F.shape[axis]
    rhs = (a * (np.roll(F, -2, axis) - 2 * np.roll(F, -1, axis) + 2 * np.roll(F, 1, axis) - np.roll(F, 2, axis)) / (2 * dx**3)
           + b * (np.roll(F, -3, axis) - 3 * np.roll(F, -1, axis) + 3 * np.roll(F, 1, axis) - np.roll(F, 3, axis)) / (8 * dx**3)
           + c * (np.roll(F, -4, axis) - 4 * np.roll(F, -1, axis) + 4 * np.roll(F, 1, axis) - np.roll(F, 4, axis)) / (20 * dx**3))
    lu = _get_pentadiag_lu(N, alpha, beta)
    Ainv = lu.Ainv
    if axis == 0:
        return Ainv @ rhs
    else:
        return rhs @ Ainv.T


def third_derivative_TDCCS(f_node, f_half, dx, coeffs):
    """
    The paper's new scheme, Eqs. (3.1) [nodes] and (3.2) [centers], solved
    as ONE coupled circulant system of size 2N (nodes + half-grid points
    interleaved), since (3.1) needs f_half and (3.2) needs f_node — no
    interpolation is used, exactly as described in Section 3.

    f_node : values at x_0..x_{N-1}
    f_half : values at x_{1/2}, x_{3/2}, ..., x_{N-1/2}  (same length N)

    Returns (f3_node, f3_half): third derivatives at nodes and at centers.
    """
    a, b, c, alpha, beta, _ = coeffs
    N = len(f_node)

    # RHS of (3.1) at nodes, using f_half at j-1/2, j+1/2, j+3/2, j+5/2, etc.
    # index convention: f_half[k] stores f_{k+1/2}
    fh = f_half
    fn = f_node
    rhs_node = (a * (4 * np.roll(fn, -1) - 8 * fh + 8 * np.roll(fh, 1) - 4 * np.roll(fn, 1)) / dx**3
                + b * (8 * np.roll(fh, -1) - 12 * np.roll(fn, -1) + 12 * np.roll(fn, 1) - 8 * np.roll(fh, 2)) / (5 * dx**3)
                + c * (8 * np.roll(fh, -2) - 20 * np.roll(fn, -1) + 20 * np.roll(fn, 1) - 8 * np.roll(fh, 3)) / (35 * dx**3))
    # RHS of (3.2) at centers. Eq. (3.2) is Eq. (3.1) re-centered at the
    # half-grid point j-1/2; writing k = j-1 (so the LHS pentadiagonal
    # operator on f3_half is centered at index k, i.e. at f'''_{k+1/2})
    # gives, after relabelling every node/center offset accordingly:
    rhs_half = (a * (4 * np.roll(fh, -1) - 8 * np.roll(fn, -1) + 8 * fn - 4 * np.roll(fh, 1)) / dx**3
                + b * (8 * np.roll(fn, -2) - 12 * np.roll(fh, -1) + 12 * np.roll(fh, 1) - 8 * np.roll(fn, 1)) / (5 * dx**3)
                + c * (8 * np.roll(fn, -3) - 20 * np.roll(fh, -1) + 20 * np.roll(fh, 1) - 8 * np.roll(fn, 2)) / (35 * dx**3))

    lu = _get_pentadiag_lu(N, alpha, beta)
    f3_node = lu.solve(rhs_node)
    f3_half = lu.solve(rhs_half)
    return f3_node, f3_half


def apply_filter_F12(f, alphaF=0.4):
    """The 12th-order low-pass filter of Eq. (5.1), Table 7 coefficients.

    WHY A FILTER IS NEEDED AT ALL
    -----------------------------
    These compact schemes are CENTRAL, meaning symmetric about each grid
    point. Symmetry is desirable because it adds no artificial damping,
    so wave amplitudes are not spuriously reduced. The drawback is that
    nothing damps genuine numerical noise either: small wiggles at the
    shortest wavelength the grid can represent, generated near steep
    gradients, simply accumulate.

    This filter is a deliberate, controlled way to remove just those
    wiggles. It is built so that its transfer function vanishes at the
    highest frequency the grid supports, T(pi) = 0, while staying
    essentially equal to 1 across the smooth, well-resolved part of the
    spectrum. In short: it removes the noise and leaves the physics.

    alphaF is a free parameter with |alphaF| < 0.5; the paper uses 0.4.
    Like the derivative operators the filter is implicit, so applying it
    means solving a small tridiagonal system.
    """
    a = filter_F12_coeffs(alphaF)
    N = len(f)
    rhs = a[0] * f.copy()
    for n in range(1, 7):
        rhs += (a[n] / 1) * (np.roll(f, -n) + np.roll(f, n)) / 1.0
    # NOTE: Eq. (5.1) RHS is sum_{n=0}^{N} (a_n/2)(f_{j+n}+f_{j-n}); a0 term
    # counted once (n=0 gives f_j+f_j -> a0*f_j already handled above with /2
    # folded into coefficients a_n as tabulated for n>=1, and a0 as the n=0
    # "diagonal" weight). We reconstruct precisely below instead:
    rhs = a[0] * f.copy()
    for n in range(1, 7):
        rhs += (a[n] / 2.0) * (np.roll(f, -n) + np.roll(f, n))
    Nn = len(f)
    idx = np.arange(Nn)
    rows = np.concatenate([idx, idx, idx])
    cols = np.concatenate([idx, (idx - 1) % Nn, (idx + 1) % Nn])
    data = np.concatenate([np.ones(Nn), np.full(Nn, alphaF), np.full(Nn, alphaF)])
    A = sparse.coo_matrix((data, (rows, cols)), shape=(Nn, Nn)).tocsc()
    lu = splu(A)
    return lu.solve(rhs)


def tvdrk3_step(u, dt, rhs_func):
    """One step of the third-order TVD Runge-Kutta method, Eq. (6.2).

    Runge-Kutta methods improve on the crude Euler step
    (u_new = u + dt*S(u)) by evaluating the right-hand side at several
    intermediate states and combining them, which cancels the leading
    error terms. This one uses three stages and is third-order accurate.

    TVD stands for Total Variation Diminishing: the method will not
    manufacture new oscillations that the equation did not already imply.
    That is exactly the property wanted near the steep fronts of
    Examples 7.4 and 7.6.

    Each stage below is a weighted average of the old value and an Euler
    step taken from the previous stage. The particular weights (3/4 and
    1/4, then 1/3 and 2/3) are what make the combination third-order
    accurate while preserving the TVD property.

    rhs_func : the function S(u) giving du/dt; see nonlinear_kdv.py.
    """
    u1 = u + dt * rhs_func(u)
    u2 = 0.75 * u + 0.25 * u1 + 0.25 * dt * rhs_func(u1)
    u_new = (1 / 3) * u + (2 / 3) * u2 + (2 / 3) * dt * rhs_func(u2)
    return u_new


# =======================================================================
# First-derivative compact schemes -- explicit coefficients from the two
# papers this paper cites (but does not itself tabulate numerically) for
# the convective term g(u)_x in the nonlinear examples:
#
#   [1]  S.K. Lele, "Compact finite difference schemes with spectral-like
#        resolution," J. Comput. Phys. 103 (1992) 16-42.
#        Node-only ("cell-node compact scheme", CNCS) family:
#          beta f'_{j-2} + alpha f'_{j-1} + f'_j + alpha f'_{j+1} + beta f'_{j+2}
#            = a (f_{j+1}-f_{j-1})/(2h) + b (f_{j+2}-f_{j-2})/(4h)
#                                       + c (f_{j+3}-f_{j-3})/(6h)
#        The eighth-order TRIDIAGONAL member (beta=0) has
#            alpha = 3/8,  a = 25/16,  b = 1/5,  c = -1/80
#        (verified here directly against Lele's order-condition ladder:
#         a+b+c = 1+2 alpha; a+4b+9c = 6 alpha; a+16b+81c = 10 alpha;
#         a+64b+729c = 14 alpha -- all four hold exactly with these values).
#
#   [23] X. Liu, S. Zhang, H. Zhang, C.-W. Shu, "A new class of central
#        compact schemes with spectral-like resolution I: linear schemes,"
#        Commun. Comput. Phys., preprint (2013 conference version used
#        here). Their "CCS" (central compact scheme) couples NODE and
#        CENTER values, Eqs. (2.5) [nodes] / (2.7) [centers]:
#          beta f'_{j-2} + alpha f'_{j-1} + f'_j + alpha f'_{j+1} + beta f'_{j+2}
#            = a (f_{j+1/2}-f_{j-1/2})/h + b (f_{j+1}-f_{j-1})/(2h)
#            + c (f_{j+3/2}-f_{j-3/2})/(3h) + d (f_{j+2}-f_{j-2})/(4h)
#            + e (f_{j+5/2}-f_{j-5/2})/(5h)
#        Their Table 2.2, row "CCS-T8" (eighth-order, tridiagonal):
#            alpha = -3/20,  a = 2,  b = -61/50,  c = -2/25,  d = e = 0
#        (also verified here against their own Eq. 2.8: a+b+c = 1+2alpha).
# =======================================================================

_LELE_CNCS_T8 = dict(alpha=3 / 8, a=25 / 16, b=1 / 5, c=-1 / 80)
_LIU_CCS_T8 = dict(alpha=-3 / 20, a=2.0, b=-61 / 50, c=-2 / 25)


def first_derivative_Lele_CNCS8(f, dx):
    """
    Node-only 8th-order tridiagonal compact first derivative, from Lele
    (1992) [ref 1 in the paper] -- used for the convective term g(u)_x in
    TDCNCS's nonlinear examples, since the paper states TDCNCS pairs its
    Eq. (2.4) third-derivative scheme with "the existing eighth-order
    cell-node compact scheme [1]" for the first derivative.
    """
    p = _LELE_CNCS_T8
    N = len(f)
    rhs = (p['a'] * (np.roll(f, -1) - np.roll(f, 1)) / (2 * dx)
           + p['b'] * (np.roll(f, -2) - np.roll(f, 2)) / (4 * dx)
           + p['c'] * (np.roll(f, -3) - np.roll(f, 3)) / (6 * dx))
    lu = _get_pentadiag_lu(N, p['alpha'], 0.0)
    return lu.solve(rhs)


def first_derivative_Liu_CCS8(f_node, f_half, dx):
    """
    Coupled node+center 8th-order tridiagonal compact first derivative,
    from Liu, Zhang, Zhang & Shu (2013) [ref 23 in the paper] -- used for
    the convective term g(u)_x in TDCCS's nonlinear examples, since the
    paper states TDCCS pairs its Eqs. (3.1)-(3.2) third-derivative scheme
    with "the existing eighth-order central compact scheme [23]" for the
    first derivative. Structurally identical in spirit to third_derivative_
    TDCCS: nodes and centers are evolved together, no interpolation.

    f_node : values at x_0..x_{N-1}
    f_half : values at x_{1/2}, x_{3/2}, ..., x_{N-1/2} (f_half[k] = f_{k+1/2})
    Returns (f1_node, f1_half).
    """
    p = _LIU_CCS_T8
    N = len(f_node)
    fn, fh = f_node, f_half
    # Eq. (2.5) at node j (d=e=0 for the T8 member):
    rhs_node = (p['a'] * (fh - np.roll(fh, 1)) / dx
                + p['b'] * (np.roll(fn, -1) - np.roll(fn, 1)) / (2 * dx)
                + p['c'] * (np.roll(fh, -1) - np.roll(fh, 2)) / (3 * dx))
    # Eq. (2.7), re-centered at half-grid index k = j-1 exactly as we did
    # for the third-derivative TDCCS operator (see third_derivative_TDCCS):
    rhs_half = (p['a'] * (np.roll(fn, -1) - fn) / dx
                + p['b'] * (np.roll(fh, -1) - np.roll(fh, 1)) / (2 * dx)
                + p['c'] * (np.roll(fn, -2) - np.roll(fn, 1)) / (3 * dx))
    lu = _get_pentadiag_lu(N, p['alpha'], 0.0)
    f1_node = lu.solve(rhs_node)
    f1_half = lu.solve(rhs_half)
    return f1_node, f1_half


