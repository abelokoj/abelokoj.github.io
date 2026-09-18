"""
Reproduces Figure 4 and Eq. (6.3) (linear stability / CFL bound) of
Salian, Samala & Ghosh (2026).

The TDCNCS third-derivative operator is a circulant matrix (both
its implicit pentadiagonal LHS and explicit-stencil RHS are circulant), so
its eigenvalues equal i * omega'''_TDCNCS(omega)/Delta x^3 for the discrete
wavenumbers omega = 2*pi*m/N, m = -N/2..N/2 -- this is just Eq. (2.6)
evaluated on the grid.

TDCCS, however, evolves NODE and CENTER values as one coupled 2N-vector
(Eqs. 3.1 and 3.2 together), so its symbol is a  2x2 matrix per
Fourier mode rather than a scalar. Rather than re-derive that 2x2 block
symbol by hand (easy to get a sign/phase wrong), we build the actual
2N x 2N linear operator numerically (by applying third_derivative_TDCCS to
unit vectors) and take its eigenvalues directly with numpy -- this is
guaranteed to match whatever Eqs. (3.1)-(3.2) actually implement.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pub_style import subcaption, apply_style, savefig_all, FIGDIR
from tdccs_lib import TDCNCS, TDCCS, third_derivative_TDCNCS, third_derivative_TDCCS

apply_style()
OUT = FIGDIR   # <project>/figs, resolved relative to pub_style.py

# ---------------------------------------------------------------------
# TDCNCS eigenvalues: circulant -> diagonalized by omega'''(omega)
# ---------------------------------------------------------------------
from tdccs_lib import omega3_TDCNCS
omega_grid = np.linspace(-np.pi, np.pi, 2001)
eig_tdcncs = 1j * omega3_TDCNCS(omega_grid, TDCNCS["T8"])   # dx = 1 reference
lam_max_tdcncs = np.max(np.abs(eig_tdcncs))

# ---------------------------------------------------------------------
# TDCCS eigenvalues: build the 2N x 2N operator explicitly and diagonalize
# ---------------------------------------------------------------------
def tdccs_operator_matrix(N, coeffs):
    """
    Returns the 2N x 2N matrix L such that
        [f3_node ; f3_half] = L @ [f_node ; f_half]      (dx = 1)
    by applying third_derivative_TDCCS to unit basis vectors.
    """
    L = np.zeros((2 * N, 2 * N))
    for k in range(2 * N):
        e = np.zeros(2 * N)
        e[k] = 1.0
        fn = e[:N]
        fh = e[N:]
        f3n, f3h = third_derivative_TDCCS(fn, fh, 1.0, coeffs)
        L[:N, k] = f3n
        L[N:, k] = f3h
    return L

N = 48
L = tdccs_operator_matrix(N, TDCCS["T8"])
eig_tdccs = np.linalg.eigvals(L)
lam_max_tdccs = np.max(np.abs(eig_tdccs))

print(f"max |eigenvalue|  TDCNCS-T8 : {lam_max_tdcncs:.3f}   (paper: 15.157)")
print(f"max |eigenvalue|  TDCCS-T8  : {lam_max_tdccs:.3f}   (paper: 147.168)")

# ---------------------------------------------------------------------
# TVDRK3 stability region + imaginary-axis intercept
# ---------------------------------------------------------------------
def rk3_stability_poly(z):
    return 1 + z + z**2 / 2 + z**3 / 6

# Solve |R(iy)| = 1 for the largest real y>0 root -> stability boundary
# intercept with the imaginary axis
from scipy.optimize import brentq
def f(y):
    return abs(rk3_stability_poly(1j * y)) - 1
y_intercept = brentq(f, 1.0, 2.0)
print(f"TVDRK3 stability boundary intercepts imaginary axis at +/-{y_intercept:.4f} (paper: 1.732)")

dt_dx3_tdcncs = y_intercept / lam_max_tdcncs
dt_dx3_tdccs = y_intercept / lam_max_tdccs
print(f"CFL bound: dt/dx^3 <= {dt_dx3_tdcncs:.3f} (TDCNCS), {dt_dx3_tdccs:.4f} (TDCCS)   "
      f"[paper: 0.11, 0.011]")

# ---------------------------------------------------------------------
# Figure 4a: eigenvalues (exact / TDCCS / TDCNCS)
# ---------------------------------------------------------------------
# The paper plots the scaled eigenvalues over omega = k*dx in [0, 2*pi],
# not [-pi, pi]. That choice is what puts the exact curve at -(2*pi)^3,
# about -248, and matches the -250 to 0 vertical range of its Figure 4a.
# Every eigenvalue is purely imaginary because these are central schemes,
# so all three sets collapse onto the vertical line Real = 0 and the plot
# reads as three line segments of differing length.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

w = np.linspace(0.0, 2.0 * np.pi, 4001)
exact_imag = -(w ** 3)
tdcncs_imag = -np.abs(omega3_TDCNCS(w, TDCNCS["T8"]))
# TDCCS has no closed-form symbol here; its eigenvalues come from the
# explicit 2N x 2N operator diagonalized above. They are purely
# imaginary, so the negative branch is what the paper plots.
tdccs_imag = -np.sort(np.abs(eig_tdccs.imag))

axes[0].plot(np.zeros_like(exact_imag), exact_imag, color="green", lw=1.4,
             label="Exact")
axes[0].plot(np.zeros_like(tdccs_imag), tdccs_imag, color="tab:blue", lw=1.4,
             label="TDCCS")
axes[0].plot(np.zeros_like(tdcncs_imag), tdcncs_imag, color="red", lw=1.4,
             linestyle="--", label="TDCNCS")

# Double-headed arrows marking the extent of each set, as the paper does.
for xpos, lo, tag in ((1.15, tdcncs_imag.min(), r"$\langle i \rangle$"),
                      (1.60, tdccs_imag.min(), r"$\langle ii \rangle$"),
                      (2.05, exact_imag.min(), r"$\langle iii \rangle$")):
    axes[0].annotate("", xy=(xpos, 0.0), xytext=(xpos, lo),
                     arrowprops=dict(arrowstyle="<->", color="0.25", lw=0.9))
    axes[0].text(xpos + 0.10, 0.5 * lo, tag, va="center", fontsize=9)

axes[0].set_xlabel("Real")
axes[0].set_ylabel("Imaginary")
axes[0].set_xlim(-1, 3)
axes[0].set_ylim(1.05 * exact_imag.min(), 12)
axes[0].legend(loc="lower right", framealpha=0.95)
subcaption(axes[0], "(a)")

# ---------------------------------------------------------------------
# Figure 4b: TVDRK3 stability region + scaled eigenvalues
# ---------------------------------------------------------------------
xg = np.linspace(-3.5, 1, 500)
yg = np.linspace(-3, 3, 500)
X, Y = np.meshgrid(xg, yg)
Rmag = np.abs(rk3_stability_poly(X + 1j * Y))
axes[1].contour(X, Y, Rmag, levels=[1.0], colors="k", linestyles="--")
axes[1].plot([], [], "k--", label="RK3")

# Each scheme's eigenvalues scaled by its own maximum stable CFL, so both
# just reach the imaginary-axis intercept of the stability boundary.
sc_n = tdcncs_imag / lam_max_tdcncs * y_intercept
sc_c = tdccs_imag / lam_max_tdccs * y_intercept
axes[1].plot(np.zeros_like(sc_n), sc_n, color="red", lw=1.4, label="TDCNCS")
axes[1].plot(np.zeros_like(sc_n), -sc_n, color="red", lw=1.4)
axes[1].plot(np.zeros_like(sc_c), sc_c, ".", ms=2, color="tab:blue", label="TDCCS")
axes[1].plot(np.zeros_like(sc_c), -sc_c, ".", ms=2, color="tab:blue")
axes[1].set_xlabel("Real")
axes[1].set_ylabel("Imaginary")
axes[1].set_xlim(-3, 0.5)
axes[1].set_ylim(-2.5, 2.5)
axes[1].legend(loc="lower left", framealpha=0.95)
subcaption(axes[1], "(b)")

fig.tight_layout()
savefig_all(fig, "figure4_stability", outdir=OUT)
plt.close(fig)
print("Figure 4 written.")


