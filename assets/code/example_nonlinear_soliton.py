"""
Nonlinear KdV examples (reproduction of Examples 7.2/7.3/7.4/7.6 of
Salian, Samala & Ghosh 2026): single-soliton propagation, double-soliton
collision, soliton splitting with filtering, the zero-dispersion limit and
top-hat breakup, and the Ito-type coupled system.

*** First-derivative operators now match the paper's own citations ***
These equations need a first-derivative operator for the nonlinear flux
term g(u)_x, e.g. u_t + (u^2/2)_x + eps*u_xxx = 0. The paper states it uses:
  - "the existing eighth-order cell-node compact scheme [1]" (Lele 1992)
    for TDCNCS, and
  - "the existing eighth-order central compact scheme [23]" (Liu, Zhang,
    Zhang & Shu 2013) for TDCCS,
paired with its own third-derivative operators (Eq. 2.4 for TDCNCS,
Eqs. 3.1-3.2 for TDCCS). Neither first-derivative scheme's coefficients are
tabulated numerically *in this paper* -- they live in refs [1] and [23] --
so an earlier version of this file substituted a spectral (FFT) first
derivative and flagged the substitution explicitly. That placeholder has
now been replaced with the two schemes' actual published coefficients,
implemented in tdccs_lib.py:

  - first_derivative_Lele_CNCS8: Lele (1992) J. Comput. Phys. 103, 16-42,
    the node-only 8th-order tridiagonal member of his classical family
    (alpha=3/8, a=25/16, b=1/5, c=-1/80). Verified here against Lele's own
    Taylor order-condition ladder before use (see tdccs_lib.py docstring).

  - first_derivative_Liu_CCS8: Liu, Zhang, Zhang & Shu (2013), "A new class
    of central compact schemes with spectral-like resolution I: linear
    schemes" -- their Table 2.2, row CCS-T8 (alpha=-3/20, a=2, b=-61/50,
    c=-2/25, d=e=0), which couples node and cell-center values exactly the
    way this paper's own TDCCS couples them for the third derivative (no
    interpolation). Verified against their Eq. (2.8) order-2 condition
    before use.

With these in place, TDCNCS solves use ONLY node values throughout (Lele
CNCS8 + this paper's Eq. 2.4), and TDCCS solves evolve node AND center
values together throughout (Liu CCS8 + this paper's Eqs. 3.1-3.2) -- an
internally consistent, fully-cited discretization on both sides, matching
the paper's own scheme pairing. Grid sizes and integration windows are
still reduced from the paper's in several examples purely for wall-clock
budget in this session (the dt ~ CFL*Delta x^3 restriction is the binding
constraint); each such reduction is noted at the point it's used.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pub_style import apply_style, savefig_all, FIGDIR
from tdccs_lib import (TDCNCS, TDCCS, third_derivative_TDCNCS, third_derivative_TDCCS,
                        first_derivative_Lele_CNCS8, first_derivative_Liu_CCS8,
                        tvdrk3_step, apply_filter_F12)

apply_style()
OUT = FIGDIR   # <project>/figs, resolved relative to pub_style.py


# =======================================================================
# TDCNCS-based nonlinear solvers (node values only)
# =======================================================================
def solve_generic_TDCNCS(N, xmin, xmax, T, u0_func, flux_coef, disp_coef,
                          cfl=0.05, filter_every=None, alphaF=0.4):
    """
    Solves u_t + flux_coef*(u^2)_x + disp_coef*u_xxx = 0 using Lele's (1992)
    eighth-order cell-node compact scheme for the convective term and this
    paper's TDCNCS (Eq. 2.4) for the dispersive term -- exactly the pairing
    the paper describes for TDCNCS. Optionally applies the F12 low-pass
    filter (Eq. 5.1, Table 7) every `filter_every` RK3 steps, as in the
    paper's Fig. 10.
    """
    L = xmax - xmin
    dx = L / N
    x = xmin + np.arange(N) * dx
    u = u0_func(x)

    def rhs(u):
        conv = flux_coef * first_derivative_Lele_CNCS8(u**2, dx)
        disp = disp_coef * third_derivative_TDCNCS(u, dx, TDCNCS["T8"])
        return -(conv + disp)

    dt = cfl * dx**3
    nsteps = max(1, int(np.ceil(T / dt)))
    dt = T / nsteps
    for step in range(nsteps):
        u = tvdrk3_step(u, dt, rhs)
        if filter_every and (step + 1) % filter_every == 0:
            u = apply_filter_F12(u, alphaF)
    return x, u


def solve_soliton_TDCNCS(N, xmin, xmax, T, u0_func, eps, cfl=0.05):
    L = xmax - xmin
    dx = L / N
    x = xmin + np.arange(N) * dx
    u = u0_func(x)

    def rhs(u):
        conv = first_derivative_Lele_CNCS8(0.5 * u**2, dx)
        disp = third_derivative_TDCNCS(u, dx, TDCNCS["T8"])
        return -(conv + eps * disp)

    dt = cfl * dx**3
    nsteps = max(1, int(np.ceil(T / dt)))
    dt = T / nsteps
    for _ in range(nsteps):
        u = tvdrk3_step(u, dt, rhs)
    return x, u


def solve_ito_TDCNCS(N, xmin, xmax, T, u0_func, v0_func, cfl=0.05):
    """
    Ito-type coupled system (Eq. 7.12):
        u_t - (3u^2+v^2)_x - u_xxx = 0
        v_t - 2(uv)_x = 0
    Dispersive term uses TDCNCS (Eq. 2.4); both flux derivatives use Lele's
    eighth-order cell-node compact scheme, consistent with the rest of this
    module's TDCNCS solves.
    """
    L = xmax - xmin
    dx = L / N
    x = xmin + np.arange(N) * dx
    u = u0_func(x)
    v = v0_func(x)

    def rhs(state):
        u, v = state[:N], state[N:]
        du = first_derivative_Lele_CNCS8(3 * u**2 + v**2, dx) + third_derivative_TDCNCS(u, dx, TDCNCS["T8"])
        dv = 2 * first_derivative_Lele_CNCS8(u * v, dx)
        return np.concatenate([du, dv])

    dt = cfl * dx**3
    nsteps = max(1, int(np.ceil(T / dt)))
    dt = T / nsteps
    state = np.concatenate([u, v])
    for _ in range(nsteps):
        state = tvdrk3_step(state, dt, rhs)
    return x, state[:N], state[N:]


# =======================================================================
# TDCCS-based nonlinear solver (node + center values, coupled throughout)
# =======================================================================
def solve_soliton_TDCCS(N, xmin, xmax, T, u0_func, eps, cfl=0.008):
    """
    Same equation as solve_soliton_TDCNCS, but using Liu et al.'s (2013)
    eighth-order central compact scheme for the convective term and this
    paper's TDCCS (Eqs. 3.1-3.2) for the dispersive term -- both node and
    center values are evolved together throughout, exactly the pairing the
    paper describes for TDCCS. No interpolation anywhere.
    """
    L = xmax - xmin
    dx = L / N
    x = xmin + np.arange(N) * dx
    xh = x + dx / 2
    un = u0_func(x)
    uh = u0_func(xh)

    def rhs(state):
        un, uh = state[:N], state[N:]
        conv_n, conv_h = first_derivative_Liu_CCS8(0.5 * un**2, 0.5 * uh**2, dx)
        f3n, f3h = third_derivative_TDCCS(un, uh, dx, TDCCS["T8"])
        return np.concatenate([-(conv_n + eps * f3n), -(conv_h + eps * f3h)])

    dt = cfl * dx**3
    nsteps = max(1, int(np.ceil(T / dt)))
    dt = T / nsteps
    state = np.concatenate([un, uh])
    for _ in range(nsteps):
        state = tvdrk3_step(state, dt, rhs)
    return x, state[:N]


if __name__ == "__main__":
    # ================================================================
    # Figure 7 -- Example 7.2 style: u_t - 3(u^2)_x + u_xxx = 0
    #   IC: u(x,0) = -2 sech^2(x), exact: u(x,t) = -2 sech^2(x-4t)
    # ================================================================
    def u0_72(x):
        return -2 / np.cosh(x)**2

    def exact_72(x, t):
        return -2 / np.cosh(x - 4 * t)**2

    N, T = 80, 0.5
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7))
    times = [0, 0.25, 0.5]
    colors = ['k', 'b', 'r']
    for t, cc in zip(times, colors):
        x, u = solve_generic_TDCNCS(N, -10, 12, t, u0_72, flux_coef=-3.0, disp_coef=1.0)
        axes[0, 0].plot(x, exact_72(x, t), '-', color=cc, lw=1.0)
        axes[0, 0].plot(x, u, 'o', ms=2.5, color=cc, mfc='none', label=fr'$t={t}$')
        axes[1, 0].plot(x, np.abs(u - exact_72(x, t)), color=cc)
    axes[0, 0].set_title("(a) TDCNCS solution")
    axes[1, 0].set_title("(b) TDCNCS pointwise error")
    axes[0, 0].set_xlabel(r"$x$"); axes[0, 0].set_ylabel(r"$u(x,t)$")
    axes[1, 0].set_xlabel(r"$x$"); axes[1, 0].set_ylabel("Error")
    axes[0, 0].legend(fontsize=8, loc="lower right", framealpha=0.9)
    Ns = [40, 60, 80, 100]
    errs = []
    for Nc in Ns:
        x, u = solve_generic_TDCNCS(Nc, -10, 12, 0.5, u0_72, flux_coef=-3.0, disp_coef=1.0)
        errs.append(np.max(np.abs(u - exact_72(x, 0.5))))
    axes[0, 1].loglog(Ns, errs, 'o-', color='tab:blue', label=r'$L^\infty$ error')
    axes[0, 1].set_xlabel(r"$N$"); axes[0, 1].set_ylabel("Error"); axes[0, 1].legend()
    axes[0, 1].set_title("(c) Grid convergence, $t=0.5$")
    axes[1, 1].axis('off')
    fig.tight_layout()
    savefig_all(fig, "figure7_soliton_example72", outdir=OUT)
    plt.close(fig)
    print("Figure 7 written.")

    # ================================================================
    # Figure 8 -- single soliton at small epsilon (paper's Example 7.3
    #   scaling), TDCNCS vs TDCCS, both now using their fully-cited
    #   convective operators instead of the earlier spectral placeholder
    # ================================================================
    c, eps = 0.3, 5e-4
    k = 0.5 * np.sqrt(c / eps)
    x0 = 0.5

    def u0(x):
        return 3 * c / np.cosh(k * (x - x0))**2

    def exact(x, t):
        return 3 * c / np.cosh(k * ((x - x0) - c * t))**2

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    N_ncs, T_ncs = 80, 0.2
    x, u = solve_soliton_TDCNCS(N_ncs, 0, 2, T_ncs, u0, eps, cfl=0.05)
    axes[0, 0].plot(x, u0(x), 'k--', label='$t=0$ (initial)')
    axes[0, 0].plot(x, u, 'r-o', ms=3, label=fr'$t={T_ncs}$ (TDCNCS)')
    axes[0, 0].plot(x, exact(x, T_ncs), 'b:', label='exact')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title(f"TDCNCS + Lele CNCS8 convection, $N={N_ncs}$")
    axes[1, 0].plot(x, np.abs(u - exact(x, T_ncs)))
    axes[1, 0].set_title("TDCNCS pointwise error")
    err_ncs = np.max(np.abs(u - exact(x, T_ncs)))
    print(f"TDCNCS: Linf error at t={T_ncs}, N={N_ncs}: {err_ncs:.4e}")

    N_ccs, T_ccs = 60, 0.03
    x2, u2 = solve_soliton_TDCCS(N_ccs, 0, 2, T_ccs, u0, eps, cfl=0.008)
    axes[0, 1].plot(x2, u0(x2), 'k--', label='$t=0$ (initial)')
    axes[0, 1].plot(x2, u2, 'r-o', ms=3, label=fr'$t={T_ccs}$ (TDCCS)')
    axes[0, 1].plot(x2, exact(x2, T_ccs), 'b:', label='exact')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_title(f"TDCCS + Liu CCS8 convection, $N={N_ccs}$")
    axes[1, 1].plot(x2, np.abs(u2 - exact(x2, T_ccs)))
    axes[1, 1].set_title("TDCCS pointwise error")
    err_ccs = np.max(np.abs(u2 - exact(x2, T_ccs)))
    print(f"TDCCS: Linf error at t={T_ccs}, N={N_ccs}: {err_ccs:.4e}")

    fig.tight_layout()
    savefig_all(fig, "figure8_soliton_illustrative", outdir=OUT)
    plt.close(fig)
    print("Figure 8 written.")

    # ================================================================
    # Figure 9 -- double soliton collision, Eq. (7.7)
    #   u_t + (u^2/2)_x + eps u_xxx = 0
    # ================================================================
    eps97 = 4.84e-4
    c1, c2, x1, x2 = 0.3, 0.1, 0.4, 0.8
    k1, k2 = 0.5 * np.sqrt(c1 / eps97), 0.5 * np.sqrt(c2 / eps97)

    def u0_double(x):
        return 3 * c1 / np.cosh(k1 * (x - x1))**2 + 3 * c2 / np.cosh(k2 * (x - x2))**2

    N9, T9 = 40, 0.2   # reduced from the paper's N=100, T up to 4, for runtime
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
    times9 = [0, 0.1, 0.2]
    snaps = []
    for t in times9:
        x, u = solve_generic_TDCNCS(N9, 0, 2, t, u0_double, flux_coef=0.5, disp_coef=eps97, cfl=0.05)
        snaps.append(u)
        ax[0].plot(x, u, label=fr'$t={t}$')
    ax[0].set_title("(a) TDCNCS: double-soliton collision")
    ax[0].set_xlabel(r"$x$"); ax[0].set_ylabel(r"$u(x,t)$"); ax[0].legend(fontsize=8)

    Nt_frames = 8
    Xall = np.linspace(0, T9, Nt_frames)
    Umat = np.zeros((Nt_frames, N9))
    for i, t in enumerate(Xall):
        _, u = solve_generic_TDCNCS(N9, 0, 2, t, u0_double, flux_coef=0.5, disp_coef=eps97, cfl=0.05)
        Umat[i] = u
    Tg, Xg = np.meshgrid(Xall, x, indexing='ij')
    cf = ax[1].contourf(Xg, Tg, Umat, levels=30, cmap='turbo')
    fig.colorbar(cf, ax=ax[1])
    ax[1].set_xlabel(r"$x$"); ax[1].set_ylabel(r"$t$")
    ax[1].set_title("(b) space-time contour")
    fig.tight_layout()
    savefig_all(fig, "figure9_double_soliton", outdir=OUT)
    plt.close(fig)
    print("Figure 9 written.")

    # ================================================================
    # Figure 10 -- soliton splitting + effect of the F12 low-pass filter
    #   (reduced N, T from the paper's N=150, t up to 4, for runtime)
    # ================================================================
    eps10 = 1e-4

    def u0_split(x):
        return (2 / 3) / np.cosh((x - 1) / np.sqrt(108 * eps10))**2

    N10, T10 = 80, 0.3
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    x_nf, u_nf = solve_generic_TDCNCS(N10, 0, 3, T10, u0_split, flux_coef=0.5, disp_coef=eps10, cfl=0.05)
    x_f, u_f = solve_generic_TDCNCS(N10, 0, 3, T10, u0_split, flux_coef=0.5, disp_coef=eps10, cfl=0.05,
                                     filter_every=20, alphaF=0.4)
    axes[0].plot(x_nf, u0_split(x_nf), 'k--', label='$t=0$')
    axes[0].plot(x_nf, u_nf, 'b-', label=f'$t={T10}$, unfiltered')
    axes[0].set_title("(a) TDCNCS, no filter")
    axes[0].set_xlabel(r"$x$"); axes[0].set_ylabel(r"$u(x,t)$"); axes[0].legend(fontsize=8)
    axes[1].plot(x_f, u0_split(x_f), 'k--', label='$t=0$')
    axes[1].plot(x_f, u_f, 'r-', label=f'$t={T10}$, F12 filtered')
    axes[1].set_title("(b) TDCNCS + F12 filter every 20 steps")
    axes[1].set_xlabel(r"$x$"); axes[1].legend(fontsize=8)
    fig.tight_layout()
    savefig_all(fig, "figure10_filter_effect", outdir=OUT)
    plt.close(fig)
    print("Figure 10 written.")

    # ================================================================
    # Figure 11 -- zero-dispersion limit, illustrative (heavily reduced
    #   scale vs. the paper: eps here is far larger and N far coarser,
    #   because the true eps=1e-4 case needs N~1000+ and dt ~ dx^3 --
    #   this is flagged clearly as a qualitative demo of the oscillatory
    #   dispersive-wave-train onset, not a resolution-matched reproduction.
    #   NOTE: this limitation is about GRID RESOLUTION, not about which
    #   convective scheme is used -- switching to the exact Lele operator
    #   does not change that, so this figure remains illustrative-scale.)
    # ================================================================
    eps11 = 5e-2

    def u0_zd(x):
        return 2 + 0.5 * np.sin(2 * np.pi * x)

    N11 = 32
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.2))
    for T11, cc in zip([0, 0.02, 0.05], ['k', 'b', 'r']):
        if T11 == 0:
            x = np.arange(N11) / N11
            u = u0_zd(x)
        else:
            x, u = solve_generic_TDCNCS(N11, 0, 1, T11, u0_zd, flux_coef=0.5, disp_coef=eps11, cfl=0.05)
        ax.plot(x, u, color=cc, label=fr'$t={T11}$')
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$u(x,t)$")
    ax.set_title(fr"Dispersive oscillation onset, illustrative ($\epsilon={eps11}$)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig_all(fig, "figure11_zero_dispersion_illustrative", outdir=OUT)
    plt.close(fig)
    print("Figure 11 written (illustrative scale, see caption).")

    # ================================================================
    # Figure 12 -- top-hat initial condition, illustrative (heavily
    #   reduced scale vs. the paper's N=1000, eps=1e-4, for the same
    #   dt~dx^3 runtime reason as Figure 11)
    # ================================================================
    eps12 = 5e-3

    def u0_th(x):
        return np.where((x > 0.5) & (x < 3.5), 1.0, 0.0)

    N12 = 200
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
    for col, filt in enumerate([None, 20]):
        for T12, cc in zip([0, 0.02], ['k', 'b']):
            if T12 == 0:
                x = np.arange(N12) * (5.0 / N12)
                u = u0_th(x)
            else:
                x, u = solve_generic_TDCNCS(N12, 0, 5, T12, u0_th, flux_coef=0.5, disp_coef=eps12,
                                             cfl=0.05, filter_every=filt, alphaF=0.4)
            ax[col].plot(x, u, color=cc, label=fr'$t={T12}$')
        ax[col].set_title("(a) unfiltered" if filt is None else "(b) F12 filtered")
        ax[col].set_xlabel(r"$x$"); ax[col].legend(fontsize=8)
    ax[0].set_ylabel(r"$u(x,t)$")
    fig.tight_layout()
    savefig_all(fig, "figure12_tophat_illustrative", outdir=OUT)
    plt.close(fig)
    print("Figure 12 written (illustrative scale, see caption).")

    # ================================================================
    # Figure 14 -- Ito-type coupled system, trigonometric IC (Eq. 7.13)
    # ================================================================
    N14, T14 = 80, 1.0
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    for t, cc in zip([0, 0.5, 1.0], ['k', 'b', 'r']):
        x, u, v = solve_ito_TDCNCS(N14, 0, 2 * np.pi, t, np.cos, np.cos, cfl=0.05)
        axes[0].plot(x, u, color=cc, label=fr'$t={t}$')
        axes[1].plot(x, v, color=cc, label=fr'$t={t}$')
    axes[0].set_title(r"(a) $u(x,t)$ -- dispersive component")
    axes[1].set_title(r"(b) $v(x,t)$ -- shock-type component")
    for a in axes:
        a.set_xlabel(r"$x$"); a.legend(fontsize=8)
    fig.tight_layout()
    savefig_all(fig, "figure14_ito_trig", outdir=OUT)
    plt.close(fig)
    print("Figure 14 written.")

    # ================================================================
    # Figure 15 -- Ito-type coupled system, Gaussian IC (Eq. 7.14)
    # ================================================================
    N15, T15 = 160, 2.0

    def gauss(x):
        return np.exp(-x**2)

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
    for t, cc in zip([0, 1.0, 2.0], ['k', 'b', 'r']):
        x, u, v = solve_ito_TDCNCS(N15, -15, 15, t, gauss, gauss, cfl=0.05)
        ax[0].plot(x, u, color=cc, label=fr'$t={t}$')
        ax[1].plot(x, v, color=cc, label=fr'$t={t}$')
    ax[0].set_title(r"(a) $u(x,t)$")
    ax[1].set_title(r"(b) $v(x,t)$")
    for a in ax:
        a.set_xlabel(r"$x$"); a.legend(fontsize=8)
    fig.tight_layout()
    savefig_all(fig, "figure15_ito_gaussian", outdir=OUT)
    plt.close(fig)
    print("Figure 15 written.")


