"""
kh_core.py
==========
Core routines for the Kelvin-Helmholtz mini-project (blog post p10).

Two independent pieces live here, and the point of the project is that they agree:

1. Linear stability theory for the hyperbolic-tangent shear layer U(y) = tanh(y), solved with Chebyshev collocation. The inviscid Rayleigh equation is solved on the whole real line through an algebraic mapping; the viscous Orr-Sommerfeld equation is solved on a truncated interval with clamped ends.

2. A pseudo-spectral (Fourier-Fourier) solver for the two-dimensional incompressible Navier-Stokes equations in vorticity-streamfunction form on a doubly periodic box, with 2/3-rule dealiasing and a fourth-order Runge-Kutta scheme with an integrating factor for the viscous term. A passive scalar is advected alongside the vorticity for visualization.

All quantities are nondimensionalized by the free-stream speed U0 (half the velocity difference) and the shear-layer half-thickness delta, so the base profile is U = tanh(y) and the Reynolds number is Re = U0 delta / nu.
"""
import numpy as np
import scipy.fft as sfft
import scipy.linalg as sla
from scipy.interpolate import BarycentricInterpolator


# ----------------------------------------------------------------------
# Chebyshev differentiation (Trefethen, Spectral Methods in MATLAB, 2000)
# ----------------------------------------------------------------------
def cheb(N):
    """Chebyshev points x_j = cos(pi j / N) and the differentiation matrix."""
    if N == 0:
        return np.zeros((1, 1)), np.array([1.0])
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack([2.0, np.ones(N - 1), 2.0]) * (-1.0) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(N + 1))
    D -= np.diag(D.sum(axis=1))
    return D, x


# ----------------------------------------------------------------------
# Inviscid theory: Rayleigh equation on the whole line
# ----------------------------------------------------------------------
def rayleigh_growth(alpha, N=160, a=4.0):
    """Most unstable temporal eigenvalue of the Rayleigh equation for U = tanh(y).

    (U - c)(phi'' - alpha^2 phi) - U'' phi = 0,   phi -> 0 as |y| -> inf.

    The real line is mapped onto (-1, 1) by y = a xi / sqrt(1 - xi^2), so no truncation of the domain is needed; phi = 0 at xi = +-1 removes the end points. Returns the complex phase speed c and the growth rate omega_i = alpha c_i.
    """
    D, xi = cheb(N)
    xi_i = xi[1:-1]
    y = a * xi_i / np.sqrt(1.0 - xi_i**2)
    # derivatives of the inverse map xi(y) = y / sqrt(a^2 + y^2)
    dxi = a**2 / (a**2 + y**2) ** 1.5
    d2xi = -3.0 * a**2 * y / (a**2 + y**2) ** 2.5
    D1 = D[1:-1, 1:-1]
    D2 = (D @ D)[1:-1, 1:-1]
    Dyy = np.diag(dxi**2) @ D2 + np.diag(d2xi) @ D1
    U = np.tanh(y)
    Upp = -2.0 * np.tanh(y) / np.cosh(np.clip(y, -300.0, 300.0)) ** 2
    I = np.eye(N - 1)
    B = Dyy - alpha**2 * I
    A = np.diag(U) @ B - np.diag(Upp)
    c = sla.eig(A, B, right=False)
    c = c[np.isfinite(c) & (np.abs(c) < 5.0)]
    k = np.argmax(c.imag)
    return c[k], alpha * c[k].imag


# ----------------------------------------------------------------------
# Viscous theory: Orr-Sommerfeld equation on a truncated interval
# ----------------------------------------------------------------------
def orr_sommerfeld(alpha, Re, Y=16.0, N=200, return_mode=False):
    """Most unstable temporal mode of the Orr-Sommerfeld equation for U = tanh(y).

    (U - c)(phi'' - a^2 phi) - U'' phi = (phi'''' - 2 a^2 phi'' + a^4 phi) / (i a Re)

    on -Y < y < Y with phi = phi' = 0 at both ends. The clamped fourth-derivative operator follows Trefethen (2000, program 40). Returns c, omega_i = alpha c_i and, optionally, the eigenfunction (the perturbation streamfunction amplitude) on the Chebyshev grid.
    """
    D, x = cheb(N)
    D2 = D @ D
    D3 = D2 @ D
    D4c = D3 @ D
    S = np.diag(np.hstack([0.0, 1.0 / (1.0 - x[1:-1] ** 2), 0.0]))
    D4 = (np.diag(1.0 - x**2) @ D4c - 8.0 * np.diag(x) @ D3 - 12.0 * D2) @ S
    D2 = D2[1:-1, 1:-1] / Y**2
    D4 = D4[1:-1, 1:-1] / Y**4
    y = Y * x[1:-1]
    U = np.tanh(y)
    Upp = -2.0 * np.tanh(y) / np.cosh(np.clip(y, -300.0, 300.0)) ** 2
    I = np.eye(N - 1)
    B = D2 - alpha**2 * I
    L = D4 - 2.0 * alpha**2 * D2 + alpha**4 * I
    A = np.diag(U) @ B - np.diag(Upp) - L / (1j * alpha * Re)
    c, V = sla.eig(A, B)
    ok = np.isfinite(c) & (np.abs(c) < 5.0)
    idx = np.where(ok)[0][np.argmax(c[ok].imag)]
    cc = c[idx]
    if not return_mode:
        return cc, alpha * cc.imag
    phi = np.hstack([0.0, V[:, idx], 0.0])
    phi = phi / phi[np.argmax(np.abs(phi))]
    return cc, alpha * cc.imag, Y * x, phi


# ----------------------------------------------------------------------
# Pseudo-spectral Navier-Stokes solver (vorticity form)
# ----------------------------------------------------------------------
class SpectralNS2D:
    """Doubly periodic 2-D Navier-Stokes in vorticity-streamfunction form.

    omega_t + u omega_x + v omega_y = nu Lap(omega) + F,   Lap(psi) = -omega,
    u = psi_y + Ubar,   v = -psi_x,

    plus a passive scalar  c_t + u c_x + v c_y = kappa Lap(c).

    Ubar is the box-averaged streamwise velocity, which the streamfunction cannot carry and which is conserved. Time integration is RK4 with an integrating factor exp(-nu k^2 t), so the viscous term is treated exactly and the time step is limited only by advection. Nonlinear products are dealiased with the 2/3 rule.
    """

    def __init__(self, nx, ny, Lx, Ly, nu, kappa=None, workers=-1):
        self.nx, self.ny, self.Lx, self.Ly = nx, ny, Lx, Ly
        self.nu = nu
        self.kappa = nu if kappa is None else kappa
        self.workers = workers
        self.x = np.arange(nx) * Lx / nx
        self.y = np.arange(ny) * Ly / ny
        kx = 2.0 * np.pi / Lx * np.arange(nx // 2 + 1)
        ky = 2.0 * np.pi / Ly * sfft.fftfreq(ny, 1.0 / ny)
        self.KX, self.KY = np.meshgrid(kx, ky)          # shape (ny, nx//2+1)
        self.K2 = self.KX**2 + self.KY**2
        self.K2inv = np.where(self.K2 > 0, 1.0 / np.where(self.K2 > 0, self.K2, 1.0), 0.0)
        kxmax, kymax = kx.max(), np.abs(ky).max()
        self.dealias = (np.abs(self.KX) < 2.0 / 3.0 * kxmax) & (np.abs(self.KY) < 2.0 / 3.0 * kymax)
        self.Ubar = 0.0
        self.forcing_hat = None
        self.ikx = 1j * self.KX
        self.iky = 1j * self.KY

    # transforms -------------------------------------------------------
    def fft(self, f):
        return sfft.rfft2(f, workers=self.workers)

    def ifft(self, fh):
        return sfft.irfft2(fh, s=(self.ny, self.nx), workers=self.workers)

    # velocity ---------------------------------------------------------
    def velocity_hat(self, wh):
        psih = wh * self.K2inv
        return self.iky * psih, -self.ikx * psih

    def velocity(self, wh):
        uh, vh = self.velocity_hat(wh)
        return self.ifft(uh) + self.Ubar, self.ifft(vh)

    # right-hand side (advection + forcing) -----------------------------
    def _advect(self, qh, u, v):
        qx = self.ifft(self.ikx * qh)
        qy = self.ifft(self.iky * qh)
        return -self.fft(u * qx + v * qy) * self.dealias

    def rhs(self, wh, ch):
        u, v = self.velocity(wh)
        nw = self._advect(wh, u, v)
        if self.forcing_hat is not None:
            nw = nw + self.forcing_hat
        nc = self._advect(ch, u, v) if ch is not None else None
        return nw, nc

    # time stepping ----------------------------------------------------
    def step(self, wh, ch, dt):
        """One RK4 step with integrating factors for both fields."""
        Ew = np.exp(-self.nu * self.K2 * dt / 2.0)
        Ec = np.exp(-self.kappa * self.K2 * dt / 2.0)
        scal = ch is not None

        k1w, k1c = self.rhs(wh, ch)
        w2 = Ew * (wh + 0.5 * dt * k1w)
        c2 = Ec * (ch + 0.5 * dt * k1c) if scal else None
        k2w, k2c = self.rhs(w2, c2)
        w3 = Ew * wh + 0.5 * dt * k2w
        c3 = Ec * ch + 0.5 * dt * k2c if scal else None
        k3w, k3c = self.rhs(w3, c3)
        w4 = Ew**2 * wh + dt * Ew * k3w
        c4 = Ec**2 * ch + dt * Ec * k3c if scal else None
        k4w, k4c = self.rhs(w4, c4)
        wn = Ew**2 * wh + dt / 6.0 * (Ew**2 * k1w + 2.0 * Ew * (k2w + k3w) + k4w)
        cn = None
        if scal:
            cn = Ec**2 * ch + dt / 6.0 * (Ec**2 * k1c + 2.0 * Ec * (k2c + k3c) + k4c)
        return wn, cn

    # diagnostics ------------------------------------------------------
    def energy_enstrophy(self, wh):
        """Box-averaged kinetic energy E = <|u|^2>/2 and enstrophy Z = <omega^2>/2."""
        u, v = self.velocity(wh)
        w = self.ifft(wh)
        return 0.5 * np.mean(u**2 + v**2), 0.5 * np.mean(w**2)

    def mode_amplitude(self, wh, m):
        """L2 amplitude of streamwise Fourier mode m (all ky) of the vorticity."""
        col = wh[:, m] / (self.nx * self.ny)
        return np.sqrt(np.sum(np.abs(col) ** 2))


def double_shear_layer(y, Ly):
    """Periodic double shear layer: U = -1, +1, -1 with tanh layers at Ly/4, 3Ly/4."""
    return np.tanh(y - Ly / 4.0) - np.tanh(y - 3.0 * Ly / 4.0) - 1.0


def double_shear_vorticity(y, Ly):
    """Base vorticity omega = -dU/dy of the double shear layer."""
    return -(1.0 / np.cosh(y - Ly / 4.0) ** 2 - 1.0 / np.cosh(y - 3.0 * Ly / 4.0) ** 2)


def eigenmode_streamfunction(solver, alpha, Re, amp, phase=0.0, Nc=200):
    """Perturbation streamfunction built from the Orr-Sommerfeld eigenmode of each layer.

    The mode of the lower layer (U = tanh(y - Ly/4)) is used on 0 <= y < Ly/2, clamped at the mid-plane y = Ly/2, and its mirror image phi(-eta) on the upper layer (U = -tanh(y - 3Ly/4)), which has the same eigenvalue. Both modes are normalized to max|phi| = 1 and scaled by amp.
    """
    Ly = solver.Ly
    Y = Ly / 4.0
    c, wi, yc, phi = orr_sommerfeld(alpha, Re, Y=Y, N=Nc, return_mode=True)
    order = np.argsort(yc)
    re = BarycentricInterpolator(yc[order], phi.real[order])
    im = BarycentricInterpolator(yc[order], phi.imag[order])
    y = solver.y
    eta = np.where(y < Ly / 2.0, y - Ly / 4.0, -(y - 3.0 * Ly / 4.0))
    eta = np.clip(eta, -Y, Y)
    prof = re(eta) + 1j * im(eta)
    X, _ = np.meshgrid(solver.x, y)
    psi = amp * np.real(prof[:, None] * np.exp(1j * (alpha * X + phase)))
    return psi, c, wi


def momentum_thickness(solver, wh, Ly):
    """Momentum thickness of the lower layer, theta = int_0^{Ly/2} (1 - Ubar(y)^2)/4 dy.

    Ubar is the x-averaged streamwise velocity; for U = tanh(y), theta = 1/2.
    """
    u, _ = solver.velocity(wh)
    U = u.mean(axis=1)
    y = solver.y
    sel = y < Ly / 2.0
    dy = y[1] - y[0]
    return np.sum((1.0 - U[sel] ** 2) / 4.0) * dy
