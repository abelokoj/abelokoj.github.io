"""
Lid-driven cavity flow solver (2D, incompressible Navier-Stokes)
Fractional-step / pressure-projection method, finite differences.

Re = U * L / nu   (defaults give Re = 100)

Dependencies: numpy, matplotlib
Run:  python cavity_flow.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Publication-style plot settings (LaTeX-like serif rendering without
# requiring a system LaTeX install).
# ---------------------------------------------------------------------
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = [
    "New Computer Modern", "Latin Modern Roman", "CMU Serif", "STIXGeneral",
]
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["axes.labelsize"] = 11
matplotlib.rcParams["axes.titlesize"] = 12
matplotlib.rcParams["legend.fontsize"] = 9
matplotlib.rcParams["xtick.labelsize"] = 9.5
matplotlib.rcParams["ytick.labelsize"] = 9.5
matplotlib.rcParams["axes.linewidth"] = 0.8
matplotlib.rcParams["figure.dpi"] = 150


def savefig_all(fig, basename, **kwargs):
    """Save a figure as both a high-resolution raster (PNG) and a vector
    (SVG) copy, matching the naming convention used across the post."""
    fig.savefig(f"{basename}.png", dpi=600, bbox_inches="tight", **kwargs)
    fig.savefig(f"{basename}.svg", bbox_inches="tight", **kwargs)


class CavityFlowSolver:
    """Finite-difference lid-driven cavity solver with pressure projection."""

    def __init__(self, nx=61, ny=61, nu=0.01, rho=1.0, dt=0.0008,
                 nit=60, lid_velocity=1.0):
        self.nx, self.ny = nx, ny
        self.dx = 1.0 / (nx - 1)
        self.dy = 1.0 / (ny - 1)
        self.x = np.linspace(0.0, 1.0, nx)
        self.y = np.linspace(0.0, 1.0, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.nu, self.rho, self.dt, self.nit = nu, rho, dt, nit
        self.lid_velocity = lid_velocity
        self.Re = lid_velocity * 1.0 / nu

        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))
        self.residuals = []

    def _poisson_rhs(self, u, v):
        dx, dy, dt, rho = self.dx, self.dy, self.dt, self.rho
        b = np.zeros_like(u)
        b[1:-1, 1:-1] = rho * (
            (1.0 / dt) * (
                (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)
                + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)
            )
            - ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx)) ** 2
            - 2 * (
                (u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy)
                * (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)
            )
            - ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) ** 2
        )
        return b

    def _solve_pressure(self, b):
        dx, dy = self.dx, self.dy
        p = self.p
        for _ in range(self.nit):
            pn = p.copy()
            p[1:-1, 1:-1] = (
                ((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy ** 2
                 + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx ** 2)
                / (2 * (dx ** 2 + dy ** 2))
                - (dx ** 2 * dy ** 2) / (2 * (dx ** 2 + dy ** 2)) * b[1:-1, 1:-1]
            )
            p[:, -1] = p[:, -2]   # dp/dx = 0 at x = 1
            p[:, 0] = p[:, 1]     # dp/dx = 0 at x = 0
            p[0, :] = p[1, :]     # dp/dy = 0 at y = 0
            p[-1, :] = 0.0        # reference pressure at lid
        self.p = p
        return p

    def _apply_bcs(self):
        u, v, U = self.u, self.v, self.lid_velocity
        u[0, :] = 0.0
        u[:, 0] = 0.0
        u[:, -1] = 0.0
        u[-1, :] = U
        v[0, :] = 0.0
        v[-1, :] = 0.0
        v[:, 0] = 0.0
        v[:, -1] = 0.0

    def step(self):
        dx, dy, dt, rho, nu = self.dx, self.dy, self.dt, self.rho, self.nu
        un, vn = self.u.copy(), self.v.copy()

        b = self._poisson_rhs(un, vn)
        p = self._solve_pressure(b)

        self.u[1:-1, 1:-1] = (
            un[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt / dx * (un[1:-1, 1:-1] - un[1:-1, 0:-2])
            - vn[1:-1, 1:-1] * dt / dy * (un[1:-1, 1:-1] - un[0:-2, 1:-1])
            - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2])
            + nu * (
                dt / dx ** 2 * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
                + dt / dy ** 2 * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])
            )
        )
        self.v[1:-1, 1:-1] = (
            vn[1:-1, 1:-1]
            - un[1:-1, 1:-1] * dt / dx * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2])
            - vn[1:-1, 1:-1] * dt / dy * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1])
            - dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1])
            + nu * (
                dt / dx ** 2 * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2])
                + dt / dy ** 2 * (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])
            )
        )
        self._apply_bcs()
        self.residuals.append(np.max(np.abs(self.u - un)))

    def run(self, nt=700):
        for _ in range(nt):
            self.step()
        return self.u, self.v, self.p

    def centerline_profile(self):
        mid = self.nx // 2
        return self.y, self.u[:, mid]

    def verify(self):
        """Sanity checks that should always hold for this benchmark."""
        checks = {
            "lid_velocity_enforced": np.isclose(self.u[-1, self.nx // 2], self.lid_velocity),
            "walls_no_slip": np.allclose(self.u[1:-1, 0], 0) and np.allclose(self.u[1:-1, -1], 0),
            "bounded_velocity": np.max(np.abs(self.u)) <= self.lid_velocity + 1e-6,
            "recirculation_present": np.min(self.centerline_profile()[1]) < -0.02,
        }
        return checks


if __name__ == "__main__":
    solver = CavityFlowSolver(nx=61, ny=61, nu=0.01, dt=0.0008, nit=60)
    print(f"Reynolds number: {solver.Re:.1f}")

    solver.run(nt=700)

    checks = solver.verify()
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    assert all(checks.values()), "Verification failed -- inspect parameters."

    y, u_centerline = solver.centerline_profile()
    print(f"max|u| = {np.max(np.abs(solver.u)):.6f}")
    print(f"min u(x=0.5) = {np.min(u_centerline):.6f}  (recirculation strength)")
    print(f"final step residual = {solver.residuals[-1]:.3e}")

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))

    speed = np.sqrt(solver.u**2 + solver.v**2)
    cf = axes[0].contourf(solver.X, solver.Y, speed, levels=25, cmap="turbo")
    fig.colorbar(cf, ax=axes[0], label=r"$|\mathbf{u}|$")
    axes[0].streamplot(solver.X, solver.Y, solver.u, solver.v,
                        color="white", linewidth=0.6, density=1.2)
    axes[0].set_title(rf"Velocity field, $\mathrm{{Re}} = {solver.Re:.0f}$")
    axes[0].set_xlabel(r"$x$"); axes[0].set_ylabel(r"$y$"); axes[0].set_aspect("equal")

    axes[1].plot(u_centerline, y, "b-", lw=1.8)
    axes[1].axvline(0, color="gray", lw=0.7, ls="--")
    axes[1].set_xlabel(r"$u$ velocity"); axes[1].set_ylabel(r"$y$")
    axes[1].set_title(r"Centerline $u$-profile ($x = 0.5$)")
    axes[1].grid(alpha=0.25, lw=0.6)

    axes[2].semilogy(solver.residuals, "r-", lw=1.2)
    axes[2].set_xlabel("time step"); axes[2].set_ylabel(r"max $|\Delta u|$ (log scale)")
    axes[2].set_title("Convergence history")
    axes[2].grid(alpha=0.25, lw=0.6)

    fig.tight_layout()
    savefig_all(fig, "cavity_flow_v2_p10")
    print("done")
