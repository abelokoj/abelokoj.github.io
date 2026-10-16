"""
lbm_core.py
===========
A D2Q9 lattice Boltzmann solver for incompressible flow past a circular cylinder (blog post p12), compiled with Numba.

Ingredients:
- two-relaxation-time (TRT) collision with the "magic" parameter Lambda = 3/16, which places a bounce-back wall exactly halfway between nodes and keeps the low-viscosity runs stable;
- Bouzidi-Firdaouss-Lallemand (2001) linearly interpolated bounce-back on the curved cylinder surface, using the exact wall distance q of every boundary link;
- force on the cylinder by momentum exchange over the boundary links;
- Zou-He (1997) velocity inlet and constant-density (pressure) outlet;
- halfway bounce-back (no-slip) or specular reflection (free-slip) on the side walls.

Lattice units throughout: dx = dt = 1, reference density 1, sound speed c_s = 1/sqrt(3). Node (i, j) sits at x = i, y = j + 1/2, so a halfway wall lies at y = 0 and at y = ny.
"""
import numpy as np
from numba import njit, prange

# D2Q9 lattice --------------------------------------------------------
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int64)
W = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
MIRY = np.array([0, 1, 4, 3, 2, 8, 7, 6, 5], dtype=np.int64)   # reflection y -> -y


@njit(inline="always")
def _feq(i, rho, ux, uy):
    cu = CX[i] * ux + CY[i] * uy
    return W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (ux * ux + uy * uy))


@njit(parallel=True, cache=True)
def init_equilibrium(f, rho, ux, uy):
    nx, ny = rho.shape
    for x in prange(nx):
        for y in range(ny):
            for i in range(9):
                f[i, x, y] = _feq(i, rho[x, y], ux[x, y], uy[x, y])


@njit(parallel=True, cache=True)
def collide_trt(f, fpost, solid, omp, omm):
    """TRT collision: symmetric part relaxes with omp, antisymmetric with omm.

    The equilibrium is evaluated once per direction per node; pairs (i, OPP[i]) are updated together.
    """
    nx, ny = solid.shape
    for x in prange(nx):
        fe = np.empty(9)
        for y in range(ny):
            if solid[x, y]:
                continue
            rho = 0.0
            mx = 0.0
            my = 0.0
            for i in range(9):
                fi = f[i, x, y]
                rho += fi
                mx += CX[i] * fi
                my += CY[i] * fi
            ux = mx / rho
            uy = my / rho
            usq = 1.5 * (ux * ux + uy * uy)
            for i in range(9):
                cu = CX[i] * ux + CY[i] * uy
                fe[i] = W[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - usq)
            fpost[0, x, y] = f[0, x, y] - omp * (f[0, x, y] - fe[0])
            for i in (1, 2, 5, 6):
                j = OPP[i]
                fi = f[i, x, y]
                fj = f[j, x, y]
                dp = 0.5 * (fi + fj - fe[i] - fe[j])
                dm = 0.5 * (fi - fj - fe[i] + fe[j])
                fpost[i, x, y] = fi - omp * dp - omm * dm
                fpost[j, x, y] = fj - omp * dp + omm * dm


@njit(parallel=True, cache=True)
def stream(fpost, f, solid, freeslip):
    """Pull streaming. Side walls: halfway bounce-back, or specular reflection if freeslip.

    Populations whose source node is solid are left for the Bouzidi step; populations entering through the inlet or outlet are left for the Zou-He step.
    """
    nx, ny = solid.shape
    for xp in prange(nx):
        x = np.int64(xp)
        for y in range(ny):
            if solid[x, y]:
                continue
            for i in range(9):
                xs = x - CX[i]
                ys = y - CY[i]
                if ys < 0 or ys >= ny:
                    if freeslip:
                        m = MIRY[i]
                        xm = np.int64(xs)
                        if xm < 0 or xm >= nx:
                            xm = np.int64(x)
                        f[i, x, y] = fpost[m, xm, y]
                    else:
                        f[i, x, y] = fpost[OPP[i], x, y]
                    continue
                if xs < 0 or xs >= nx:
                    continue
                if solid[xs, ys]:
                    continue
                f[i, x, y] = fpost[i, xs, ys]


@njit(cache=True)
def bouzidi(fpost, f, lx, ly, ld, lq):
    """Interpolated bounce-back on the cylinder and momentum-exchange force.

    Each link (lx, ly, ld, lq): fluid node (lx, ly), direction ld pointing into the solid, wall at fraction lq of the link. The unknown population is the one in direction OPP[ld].
    """
    fx = 0.0
    fy = 0.0
    for n in range(lx.shape[0]):
        x = lx[n]
        y = ly[n]
        d = ld[n]
        q = lq[n]
        o = OPP[d]
        if q < 0.5:
            val = 2.0 * q * fpost[d, x, y] + (1.0 - 2.0 * q) * fpost[d, x - CX[d], y - CY[d]]
        else:
            val = fpost[d, x, y] / (2.0 * q) + (2.0 * q - 1.0) / (2.0 * q) * fpost[o, x, y]
        f[o, x, y] = val
        fx += CX[d] * (fpost[d, x, y] + val)
        fy += CY[d] * (fpost[d, x, y] + val)
    return fx, fy


@njit(cache=True)
def zou_he(f, uin, rho_out):
    """Velocity inlet at x = 0 (u = (uin[y], 0)) and density outlet at x = nx-1 (rho = rho_out)."""
    _, nx, ny = f.shape
    for y in range(ny):
        # inlet: unknown f1, f5, f8
        ux = uin[y]
        rho = (f[0, 0, y] + f[2, 0, y] + f[4, 0, y] + 2.0 * (f[3, 0, y] + f[6, 0, y] + f[7, 0, y])) / (1.0 - ux)
        f[1, 0, y] = f[3, 0, y] + 2.0 / 3.0 * rho * ux
        f[5, 0, y] = f[7, 0, y] - 0.5 * (f[2, 0, y] - f[4, 0, y]) + rho * ux / 6.0
        f[8, 0, y] = f[6, 0, y] + 0.5 * (f[2, 0, y] - f[4, 0, y]) + rho * ux / 6.0
        # outlet: unknown f3, f6, f7
        e = nx - 1
        uo = -1.0 + (f[0, e, y] + f[2, e, y] + f[4, e, y] + 2.0 * (f[1, e, y] + f[5, e, y] + f[8, e, y])) / rho_out
        f[3, e, y] = f[1, e, y] - 2.0 / 3.0 * rho_out * uo
        f[7, e, y] = f[5, e, y] + 0.5 * (f[2, e, y] - f[4, e, y]) - rho_out * uo / 6.0
        f[6, e, y] = f[8, e, y] - 0.5 * (f[2, e, y] - f[4, e, y]) - rho_out * uo / 6.0


@njit(parallel=True, cache=True)
def macros(f, solid, rho, ux, uy):
    nx, ny = solid.shape
    for x in prange(nx):
        for y in range(ny):
            if solid[x, y]:
                rho[x, y] = 1.0
                ux[x, y] = 0.0
                uy[x, y] = 0.0
                continue
            r = 0.0
            mx = 0.0
            my = 0.0
            for i in range(9):
                fi = f[i, x, y]
                r += fi
                mx += CX[i] * fi
                my += CY[i] * fi
            rho[x, y] = r
            ux[x, y] = mx / r
            uy[x, y] = my / r


def cylinder_geometry(nx, ny, xc, yc, R):
    """Solid mask and Bouzidi boundary links for a cylinder of radius R centered at (xc, yc).

    Node (i, j) is at (i, j + 1/2). For every fluid node and every direction whose neighbor is solid, the wall fraction q solves |p + q c - center| = R with 0 < q <= 1.
    """
    X = np.arange(nx)[:, None] * np.ones((1, ny))
    Y = np.ones((nx, 1)) * (np.arange(ny)[None, :] + 0.5)
    solid = (X - xc) ** 2 + (Y - yc) ** 2 <= R**2
    lx, ly, ld, lq = [], [], [], []
    xs, ys = np.where(solid)
    i0, i1 = max(xs.min() - 2, 0), min(xs.max() + 3, nx)
    j0, j1 = max(ys.min() - 2, 0), min(ys.max() + 3, ny)
    for i in range(i0, i1):
        for j in range(j0, j1):
            if solid[i, j]:
                continue
            px, py = i - xc, j + 0.5 - yc
            for d in range(1, 9):
                ni, nj = i + CX[d], j + CY[d]
                if not solid[ni, nj]:
                    continue
                cx, cy = CX[d], CY[d]
                a = cx * cx + cy * cy
                b = 2.0 * (px * cx + py * cy)
                c = px * px + py * py - R * R
                disc = b * b - 4.0 * a * c
                q = (-b - np.sqrt(disc)) / (2.0 * a)
                if not (0.0 < q <= 1.0 + 1e-12):
                    raise RuntimeError("bad wall fraction")
                if solid[i - cx, j - cy]:
                    raise RuntimeError("cylinder too coarse: second fluid node missing")
                lx.append(i)
                ly.append(j)
                ld.append(d)
                lq.append(min(q, 1.0))
    return solid, (np.array(lx), np.array(ly), np.array(ld), np.array(lq))


def trt_rates(nu, magic=3.0 / 16.0):
    """Relaxation rates for kinematic viscosity nu and the TRT magic parameter."""
    tau_p = 3.0 * nu + 0.5
    tau_m = magic / (tau_p - 0.5) + 0.5
    return 1.0 / tau_p, 1.0 / tau_m


def vorticity(ux, uy):
    """Second-order central-difference vorticity on the lattice (dx = 1)."""
    w = np.zeros_like(ux)
    w[1:-1, 1:-1] = 0.5 * (uy[2:, 1:-1] - uy[:-2, 1:-1]) - 0.5 * (ux[1:-1, 2:] - ux[1:-1, :-2])
    return w
