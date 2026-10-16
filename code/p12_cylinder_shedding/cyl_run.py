"""
cyl_run.py
==========
Flow past a circular cylinder at Re = 100 with the lattice Boltzmann solver in lbm_core.py.

Two configurations:

  --case st    The Schaefer-Turek (1996) DFG benchmark 2D-2: a cylinder of diameter D = 0.1 at (0.2, 0.2) in a channel 2.2 x 0.41, parabolic inflow with mean velocity 1 (maximum 1.5), nu = 1e-3, so Re = U_mean D / nu = 100. The cylinder sits 0.005 below the channel axis, which breaks the symmetry and starts the shedding.

  --case free  An effectively unconfined cylinder: uniform inflow, free-slip side boundaries 40 D apart (blockage 2.5 percent), inlet 15 D upstream and outlet 30 D downstream. A short transverse velocity bump behind the cylinder starts the shedding.

Usage:
    python cyl_run.py --case st --N 40          (N = lattice nodes per diameter)
    python cyl_run.py --case free --N 40 --T 300

Output: data/cyl_<case>_N<N>.npz with the force and probe histories, the vorticity over the last shedding cycle, and the run parameters.
"""
import argparse
import time

import numpy as np

import lbm_core as L
from pub_style import save_data

RE = 100.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=["st", "free"], default="st")
    p.add_argument("--N", type=int, default=40, help="lattice nodes per cylinder diameter")
    p.add_argument("--u", type=float, default=0.04, help="mean inflow speed in lattice units")
    p.add_argument("--T", type=float, default=None, help="run length: seconds (st) or D/U units (free)")
    p.add_argument("--tag", default="")
    a = p.parse_args()

    N, u0 = a.N, a.u
    nu = u0 * N / RE
    omp, omm = L.trt_rates(nu)
    if a.case == "st":
        if (41 * N) % 10:
            raise SystemExit("N must make 4.1 N an integer (use a multiple of 10)")
        nx, ny = 22 * N + 1, 41 * N // 10
        xc, yc, R = 2.0 * N, 2.0 * N, 0.5 * N
        H = float(ny)
        yy = np.arange(ny) + 0.5
        profile = 6.0 * u0 * yy * (H - yy) / H**2          # mean u0, maximum 1.5 u0
        freeslip = False
        T = 20.0 if a.T is None else a.T                  # physical seconds
        t_unit = 0.1 / 1.0                                 # D / U_mean in seconds
        ramp = 0.5                                         # seconds
    else:
        nx, ny = 45 * N + 1, 40 * N
        xc, yc, R = 15.0 * N, 20.0 * N, 0.5 * N
        profile = u0 * np.ones(ny)
        freeslip = True
        T = 300.0 if a.T is None else a.T                 # convective units D / U
        t_unit = 1.0
        ramp = 0.0
    steps_per_unit = N / u0                                # lattice steps per D/U
    nsteps = int(round(T / t_unit * steps_per_unit))
    print(f"case {a.case}: grid {nx} x {ny}, N = {N}, u = {u0}, nu = {nu:.5f}, tau+ = {1 / omp:.4f}, Ma = {u0 * np.sqrt(3):.3f}", flush=True)

    solid, links = L.cylinder_geometry(nx, ny, xc, yc, R)
    lx, ly, ld, lq = links
    print(f"  {lx.size} boundary links, {solid.sum()} solid nodes", flush=True)

    rho = np.ones((nx, ny))
    ux = np.zeros((nx, ny))
    uy = np.zeros((nx, ny))
    if a.case == "free":
        ux[:] = u0
        X = np.arange(nx)[:, None]
        Y = np.arange(ny)[None, :] + 0.5
        uy += 0.1 * u0 * np.exp(-((X - xc - 2.0 * N) ** 2 + (Y - yc) ** 2) / (0.5 * N) ** 2)
        ux[solid] = 0.0
        uy[solid] = 0.0
    f = np.zeros((9, nx, ny))
    L.init_equilibrium(f, rho, ux, uy)
    fpost = f.copy()

    # probes for the benchmark pressure difference (front and rear stagnation points)
    jp = int(round(yc - 0.5))
    xa1, xa2 = int(round(xc - R)), int(round(xc + R))

    rec_every = max(1, int(steps_per_unit / 200))         # about 200 samples per D/U
    T_rec, CD, CL, DP = [], [], [], []
    # vorticity snapshots: channel, 8 per D/U over the last 8 D/U (enough for an animation);
    # unconfined, 4 per D/U over the last 4 D/U. Stored at no more than 40 nodes per diameter.
    per_unit, span = (8, 8.0) if a.case == "st" else (4, 4.0)
    snap_every = max(1, int(round(steps_per_unit / per_unit)))
    stride = max(1, N // 40)
    snaps_t, snaps_w = [], []
    snap_start = nsteps - int(round(span * steps_per_unit))
    if a.case == "st":
        crop = (slice(0, nx), slice(0, ny))
    else:
        crop = (slice(int(xc - 3 * N), int(xc + 25 * N)), slice(int(yc - 6 * N), int(yc + 6 * N)))

    meta = {"case": a.case, "N": N, "u0": u0, "nu": nu, "nx": nx, "ny": ny, "xc": xc, "yc": yc, "R": R, "T": T, "t_unit": t_unit, "Re": RE, "crop_x0": crop[0].start, "crop_y0": crop[1].start, "stride": stride, "freeslip": freeslip}
    tag = a.tag or f"cyl_{a.case}_N{N}"

    t0 = time.perf_counter()
    print(f"  {nsteps:,} steps ...", flush=True)
    for n in range(nsteps + 1):
        t_lat = n
        t_phys = t_lat / steps_per_unit * t_unit
        s = 1.0 if (ramp == 0.0 or t_phys >= ramp) else np.sin(0.5 * np.pi * t_phys / ramp) ** 2
        L.collide_trt(f, fpost, solid, omp, omm)
        L.stream(fpost, f, solid, freeslip)
        fx, fy = L.bouzidi(fpost, f, lx, ly, ld, lq)
        L.zou_he(f, s * profile, 1.0)
        if n % rec_every == 0:
            T_rec.append(t_phys)
            CD.append(2.0 * fx / (u0**2 * N))
            CL.append(2.0 * fy / (u0**2 * N))
            r1 = f[:, xa1, jp].sum() + f[:, xa1, jp + 1].sum()
            r2 = f[:, xa2, jp].sum() + f[:, xa2, jp + 1].sum()
            # pressure difference in physical units (rho = 1): (c_s^2 drho) (U / u0)^2 with U = 1
            DP.append(0.5 * (r1 - r2) / 3.0 / u0**2)
        if n >= snap_start and (n - snap_start) % snap_every == 0:
            rr = np.empty((nx, ny))
            vx = np.empty((nx, ny))
            vy = np.empty((nx, ny))
            L.macros(f, solid, rr, vx, vy)
            w = L.vorticity(vx, vy) * N / u0              # vorticity in units of U / D
            w[solid] = np.nan
            snaps_t.append(t_phys)
            snaps_w.append(w[crop][::stride, ::stride].astype(np.float32))
        if n % (20 * int(steps_per_unit)) == 0 and n > 0:
            el = time.perf_counter() - t0
            mlups = n * (nx * ny - solid.sum()) / el / 1e6
            print(f"    t = {t_phys:8.3f}  Cd = {CD[-1]:.4f}  Cl = {CL[-1]:+.4f}  ({el:.0f} s, {mlups:.0f} MLUPS)", flush=True)
            # checkpoint the force history, so an interrupted run (for example a cluster job that hits its wall-time limit) still leaves usable data
            save_data(f"{tag}_partial", t=np.array(T_rec), Cd=np.array(CD), Cl=np.array(CL), dp=np.array(DP), meta={**meta, "partial": True})

    save_data(tag, t=np.array(T_rec), Cd=np.array(CD), Cl=np.array(CL), dp=np.array(DP), snap_t=np.array(snaps_t), snap_w=np.array(snaps_w), meta=meta)
    import os
    from pub_style import DATADIR
    part = os.path.join(DATADIR, f"{tag}_partial.npz")
    if os.path.exists(part):
        os.remove(part)                                    # the complete file supersedes the checkpoint
    print(f"done in {time.perf_counter() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
