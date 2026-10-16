"""
kh_linear.py
============
Linear stage of the Kelvin-Helmholtz instability: theory against simulation.

1. Dispersion relation omega_i(alpha) of the tanh shear layer from the inviscid Rayleigh equation (whole line) and from the viscous Orr-Sommerfeld equation at Re = 1000 (truncated at |y| = 16).
2. Growth rates measured from the nonlinear pseudo-spectral solver, started from the Orr-Sommerfeld eigenmode at amplitude 1e-6, with a body force that holds the base profile fixed so the comparison with temporal theory is exact in principle.
3. A resolution check of the measured growth rate at the most unstable wavenumber.

Run:  python kh_linear.py        (about two minutes on a laptop)
Output: data/kh_dispersion.npz and a printed table.
"""
import time

import numpy as np
from scipy.optimize import minimize_scalar

import kh_core as k
from pub_style import save_data

RE = 1000.0
LY = 64.0          # four layer half-widths of clearance: Y = Ly/4 = 16
NC = 240           # Chebyshev resolution of the Orr-Sommerfeld solve


def measure_growth(alpha, ny=512, nx=32, amp=1e-6, dt=0.05, T=40.0, fit=(10.0, 40.0)):
    """Growth rate of streamwise mode 1 from a forced-base-flow simulation."""
    Lx = 2.0 * np.pi / alpha
    s = k.SpectralNS2D(nx, ny, Lx, LY, nu=1.0 / RE)
    w0 = k.double_shear_vorticity(s.y, LY)[:, None] * np.ones((1, nx))
    w0h = s.fft(w0)
    s.Ubar = np.mean(k.double_shear_layer(s.y, LY))
    s.forcing_hat = s.nu * s.K2 * w0h          # keeps the base flow steady
    psi, c, wi_os = k.eigenmode_streamfunction(s, alpha, RE, amp, Nc=NC)
    wh = w0h + s.K2 * s.fft(psi)               # omega' = -Lap(psi')
    nsteps = int(round(T / dt))
    t = np.zeros(nsteps + 1)
    A = np.zeros(nsteps + 1)
    A[0] = s.mode_amplitude(wh, 1)
    for n in range(nsteps):
        wh, _ = s.step(wh, None, dt)
        t[n + 1] = (n + 1) * dt
        A[n + 1] = s.mode_amplitude(wh, 1)
    sel = (t >= fit[0]) & (t <= fit[1])
    slope = np.polyfit(t[sel], np.log(A[sel]), 1)[0]
    return slope, wi_os, t, A


def main():
    t0 = time.perf_counter()
    # --- theory curves -------------------------------------------------
    alphas = np.linspace(0.02, 0.98, 49)
    w_ray = np.array([k.rayleigh_growth(a, N=200)[1] for a in alphas])
    w_os = np.array([k.orr_sommerfeld(a, RE, Y=LY / 4.0, N=NC)[1] for a in alphas])
    r = minimize_scalar(lambda a: -k.rayleigh_growth(a, N=200)[1], bounds=(0.3, 0.6), method="bounded", options={"xatol": 1e-7})
    a_ray, wmax_ray = r.x, -r.fun
    r = minimize_scalar(lambda a: -k.orr_sommerfeld(a, RE, Y=LY / 4.0, N=NC)[1], bounds=(0.3, 0.6), method="bounded", options={"xatol": 1e-7})
    a_os, wmax_os = r.x, -r.fun
    print(f"Rayleigh (inviscid): max omega_i = {wmax_ray:.5f} at alpha = {a_ray:.4f}")
    print(f"Orr-Sommerfeld Re={RE:.0f}: max omega_i = {wmax_os:.5f} at alpha = {a_os:.4f}", flush=True)

    # --- simulations ---------------------------------------------------
    a_dns = np.array([0.15, 0.25, 0.35, a_os, 0.55, 0.65, 0.75, 0.85])
    w_dns, w_os_pts = [], []
    hist = {}
    print(f"{'alpha':>8} {'OS':>10} {'DNS':>10} {'rel.diff':>10}")
    for a in a_dns:
        g, wi, t, A = measure_growth(a)
        w_dns.append(g)
        w_os_pts.append(wi)
        hist[float(a)] = A
        print(f"{a:8.4f} {wi:10.6f} {g:10.6f} {abs(g - wi) / wi:10.2e}", flush=True)

    # --- resolution check at the most unstable wavenumber ---------------
    res = []
    for ny in (256, 512, 1024):
        for dt in (0.1, 0.05):
            g, wi, _, _ = measure_growth(a_os, ny=ny, dt=dt)
            res.append((ny, dt, g))
            print(f"  resolution ny={ny:5d} dt={dt:5.3f}: omega_i = {g:.6f} (OS {wi:.6f})", flush=True)

    save_data("kh_dispersion", alphas=alphas, w_ray=w_ray, w_os=w_os, a_dns=a_dns, w_dns=np.array(w_dns), w_os_pts=np.array(w_os_pts), t_hist=t, A_hist=hist, res=np.array(res), meta={"Re": RE, "Ly": LY, "Nc": NC, "a_ray": a_ray, "wmax_ray": wmax_ray, "a_os": a_os, "wmax_os": wmax_os})
    print(f"done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
