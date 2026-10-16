"""
kh_rollup.py
============
Nonlinear stage of the Kelvin-Helmholtz instability: roll-up, saturation and vortex pairing in a temporally evolving shear layer.

The box holds two wavelengths of the most unstable Orr-Sommerfeld mode (Re = 1000), so the fundamental can roll up into two vortices and its subharmonic can then merge them. The run is unforced: the mean profile is free to spread.

Usage:
    python kh_rollup.py --nx 512 --ny 1024 --tag main      (about 20-30 minutes on a laptop)
    python kh_rollup.py --nx 256 --ny 512 --tag coarse     (resolution check)

Output: data/kh_rollup_<tag>.npz (time series and snapshots) and, for the main run, data/kh_frames_<tag>.npz (downsampled frames for an animation).
"""
import argparse
import time

import numpy as np
from scipy.optimize import minimize_scalar

import kh_core as k
from pub_style import save_data

RE = 1000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=512)
    p.add_argument("--ny", type=int, default=1024)
    p.add_argument("--T", type=float, default=200.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--amp1", type=float, default=1e-3, help="fundamental amplitude (max |psi'|)")
    p.add_argument("--amp2", type=float, default=1e-4, help="subharmonic amplitude (max |psi'|)")
    p.add_argument("--phase", type=float, default=0.0, help="subharmonic phase, radians")
    p.add_argument("--tag", default="main")
    p.add_argument("--frames", action="store_true", help="also store animation frames every 1.0")
    a = p.parse_args()

    # most unstable wavenumber at this Reynolds number (Y = 16, as in kh_linear.py)
    r = minimize_scalar(lambda al: -k.orr_sommerfeld(al, RE, Y=16.0, N=240)[1], bounds=(0.3, 0.6), method="bounded", options={"xatol": 1e-7})
    alpha = r.x
    Lx = 2.0 * (2.0 * np.pi / alpha)        # two fundamental wavelengths
    Ly = 2.0 * Lx
    s = k.SpectralNS2D(a.nx, a.ny, Lx, Ly, nu=1.0 / RE)
    print(f"alpha = {alpha:.4f}, Lx = {Lx:.3f}, Ly = {Ly:.3f}, grid {a.nx} x {a.ny}, dx = {Lx / a.nx:.4f}", flush=True)

    # base flow, eigenmode perturbations and passive scalar
    U = k.double_shear_layer(s.y, Ly)
    w0 = k.double_shear_vorticity(s.y, Ly)[:, None] * np.ones((1, a.nx))
    s.Ubar = np.mean(U)
    psi1, c1, wi1 = k.eigenmode_streamfunction(s, alpha, RE, a.amp1)
    psi2, c2, wi2 = k.eigenmode_streamfunction(s, alpha / 2.0, RE, a.amp2, phase=a.phase)
    print(f"fundamental omega_i = {wi1:.5f}, subharmonic omega_i = {wi2:.5f}", flush=True)
    wh = s.fft(w0) + s.K2 * s.fft(psi1 + psi2)
    ch = s.fft(0.5 * (1.0 + U)[:, None] * np.ones((1, a.nx)))

    nsteps = int(round(a.T / a.dt))
    every = 5
    rec = {key: [] for key in ("t", "E", "Z", "A1", "A2", "theta", "umax")}
    snap_times = [0.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 200.0]
    snaps_w, snaps_c = {}, {}
    frames_w, frames_c, frame_t = [], [], []

    def record(n, wh, ch):
        t = n * a.dt
        E, Z = s.energy_enstrophy(wh)
        u, v = s.velocity(wh)
        rec["t"].append(t)
        rec["E"].append(E)
        rec["Z"].append(Z)
        rec["A1"].append(s.mode_amplitude(wh, 2))    # fundamental: two wavelengths per box
        rec["A2"].append(s.mode_amplitude(wh, 1))    # subharmonic
        rec["theta"].append(k.momentum_thickness(s, wh, Ly))
        rec["umax"].append(np.sqrt(u**2 + v**2).max())

    t0 = time.perf_counter()
    print(f"  {nsteps:,} RK4 steps of dt = {a.dt} ...", flush=True)
    for n in range(nsteps + 1):
        t = n * a.dt
        if n % every == 0:
            record(n, wh, ch)
        for ts in snap_times:
            if abs(t - ts) < 0.5 * a.dt:
                snaps_w[ts] = s.ifft(wh).astype(np.float32)
                snaps_c[ts] = s.ifft(ch).astype(np.float32)
        if a.frames and abs(t - round(t)) < 0.5 * a.dt:
            frames_w.append(s.ifft(wh)[::2, ::2].astype(np.float32))
            frames_c.append(s.ifft(ch)[::2, ::2].astype(np.float32))
            frame_t.append(round(t))
        if n % 1000 == 0:
            el = time.perf_counter() - t0
            print(f"    t = {t:7.2f}  E = {rec['E'][-1]:.6f}  theta = {rec['theta'][-1]:.4f}  umax = {rec['umax'][-1]:.3f}  ({el:.0f} s)", flush=True)
            # checkpoint the time series, so an interrupted run still leaves usable data
            save_data(f"kh_rollup_{a.tag}_partial", **{k: np.array(v) for k, v in rec.items()}, meta={"partial": True, "Lx": Lx, "Ly": Ly, "nx": a.nx, "ny": a.ny})
        if n < nsteps:
            wh, ch = s.step(wh, ch, a.dt)

    rec = {key: np.array(v) for key, v in rec.items()}
    # energy budget: dE/dt = -2 nu Z for unforced periodic 2-D flow
    dEdt = np.gradient(rec["E"], rec["t"])
    resid = np.abs(dEdt + 2.0 * s.nu * rec["Z"]) / (2.0 * s.nu * rec["Z"])
    print(f"energy budget: max relative residual |dE/dt + 2 nu Z| / (2 nu Z) = {resid[2:-2].max():.2e}", flush=True)

    save_data(f"kh_rollup_{a.tag}", x=s.x, y=s.y, w=snaps_w, c=snaps_c, budget_resid=resid, **rec, meta={"Re": RE, "alpha": alpha, "Lx": Lx, "Ly": Ly, "nx": a.nx, "ny": a.ny, "dt": a.dt, "T": a.T, "amp1": a.amp1, "amp2": a.amp2, "phase": a.phase, "wi1": wi1, "wi2": wi2})
    if a.frames:
        save_data(f"kh_frames_{a.tag}", t=np.array(frame_t), w=np.array(frames_w), c=np.array(frames_c), x=s.x[::2], y=s.y[::2], meta={"Lx": Lx, "Ly": Ly})
    import os
    from pub_style import DATADIR
    part = os.path.join(DATADIR, f"kh_rollup_{a.tag}_partial.npz")
    if os.path.exists(part):
        os.remove(part)                                    # the complete file supersedes the checkpoint
    print(f"done in {time.perf_counter() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
