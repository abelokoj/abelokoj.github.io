"""
cyl_analyze.py
==============
Benchmark quantities from the force histories written by cyl_run.py.

For each run, the last `--window` time units are split into shedding cycles at successive minima of the lift coefficient (the cycle definition of the DFG benchmark). Over the last full cycles it reports min, max, mean = (max + min)/2 and amplitude = max - min of Cd and Cl, the time-averaged Cd, the r.m.s. lift fluctuation, the shedding period and the Strouhal number St = f D / U, and the cycle-to-cycle spread of these numbers as a check that the flow is periodic.

Usage: python cyl_analyze.py data/cyl_st_N40.npz [more files ...]
Output: a printed table and data/cyl_summary.json
"""
import json
import os
import sys

import numpy as np

from pub_style import load_data


def refine_min(t, y, i):
    """Parabolic refinement of a discrete minimum at index i."""
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = y0 - 2.0 * y1 + y2
    s = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    h = t[i + 1] - t[i]
    return t[i] + s * h, y1 - 0.25 * (y0 - y2) * s


def refine_ext(t, y, i):
    return refine_min(t, y, i)


def analyze(path, window=None):
    name = os.path.splitext(os.path.basename(path))[0]
    d = load_data(name, outdir=os.path.dirname(path) or None)
    m = d["meta"]
    t, Cd, Cl, dp = d["t"], d["Cd"], d["Cl"], d["dp"]
    # time in units of D/U for the Strouhal number
    tc = t / m["t_unit"]
    if window is None:
        window = 5.0 if m["case"] == "st" else 60.0
    sel = t >= t[-1] - window
    ts, cd, cl, pp = t[sel], Cd[sel], Cl[sel], dp[sel]
    tcs = tc[sel]
    # lift minima
    idx = np.where((cl[1:-1] < cl[:-2]) & (cl[1:-1] <= cl[2:]))[0] + 1
    tmins = np.array([refine_min(tcs, cl, i)[0] for i in idx])
    periods = np.diff(tmins)
    cycles = []
    for a, b in zip(idx[:-1], idx[1:]):
        c = slice(a, b + 1)
        cycles.append({"Cd_min": cd[c].min(), "Cd_max": cd[c].max(), "Cl_min": cl[c].min(), "Cl_max": cl[c].max(), "dp_min": pp[c].min(), "dp_max": pp[c].max()})
    last = cycles[-3:]
    res = {"name": name, "case": m["case"], "N": m["N"], "u0": m["u0"], "ncycles": len(cycles)}
    for key in ("Cd_min", "Cd_max", "Cl_min", "Cl_max", "dp_min", "dp_max"):
        vals = np.array([c[key] for c in last])
        res[key] = float(vals.mean())
        res[key + "_spread"] = float(vals.max() - vals.min())
    res["Cd_mean_benchmark"] = 0.5 * (res["Cd_max"] + res["Cd_min"])
    res["Cd_amp"] = res["Cd_max"] - res["Cd_min"]
    res["Cl_mean_benchmark"] = 0.5 * (res["Cl_max"] + res["Cl_min"])
    res["Cl_amp"] = res["Cl_max"] - res["Cl_min"]
    # time averages over whole cycles
    a, b = idx[-1 - min(5, len(idx) - 1)], idx[-1]
    res["Cd_timeavg"] = float(np.mean(cd[a:b]))
    res["Cl_timeavg"] = float(np.mean(cl[a:b]))
    res["Cl_rms"] = float(np.sqrt(np.mean((cl[a:b] - np.mean(cl[a:b])) ** 2)))
    P = float(np.mean(periods[-5:]))
    res["period_DU"] = P
    res["period_spread"] = float(periods[-5:].max() - periods[-5:].min())
    res["St"] = 1.0 / P            # time already in D/U units
    if m["case"] == "st":
        res["period_s"] = P * m["t_unit"]
    # spectral estimate of the shedding frequency: peak of the lift spectrum (zero-padded, Hann window,
    # parabolic refinement). This is insensitive to the small high-frequency (acoustic) noise that the
    # weakly compressible method superposes on the forces, which corrupts minimum-to-minimum cycle detection
    # in the unconfined box.
    dt = tcs[1] - tcs[0]
    x = (cl - cl.mean()) * np.hanning(cl.size)
    nfft = 16 * cl.size
    S = np.abs(np.fft.rfft(x, nfft)) ** 2
    fr = np.fft.rfftfreq(nfft, dt)
    k = int(np.argmax(S[1:]) + 1)
    a, b, c = np.log(S[k - 1]), np.log(S[k]), np.log(S[k + 1])
    res["St_spectral"] = float(fr[k] + 0.5 * (a - c) / (a - 2 * b + c) * (fr[1] - fr[0]))
    # low-passed forces (cut-off at twice the shedding frequency for lift, four times for drag)
    def lowpass(y, fc):
        Y = np.fft.rfft(y)
        f = np.fft.rfftfreq(y.size, dt)
        Y[f > fc] = 0.0
        return np.fft.irfft(Y, y.size)
    St = res["St_spectral"]
    ncyc = int((tcs[-1] - tcs[0]) * St)
    n_int = int(round(ncyc / St / dt))           # whole number of shedding periods
    cl_f = lowpass(cl, 2.0 * St)[-n_int:]
    cd_f = lowpass(cd, 4.0 * St)[-n_int:]
    res["Cd_timeavg_whole_periods"] = float(np.mean(cd[-n_int:]))
    res["Cl_rms_lowpass"] = float(np.sqrt(np.mean((cl_f - cl_f.mean()) ** 2)))
    res["Cl_amp_lowpass"] = float(0.5 * (cl_f.max() - cl_f.min()))
    res["Cd_amp_lowpass"] = float(0.5 * (cd_f.max() - cd_f.min()))
    res["Cd_noise_rms"] = float(np.sqrt(np.mean((cd[-n_int:] - lowpass(cd, 4.0 * St)[-n_int:]) ** 2)))
    return res


def main():
    files = sys.argv[1:]
    out = []
    for f in files:
        r = analyze(f)
        out.append(r)
        print(f"{r['name']}: N={r['N']} u0={r['u0']}  cycles={r['ncycles']}")
        print(f"   Cd min/max/mean/amp = {r['Cd_min']:.4f} {r['Cd_max']:.4f} {r['Cd_mean_benchmark']:.4f} {r['Cd_amp']:.4f}   (time avg {r['Cd_timeavg']:.4f})")
        print(f"   Cl min/max/mean/amp = {r['Cl_min']:.4f} {r['Cl_max']:.4f} {r['Cl_mean_benchmark']:.4f} {r['Cl_amp']:.4f}   (rms {r['Cl_rms']:.4f})")
        print(f"   St = {r['St']:.4f}  period spread {r['period_spread']:.2e}  Cd_max spread {r['Cd_max_spread']:.1e}  dp min/max {r['dp_min']:.4f} {r['dp_max']:.4f}")
        print(f"   spectral St = {r['St_spectral']:.4f}  Cd avg (whole periods) = {r['Cd_timeavg_whole_periods']:.4f}  Cl rms (low-passed) = {r['Cl_rms_lowpass']:.4f}  Cl amp = {r['Cl_amp_lowpass']:.4f}  Cd amp (low-passed) = {r['Cd_amp_lowpass']:.4f}  Cd noise rms = {r['Cd_noise_rms']:.4f}")
    path = os.path.join("data", "cyl_summary.json")
    old = []
    if os.path.exists(path):
        old = [o for o in json.load(open(path)) if o["name"] not in {r["name"] for r in out}]
    json.dump(old + out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
