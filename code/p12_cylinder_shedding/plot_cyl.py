"""
plot_cyl.py
===========
Figures for blog post p12 from the saved data (run cyl_run.py and cyl_analyze.py first).

    python plot_cyl.py               all figures
    python plot_cyl.py convergence   one figure

Alongside every figure, the exact arrays it plots are written to figs/<figure>_plotdata.npz (read them back with pub_style.load_data("<figure>_plotdata", outdir="figs")), so a figure can be redrawn elsewhere without rerunning or re-analyzing anything.
"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from pub_style import apply_style, savefig_all, cm_both, cm_x, cm_y, subcaption, load_data, has_data, save_data, FIGDIR

apply_style()


def export(figname, meta=None, **arrays):
    """Save the arrays a figure plots next to the figure itself."""
    save_data(f"{figname}_plotdata", outdir=FIGDIR, meta=meta, **arrays)

# FeatFlow reference for DFG 2D-2, level 6, k = 1/200 (finest level in the published table)
REF = {"Cd_max": 3.2200, "Cd_min": 3.1569, "Cl_max": 0.9859, "Cl_min": -1.0206, "St": 0.30188}
# Unconfined cylinder at Re = 100: Qu, Norberg, Davidson et al. (2013), Table 1, case D9
QU = {"Cd": 1.319, "St": 0.1648, "Cl_rms": 0.225}


def summary():
    return {r["name"]: r for r in json.load(open("data/cyl_summary.json"))}


def best_st():
    """Finest channel run available, preferring the low-Mach (u0 = 0.02) runs."""
    for tag in ("cyl_st_N80_u002", "cyl_st_N60_u002", "cyl_st_N40_u002", "cyl_st_N60", "cyl_st_N40", "cyl_st_N20"):
        if has_data(tag):
            return tag
    return None


def field_panel(ax, d, k, extent_from_meta=True, vmax=5.0):
    m = d["meta"]
    N = m["N"]
    w = d["snap_w"][k]
    x0, y0 = m["crop_x0"], m["crop_y0"]
    # lattice node (i, j) sits at (i, j + 1/2); convert to diameters measured from the cylinder center
    st = m.get("stride", 1)
    nxw, nyw = w.shape
    ext = [(x0 - m["xc"]) / N, (x0 + st * (nxw - 1) - m["xc"]) / N, (y0 + 0.5 - m["yc"]) / N, (y0 + st * (nyw - 1) + 0.5 - m["yc"]) / N]
    im = ax.imshow(w.T, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal", interpolation="bilinear")
    ax.add_patch(plt.Circle((0, 0), 0.5, color="0.35", zorder=3))
    ax.grid(False)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    return im


def st_vorticity():
    d = load_data(best_st())
    k = len(d["snap_w"]) - 1                      # last stored instant
    fig, ax = plt.subplots(figsize=(11.0, 2.6))
    im = field_panel(ax, d, k)
    ax.set_xlabel("$(x - x_c)/D$")
    ax.set_ylabel("$(y - y_c)/D$")
    cm_both(ax)
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
    cb.set_label(r"$\omega D / U$")
    fig.tight_layout()
    savefig_all(fig, "cyl_st_vorticity_p12", close=True)
    export("cyl_st_vorticity_p12", meta={**d["meta"], "extent_D": list(im.get_extent()), "t": float(d["snap_t"][k])}, vorticity=d["snap_w"][k])


def st_forces():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    cols = {20: "#3EA44E", 40: "#009AFA", 60: "#FF0000", 80: "#000000"}
    series = {}
    for n in (20, 40, 60, 80):
        if not has_data(f"cyl_st_N{n}_u002"):
            continue
        d = load_data(f"cyl_st_N{n}_u002")
        t, Cd, Cl = d["t"], d["Cd"], d["Cl"]
        sel = t >= t[-1] - 1.2
        ts, cd, cl = t[sel], Cd[sel], Cl[sel]
        i0 = np.argmin(cl[: int(0.4 * len(cl))])        # align on a lift minimum
        tt = ts - ts[i0]
        keep = (tt >= 0) & (tt <= 0.66)
        axes[0].plot(tt[keep], cd[keep], "-", color=cols[n], lw=1.2, label=f"$N = {n}$")
        axes[1].plot(tt[keep], cl[keep], "-", color=cols[n], lw=1.2, label=f"$N = {n}$")
        series[f"t_N{n}"], series[f"Cd_N{n}"], series[f"Cl_N{n}"] = tt[keep], cd[keep], cl[keep]
    for key, ax in (("Cd", axes[0]), ("Cl", axes[1])):
        ax.axhline(REF[key + "_max"], color="k", ls="--", lw=0.8)
        ax.axhline(REF[key + "_min"], color="k", ls="--", lw=0.8, label="FeatFlow max/min")
        ax.set_xlim(0, 0.66)
        ax.set_xlabel(r"$t - t_0$ (s)")
        cm_both(ax)
        ax.legend(frameon=False, loc="lower right" if key == "Cd" else "upper right", ncol=2)
    axes[0].set_ylabel("$C_D$")
    axes[1].set_ylabel("$C_L$")
    axes[0].set_ylim(3.14, 3.29)
    axes[1].set_ylim(-1.15, 1.35)
    subcaption(axes[0], "(a) drag coefficient, two shedding cycles")
    subcaption(axes[1], "(b) lift coefficient, two shedding cycles")
    fig.tight_layout()
    savefig_all(fig, "cyl_st_forces_p12", close=True)
    export("cyl_st_forces_p12", meta={"reference": REF}, **series)


def convergence():
    S = summary()
    runs = {"$u_0 = 0.04$": ("", "o-", "#009AFA"), "$u_0 = 0.02$": ("_u002", "s-", "#FF0000")}
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))
    out = {}
    allN = sorted({n for n in (20, 40, 60, 80) for suf, _, _ in runs.values() if f"cyl_st_N{n}{suf}" in S})
    for ax, key, lab in zip(axes, ("Cd_max", "Cl_max", "St"), (r"$\max C_D$", r"$\max C_L$", "St")):
        vals = [REF[key]]
        for name, (suf, sty, col) in runs.items():
            Ns = [n for n in (20, 40, 60, 80) if f"cyl_st_N{n}{suf}" in S]
            v = [S[f"cyl_st_N{n}{suf}"][key] for n in Ns]
            vals += v
            ax.plot(Ns, v, sty, ms=6, color=col, mfc="none" if suf else col, label=name)
            out[f"N{suf or '_u004'}"] = np.array(Ns)
            out[f"{key}{suf or '_u004'}"] = np.array(v)
        ax.axhline(REF[key], color="k", ls="--", lw=0.9, label="FeatFlow (level 6)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(allN)
        ax.set_xticklabels([str(n) for n in allN])
        ax.minorticks_off()
        ax.set_xlim(allN[0] / 1.2, allN[-1] * 1.2)
        ax.set_xlabel("nodes per diameter $N$")
        ax.set_ylabel(lab)
        ax.legend(frameon=False)
        span = max(vals) - min(vals)
        ax.set_ylim(min(vals) - 0.15 * span, max(vals) + 0.25 * span)
        cm_y(ax)
    subcaption(axes[0], "(a) maximum drag")
    subcaption(axes[1], "(b) maximum lift")
    subcaption(axes[2], "(c) Strouhal number")
    fig.tight_layout()
    savefig_all(fig, "cyl_st_convergence_p12", close=True)
    export("cyl_st_convergence_p12", meta={"reference": REF}, **out)


def free():
    tag = next(t for t in ("cyl_free_N30_u002", "cyl_free_N20_u002", "cyl_free_N20") if has_data(t))
    d = load_data(tag)
    n = d["meta"]["N"]
    S = summary()[tag]
    fig = plt.figure(figsize=(12.0, 6.0))
    ax = fig.add_subplot(2, 1, 1)
    im = field_panel(ax, d, len(d["snap_w"]) - 1, vmax=3.0)
    ax.set_xlabel("$x/D$")
    ax.set_ylabel("$y/D$")
    cm_both(ax)
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
    cb.set_label(r"$\omega D / U$")
    ax.set_ylim(-3.5, 3.5)
    subcaption(ax, f"(a) vorticity, unconfined cylinder, $N = {n}$")

    t, Cd, Cl = d["t"], d["Cd"], d["Cl"]
    ax = fig.add_subplot(2, 2, 3)
    sel = t >= t[-1] - 30.0
    dts = t[1] - t[0]
    Y = np.fft.rfft(Cd[sel])
    Y[np.fft.rfftfreq(sel.sum(), dts) > 4.0 * S["St_spectral"]] = 0.0
    cd_lp = np.fft.irfft(Y, sel.sum())
    ax.plot(t[sel], Cd[sel], "-", color="skyblue", lw=0.6, label="$C_D$ (raw)")
    ax.plot(t[sel], cd_lp, "-", color="blue", lw=1.4, label="$C_D$ (low-passed)")
    ax.plot(t[sel], Cl[sel], "-", color="#FF0000", lw=1.2, label="$C_L$")
    ax.set_xlim(t[sel][0], t[sel][-1])
    ax.set_ylim(-0.6, 1.8)
    ax.set_xlabel("$tU/D$")
    ax.legend(frameon=False, ncol=3, loc="center right", fontsize=7)
    cm_both(ax)
    subcaption(ax, "(b) force coefficients")

    ax = fig.add_subplot(2, 2, 4)
    sel = t >= t[-1] - 120.0
    cl = Cl[sel] - Cl[sel].mean()
    dt = t[1] - t[0]
    spec = np.abs(np.fft.rfft(cl * np.hanning(cl.size))) ** 2
    fr = np.fft.rfftfreq(cl.size, dt)
    ax.semilogy(fr, spec / spec.max(), "-", color="k", lw=1.0)
    ax.axvline(S["St_spectral"], color="#009AFA", ls="--", lw=1.0, label=f"this run: St = {S['St_spectral']:.4f}")
    ax.axvline(QU["St"], color="#FF0000", ls=":", lw=1.2, label=f"Qu et al. (2013): {QU['St']}")
    ax.set_xlim(0, 0.6)
    ax.set_ylim(1e-8, 2)
    ax.set_xlabel("frequency $fD/U$")
    ax.set_ylabel("lift spectrum (normalized)")
    ax.legend(frameon=False)
    ax.grid(which="both")
    cm_x(ax)
    subcaption(ax, "(c) lift spectrum")
    fig.tight_layout()
    savefig_all(fig, "cyl_free_p12", close=True)
    win = t >= t[-1] - 30.0
    export("cyl_free_p12", meta={**d["meta"], "extent_D": list(im.get_extent()), "St_spectral": S["St_spectral"], "reference": QU}, vorticity=d["snap_w"][-1], t=t[win], Cd=Cd[win], Cd_lowpass=cd_lp, Cl=Cl[win], frequency=fr, lift_spectrum=spec / spec.max())


def animate():
    d = load_data(best_st())
    plt.rcParams["figure.dpi"] = 100
    fig, ax = plt.subplots(figsize=(8.8, 1.9))
    im = field_panel(ax, d, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    W = d["snap_w"]

    def update(k):
        im.set_data(W[k].T)
        return (im,)

    ani = animation.FuncAnimation(fig, update, frames=len(W), blit=True)
    ani.save("figs/cyl_st_p12.gif", writer=animation.PillowWriter(fps=15), dpi=100)
    plt.close(fig)
    print("    figs/cyl_st_p12.gif")


FIGURES = {"st_vorticity": st_vorticity, "st_forces": st_forces, "convergence": convergence, "free": free, "animation": animate}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(FIGURES):
        try:
            FIGURES[name]()
            print(f"  drew {name}")
        except (FileNotFoundError, KeyError) as e:
            print(f"  skipped {name}: missing data ({e})")
