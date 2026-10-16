"""
plot_kh.py
==========
Figures for blog post p10 from the saved data (no simulation is rerun).

    python plot_kh.py                 all figures
    python plot_kh.py dispersion      one figure

Figures are written to figs/ in PNG, PDF and SVG; the animation is written to figs/kh_rollup_p10.gif.

Alongside every figure, the exact arrays it plots are written to figs/<figure>_plotdata.npz (read them back with pub_style.load_data("<figure>_plotdata", outdir="figs")), so a figure can be redrawn in another tool, or combined with other results, without touching the simulation data.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from pub_style import apply_style, savefig_all, cm_both, cm_x, subcaption, load_data, has_data, save_data, FIGDIR

apply_style()


def export(figname, meta=None, **arrays):
    """Save the arrays a figure plots next to the figure itself."""
    save_data(f"{figname}_plotdata", outdir=FIGDIR, meta=meta, **arrays)


def dispersion():
    d = load_data("kh_dispersion")
    m = d["meta"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    ax = axes[0]
    keep = d["alphas"] <= 0.9          # near alpha = 1 both modes become neutral and the eigenvalue solves lose accuracy
    ax.plot(d["alphas"][keep], d["w_ray"][keep], "k-", lw=1.2, label="Rayleigh (inviscid)")
    ax.plot(d["alphas"][keep], d["w_os"][keep], "-", color="#009AFA", lw=1.2, label=f"Orr-Sommerfeld, Re = {m['Re']:.0f}")
    ax.plot(d["a_dns"], d["w_dns"], "o", ms=6, mfc="none", mec="#FF0000", mew=1.2, label="simulation")
    ax.plot([m["a_ray"]], [m["wmax_ray"]], "k+", ms=9)
    ax.set_xlim(0.0, 0.9)
    ax.set_ylim(0.0, 0.2)
    ax.set_xlabel(r"wavenumber $\alpha$")
    ax.set_ylabel(r"growth rate $\omega_i$")
    ax.legend(frameon=False, loc="lower center")
    cm_both(ax)
    subcaption(ax, "(a) dispersion relation")

    ax = axes[1]
    t = d["t_hist"]
    keys = sorted(d["A_hist"])
    show = [keys[0], keys[3], keys[-1]]
    for kk, col in zip(show, ["#3EA44E", "#009AFA", "mediumvioletred"]):
        A = d["A_hist"][kk]
        ax.semilogy(t, A / A[0], "-", color=col, lw=1.2, label=rf"$\alpha = {kk:.3f}$")
        i = list(d["a_dns"]).index(kk)
        ax.semilogy(t, np.exp(d["w_os_pts"][i] * t), "k--", lw=0.8)
    ax.set_xlim(0, t.max())
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"mode amplitude $A(t)/A(0)$")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(which="both")
    cm_x(ax)
    subcaption(ax, r"(b) growth from the eigenmode; dashed: $e^{\omega_i t}$")
    fig.tight_layout()
    savefig_all(fig, "kh_dispersion_p10", close=True)
    export("kh_dispersion_p10", meta=m, alpha_curve=d["alphas"][keep], w_rayleigh=d["w_ray"][keep], w_orr_sommerfeld=d["w_os"][keep], alpha_sim=d["a_dns"], w_sim=d["w_dns"], w_theory_at_sim=d["w_os_pts"], t=t, alpha_shown=np.array(show), A_over_A0=np.array([d["A_hist"][kk] / d["A_hist"][kk][0] for kk in show]))


def _crop(d):
    m = d["meta"]
    y = d["y"]
    yc = m["Ly"] / 4.0
    sel = np.abs(y - yc) <= 0.42 * m["Lx"]
    return sel, y[sel] - yc


def snapshots(field="c"):
    d = load_data("kh_rollup_main")
    m = d["meta"]
    sel, yy = _crop(d)
    x = d["x"]
    times = [20.0, 40.0, 60.0, 80.0, 100.0, 140.0]
    fig, axes = plt.subplots(3, 2, figsize=(10.0, 9.4), sharex=True, sharey=True)
    for ax, t in zip(axes.flat, times):
        F = d[field][t][sel]
        if field == "c":
            im = ax.imshow(F, origin="lower", extent=[0, m["Lx"], yy[0], yy[-1]], cmap="jet", vmin=0, vmax=1, aspect="equal", interpolation="bilinear")
        else:
            im = ax.imshow(F, origin="lower", extent=[0, m["Lx"], yy[0], yy[-1]], cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal", interpolation="bilinear")
        ax.grid(False)
        ax.set_xlim(0, m["Lx"])
        subcaption(ax, f"$t = {t:.0f}$")
        cm_both(ax)
    for ax in axes[-1]:
        ax.set_xlabel("$x$")
    for ax in axes[:, 0]:
        ax.set_ylabel("$y$")
    fig.tight_layout()
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("passive scalar $c$" if field == "c" else r"vorticity $\omega$")
    name = f"kh_snapshots_{'scalar' if field == 'c' else 'vorticity'}_p10"
    savefig_all(fig, name, close=True)
    export(name, meta={"Lx": m["Lx"], "times": times, "field": field}, x=x, y=yy, fields={t: d[field][t][sel] for t in times})


def diagnostics():
    d = load_data("kh_rollup_main")
    m = d["meta"]
    dc = load_data("kh_rollup_coarse") if has_data("kh_rollup_coarse") else None
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    t = d["t"]
    ax = axes[0]
    ax.semilogy(t, d["A1"], "-", color="#009AFA", lw=1.2, label=r"fundamental $\alpha$")
    ax.semilogy(t, d["A2"], "-", color="#FF0000", lw=1.2, label=r"subharmonic $\alpha/2$")
    tt = np.linspace(0, 60, 50)
    ax.semilogy(tt, d["A1"][0] * np.exp(m["wi1"] * tt), "k--", lw=0.8, label="linear theory")
    tt2 = np.linspace(0, 110, 50)
    ax.semilogy(tt2, d["A2"][0] * np.exp(m["wi2"] * tt2), "k:", lw=0.9)
    ax.set_xlim(0, t.max())
    ax.set_ylim(1e-5, 1.0)
    ax.set_xlabel("time $t$")
    ax.set_ylabel("mode amplitude")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(which="both")
    cm_x(ax)
    subcaption(ax, "(a) modal amplitudes")

    ax = axes[1]
    ax.plot(t, d["theta"] / d["theta"][0], "-", color="#009AFA", lw=1.2, label=f"{m['nx']} x {m['ny']}")
    if dc is not None:
        ax.plot(dc["t"], dc["theta"] / dc["theta"][0], "--", color="k", lw=1.0, label=f"{dc['meta']['nx']} x {dc['meta']['ny']}")
    ax.set_xlim(0, t.max())
    ax.set_ylim(0.9, None)
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"momentum thickness $\theta/\theta_0$")
    ax.legend(frameon=False, loc="upper left")
    cm_both(ax)
    subcaption(ax, "(b) layer growth")

    ax = axes[2]
    ax.semilogy(t[2:-2], d["budget_resid"][2:-2], "-", color="#3EA44E", lw=1.0)
    ax.set_xlim(0, t.max())
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"$|dE/dt + 2\nu Z| \,/\, 2\nu Z$")
    ax.grid(which="both")
    cm_x(ax)
    subcaption(ax, "(c) energy budget residual")
    fig.tight_layout()
    savefig_all(fig, "kh_diagnostics_p10", close=True)
    extra = {}
    if dc is not None:
        extra = {"t_coarse": dc["t"], "theta_over_theta0_coarse": dc["theta"] / dc["theta"][0]}
    export("kh_diagnostics_p10", meta=m, t=t, A_fundamental=d["A1"], A_subharmonic=d["A2"], t_theory_fund=tt, A_theory_fund=d["A1"][0] * np.exp(m["wi1"] * tt), t_theory_sub=tt2, A_theory_sub=d["A2"][0] * np.exp(m["wi2"] * tt2), theta_over_theta0=d["theta"] / d["theta"][0], budget_residual=d["budget_resid"], **extra)


def animate():
    d = load_data("kh_frames_main")
    m = d["meta"]
    y = d["y"]
    yc = m["Ly"] / 4.0
    sel = np.abs(y - yc) <= 0.42 * m["Lx"]
    frames = d["c"][:, sel, :]
    plt.rcParams["figure.dpi"] = 100
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    im = ax.imshow(frames[0], origin="lower", extent=[0, m["Lx"], y[sel][0] - yc, y[sel][-1] - yc], cmap="jet", vmin=0, vmax=1, aspect="equal", interpolation="bilinear")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    txt = ax.text(0.01, 0.95, "", transform=ax.transAxes, color="w", va="top", fontsize=9)
    fig.tight_layout()

    def update(k):
        im.set_data(frames[k])
        txt.set_text(f"t = {d['t'][k]:.0f}")
        return im, txt

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=True)
    ani.save("figs/kh_rollup_p10.gif", writer=animation.PillowWriter(fps=12), dpi=100)
    plt.close(fig)
    print("    figs/kh_rollup_p10.gif")
    # the frames themselves are already stored in data/kh_frames_main.npz; nothing new to export here


FIGURES = {"dispersion": ("kh_dispersion", dispersion), "snapshots": ("kh_rollup_main", lambda: (snapshots("c"), snapshots("w"))), "diagnostics": ("kh_rollup_main", diagnostics), "animation": ("kh_frames_main", animate)}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGURES)
    missing = []
    for n in names:
        datafile, fn = FIGURES[n]
        if not has_data(datafile):
            missing.append(n)
            continue
        fn()
        print(f"  drew {n}")
    if missing:
        print("missing data for:", ", ".join(missing))
