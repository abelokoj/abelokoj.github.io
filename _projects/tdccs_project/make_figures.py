"""
make_figures.py
===============
Draw every figure in the project from the data files written by the
example scripts. This is the ONLY file that plots.

Why the split
-------------
The solves in this project are expensive, because the stable time step
scales like dx^3 and the finest runs need hundreds of thousands of
Runge-Kutta steps. Recomputing a solution merely to change a colour, a
legend position or an axis label wastes hours. So the example scripts
compute and save; this script reads and draws.

Usage
-----
    python example_7_1_linear_kdv.py        # writes data/figure5..., etc.
    python example_7_2_soliton.py
    ...
    python make_figures.py                  # draws everything available

    python make_figures.py figure7 figure13 # draw only these

Figures whose data files are absent are reported and skipped, so this can
be run at any point while the longer solves are still going.

Figures 2, 3 and 4 are NOT produced here. Those depend only on the
coefficient tables and cost seconds to recompute, so `fig_fourier_analysis.py`
and `fig_stability.py` still plot directly.

Paper conventions
-----------------
Three conventions are matched deliberately, since the source figures are
MATLAB output: surfaces use the `turbo` colormap (errors in Figure 13 use `cool`), each 3-D figure uses the camera angle fitted to the paper's panel, with white walls, the z-axis on the left and a tall colorbar hugging the box, and panel labels carry the "(a) t = 0" form. All of these live in `pub_style.py`.
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")          # write files, never open a window
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

from figdata import load_data, has_data
from pub_style import (apply_style, savefig_all, FIGDIR,
                       SURFACE_CMAP, ERROR_CMAP, VIEW_FIG9, VIEW_FIG10, VIEW_FIG13,
                       VIEW_FIG14_U, VIEW_FIG14_V, VIEW_FIG15_U, VIEW_FIG15_V,
                       FIG13_CMAP, FIG13_ALPHA, matlab_zlim, matlab_view, matlab_axes3d, matlab_colorbar,
                       subcaption)

apply_style()
OUT = FIGDIR

# Colours used for successive times, following the paper: black, blue,
# green, red, magenta.
TIME_COLORS = ["k", "b", "r", "g", "m"]


# ======================================================================
# Small helpers shared by several figures
# ======================================================================
def _line_panel(ax, x, y, color, label=None, lw=1.1, style="-"):
    """One curve on a 2-D panel, with the axis clamped to the data range."""
    ax.plot(x, y, style, color=color, lw=lw, label=label)
    ax.set_xlim(float(x.min()), float(x.max()))


def _surface_panel(fig, ax, x, times, Z, zlabel, view=None, zlim=None):
    """A space-time surface drawn the way the paper draws them.

    x runs along one horizontal axis, time along the other, and the
    solution is the height. The colormap and camera come from
    pub_style so that every surface in the project matches.

    shade=False because MATLAB's surf applies no lighting unless a light is added; matplotlib's default shading darkens every slope and made the flat floors of the v rows look near-black. `zlim` is "nice" (MATLAB auto limits, the default), "tight", or an explicit pair.
    """
    Tg, Xg = np.meshgrid(times, x, indexing="ij")
    surf = ax.plot_surface(Xg, Tg, Z, cmap=SURFACE_CMAP, linewidth=0,
                           antialiased=False, rstride=1, cstride=1,
                           shade=False, vmin=float(Z.min()),
                           vmax=float(Z.max()))
    matlab_view(ax, **(view or {}))
    if zlim is not None:
        if isinstance(zlim, str):
            zlim = matlab_zlim(float(Z.min()), float(Z.max()),
                               tight=(zlim == "tight"))
        ax.set_zlim(*zlim)
    matlab_axes3d(ax)
    ax.set_xlabel("$x$", labelpad=-2)
    ax.set_ylabel("$t$", labelpad=-2)
    if -90.0 < ax.azim < 0.0:
        # z-axis was moved to the left edge; matplotlib would still put its label on the right beside the colorbar, so place it by hand.
        ax.text2D((view or {}).get("zlabel_x", -0.08), 0.5, zlabel,
                  transform=ax.transAxes, rotation=90,
                  ha="center", va="center", fontsize=11)
    else:
        ax.set_zlabel(zlabel, labelpad=0)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_ylim(float(times.min()), float(times.max()))
    matlab_colorbar(fig, surf, ax, pad=(view or {}).get("cbpad") or 0.02)
    return surf


def _solution_and_error(basename, title_stub, ylabel="$u(x,t)$"):
    """Draw the two-by-two solution-and-error layout.

    Used by Figures 5, 6, 7 and 8, which share an identical arrangement:
    the top row holds the numerical solution (open circles) over the
    exact solution (solid line) for each scheme, and the bottom row holds
    the pointwise error at the same times.

    The two error axes are scaled independently, exactly as in the paper,
    so the numbers on the axis matter rather than the curve heights.
    """
    d = load_data(basename)
    x = d["x"]
    times = d["meta"]["times"]
    colors = TIME_COLORS[:len(times)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, scheme in enumerate(("TDCNCS", "TDCCS")):
        snaps = d[f"u_{scheme}"]
        for t, cc in zip(times, colors):
            tf = float(t)
            u = snaps[tf]
            ex = d["exact"][tf]
            # Top: exact as a line, numerical as open circles on top.
            axes[0, col].plot(x, ex, "-", color=cc, lw=1.2)
            axes[0, col].plot(x, u, "o", ms=5, mfc="none", color=cc)
            # Bottom: the gap between them.
            axes[1, col].plot(x, np.abs(ex - u), color=cc, lw=1.0)

        for row, what in ((0, "Numerical solution"), (1, "Pointwise error")):
            axes[row, col].set_xlabel("$x$")
            axes[row, col].set_xlim(float(x.min()), float(x.max()))
            axes[row, col].set_ylabel(ylabel if row == 0 else "Error")
            tag = "abcd"[row * 2 + col]
            subcaption(axes[row, col], f"({tag}) {scheme} - {what}")

    fig.tight_layout()
    savefig_all(fig, basename, outdir=OUT)
    plt.close(fig)
    print(f"  {basename}  ({title_stub})")


# ======================================================================
# Figures 5 and 6: Example 7.1, linear KdV
# ======================================================================
def figure5():
    _solution_and_error("figure5_example71_c1", "Example 7.1, c = 1")


def figure6():
    _solution_and_error("figure6_example71_c8", "Example 7.1, c = 8")


# ======================================================================
# Figure 7: Example 7.2, the classical soliton
# ======================================================================
def figure7():
    _solution_and_error("figure7_soliton_example72", "Example 7.2")


# ======================================================================
# Figure 8: Example 7.3, single soliton
# ======================================================================
def figure8():
    _solution_and_error("figure8_soliton_illustrative",
                        "Example 7.3, single soliton")


# ======================================================================
# Figure 9: Example 7.3, double soliton collision
# ======================================================================
def figure9():
    """Two rows by four columns: three snapshots then a surface, per scheme."""
    d = load_data("figure9_double_soliton")
    x = d["x"]
    snap_times = d["meta"]["snap_times"]
    surf_times = d["surf_times"]

    fig = plt.figure(figsize=(15, 6.4))
    labels = "abcdefgh"
    idx = 0
    for row, scheme in enumerate(("TDCNCS", "TDCCS")):
        snaps = d[f"u_{scheme}"]
        color = "k" if scheme == "TDCNCS" else "tab:blue"

        for t in snap_times:
            ax = fig.add_subplot(2, 4, row * 4 + idx % 4 + 1)
            _line_panel(ax, x, snaps[float(t)], color, label=scheme)
            ax.set_xlabel("$x$"); ax.set_ylabel("$u(x,t)$")
            ax.set_ylim(-0.1, 1.0)
            ax.legend(loc="upper right", fontsize=8, framealpha=1.0)
            subcaption(ax, f"({labels[idx]}) $t = {t:g}$")
            idx += 1

        ax = fig.add_subplot(2, 4, row * 4 + 4, projection="3d")
        Z = np.array([snaps[float(t)] for t in surf_times])
        _surface_panel(fig, ax, x, surf_times, Z, "$u(x,t)$",
                       view=VIEW_FIG9)
        subcaption(ax, f"({labels[idx]}) $t = {surf_times.max():g}$")
        idx += 1

    fig.tight_layout()
    savefig_all(fig, "figure9_double_soliton", outdir=OUT)
    plt.close(fig)
    print("  figure9_double_soliton  (Example 7.3, collision)")


# ======================================================================
# Figure 10: Example 7.3, triple splitting with and without the filter
# ======================================================================
def figure10():
    """Three rows: TDCNCS snapshots, TDCCS snapshots, then two surfaces.

    Each line panel overlays the unfiltered run (solid) with the filtered
    run (dotted red), so the filter's effect reads as the gap between
    them. The surfaces show the unfiltered runs.
    """
    d = load_data("figure10_filter_effect")
    x = d["x"]
    snap_times = d["meta"]["snap_times"]
    surf_times = d["surf_times"]

    fig = plt.figure(figsize=(14, 12))
    labels = "abcdefgh"
    idx = 0
    for row, scheme in enumerate(("TDCNCS", "TDCCS")):
        color = "k" if scheme == "TDCNCS" else "tab:blue"
        raw = d[f"u_{scheme}_raw"]
        f12 = d[f"u_{scheme}_f12"]
        for j, t in enumerate(snap_times):
            ax = fig.add_subplot(3, 3, row * 3 + j + 1)
            _line_panel(ax, x, raw[float(t)], color, label=scheme)
            ax.plot(x, f12[float(t)], ":", color="r", lw=1.1,
                    label=f"{scheme}-F12")
            ax.set_xlabel("$x$"); ax.set_ylabel("$u(x,t)$")
            ax.set_ylim(-0.1, 1.0)
            ax.legend(loc="upper left", fontsize=7.5, framealpha=1.0)
            subcaption(ax, f"({labels[idx]}) $t = {t:g}$")
            idx += 1

    for col, scheme in enumerate(("TDCNCS", "TDCCS")):
        raw = d[f"u_{scheme}_raw"]
        ax = fig.add_subplot(3, 2, 5 + col, projection="3d")
        Z = np.array([raw[float(t)] for t in surf_times])
        _surface_panel(fig, ax, x, surf_times, Z, "$u(x,t)$",
                       view=VIEW_FIG10)
        subcaption(ax, f"({labels[6 + col]}) $t = {surf_times.max():g}$ "
                       f"({scheme})")

    fig.tight_layout()
    savefig_all(fig, "figure10_filter_effect", outdir=OUT)
    plt.close(fig)
    print("  figure10_filter_effect  (Example 7.3, splitting)")


# ======================================================================
# Figure 11: Example 7.4, zero-dispersion sweep
# ======================================================================
def figure11():
    """Two rows by three columns, with panels assigned COLUMN-wise.

    Column 1 holds the two mildest cases with both schemes and the
    refined reference on one axis. Columns 2 and 3 hold the two finest
    cases, one scheme per panel, because the oscillations there are far
    too dense for overlaid curves to be read.
    """
    d = load_data("figure11_zero_dispersion_illustrative")
    cases = d["meta"]["cases"]
    xlo, xhi = d["meta"]["xlo"], d["meta"]["xhi"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    with_ref = [c for c in cases if c["with_ref"]]
    fine = [c for c in cases if not c["with_ref"]]

    for k, c in enumerate(with_ref):
        tag, ax = c["tag"], axes[k, 0]
        x = d[f"x_{tag}"]
        ax.plot(x, d[f"u_TDCCS_{tag}"], "-", color="tab:blue", lw=1.0,
                label="TDCCS")
        ax.plot(x, d[f"u_TDCNCS_{tag}"], "k-", lw=1.0, label="TDCNCS")
        ax.plot(d[f"x_ref_{tag}"], d[f"u_ref_{tag}"], "r:", lw=1.2,
                label="TDCNCS-Ref")
        ax.set_xlabel("$x$"); ax.set_ylabel("$u(x,t)$")
        ax.set_xlim(xlo, xhi); ax.set_ylim(1.4, 3.4)
        ax.legend(loc="upper right", fontsize=7.5, framealpha=1.0)
        expo = int(np.log10(c["eps"]))
        subcaption(ax, f"({'ab'[k]}) $\\epsilon = 10^{{{expo}}}$, "
                       f"$N = {c['N']}$")

    panel = [("c", "e"), ("d", "f")]
    for k, c in enumerate(fine):
        tag = c["tag"]; x = d[f"x_{tag}"]
        expo = int(np.log10(c["eps"]))
        for j, (scheme, color) in enumerate((("TDCNCS", "k"),
                                             ("TDCCS", "tab:blue"))):
            ax = axes[k, 1 + j]
            ax.plot(x, d[f"u_{scheme}_{tag}"], "-", color=color, lw=0.7,
                    label=scheme)
            ax.set_xlabel("$x$"); ax.set_ylabel("$u(x,t)$")
            ax.set_xlim(xlo, xhi); ax.set_ylim(1.4, 3.4)
            ax.legend(loc="upper right", fontsize=7.5, framealpha=1.0)
            subcaption(ax, f"({panel[j][k]}) $\\epsilon = 10^{{{expo}}}$, "
                           f"$N = {c['N']}$")

    fig.tight_layout()
    savefig_all(fig, "figure11_zero_dispersion_illustrative", outdir=OUT)
    plt.close(fig)
    print("  figure11_zero_dispersion_illustrative  (Example 7.4)")


# ======================================================================
# Figure 12: Example 7.4, top-hat breakup
# ======================================================================
def figure12():
    """Two rows by four columns: unfiltered and filtered at each time."""
    d = load_data("figure12_tophat_illustrative")
    x = d["x"]
    times = d["meta"]["times"]
    xlo, xhi = d["meta"]["xlo"], d["meta"]["xhi"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    labels = "abcdefgh"
    idx = 0
    for row, scheme in enumerate(("TDCNCS", "TDCCS")):
        color = "k" if scheme == "TDCNCS" else "tab:blue"
        col = 0
        for t in times:
            for tag, lab in (("raw", scheme), ("f12", f"{scheme}-F12")):
                ax = axes[row, col]
                ax.plot(x, d[f"u_{scheme}_{tag}"][float(t)], color=color,
                        lw=0.8, label=lab)
                ax.set_xlabel("$x$"); ax.set_ylabel("$u(x,t)$")
                ax.set_xlim(xlo, xhi); ax.set_ylim(-0.4, 1.4)
                ax.legend(loc="lower right", fontsize=7, framealpha=1.0)
                subcaption(ax, f"({labels[idx]}) $t = {t:g}$")
                idx += 1
                col += 1

    fig.tight_layout()
    savefig_all(fig, "figure12_tophat_illustrative", outdir=OUT)
    plt.close(fig)
    print("  figure12_tophat_illustrative  (Example 7.4)")


# ======================================================================
# Figure 13: Example 7.5, two-dimensional surfaces
# ======================================================================
def figure13():
    """Two by two: solution surfaces on top, error surfaces below.

    The diagonal ridges in the error panels are not a defect. The exact
    solution depends on x and y only through x + y, so it is constant
    along lines of constant x + y, and the error inherits that structure
    from the way such a wave meets a square grid.
    """
    d = load_data("figure13_example75")
    X, Y = d["X"], d["Y"]
    L = d["meta"]["L_domain"]

    fig = plt.figure(figsize=(12, 9))
    panels = [
        (1, d["u_TDCNCS"], "(a) TDCNCS", "$u(x,y)$", False),
        (2, d["u_TDCCS"], "(b) TDCCS", "$u(x,y)$", False),
        (3, d["err_TDCNCS"], "(c) TDCNCS", r"$|u_{exact}-u_{num}|$", True),
        (4, d["err_TDCCS"], "(d) TDCCS", r"$|u_{exact}-u_{num}|$", True),
    ]
    for pos, Z, tag, zlab, is_err in panels:
        ax = fig.add_subplot(2, 2, pos, projection="3d")
        # Axes3D reserves a bounding box much larger than the surface actually
        # fills, which is why the panel label ends up floating well above the
        # plot. zoom enlarges the drawing within that box and closes the gap.
        ax.set_box_aspect(VIEW_FIG13["aspect"], zoom=1.15)
        # Draw the z-axis on the LEFT of the box instead of the right, so it
        # does not sit next to the colorbar. This swaps which pair of vertical
        # planes matplotlib attaches the z-axis to; the camera is unchanged,
        # so the surface itself looks exactly the same.
        # _p = ax.zaxis._PLANES
        # ax.zaxis._PLANES = (_p[2], _p[3], _p[0], _p[1], _p[4], _p[5])
        # The paper uses two colormaps in this figure: the solution surfaces
        # are blue-to-red, while the error surfaces are cyan-to-magenta.
        # MATLAB's "cool" is the latter, and matplotlib's is identical.
        # Figure 13 is an RGB image in the PDF, so its solution surfaces use true turbo (not the print-muted PAPER_TURBO), drawn semi-transparent as in the paper.
        cmap = ERROR_CMAP if is_err else FIG13_CMAP
        alpha = 1.0 if is_err else FIG13_ALPHA

        surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0,
                               antialiased=False, rstride=1, cstride=1,
                               shade=False, alpha=alpha,
                               vmin=float(Z.min()), vmax=float(Z.max()))

        ax.view_init(elev=VIEW_FIG13["elev"], azim=VIEW_FIG13["azim"])
        ax.set_proj_type("ortho")
        if not is_err:
            ax.set_zlim(-1, 1)
        matlab_axes3d(ax)
        ax.set_xlabel("$x$", labelpad=-2); ax.set_ylabel("$y$", labelpad=-2)
        ax.set_xlim(0, L); ax.set_ylim(0, L)
        # With the z-axis moved to the left edge, matplotlib still anchors its label and its "x10^n" offset to the right edge, on top of the colorbar. Both are placed by hand on the left instead, where the paper has them.
        ax.set_zlabel("")
        ax.text2D(-0.15, 0.5, zlab, transform=ax.transAxes, rotation=90,
                  ha="center", va="center", fontsize=11)
        if is_err:
            expo = int(np.floor(np.log10(np.abs(Z).max())))
            ax.zaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _p, s=10.0 ** expo: f"{v / s:g}"))
            ax.text2D(0.0, 0.80, f"$\\times10^{{{expo}}}$",
                      transform=ax.transAxes, ha="left", va="bottom",
                      fontsize=8)
        # Built from an opaque mappable so the bar stays solid even though the surface is translucent, as in the paper.
        sm = matplotlib.cm.ScalarMappable(norm=surf.norm, cmap=surf.cmap)
        cb = matlab_colorbar(fig, sm, ax)
        if is_err:
            # MATLAB prints the shared power of ten above the bar.
            cb.formatter.set_powerlimits((0, 0))
            cb.formatter.set_useMathText(True)
            cb.update_ticks()
        subcaption(ax, tag)

    fig.tight_layout()
    savefig_all(fig, "figure13_example75", outdir=OUT)
    plt.close(fig)
    plt.show()
    print("  figure13_example75  (Example 7.5)")


# ======================================================================
# Figures 14 and 15: Example 7.6, the Ito system
# ======================================================================
def _ito_figure(basename, stub, views, zlims):
    """Four rows by four columns.

    Rows, from the top: TDCNCS u, TDCNCS v, TDCCS u, TDCCS v. The first
    three columns are snapshots, the fourth a space-time surface.
    Stacking u directly above v for each scheme is what makes the central
    contrast legible: only the u equation carries a third derivative, so
    the u rows stay smooth while the v rows steepen into shock-like fronts.
    """
    d = load_data(basename)
    x = d["x"]
    snap_times = d["meta"]["snap_times"]
    surf_times = d["surf_times"]
    xlo, xhi = d["meta"]["xlo"], d["meta"]["xhi"]

    fig = plt.figure(figsize=(16, 13))
    rows = [("TDCNCS", "u", "$u(x,t)$", "k"),
            ("TDCNCS", "v", "$v(x,t)$", "k"),
            ("TDCCS", "u", "$u(x,t)$", "tab:blue"),
            ("TDCCS", "v", "$v(x,t)$", "tab:blue")]
    labels = "abcdefghijklmnop"
    idx = 0

    for r, (scheme, comp, ylab, color) in enumerate(rows):
        snaps = d[f"{comp}_{scheme}"]
        # A common y range per row, so the three snapshots are comparable.
        vals = np.concatenate([snaps[float(t)] for t in snap_times])
        lo, hi = float(vals.min()), float(vals.max())
        pad = 0.08 * (hi - lo if hi > lo else 1.0)

        for t in snap_times:
            ax = fig.add_subplot(4, 4, r * 4 + idx % 4 + 1)
            _line_panel(ax, x, snaps[float(t)], color, label=scheme, lw=1.0)
            ax.set_xlabel("$x$"); ax.set_ylabel(ylab)
            ax.set_xlim(xlo, xhi); ax.set_ylim(lo - pad, hi + pad)
            ax.legend(loc="upper left", fontsize=7, framealpha=1.0)
            subcaption(ax, f"({labels[idx]}) $t = {t:g}$")
            idx += 1

        ax = fig.add_subplot(4, 4, r * 4 + 4, projection="3d")
        Z = np.array([snaps[float(t)] for t in surf_times])
        _surface_panel(fig, ax, x, surf_times, Z, ylab, view=views[comp],
                       zlim=zlims[comp])
        subcaption(ax, f"({labels[idx]}) $t = {surf_times.max():g}$")
        idx += 1

    fig.tight_layout()
    savefig_all(fig, basename, outdir=OUT)
    plt.close(fig)
    print(f"  {basename}  ({stub})")


def figure14():
    _ito_figure("figure14_ito_trig", "Example 7.6, trigonometric",
                views={"u": VIEW_FIG14_U, "v": VIEW_FIG14_V},
                zlims={"u": "nice", "v": "nice"})


def figure15():
    # The paper's u panels here are drawn with `axis tight`; the v panels keep MATLAB's automatic limits ([-0.5, 2]).
    _ito_figure("figure15_ito_gaussian", "Example 7.6, Gaussian",
                views={"u": VIEW_FIG15_U, "v": VIEW_FIG15_V},
                zlims={"u": "tight", "v": "nice"})


# ======================================================================
# Driver
# ======================================================================
#: Maps a short name to (data file it needs, function that draws it).
FIGURES = {
    # "figure5": ("figure5_example71_c1", figure5),
    # "figure6": ("figure6_example71_c8", figure6),
    # "figure7": ("figure7_soliton_example72", figure7),
    # "figure8": ("figure8_soliton_illustrative", figure8),
    "figure9": ("figure9_double_soliton", figure9),
    "figure10": ("figure10_filter_effect", figure10),
    # "figure11": ("figure11_zero_dispersion_illustrative", figure11),
    # "figure12": ("figure12_tophat_illustrative", figure12),
    "figure13": ("figure13_example75", figure13),
    "figure14": ("figure14_ito_trig", figure14),
    "figure15": ("figure15_ito_gaussian", figure15),
}


def main(names=None):
    """Draw every requested figure whose data file exists.

    Missing data files are reported rather than raising, so this can be
    run while the longer solves are still in progress.
    """
    names = list(FIGURES) if not names else names
    drawn, missing = 0, []

    print("Drawing figures from saved data:\n")
    for name in names:
        if name not in FIGURES:
            print(f"  unknown figure {name!r}; known names: "
                  f"{', '.join(FIGURES)}")
            continue
        datafile, fn = FIGURES[name]
        if not has_data(datafile):
            missing.append((name, datafile))
            continue
        fn()
        drawn += 1

    print(f"\n{drawn} figure(s) written to {OUT}")
    if missing:
        print("\nSkipped, data file not found (run the matching example "
              "script first):")
        for name, datafile in missing:
            print(f"  {name:9s} needs data/{datafile}.npz")


if __name__ == "__main__":
    main(sys.argv[1:])
