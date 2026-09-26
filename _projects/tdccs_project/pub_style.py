"""
pub_style.py
============
Shared publication-quality matplotlib style for every figure in this
project, plus the Computer-Modern tick-label formatter and a single
multi-format figure saver. Import and call `apply_style()` once at the
top of any plotting script, then save every figure with `savefig_all`.

Every figure is written in three formats:
  .png  600 dpi raster, for web embedding
  .pdf  vector, for LaTeX inclusion
  .svg  vector, for web embedding and further editing
"""
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator
from matplotlib.colors import LinearSegmentedColormap

# Default output directory, resolved relative to this file rather than to
# the caller's working directory, so the scripts run from anywhere.
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")

FORMATS = ("png", "pdf", "svg")


def apply_style():
    plt.rcParams.update({
        "text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Serif"],
        "axes.formatter.use_mathtext": True, "font.size": 11,
        "axes.labelsize": 11, "legend.fontsize": 9,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.linewidth": 0.8, "lines.linewidth": 1.15,
        "figure.dpi": 600, "savefig.dpi": 600, "savefig.bbox": "tight",
        "axes.grid": True, "grid.alpha": 0.5, "grid.linestyle": "--",
        # Matplotlib pads every axis by 5 percent of the data range by
        # default, which leaves visible slack between the curve and the
        # frame. The source paper's MATLAB figures have no such padding:
        # the data runs to the box. Setting both margins to zero matches
        # that, and each script then sets its own limits explicitly.
        "axes.xmargin": 0.0, 
        # "axes.ymargin": 0.0,
        # "axes.autolimit_mode": "data",
        # Ticks point INTO the axes and appear on all four sides, as in
        # the source paper. Note that "xtick.direction" governs BOTH the
        # major and the minor ticks; there is no separate
        # "xtick.minor.direction" rcParam, and setting one raises a
        # KeyError. This block applies to 2-D axes only: matplotlib's
        # Axes3D ignores tick.direction, so the surface panels keep
        # outward ticks, which is what MATLAB draws there in any case.
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": False, "ytick.minor.visible": False,
    })


class CMTickFormatter(ScalarFormatter):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.set_useMathText(False)

    def __call__(self, x, pos=None):
        return f"${super().__call__(x, pos)}$"


def cm_x(ax):
    ax.xaxis.set_major_formatter(CMTickFormatter())


def cm_y(ax):
    ax.yaxis.set_major_formatter(CMTickFormatter())


def cm_both(ax):
    cm_x(ax)
    cm_y(ax)


def savefig_all(fig, basename, outdir=None, formats=FORMATS, close=False, **kwargs):
    """Save `fig` as `basename` in every format in `formats`.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    basename : str
        Filename stem, without extension. May include subdirectories.
    outdir : str, optional
        Destination directory. Defaults to the module-level FIGDIR, which
        resolves to `<project>/figs` regardless of working directory.
    formats : iterable of str
        Extensions to write. Defaults to PNG, PDF, and SVG.
    close : bool
        If True, close the figure after saving.

    Returns
    -------
    list of str : the paths written.
    """
    outdir = FIGDIR if outdir is None else outdir
    os.makedirs(outdir, exist_ok=True)
    written = []
    for ext in formats:
        path = os.path.join(outdir, f"{basename}.{ext}")
        # dpi is ignored for the vector formats; bbox comes from rcParams.
        fig.savefig(path, format=ext, **kwargs)
        written.append(path)
    if close:
        plt.close(fig)
    return written


# ----------------------------------------------------------------------
# Conventions taken from the source paper's figures
# ----------------------------------------------------------------------
# The paper's plots are MATLAB output, and several of its conventions
# have to be matched deliberately rather than left to matplotlib's
# defaults.

#: The paper's surfaces use MATLAB's "turbo" colormap. Figure 13 is stored in the PDF as an RGB image and its colorbar matches matplotlib's "turbo" exactly. Figures 9, 10, 14 and 15, however, are stored as CMYK images, and the print conversion mutes turbo noticeably (a navy floor instead of near-black indigo, softer greens and reds). To match those figures as they actually appear in the paper, PAPER_TURBO below is sampled directly from their colorbars (median of the eight bars in Figures 14 and 15, lightly smoothed, 32 stops).
_PAPER_TURBO_STOPS = [
    (0.224, 0.208, 0.486),
    (0.240, 0.276, 0.609),
    (0.253, 0.327, 0.656),
    (0.269, 0.367, 0.681),
    (0.301, 0.417, 0.714),
    (0.330, 0.485, 0.753),
    (0.337, 0.563, 0.795),
    (0.305, 0.671, 0.847),
    (0.271, 0.713, 0.758),
    (0.296, 0.724, 0.626),
    (0.315, 0.731, 0.559),
    (0.332, 0.732, 0.508),
    (0.366, 0.746, 0.480),
    (0.395, 0.756, 0.446),
    (0.445, 0.761, 0.390),
    (0.492, 0.780, 0.351),
    (0.538, 0.794, 0.322),
    (0.592, 0.806, 0.283),
    (0.714, 0.829, 0.206),
    (0.841, 0.802, 0.169),
    (0.897, 0.713, 0.173),
    (0.914, 0.625, 0.181),
    (0.919, 0.539, 0.184),
    (0.909, 0.443, 0.182),
    (0.906, 0.369, 0.172),
    (0.887, 0.304, 0.155),
    (0.864, 0.251, 0.145),
    (0.816, 0.197, 0.136),
    (0.748, 0.158, 0.130),
    (0.672, 0.129, 0.122),
    (0.583, 0.110, 0.112),
    (0.488, 0.087, 0.102)
]
PAPER_TURBO = LinearSegmentedColormap.from_list("paper_turbo", _PAPER_TURBO_STOPS, N=256)
SURFACE_CMAP = PAPER_TURBO        # Figures 9, 10, 14, 15
FIG13_CMAP = "turbo"              # Figure 13 solution surfaces (RGB in the PDF)

#: Colormap for the error surfaces of Figure 13 (cyan to magenta). MATLAB's "cool" and matplotlib's "cool" are the same map.
ERROR_CMAP = "cool"

#: Figure 13's solution surfaces are drawn semi-transparent in the paper (the fold behind the front sheet shows through, and the colours look paler than the colorbar), which is MATLAB's FaceAlpha.
FIG13_ALPHA = 0.75

#: Camera and plot-box shape for every 3-D panel, MEASURED from the paper's embedded panel images rather than eyeballed. For each panel the on-screen direction and length of the x, y (or t) and z axes were read off the tick marks, and the orthographic projection equations were solved for elevation and azimuth; the axis lengths give the box aspect. Matplotlib convention throughout: azimuth from +x, and MATLAB's azimuth = matplotlib's + 90. Panels in the same row pair (TDCNCS / TDCCS) share a camera, but the u and v panels of Figures 14 and 15 do NOT: the paper rotated them separately.
VIEW_FIG9 = dict(elev=28, azim=-112, aspect=None)
VIEW_FIG10 = dict(elev=50, azim=-102, aspect=None)
VIEW_FIG13 = dict(elev=13.7, azim=-64.5, aspect=(1.0, 1.04, 0.91))
VIEW_FIG14_U = dict(elev=22.6, azim=-127.8, aspect=(1.0, 0.98, 0.76))
VIEW_FIG14_V = dict(elev=10.6, azim=-122.2, aspect=(1.0, 1.03, 1.00))
VIEW_FIG15_U = dict(elev=26.6, azim=-81.5, aspect=(1.0, 0.91, 0.57), zlabel_x=-0.17,
                    cbpad=0.07)
VIEW_FIG15_V = dict(elev=12.7, azim=-75.5, aspect=(1.0, 0.90, 0.86), tstep=1.0)
VIEW_ELEV, VIEW_AZIM = VIEW_FIG14_U["elev"], VIEW_FIG14_U["azim"]


def matlab_zlim(zmin, zmax, tight=False):
    """z limits the way MATLAB picks them: out to the nearest 'nice' tick beyond the data (e.g. [-0.095, 1.78] -> [-0.5, 2], [-2.55, 4.97] -> [-3, 5]), or the bare data range when the paper used `axis tight`."""
    if tight:
        return zmin, zmax
    ticks = MaxNLocator(nbins=8, steps=[1, 2, 5, 10]).tick_values(zmin, zmax)
    return float(ticks[ticks <= zmin + 1e-12].max()), float(ticks[ticks >= zmax - 1e-12].min())


def matlab_view(ax, elev=VIEW_ELEV, azim=VIEW_AZIM, aspect=None, tstep=None,
                zlabel_x=None, cbpad=None):
    """Point a 3-D axis at a camera measured from the paper, with MATLAB's orthographic projection (matplotlib defaults to perspective, which bends the box edges) and, when given, the paper's plot-box proportions."""
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    if aspect is not None:
        # zoom compensates for the smaller footprint of a flat box.
        ax.set_box_aspect(aspect, zoom=1.12)
    if tstep is not None:
        # The paper's near-edge-on panels label the receding axis sparsely (0, 1, 2).
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_major_locator(MultipleLocator(tstep))


def matlab_axes3d(ax):
    """Make an Axes3D look like a MATLAB surf plot.

    MATLAB draws white back walls with thin light-grey grid lines and a thin dark box edge, while matplotlib fills the walls grey. This only touches the 3-D panel's decoration, never the data.
    """
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((0.15, 0.15, 0.15, 1.0))
        axis._axinfo["grid"].update(color=(0.85, 0.85, 0.85, 1.0),
                                    linewidth=0.5, linestyle="-")
        axis.line.set_linewidth(0.6)
    ax.tick_params(axis="both", which="major", labelsize=8, pad=0)
    ax.tick_params(axis="z", which="major", labelsize=8, pad=2)
    # MATLAB always draws the z-axis on the LEFT edge of the box, away from the colorbar. Matplotlib picks the side from the camera, and for azimuths between -90 and 0 it lands on the right; "lower" moves it back to the left there. Call this after view_init.
    if -90.0 < ax.azim < 0.0:
        ax.zaxis.set_ticks_position("lower")
    # MATLAB writes the z label reading bottom-to-top; matplotlib auto-rotates it and often flips it upside down.
    ax.zaxis.set_rotate_label(False)
    ax.zaxis.label.set_rotation(90)


def matlab_colorbar(fig, surf, ax, shrink=0.80, pad=0.02, aspect=22):
    """Colorbar placed the way MATLAB's `colorbar` places it.

    In the paper each bar is a narrow strip that stands immediately to the right of the 3-D box and runs almost its full height. Matplotlib's defaults leave a wide gap and a short bar, so the gap is pulled in and the bar lengthened.
    """
    cb = fig.colorbar(surf, ax=ax, shrink=shrink, pad=pad, aspect=aspect)
    cb.ax.tick_params(labelsize=8, direction="out", length=2.5)
    cb.outline.set_linewidth(0.6)
    return cb


def subcaption(ax, text, y=1.02):
    """Place a panel label ABOVE the axis.

    The source paper prints its panel labels ("(a) t = 0") underneath
    each box. They are placed above here instead, which reads better on
    a web page where the figure caption already sits below the image.

    For 2-D axes the label is set as an ordinary axes title, so it
    inherits the usual title spacing and cannot collide with the frame.

    The `y` argument is retained because the 3-D call sites pass their
    own offsets, several of which are negative from when labels sat
    below. Clamping with max(y, 1.0) keeps those panels' labels above the
    box as well, so no figure script needs editing.
    """
    if hasattr(ax, "get_zlim"):
        # Axes3D positions titles poorly and its text() signature differs
        # from the 2-D one, so the label is placed manually in axis
        # coordinates just above the box.
        ax.text2D(0.5, y, text, transform=ax.transAxes,
          ha="center", va="bottom", fontsize=10)
        # ax.text2D(0.5, max(y, 1.0), text, transform=ax.transAxes,
        #           ha="center", va="bottom", fontsize=10)
    else:
        ax.set_title(text, fontsize=10, pad=6)


def tight_limits(ax, x, y=None, pad_y=0.0):
    """Set x limits exactly to the data range, with no padding.

    Together with the zero margins set in apply_style, this makes the
    curve meet the frame on the left and right, as in the paper.
    """
    ax.set_xlim(float(min(x)), float(max(x)))
    if y is not None:
        lo, hi = float(min(y)), float(max(y))
        if pad_y:
            span = hi - lo
            lo, hi = lo - pad_y * span, hi + pad_y * span
        ax.set_ylim(lo, hi)