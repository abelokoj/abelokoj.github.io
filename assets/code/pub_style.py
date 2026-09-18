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
from matplotlib.ticker import ScalarFormatter

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

#: Colormap used for every surface plot in the paper (blue-cyan-green-
#: yellow-red). Matplotlib's "jet" is the closest direct equivalent.
SURFACE_CMAP = "jet"

#: MATLAB's default 3-D camera, view(-37.5, 30), expressed as matplotlib
#: (elevation, azimuth). Every surface panel in the paper uses it, so the
#: same numbers are used here for every surface panel.
VIEW_ELEV, VIEW_AZIM = 30, -37.5


def matlab_view(ax):
    """Point a 3-D axis at MATLAB's default camera angle."""
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)


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
        ax.text2D(0.5, max(y, 1.0), text, transform=ax.transAxes,
                  ha="center", va="bottom", fontsize=10)
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

