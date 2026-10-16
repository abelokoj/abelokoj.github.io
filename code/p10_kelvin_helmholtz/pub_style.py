"""
pub_style.py
============
Shared publication-quality matplotlib style, together with the small persistence layer used to keep expensive computations separate from plotting. Copy this file into the project and import from it rather than repeating rcParams in each script.

    from pub_style import apply_style, savefig_all, cm_both
    apply_style()

Every figure is written in three formats by `savefig_all`:

    .png   600 dpi raster, for web embedding
    .pdf   vector, for LaTeX inclusion
    .svg   vector, for web embedding and further editing

Mathematics is rendered through matplotlib's own mathtext engine with the Computer Modern font set rather than through an external LaTeX installation, which keeps the module portable to machines carrying no TeX distribution. Set `text.usetex` to True manually where full LaTeX macro support is required and available.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

#: Default output directory, resolved relative to THIS file rather than the caller's working directory, so scripts run from anywhere.
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")

#: Companion directory for saved figure data, resolved the same way.
DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

FORMATS = ("png", "pdf", "svg")


# ----------------------------------------------------------------------
# Colour cycle
# ----------------------------------------------------------------------
#: House palette, ordered so that the earliest slots stay distinguishable both in normal vision and under deuteranopia, and so that faint colours appear late. A single-series plot therefore receives #009AFA, which is more decisive than matplotlib's muted default and clears the 3:1 contrast threshold a thin line on white requires.
#:
#: The order was chosen by measurement rather than taste: perceptual distance in CIE Lab, computed both normally and through a deuteranopia simulation, with colours below 3:1 contrast against white penalised. Two orderings mattered. Red (#FF0000) and olive (#AC8E18) are almost indistinguishable to a deuteranope at dE 2.2, so they are placed far apart. The three blues, two oranges and two cyans in the original list are likewise separated, since adjacent near-duplicates are what render a legend unreadable.
#:
#: Green occupies slot 2, immediately after red. Under simulated deuteranopia this reduces the worst-case separation among the leading entries: measured values are dE 170.5, 77.1, 46.1, 42.6 and 41.3 for two through six series respectively, falling to 22.9 at seven series, which is at the approximate collision threshold of 22. The ordering therefore remains workable through six series, but a figure whose argument depends on distinguishing two particular curves should carry markers or dash patterns rather than relying on hue alone.
PALETTE = [
    "#009AFA",          # azure          3.00:1
    "#FF0000",          # red            4.00:1
    "#3EA44E",          # green          3.17:1
    "#000000",          # black         21.00:1
    "blue",             # #0000ff        8.59:1
    "mediumvioletred",  # #c71585        5.42:1
    "saddlebrown",      # #8b4513        7.10:1
    "teal",             # #008080        4.77:1
    "#7F00FF",          # violet         6.28:1
    "crimson",          # #dc143c        4.99:1
    "darkorange",       # #ff8c00        2.33:1  faint
    "#AC8E18",          # olive          3.16:1
    "darkturquoise",    # #00ced1        1.95:1  faint
    "orange",           # #ffa500        1.97:1  faint
    "#05fbff",          # bright cyan    1.29:1  faint: fills, not lines
    "skyblue",          # #87ceeb        1.74:1  faint
]

#: Line styles cycled alongside colour once a plot carries many series. Beyond roughly six or seven saturated hues, colour alone cannot keep series distinct under colour-blind vision or in greyscale print: at eight colours the closest pair here falls to dE 4.4 under deuteranopia. Style carries the distinction colour can no longer carry.
LINESTYLES = ["-", "--", "-.", ":"]


def apply_style():
    """Install the house style. Call once, at the top of a script."""
    plt.rcParams.update({
        # Computer Modern throughout, without requiring a LaTeX install.
        "text.usetex": False, "mathtext.fontset": "cm",
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Serif"],

        # cmr10 carries no proper minus glyph and expects mathtext-rendered tick labels. Without both settings below, every figure emits a warning and drops a character.
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,

        # axes.titlesize is set explicitly rather than left at matplotlib's default of "large", which resolves to 1.2 times font.size and therefore drifts whenever the base size changes.
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9,

        # A 1.0 pt data line reads cleanly at single-column width without curves bleeding together where they overlap, while the thinner frame keeps the box from competing with the data.
        "axes.linewidth": 0.8, "lines.linewidth": 1.0,

        # Resolution and bounding box live here, not at the call site, so changing them later is a one-line edit.
        "figure.dpi": 600, "savefig.dpi": 600, "savefig.bbox": "tight",

        # Alpha stays at 1.0 so grid.color is the rendered colour. Compositing a grey at partial opacity against white lightens it, which makes the requested value and the perceived value disagree.
        "axes.grid": True, "grid.linestyle": "--",
        "grid.color": "0.5", "grid.linewidth": 0.5, "grid.alpha": 1.0,

        # Ticks point INTO the axes and appear on all four sides. Note that xtick.direction governs both major and minor ticks; there is no xtick.minor.direction rcParam, and setting one raises a KeyError. Axes3D ignores these entirely, which is correct, since MATLAB draws three-dimensional ticks outward too.
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,

        # Matplotlib pads every axis by five percent of the data range, leaving visible slack between the curve and the frame. Printed figures run the data to the box. Each script then sets its own limits explicitly.
        "axes.xmargin": 0.0, "axes.ymargin": 0.0,

        # House colour cycle, replacing matplotlib's default.
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })


# ----------------------------------------------------------------------
# Tick labels in Computer Modern
# ----------------------------------------------------------------------
class CMTickFormatter(ScalarFormatter):
    """Wrap each tick label in math mode so it renders in Computer Modern.

    Without this the tick numbers are typeset in the default sans font while every other label uses the serif family, which is plainly visible at 600 dpi.
    """

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.set_useMathText(False)

    def __call__(self, x, pos=None):
        return f"${super().__call__(x, pos)}$"


def cm_x(ax):
    """Apply the Computer-Modern tick formatter to the x-axis only."""
    ax.xaxis.set_major_formatter(CMTickFormatter())


def cm_y(ax):
    """Apply it to the y-axis only."""
    ax.yaxis.set_major_formatter(CMTickFormatter())


def cm_both(ax):
    """Apply it to both axes, for panels whose ticks are all linear numeric.

    Do NOT use on a log axis, which needs its power-of-ten labels, nor on an axis carrying category names, which would be mangled into numbers.
    """
    cm_x(ax)
    cm_y(ax)


# ----------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------
def savefig_all(fig, basename, outdir=None, formats=FORMATS, close=False,
                **kwargs):
    """Save `fig` as `basename` in every format in `formats`.

    dpi and bbox come from the rcParams block, so they are not repeated here. Returns the list of paths written.
    """
    outdir = FIGDIR if outdir is None else outdir
    os.makedirs(outdir, exist_ok=True)
    written = []
    for ext in formats:
        path = os.path.join(outdir, f"{basename}.{ext}")
        fig.savefig(path, format=ext, **kwargs)
        written.append(path)
    if close:
        plt.close(fig)
    return written


# ----------------------------------------------------------------------
# Three-dimensional conventions, matched to MATLAB
# ----------------------------------------------------------------------
#: Solution surfaces: blue through cyan, green, yellow, to red.
SURFACE_CMAP = "jet"

#: Error surfaces: cyan to magenta. MATLAB's "cool", identical here.
ERROR_CMAP = "cool"

#: MATLAB's default 3-D camera, view(-37.5, 30), as (elevation, azimuth).
VIEW_ELEV, VIEW_AZIM = 30, -37.5

#: Tested pairing. Higher zoom collides the z-axis labels with the colorbar.
SURFACE_ZOOM, COLORBAR_PAD = 1.15, 0.22


def matlab_view(ax):
    """Point a 3-D axis at MATLAB's default camera angle."""
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)


def zaxis_left(ax):
    """Draw the z-axis on the LEFT of the box instead of the right.

    Useful when a colorbar sits on the right. The camera is unchanged, so the surface looks identical; only the axis moves.

    This manipulates a private attribute and is not part of matplotlib's public API. It has been stable across many versions, but if a future release breaks a 3-D figure, look here first. Flipping the azimuth sign also moves the axis, but swings the camera to a different corner and mirrors the surface, so it is not an equivalent alternative.
    """
    p = ax.zaxis._PLANES
    ax.zaxis._PLANES = (p[2], p[3], p[0], p[1], p[4], p[5])


def surface(fig, ax, X, Y, Z, zlabel="", cmap=None, colorbar=True):
    """Draw a surface with every house convention applied at once."""
    surf = ax.plot_surface(X, Y, Z, cmap=cmap or SURFACE_CMAP, linewidth=0,
                           antialiased=True, rstride=1, cstride=1)
    matlab_view(ax)
    # Axes3D reserves a bounding box much larger than the surface fills, which is why panel labels otherwise float far above the plot.
    ax.set_box_aspect(None, zoom=SURFACE_ZOOM)
    if zlabel:
        ax.set_zlabel(zlabel)
    if colorbar:
        fig.colorbar(surf, ax=ax, shrink=0.60, pad=COLORBAR_PAD)
    return surf


# ----------------------------------------------------------------------
# Panel labels
# ----------------------------------------------------------------------
def subcaption(ax, text, y=1.02, **kwargs):
    """Place a panel label such as "(a) TDCNCS" ABOVE the axis.

    For 2-D axes this routes through set_title, so the label inherits the usual title spacing and cannot collide with the frame. For 3-D axes it is placed manually, because Axes3D positions titles poorly and its text() signature differs from the 2-D one.

    The clamp on y keeps labels above the box even when a caller passes a negative offset left over from a layout that put them below.

    Size is taken from the axes.titlesize rcParam rather than fixed here, so panel labels track the rest of the style. A hard-coded size would silently override any change to that rcParam. Pass fontsize explicitly to override for one panel.
    """
    kwargs.setdefault("fontsize", plt.rcParams["axes.titlesize"])
    if hasattr(ax, "get_zlim"):
        ax.text2D(0.5, max(y, 1.0), text, transform=ax.transAxes,
                  ha="center", va="bottom", **kwargs)
    else:
        ax.set_title(text, pad=6, **kwargs)


def tight_limits(ax, x, y=None, pad_y=0.0):
    """Set the x limits exactly to the data range, with no padding.

    Together with the zero margins set in apply_style, this makes the curve meet the frame on the left and right.
    """
    ax.set_xlim(float(min(x)), float(max(x)))
    if y is not None:
        lo, hi = float(min(y)), float(max(y))
        if pad_y:
            span = hi - lo
            lo, hi = lo - pad_y * span, hi + pad_y * span
        ax.set_ylim(lo, hi)


def style_cycle(n_series):
    """Return a prop_cycle pairing colour with line style for n_series.

    Use when a single axis carries more than about six curves. Colour alone stops separating them reliably at that point, so each block of len(PALETTE) reuses the colours with a different dash pattern. The result stays readable in greyscale and under colour-blind vision.

        ax.set_prop_cycle(style_cycle(12))
    """
    colours, styles = [], []
    for i in range(n_series):
        colours.append(PALETTE[i % len(PALETTE)])
        styles.append(LINESTYLES[(i // len(PALETTE)) % len(LINESTYLES)])
    return plt.cycler(color=colours, linestyle=styles)


def paired_cycle(n_series, stride=4):
    """Return a prop_cycle that rotates line style within the first pass.

    style_cycle changes dash pattern only after the palette is exhausted, which leaves twelve curves solid and separated by hue alone. Where hue separation is already marginal, as it is beyond roughly six series, this variant advances the dash pattern every `stride` curves, so neighbouring entries in the cycle differ in stroke as well as colour.

        ax.set_prop_cycle(paired_cycle(12))
    """
    colours, styles = [], []
    for i in range(n_series):
        colours.append(PALETTE[i % len(PALETTE)])
        styles.append(LINESTYLES[(i // stride) % len(LINESTYLES)])
    return plt.cycler(color=colours, linestyle=styles)


# ----------------------------------------------------------------------
# Figure data: persisting the numbers behind a figure
# ----------------------------------------------------------------------
# Several runs in a spectral or compact-difference project take minutes to hours, because the stable time step scales like dx^3. Re-running a solver only to move a legend or change a colormap is wasteful. The two functions below write every array a figure needs into a single compressed .npz file, and read it back.
#
# Anything that is not an array (an int, a float, a string) is stored as a zero-dimensional array, so read it back with int(...), float(...) or str(...) as appropriate.


def save_data(basename, outdir=None, meta=None, **arrays):
    """Write every keyword argument into <DATADIR>/<basename>.npz.

    Parameters
    ----------
    basename : str
        Filename stem, without extension. Use the figure name, so the data file sits alongside the figure it produced.
    outdir : str, optional
        Destination directory. Defaults to DATADIR.
    meta : dict, optional
        Small scalars worth recording, for example {"N": 80, "eps": 5e-4, "scheme": "TDCCS"}. Stored as a JSON string under the key "__meta__" so it survives the round trip unchanged.
    **arrays
        The arrays to store. Dictionaries keyed by time are flattened, since .npz keys must be strings: pass snapshots as, for example, u_TDCNCS={0.0: arr, 0.5: arr} and they are stored as "u_TDCNCS@0.0" and "u_TDCNCS@0.5".

    Returns
    -------
    str : the path written.
    """
    outdir = DATADIR if outdir is None else outdir
    os.makedirs(outdir, exist_ok=True)

    flat = {}
    for key, val in arrays.items():
        if isinstance(val, dict):
            # A {time: array} snapshot dictionary. Record the keys too, so load_data can rebuild the dictionary in order.
            stamps = sorted(float(t) for t in val)
            flat[f"{key}__keys"] = np.asarray(stamps, dtype=float)
            for t in stamps:
                flat[f"{key}@{t}"] = np.asarray(val[t])
        else:
            flat[key] = np.asarray(val)

    flat["__meta__"] = np.asarray(json.dumps(meta or {}))

    path = os.path.join(outdir, f"{basename}.npz")
    np.savez_compressed(path, **flat)
    print(f"    data saved -> {path}", flush=True)
    return path


def load_data(basename, outdir=None):
    """Read back what save_data wrote.

    Returns a plain dict. Snapshot dictionaries are reassembled, so a field saved as u_TDCNCS={0.0: ..., 0.5: ...} comes back in that same form, keyed by float. Metadata is returned under "meta".
    """
    outdir = DATADIR if outdir is None else outdir
    path = os.path.join(outdir, f"{basename}.npz")
    with np.load(path, allow_pickle=False) as z:
        raw = {k: z[k] for k in z.files}

    out = {"meta": json.loads(str(raw.pop("__meta__")))}

    # Rebuild the {time: array} dictionaries first.
    for key in [k for k in raw if k.endswith("__keys")]:
        field = key[: -len("__keys")]
        stamps = raw.pop(key)
        out[field] = {float(t): raw.pop(f"{field}@{t}") for t in stamps}

    out.update(raw)
    return out


def has_data(basename, outdir=None):
    """True if a saved file already exists, for skip-if-present logic."""
    outdir = DATADIR if outdir is None else outdir
    return os.path.exists(os.path.join(outdir, f"{basename}.npz"))
