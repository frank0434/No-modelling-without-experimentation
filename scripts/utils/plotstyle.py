"""
utils/plotstyle.py
------------------
One definition of how every figure in this repo encodes the experiment.

The point is that each dimension gets exactly one visual channel, so the same
experiment reads the same way in every figure:

    location        marker shape        IND o, NLD s, RSA ^
    cultivar        (no channel)        one cultivar per location in this
                                        pipeline (IND Kennebec, NLD Innovator,
                                        RSA Mondial), so the location shape
                                        carries it too
    season          panel column        never a colour
    field (setup)   line style + fill   OpenField solid/filled,
                                        RainShelter dashed/open
    before/after    colour              see MODE below
    observations    black points        always
    ordered levels  grey ramp           darkest = highest (grey_ramp)
    unordered series colour             matplotlib default cycle (SERIES_COLORS)

Grey levels only encode an order. Unordered categories drawn in one panel --
cultivars, treatments, variants -- need colour to stay separable, and use the
default cycle. Where several locations share one axis, colour is SITE_COLOR.

MODE switches the before/after pair between print and screen. Set it once, at
the top of a script, before drawing:

    from utils import plotstyle
    plotstyle.MODE = "slides"
"""

import matplotlib.pyplot as plt

# "paper" for the manuscript, "slides" for presentations.
MODE = "paper"

# Row order for every multi-location figure: temperate first, then subtropical.
LOC_ORDER = ("NLD", "IND", "RSA")

# Location -> marker.
SITE_MARKER = {"IND": "o", "NLD": "s", "RSA": "^"}
TRT_MARKER = {"Full_0 Days": "o", "Full_6 Days": "s", "Full_12 Days": "^",
              "Reduced_0 Days": "+", "Reduced_6 Days": "x", "Reduced_12 Days": "*"}
DEFAULT_MARKER = "o"

# Experimental setup -> line style and marker fill.
FIELD_LS = {"OpenField": "-", "RainShelter": "--"}
FIELD_FILLED = {"OpenField": True, "RainShelter": False}

# Colour, only where shape and fill cannot separate the series. Same three
# values as the weighted-NRMSE figures in 06_overview.py.
SITE_COLOR = {"IND": "#000080", "NLD": "#FF4F00", "RSA": "#007A4D"}

# Unordered series drawn in one panel (cultivar, treatment, variant), in the
# matplotlib default cycle order.
SERIES_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Before/after comparison. Colour is the cue in both modes; where a panel also
# separates fields, the field keeps the line style and only the colour changes.
_BEFORE_AFTER = {
    "paper": {"before": dict(color="0.65", linestyle="-"),
              "after":  dict(color="black", linestyle="-")},
    "slides": {"before": dict(color="#1f77b4", linestyle="--"),
               "after":  dict(color="#d62728", linestyle="-")},
}

# Line styles for distinguishing simulation variants drawn in one colour.
SIM_LINE_STYLES = ["-", "--", ":", "-."]

# Stacked decomposition segments (bias / structural / unsystematic), in print
# order. Red, green, blue with no hatch: three components in one bar need to
# separate at a glance, which grey levels plus hatching did not do.
DECOMP_SEGMENTS = [("red", ""), ("green", ""), ("blue", "")]

# Reference lines (1:1, zero line) and observed points.
REF_COLOR = "0.4"
OBS_COLOR = "black"

# Line widths and sizes, shared with utils/visualise.py's VIZ_* dicts.
LW_SIM = 1.2
LW_REF = 1.0
MS_POINT = 5


def display_season(season):
    """Show winter seasons by planting year; retain labels such as RSA 2025-A/S."""
    season = str(season)
    return season[:4] if season[4:5] in ("-", "_") and season[5:].isdigit() else season


def field_key(field):
    """Accept either spelling of the shelter ('RainShelter' / 'Rainshelter')."""
    return "RainShelter" if str(field).lower() == "rainshelter" else str(field)


def grey_ramp(n, lightest=0.75, darkest=0.0):
    """n grey levels, light to dark -- for ordered levels (irrigation, variants).

    Returns matplotlib grey strings, so the darkest entry is the highest level.
    """
    if n <= 1:
        return [str(darkest)]
    step = (lightest - darkest) / (n - 1)
    return [f"{lightest - i * step:.3f}" for i in range(n)]


def before_after(role):
    """Colour/line-style kwargs for a 'before' or 'after' series under MODE."""
    return dict(_BEFORE_AFTER[MODE][role])


def use_style():
    """Set the shared rcParams. Call once at the top of a plotting script."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        # plain white ground, no grid
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": False,
        # full box: all four spines drawn round every panel
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.edgecolor": "black",
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": LW_SIM,
        "lines.markersize": MS_POINT,
    })
