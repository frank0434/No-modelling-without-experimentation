"""
utils/visualise.py
------------------
Three figures, the three questions this project asks over and over.

  plot_ts_pp                simulated time series with observations on top
  plot_sim_vs_obs_grid      simulated vs observed against a 1:1 line
  plot_decomposition_sites  MAE and MSE split into their components

The first two share one layout: rows = location, columns = that location's
seasons. `panels` builds it, the plot function loops over the cells and draws.
To change a figure, edit the loop -- there is no dispatch to trace.

Colours, markers and line styles come from utils/plotstyle.py.
Every function returns the figure.

A plotting helper used by only one script lives in that script:
plot_obs_benchmark in 00_summarise_obs.py, plot_day_residuals in
UseDaysResidual.py.
"""

import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

from utils.plotstyle import display_season
from utils.metrics import mae, rmse, MAEDecomposition, MSEDecomposition
from utils import plotstyle
from utils.plotstyle import (
    SITE_MARKER, FIELD_LS, FIELD_FILLED, DECOMP_SEGMENTS, SERIES_COLORS,
    REF_COLOR, OBS_COLOR, LOC_ORDER, field_key, use_style,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Series drawn in one panel are separated by colour, from the matplotlib
# default cycle -- see utils/plotstyle.py. Kept under the old name because
# several scripts import PALETTE directly. Grey levels (grey_ramp) are for
# ordered levels only.
PALETTE = list(SERIES_COLORS)

# Dataset codes: one per location x season x field. Every figure labels an
# experiment with the code from here, so the same dataset reads the same way in
# every figure. RS = rain shelter; -S = the second (short) 2025 RSA planting.
# Keys are (location, season_key, field) as used in MainTable.xlsx and the obs
# tables. 01_data_assessment.py's IND-1/RSA-4 style labels are the same ten
# datasets under the older naming.
DATASETS = {
    ("IND", "2022-2023", "OpenField"):   "IND2022",
    ("IND", "2024-2025", "OpenField"):   "IND2024",
    ("NLD", "2024",      "OpenField"):   "NLD2024",
    ("NLD", "2025",      "OpenField"):   "NLD2025",
    ("RSA", "2024",      "OpenField"):   "RSA2024",
    ("RSA", "2025-A",    "OpenField"):   "RSA2025",
    ("RSA", "2025-S",    "OpenField"):   "RSA2025-S",
    ("RSA", "2024",      "RainShelter"): "RSA2024RS",
    ("RSA", "2025-A",    "RainShelter"): "RSA2025RS",
    ("RSA", "2025-S",    "RainShelter"): "RSA2025RS-S",
}

# Plot order for the ten datasets (open field first, then rain shelter).
DATASET_ORDER = list(DATASETS.values())


def dataset_code(location, season, field="OpenField"):
    """Figure label for one experiment, e.g. dataset_code("RSA", "2025-S") -> "RSA2025-S".

    field accepts either spelling of the rain shelter ("RainShelter" in the obs
    tables, "Rainshelter" in MainTable.xlsx). Falls back to "<location> <season>"
    for a combination that is not one of the ten defined datasets.
    """
    field = "RainShelter" if str(field).lower() == "rainshelter" else str(field)
    return DATASETS.get((str(location), str(season), field), f"{location} {display_season(season)}")

# Line widths, marker sizes and font sizes. Edit here (or override after import)
# to change every figure at once.
VIZ_LW = {
    "sim":    plotstyle.LW_SIM,   # simulation / data lines
    "ref":    plotstyle.LW_REF,   # 1:1 and reference dashed lines
    "legend": 1.5,                # proxy lines in legend handles
}
VIZ_MS = {
    "obs":      plotstyle.MS_POINT ** 2,   # scatter s= for observed points (area in pt²)
    "legend":   plotstyle.MS_POINT + 1,    # markersize= for legend dot/line handles
    "errorbar": plotstyle.MS_POINT,        # markersize= for errorbar plot markers
}
VIZ_FS = {
    "title":  8,   # title font size
    "legend": 7,   # legend font size
    "text":   7,   # text font size
}


def _series_colors(values, explicit=None):
    """Colour per series drawn in one panel.

    An explicit {value: colour} mapping wins (pass plotstyle.before_after for a
    before/after pair, or grey_ramp for an ordered level). Otherwise a single
    series is black and several take the default colour cycle.
    """
    values = list(values)
    if explicit is not None:
        return {v: explicit.get(v, "black") for v in values}
    if len(values) <= 1:
        return {v: "black" for v in values}
    return {v: SERIES_COLORS[i % len(SERIES_COLORS)] for i, v in enumerate(values)}


def _fields_in(df, field_col):
    """Experimental setups present in a frame, or [None] when there is no split."""
    if not field_col or field_col not in df.columns:
        return [None]
    vals = sorted(df[field_col].dropna().astype(str).unique())
    return vals or [None]


def _field_handles(fields, color="black"):
    """Legend proxies naming the line style of each experimental setup."""
    return [mlines.Line2D([], [], color=color, linewidth=VIZ_LW["legend"],
                          linestyle=FIELD_LS.get(field_key(f), "-"), label=f)
            for f in fields if f is not None]


def _padded_limits(lo: float, hi: float, pad_frac: float = 0.06) -> tuple[float, float]:
    """Return axis limits padded so edge markers are not clipped."""
    if pd.isna(lo) or pd.isna(hi):
        return lo, hi
    span = hi - lo
    if span == 0:
        pad = max(abs(lo) * pad_frac, 0.1)
    else:
        pad = span * pad_frac
    return lo - pad, hi + pad


def add_identity_line(ax, lo: float, hi: float, pad_frac: float = 0.08) -> tuple[float, float]:
    """Draw a dashed 1:1 line over [lo, hi] (padded) and set equal, matching x/y limits."""
    lo, hi = _padded_limits(lo, hi, pad_frac)
    ax.plot([lo, hi], [lo, hi], color=REF_COLOR, lw=VIZ_LW["ref"], ls="--", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    return lo, hi


def annotate_metrics(ax, obs, sim, unit: str = "", loc: str = "upper left", extra: str = "") -> dict:
    """Compute MAE/RMSE/Bias (utils.metrics) for a 1:1 scatter and print them in a text box.

    Returns the metrics dict so callers can also tabulate/print them.
    """
    obs_a, sim_a = np.asarray(obs, dtype=float), np.asarray(sim, dtype=float)
    bias = float(np.nanmean(sim_a - obs_a))
    m = {"MAE": mae(obs_a, sim_a), "RMSE": rmse(obs_a, sim_a), "Bias": bias, "n": int(np.sum(np.isfinite(obs_a) & np.isfinite(sim_a)))}
    txt = f"n={m['n']}\nMAE={m['MAE']:.1f}{unit}\nRMSE={m['RMSE']:.1f}{unit}\nBias={m['Bias']:+.1f}{unit}"
    if extra:
        txt += f"\n{extra}"
    va = "top" if "upper" in loc else "bottom"
    ha = "left" if "left" in loc else "right"
    x = 0.03 if ha == "left" else 0.97
    y = 0.97 if va == "top" else 0.03
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=VIZ_FS["text"] + 2, va=va, ha=ha,
            bbox=dict(fc="white", ec="0.7", alpha=0.85, boxstyle="round,pad=0.3"), zorder=6)
    return m


def _grid_spacing(fig, left=0.08, right=0.88, bottom=0.08, top=0.94,
                  wspace=0.05, hspace=0.30):
    """Spacing for a 3-column grid with legends down the right-hand side."""
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                        wspace=wspace, hspace=hspace)


# -- shared grid ---------------------------------------------------------------
# Three figures use the same layout: rows = location, columns = that location's
# seasons. panels() builds it; the caller loops over the cells and draws.

def panels(df, loc_order=LOC_ORDER, ncols=3, figsize=(10, 9), sharey=True):
    """Grid of rows = location, cols = season. Panels with no data are hidden.

    Returns (fig, axes, cells); cells is [(row, col, ax, location, season, sub_df)]
    for the panels that have data, titled with dataset_code.
    """
    seasons = {loc: sorted(df.loc[df["location"] == loc, "season"].dropna().unique())
               for loc in loc_order}
    ncols = max(ncols, max((len(s) for s in seasons.values()), default=1))
    fig, axes = plt.subplots(len(loc_order), ncols, figsize=figsize,
                             sharey=sharey, squeeze=False)
    cells = []
    for r, loc in enumerate(loc_order):
        for c in range(ncols):
            ax = axes[r][c]
            if c >= len(seasons[loc]):
                ax.set_visible(False)
                continue
            season = seasons[loc][c]
            sub = df[(df["location"] == loc) & (df["season"].astype(str) == str(season))]
            ax.set_title(dataset_code(loc, season), fontsize=VIZ_FS["title"])
            cells.append((r, c, ax, loc, season, sub))
    return fig, axes, cells


def outer_labels(ax, col, row, last_row, ylabel, xlabel=None):
    """y label on the first column only, x label on the bottom row only."""
    if col == 0:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    ax.set_xlabel(xlabel if (xlabel and row == last_row) else "")


def row_legend(axes, row, n_seasons, handles, title=None):
    """Legend to the right of the last visible panel in one row."""
    if not handles:
        return
    axes[row][min(n_seasons - 1, axes.shape[1] - 1)].legend(
        handles=handles, title=title, fontsize=VIZ_FS["legend"],
        loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)


def line_handles(values, colors):
    """Legend proxies: one line per series value."""
    return [mlines.Line2D([], [], color=colors.get(v, "black"),
                          linewidth=VIZ_LW["legend"], label=v) for v in values]


def point_handles(values, colors):
    """Legend proxies: one marker per series value."""
    return [mlines.Line2D([], [], marker="o", linestyle="None",
                          color=colors.get(v, OBS_COLOR), label=v,
                          markersize=VIZ_MS["legend"]) for v in values]


def obs_points(ax, frame, ycol, err=None, field_col=None, color=OBS_COLOR,
               by=None, colors=None, x="day"):
    """Observed points with error bars; filled for the open field, open for the shelter."""
    if by is not None and by in frame.columns:
        groups = [((colors or {}).get(v, OBS_COLOR), frame[frame[by] == v])
                  for v in frame[by].unique()]
    else:
        groups = [(color, frame)]
    for col, grp in groups:
        for fld in _fields_in(grp, field_col):
            g = grp if fld is None else grp[grp[field_col].astype(str) == fld]
            filled = fld is None or FIELD_FILLED.get(field_key(fld), True)
            ax.errorbar(g[x], g[ycol], yerr=g[err] if err in g.columns else None,
                        fmt="o", color=col, markerfacecolor=col if filled else "none",
                        zorder=5, markersize=VIZ_MS["errorbar"], capsize=3)


def date_axis(ax):
    """Readable date ticks on a time-series panel."""
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)


def _save(fig, output, OUTPUT_DIR, name):
    if output:
        fig.savefig(OUTPUT_DIR / f"{name}.svg", dpi=300, bbox_inches="tight")


def _shared_limits(sim_vals, obs_df=None, obs_col=None, err_col=None):
    """One y-range across every panel, including the error-bar extent."""
    vals = [sim_vals.dropna()]
    if obs_df is not None and obs_col is not None:
        o = obs_df.dropna(subset=[obs_col])
        if err_col is not None and err_col in o.columns:
            e = o[err_col].fillna(0)
            vals += [o[obs_col] - e, o[obs_col] + e]
        else:
            vals.append(o[obs_col])
    allv = pd.concat(vals)
    return _padded_limits(allv.min(), allv.max()) if not allv.empty else (0, 1)


def plot_ts_pp(df_pp, yvar, hue="Treatment", output=False, OUTPUT_DIR=FIGURES_DIR, RUN_ID="",
               obs_df=None, obs_yvar=None, obs_color=None, obs_label=None,
               obs_err=None, field_col=None, hue_colors=None, loc_order=LOC_ORDER,
               xvar="day", xlabel=None):
    """Simulated time series, one panel per location x season, observations overlaid.

    - one line per `hue` value, separated by colour (or by hue_colors)
    - field_col: open field solid, rain shelter dashed, both in the same panel
    - obs_color: one colour for every observed point; otherwise coloured by `hue`
    - obs_label: legend text for the observed points, or a dict keyed by location
    - loc_order: row order; a location left out of it is dropped from the figure
    - xvar: x column; date ticks are used for "day" only. Any other column (days
      after emergence, thermal time) needs it in both df_pp and obs_df
    - xlabel: x axis label; defaults to "Date" for "day", otherwise the column name
    """
    use_style()
    y_lo, y_hi = _shared_limits(df_pp[yvar], obs_df, obs_yvar, obs_err)
    fig, axes, cells = panels(df_pp, loc_order=loc_order)
    last_row = axes.shape[0] - 1

    for r, c, ax, loc, season, sub in cells:
        hue_vals = sorted(sub[hue].dropna().unique()) if hue in sub.columns else []
        colors = _series_colors(hue_vals, hue_colors)

        # simulated lines: one pass per setup, so both keep their own line style
        for fld in _fields_in(sub, field_col):
            fsub = sub if fld is None else sub[sub[field_col].astype(str) == fld]
            ls = "-" if fld is None else FIELD_LS.get(field_key(fld), "-")
            for hv in (hue_vals or [None]):
                grp = fsub if hv is None else fsub[fsub[hue] == hv]
                grp = grp.dropna(subset=[xvar, yvar]).groupby(xvar, as_index=False)[yvar].mean()
                if not grp.empty:
                    ax.plot(grp[xvar], grp[yvar], color=colors.get(hv, "black"),
                            linestyle=ls, linewidth=VIZ_LW["sim"])

        # observed points
        if obs_df is not None and obs_yvar is not None:
            osub = obs_df[(obs_df["location"] == loc)
                          & (obs_df["season"].astype(str) == str(season))].dropna(subset=[obs_yvar])
            if not osub.empty:
                obs_points(ax, osub, obs_yvar, obs_err, field_col,
                           color=obs_color or OBS_COLOR,
                           by=None if obs_color else hue, colors=colors, x=xvar)

        ax.set_ylim(y_lo, y_hi)
        outer_labels(ax, c, r, last_row, yvar,
                     xlabel or ("Date" if xvar == "day" else xvar))
        if xvar == "day":
            date_axis(ax)

    # one legend per row: series, then setups, then the observed marker
    for r, loc in enumerate(LOC_ORDER):
        loc_data = df_pp[df_pp["location"] == loc]
        if loc_data.empty:
            continue
        hue_vals = sorted(loc_data[hue].dropna().unique()) if hue in loc_data.columns else []
        colors = _series_colors(hue_vals, hue_colors)
        handles = line_handles(hue_vals, colors)
        fields = _fields_in(loc_data, field_col)
        if len(fields) > 1:
            handles += _field_handles(fields)
        if obs_df is not None and obs_yvar is not None:
            if obs_color is not None:
                label = obs_label.get(loc, "observed") if isinstance(obs_label, dict) else (
                    obs_label or "observed")
                handles += point_handles([label], {label: obs_color})
            elif hue in obs_df.columns:
                handles += point_handles(hue_vals, colors)
        row_legend(axes, r, len(loc_data["season"].dropna().unique()), handles, title=hue)

    _grid_spacing(fig, hspace=0.36)
    _save(fig, output, OUTPUT_DIR, f"pp_timeseries_{yvar}_{RUN_ID}")
    return fig


def plot_sim_vs_obs_grid(df, obs_col, sim_col, hue="cultivar", err_col=None,
                         xlabel=None, ylabel=None, output=False, OUTPUT_DIR=FIGURES_DIR,
                         RUN_ID="", field_col=None, hue_colors=None):
    """Simulated vs observed scatter against a 1:1 line, one panel per location x season.

    - marker shape is the location, fill the setup (open field filled, shelter open)
    - one colour per `hue` value, or the explicit hue_colors mapping
    - all panels share one axis range so they compare directly
    """
    use_style()
    xlabel, ylabel = xlabel or obs_col, ylabel or sim_col
    allv = pd.concat([df[obs_col], df[sim_col]]).dropna()
    lo, hi = _padded_limits(allv.min(), allv.max()) if not allv.empty else (0, 1)

    fig, axes, cells = panels(df, figsize=(16, 10), sharey=False)
    last_row = axes.shape[0] - 1

    for r, c, ax, loc, season, sub in cells:
        sub = sub.dropna(subset=[obs_col, sim_col])
        hue_vals = sorted(sub[hue].dropna().unique()) if hue in sub.columns else []
        colors = _series_colors(hue_vals, hue_colors)
        marker = SITE_MARKER.get(loc, "o")
        for hv in (hue_vals or [None]):
            grp = sub if hv is None else sub[sub[hue] == hv]
            color = colors.get(hv, "black")
            for fld in _fields_in(grp, field_col):
                g = grp if fld is None else grp[grp[field_col].astype(str) == fld]
                filled = fld is None or FIELD_FILLED.get(field_key(fld), True)
                xerr = g[err_col] if (err_col in g.columns and g[err_col].notna().any()) else None
                ax.errorbar(g[obs_col], g[sim_col], xerr=xerr, fmt=marker, color=color,
                            markerfacecolor=color if filled else "none",
                            capsize=3, markersize=VIZ_MS["errorbar"])
        ax.plot([lo, hi], [lo, hi], "--", color=REF_COLOR, linewidth=VIZ_LW["ref"])
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        outer_labels(ax, c, r, last_row, ylabel, xlabel)
        ax.set_xlabel(xlabel)

    for r, loc in enumerate(LOC_ORDER):
        loc_data = df[df["location"] == loc]
        if loc_data.empty:
            continue
        hue_vals = sorted(loc_data[hue].dropna().unique()) if hue in loc_data.columns else []
        colors = _series_colors(hue_vals, hue_colors)
        marker = SITE_MARKER.get(loc, "o")
        handles = [mlines.Line2D([], [], marker=marker, linestyle="None", color=colors[v],
                                 label=v, markersize=VIZ_MS["legend"]) for v in hue_vals]
        fields = _fields_in(loc_data, field_col)
        if len(fields) > 1:
            handles += [mlines.Line2D([], [], marker=marker, linestyle="None", color=OBS_COLOR,
                                      markerfacecolor=OBS_COLOR if FIELD_FILLED.get(field_key(f), True)
                                      else "none", label=f, markersize=VIZ_MS["legend"])
                        for f in fields]
        row_legend(axes, r, len(loc_data["season"].dropna().unique()), handles, title=loc)

    plt.tight_layout()
    _save(fig, output, OUTPUT_DIR, f"pp_sim_vs_obs_{obs_col}_{RUN_ID}")
    return fig


def stacked_bars(ax, values, labels, names, ylabel, title, ylim=None, blank="n<3"):
    """One stacked bar per group: three segments, greys and hatches from plotstyle.

    values: [(v1, v2, v3) or None] in the order of `labels`; None leaves a blank bar
    names:  the three segment names, for the legend
    """
    for i, vals in enumerate(values):
        if vals is None:
            ax.text(i, 0, blank, ha="center", va="bottom", fontsize=VIZ_FS["text"])
            continue
        bottom = 0.0
        for val, (color, hatch) in zip(vals, DECOMP_SEGMENTS):
            ax.bar(i, val, bottom=bottom, width=0.6, color=color, hatch=hatch,
                   edgecolor="white")
            bottom += val
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(handles=[mpatches.Patch(facecolor=c, hatch=h, label=n)
                       for n, (c, h) in zip(names, DECOMP_SEGMENTS)],
              fontsize=VIZ_FS["legend"], loc="upper center",
              bbox_to_anchor=(0.5, -0.15), ncol=1, frameon=False)


def plot_decomposition_sites(site_pairs, title=None, output_path=None, mae_ylim=None,
                             mse_ylim=None, var_label="DVS", unit=None):
    """Error decomposition for several groups in one figure.

    site_pairs: {label: (obs, sim)}, each pooled over that group's seasons. Panel order
    follows the dict.

    - top row: one obs-vs-sim scatter per group, 1:1 line, OLS fit, centroid
    - bottom left: MAE split (bias / proportional / unsystematic), in `unit`
    - bottom right: MSE split (bias / distribution / sequence), in `unit` squared
    - separate axes and separate units: MAE and MSE are never put on one scale
    - a group with fewer than 2 pairs gets no MAE bar, fewer than 3 gets no MSE bar
    - mae_ylim / mse_ylim: pass the same limits to several calls so bar heights compare
    """
    unit = unit or var_label
    labels = list(site_pairs)
    n = len(labels)

    # use_style mutates global rcParams; snapshot so this figure's styling stays local
    _prev_rc = plt.rcParams.copy()
    use_style()

    # clean each group's pairs, decompose, and find one shared scatter range
    pairs, maes, mses = {}, {}, {}
    gmin, gmax = np.inf, -np.inf
    for lab in labels:
        obs, sim = (np.asarray(v, dtype=float) for v in site_pairs[lab])
        mask = np.isfinite(obs) & np.isfinite(sim)
        obs, sim = obs[mask], sim[mask]
        pairs[lab] = (obs, sim)
        for store, cls in ((maes, MAEDecomposition), (mses, MSEDecomposition)):
            try:
                store[lab] = cls(obs, sim)
            except ValueError:
                store[lab] = None
        if obs.size:
            gmin, gmax = min(gmin, obs.min(), sim.min()), max(gmax, obs.max(), sim.max())
    lo, hi = _padded_limits(gmin, gmax)

    # scatter row spans 2 columns per group; the two bar panels split the row below
    fig, axd = plt.subplot_mosaic(
        [[f"sc{i}" for i in range(n) for _ in range(2)], ["mae"] * n + ["mse"] * n],
        figsize=(3.4 * n, 8), gridspec_kw={"hspace": 0.35, "wspace": 1.2})

    for i, lab in enumerate(labels):
        obs, sim = pairs[lab]
        d = maes[lab]
        ax = axd[f"sc{i}"]
        ax.plot([lo, hi], [lo, hi], "--", color=REF_COLOR, lw=VIZ_LW["ref"],
                label="1:1" if i == 0 else None)
        if d is not None:
            xs = np.array([lo, hi])
            ax.plot(xs, d.intercept + d.mbe + d.slope * xs, "-", color="black", lw=1.2,
                    label="OLS fit" if i == 0 else None)
            ax.scatter(obs.mean(), sim.mean(), marker="X", color="0.5", s=90, zorder=6,
                       label="centroid" if i == 0 else None)
        else:
            ax.text(0.5, 0.02, f"n={obs.size}", ha="center", va="bottom",
                    fontsize=VIZ_FS["text"], transform=ax.transAxes)
        ax.scatter(obs, sim, marker=SITE_MARKER.get(str(lab).split()[0], "o"),
                   color=OBS_COLOR, s=VIZ_MS["obs"], zorder=5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(lab, fontsize=11)
        ax.set_xlabel(f"Observed {var_label}")
        ax.set_ylabel(f"Simulated {var_label}" if i == 0 else "")
        if i == 0:
            ax.legend(fontsize=VIZ_FS["legend"], loc="upper left")

    stacked_bars(axd["mae"],
                 [None if maes[l] is None else (maes[l].MAE_b, maes[l].MAE_p, maes[l].MAE_u)
                  for l in labels],
                 labels, ("bias", "proportional", "unsystematic"),
                 f"MAE ({unit})", "MAE decomposition", mae_ylim,
                 blank="n<2")
    stacked_bars(axd["mse"],
                 [None if mses[l] is None else
                  (mses[l].Bias_component, mses[l].Distribution, mses[l].Sequence)
                  for l in labels],
                 labels, ("bias²", "distribution", "sequence"),
                 f"MSE ({unit}²)", "MSE decomposition", mse_ylim)
    axd["mse"].yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    axd["mse"].ticklabel_format(style="sci", axis="y", scilimits=(-3, 3))

    if title:
        fig.suptitle(title, fontsize=13)

    # small gap between the MAE and MSE panels, taken from their facing edges so the
    # row stays aligned with the scatter row above
    for key, shift in (("mae", 0.0), ("mse", 0.01)):
        pos = axd[key].get_position()
        axd[key].set_position([pos.x0 + shift, pos.y0, pos.width - 0.01, pos.height])

    if output_path:
        fig.savefig(FIGURES_DIR / output_path, dpi=300, bbox_inches="tight")

    plt.rcParams.update(_prev_rc)   # restore global style
    return fig
