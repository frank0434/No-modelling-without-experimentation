# %% Imports, paths, and one shared style
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
_root = next(p for p in (_here, *_here.parents)
             if (p / "scripts").is_dir() and (p / "data").is_dir())
sys.path.insert(0, str(_root / "scripts"))
from manuscript_common import (  
    MAIN_FIG_DIR, MANUSCRIPT_RC, OUTPUT_DIRECTORIES, ROOT, LINE_STYLE,
    SITE_COLOR, SITE_MARKER, SEASON_MARKER, DAYLENGTH_YEAR_LINE,
    PALETTE, HEATING_COLOR, HEATING_MARKER, HEATING_FILLED,
    HEATING_LINE, TREATMENT_COLOR, TREATMENT_MARKER, 
    NUMBER_DIR, SOURCE_DIR, TREATMENT_ORDER, REF_COLOR,
)
from utils.plotstyle import display_season
from utils.metrics import nrmse

for directory in OUTPUT_DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(MANUSCRIPT_RC)
PHENOLOGY = ROOT / "output" / "phenology"
WLD_IND = ROOT / "output" / "wld_growth" / "ind"


# %% Day length over days after observed emergence (prepared by 01_weather.py)
df_daylength_dae = pd.read_csv(
    SOURCE_DIR / "weather_daylength_DAE.csv", dtype={"season": str},
    parse_dates=["day", "Emergence", "TuberInitiation", "Maturity"],
)
df_daylength_dae = df_daylength_dae[
    ~((df_daylength_dae["location"].isin(["NLD", "RSA"])) & (df_daylength_dae["season"] == "2024"))
].copy()
df_rsa_daylength_ti = pd.read_csv(
    PHENOLOGY / "rsa_photoperiod_selected.csv", dtype={"season": str},
    parse_dates=["initiation"],
)
df_rsa_daylength_ti = df_rsa_daylength_ti[
    df_rsa_daylength_ti["field"].eq("OpenField")
    & df_rsa_daylength_ti["scenario"].eq("TSUM1-only effect")
].set_index("season")
fig, axes = plt.subplots(2, 2, figsize=(4.5, 3), sharex=True, sharey=True,
                         layout="constrained")
for panel, (ax, location) in enumerate(zip(axes.flat, ["NLD", "CAN", "IND", "RSA"])):
    seasons = sorted(
        df_daylength_dae.loc[df_daylength_dae["location"] == location,
                             "season"].unique()
    )
    for i, season in enumerate(seasons):
        season_color = PALETTE["black"] if i == len(seasons) - 1 else "0.5"
        season_label = display_season(season)
        marker_season = season_label.removesuffix("-A").removesuffix("-S")
        style_season = season_label if location == "RSA" else marker_season
        season_marker = SEASON_MARKER[style_season]
        season_line = DAYLENGTH_YEAR_LINE[style_season]
        sub = df_daylength_dae[
            (df_daylength_dae["location"] == location)
            & (df_daylength_dae["season"] == season)
        ]
        ax.plot(
            sub["DAE"], sub["DAYL"],
            color=season_color,
            linestyle=season_line,
            marker=season_marker,
            markevery=(i * 5, 15),
            markersize=3.5,
            markerfacecolor="white", markeredgewidth=0.7,
            linewidth=0.8, zorder=3,
            label=season_label,)
        ti_date = (df_rsa_daylength_ti.loc[season, "initiation"]
                   if location == "RSA" else sub["TuberInitiation"].iloc[0])
        ti_sub = sub[sub["day"] == ti_date]
        if not ti_sub.empty and (location != "RSA" or season == "2025-A"):
            ti_point = ti_sub.iloc[0]
            ax.plot(
                ti_point["DAE"], ti_point["DAYL"],
                marker=season_marker, color=season_color,
                markersize=3.5, markerfacecolor=season_color, markeredgewidth=0.7,
                linestyle="none", zorder=5,
            )
            ax.annotate(
                "", xy=(ti_point["DAE"], ti_point["DAYL"]),
                xytext=((i - (len(seasons) - 1) / 2) * 10, 20)
                if location in ["NLD", "IND"] else (0, 20),
                textcoords="offset points",
                ha="center", va="top",
                arrowprops=dict(arrowstyle="->", color=season_color, lw=0.8,
                                linestyle=season_line,
                                shrinkA=0, shrinkB=2),
            )
    ax.set_title(location)
    ax.text(0.01, 0.99, f"({chr(97 + panel)})", transform=ax.transAxes,
            ha="left", va="top", fontsize=8)
    ax.set_xlim(0, 125)
    ax.set_ylim(10, 24)
    ax.legend(loc="upper right", handlelength=2,frameon=False)

for ax in axes[1]:
    ax.set_xlabel("Days after emergence")
for ax in axes[:, 0]:
    ax.set_ylabel("Day length (h)")

output_path = ROOT / "manuscript_outputs" / "figures" / "weather_daylength_DAE.svg"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Wrote {output_path}")


# %% Figure 1 - emergence response-shape consistency
path = SOURCE_DIR / "fig01_emergence_consistency.csv"
data = pd.read_csv(path)
rule_order = ['clip@18', 'clip@25', 'Kooman']
data = data.loc[
    data['site'].isin(['CAN', 'IND', 'RSA'])
    & data['source'].eq('air')
    & data['rule'].isin(rule_order)
].copy()
mapping = {'clip@18': 'TEFFMX = 18 °C',
           'clip@25': 'TEFFMX = 25 °C',
           'Kooman': 'Kooman et al 1995'}
data['legend'] = data['rule'].replace(mapping)
legend_order = [mapping.get(rule, rule) for rule in rule_order]
data['experiment'] = data['site'].str.cat(data['field'], sep='\n')
plot_data = (
    data.pivot(
        index='experiment', columns='legend', values=['cv', 'abs_off'],
    )
)
plot_data = plot_data[
    [(metric, legend) for metric in ['cv', 'abs_off'] for legend in legend_order]
]
figure1_site_order = {'CAN': 0, 'IND': 1, 'RSA': 2}
plot_data = plot_data.sort_index(
    key=lambda labels: labels.str.split('\n').str[0].map(figure1_site_order))
plot_data.index = plot_data.index.str.replace('\nOpenField', '', regex=False)
from matplotlib.ticker import PercentFormatter

fig, axes = plt.subplots(1, 2, figsize=(6, 3.0))
plot_data['cv'].plot.bar(
    ax=axes[0], color=['white', 'white', '0.45'], edgecolor='black',
    linewidth=0.6, rot=0)
plot_data['abs_off'].plot.bar(
    ax=axes[1], color=['white', 'white', '0.45'], edgecolor='black',
    linewidth=0.6, rot=0, legend=False)
plt.setp(axes[0].containers[1].patches, hatch='///')
plt.setp(axes[1].containers[1].patches, hatch='///')
axes[0].set_xlabel('')
axes[1].set_xlabel('')
axes[0].set_ylabel("Coefficient of variation across seasons")
axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
axes[1].set_ylabel("Mean absolute prediction error (days)")
axes[0].text(0.01, 0.99, "(a)", transform=axes[0].transAxes,
             ha="left", va="top")
axes[1].text(0.01, 0.99, "(b)", transform=axes[1].transAxes,
             ha="left", va="top")
axes[0].set_ylim(0, 0.4)
axes[1].set_ylim(0, 15)
axes[0].legend(title='Assumptions', loc='upper left',
               bbox_to_anchor=(0, 0.95))
fig.savefig(MAIN_FIG_DIR / "Figure_01.svg", format="svg", bbox_inches="tight")
plt.show()
print(f"wrote {MAIN_FIG_DIR / 'Figure_01.svg'}")


# %% Figure 2 - render the saved comparison from 02.7_pp_photoperiod.py
# Run 02.7 first to update the predictions and plotted-point table.
duration = pd.read_csv(SOURCE_DIR / "fig02_photoperiod_comparison.csv", dtype={"season": str})
df_figure2_labelled = duration.copy()
figure2_site_order = ["NLD", "CAN", "IND", "RSA"]
df_figure2_labelled_params = pd.read_csv(
    ROOT / "output/phenology/phase_duration_check/phase_duration_parameters.csv", dtype={"season": str},
)
df_figure2_labelled_rsa = pd.read_csv(
    ROOT / "output/phenology/rsa_photoperiod_selected.csv", dtype={"season": str},
)
df_figure2_labelled_spring = df_figure2_labelled_rsa[
    df_figure2_labelled_rsa.season.eq("2025-S")
    & df_figure2_labelled_rsa.field.eq("OpenField")
    & df_figure2_labelled_rsa.scenario.ne("Temperature only")
].set_index("scenario").loc[["TSUM1-only effect", "TSUM2-only effect"]]

fig_figure2_labelled, axes_figure2_labelled = plt.subplots(
    2, 2, figsize=(5, 5), sharex="row", sharey="row"
)
for figure2_row, figure2_panels in enumerate([["P1", "P2"], ["P3", "P4"]]):
    df_figure2_labelled_values = df_figure2_labelled[
        df_figure2_labelled["panel"].isin(figure2_panels)
    ][["obs_days", "pred_days"]].stack().dropna()
    figure2_labelled_pad = max(float(df_figure2_labelled_values.max()
                                   - df_figure2_labelled_values.min()), 1.0) * 0.12
    figure2_labelled_limits = (
        float(df_figure2_labelled_values.min()) - figure2_labelled_pad,
        float(df_figure2_labelled_values.max()) + figure2_labelled_pad,
    )
    for figure2_col, figure2_panel in enumerate(figure2_panels):
        figure2_ax = axes_figure2_labelled[figure2_row, figure2_col]
        figure2_ax.plot(figure2_labelled_limits, figure2_labelled_limits,
                       color=REF_COLOR, ls="--", lw=0.7)
        df_figure2_labelled_panel = df_figure2_labelled[
            df_figure2_labelled["panel"].eq(figure2_panel)
        ]
        for figure2_site, df_figure2_labelled_group in df_figure2_labelled_panel.groupby("site"):
            # RSA setups and hypotheses are drawn separately below.
            df_figure2_labelled_plain = df_figure2_labelled_group[
                df_figure2_labelled_group["site"].ne("RSA")
            ]
            figure2_ax.scatter(
                df_figure2_labelled_plain["obs_days"], df_figure2_labelled_plain["pred_days"],
                marker=SITE_MARKER[figure2_site], facecolors="white",
                edgecolors=SITE_COLOR[figure2_site], s=24, linewidths=0.7, label=figure2_site,
            )
        if figure2_panel in ["P1", "P2", "P3", "P4"]:
            for record in df_figure2_labelled_panel[df_figure2_labelled_panel.site.eq("RSA")].itertuples():
                filled = record.field == "RainShelter"
                figure2_ax.scatter(
                    record.obs_days, record.pred_days,
                    marker=SITE_MARKER["RSA"],
                    facecolors=SITE_COLOR["RSA"] if filled else "white",
                    edgecolors=SITE_COLOR["RSA"], s=24, linewidths=0.7, zorder=5,
                )
                if record.season == "2025-S" and figure2_panel == "P4" and record.field == "OpenField":
                    offset = (20, -0) if record.scenario == "TSUM1-only effect" else (-20, -0)
                    figure2_ax.annotate(
                        "Hypothesis 1" if record.scenario == "TSUM1-only effect" else "Hypothesis 2",
                        (record.obs_days, record.pred_days), xytext=offset,
                        textcoords="offset points", fontsize=7, color="black", zorder=7,
                        annotation_clip=False, ha="left" if offset[0] > 0 else "right",
                        arrowprops=dict(arrowstyle="->", color="black", lw=0.5,
                                        mutation_scale=6, shrinkA=2, shrinkB=4),
                    )
        if figure2_panel == "P4":
            figure2_table = figure2_ax.table(
                cellText=[["RSA",
                           f"{row.DLO:.1f}", f"{row.DLC:.1f}"]
                          for method, row in df_figure2_labelled_spring.loc[["TSUM2-only effect"]].iterrows()],
                colLabels=["Location", "DLO", "DLC"],
                cellLoc="center", colWidths=[0.4, 0.3, 0.3],
                bbox=[0.38, 0.02, 0.61, 0.128],
            )
            figure2_table.auto_set_font_size(False)
            figure2_table.set_fontsize(6)

            plt.setp(
                figure2_table.get_celld().values(),
                linewidth=0.4,
                edgecolor="0.5",
            )
        if figure2_panel == "P2":
            df_figure2_hypothesis1 = df_figure2_labelled_panel[
                df_figure2_labelled_panel.site.eq("RSA")
                & df_figure2_labelled_panel.season.eq("2025-S")
            ]
            hypothesis1_targets = [
                (record.obs_days, record.pred_days)
                for record in df_figure2_labelled_panel[df_figure2_labelled_panel.site.eq("CAN")].itertuples()
            ] + [(df_figure2_hypothesis1.obs_days.mean(), df_figure2_hypothesis1.pred_days.mean())]
            figure2_ax.text(0.35, 0.8, "Hypothesis 1",
                             transform=figure2_ax.transAxes,
                           fontsize=7, ha="center", va="bottom")
            for target in hypothesis1_targets:
                figure2_ax.annotate(
                    "", target, xytext=(0.5, 0.79), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.5,
                                    mutation_scale=6, shrinkA=2, shrinkB=4),
                )
            df_figure2_labelled_table = df_figure2_labelled_params[
                df_figure2_labelled_params["site"].ne("RSA")
            ][["site", "init_DLO_h", "init_DLC_h"]].drop_duplicates().sort_values(
                "site", key=lambda s: s.map({"NLD": 0, "CAN": 1, "IND": 2}))
            hypothesis1 = df_figure2_labelled_spring.loc["TSUM1-only effect"]
            figure2_table = figure2_ax.table(
                cellText=[[row.site, f"{row.init_DLO_h:.1f}", f"{row.init_DLC_h:.1f}"]
                          for row in df_figure2_labelled_table.itertuples()]
                         + [["RSA", f"{hypothesis1.DLO:.1f}", f"{hypothesis1.DLC:.1f}"]],
                colLabels=["Location", "DLO", "DLC"], cellLoc="center",
                  colWidths=[0.4, 0.3, 0.3],
                bbox=[0.38, 0.02, 0.61, 0.32],
            )
            figure2_table.auto_set_font_size(False)
            figure2_table.set_fontsize(6)
            plt.setp(figure2_table.get_celld().values(), linewidth=0.4, edgecolor="0.5")
        figure2_ax.set(xlim=figure2_labelled_limits, ylim=figure2_labelled_limits, aspect="equal")
        figure2_ax.text(0.01, 0.99, f"({chr(97 + figure2_row * 2 + figure2_col)})",
                       transform=figure2_ax.transAxes, ha="left", va="top")
        figure2_missing = df_figure2_labelled_panel[
            df_figure2_labelled_panel[["obs_days", "pred_days"]].isna().any(axis=1)
        ]
        if figure2_panel == "P3":
            df_figure2_spring_point = df_figure2_labelled_panel[
                df_figure2_labelled_panel.site.eq("RSA")
                & df_figure2_labelled_panel.season.eq("2025-S")
                & df_figure2_labelled_panel.field.eq("OpenField")
            ]
            figure2_ax.scatter(
                df_figure2_spring_point.obs_days, df_figure2_spring_point.pred_days,
                marker="o", facecolors="none", edgecolors="0",
                s=120, linewidths=1, linestyle=":",
            )

axes_figure2_labelled[0, 0].set_title("Thermal only")
axes_figure2_labelled[0, 1].set_title("Thermal + photoperiod")
axes_figure2_labelled[0, 0].set_ylabel("Predicted EM-TI duration (days)")
axes_figure2_labelled[1, 0].set_ylabel("Predicted TI-MA duration (days)")
for ax in axes_figure2_labelled[1]:
    ax.set_xlabel("Reference TI-MA window (days)")
for ax in axes_figure2_labelled[0]:
    ax.set_xlabel("Observed EM-TI duration (days)")
axes_figure2_labelled[0, 0].legend(
    [Line2D([], [], marker=SITE_MARKER[site], markerfacecolor="white",
            markeredgecolor=SITE_COLOR[site], markersize=np.sqrt(24), markeredgewidth=0.7,
            linestyle="none") for site in figure2_site_order]
    + [Line2D([], [], marker=SITE_MARKER["RSA"], markerfacecolor=SITE_COLOR["RSA"],
              markeredgecolor=SITE_COLOR["RSA"], markersize=np.sqrt(24), markeredgewidth=0.7,
              linestyle="none")],
    ["RSA OF" if site == "RSA" else site for site in figure2_site_order]
    + ["RSA RS "],
    title="Location", frameon=True, loc="upper right")
fig_figure2_labelled.subplots_adjust(hspace=0.25, wspace=0.1)
fig_figure2_labelled.savefig(MAIN_FIG_DIR / "Figure_02.svg", format="svg", bbox_inches="tight")
# Retain the filename of the attached figure as an alias of the main figure.
fig_figure2_labelled.savefig(
    MAIN_FIG_DIR / "Figure_02_methods_labelled.svg",
    format="svg", bbox_inches="tight")
plt.show()
print(f"wrote {MAIN_FIG_DIR / 'Figure_02.svg'}")


# %% Figure 2 Results values - endpoint differences using a common start
df_figure2_duration_diff = duration[
    ["panel", "site", "season", "field", "role", "scenario", "obs_days", "pred_days"]
].copy()
df_figure2_duration_diff["pred_minus_obs_days"] = (
    df_figure2_duration_diff["pred_days"] - df_figure2_duration_diff["obs_days"]
)
print("\nFigure 2 differences (predicted - reference, days; common start):")
print(df_figure2_duration_diff.round(1).to_string(index=False))


# %% Figure 3 - produced by 03.1_p_growth_corrections.py
# IND2024 only: LAI on the left, TWSO on the right.
# That script owns the three correction runs, rendering and matched error summaries.

# %% Figure 3 Results text - the same IND2024 comparisons used in the figure
figure3_numbers = pd.read_csv(NUMBER_DIR / "figure3_writing_numbers.csv", dtype={"season": str})
print("\nFigure 3 IND2024 TDWI and SLATB corrections")
print(figure3_numbers.round(3).to_string(index=False))


# %% Figure 4 - produced by 03.2_p_growth_AMAX.py
# NLD2025 light-response fits only; NLD2024 temperature comparison is supplementary.
figure4_summary = pd.read_csv(NUMBER_DIR / "figure4_amax_summary.csv")
print("\nFigure 4 retained AMAX observations")
print(figure4_summary.round(3).to_string(index=False))
figure4_writing_numbers = pd.read_csv(NUMBER_DIR / "figure4_writing_numbers.csv")
print("Gross AMAX per unit leaf area, in kg CO2 ha(leaf)-1 h-1 and umol CO2 m(leaf)-2 s-1:")
print(figure4_writing_numbers.round(3).to_string(index=False))


# %% Figure 5 - fitted allocation fractions, effective SPAN and LAI
figure5_targets = {("NLD", "2025"), ("IND", "2024-2025")}
allocation = pd.read_csv(SOURCE_DIR / "s10_partition_curves.csv", dtype={"season": str})
allocation = allocation.loc[[key in figure5_targets for key in zip(allocation.site, allocation.season)]]
assert set(zip(allocation.site, allocation.season)) == figure5_targets
assert not allocation.duplicated(["site", "season", "table", "variant", "dvs"]).any()
span_parameters = pd.read_csv(SOURCE_DIR / "table_s02_parameters.csv", dtype={"season": str})
span_parameters = span_parameters.loc[
    span_parameters.param.eq("SPAN")
    & np.array([key in figure5_targets for key in zip(span_parameters.site, span_parameters.season)])
].set_index("site")
assert set(span_parameters.index) == {"NLD", "IND"} and span_parameters.index.is_unique
data = pd.read_csv(SOURCE_DIR / "fig05_growth_timeseries.csv", dtype={"season": str})
data = data.loc[[key in figure5_targets for key in zip(data.site, data.season)]].copy()
assert set(zip(data.site, data.season)) == figure5_targets
assert not data.duplicated(["site", "season", "variant", "variable", "series_kind", "DAE"]).any()
figure5_lai = data[data.variable.eq("LAI")]
assert figure5_lai.DVS.notna().all()
dvs_ticks = np.arange(0, 2.01, 0.5)
sites = ["NLD", "IND"]
organ_styles = {"FLTB": ("Leaf", PALETTE["green"], "-"),
                "FSTB": ("Stem", PALETTE["orange"], "--"),
                "FOTB": ("Tuber", PALETTE["blue"], ":")}
fig, axes = plt.subplots(2, 2, figsize=(5.5, 5), sharex=True, sharey="row")
for c, site in enumerate(sites):
    season = "2025" if site == "NLD" else "2024-2025"
    ax = axes[0, c]
    for table, (label, color, linestyle) in organ_styles.items():
        curve = allocation[allocation.site.eq(site) & allocation.table.eq(table)
                           & allocation.variant.eq("calibrated")].sort_values("dvs")
        assert not curve.empty
        ax.plot(curve.dvs, curve.value, color=color, linestyle=linestyle,
                linewidth=1, label=label)
    ax.set_xlim(0, 2)
    ax.set_xticks(dvs_ticks)
    ax.tick_params(axis="x", labelbottom=True)
    ax.set_ylim(0, 1)
    ax.set_xlabel("DVS")
    if c == 0:
        ax.set_ylabel("Fraction of newly produced\nnon-root dry matter")
    span = span_parameters.loc[site]
    dataset = f"IND{season.split('-')[0]}" if site == "IND" else f"NLD{season}"
    ax.set_title(f"{dataset}\n"
                 f"SPAN: fitted {span.calibrated:.1f}; default {span.default:.1f}", fontsize=8)
    ax = axes[1, c]
    dsite = figure5_lai[figure5_lai.site.eq(site)]
    for variant, style in (("default", LINE_STYLE["default"]),
                           ("calibrated", LINE_STYLE["after"])):
        d = dsite[dsite.variant.eq(variant)].sort_values("DAE")
        assert not d.empty
        label = "Default" if variant == "default" else "Optimised"
        ax.plot(d.DVS, d.value, label=label, **style, linewidth=0.6)
    obs = dsite[dsite.series_kind.eq("observation")].sort_values("DAE")
    assert not obs.empty
    ax.errorbar(obs.DVS, obs.value, yerr=obs.uncertainty,
                label="Observed", **LINE_STYLE["observed"], linewidth=0.5,
                capsize=2, clip_on=False, zorder=2)
    if c == 0:
        ax.set_ylabel("LAI (m$^2$ m$^{-2}$)")
    ax.set_ylim(0, 1.08 * (figure5_lai.value + figure5_lai.uncertainty.fillna(0)).max())
    ax.set_xlim(0, 2)
    ax.set_xlabel("DVS")
    # DAE labels use the first calibrated day at each DVS; no extrapolation beyond the run.
    time = (dsite[dsite.variant.eq("calibrated")][["DVS", "DAE"]]
            .sort_values("DAE").drop_duplicates("DVS", keep="first"))
    assert time.DVS.is_monotonic_increasing
    time_ticks = dvs_ticks[(dvs_ticks >= time.DVS.min()) & (dvs_ticks <= time.DVS.max())]
    dae_axis = ax.secondary_xaxis(-0.32)
    dae_axis.set_xticks(time_ticks, labels=[
        f"{dae:.0f}" for dae in np.interp(time_ticks, time.DVS, time.DAE)
    ])
    dae_axis.set_xlabel("Days after emergence")
for ax, panel_label in zip(axes.flat, "abcd"):
    ax.text(0.02, 0.98, f"({panel_label})", transform=ax.transAxes,
            ha="left", va="top", fontsize=8)
allocation_handles, allocation_labels = axes[0, 0].get_legend_handles_labels()
fig.legend(allocation_handles, allocation_labels, loc="upper center",
           bbox_to_anchor=(0.5, 1), ncol=3, frameon=False, fontsize=8)
lai_handles, lai_labels = axes[1, 0].get_legend_handles_labels()
fig.legend(lai_handles, lai_labels, loc="upper center",
           bbox_to_anchor=(0.5, 0.97), ncol=3, frameon=False, fontsize=8)
fig.subplots_adjust(top=0.86, bottom=0.18, wspace=0.10, hspace=0.30)
fig.savefig(MAIN_FIG_DIR / "Figure_05.svg", format="svg", bbox_inches="tight")
plt.show()
print(f"wrote {MAIN_FIG_DIR / 'Figure_05.svg'}")

# %% Figure 5 - retained calibration numbers for all four growth variables
df_figure5_writing_pairs = (
    data.assign(field="OpenField")
    .pivot(index=["site", "season", "field", "variable", "DAE"],
           columns="variant", values="value")
    .reset_index()
)
figure5_number_rows = []
for (site, season, field, variable), df_variable in df_figure5_writing_pairs.groupby(
        ["site", "season", "field", "variable"], sort=False):
    df_variable = df_variable.dropna(subset=["observed", "default", "calibrated"])
    if df_variable.empty:
        print(f"SKIP Figure 5 numbers -- no pairs: {site}/{season}/{field}/{variable}")
        continue
    baseline = nrmse(df_variable["observed"], df_variable["default"])
    optimised = nrmse(df_variable["observed"], df_variable["calibrated"])
    figure5_number_rows.append({
        "dataset": f"{site}{season.split('-')[0] if site == 'IND' else season}",
        "site": site, "season": season, "field": field, "variable": variable,
        "default_NRMSE": round(baseline, 2),
        "calibrated_NRMSE": round(optimised, 2),
        "NRMSE_reduction_percent": round(100 * (baseline - optimised) / baseline, 1),
    })

df_figure5_writing_numbers = pd.DataFrame(figure5_number_rows)
df_figure5_writing_numbers["site"] = pd.Categorical(
    df_figure5_writing_numbers["site"], categories=["NLD", "IND"], ordered=True,
)
df_figure5_writing_numbers = (
    df_figure5_writing_numbers.sort_values(["site", "season", "field", "variable"])
    .drop(columns=["site", "season"]).reset_index(drop=True)
)
df_figure5_writing_numbers.to_csv(NUMBER_DIR / "figure5_writing_numbers.csv", index=False)

duration = pd.read_csv(PHENOLOGY / "photoperiod_duration_1to1_photoperiod_v1.csv")
duration = duration[(duration["site"] == "NLD")
                    & (duration["season"].astype(str) == "2025")
                    & duration["panel"].isin(["P2", "P4"])]
nld_lai = data[(data["site"] == "NLD") & (data["variable"] == "LAI")]
canopy_dae = nld_lai[(nld_lai["variant"] == "calibrated")
                     & (nld_lai["value"] <= 0.01)]
print(f"Canopy decline to LAI <= 0.01 occurs at DAE {canopy_dae['DAE'].min()}")
default_last_dae = nld_lai[nld_lai["variant"] == "default"]["DAE"].max()
df_figure5_nld_timing = pd.DataFrame([{
    "site": "NLD", "season": "2025", "field": "OpenField",
    "observed_maturity_DAE": int(duration["obs_days"].sum()),
    "calibrated_LAI_0_DAE": int(canopy_dae['DAE'].min()),
    "default_end_minus_calibrated_canopy_end_days": int(default_last_dae - canopy_dae['DAE'].min()),
}])

partition_params = pd.read_csv(ROOT / "output" / "pp_growth" / "growth_best_params_v1.csv")
partition_order = ["FLTBa", "FLTBb", "FLTBc", "FLTBd", "FOTBa", "FOTBb", "SPAN"]

df_figure5_partition_parameters = (
    partition_params[partition_params["site"].isin(["NLD", "IND"])
                     & partition_params["layer"].eq("partitioning")
                     & partition_params["param"].isin(partition_order)]
    .set_index(["site", "param"])
    .loc[pd.MultiIndex.from_product([["NLD", "IND"], partition_order]),
         ["season", "value"]]
    .rename(columns={"value": "optimised_value"})
    .reset_index().pivot(index="param", columns= ["site", "season"], values="optimised_value")
    .astype(float)
)
print("\nFigure 5 numbers for manuscript writing")
print(df_figure5_writing_numbers.to_string(index=False))
print("\nNLD 2025 maturity and canopy timing")
print(df_figure5_nld_timing.to_string(index=False))
print("\nNLD and IND optimised partitioning and SPAN parameters")
print(df_figure5_partition_parameters.to_string())

# %% Figure 5 - when net leaf-DW accumulation stopped in NLD versus IND
df_figure5_dw_daily = (
    data.loc[
        data["series_kind"].eq("simulation")
        & data["variant"].eq("calibrated")
        & data["variable"].isin(["WLV", "TWSO"]),
        ["site", "season", "DAE", "variable", "value"],
    ]
    .pivot(index=["site", "season", "DAE"], columns="variable", values="value")
    .reset_index()
    .sort_values(["site", "season", "DAE"])
)
df_figure5_dw_daily["daily_leaf_DW_change_kg_ha"] = (
    df_figure5_dw_daily.groupby(["site", "season"])["WLV"].diff()
)
df_figure5_dw_daily["daily_tuber_DW_change_kg_ha"] = (
    df_figure5_dw_daily.groupby(["site", "season"])["TWSO"].diff()
)

figure5_leaf_dw_stop_rows = []
for (site, season), df_site_season in df_figure5_dw_daily.groupby(
        ["site", "season"], sort=False):
    leaf_stop = df_site_season[
        df_site_season["daily_leaf_DW_change_kg_ha"] > 0
    ].iloc[-1]
    last_tuber_gain = df_site_season[
        df_site_season["daily_tuber_DW_change_kg_ha"] > 0
    ].iloc[-1]
    simulation_end = df_site_season.iloc[-1]
    tuber_gain_after_leaf_stop = simulation_end["TWSO"] - leaf_stop["TWSO"]
    figure5_leaf_dw_stop_rows.append({
        "site": site,
        "season": season,
        "net_leaf_DW_gain_stop_DAE": int(leaf_stop["DAE"]),
        "leaf_DW_at_stop_kg_ha": leaf_stop["WLV"],
        "tuber_DW_at_leaf_stop_kg_ha": leaf_stop["TWSO"],
        "last_tuber_DW_gain_DAE": int(last_tuber_gain["DAE"]),
        "simulation_end_DAE": int(simulation_end["DAE"]),
        "simulation_end_tuber_DW_kg_ha": simulation_end["TWSO"],
        "days_tuber_DW_continued_after_leaf_stop": int(
            last_tuber_gain["DAE"] - leaf_stop["DAE"]),
        "tuber_DW_gained_after_leaf_stop_kg_ha": tuber_gain_after_leaf_stop,
        "tuber_DW_gained_after_leaf_stop_percent": 100 * (
            tuber_gain_after_leaf_stop / simulation_end["TWSO"]),
    })
df_figure5_leaf_dw_stop = pd.DataFrame(figure5_leaf_dw_stop_rows)

leaf_stop_by_site = df_figure5_leaf_dw_stop.set_index("site")["net_leaf_DW_gain_stop_DAE"]
nld_leaf_dw_stopped_earlier_days = int(leaf_stop_by_site["IND"] - leaf_stop_by_site["NLD"])
print("\nFigure 5 net leaf-DW accumulation evidence")
print("Positive daily WLV change denotes net leaf-DW accumulation; it does not isolate gross allocation from senescence.")
print(df_figure5_leaf_dw_stop.round(1).to_string(index=False))
print(f"NLD stopped net leaf-DW accumulation {nld_leaf_dw_stopped_earlier_days} days earlier than IND.")

# %% Figure 6 - shared NLD data and soil-water reference levels
timeseries_path = SOURCE_DIR / "s13_s14_nld_timeseries.csv"
retention_path = SOURCE_DIR / "s11_nld_retention_curves.csv"
figure6_series = pd.read_csv(timeseries_path, dtype={"season": str})
figure6_retention = pd.read_csv(retention_path)
figure6_seasons = ["2025"]
figure6_treatments = ["AC", "HTI"]
figure6_depths = {0: 20.0, 1: 40.0}
figure6_colors = HEATING_COLOR
figure6_linestyles = HEATING_LINE
figure6_markers = HEATING_MARKER
figure6_dodge = {"AC": -0.6, "HTI": 0.6}
figure6_obs_style = dict(
    linestyle="none", markeredgewidth=0.5,
    markersize=3, zorder=10,
)

figure6_series = figure6_series[
    figure6_series["site"].eq("NLD")
    & figure6_series["season"].isin(figure6_seasons)
    & figure6_series["treatment"].isin(figure6_treatments)
].copy()
figure6_sim = figure6_series[
    figure6_series["soil"].eq("FC corrected 3-layer")
    & figure6_series["series_kind"].eq("simulation")
    & figure6_series["variable"].isin(["SM_L0", "SM_L1", "TWSO"])
].pivot(index=["season", "treatment", "day", "DAE"],
        columns="variable", values="value").reset_index()
figure6_sim.columns.name = None

figure6_obs = figure6_series[
    figure6_series["series_kind"].eq("observation")
    & (figure6_series["variable"].eq("TWSO")
       | (figure6_series["variable"].eq("VWC")
          & figure6_series["depth"].isin(figure6_depths.values())))
].copy()
figure6_levels = {0: [], 1: []}
for layer, depth in figure6_depths.items():
    layer_curve = figure6_retention[
        (figure6_retention["site"] == "NLD")
        & (figure6_retention["geometry"] == "3layer")
        & (figure6_retention["soil"] == "FC corrected")
        & np.isclose(figure6_retention["layer_top"], depth - 20.0)
        & np.isclose(figure6_retention["layer_bottom"], depth)
    ]
    for level_name, pf in (("SM0", -1.0), ("FC", 2.0), ("WP", 4.2)):
        level = layer_curve[np.isclose(layer_curve["pF"], pf)]
        figure6_levels[layer].append((level_name, level["vwc"].iloc[0]))

# %% Figure 6 - two VWC panels beside one full-height tuber dry-weight panel
fig = plt.figure(figsize=(6, 3))
grid = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=0.12, wspace=0.35)
ax_vwc_top = fig.add_subplot(grid[0, 0])
ax_vwc_bottom = fig.add_subplot(grid[1, 0], sharex=ax_vwc_top, sharey=ax_vwc_top)
ax_twso = fig.add_subplot(grid[:, 1], sharex=ax_vwc_top)
vwc_axes = [ax_vwc_top, ax_vwc_bottom]
season = figure6_seasons[0]
for treatment in figure6_treatments:
    sim_treatment = figure6_sim[
        figure6_sim.season.eq(season) & figure6_sim.treatment.eq(treatment)
    ]
    obs_treatment = figure6_obs[
        figure6_obs.season.eq(season) & figure6_obs.treatment.eq(treatment)
    ]
    line_style = dict(
        color=figure6_colors[treatment],
        linestyle=figure6_linestyles[treatment], linewidth=0.6,
    )
    obs_style = dict(
        color=figure6_colors[treatment], marker=figure6_markers[treatment],
        markerfacecolor=figure6_colors[treatment] if HEATING_FILLED[treatment] else "white",
        **figure6_obs_style,
    )
    for layer, depth in figure6_depths.items():
        obs_layer = obs_treatment[
            obs_treatment.variable.eq("VWC") & obs_treatment.depth.eq(depth)
        ]
        end_dae = min(int(sim_treatment.DAE.max()), int(obs_layer.DAE.max()))
        sim_layer = sim_treatment[sim_treatment.DAE <= end_dae]
        obs_layer = obs_layer[obs_layer.DAE <= end_dae]
        vwc_axes[layer].plot(sim_layer.DAE, sim_layer[f"SM_L{layer}"], **line_style)
        vwc_axes[layer].plot(obs_layer.DAE, obs_layer.value, **obs_style)
    tuber_obs = obs_treatment[obs_treatment.variable.eq("TWSO")]
    ax_twso.plot(sim_treatment.DAE, sim_treatment.TWSO, **line_style)
    ax_twso.errorbar(
        tuber_obs.DAE + figure6_dodge[treatment], tuber_obs.value,
        yerr=tuber_obs.SD, capsize=2, elinewidth=0.5, capthick=0.5,
        clip_on=False, **obs_style,
    )
for layer, layer_levels in figure6_levels.items():
    for level_name, level_vwc in layer_levels:
        vwc_axes[layer].axhline(level_vwc, color="0.5", linestyle=":", linewidth=0.8, zorder=0)
        vwc_axes[layer].text(
            0.99, level_vwc, "Saturation" if level_name == "SM0" else level_name,
            transform=vwc_axes[layer].get_yaxis_transform(),
            ha="right", va="bottom", color="0.5",
        )
ax_vwc_top.set_ylabel("VWC, 0-20 cm\n(cm$^3$ cm$^{-3}$)")
ax_vwc_bottom.set_ylabel("VWC, 20-40 cm\n(cm$^3$ cm$^{-3}$)")
ax_twso.set_ylabel("Tuber dry weight (kg ha$^{-1}$)")
ax_vwc_top.set_xlim(0, 105)
ax_vwc_top.set_ylim(0, 0.5)
ax_vwc_top.tick_params(labelbottom=False)
ax_vwc_bottom.set_xlabel("Days after emergence")
ax_twso.set_xlabel("Days after emergence")
ax_twso.set_ylim(0, 15500)
ax_twso.set_yticks(np.arange(0, 15500, 3000))
for ax, label in zip([ax_vwc_top, ax_vwc_bottom, ax_twso], "abc"):
    ax.text(0.01, 0.99, f"({label})", transform=ax.transAxes, ha="left", va="top")
handles = [ax_vwc_top.plot(
    [], [], color=figure6_colors[treatment],
    linestyle=figure6_linestyles[treatment], marker=figure6_markers[treatment],
    markerfacecolor=figure6_colors[treatment] if HEATING_FILLED[treatment] else "white",
    markeredgewidth=0.9, markersize=6, label=treatment,
)[0] for treatment in figure6_treatments]
handles.append(ax_vwc_top.plot(
    [], [], color="0.5", linestyle=":", linewidth=0.8,
    label="Soil hydraulic properties",
)[0])
fig.legend(handles, [*figure6_treatments, "Soil hydraulic properties"], loc="upper center",
           ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08), handlelength=4)
fig.subplots_adjust(left=0.12, right=0.99, bottom=0.19, top=0.95)
fig.savefig(MAIN_FIG_DIR / "Figure_06.svg", format="svg", bbox_inches="tight")
plt.show()
print(f"wrote {MAIN_FIG_DIR / 'Figure_06.svg'}")
# %% Figure 6 - seasonal water totals and layer contributions
figure6_water = figure6_series[
    (figure6_series["soil"] == "FC corrected 3-layer")
    & (figure6_series["series_kind"] == "simulation")
    & figure6_series["variable"].isin(
        ["ET0", "RAIN", "IRRIGATION", "TRAMX", "TRA",
         "TRALY_L0", "TRALY_L1", "TRALY_L2"])
].copy()
df_figure6_water = (figure6_water.groupby(
    ["season", "treatment", "variable"])["value"].sum().mul(10).unstack()
    .rename(columns={"ET0": "ET0_mm", "RAIN": "rain_mm",
                     "IRRIGATION": "irrigation_mm",
                     "TRAMX": "TRAMX_mm", "TRA": "TRA_mm",
                     "TRALY_L0": "TRA_0_20_mm", "TRALY_L1": "TRA_20_40_mm",
                     "TRALY_L2": "TRA_40_60_mm"}).reset_index())
df_figure6_water['supply'] = df_figure6_water['rain_mm'] + df_figure6_water['irrigation_mm']
for depth in ("0_20", "20_40", "40_60"):
    df_figure6_water[f"TRA_{depth}_pct"] = (
        100 * df_figure6_water[f"TRA_{depth}_mm"] / df_figure6_water["TRA_mm"])
df_figure6_water = df_figure6_water[[
    "season", "treatment", "rain_mm", "irrigation_mm",
    "supply",
    "ET0_mm", "TRAMX_mm", "TRA_mm",
    "TRA_0_20_mm", "TRA_20_40_mm", "TRA_40_60_mm",
    "TRA_0_20_pct", "TRA_20_40_pct", "TRA_40_60_pct"
]]
print("\nFigure 6 seasonal water totals and layer contributions")
print(df_figure6_water.round(1).to_string(index=False))
# %% Figure 7 - measured and simulated final IND tuber yield
# IND2024: FC+WP soil, default versus fitted CFET; PERDL stays at its crop default.
df_rank = pd.read_csv(WLD_IND / "opt_yield_order_ind_2024-2025.csv")
df_rank = df_rank[(df_rank["location"] == "IND") & (df_rank["season"].astype(str) == "2024-2025")].copy()

df_manuscript_markers = pd.DataFrame([
    {"treatment": treatment, "marker": TREATMENT_MARKER[treatment]}
    for treatment in TREATMENT_ORDER
])

df_manuscript_panels = pd.DataFrame([
    {"comparison": "layered, crop default", "panel": "a", "panel_order": 0,
     "panel_title": f"(a) 2024: FC + WP corrected soil"},
    {"comparison": "layered, CFET_fit", "panel": "b", "panel_order": 1,
     "panel_title": f"(b) 2024: FC + WP soil + fitted CFET, default PERDL"},
])

df_manuscript_twso = df_rank.loc[
    df_rank["version"].isin(["layered, crop default", "layered, CFET_fit"]),
    ["location", "season", "version", "treatment", "day",
     "TWSO_obs", "TWSO_obs_sd", "TWSO_sim", "CFET", "PERDL"],
].rename(columns={"version": "comparison"})
df_manuscript_twso = (
    df_manuscript_twso
    .merge(df_manuscript_panels, on="comparison", how="left", validate="many_to_one")
    .merge(df_manuscript_markers, on="treatment", how="left", validate="many_to_one")
)
df_manuscript_twso["treatment_order"] = df_manuscript_twso["treatment"].map(
    dict(zip(TREATMENT_ORDER, range(len(TREATMENT_ORDER))))
)
df_manuscript_twso = (
    df_manuscript_twso
    .sort_values(["panel_order", "treatment_order"])
    .reset_index(drop=True)
)

df_manuscript_results = df_manuscript_twso.copy()

_manuscript_values = pd.concat([
    df_manuscript_results["TWSO_obs"],
    df_manuscript_results["TWSO_sim"],
]).dropna() / 1000.0
MANUSCRIPT_LIM = [0, 10]

spearman_rows = []
for (panel, comparison), d in df_manuscript_results.groupby(
        ["panel", "comparison"], sort=False):
    valid = d[["TWSO_obs", "TWSO_sim"]].dropna()
    spearman_rows.append({
        "panel": panel,
        "season": d["season"].iloc[0],
        "comparison": comparison,
        "n_pairs": len(valid),
        "spearman_rho": valid["TWSO_obs"].corr(valid["TWSO_sim"], method="spearman"),
    })
df_manuscript_spearman = pd.DataFrame(spearman_rows)

with plt.rc_context({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "svg.fonttype": "none",
                     "figure.dpi": 300, "axes.grid": False}):
    # IND publication colours; look up by treatment rather than dataframe columns.
    figure7_amount_color = {"Full": "#1f78b4", "Reduced": "#E31A1C"}
    fig, axes = plt.subplots(1, 2, figsize=(5, 3), sharex=True, sharey=True)

    for ax, (_, panel_row) in zip(axes.flat,
                                  df_manuscript_panels.sort_values("panel_order").iterrows()):
        d = df_manuscript_twso[
            df_manuscript_twso["panel"] == panel_row["panel"]
        ]
        ax.plot(MANUSCRIPT_LIM, MANUSCRIPT_LIM, color=REF_COLOR, linestyle="--",
                linewidth=1.0, zorder=0)
        for _, r in d.iterrows():
            ax.errorbar(r["TWSO_obs"] / 1000.0, r["TWSO_sim"] / 1000.0,
                        xerr=r["TWSO_obs_sd"] / 1000.0,
                        fmt=r["marker"], color=figure7_amount_color[r["treatment"].split("_")[0]],
                        markerfacecolor="white", markeredgecolor=figure7_amount_color[r["treatment"].split("_")[0]],
                        markeredgewidth=0.9, capsize=2,
                        linewidth=0.4, label=r["treatment"], zorder=2)
        ax.text(0.03, 0.97, f"({panel_row['panel']})", transform=ax.transAxes,
                ha="left", va="top")
        ax.set_xlim(*MANUSCRIPT_LIM)
        ax.set_ylim(*MANUSCRIPT_LIM)
        ax.set_aspect("equal")
        params = d.iloc[0]
        ax.text(0.97, 0.03, f"CFET={params['CFET']:.2f}\nPERDL={params['PERDL']:.2f}",
                transform=ax.transAxes, ha="right", va="bottom")

    for ax in axes:
        ax.set_xlabel("Observed tuber yield (t ha$^{-1}$)")
    axes[0].set_ylabel("Simulated tuber yield (t ha$^{-1}$)")
    legend_treatments = [
        "Full_0 Days", "Reduced_0 Days",
        "Full_6 Days", "Reduced_6 Days",
        "Full_12 Days", "Reduced_12 Days",
    ]
    df_manuscript_legend = (
        df_manuscript_markers.set_index("treatment")
        .loc[legend_treatments]
        .reset_index()
    )
    legend_handles = []
    for _, r in df_manuscript_legend.iterrows():
        legend_handles.append(plt.Line2D(
            [], [], marker=r["marker"], linestyle="none", color=figure7_amount_color[r["treatment"].split("_")[0]],
            markerfacecolor="white", markeredgecolor=figure7_amount_color[r["treatment"].split("_")[0]],
            markeredgewidth=0.9, markersize=6,
            label=r["treatment"].replace("_", " "),
        ))
    fig.legend(handles=legend_handles, loc="outside upper center", ncol=3,
               frameon=False)
    fig.subplots_adjust(top=0.85, left=0.12, right=0.98, bottom=0.19,
                        wspace=0)

    fig.savefig(MAIN_FIG_DIR / "Figure_07.svg", format="svg",
                dpi=300, bbox_inches="tight")
    plt.show()

print("\nSpearman ranking for manuscript TWSO panels:")
print(df_manuscript_spearman.to_string(index=False,
      formatters={"spearman_rho": lambda x: f"{x:+.2f}"}))