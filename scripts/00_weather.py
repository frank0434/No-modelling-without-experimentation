# %%
# Weather diagnostics: visualise the WOFOST weather inputs themselves (not
# simulation output). Four temperature series per site x season:
#   Tmin, Tmax          - straight from the weather file
#   Daily mean (TEMP)   - (TMIN+TMAX)/2 if not already in the file
#   Daytime mean (DTEMP)- (TEMP+TMAX)/2, WOFOST's own daytime-temperature
#                         approximation used by the assimilation routines
#                         (pcse.engine.Engine._get_driving_variables)

import sys
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.lines as mlines

_here = pathlib.Path().resolve()
sys.path.insert(0, str(next(p for p in (_here, *_here.parents)
                            if (p/"scripts").exists() and (p/"data").exists()) / "scripts"))
from pcse.util import daylength

from utils.plotstyle import display_season
from utils.paths import ROOT
from utils.loaders import load_weather, load_obs_experiments
from utils.phenology_helpers import rsa_tuber_init, RSA_TI_SEASON
from utils.visualise import dataset_code

# Same site/season -> file mapping as 00_summarise_obs.py; 2023-2024 IND
# season is excluded there too (dropped from the analysis).
WEATHER_SOURCES = [
    {"path": "India2022_23_wofost.xlsx",      "location": "IND", "season": "2022-2023"},
    {"path": "India2024_25_wofost.xlsx",      "location": "IND", "season": "2024-2025"},
    {"path": "WUR2024_wofost.xlsx",           "location": "NLD", "season": "2024"},
    {"path": "WUR2025_wofost.xlsx",           "location": "NLD", "season": "2025"},
    {"path": "SouthAfrica2024_wofost.xlsx",   "location": "RSA", "season": "2024"},
    {"path": "SouthAfrica2025_wofost.xlsx",   "location": "RSA", "season": "2025-A"},
    {"path": "SouthAfrica2025_2_wofost.xlsx", "location": "RSA", "season": "2025-S"},
    {"path": "CAN_QC_7016932_1992_P1D.csv", "location": "CAN", "season": "1992"},
    {"path": "CAN_QC_7016932_1993_P1D.csv", "location": "CAN", "season": "1993"},
]
WEATHER_DIR = "data/Weather"
FIGURES_DIR = ROOT / "figures" / "weather"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

VARIABLE_LABELS = {"TMIN": "Tmin", "TMAX": "Tmax", "TEMP": "Daily mean", "DTEMP": "Daytime mean"}
COLOR_MAP = {"Tmax": "orange", "Tmin": "lightblue", "Daily mean": "black", "Daytime mean": "red"}

# %%
# -- Build a long dataframe: location, season, day, variable, Temperature -----
# CAN planting -> harvest, inclusive; also used in the phenology export below.
CAN_CROP_WINDOWS = {
    "1992": (pd.Timestamp("1992-05-15"), pd.Timestamp("1992-09-16")),
    "1993": (pd.Timestamp("1993-05-16"), pd.Timestamp("1993-09-13")),
}
# %%

frames = []
wide_frames = []
for spec in WEATHER_SOURCES:
    if spec["location"] == "CAN":
        weather_CAN = pd.read_csv(
            ROOT / WEATHER_DIR / spec["path"], parse_dates=["Date/Time"],
        )
        weather_CAN["day"] = weather_CAN["Date/Time"]
        weather_CAN["TMIN"] = weather_CAN["Min Temp (°C)"]
        weather_CAN["TMAX"] = weather_CAN["Max Temp (°C)"]
        df = weather_CAN.loc[:, ["day", "TMIN", "TMAX"]].copy()
        start, end = CAN_CROP_WINDOWS[spec["season"]]
        df = df[df["day"].between(start, end)].copy()
        latitude = float(weather_CAN["Latitude (y)"].iloc[0])
    else:
        wdp = load_weather(f"{WEATHER_DIR}/{spec['path']}")
        df = pd.DataFrame(wdp.export())
        df["day"] = pd.to_datetime(df["DAY"])
        latitude = float(wdp.latitude)
    if "TEMP" not in df.columns:
        df["TEMP"] = (df["TMIN"] + df["TMAX"]) / 2.0
    df["DTEMP"] = (df["TEMP"] + df["TMAX"]) / 2.0
    df["DAYL"] = [daylength(d.date(), latitude) for d in df["day"]]
    df["location"] = spec["location"]
    df["season"] = spec["season"]
    wide_frames.append(df[["location", "season", "day", "TMIN", "TMAX", "TEMP", "DTEMP", "DAYL"]])
    long = df.melt(
        id_vars=["location", "season", "day"],
        value_vars=list(VARIABLE_LABELS),
        var_name="variable",
        value_name="Temperature",
    )
    long["variable"] = long["variable"].map(VARIABLE_LABELS)
    frames.append(long)

df_weather = pd.concat(frames, ignore_index=True)

df_weather_wide = pd.concat(wide_frames, ignore_index=True)
_manuscript_source = ROOT / "manuscript_outputs" / "source_data"
_manuscript_source.mkdir(parents=True, exist_ok=True)
(df_weather.rename(columns={"location": "site", "Temperature": "value"})[
    ["site", "season", "day", "variable", "value"]
 ].to_csv(_manuscript_source / "s03_weather_temperature.csv", index=False))
print(f"Built weather series: {len(df_weather)} rows across "
      f"{df_weather['location'].nunique()} locations, {df_weather['season'].nunique()} site-seasons")

# %%
# -- Plot: rows = locations, cols = seasons, one line per temperature series --

loc_order = ["NLD","CAN","IND", "RSA"]
ncols = max(df_weather.groupby("location")["season"].nunique())
fig, axes = plt.subplots(len(loc_order), ncols, figsize=(4.2 * ncols, 3.2 * len(loc_order)), squeeze=False)

for r, loc in enumerate(loc_order):
    loc_data = df_weather[df_weather["location"] == loc]
    seasons = sorted(loc_data["season"].unique())
    for c in range(ncols):
        ax = axes[r][c]
        if c >= len(seasons):
            ax.set_visible(False)
            continue
        season = seasons[c]
        sub = loc_data[loc_data["season"] == season]
        for var, color in COLOR_MAP.items():
            grp = sub[sub["variable"] == var].sort_values("day")
            ax.plot(grp["day"], grp["Temperature"], color=color, linewidth=1, label=var)
        ax.set_title(dataset_code(loc, display_season(season)))
        if c == 0:
            ax.set_ylabel("Temperature (°C)")
        else:
            ax.tick_params(labelleft=False)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    proxy_handles = [mlines.Line2D([], [], color=color, linewidth=2, label=var)
                      for var, color in COLOR_MAP.items()]
    axes[r][min(len(seasons) - 1, ncols - 1)].legend(
        handles=proxy_handles, fontsize=6, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0,
    )

fig.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.94, wspace=0.05, hspace=0.35)
output_path = FIGURES_DIR / "weather_temperature.svg"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Wrote {output_path}")

# %%
# -- Seasonal summary table (publication-ready, for direct paste into Word) ----
# RAIN is cm/day (PCSE convention) -> mm via *10. IRRAD is J/m2/day -> MJ/m2/day via /1e6.
# Rainfall is a season total; temperature and radiation are season means (radiation
# reported as a mean daily rate, not a season total, so sites/seasons of different
# length remain comparable -- flag if a cumulative total is wanted instead).
#
# All columns are computed over the growing window, planting (Start) -> harvest
# (Maturity, i.e. End_Leaf_removal) from the Experiments sheet. The weather records
# run 1-17 days past harvest, and those days are excluded here.
#
# Daytime mean temperature is DTEMP = (TEMP+TMAX)/2, WOFOST's own approximation.
# Day length is the astronomical sunrise-to-sunset value at the site latitude
# (pcse.util.daylength, civil-twilight angle -4 deg), averaged over the observed
# emergence -> tuber-initiation window, (emergence, tuber initiation], the same
# window convention as 02.8_pp_photoperiod_check.py. RSA 2024 uses the first
# tuber-bearing harvest as an upper bound; RSA 2025 uses the whole growing window.
_LOC_ORDER = {"NLD": 0, "IND": 1, "RSA": 2, "CAN": 3}
_pheno = load_obs_experiments()
_pheno = _pheno[_pheno["field"] == "OpenField"].set_index(["location", "season_key"])

# CAN dates follow the observed day-of-year values in 02.7_pp_photoperiod_check.py.
can_frames = []
can_pheno = []
for season, ti_doy in [("1992", 204), ("1993", 198)]:
    df_can = pd.read_csv(
        ROOT / WEATHER_DIR / f"CAN_QC_7016932_{season}_P1D.csv",
        parse_dates=["Date/Time"],
    )
    df_can["day"] = df_can["Date/Time"]
    df_can["TMIN"] = df_can["Min Temp (°C)"]
    df_can["TMAX"] = df_can["Max Temp (°C)"]
    df_can["TEMP"] = (df_can["TMIN"] + df_can["TMAX"]) / 2.0
    df_can["DTEMP"] = (df_can["TEMP"] + df_can["TMAX"]) / 2.0
    df_can["DAYL"] = [
        daylength(d.date(), float(df_can["Latitude (y)"].iloc[0]))
        for d in df_can["day"]
    ]
    df_can["RAIN"] = df_can["Total Rain (mm)"] / 10.0
    df_can["IRRAD"] = float("nan")  # Solar radiation is absent from the CAN files.
    df_can["location"] = "CAN"
    df_can["season"] = season
    can_frames.append(df_can[df_weather_wide.columns])
    can_pheno.append({
        "location": "CAN", "season_key": season,
        "Start": CAN_CROP_WINDOWS[season][0],
        "Emergence": pd.Timestamp(f"{season}-01-01") + pd.Timedelta(days=163),
        "TuberInitiation": pd.Timestamp(f"{season}-01-01") + pd.Timedelta(days=ti_doy - 1),
        "Maturity": CAN_CROP_WINDOWS[season][1],
    })
df_environment = pd.concat([
    df_weather_wide[df_weather_wide["location"] != "CAN"], *can_frames,
], ignore_index=True)
_pheno = pd.concat([
    _pheno, pd.DataFrame(can_pheno).set_index(["location", "season_key"]),
])
_pheno["TI_source"] = "MainTable Experiments"
_pheno.loc[("CAN", slice(None)), "TI_source"] = "CAN literature"
_pheno.loc[("RSA", RSA_TI_SEASON), "TuberInitiation"] = rsa_tuber_init()
_pheno.loc[("RSA", RSA_TI_SEASON), "TI_source"] = "First tuber-bearing harvest (upper bound)"
_pheno.loc[_pheno["TuberInitiation"].isna(), "TI_source"] = ""

# %%
# -- Day length over days after observed emergence ----------------------------
# Keep the observed emergence -> maturity window so all curves use the same
# crop-relative period even though the calendar dates differ among sites.
df_daylength_dae = df_environment.merge(
    _pheno[["Emergence", "TuberInitiation", "TI_source", "Maturity"]].reset_index(),
    left_on=["location", "season"],
    right_on=["location", "season_key"],
    how="inner",
    validate="many_to_one",
)
df_daylength_dae = df_daylength_dae[
    (df_daylength_dae["day"] >= df_daylength_dae["Emergence"])
    & (df_daylength_dae["day"] <= df_daylength_dae["Maturity"])
    
].copy()
df_daylength_dae["DAE"] = (
    df_daylength_dae["day"] - df_daylength_dae["Emergence"]
).dt.days
df_daylength_dae = df_daylength_dae[
    ["location", "season", "day", "Emergence", "TuberInitiation", "TI_source",
     "Maturity", "DAE", "DAYL"]
].sort_values(["location", "season", "DAE"]).reset_index(drop=True)

df_daylength_dae.to_csv(_manuscript_source / "weather_daylength_DAE.csv", index=False)
df_environment.to_csv(_manuscript_source / "weather_environment_daily.csv", index=False)
_pheno[["Start", "Emergence", "TuberInitiation", "TI_source", "Maturity"]].to_csv(
    _manuscript_source / "weather_environment_phenology.csv",
)

# %%
# -- Small single-panel figure: daytime mean temperature, IND 2024-2025 --------
sub = df_weather_wide[(df_weather_wide["location"] == "IND")
                      & (df_weather_wide["season"] == "2024-2025")].sort_values("day")

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(sub["day"], sub["DTEMP"], color=COLOR_MAP["Daytime mean"], linewidth=1)
ax.axhline(25, color="black", linewidth=1, linestyle="--")
ax.set_ylabel("Daytime mean temperature (°C)")
ax.set_title("IND 2024")
ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.tick_params(axis="x", labelrotation=30, labelsize=8)

output_path = FIGURES_DIR / "weather_dtemp_IND_2024-2025.svg"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Wrote {output_path}  (season mean DTEMP = {sub['DTEMP'].mean():.1f} °C)")

# %%
# -- Share of growing-season days above 24 °C ---------------------------------
# Window is planting (Start) -> harvest (Maturity, i.e. End_Leaf_removal) from the
# Experiments sheet, so seasons of different length stay comparable. This differs
# from the summary table above, which counts over the whole weather record (each
# record runs 5-20 days past harvest). Reported for the three temperature
# definitions because the site contrast changes size between them: state which
# variable is meant wherever the number is used.
hot_rows = []
for (loc, season), sub in df_weather_wide.groupby(["location", "season"]):
    start = _pheno.loc[(loc, season), "Start"]
    end = _pheno.loc[(loc, season), "Maturity"]
    win = sub[(sub["day"] >= start) & (sub["day"] <= end)]
    row = {"Location": loc, "Season": season,
           "Planting": start.date(), "Harvest": end.date(), "Season days": len(win)}
    for var in ["DTEMP", "TMAX", "TEMP"]:
        row[f"Days {var} > 24°C"] = (win[var] > 24).sum()
        row[f"% days {var} > 24°C"] = round(100 * (win[var] > 24).mean(), 1)
    hot_rows.append(row)

df_hot = pd.DataFrame(hot_rows).sort_values(
    by=["Location", "Season"], key=lambda s: s.map(_LOC_ORDER) if s.name == "Location" else s
).reset_index(drop=True)
print(df_hot.to_string(index=False))

# Location means (across that location's seasons) and pairwise differences, for the
# summary text. Seasons are weighted equally, not by season length.
pct_cols = [f"% days {v} > 24°C" for v in ["DTEMP", "TMAX", "TEMP"]]
loc_mean = df_hot.groupby("Location")[pct_cols].mean().round(1)
loc_mean = loc_mean.reindex(sorted(loc_mean.index, key=lambda loc: _LOC_ORDER[loc]))
print("\nLocation means (% of growing-season days):")
print(loc_mean.to_string())

print("\nPairwise differences in percentage points:")
for a, b in [("IND", "NLD"), ("RSA", "NLD"), ("IND", "RSA")]:
    diffs = "  ".join(f"{c.split()[2]}: {loc_mean.loc[a, c] - loc_mean.loc[b, c]:+.1f}" for c in pct_cols)
    print(f"{a} - {b}   {diffs}")
