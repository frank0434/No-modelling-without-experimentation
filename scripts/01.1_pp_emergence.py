# %% Shared paths and Figure 1 response rules
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next(p for p in (_here, *_here.parents)
            if (p / "scripts").is_dir() and (p / "data").is_dir())
sys.path.insert(0, str(ROOT / "scripts"))
SOURCE_DIR = ROOT / "manuscript_outputs" / "source_data"
PHENOLOGY_DIR = ROOT / "output" / "phenology"
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
PHENOLOGY_DIR.mkdir(parents=True, exist_ok=True)

TBASEM = 3.0  # Crop default in data/crops_calibration/potato.yaml.
RULES = ["clip@18", "clip@25", "Kooman"]
MARCH_BUFFER = 30


def apply_rule(rule, temperature):
    temperature = np.asarray(temperature, dtype=float)
    if rule == "Kooman":
        return np.interp(
            temperature, [2.0, 20.0, 25.0, 35.0], [0.0, 0.9, 1.0, 0.0],
            left=0.0, right=0.0,
        ) * (25.0 - TBASEM)
    ceiling = 18.0 if rule == "clip@18" else 25.0
    return np.clip(temperature - TBASEM, 0.0, ceiling - TBASEM)


# %% Shared planting-to-emergence windows
from utils.loaders import load_obs_experiments

df_windows = pd.DataFrame([
    ("CAN", "1992", "OpenField", "Kennebec", "CAN_QC_7016932_1992_P1D.csv"),
    ("CAN", "1993", "OpenField", "Kennebec", "CAN_QC_7016932_1993_P1D.csv"),
    ("NLD", "2025", "OpenField", "Innovator", "WUR2025_wofost.xlsx"),
    ("IND", "2022-2023", "OpenField", "Kennebec", "India2022_23_wofost.xlsx"),
    ("IND", "2024-2025", "OpenField", "Kennebec", "India2024_25_wofost.xlsx"),
    ("RSA", "2025-A", "OpenField", "Mondial", "SouthAfrica2025_wofost.xlsx"),
    ("RSA", "2025-S", "OpenField", "Mondial", "SouthAfrica2025_2_wofost.xlsx"),
    ("RSA", "2025-A", "RainShelter", "Mondial", "SouthAfrica2025_wofost.xlsx"),
    ("RSA", "2025-S", "RainShelter", "Mondial", "SouthAfrica2025_2_wofost.xlsx"),
], columns=["site", "season", "field", "cultivar", "weather_file"])

observed = load_obs_experiments()[
    ["location", "season_key", "field", "cultivar", "Start", "Emergence"]
].rename(columns={"location": "site", "season_key": "season"})
observed["season"] = observed["season"].astype(str)
observed["field"] = observed["field"].replace("Rainshelter", "RainShelter")
df_windows = df_windows.merge(
    observed.drop_duplicates(), on=["site", "season", "field", "cultivar"], how="left",
)

# CAN literature dates are DOY 136 and 164 in both years; calendar dates differ in leap years.
can = df_windows["site"].eq("CAN")
year_start = pd.to_datetime(df_windows.loc[can, "season"] + "-01-01")
df_windows.loc[can, "Start"] = year_start + pd.Timedelta(days=135)
df_windows.loc[can, "Emergence"] = year_start + pd.Timedelta(days=163)
df_windows["Start"] = pd.to_datetime(df_windows["Start"])
df_windows["Emergence"] = pd.to_datetime(df_windows["Emergence"])
print(df_windows[["site", "season", "field", "Start", "Emergence"]].to_string(index=False))


# %% Figure 1: daily air temperature and observed thermal requirements
daily_parts = []
requirement_rows = []
for row in df_windows.loc[df_windows["site"].isin(["CAN", "IND", "RSA"])].itertuples(index=False):
    weather_path = ROOT / "data" / "Weather" / row.weather_file
    if row.site == "CAN":
        weather = pd.read_csv(weather_path, parse_dates=["Date/Time"])
        weather = weather.rename(columns={
            "Date/Time": "day", "Min Temp (°C)": "TMIN", "Max Temp (°C)": "TMAX",
        })
    else:
        weather = pd.read_excel(weather_path, header=10, skiprows=[11])
        weather = weather.rename(columns={"DAY": "day"})
    days = pd.date_range(row.Start, row.Emergence + pd.Timedelta(days=MARCH_BUFFER))
    weather = weather.set_index("day").reindex(days)
    temperature = (weather["TMIN"] + weather["TMAX"]) / 2.0
    daily = pd.DataFrame({
        "site": row.site, "season": row.season, "field": row.field,
        "day": days, "air": temperature.to_numpy(),
    })
    daily_parts.append(daily)
    for rule in RULES:
        thermal_time = apply_rule(rule, daily.loc[daily["day"] < row.Emergence, "air"])
        requirement_rows.append({
            "site": row.site, "season": row.season, "field": row.field,
            "source": "air", "rule": rule,
            "accumulated": np.nan if np.isnan(thermal_time).any() else thermal_time.sum(),
        })

df_days = pd.concat(daily_parts, ignore_index=True)
df_requirements = pd.DataFrame(requirement_rows)


# %% Figure 1: population CV and leave-one-season-out emergence error
keys = ["site", "field", "source", "rule"]
scored = df_requirements.dropna(subset=["accumulated"])
df_consistency = scored.groupby(keys)["accumulated"].agg(
    n_seasons="size", mean="mean", sd=lambda values: values.std(ddof=0),
).reset_index()
df_consistency["cv"] = df_consistency["sd"] / df_consistency["mean"]

loo_rows = []
for (site, field, source, rule), group in scored.groupby(keys):
    for index, row in group.iterrows():
        target = group.drop(index=index)["accumulated"].mean()
        daily = df_days[
            df_days["site"].eq(site) & df_days["season"].eq(row["season"])
            & df_days["field"].eq(field)
        ]
        cumulative = np.cumsum(apply_rule(rule, daily["air"]))
        hits = np.flatnonzero(cumulative >= target)
        # Preserve Figure 1's convention: the weather day reaching the target is the predicted date.
        predicted = daily["day"].iloc[hits[0]] if len(hits) else pd.NaT
        emergence = df_windows.loc[
            df_windows["site"].eq(site) & df_windows["season"].eq(row["season"])
            & df_windows["field"].eq(field), "Emergence",
        ].iloc[0]
        loo_rows.append({
            "site": site, "season": row["season"], "field": field,
            "source": source, "rule": rule, "target": target, "predicted": predicted,
            "days_off": np.nan if pd.isna(predicted) else (predicted - emergence).days,
        })

df_loo = pd.DataFrame(loo_rows)
mean_errors = df_loo.assign(abs_off=df_loo["days_off"].abs()).groupby(keys)["abs_off"].mean()
df_consistency = df_consistency.merge(mean_errors.reset_index(), on=keys)
df_consistency.to_csv(SOURCE_DIR / "fig01_emergence_consistency.csv", index=False)
print(df_consistency.round(3).to_string(index=False))


# %% Figure S6: response curves for the same three rules
temperature_grid = np.linspace(0, 35, 351)
df_response_curves = pd.concat([
    pd.DataFrame({"temperature": temperature_grid, "rule": rule,
                  "effective_thermal_time": apply_rule(rule, temperature_grid)})
    for rule in RULES
], ignore_index=True)
df_response_curves.to_csv(SOURCE_DIR / "s06_emergence_response_curves.csv", index=False)


# %% Figure S4: Sobol inputs and daily emergence equation
from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze
from utils.loaders import load_weather

MAX_DAYS = 90
PROBLEM = {
    "num_vars": 3,
    "names": ["TSUMEM", "TBASEM", "TEFFMX"],
    "bounds": [[170.0, 320.0], [2.0, 4.0], [18.0, 32.0]],
}
N_SAMPLES = 2**11  # N*(D+2) = 10240 evaluations per crop for S1 and ST.
sobol_windows = df_windows[
    df_windows["site"].isin(["CAN", "NLD", "IND", "RSA"]) & df_windows["field"].eq("OpenField")
]


def days_to_emergence(tavg, tsumem, tbasem, teffmx):
    dvs = -0.1
    cap = teffmx - tbasem
    for day, t in enumerate(tavg, start=1):
        dtsume = min(max(t - tbasem, 0.0), cap)
        dvs += 0.1 * dtsume / tsumem
        if dvs >= 0.0:
            return day
    return None


# %% Figure S4: first-order and total-order indices with confidence intervals
# Preserve the planting+1 temperature window and original absence of an explicit random seed.
main_rows = []
for row in sobol_windows.itertuples(index=False):
    location, season, cultivar = row.site, row.season, row.cultivar
    days = pd.date_range(row.Start + pd.Timedelta(days=1), periods=MAX_DAYS, freq="D")
    if location == "CAN":
        weather = pd.read_csv(ROOT / "data" / "Weather" / row.weather_file,
                              parse_dates=["Date/Time"]).set_index("Date/Time")
        tavg = ((weather.loc[days, "Min Temp (°C)"]
                 + weather.loc[days, "Max Temp (°C)"]) / 2.0).to_numpy()
    else:
        wdp = load_weather(f"data/Weather/{row.weather_file}")
        tavg = np.array([(wdp(d.date()).TMIN + wdp(d.date()).TMAX) / 2.0 for d in days])

    param_values = sobol_sample.sample(PROBLEM, N_SAMPLES, calc_second_order=False)
    Y = np.empty(param_values.shape[0])
    n_censored = 0
    for i, (tsumem, tbasem, teffmx) in enumerate(param_values):
        d = days_to_emergence(tavg, tsumem, tbasem, teffmx)
        if d is None:
            d = MAX_DAYS
            n_censored += 1
        Y[i] = d

    if n_censored:
        print(f"[{location} {season}] WARNING: {n_censored}/{len(Y)} samples did not reach "
              f"emergence within {MAX_DAYS} days (censored at {MAX_DAYS}) -- Sobol indices below "
              f"are biased by this.")

    Si = sobol_analyze.analyze(PROBLEM, Y, calc_second_order=False, print_to_console=False)
    for j, name in enumerate(PROBLEM["names"]):
        main_rows.append({
            "location": location, "season": season, "cultivar": cultivar, "param": name,
            "S1": Si["S1"][j], "S1_conf": Si["S1_conf"][j],
            "ST": Si["ST"][j], "ST_conf": Si["ST_conf"][j],
        })

main_df = pd.DataFrame(main_rows).round(4)
main_df.to_csv(PHENOLOGY_DIR / "emergence_sobol_v1_main_effects.csv", index=False)
print(main_df.to_string(index=False))


# %% Figure S5: uncapped-TSUMEM identifiability inputs
# This profile assumption is separate from the TEFFMX=25 C production calculation.
from utils.visualise import dataset_code
from utils.phenology_helpers import (
    tavg_window, obs_dates, site_dtsmtb, derive_obs_fixed, ind_maturity_dvs,
)
from optimization.config import OptimizationConfig, load_param_specs
from optimization.resolver import ParameterSetResolver
from optimization.selector import ExperimentSelector
from optimization.runner import PhenologySimulationRunner
from optimization.objective import PhenologyObjective

CROP_DIR = "data/crops_calibration"
TEFFMX_GRID = np.arange(18.0, 32.01, 0.25)
profile_windows = df_windows[df_windows["site"].isin(["IND", "RSA"])]

FIXED, window_rows = {}, []
for row in profile_windows.itertuples(index=False):
    site, season, field, cultivar = row.site, row.season, row.field, row.cultivar
    FIXED[(site, season, field)] = derive_obs_fixed(site, cultivar, [season], np.inf, field=field)
    d = obs_dates(site, season, cultivar, field)
    temps = tavg_window(site, season, cultivar, d["Start"], d["Emergence"])
    window_rows.append({"site": site, "season": season, "field": field,
                        "code": dataset_code(site, season, field),
                        "max_recorded_tavg": max(temps)})

window_df = pd.DataFrame(window_rows)
window_df.to_csv(PHENOLOGY_DIR / "teffmx_identifiability_teffmx_identifiability_v1.csv", index=False)


# %% Figure S5: calibration error over the TEFFMX grid
_teffmx_spec = load_param_specs(["TEFFMX"])
profile_rows = []
for row in profile_windows.itertuples(index=False):
    site, season, field, cultivar = row.site, row.season, row.field, row.cultivar
    fixed = FIXED[(site, season, field)]

    config = OptimizationConfig(
        target=f"{site}-{cultivar}-teffmx-check-{season}-{field}", location=site,
        cultivar=cultivar, seasons=[season], obs_field=field, active_params=_teffmx_spec,
        fixed_overrides=fixed, crop_dir=CROP_DIR, cache=False,
    )
    selector = ExperimentSelector(config)
    objective = PhenologyObjective(config, ParameterSetResolver(config),
                                   PhenologySimulationRunner(CROP_DIR), selector)

    if site == "IND":
        # IND's observed Maturity is a pre-maturity haulm kill; adjust before scoring.
        theo_dvs = ind_maturity_dvs(season, cultivar, fixed["TSUM1"], fixed["TSUM2"],
                                   site_dtsmtb(site, season, cultivar))
        df_obs = selector.obs()
        df_obs["obs_dvs"] = df_obs["obs_dvs"].astype(float)
        df_obs.loc[df_obs["obs_dvs"] == 2.0, "obs_dvs"] = theo_dvs

    for v in TEFFMX_GRID:
        rec = objective.evaluate(np.array([v]))
        profile_rows.append({"site": site, "season": season, "field": field, "TEFFMX": v,
                             "RMSE": rec["rmse"], "MAE": rec["mae"]})

profile_df = pd.DataFrame(profile_rows)
profile_df.to_csv(SOURCE_DIR / "s05_teffmx_profiles.csv", index=False)
print(f"{len(profile_windows)} datasets x {len(TEFFMX_GRID)} grid points = {len(profile_df)} runs.")
