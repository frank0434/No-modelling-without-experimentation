# %% Photoperiod: main four-panel figure and appendix cross-validation
# CAN/IND use the existing paired calibration/evaluation design.
# RSA: autumn thermal references, spring open-field DLO/DLC fits,
# then frozen-parameter evaluation in both rain-shelter seasons.
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pcse.util import daylength
from pcse.base import ParameterProvider, WeatherDataProvider
from pcse.models import Wofost72_Phenology

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next(p for p in (_here, *_here.parents)
            if (p / "scripts").is_dir() and (p / "data").is_dir())
sys.path.insert(0, str(ROOT / "scripts"))
from utils.loaders import (load_experiments, find_experiment, load_weather,
    load_soil, load_site_data, load_crop_data, build_parameters, load_obs_experiments)
from utils.phenology_helpers import obs_dates, site_dtsmtb
from utils.plotstyle import display_season
from manuscript_common import (MAIN_FIG_DIR, SUPP_FIG_DIR, SOURCE_DIR,
    MANUSCRIPT_RC, SITE_MARKER, SITE_COLOR, REF_COLOR)

OUTPUT_DIR = ROOT / "output" / "phenology"
PHASE_DIR = OUTPUT_DIR / "phase_duration_check"
for directory in (OUTPUT_DIR, PHASE_DIR, MAIN_FIG_DIR, SUPP_FIG_DIR, SOURCE_DIR):
    directory.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(MANUSCRIPT_RC)
lw = 0.5

# %% Kennebec: the four existing calibration/evaluation pairs
FACTORIAL_SPLITS = [
    ("A", "2022-2023", "1992", "2024-2025", "1993"),
    ("B", "2024-2025", "1993", "2022-2023", "1992"),
    ("C", "2022-2023", "1993", "2024-2025", "1992"),
    ("D", "2024-2025", "1992", "2022-2023", "1993"),
]
MAIN_SPLIT = "A"  # Fit IND2022/CAN1992; evaluate IND2024/CAN1993.


DLO_GRID = np.round(np.arange(9.0, 15.01, 0.1), 1)
DLC_GRID = np.round(np.arange(12.0, 20.01, 0.1), 1)
DTSMTB = site_dtsmtb("IND", "2024-2025", "Kennebec")  # Crop-default table, not fitted.


# %%
# 1. Four daily datasets, starting at observed emergence.
# CAN literature DOY: Mahdian & Gallichand (1997), as in 02.7.
CAN_DATES = [("1992", 164, 204), ("1993", 164, 198)]
daily, dates = {}, {}
for season in ("2022-2023", "2024-2025"):
    observed = obs_dates("IND", season, "Kennebec")
    exp = find_experiment(load_experiments("IND"), "IND", season, "Kennebec")
    weather = load_weather(exp["weather_file"])
    days = pd.date_range(observed["Emergence"], pd.Timestamp(weather.last_date))
    frame = pd.DataFrame({
        "day": days,
        "temperature": [(weather(d.date()).TMIN + weather(d.date()).TMAX) / 2 for d in days],
        "daylength": [daylength(d.date(), float(weather.latitude)) for d in days],
    })
    daily[("IND", season)] = frame
    dates[("IND", season)] = (observed["Emergence"], observed["TuberInitiation"])

for season, emergence_doy, initiation_doy in CAN_DATES:
    year_start = pd.Timestamp(f"{season}-01-01")
    emergence = year_start + pd.Timedelta(days=emergence_doy - 1)
    initiation = year_start + pd.Timedelta(days=initiation_doy - 1)
    weather = pd.read_csv(ROOT / "data" / "Weather" / f"CAN_QC_7016932_{season}_P1D.csv",
                          parse_dates=["Date/Time"])
    latitude = float(weather["Latitude (y)"].iloc[0])
    weather = weather[weather["Date/Time"] >= emergence].sort_values("Date/Time")
    frame = pd.DataFrame({
        "day": weather["Date/Time"],
        "temperature": (weather["Max Temp (°C)"] + weather["Min Temp (°C)"]) / 2,
        "daylength": [daylength(d.date(), latitude) for d in weather["Date/Time"]],
    }).reset_index(drop=True)
    daily[("CAN", season)] = frame
    dates[("CAN", season)] = (emergence, initiation)

for key, frame in daily.items():
    frame["thermal"] = np.interp(frame["temperature"], DTSMTB[::2], DTSMTB[1::2])
    # Observed window includes emergence day and excludes initiation day.
    frame["in_window"] = frame["day"] < dates[key][1]


# %% Fit each calibration pair; evaluation endpoints never enter the search.
factorial_parameter_rows = []
for split, ind_cal, can_cal, ind_eval, can_eval in FACTORIAL_SPLITS:
    ind = daily[("IND", ind_cal)]
    can = daily[("CAN", can_cal)]
    tsum1 = ind.loc[ind["in_window"], "thermal"].sum()
    can_window = can[can["in_window"]]
    best_score = np.inf
    for dlo in DLO_GRID:
        for dlc in DLC_GRID:
            if dlo >= dlc:
                continue
            reduction = np.clip((dlc - can_window["daylength"]) / (dlc - dlo), 0, 1)
            requirement = (can_window["thermal"] * reduction).sum()
            score = (requirement - tsum1) ** 2
            # Strict improvement: exact ties keep the first pair in ascending grid order.
            if score < best_score:
                best_score, best_dlo, best_dlc = score, dlo, dlc
    factorial_parameter_rows.append({
        "split": split, "IND_cal": ind_cal, "CAN_cal": can_cal,
        "IND_eval": ind_eval, "CAN_eval": can_eval,
        "TSUM1": tsum1, "DLO": best_dlo, "DLC": best_dlc,
        "CAN_cal_squared_error": best_score,
    })

factorial_parameters = pd.DataFrame(factorial_parameter_rows)
print("Full-factorial calibration parameters:")
print(factorial_parameters.round(3).to_string(index=False))


# %% Frozen-parameter held-out predictions
factorial_evaluation_rows = []
for p in factorial_parameters.itertuples(index=False):
    for site, season in [("IND", p.IND_eval), ("CAN", p.CAN_eval)]:
        frame = daily[(site, season)]
        observed = dates[(site, season)][1]
        for method in ("Temperature only", "Fitted photoperiod"):
            reduction = (np.ones(len(frame)) if method == "Temperature only" else
                         np.clip((p.DLC - frame["daylength"].to_numpy()) / (p.DLC - p.DLO), 0, 1))
            effective = frame["thermal"].to_numpy() * reduction
            hit = np.flatnonzero(np.cumsum(effective) >= p.TSUM1)
            # Same next-day stage crossing as 02.7, with no fitted emergence phase.
            predicted = frame["day"].iloc[hit[0]] + pd.Timedelta(days=1) if len(hit) else pd.NaT
            factorial_evaluation_rows.append({
                "split": p.split, "method": method, "site": site, "season": season,
                "requirement": effective[frame["in_window"].to_numpy()].sum(),
                "observed": observed, "predicted": predicted,
                "days_off": (predicted - observed).days if pd.notna(predicted) else np.nan,
            })

factorial_evaluation = pd.DataFrame(factorial_evaluation_rows)
print("\nFull-factorial held-out evaluation (positive days off = late; NaT = target not reached):")
print(factorial_evaluation.round(3).to_string(index=False))
factorial_parameters.to_csv(OUTPUT_DIR / "kennebec_factorial_parameters.csv", index=False)
factorial_evaluation.to_csv(OUTPUT_DIR / "kennebec_factorial_evaluation.csv", index=False)


# %% Held-out requirement CV and absolute date error
factorial_score_rows = []
for (split, method), group in factorial_evaluation.groupby(["split", "method"], sort=False):
    requirements = group["requirement"].to_numpy()
    factorial_score_rows.append({
        "split": split, "method": method,
        "cv": np.std(requirements, ddof=0) / np.mean(requirements),
        "mean_absolute_days_off": np.mean(np.abs(group["days_off"].to_numpy())),
    })
factorial_scores = pd.DataFrame(factorial_score_rows)
print("\nFull-factorial held-out pair scores:")
print(factorial_scores.round(3).to_string(index=False))


# %% Preserve A/B results separately from all four split cases
parameters = factorial_parameters[factorial_parameters["split"].isin(["A", "B"])].copy()
evaluation = factorial_evaluation[factorial_evaluation["split"].isin(["A", "B"])].copy()
scores = factorial_scores[factorial_scores["split"].isin(["A", "B"])].copy()
parameters.to_csv(OUTPUT_DIR / "kennebec_pair_parameters.csv", index=False)
evaluation.to_csv(OUTPUT_DIR / "kennebec_pair_evaluation.csv", index=False)
scores.to_csv(OUTPUT_DIR / "kennebec_pair_scores.csv", index=False)
factorial_scores.to_csv(OUTPUT_DIR / "kennebec_factorial_scores.csv", index=False)

# %% Figure 2: original retained reference parameters and source weather


PARAMETERS = pd.DataFrame([
    ("IND", "2022-2023", "Kennebec", 138.7, 1561.3, 12.7, 17.7, 11.36, 11.36),
    ("IND", "2024-2025", "Kennebec", 138.7, 1561.3, 12.7, 17.7, 11.36, 11.36),
    ("NLD", "2025", "Innovator", 214.7, 1149.7, 17.43, 17.43, 16.58, 16.58),
    ("RSA", "2025-A", "Mondial", 398.7, 1244.0, 13.22, 13.22, 13.4, 15.9),
    ("RSA", "2025-S", "Mondial", 398.7, 1244.0, 13.22, 13.22, 13.4, 15.9),
    ("CAN", "1992", "Kennebec", 138.7, 789.2, 12.7, 17.7, 14.69, 14.69),
    ("CAN", "1993", "Kennebec", 138.7, 934.8, 12.7, 17.7, 14.69, 14.69),
], columns=["site", "season", "cultivar", "TSUM1", "TSUM2",
            "init_DLO_h", "init_DLC_h", "repro_DLO_h", "repro_DLC_h"]).set_index(["site", "season"])
print("Retained reference parameters; Kennebec is replaced below by the selected calibration split:")
print(PARAMETERS.to_string())
CAN = {
    "1992": {"csv": "CAN_QC_7016932_1992_P1D.csv", "sow_doy": 136,
             "emg_doy": 164, "ti_doy": 204, "harvest_doy": 260},
    "1993": {"csv": "CAN_QC_7016932_1993_P1D.csv", "sow_doy": 136,
             "emg_doy": 164, "ti_doy": 198, "harvest_doy": 256},
}


cropd = load_crop_data("data/crops_calibration")
_exp0 = find_experiment(load_experiments("IND"), "IND", "2024-2025", "Kennebec")
_crop0 = build_parameters(cropd, load_soil(_exp0["soil_file"]),
                          load_site_data(10.0, _exp0["CO2"]), "Kennebec", {})
DTSMTB = list(_crop0["DTSMTB"])
_XT, _YT = DTSMTB[0::2], DTSMTB[1::2]
_defaults = {k: float(_crop0[k]) for k in ("TBASEM", "TEFFMX", "TSUMEM")}
_obs_all = load_obs_experiments()

def dtsm(t):
    """Daily effective temperature from DTSMTB (linear interpolation), scalar or array."""
    return np.interp(t, _XT, _YT)

def dvred(P, dlo, dlc):
    """WOFOST linear day-length reduction factor (short-day: dlo < dlc)."""
    if dlc == dlo:
        return np.ones_like(np.asarray(P, dtype=float))
    return np.clip((np.asarray(P, dtype=float) - dlc) / (dlo - dlc), 0.0, 1.0)

def obs_row(site, season, cultivar, field):
    """Observed phenology row for one setup (CAN comes from the paper, not MainTable)."""
    if site == "CAN":
        c = CAN[season]
        doy = lambda n: pd.Timestamp(f"{season}-01-01") + pd.Timedelta(days=n - 1)
        return pd.Series({"Start": doy(c["sow_doy"]), "Emergence": doy(c["emg_doy"]),
                          "TuberInitiation": doy(c["ti_doy"]), "Maturity": doy(c["harvest_doy"])})
    return _obs_all[(_obs_all["location"] == site) & (_obs_all["field"] == field)
                    & (_obs_all["season_key"] == season)
                    & (_obs_all["cultivar"] == cultivar)].iloc[0]

def _field_daily(site, season, cultivar):
    """(day, tavg, P) from sowing to the last day of the weather record, plus the latitude."""
    exp = find_experiment(load_experiments(site), site, season, cultivar)
    wdp = load_weather(exp["weather_file"])
    lat = float(wdp.latitude)
    r = obs_row(site, season, cultivar, "OpenField")
    days = pd.date_range(r["Start"], pd.Timestamp(wdp.last_date))
    return pd.DataFrame([{"day": d, "tavg": (wdp(d.date()).TMIN + wdp(d.date()).TMAX) / 2.0,
                          "P": daylength(d.date(), lat)} for d in days]), lat

def _can_daily(season):
    """Same for a Canadian season, read straight from the Environment Canada daily CSV."""
    c = CAN[season]
    df = pd.read_csv(ROOT / "data" / "Weather" / c["csv"], parse_dates=["Date/Time"])
    lat = float(df["Latitude (y)"].iloc[0])
    df["doy"] = df["Date/Time"].dt.dayofyear
    sub = df[(df["doy"] >= c["sow_doy"])]
    return pd.DataFrame([{"day": d, "tavg": (mx + mn) / 2.0, "P": daylength(d.date(), lat)}
                         for d, mx, mn in zip(sub["Date/Time"], sub["Max Temp (°C)"],
                                              sub["Min Temp (°C)"])]), lat

def season_daily(key):
    """The rows of `daily` for one (site, season)."""
    return daily[(daily["site"] == key[0]) & (daily["season"] == key[1])]

def accumulate(key, start, end, factor=None):
    """Daily thermal accumulation over [start, end) for one season.
    factor: None for temperature only, or a function of photoperiod returning the daily
    development-rate multiplier (dvred for the original linear daylength response).
    Returns the windowed daily frame with columns red, dtsum, cum added.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    d = season_daily(key)
    win = d[(d["day"] >= start) & (d["day"] < end)].copy()
    win["red"] = 1.0 if factor is None else factor(win["P"].to_numpy())
    win["dtsum"] = dtsm(win["tavg"].to_numpy()) * win["red"]
    win["cum"] = win["dtsum"].cumsum()
    return win

def crossing_day(key, start, threshold, factor=None):
    """Day the accumulation from `start` first reaches `threshold`, or None if it never does.

    Return the weather day reaching the threshold plus one integration day.
    Native DVS floating-point boundaries can differ; do not adjust either date.
    """
    d = accumulate(key, start, season_daily(key)["day"].max() + pd.Timedelta(days=1), factor)
    hit = np.flatnonzero(d["cum"].to_numpy() >= threshold)
    return d["day"].iloc[hit[0]] + pd.Timedelta(days=1) if len(hit) else None


# Freeze the Kennebec split before predicting either calibration or evaluation crops.
main_parameters = factorial_parameters.set_index("split").loc[MAIN_SPLIT]
kennebec = PARAMETERS.index.get_level_values("site").isin(["IND", "CAN"])
PARAMETERS.loc[kennebec, "TSUM1"] = main_parameters.TSUM1
PARAMETERS.loc[kennebec, "init_DLO_h"] = main_parameters.DLO
PARAMETERS.loc[kennebec, "init_DLC_h"] = main_parameters.DLC
PARAMETERS.loc[PARAMETERS.index.get_level_values("site") == "IND", "TSUM2"] = 1700.0 - main_parameters.TSUM1


# %% Load all source weather and observations (the only season ingestion loop)
daily_parts, SEASON, WEATHER = [], {}, {}
for (site, season), p in PARAMETERS.iterrows():
    r = obs_row(site, season, p.cultivar, "OpenField")
    d, lat = _can_daily(season) if site == "CAN" else _field_daily(site, season, p.cultivar)
    assert not d.day.duplicated().any()
    assert d.day.diff().dropna().eq(pd.Timedelta(days=1)).all()
    assert np.isfinite(d.tavg).all()
    endpoint_kind = {"IND": "haulm-kill", "NLD": "senescence",
                     "RSA": "senescence", "CAN": "harvest"}[site]
    SEASON[(site, season)] = {
        "site": site, "season": season, "cultivar": p.cultivar, "field": "OpenField",
        "sowing": pd.Timestamp(r["Start"]), "emergence": pd.Timestamp(r["Emergence"]),
        "ti_obs": pd.Timestamp(r["TuberInitiation"]), "ti_start": pd.Timestamp(r["TuberInitiation"]),
        "endpoint": pd.Timestamp(r["Maturity"]), "endpoint_kind": endpoint_kind,
        "reference_maturity": pd.Timestamp(r["Maturity"]), "reference_endpoint_kind": endpoint_kind,
        "ti_source": "CAN literature" if site == "CAN" else "MainTable Experiments",
        "last_weather": d.day.max(),
    }
    daily_parts.append(d.assign(site=site, season=season))
    provider = WeatherDataProvider()
    provider.latitude = lat
    for day, temp, photoperiod in d.itertuples(index=False, name=None):
        provider._store_WeatherDataContainer(
            SimpleNamespace(DAY=day.date(), LAT=lat, TMIN=float(temp), TMAX=float(temp),
                            TEMP=float(temp), DTEMP=float(temp)), day.date())
    WEATHER[(site, season)] = provider
    print(site, season, "weather:", d.day.min(), "through", d.day.max())
daily = pd.concat(daily_parts, ignore_index=True)


# %% Refresh NLD's thermal windows after its 50% emergence-date update.
# CAN reproductive requirements remain own-season harvest references, not held-out maturity tests.
for key in [("NLD", "2025"), ("CAN", "1992"), ("CAN", "1993")]:
    s = SEASON[key]
    if key[0] == "NLD":
        PARAMETERS.loc[key, "TSUM1"] = accumulate(key, s["emergence"], s["ti_obs"])["dtsum"].sum()
    PARAMETERS.loc[key, "TSUM2"] = accumulate(key, s["ti_obs"], s["endpoint"])["dtsum"].sum()


# %% Reference dates for all locations: IND assumed maturity and RSA reference TI
key = ("IND", "2022-2023")
s = SEASON[key]
ind_2223_seijo = season_daily(key).loc[lambda d: d.day.ge(s["emergence"])].copy()
ind_2223_seijo["seijo_daily"] = np.clip(ind_2223_seijo.tavg - 2.0, 0.0, 24.0)
ind_2223_seijo["seijo_cumulative"] = ind_2223_seijo.seijo_daily.cumsum()
hits = ind_2223_seijo.loc[ind_2223_seijo.seijo_cumulative.ge(1700.0), "day"]
s["reference_maturity"] = hits.iloc[0] + pd.Timedelta(days=1) if len(hits) else pd.NaT
s["reference_endpoint_kind"] = "assumed Seijo maturity: base 2, cap 26, 1700 GDD"

# IND 2024-2025: independent assumed maturity, separate from haulm-kill
key = ("IND", "2024-2025")
s = SEASON[key]
ind_2425_seijo = season_daily(key).loc[lambda d: d.day.ge(s["emergence"])].copy()
ind_2425_seijo["seijo_daily"] = np.clip(ind_2425_seijo.tavg - 2.0, 0.0, 24.0)
ind_2425_seijo["seijo_cumulative"] = ind_2425_seijo.seijo_daily.cumsum()
hits = ind_2425_seijo.loc[ind_2425_seijo.seijo_cumulative.ge(1700.0), "day"]
s["reference_maturity"] = hits.iloc[0] + pd.Timedelta(days=1) if len(hits) else pd.NaT
s["reference_endpoint_kind"] = "assumed Seijo maturity: base 2, cap 26, 1700 GDD"


# RSA 2025-A: inferred TI from the fixed RSA2024 first-harvest proxy
key = ("RSA", "2025-A")
SEASON[key]["ti_obs"] = pd.NaT
SEASON[key]["ti_start"] = crossing_day(key, SEASON[key]["emergence"], PARAMETERS.loc[key, "TSUM1"])
SEASON[key]["ti_source"] = "assumed TI from fixed RSA 2024 TSUM1"

# RSA 2025-S: inferred TI; not an observed EM-TI scoring target
key = ("RSA", "2025-S")
SEASON[key]["ti_obs"] = pd.NaT
SEASON[key]["ti_start"] = crossing_day(key, SEASON[key]["emergence"], PARAMETERS.loc[key, "TSUM1"])
SEASON[key]["ti_source"] = "assumed TI from fixed RSA 2024 TSUM1"
references = pd.DataFrame(SEASON.values())
print("Reference dates and provenance (IND endpoint is haulm-kill):")
print(references[["site", "season", "emergence", "ti_obs", "ti_start", "ti_source",
                  "endpoint", "endpoint_kind", "reference_maturity", "reference_endpoint_kind"]].to_string(index=False))


def _run_pcse(site, season, cultivar, overrides, field="OpenField"):
    """Native continuous phase simulation from observed emergence."""
    key = (site, season)
    p = PARAMETERS.loc[key]
    crop_start = pd.Timestamp(obs_row(site, season, cultivar, field)["Emergence"])
    if pd.isna(crop_start):
        return pd.DataFrame(columns=["day", "DVS", "TSUM", "STAGE", "DOE", "DOA", "DOM"])
    provider = WEATHER[key]
    last_day = pd.Timestamp(provider.last_date)
    if crop_start > last_day:
        return pd.DataFrame(columns=["day", "DVS", "TSUM", "STAGE", "DOE", "DOA", "DOM"])
    params = {**_defaults, "DTSMTB": DTSMTB, "TSUM1": float(p.TSUM1), "TSUM2": float(p.TSUM2),
              "DVSEND": 2.0, "DVSI": 0.0, "IDSL": 0,
              "DLO": float(p.init_DLO_h), "DLC": float(p.init_DLC_h), **overrides}
    # Equal thresholds mean no reduction; native IDSL=1 would divide by zero.
    if params["DLO"] == params["DLC"]:
        params["IDSL"] = 0
    calendar = [{crop_start.date(): {
        "CropCalendar": {"crop_name": "potato", "variety_name": cultivar,
            "crop_start_date": crop_start.date(), "crop_start_type": "emergence",
            "crop_end_date": None, "crop_end_type": "maturity",
            "max_duration": (last_day - crop_start).days + 1},
        "TimedEvents": None, "StateEvents": None,
    }}]
    model = Wofost72_Phenology(ParameterProvider(cropdata=params), provider, calendar,
                               output_vars=["STAGE", "DOE", "DOA", "DOM"])
    model.run((last_day - crop_start).days)
    frame = pd.DataFrame(model.get_output())
    frame["day"] = pd.to_datetime(frame["day"])
    return frame

def first_day_at(sim, dvs):
    """First simulated day reaching `dvs`, or None if the run ends first."""
    hit = sim.loc[sim["DVS"] >= dvs, "day"]
    return hit.iloc[0] if len(hit) else None


# %% Continuous native a/b/c and existing external d calculations
continuous_dates = {}
for key, s in SEASON.items():
    p = PARAMETERS.loc[key]
    thermal = _run_pcse(*key, s["cultivar"], {"IDSL": 0})
    linear = _run_pcse(*key, s["cultivar"], {"IDSL": 1})
    ti_thermal, ti_linear = first_day_at(thermal, 1.0), first_day_at(linear, 1.0)
    factor = lambda P: dvred(P, p.repro_DLO_h, p.repro_DLC_h)
    continuous_dates[key] = {
        "ti_thermal": ti_thermal, "ma_thermal": first_day_at(thermal, 2.0),
        "ti_linear": ti_linear, "ma_external": crossing_day(key, ti_linear, p.TSUM2, factor),
        "unity_maturity": crossing_day(key, ti_thermal, p.TSUM2),
    }

# %% Phase dates, provenance and the existing manuscript handoffs
phase_parameters = PARAMETERS.reset_index().merge(references, on=["site", "season", "cultivar"])
phase_parameters["parameter_source"] = np.where(
    phase_parameters["site"].isin(["CAN", "IND"]), "Kennebec calibration split " + MAIN_SPLIT,
    np.where(phase_parameters["site"].eq("RSA"), "fixed autumn thermal references", "NLD2025 observed windows"))


continuous_stages = pd.DataFrame.from_dict(continuous_dates, orient="index").rename_axis(["site", "season"]).reset_index()
base_columns = ["site", "season", "panel", "prediction_start", "prediction_end", "external_unity_maturity"]
continuous_stages = continuous_stages.merge(references[["site", "season", "emergence"]], on=["site", "season"])
phase_duration = pd.concat([
    continuous_stages.assign(panel="P1", prediction_start=continuous_stages.emergence,
        prediction_end=continuous_stages.ti_thermal, external_unity_maturity=pd.NaT)[base_columns],
    continuous_stages.assign(panel="P2", prediction_start=continuous_stages.emergence,
        prediction_end=continuous_stages.ti_linear, external_unity_maturity=pd.NaT)[base_columns],
    continuous_stages.assign(panel="P3", prediction_start=continuous_stages.ti_thermal,
        prediction_end=continuous_stages.ma_thermal, external_unity_maturity=continuous_stages.unity_maturity)[base_columns],
    continuous_stages.assign(panel="P4", prediction_start=continuous_stages.ti_linear,
        prediction_end=continuous_stages.ma_external, external_unity_maturity=pd.NaT)[base_columns],
], ignore_index=True)
phase_duration["check"] = "predicted_ti"
phase_duration = phase_duration.merge(references, on=["site", "season"]).merge(
    PARAMETERS.reset_index(), on=["site", "season", "cultivar"])
phase_duration = phase_duration.rename(columns={"emergence": "reference_emergence", "ti_start": "reference_ti",
                                                "ti_obs": "observed_ti"})
phase_duration[["prediction_start", "prediction_end", "external_unity_maturity"]] = phase_duration[
    ["prediction_start", "prediction_end", "external_unity_maturity"]].apply(pd.to_datetime)
is_init = phase_duration.panel.isin(["P1", "P2"])
phase_duration["reference_start"] = phase_duration.reference_ti.where(~is_init, phase_duration.reference_emergence)
# RSA 2025's assumed TI is never used as an observed EM-TI scoring target.
phase_duration["reference_end"] = phase_duration.reference_maturity.where(~is_init, phase_duration.observed_ti)
phase_duration["simulated_ti"] = phase_duration.prediction_start.where(~is_init, phase_duration.prediction_end)
phase_duration["simulated_maturity"] = phase_duration.prediction_end.where(~is_init, pd.NaT)
phase_duration["obs_days"] = (phase_duration.reference_end - phase_duration.reference_start).dt.days
phase_duration["pred_days"] = (phase_duration.prediction_end - phase_duration.prediction_start).dt.days
phase_duration["duration_error_d"] = phase_duration.pred_days - phase_duration.obs_days
# Four-panel comparisons use the same TI origin on both reproductive axes, as
# the original comparison did for CAN. Both reproductive axes use predicted TI.
# Keep obs_days above as the actual observed/assumed phase duration: subtracting
# different TI dates on the two axes can hide a shared shift in TI and maturity.
phase_duration["comparison_start"] = phase_duration.prediction_start.where(~is_init, phase_duration.reference_emergence)
phase_duration["comparison_reference_days"] = (phase_duration.reference_end - phase_duration.comparison_start).dt.days
phase_duration["comparison_simulated_days"] = (phase_duration.prediction_end - phase_duration.comparison_start).dt.days
phase_duration["comparison_error_d"] = phase_duration.comparison_simulated_days - phase_duration.comparison_reference_days
scoring_ti = phase_duration.reference_ti.where(~is_init, phase_duration.observed_ti)
phase_duration["ti_error_d"] = (phase_duration.simulated_ti - scoring_ti).dt.days
phase_duration["maturity_error_d"] = (phase_duration.simulated_maturity - phase_duration.reference_maturity).dt.days
phase_duration["native_minus_external_days"] = (
    phase_duration.prediction_end - phase_duration.external_unity_maturity).dt.days
phase_duration["engine"] = np.where(phase_duration.panel.eq("P4"), "external existing functions", "Wofost72_Phenology")
phase_duration["crossing_status"] = np.select([
    phase_duration.prediction_start.isna(), phase_duration.prediction_end.isna(),
    phase_duration.prediction_end.gt(phase_duration.last_weather),
], ["TI start unavailable", "stage not reached in available weather", "next integration after last weather day"],
    default="reached within available weather")
phase_dur_predicted = phase_duration.copy()
assert phase_dur_predicted.groupby("panel").size().eq(7).all()
assert phase_dur_predicted.loc[~is_init, "comparison_error_d"].equals(
    phase_dur_predicted.loc[~is_init, "maturity_error_d"].rename("comparison_error_d"))
print("Continuous approach: all locations, predicted TI, no reproductive restart")
print(phase_dur_predicted[["panel", "site", "season", "comparison_reference_days", "comparison_simulated_days",
                           "ti_error_d", "maturity_error_d", "duration_error_d", "native_minus_external_days"]].to_string(index=False))
phase_dur_predicted.to_csv(PHASE_DIR / "phase_duration_predicted_ti.csv", index=False)
phase_parameters.to_csv(PHASE_DIR / "phase_duration_parameters.csv", index=False)


# Preserve the NLD timing handoff used outside Figure 2.
phase_duration.to_csv(OUTPUT_DIR / "photoperiod_duration_1to1_photoperiod_v1.csv", index=False)

# %% RSA calibration: spring open field only, with fixed autumn thermal thresholds.
SEASONS = ["2025-A", "2025-S"]
SCENARIOS = ["Temperature only", "TSUM1-only effect", "TSUM2-only effect"]
TSUM1_FIXED, TSUM2_SHARED = 398.7, 1244.0
DLO_VALUES = np.round(np.arange(9.0, 14.01, 0.1), 1)
DLC_VALUES = np.round(np.arange(10.0, 18.01, 0.1), 1)
df_daily = daily[daily["site"].eq("RSA")].rename(columns={"tavg": "TEMP", "P": "DAYLP"}).copy()
df_daily["DTSM"] = dtsm(df_daily["TEMP"])
season_dates = {season: {"emergence": SEASON[("RSA", season)]["emergence"],
                        "maturity": SEASON[("RSA", season)]["endpoint"]} for season in SEASONS}
temperature_only_dvs1 = {season: crossing_day(("RSA", season), season_dates[season]["emergence"], TSUM1_FIXED)
                        for season in SEASONS}


spring = df_daily[
    df_daily["season"].eq("2025-S")
    & df_daily["day"].ge(season_dates["2025-S"]["emergence"])
].copy()
grid_rows = []
for mechanism in SCENARIOS[1:]:
    for dlo in DLO_VALUES:
        for dlc in DLC_VALUES:
            if dlo >= dlc:
                continue
            red = np.clip((spring["DAYLP"].to_numpy() - dlc) / (dlo - dlc), 0.0, 1.0)
            effective = spring["DTSM"].to_numpy() * (red if mechanism == "TSUM1-only effect" else 1.0)
            hit = np.flatnonzero(np.cumsum(effective) >= TSUM1_FIXED)
            initiation = spring["day"].iloc[hit[0]] + pd.Timedelta(days=1) if len(hit) else pd.NaT
            row = {"mechanism": mechanism, "DLO": dlo, "DLC": dlc,
                   "TSUM1": TSUM1_FIXED, "shared_TSUM2": TSUM2_SHARED,
                   "initiation": initiation, "status": "initiation not reached"}
            if pd.notna(initiation):
                before_maturity = ((spring["day"] >= initiation)
                                   & (spring["day"] < season_dates["2025-S"]["maturity"]))
                row["status"] = "no reproductive window"
                if before_maturity.any():
                    effective = spring["DTSM"].to_numpy() * (red if mechanism == "TSUM2-only effect" else 1.0)
                    apparent = float(effective[before_maturity].sum())
                    reproductive = spring["day"] >= initiation
                    hit = np.flatnonzero(np.cumsum(effective[reproductive]) >= TSUM2_SHARED)
                    maturity = (spring.loc[reproductive, "day"].iloc[hit[0]] + pd.Timedelta(days=1)
                                if len(hit) else pd.NaT)
                    error = ((maturity - season_dates["2025-S"]["maturity"]).days
                             if pd.notna(maturity) else np.nan)
                    row.update({"status": "valid", "apparent_TSUM2": apparent,
                                "TSUM2_difference": apparent - TSUM2_SHARED,
                                "abs_TSUM2_difference": abs(apparent - TSUM2_SHARED),
                                "within_1pct": abs(apparent - TSUM2_SHARED) <= 0.01 * TSUM2_SHARED,
                                "predicted_maturity": maturity, "maturity_error_days": error,
                                "abs_maturity_error_days": abs(error),
                                "initiation_delay_days": (initiation - temperature_only_dvs1["2025-S"]).days})
            grid_rows.append(row)

df_grid = pd.DataFrame(grid_rows)
rank_columns = ["abs_TSUM2_difference", "abs_maturity_error_days", "DLO", "DLC"]
df_valid = df_grid[df_grid["status"].eq("valid") & df_grid["predicted_maturity"].notna()]
if set(df_valid["mechanism"]) != set(SCENARIOS[1:]):
    raise RuntimeError("A mechanism has no valid spring candidate; inspect df_grid")
df_valid = df_valid.sort_values(rank_columns, na_position="last")
df_best = df_valid.drop_duplicates("mechanism").set_index("mechanism")
df_best.reset_index().to_csv(OUTPUT_DIR / "rsa_photoperiod_fit_parameters.csv", index=False)


# %% Selected RSA dates; autumn remains unreduced for both hypotheses.
df_best = pd.read_csv(OUTPUT_DIR / "rsa_photoperiod_fit_parameters.csv",
                      parse_dates=["initiation", "predicted_maturity"]).set_index("mechanism")
selected_rows = []
for field in ["OpenField", "RainShelter"]:
    for season in SEASONS:
        observed = obs_row("RSA", season, "Mondial", field)
        emergence, endpoint = pd.Timestamp(observed["Emergence"]), pd.Timestamp(observed["Maturity"])
        key = ("RSA", season)
        thermal_ti = crossing_day(key, emergence, TSUM1_FIXED)
        if field == "OpenField":
            native_ti, native_maturity = continuous_dates[key]["ti_thermal"], continuous_dates[key]["ma_thermal"]
        else:
            thermal = _run_pcse("RSA", season, "Mondial", {"IDSL": 0}, field=field)
            native_ti, native_maturity = first_day_at(thermal, 1.0), first_day_at(thermal, 2.0)
        for scenario in SCENARIOS:
            active = season == "2025-S" and scenario != "Temperature only"
            dlo, dlc = (df_best.loc[scenario, ["DLO", "DLC"]]
                        if scenario != "Temperature only" else (np.nan, np.nan))
            factor = (lambda P: dvred(P, dlo, dlc)) if active else None
            initiation = crossing_day(key, emergence, TSUM1_FIXED,
                                      factor if scenario == "TSUM1-only effect" else None)
            reproductive_factor = factor if scenario == "TSUM2-only effect" else None
            maturity = crossing_day(key, initiation, TSUM2_SHARED, reproductive_factor)
            apparent = accumulate(key, initiation, endpoint, reproductive_factor)["dtsum"].sum()
            role = ("evaluation" if field == "RainShelter" else
                    "evaluation" if scenario == "Temperature only" and season == "2025-S" else "calibration")
            selected_rows.append({
                "site": "RSA", "season": season, "field": field, "scenario": scenario,
                "role": role, "DLO": dlo, "DLC": dlc, "emergence": emergence,
                "TSUM1": TSUM1_FIXED, "shared_TSUM2": TSUM2_SHARED, "initiation": initiation,
                "observed_initiation": pd.NaT,
                "initiation_delay_days": (initiation - thermal_ti).days,
                "native_thermal_initiation": native_ti,
                "native_thermal_maturity": native_maturity,
                "apparent_TSUM2": apparent, "TSUM2_difference": apparent - TSUM2_SHARED,
                "predicted_maturity": maturity, "observed_maturity": endpoint,
                "maturity_error_days": (maturity - endpoint).days if pd.notna(maturity) else np.nan,
            })

df_selected = pd.DataFrame(selected_rows)
df_selected.to_csv(OUTPUT_DIR / "rsa_photoperiod_selected.csv", index=False)
print("RSA open-field calibration and frozen-parameter rain-shelter evaluation:")
print(df_selected.to_string(index=False))


# %% Main Figure 2: retained four-panel comparison and RSA spring hypotheses
# Re-execute this cell alone to render the saved essential source tables.
main_parameters = pd.read_csv(OUTPUT_DIR / "kennebec_factorial_parameters.csv",
                              dtype={"IND_cal": str, "CAN_cal": str}).set_index("split").loc[MAIN_SPLIT]
figure2_source = OUTPUT_DIR / "phase_duration_check"
df_figure2_labelled = pd.read_csv(
    figure2_source / "phase_duration_predicted_ti.csv", dtype={"season": str},
)
# Use the common predicted-TI origin for reproductive comparisons. Comparing
# independently anchored phase durations would hide CAN's early TI and endpoint.
df_figure2_labelled["obs_days"] = df_figure2_labelled["comparison_reference_days"]
df_figure2_labelled["pred_days"] = df_figure2_labelled["comparison_simulated_days"]
df_figure2_labelled["scenario"] = "Continuous model"
df_figure2_labelled_params = pd.read_csv(
    figure2_source / "phase_duration_parameters.csv", dtype={"season": str},
)
# Shared location styles; filled/open RSA triangles distinguish the hypotheses.
figure2_site_order = ["NLD", "CAN", "IND", "RSA"]
df_figure2_labelled_rsa = pd.read_csv(
    OUTPUT_DIR / "rsa_photoperiod_selected.csv",
    parse_dates=["emergence", "initiation", "observed_initiation", "predicted_maturity", "observed_maturity",
                 "native_thermal_initiation", "native_thermal_maturity"],
)

# Keep the non-RSA comparisons; assemble both RSA setups from frozen predictions.
# Both reproductive axes use the same predicted TI origin.
df_figure2_labelled = df_figure2_labelled[df_figure2_labelled["site"].ne("RSA")].copy()
main_cal = {("IND", str(main_parameters.IND_cal)), ("CAN", str(main_parameters.CAN_cal)), ("NLD", "2025")}
df_figure2_labelled["role"] = ["calibration" if (site, season) in main_cal else "evaluation"
                              for site, season in zip(df_figure2_labelled.site, df_figure2_labelled.season)]
figure2_labelled_rows = []
for record in df_figure2_labelled_rsa.itertuples():
    panels = ["P1", "P3"] if record.scenario == "Temperature only" else ["P2", "P4"]
    if record.scenario == "TSUM2-only effect":
        panels = ["P4"]
    for panel in panels:
        thermal = record.scenario == "Temperature only"
        initiation = record.native_thermal_initiation if thermal else record.initiation
        maturity = record.native_thermal_maturity if thermal else record.predicted_maturity
        is_init = panel in ["P1", "P2"]
        start = record.emergence if is_init else initiation
        reference_end = record.native_thermal_initiation if is_init else record.observed_maturity
        if panel == "P2" and record.season == "2025-S":
            reference_end = record.emergence + pd.Timedelta(days=50)
        predicted_end = initiation if is_init else maturity
        figure2_labelled_rows.append({
            "panel": panel, "site": "RSA", "season": record.season, "field": record.field,
            "scenario": record.scenario, "role": record.role,
            "obs_days": (reference_end - start).days,
            "pred_days": (predicted_end - start).days,
            "TSUM1": record.TSUM1, "TSUM2": record.shared_TSUM2,
            "DLO": record.DLO, "DLC": record.DLC,
        })
df_figure2_labelled = pd.concat(
    [df_figure2_labelled, pd.DataFrame(figure2_labelled_rows)], ignore_index=True
)
duration = df_figure2_labelled.copy()
print("Figure 2: RSA open-field calibration / rain-shelter evaluation; TSUM1 = 398.7, TSUM2 = 1244.0.")
print("Reproductive reference windows end at reference maturity/harvest, starting at each predicted TI.")
print("IND uses assumed Seijo maturity; RSA 2025 initiation is unobserved.")

df_figure2_labelled_spring = df_figure2_labelled_rsa[
    df_figure2_labelled_rsa["season"].eq("2025-S")
    & df_figure2_labelled_rsa["field"].eq("OpenField")
    & df_figure2_labelled_rsa["scenario"].ne("Temperature only")
].set_index("scenario").loc[["TSUM1-only effect", "TSUM2-only effect"]]

figure2_point_counts = df_figure2_labelled.assign(
    plotted=df_figure2_labelled[["obs_days", "pred_days"]].notna().all(axis=1)
).groupby("panel")["plotted"].sum()

print("Figure 2 finite points by panel:", figure2_point_counts.to_dict())
df_figure2_labelled.to_csv(SOURCE_DIR / "fig02_photoperiod_comparison.csv", index=False)
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


# %% Appendix: A/B and full-factorial CV and held-out date-error figures
scores = pd.read_csv(OUTPUT_DIR / "kennebec_pair_scores.csv")
factorial_scores = pd.read_csv(OUTPUT_DIR / "kennebec_factorial_scores.csv")
factorial_parameters = pd.read_csv(OUTPUT_DIR / "kennebec_factorial_parameters.csv",
                                   dtype={"IND_eval": str, "CAN_eval": str})
for plot_scores, splits, filename, png_name in [
    (scores, ["A", "B"], "kennebec_pair_cross_validation.svg", "kennebec_pair_evaluation.png"),
    (factorial_scores, ["A", "B", "C", "D"], "kennebec_factorial_cross_validation.svg", "kennebec_factorial_evaluation.png"),
]:
    fig, axes = plt.subplots(1, 2, figsize=(2.7 * len(splits) + 2, 4))
    for method, offset, hatch in [("Temperature only", -0.18, ""), ("Fitted photoperiod", 0.18, "///")]:
        values = plot_scores[plot_scores["method"].eq(method)].set_index("split").loc[splits]
        for ax, column in zip(axes, ["cv", "mean_absolute_days_off"]):
            heights = values[column].to_numpy()
            ax.bar(np.arange(len(splits)) + offset, heights, width=0.36,
                   facecolor="white", edgecolor="black", hatch=hatch, label=method)
            for x, height in zip(np.arange(len(splits)) + offset, heights):
                ax.annotate(f"{height:.2f}" if np.isfinite(height) else "Not reached",
                            (x, height if np.isfinite(height) else 0), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=8)
    pairs = factorial_parameters.set_index("split").loc[splits]
    labels = [f"{row.split}: IND {display_season(row.IND_eval)}\n+ CAN {row.CAN_eval}"
              for row in pairs.reset_index().itertuples(index=False)]
    for ax in axes:
        ax.set_xticks(np.arange(len(splits)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("Held-out evaluation pair")
    axes[0].set_ylabel("CV of held-out thermal requirements")
    axes[1].set_ylabel("Mean absolute days off")
    axes[0].set_title("(a) Requirement consistency")
    axes[1].set_title("(b) Initiation-date prediction")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(SUPP_FIG_DIR / filename, format="svg", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / png_name, dpi=300, bbox_inches="tight")
    plt.show()
