"""
utils/loaders.py
----------------
Functions to load all inputs for the WOFOST workflow.
All paths are resolved relative to the project root (ch3_model/).
"""

import pathlib
import yaml
import pandas as pd
import numpy as np
from pcse.input import ExcelWeatherDataProvider, YAMLCropDataProvider, CABOFileReader
from pcse.input import WOFOST73SiteDataProvider
from pcse.base import ParameterProvider
import ast
# Project root = two levels up from this file (scripts/utils/ -> scripts/ -> ch3_model/)
ROOT = pathlib.Path(__file__).resolve().parents[2]


# ── Experiments ─────────────────────────────────────────────────────────────

def canonicalize_season(season) -> str:
    """Normalize season labels so observed tables and config use the same key.

    Examples:
        2022-23   -> 2022-2023
        2024      -> 2024
        2025-A    -> 2025-A
    """
    season = str(season)
    if len(season) == 7 and season[4] == "-" and season[-2:].isdigit():
        return f"{season[:4]}-{season[:2]}{season[-2:]}"
    return season

def load_experiments(location: str = None) -> list[dict]:
    """Load experiment definitions from data/params/experiments.yaml.

    Args:
        location: If given, filter to experiments for that location
                  ('IND', 'NLD', 'RSA').

    Returns:
        List of experiment dicts.
    """
    path = ROOT / "data" / "params" / "experiments.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)
    experiments = data["experiments"]
    if location:
        experiments = [e for e in experiments if e["location"] == location]
    return experiments


def find_experiment(
    experiments: list[dict],
    location: str,
    season: str,
    cultivar: str,
) -> dict | None:
    """Resolve one base experiment by location, season, and cultivar.

    The experiments.yaml file is expected to contain one base row per
    location × season × cultivar. Treatment-specific runs should be derived
    from observed tables or irrigation schedules, not duplicated here.
    """
    matches = [
        exp for exp in experiments
        if str(exp["location"]) == str(location)
        and canonicalize_season(exp["season"]) == canonicalize_season(season)
        and str(exp["cultivar"]) == str(cultivar)
    ]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            "Multiple experiments found for "
            f"{location}/{canonicalize_season(season)}/{cultivar}. "
            "experiments.yaml should only contain base experiment rows."
        )
    return matches[0]


# ── Agromanagement ───────────────────────────────────────────────────────────

AGRO_FILE = {
    "IND": "india_seasons.yaml",
    "NLD": "netherlands_seasons.yaml",
    "RSA": "southafrica_seasons.yaml",
}


def load_seasons(location: str) -> dict:
    """The 'seasons' block of one location's agro YAML."""
    fname = AGRO_FILE.get(location)
    if fname is None:
        raise ValueError(f"Unknown location '{location}'. Choose from: {list(AGRO_FILE)}")
    with open(ROOT / "data" / "agro" / fname) as f:
        return yaml.safe_load(f)["seasons"]


def run_days(location: str, season: str):
    """Number of days to run this season, or None to run till terminate."""
    return load_seasons(location)[season].get("run_days")


def load_agro(location: str, season: str, cultivar: str) -> list:
    """Load and return a parsed PCSE agromanagement object.

    Args:
        location: 'IND', 'NLD', or 'RSA'.
        season:   Season key matching the agro YAML, e.g. '2022-2023'.
        cultivar: Variety name to substitute into the agro template.

    Returns:
        Parsed agromanagement list ready for PCSE.
    """
    seasons = load_seasons(location)
    if season not in seasons:
        raise KeyError(f"Season '{season}' not found in {AGRO_FILE[location]}. "
                       f"Available: {list(seasons)}")

    agro_str = seasons[season]["agro_yaml"].format(cultivar=cultivar)
    return yaml.safe_load(agro_str)


# ── Weather ──────────────────────────────────────────────────────────────────

def load_weather(weather_file: str, force_reload: bool = False) -> ExcelWeatherDataProvider:
    """Load weather data from an Excel file.

    Args:
        weather_file: Path relative to project root, e.g.
                      'data/Weather/India2022_23_wofost.xlsx'.
        force_reload: If True, bypass the cached .cache file.

    Returns:
        ExcelWeatherDataProvider instance.
    """
    path = ROOT / weather_file
    return ExcelWeatherDataProvider(str(path), force_reload=force_reload)


def rain_free_weather(weather_file: str) -> ExcelWeatherDataProvider:
    """Weather provider with rainfall zeroed, for rainshelter runs.

    A rainshelter excludes rain, so those treatments receive water only from
    irrigation. Each call returns a fresh provider (load_weather builds a new
    object), so zeroing RAIN here does not affect the open-field driver.
    """
    wdp = load_weather(weather_file)
    for c in wdp.store.values():
        c.RAIN = 0.0
    return wdp


# ── Soil ─────────────────────────────────────────────────────────────────────

def load_soil(soil_file: str) -> CABOFileReader:
    """Load soil data from a CABO-format soil file.

    Args:
        soil_file: Path relative to project root, e.g.
                   'data/soil/india_sandyloam.soil'.

    Returns:
        CABOFileReader instance.
    """
    path = ROOT / soil_file
    return CABOFileReader(str(path))


def load_soil_mlwb(soil_file: str) -> dict:
    """Load a multi-layer soil profile for the WaterBalanceLayered (MLWB) config.

    Args:
        soil_file: Path relative to project root, e.g.
                   'data/soil/netherlands_soil_MLWB.yaml'.

    Returns:
        Plain dict with RDMSOL, omegac and SoilProfileDescription at the top level,
        usable directly as the soildata argument to build_parameters/ParameterProvider
        (see pcse.soil.soil_profile.SoilProfile, which reads these keys).
    """
    path = ROOT / soil_file
    with open(path) as f:
        return yaml.safe_load(f)


# ── Crop & Site ──────────────────────────────────────────────────────────────

def load_crop_data(crop_dir: str = "data") -> YAMLCropDataProvider:
    """Load the YAML crop data provider.

    Args:
        crop_dir: Path to the directory containing crops.yaml and potato.yaml,
                  relative to the project root.
                  - "data"                → baseline (all original cultivars)
                  - "data/crops_calibration" → calibration (Kennebec, Innovator, Mondial only)
    """
    path = ROOT / crop_dir
    return YAMLCropDataProvider(fpath=str(path))


def load_site_data(WAV: float, CO2: float) -> WOFOST73SiteDataProvider:
    """Create site data with given initial available soil water (cm) and CO2 (ppm).

    Both come from the experiment row in data/params/experiments.yaml.
    """
    return WOFOST73SiteDataProvider(WAV=WAV, CO2=CO2)


def build_parameters(
    cropd: YAMLCropDataProvider,
    soild: CABOFileReader,
    sited: WOFOST73SiteDataProvider,
    cultivar: str,

    cultivar_overrides: dict = None,
) -> ParameterProvider:
    """Build a ParameterProvider for one experiment run.

    Args:
        cropd:               YAMLCropDataProvider (already loaded).
        soild:               CABOFileReader for the soil.
        sited:               WOFOST73SiteDataProvider.
        cultivar:            Active variety name.
        TDWI:                Initial seed tuber dry weight (kg/ha).
        cultivar_overrides:  Dict of additional parameter overrides,
                             e.g. {'TSUM1': 300, 'TSUM2': 1700}.

    Returns:
        ParameterProvider ready for PCSE model initialisation.
    """
    cropd.set_active_crop("potato", cultivar)
    params = ParameterProvider(cropdata=cropd, soildata=soild, sitedata=sited)
    if cultivar_overrides:
        for key, value in cultivar_overrides.items():
            if value is None:
                continue
            if isinstance(value, str) and value.strip().startswith("["):
                value = ast.literal_eval(value)  # "[0,0,5,0]" → [0, 0, 5, 0]
            params.set_override(key, value)
    # Apply TDWI last so it always overrides any cultivar-level TDWI entry
    # params.set_override("TDWI", TDWI)
    return params


# ── Cultivar parameters ──────────────────────────────────────────────────────
def get_overrides_long_format(
    cultivar_params: pd.DataFrame,
    val_col: str = "value",
    location: str = None,
    season: str = None,
    cultivar: str = None,
) -> dict:
    """filtered the cultivar_params DataFrame to only include rows where the specified version column has non-NaN values, and return a long-format DataFrame with columns: cultivar, location, season_key, Parameter, value.

    Args:
        cultivar_params: DataFrame with columns:
                         Cultivar, Location, season_key, Parameter, <version>.
        version:         Column name to read the override value from.
        location:        Location code ('IND', 'NLD', 'RSA').
        season:          Season key (e.g. '2022-2023').
    Returns:
        Long-format DataFrame with columns:
        cultivar, location, season_key, Parameter, value
    """
    mask = (
        (cultivar_params["cultivar"] == cultivar) &
        (cultivar_params["location"].isin([location, "all"]))
    )
    if season is not None and "season_key" in cultivar_params.columns:
        sk = cultivar_params["season_key"].astype(str)
        mask &= sk.isin([str(season), "all"])

    sub = cultivar_params.loc[mask]
    if sub.empty:
        return {}

    overrides = {}
    for _, row in sub.iterrows():
        val = row.get(val_col) if val_col in sub.columns else None
        if pd.notna(val):
            overrides[row["parameter"]] = val
    return overrides

def get_cultivar_overrides(
    cultivar_params: pd.DataFrame,
    cultivar: str,
    location: str,
    season: str = None,
    version: str = "v1",
) -> dict:
    """Extract model-ready parameter overrides for a cultivar × location × season.

    Args:
        cultivar_params: Long-format DataFrame with columns:
                         Cultivar, Location, season_key, Parameter, <version>.
        cultivar:        Variety name.
        location:        Location code ('IND', 'NLD', 'RSA').
        season:          Season key (e.g. '2022-2023'). If given, only rows
                         matching this season are used, preventing duplicate
                         Parameter names from different seasons overwriting each other.
        version:         Column name to read the override value from.

    Returns:
        Flat dict of {Parameter: value} with only non-NaN entries.
    """
    mask = (
        (cultivar_params["cultivar"] == cultivar) &
        (cultivar_params["location"].isin([location, "all"]))
    )
    if season is not None and "season_key" in cultivar_params.columns:
        sk = cultivar_params["season_key"].astype(str)
        mask &= sk.isin([str(season), "all"])

    sub = cultivar_params.loc[mask]
    if sub.empty:
        return {}

    overrides = {}
    for _, row in sub.iterrows():
        val = row.get(version) if version in sub.columns else None
        if pd.notna(val):
            overrides[row["Parameter"]] = val
    return overrides


# ── Observations ─────────────────────────────────────────────────────────────
# Curated inputs live in data/obs/MainTable.xlsx:
#   Experiments sheet — phenology dates, sowing, TDWI, WAV
# Summarised observations (Yield, LAI) come from output/obs_benchmark_summary.csv,
# the single source of truth produced by scripts/00_summarise_obs.py from the raw
# replicate CSVs. Regenerate that CSV (rerun 00_summarise_obs.py) after raw-data
# changes. Yield/TWSO/TAGP are in kg/ha there — no unit conversion on read.

MAINTABLE = ROOT / "data" / "obs" / "MainTable.xlsx"
BENCHMARK = ROOT / "output" / "obs_benchmark_summary.csv"


def _read_sheet(sheet: str) -> pd.DataFrame:
    """Read a sheet from MainTable.xlsx."""
    return pd.read_excel(MAINTABLE, sheet_name=sheet)


def _benchmark_wide(variables: list[str]) -> pd.DataFrame:
    """Read obs_benchmark_summary.csv and pivot the given variables wide.

    The benchmark CSV is long-format (one row per group × variable, with
    mean/sd/n). This pivots the requested variables back to the wide layout the
    analysis scripts expect. Column names come back as ``mean__<VAR>`` /
    ``sd__<VAR>`` / ``n__<VAR>``; each loader renames them to its canonical
    columns. ``field`` is renamed to ``Field`` to match the previous schema.
    """
    df = pd.read_csv(BENCHMARK)
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df = df[df["variable"].isin(variables)]
    keys = ["location", "season", "field", "treatment", "cultivar", "day", "DAP"]
    wide = df.pivot_table(index=keys, columns="variable",
                          values=["mean", "sd", "n"], aggfunc="first")
    wide.columns = [f"{stat}__{var}" for stat, var in wide.columns]
    return wide.reset_index().rename(columns={"field": "Field"})


def load_obs_experiments() -> pd.DataFrame:
    """Load the Experiments sheet (sowing dates, phenology, TDWI, WAV).

    Returns a DataFrame with columns:
        Location, season_key, Field, Treatment, Start, Emergence,
        TuberInitiation, Maturity, Season_duration, TDWI, WAV
    """
    df = _read_sheet("Experiments")
    df = df.rename(columns={"End_Leaf_removal": "Maturity"})
    df["field"] = df["field"].replace("Rainshelter", "RainShelter")
    df["season_key"] = df["season_key"].astype(str)
    for col in ["Start", "Emergence", "TuberInitiation", "Maturity"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    # NLD2025 emergence: 50% of planted stand, interpolated between count assessments.
    counts = pd.read_csv(ROOT / "data" / "obs" / "emergence_plot_data.csv", dtype={"season": str})
    counts = counts.loc[counts["Site"].eq("NLD") & counts["season"].eq("2025")]
    assessments = counts.groupby("dap")[["emerged", "target"]].sum().sort_index()
    fraction = assessments["emerged"] / assessments["target"]
    dap50 = np.interp(0.5, fraction.to_numpy(), assessments.index.to_numpy())
    nld2025 = df["location"].eq("NLD") & df["season_key"].eq("2025")
    df.loc[nld2025, "Emergence"] = df.loc[nld2025, "Start"] + pd.to_timedelta(
        int(np.rint(dap50)), unit="D",
    )
    return df


def load_obs_phenology(location: str = None) -> pd.DataFrame:
    """Load observed phenology dates from MainTable.xlsx (Experiments sheet).

    Returns long-format DataFrame with columns:
        location, season, Field, day, obs_dvs, stage
    matching the format expected by 00_prepare_inputs.py and 03_evaluate.py.
    """
    df = load_obs_experiments()
    stage_map = {
        "Emergence":       (0, "Emergence"),
        "TuberInitiation": (1, "Tuber initiation"),
        "Maturity":         (2, "Maturity"),
    }
    rows = []
    for _, row in df.iterrows():
        for col, (dvs, stage) in stage_map.items():
            rows.append({
                "location": row["location"],
                "season":   str(row["season_key"]),
                "field":    row["field"],
                "cultivar": row["cultivar"],
                "day":      row[col],
                "obs_dvs":  dvs,
                "stage":    stage,
            })
    out = pd.DataFrame(rows)
    if location:
        out = out[out["location"] == location]
    return out.reset_index(drop=True)


def load_obs_yield(location: str = None) -> pd.DataFrame:
    """Load observed yield / harvest data from obs_benchmark_summary.csv.

    Returns wide columns: location, season, Field, treatment, cultivar, day,
    DAP, TWSO_obs, TWSO_obs_sd, TAGP_obs, TAGP_obs_sd, n. Values are in kg/ha.
    """
    wide = _benchmark_wide(["TWSO", "TAGP"]).rename(columns={
        "mean__TWSO": "TWSO_obs", "sd__TWSO": "TWSO_obs_sd",
        "mean__TAGP": "TAGP_obs", "sd__TAGP": "TAGP_obs_sd",
    })
    n_cols = [c for c in wide.columns if c.startswith("n__")]
    wide["n"] = wide[n_cols].max(axis=1)
    wide = wide.drop(columns=n_cols)
    if location:
        wide = wide[wide["location"] == location]
    return wide.reset_index(drop=True)


def load_obs_lai(location: str = None) -> pd.DataFrame:
    """Load observed LAI timeseries from obs_benchmark_summary.csv.

    Returns wide columns: location, season, Field, treatment, cultivar, day,
    DAP, LAI, LAI_sd, n.
    """
    wide = _benchmark_wide(["LAI"]).rename(columns={
        "mean__LAI": "LAI", "sd__LAI": "LAI_sd", "n__LAI": "n",
    })
    if location:
        wide = wide[wide["location"] == location]
    return wide.reset_index(drop=True)


def load_obs_sla(location: str = None) -> pd.DataFrame:
    """Load observed specific leaf area (SLA) timeseries from obs_benchmark_summary.csv.

    SLA is field-integrated leaf area / leaf dry weight in m2/kg. Returns wide
    columns: location, season, Field, treatment, cultivar, day, DAP, SLA_obs,
    SLA_obs_sd, n.
    """
    wide = _benchmark_wide(["SLA"]).rename(columns={
        "mean__SLA": "SLA_obs", "sd__SLA": "SLA_obs_sd", "n__SLA": "n",
    })
    if location:
        wide = wide[wide["location"] == location]
    return wide.reset_index(drop=True)


def load_soil_temp(site: str, field: str = "OpenField") -> pd.DataFrame:
    """Load daily control-treatment soil temperature for one site.

    Reads data/obs/{SITE}_soiltemp.csv and returns a tidy daily series for the
    CONTROL treatment only, averaged across plots/replicates:
        columns: season (str), day (datetime), Tsoil_mean
    'season' uses the soil file's own labels (IND: '2022-2023'/'2024-2025';
    NLD: '2024'/'2025'; RSA: '2024'/'2025'). Control treatments (per the phenology
    calibration decision):
        IND = 'Full_0 Days'
        NLD = 'AC'  (ambient control, excludes heated HE/HB plots)
        RSA = '100%PET_Mondial' in 2024 and 'Drip_Mondial' in 2025 for OpenField;
              '100%PET_Mondial' in both seasons for RainShelter (the shelter runs
              the PET treatments in every season).
    'field' selects the RSA setup ('OpenField' or 'RainShelter'); it is ignored for
    IND and NLD, which have one setup each. The RSA file writes the setups as
    'Open Field' / 'Rain Shelter'; both are normalised here.
    """
    path = ROOT / "data" / "obs" / f"{site}_soiltemp.csv"
    df = pd.read_csv(path)
    df["day"] = pd.to_datetime(df["day"])
    df["season"] = df["season"].astype(str)
    if site == "IND":
        df = df[df["treatment"] == "Full_0 Days"]
    elif site == "NLD":
        df = df[df["treatment"] == "AC"]
    elif site == "RSA":
        df["field"] = df["field"].replace({"Open Field": "OpenField",
                                           "Rain Shelter": "RainShelter"})
        df = df[df["field"] == field]
        ctrl = ({"2024": "100%PET_Mondial", "2025": "Drip_Mondial"} if field == "OpenField"
                else {"2024": "100%PET_Mondial", "2025": "100%PET_Mondial"})
        df = df[[t == ctrl.get(s) for s, t in zip(df["season"], df["treatment"])]]
    else:
        raise ValueError(f"Unknown site {site!r}")
    return (df.groupby(["season", "day"], as_index=False)["Tsoil_mean"].mean()
              .sort_values(["season", "day"]).reset_index(drop=True))


# ── Irrigation ───────────────────────────────────────────────────────────────

def load_irrigation_csv(path) -> pd.DataFrame:
    """Load irrigation events from a CSV file.

    Expected columns: Season, Date, DAP, Treatment, mm, Cultivar [, cum_irri]
    The Cultivar column is optional; if absent, a 'cultivar' column of NaN is
    returned (caller must fill it or filter appropriately).
    Rows with mm == 0 or NaN are dropped.

    Returns:
        DataFrame with columns: season (str), day (datetime), cultivar (str),
        treatment (str), mm (float).
    """
    df = pd.read_csv(path)
    df["day"]       = pd.to_datetime(df["Date"])
    df["season"]    = df["season"].astype(str)
    df["mm"]        = pd.to_numeric(df["mm"], errors="coerce")
    df = df.loc[df["mm"].notna() & (df["mm"] > 0)].copy()
    return df[["season", "day", "cultivar", "treatment", "mm"]].reset_index(drop=True)


def build_agro_with_irrigation(
    location: str,
    season: str,
    cultivar: str,
    irr_df: pd.DataFrame,
    efficiency: float = 0.9,
) -> list:
    """Build a PCSE agromanagement object with irrigation TimedEvents injected.

    Takes the base agro YAML for a location/season (from data/agro/*.yaml) and
    replaces the 'TimedEvents: null' placeholder with actual irrigation events
    from *irr_df*, which must contain only the events for ONE treatment.

    Args:
        location:   'IND', 'NLD', or 'RSA'.
        season:     Season key matching the agro YAML (e.g. '2024').
        cultivar:   Variety name substituted into the agro template.
        irr_df:     DataFrame with columns 'day' (datetime) and 'mm' (float),
                    containing the irrigation schedule for a single treatment.
        efficiency: Application efficiency fraction (0–1). Default 0.9.

    Returns:
        Parsed agromanagement list ready for PCSE.
    """
    import datetime as _dt

    seasons = load_seasons(location)
    if season not in seasons:
        raise KeyError(f"Season '{season}' not found in {AGRO_FILE[location]}. "
                       f"Available: {list(seasons)}")

    agro_str = seasons[season]["agro_yaml"].format(cultivar=cultivar)
    agro = yaml.safe_load(agro_str)

    # agro is a list with one {campaign_date: {CropCalendar, TimedEvents, StateEvents}}
    campaign_date = list(agro[0].keys())[0]

    events_table = []
    for _, row in irr_df.sort_values("day").iterrows():
        date_key = row["day"].date() if hasattr(row["day"], "date") else row["day"]
        amount_cm = float(row["mm"]) / 10.0  # mm → cm
        events_table.append({date_key: {"amount": round(amount_cm, 4), "efficiency": efficiency}})

    agro[0][campaign_date]["TimedEvents"] = [{
        "event_signal": "irrigate",
        "name": "Irrigation schedule",
        "comment": "Irrigation amounts in cm",
        "events_table": events_table,
    }]
    return agro
