# %%
# Phenology (DVS) calibration under potential production. Metric scope is DVS only.
#
# The script is a linear record:
#   0  baseline       crop-file template, no calibration
#   1  calibrated     site calibration, then evaluation in the other seasons
#   2  best fit       every season/setup fitted separately; authoritative downstream output
#   D  diagnostic     index 2 with IND/RSA TEFFMX=25 and seasonal TSUMEM; never exported as index 2

import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pcse.models import Wofost73_PP

_here = pathlib.Path().resolve()
_root = next(p for p in (_here, *_here.parents)
             if (p / "scripts").exists() and (p / "data").exists())
sys.path.insert(0, str(_root / "scripts"))

from utils.plotstyle import display_season
from utils.paths import ROOT
from utils.loaders import (
    load_experiments, find_experiment, load_weather, load_soil,
    load_site_data, load_agro, load_crop_data, build_parameters,
    load_obs_experiments, run_days,
)
from utils.metrics import summary_metrics, MAEDecomposition, MSEDecomposition
from utils.phenology_helpers import (
    site_dtsmtb, crop_default, derive_obs_fixed, ind_maturity_dvs,
    rsa_tuber_init, RSA_TI_SEASON, PHENO_RUN_ID,
)
from utils import plotstyle
from utils.plotstyle import before_after, grey_ramp, use_style, SITE_MARKER, FIELD_FILLED
from optimization.config import OptimizationConfig, load_param_specs
from optimization.launcher import OptimizationLauncher


# %%
# -- 1. Analysis controls and the ten experiments -----------------------------
plotstyle.MODE = "paper"       # "slides" for presentation colours
OUTPUT_DIR = ROOT / "output" / "phenology"
OUTPUT_DIR.mkdir(exist_ok=True)
RUN_ID = PHENO_RUN_ID          # shared with all downstream consumers
CROP_DIR = "data/crops_calibration"
cropd = load_crop_data(CROP_DIR)

WARM_SITE_TEFFMX_CAP = 25.0
DIAGNOSTIC_SITES = ("IND", "RSA")
SITE_ORDER = ("NLD", "IND", "RSA")
PARAMETER_COLUMNS = ["TBASEM", "TEFFMX", "TSUMEM", "TSUM1", "TSUM2"]
KEY_COLUMNS = ["location", "season", "field", "cultivar"]

BASELINE = "baseline"
CALIBRATED = "calibrated"
BEST_FIT = "best fit"
FIXED25 = "TEFFMX = 25, seasonal TSUMEM"
SCENARIO_ORDER = [BASELINE, CALIBRATED, BEST_FIT, FIXED25]

# One visible row per modelled experiment. No nested site/season dictionaries.
df_experiments = pd.DataFrame([
    ("NLD", "2024",      "OpenField",   "Innovator", "2025",      "evaluation",  "TBASEM"),
    ("NLD", "2025",      "OpenField",   "Innovator", "2025",      "calibration", "TBASEM"),
    ("IND", "2022-2023", "OpenField",   "Kennebec",  "2024-2025", "evaluation",  "TEFFMX"),
    ("IND", "2024-2025", "OpenField",   "Kennebec",  "2024-2025", "calibration", "TEFFMX"),
    ("RSA", "2024",      "OpenField",   "Mondial",   "2024",      "calibration", "TEFFMX"),
    ("RSA", "2025-A",    "OpenField",   "Mondial",   "2024",      "evaluation",  "TEFFMX"),
    ("RSA", "2025-S",    "OpenField",   "Mondial",   "2024",      "evaluation",  "TEFFMX"),
    ("RSA", "2024",      "RainShelter", "Mondial",   "2024",      "calibration", "TEFFMX"),
    ("RSA", "2025-A",    "RainShelter", "Mondial",   "2024",      "evaluation",  "TEFFMX"),
    ("RSA", "2025-S",    "RainShelter", "Mondial",   "2024",      "evaluation",  "TEFFMX"),
], columns=KEY_COLUMNS + ["calibration_season", "role", "search_param"])
df_experiments["warm_site"] = df_experiments["location"].isin(DIAGNOSTIC_SITES)

assert len(df_experiments) == 10
assert not df_experiments.duplicated(KEY_COLUMNS).any()
print("Experiment ledger:")
print(df_experiments.to_string(index=False))

IND_DTSMTB = site_dtsmtb("IND", "2022-2023", "Kennebec")

# %%
# -- 2. Observed phenology as one long dataframe ------------------------------
_STAGE_DVS = {"Emergence": 0.0, "TuberInitiation": 1.0, "Maturity": 2.0}
df_obs_wide = load_obs_experiments().copy()
df_obs_wide["field"] = df_obs_wide["field"].replace("Rainshelter", "RainShelter")
df_obs_wide["season_key"] = df_obs_wide["season_key"].astype(str)

df_obs_wide = (df_experiments[KEY_COLUMNS]
               .merge(
                   df_obs_wide[["location", "season_key", "field", "cultivar",
                                "Start", "Emergence", "TuberInitiation", "Maturity"]],
                   left_on=KEY_COLUMNS,
                   right_on=["location", "season_key", "field", "cultivar"],
                   how="left", validate="one_to_one")
               .drop(columns="season_key"))
assert df_obs_wide[["Start", "Emergence"]].notna().all().all()

df_observations = (df_obs_wide
                   .melt(id_vars=KEY_COLUMNS, value_vars=list(_STAGE_DVS),
                         var_name="stage", value_name="day")
                   .dropna(subset=["day"])
                   .reset_index(drop=True))
df_observations["day"] = pd.to_datetime(df_observations["day"])
df_observations["obs_dvs_raw"] = df_observations["stage"].map(_STAGE_DVS).astype(float)
df_observations["synthetic_anchor"] = False

# RSA has no recorded tuber-initiation date. The first tuber-bearing 2024 harvest is an upper
# bound and belongs in evaluation figures, but not in the optimizer observation vector.
df_rsa_ti = df_experiments[
    (df_experiments["location"] == "RSA")
    & (df_experiments["season"] == RSA_TI_SEASON)
][KEY_COLUMNS].copy()
df_rsa_ti["stage"] = "TuberInitiation"
df_rsa_ti["day"] = pd.to_datetime(rsa_tuber_init())
df_rsa_ti["obs_dvs_raw"] = 1.0
df_rsa_ti["synthetic_anchor"] = True
df_observations = pd.concat([df_observations, df_rsa_ti], ignore_index=True)
assert not df_observations.duplicated(KEY_COLUMNS + ["stage"]).any()

print(df_observations[KEY_COLUMNS + ["stage", "day", "obs_dvs_raw",
                                     "synthetic_anchor"]].to_string(index=False))


# %%
# -- 3. The mechanical model wrapper -------------------------------------
def run_pp(location, season, cultivar, overrides):
    """Run Wofost73_PP once and return its daily output dataframe."""
    exp = find_experiment(load_experiments(location), location, season, cultivar)
    weather = load_weather(exp["weather_file"])
    soil = load_soil(exp["soil_file"])
    site = load_site_data(exp["WAV"], exp["CO2"])
    agro = load_agro(location, season, cultivar)
    params = build_parameters(cropd, soil, site, cultivar, overrides)
    model = Wofost73_PP(params, weather, agro)
    days = run_days(location, season)
    model.run(days) if days else model.run_till_terminate()
    df = pd.DataFrame(model.get_output())
    df["day"] = pd.to_datetime(df["day"])
    return df


# %%
# -- 4. Every optimization job in one inspectable table ----------------------
# IND/RSA TSUMEM is always derived with the 25 C cap, including for the jobs that search TEFFMX.
# This keeps the observed-emergence calculation consistent across calibrated, best-fit and
# fixed-25 scenarios. TEFFMX itself remains the active search parameter.
df_calibration_jobs = df_experiments[
    (df_experiments["field"] == "OpenField")
    & (df_experiments["season"] == df_experiments["calibration_season"])
].copy()
df_calibration_jobs["search_scope"] = CALIBRATED
df_calibration_jobs["target_suffix"] = "cal"
df_calibration_jobs.loc[
    df_calibration_jobs["warm_site"], "target_suffix"
] += "-tsumem25"

df_best_fit_jobs = df_experiments.copy()
df_best_fit_jobs["search_scope"] = BEST_FIT
df_best_fit_jobs["target_suffix"] = np.where(
    df_best_fit_jobs["field"] == "OpenField",
    "all-" + df_best_fit_jobs["season"],
    "RS-all-" + df_best_fit_jobs["season"],
)
df_best_fit_jobs.loc[
    df_best_fit_jobs["warm_site"], "target_suffix"
] += "-tsumem25"

df_search_jobs = pd.concat([df_calibration_jobs, df_best_fit_jobs], ignore_index=True)
df_search_jobs[["TSUMEM", "TSUM1", "TSUM2"]] = df_search_jobs.apply(
    lambda row: pd.Series(derive_obs_fixed(
        row["location"], row["cultivar"], [row["season"]],
        teffmx=WARM_SITE_TEFFMX_CAP if row["warm_site"] else None,
        field=row["field"])),
    axis=1,
)

print("Optimization jobs and fixed thermal sums:")
print(df_search_jobs[["search_scope"] + KEY_COLUMNS + ["search_param", "TSUMEM",
                                                        "TSUM1", "TSUM2"]].to_string(index=False))

# %%
# -- 5. Run the one-parameter searches ----------------------------------------
search_records = []
for job in df_search_jobs.itertuples(index=False):
    fixed = {"TSUMEM": job.TSUMEM, "TSUM1": job.TSUM1, "TSUM2": job.TSUM2}

    # Exact observations used by this job. IND haulm-kill DVS is corrected visibly here.
    obs_job = df_observations[
        (~df_observations["synthetic_anchor"])
        & (df_observations["location"] == job.location)
        & (df_observations["season"] == job.season)
        & (df_observations["field"] == job.field)
        & (df_observations["cultivar"] == job.cultivar)
    ][KEY_COLUMNS + ["day", "stage", "obs_dvs_raw"]].copy()
    obs_job = obs_job.rename(columns={"obs_dvs_raw": "obs_dvs"})
    if job.location == "IND":
        maturity_dvs = ind_maturity_dvs(
            job.season, job.cultivar, job.TSUM1, job.TSUM2, IND_DTSMTB)
        obs_job.loc[obs_job["stage"] == "Maturity", "obs_dvs"] = maturity_dvs

    config = OptimizationConfig(
        target=f"{job.location}-{job.cultivar}-{job.target_suffix}",
        location=job.location, cultivar=job.cultivar, seasons=(job.season,),
        active_params=load_param_specs([job.search_param]),
        fixed_overrides=fixed, obs_field=job.field, crop_dir=CROP_DIR, run_id=RUN_ID,
        output_root=OUTPUT_DIR / "optimization",
        max_evals_stage1=500, max_evals_stage2=50, ftol_rel=1e-4,
        cache=True, seed=42,
    )
    launcher = OptimizationLauncher(config, observations=obs_job)
    _, result = launcher.run(resume=True)
    search_records.append({
        "search_scope": job.search_scope,
        **{key: getattr(job, key) for key in KEY_COLUMNS},
        "search_param": job.search_param,
        "searched_value": result.overrides[job.search_param],
        "objective_RMSE": result.f_best,
        "TSUMEM": job.TSUMEM, "TSUM1": job.TSUM1, "TSUM2": job.TSUM2,
        "target": config.target, "converged": result.converged,
    })

df_search_results = pd.DataFrame(search_records)
print("Optimization results:")
print(df_search_results.to_string(index=False))
df_search_results.to_csv(OUTPUT_DIR / f"phenology_search_results_{RUN_ID}.csv", index=False)


# %%
# -- 6. Scenario 1: site calibration carried to other seasons ----------------
df_params_calibrated = df_experiments.copy()
df_cal_search = df_search_results[df_search_results["search_scope"] == CALIBRATED][
    ["location", "searched_value", "objective_RMSE"]
]
df_params_calibrated = df_params_calibrated.merge(
    df_cal_search, on="location", how="left", validate="many_to_one")

df_params_calibrated[["TSUMEM", "TSUM1", "TSUM2"]] = df_params_calibrated.apply(
    lambda row: pd.Series(derive_obs_fixed(
        row["location"], row["cultivar"], [row["calibration_season"]],
        teffmx=WARM_SITE_TEFFMX_CAP if row["warm_site"] else None,
        field=row["field"])),
    axis=1,
)

# NLD 2024 uses its own emergence window; carrying 2025 TSUMEM emerges about a week early.
_nld_2024 = ((df_params_calibrated["location"] == "NLD")
             & (df_params_calibrated["season"] == "2024"))
df_params_calibrated.loc[_nld_2024, "TSUMEM"] = df_params_calibrated.loc[
    _nld_2024
].apply(lambda row: derive_obs_fixed(
    row["location"], row["cultivar"], [row["season"]], teffmx=None,
    field=row["field"])["TSUMEM"], axis=1)

df_params_calibrated["TBASEM_default"] = df_params_calibrated.apply(
    lambda row: crop_default(row["location"], row["season"], row["cultivar"], "TBASEM"), axis=1)
df_params_calibrated["TEFFMX_default"] = df_params_calibrated.apply(
    lambda row: crop_default(row["location"], row["season"], row["cultivar"], "TEFFMX"), axis=1)
df_params_calibrated["TBASEM"] = np.where(
    df_params_calibrated["search_param"] == "TBASEM",
    df_params_calibrated["searched_value"], df_params_calibrated["TBASEM_default"])
df_params_calibrated["TEFFMX"] = np.where(
    df_params_calibrated["warm_site"], WARM_SITE_TEFFMX_CAP,
    df_params_calibrated["TEFFMX_default"])
df_params_calibrated["scenario"] = CALIBRATED

print("Calibrated parameter rows used in every season:")
print(df_params_calibrated[KEY_COLUMNS + PARAMETER_COLUMNS].to_string(index=False))


# %%
# -- 7. Scenario 2: authoritative per-season best fit (index 2) ---------------
df_best_search = df_search_results[df_search_results["search_scope"] == BEST_FIT].copy()
df_params_best_fit = df_experiments.merge(
    df_best_search[KEY_COLUMNS + ["searched_value", "objective_RMSE",
                                  "TSUMEM", "TSUM1", "TSUM2"]],
    on=KEY_COLUMNS, how="left", validate="one_to_one")
df_params_best_fit["TBASEM_default"] = df_params_best_fit.apply(
    lambda row: crop_default(row["location"], row["season"], row["cultivar"], "TBASEM"), axis=1)
df_params_best_fit["TEFFMX_default"] = df_params_best_fit.apply(
    lambda row: crop_default(row["location"], row["season"], row["cultivar"], "TEFFMX"), axis=1)
df_params_best_fit["TBASEM"] = np.where(
    df_params_best_fit["search_param"] == "TBASEM",
    df_params_best_fit["searched_value"], df_params_best_fit["TBASEM_default"])
df_params_best_fit["TEFFMX"] = np.where(
    df_params_best_fit["search_param"] == "TEFFMX",
    df_params_best_fit["searched_value"], df_params_best_fit["TEFFMX_default"])
df_params_best_fit["scenario"] = BEST_FIT

print("Authoritative index-2 best-fit parameter rows:")
print(df_params_best_fit[KEY_COLUMNS + PARAMETER_COLUMNS].to_string(index=False))


# %%
# -- 8. Diagnostic only: TEFFMX=25 with seasonal TSUMEM -----------------------
df_params_fixed25 = df_params_best_fit[
    df_params_best_fit["location"].isin(DIAGNOSTIC_SITES)
].copy()
df_params_fixed25 = df_params_fixed25.join(
    df_params_fixed25[PARAMETER_COLUMNS].add_prefix("best_fit_"))

df_params_fixed25["TEFFMX"] = WARM_SITE_TEFFMX_CAP
df_params_fixed25["TSUMEM"] = df_params_fixed25.apply(
    lambda row: derive_obs_fixed(
        row["location"], row["cultivar"], [row["season"]],
        teffmx=WARM_SITE_TEFFMX_CAP, field=row["field"])["TSUMEM"],
    axis=1,
)
df_params_fixed25["scenario"] = FIXED25

assert len(df_params_fixed25) == 8
assert not (df_params_fixed25["location"] == "NLD").any()
assert np.isclose(df_params_fixed25["TEFFMX"], WARM_SITE_TEFFMX_CAP).all()
assert np.isfinite(df_params_fixed25["TSUMEM"]).all() and (df_params_fixed25["TSUMEM"] > 0).all()
assert np.allclose(df_params_fixed25["TSUMEM"], df_params_fixed25["best_fit_TSUMEM"])
assert np.allclose(
    df_params_fixed25[["TBASEM", "TSUM1", "TSUM2"]],
    df_params_fixed25[["best_fit_TBASEM", "best_fit_TSUM1", "best_fit_TSUM2"]],
)

print("Diagnostic fixed-25 parameter rows (not exported as index 2):")
print(df_params_fixed25[KEY_COLUMNS + PARAMETER_COLUMNS].to_string(index=False))

# %%
# -- 9. One visible run plan, then one flat PCSE loop -------------------------
df_params_baseline = df_experiments.copy()
df_params_baseline[PARAMETER_COLUMNS] = np.nan
df_params_baseline["scenario"] = BASELINE
df_params_baseline["role"] = "baseline"

df_params_best_fit["role"] = "fit"
df_params_fixed25["role"] = "diagnostic"

_run_columns = KEY_COLUMNS + ["role", "scenario"] + PARAMETER_COLUMNS
df_run_plan = pd.concat([
    df_params_baseline[_run_columns],
    df_params_calibrated[_run_columns],
    df_params_best_fit[_run_columns],
    df_params_fixed25[_run_columns],
], ignore_index=True)
df_run_plan["scenario_order"] = df_run_plan["scenario"].map(
    {scenario: i for i, scenario in enumerate(SCENARIO_ORDER)})

simulation_frames = []
for run in df_run_plan.itertuples(index=False):
    overrides = {parameter: float(getattr(run, parameter))
                 for parameter in PARAMETER_COLUMNS
                 if pd.notna(getattr(run, parameter))}
    df_sim = run_pp(run.location, run.season, run.cultivar, overrides)
    df_sim["location"] = run.location
    df_sim["season"] = run.season
    df_sim["field"] = run.field
    df_sim["cultivar"] = run.cultivar
    df_sim["role"] = run.role
    df_sim["scenario"] = run.scenario
    simulation_frames.append(df_sim)

df_simulations = pd.concat(simulation_frames, ignore_index=True)
print(f"Completed {len(df_run_plan)} scenario runs; daily results are in df_simulations.")


# %%
# -- 10. Scenario-specific observations and paired DVS ------------------------
# | Site | Experiments | Scenarios per experiment | Observations | Result |
# |---|---:|---:|---:|---:|
# | NLD | 2 | 3 | 3 | 18 |
# | IND | 2 | 4 | 3 | 24 |
# | RSA 2024 | 2 setups | 4 | 3, including synthetic tuber initiation | 24 |
# | RSA 2025 | 4 season/setup combinations | 4 | 2 | 32 |
# | Total | | | | 98 | 
df_observations_scenario = df_run_plan[
    KEY_COLUMNS + ["role", "scenario", "TSUM1", "TSUM2"]
].merge(df_observations, on=KEY_COLUMNS, how="inner")
df_observations_scenario["obs_dvs_used"] = df_observations_scenario["obs_dvs_raw"]
assert len(df_observations_scenario) == 98
# Baseline keeps literal DVS=2, preserving the existing index-0 metric contract. Calibrated,
# best-fit and fixed-25 IND runs use theoretical DVS at the haulm-kill date. The diagnostic copies
# best-fit TSUM1/TSUM2, so both are compared against exactly the same IND anchor.
_correct_ind = ((df_observations_scenario["location"] == "IND")
                & (df_observations_scenario["stage"] == "Maturity")
                & (df_observations_scenario["scenario"] != BASELINE))
df_observations_scenario.loc[_correct_ind, "obs_dvs_used"] = (
    df_observations_scenario.loc[_correct_ind].apply(
        lambda row: ind_maturity_dvs(
            row["season"], row["cultivar"], row["TSUM1"], row["TSUM2"], IND_DTSMTB),
        axis=1,
    )
)

_pair_keys = KEY_COLUMNS + ["role", "scenario", "day"]
df_pairs = df_observations_scenario.merge(
    df_simulations[_pair_keys + ["DVS"]], on=_pair_keys, how="left",
    validate="many_to_many",
).dropna(subset=["obs_dvs_used", "DVS"])


# %%
# -- 11. Metrics: all scenarios in memory, only indices 0-2 in the main CSV --
_MAE_KEYS = ["MAE", "MAE_b", "MAE_p", "MAE_u", "MBE", "slope", "intercept"]
_MSE_KEYS = ["MSE", "Bias", "Bias_component", "Variance", "Distribution", "Sequence"]
metric_records = []
for keys, group in df_pairs.groupby(
        ["scenario", "location", "season", "field", "role"], sort=False):
    scenario, location, season, field, role = keys
    obs = group["obs_dvs_used"].to_numpy()
    sim = group["DVS"].to_numpy()
    record = {
        "scenario": scenario, "location": location, "year": season,
        "field": field, "role": role, "n": len(group),
    }
    record.update(summary_metrics(obs, sim))
    try:
        record.update(MAEDecomposition(obs, sim).as_dict())
    except ValueError:
        record.update({key: np.nan for key in _MAE_KEYS})
    try:
        record.update(MSEDecomposition(obs, sim).as_dict())
    except ValueError:
        record.update({key: np.nan for key in _MSE_KEYS})
    metric_records.append(record)

df_metrics_all = pd.DataFrame(metric_records)
df_metrics_all["index"] = df_metrics_all["scenario"].map(
    {BASELINE: 0, CALIBRATED: 1, BEST_FIT: 2})

_metric_columns = (["index", "location", "year", "field", "role", "n",
                    "RMSE", "NRMSE", "MAE", "EF", "d"]
                   + _MAE_KEYS[1:] + _MSE_KEYS)
df_metrics = (df_metrics_all[df_metrics_all["scenario"] != FIXED25]
              .reindex(columns=_metric_columns)
              .sort_values(["index", "location", "year", "field"])
              .reset_index(drop=True)
              .round(3))
print("Authoritative metrics (indices 0-2):")
print(df_metrics.to_string(index=False))
df_metrics.to_csv(OUTPUT_DIR / f"phenology_metrics_{RUN_ID}.csv", index=False)


# %%
# -- 12. Authoritative parameter CSV: calibrated index 1 + best-fit index 2 --
df_source_1 = pd.DataFrame([
    ("IND", "OpenField",
     "TSUMEM ~ TEFFMX = 25"
     "TSUM2 = 1700 - TSUM1"),
    ("NLD", "OpenField",
     "default TBASEM and TEFFMX"),
    ("RSA", "OpenField",
     "TSUMEM ~ TEFFMX 25; TSUM1 at harvest"),
    ("RSA", "RainShelter",
     "TSUMEM ~ TEFFMX = 25"
     "TSUM1/TSUM2 observed "),
], columns=["location", "field", "note"])

df_source_2 = pd.DataFrame([
    ("IND", "OpenField",
     "overfitted"),
    ("NLD", "OpenField",
     "overfitted"),
    ("RSA", "OpenField",
     "overfitted"),
    ("RSA", "RainShelter",
     "overfitted"),
], columns=["location", "field", "note"])

df_export_1 = df_params_calibrated[
    df_params_calibrated["season"] == df_params_calibrated["calibration_season"]
][KEY_COLUMNS + PARAMETER_COLUMNS].copy()
df_export_1["index"] = 1
df_export_1 = df_export_1.merge(df_source_1, on=["location", "field"], validate="one_to_one")

df_export_2 = df_params_best_fit[KEY_COLUMNS + PARAMETER_COLUMNS].copy()
df_export_2["index"] = 2
df_export_2 = df_export_2.merge(df_source_2, on=["location", "field"], validate="many_to_one")

assert len(df_export_1) == 4
assert len(df_export_2) == 10
_active_best_fit = np.where(
    df_params_best_fit["search_param"] == "TEFFMX",
    df_params_best_fit["TEFFMX"], df_params_best_fit["TBASEM"])
assert np.allclose(_active_best_fit, df_params_best_fit["searched_value"])

df_params = pd.concat([df_export_1, df_export_2], ignore_index=True)
df_params = df_params.rename(columns={"location": "site", "season": "year"})
df_params[["TBASEM", "TEFFMX"]] = df_params[["TBASEM", "TEFFMX"]].round(2)
df_params[["TSUMEM", "TSUM1", "TSUM2"]] = df_params[
    ["TSUMEM", "TSUM1", "TSUM2"]
].round(1)
df_params = (df_params[["index", "site", "cultivar", "year", "field"]
                       + PARAMETER_COLUMNS + ["note"]]
             .sort_values(["index", "site", "year", "field"])
             .reset_index(drop=True))
assert set(df_params["index"]) == {1, 2}
assert len(df_params[df_params["index"] == 2]) == 10
print("Authoritative parameter export (diagnostic excluded):")
print(df_params.to_string(index=False))
df_params.to_csv(OUTPUT_DIR / f"phenology_params_{RUN_ID}.csv", index=False)


# %%
# -- 13. Diagnostic comparison CSV --------------------------------------------
df_diagnostic = df_params_fixed25[
    KEY_COLUMNS + [f"best_fit_{parameter}" for parameter in PARAMETER_COLUMNS]
    + PARAMETER_COLUMNS
].copy()
df_diagnostic = df_diagnostic.rename(columns={
    "season": "year", "TEFFMX": "check_TEFFMX", "TSUMEM": "check_TSUMEM",
})

df_best_metrics = df_metrics_all[df_metrics_all["scenario"] == BEST_FIT][
    ["location", "year", "field", "RMSE", "MAE"]
].rename(columns={"RMSE": "best_fit_RMSE", "MAE": "best_fit_MAE"})
df_check_metrics = df_metrics_all[df_metrics_all["scenario"] == FIXED25][
    ["location", "year", "field", "RMSE", "MAE"]
].rename(columns={"RMSE": "check_RMSE", "MAE": "check_MAE"})
df_diagnostic = (df_diagnostic
                 .merge(df_best_metrics, on=["location", "year", "field"], validate="one_to_one")
                 .merge(df_check_metrics, on=["location", "year", "field"], validate="one_to_one"))
df_diagnostic["RMSE_change"] = df_diagnostic["check_RMSE"] - df_diagnostic["best_fit_RMSE"]
df_diagnostic["MAE_change"] = df_diagnostic["check_MAE"] - df_diagnostic["best_fit_MAE"]
df_diagnostic = df_diagnostic.sort_values(["location", "year", "field"]).reset_index(drop=True)
print("Best fit versus fixed-25 diagnostic:")
print(df_diagnostic.round(3).to_string(index=False))
df_diagnostic.round(3).to_csv(
    OUTPUT_DIR / f"phenology_teffmx_check_{RUN_ID}.csv", index=False)


# %%
# -- 14. Figure 1: DVS over time, plotted manually by location ----------------
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

STEP_COLORS = {
    BASELINE: before_after("before")["color"],
    CALIBRATED: plotstyle.SERIES_COLORS[4],
    BEST_FIT: before_after("after")["color"],
    FIXED25: plotstyle.SERIES_COLORS[2],
}
OBS_COLOR = plotstyle.OBS_COLOR
df_obs_best_fit = df_observations_scenario[
    df_observations_scenario["scenario"] == BEST_FIT
]
DVS_YLIM = (-0.1, 2.2)


# NLD: baseline and per-season best fit. The fixed-25 diagnostic does not apply.
df_plot_nld = df_simulations[
    (df_simulations["location"] == "NLD")
    & (df_simulations["scenario"].isin([BASELINE, BEST_FIT]))
].copy()
df_obs_nld = df_obs_best_fit[df_obs_best_fit["location"] == "NLD"].copy()

use_style()
fig_nld, axes_nld = plt.subplots(1, 2, figsize=(6.0, 3.4), sharey=True)
for ax, season in zip(axes_nld, ["2024", "2025"]):
    season_sim = df_plot_nld[df_plot_nld["season"] == season]
    for scenario in (BASELINE, BEST_FIT):
        line = season_sim[season_sim["scenario"] == scenario].sort_values("day")
        ax.plot(line["day"], line["DVS"], color=STEP_COLORS[scenario],
                linewidth=1.7, label=scenario)

    season_obs = df_obs_nld[df_obs_nld["season"] == season]
    ax.scatter(season_obs["day"], season_obs["obs_dvs_used"], s=30,
               color=OBS_COLOR, edgecolor="white", linewidth=0.5,
               zorder=3, label="observed")

    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_title(display_season(season))
    ax.set_xlabel("Date")
    ax.set_ylim(DVS_YLIM)
    ax.grid(alpha=0.2)

axes_nld[0].set_ylabel("DVS")
handles, labels = axes_nld[0].get_legend_handles_labels()
fig_nld.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
plt.show()

#%%
# IND: add the fixed-25 diagnostic beside baseline and per-season best fit.
## Best fit and fixed teffmx at 25 give almost identifcal results
df_plot_ind = df_simulations[
    (df_simulations["location"] == "IND")
    & (df_simulations["scenario"].isin([BASELINE, BEST_FIT, FIXED25]))
].copy()
df_obs_ind = df_obs_best_fit[df_obs_best_fit["location"] == "IND"].copy()

use_style()
fig_ind, axes_ind = plt.subplots(1, 2, figsize=(6, 3.4), sharey=True)
for ax, season in zip(axes_ind, ["2022-2023", "2024-2025"]):
    season_sim = df_plot_ind[df_plot_ind["season"] == season]
    for scenario in (BASELINE, BEST_FIT, FIXED25):
        line = season_sim[season_sim["scenario"] == scenario].sort_values("day")
        ax.plot(line["day"], line["DVS"], color=STEP_COLORS[scenario],
                linewidth= 0.8, label=scenario)

    season_obs = df_obs_ind[df_obs_ind["season"] == season]
    ax.scatter(season_obs["day"], season_obs["obs_dvs_used"], s=30,
               color=OBS_COLOR, edgecolor="white", 
               zorder=3, label="literature informed")

    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_title(display_season(season))
    ax.set_xlabel("Date")
    ax.set_ylim(DVS_YLIM)
    ax.grid(alpha=0.2)

axes_ind[0].set_ylabel("DVS")
handles, labels = axes_ind[0].get_legend_handles_labels()
fig_ind.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
plt.show()
# %%

# RSA: the two field setups share panels; colour identifies the scenario and line style the setup.
df_plot_rsa = df_simulations[
    (df_simulations["location"] == "RSA")
    & (df_simulations["scenario"].isin([BASELINE, CALIBRATED, BEST_FIT, FIXED25]))
].copy()
df_obs_rsa = df_obs_best_fit[df_obs_best_fit["location"] == "RSA"].copy()

use_style()
fig_rsa, axes_rsa = plt.subplots(1, 3, figsize=(7, 3.4), sharey=True)
for ax, season in zip(axes_rsa, ["2024", "2025-A", "2025-S"]):
    season_sim = df_plot_rsa[df_plot_rsa["season"] == season]
    for field, linestyle in (("OpenField", "-"), ("RainShelter", "--")):
        field_sim = season_sim[season_sim["field"] == field]
        for scenario in (BASELINE, CALIBRATED, BEST_FIT, FIXED25):
            line = field_sim[field_sim["scenario"] == scenario].sort_values("day")
            ax.plot(line["day"], line["DVS"], color=STEP_COLORS[scenario],
                    linestyle=linestyle, linewidth=1.7)

        field_obs = df_obs_rsa[
            (df_obs_rsa["season"] == season) & (df_obs_rsa["field"] == field)
        ]
        ax.scatter(
            field_obs["day"], field_obs["obs_dvs_used"], s=30,
            facecolor=OBS_COLOR if field == "OpenField" else "none",
            edgecolor=OBS_COLOR, linewidth=0.9, zorder=3,
        )

    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_title(display_season(season))
    ax.set_xlabel("Date")
    ax.set_ylim(DVS_YLIM)
    ax.grid(alpha=0.2)
axes_rsa[2].annotate("The perfect fit \nbecause of \nTSUM2 in spring \n> " \
"\nTSUM2 in autumn.\nchecked mult times\nThe photoperiod test \nwas using " \
"Autumn TSUM2",
                      xy=(0.5, 0.3), xycoords="axes fraction",
                     ha="center", va="bottom", fontsize=9, color="0.5")
axes_rsa[0].set_ylabel("DVS")
rsa_legend = [
    Line2D([0], [0], color=STEP_COLORS[BASELINE], linewidth=1.7, label=BASELINE),
    Line2D([0], [0], color=STEP_COLORS[CALIBRATED], linewidth=1.7, label=CALIBRATED),
    Line2D([0], [0], color=STEP_COLORS[BEST_FIT], linewidth=1.7, label=BEST_FIT),
    Line2D([0], [0], color=STEP_COLORS[FIXED25], linewidth=1.7, label=FIXED25),
    Line2D([0], [0], color="0.35", linestyle="-", label="OpenField"),
    Line2D([0], [0], color="0.35", linestyle="--", label="RainShelter"),
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=OBS_COLOR,
           markeredgecolor=OBS_COLOR, label="observed OpenField"),
    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
           markeredgecolor=OBS_COLOR, label="observed RainShelter"),
]
fig_rsa.legend(handles=rsa_legend, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=4, frameon=False)
plt.show()


# %%
# -- 15. Figure 2: simulated versus observed DVS, all scenarios ---------------
from utils.visualise import plot_sim_vs_obs_grid

plot_sim_vs_obs_grid(
    df_pairs, obs_col="obs_dvs_used", sim_col="DVS", hue="scenario",
    hue_colors=STEP_COLORS, field_col="field",
    xlabel="Observed DVS", ylabel="Simulated DVS",
    output=True, OUTPUT_DIR=OUTPUT_DIR, RUN_ID=RUN_ID,
)
plt.show()


# %%
# -- 16. Figure 3: pooled DVS error decomposition, all scenarios --------------
from utils.visualise import plot_decomposition_sites

pooled = {}
for scenario in SCENARIO_ORDER:
    pooled[scenario] = {}
    scenario_pairs = df_pairs[df_pairs["scenario"] == scenario]
    for (location, field), group in scenario_pairs.groupby(["location", "field"], sort=False):
        label = location if field == "OpenField" else f"{location} RS"
        pooled[scenario][label] = (
            group["obs_dvs_used"].to_numpy(), group["DVS"].to_numpy())

_baseline_errors = [sim - obs for obs, sim in pooled[BASELINE].values()]
MAE_YLIM = (0, max(float(np.mean(np.abs(error))) for error in _baseline_errors) * 1.1)
MSE_YLIM = (0, max(float(np.mean(error ** 2)) for error in _baseline_errors) * 1.1)

df_decomposition_outputs = pd.DataFrame([
    (BASELINE, "baseline", "Baseline DVS error (all seasons)"),
    (CALIBRATED, "calibrated", "Calibrated DVS error (calibration transfer)"),
    (BEST_FIT, "bestfit", "Best-fit DVS error (per-season fit)"),
    (FIXED25, "teffmx_check", "Fixed TEFFMX = 25 DVS error (seasonal TSUMEM)"),
], columns=["scenario", "filename", "title"])

for output in df_decomposition_outputs.itertuples(index=False):
    fig = plot_decomposition_sites(
        pooled[output.scenario], title=output.title,
        mae_ylim=MAE_YLIM, mse_ylim=MSE_YLIM)
    fig.savefig(OUTPUT_DIR / f"dvs_decomp_{output.filename}_{RUN_ID}.png",
                dpi=300, bbox_inches="tight")
    plt.show()


# %%
# -- 17. Figure 4: NRMSE through every scientific scenario --------------------
METRIC = "NRMSE"
df_trend = df_metrics_all[
    (df_metrics_all["role"] != "calibration") | (df_metrics_all["MAE"] != 0)
].copy()
df_trend["scenario_order"] = df_trend["scenario"].map(
    {scenario: i for i, scenario in enumerate(SCENARIO_ORDER)})

fig, axes = plt.subplots(1, len(SITE_ORDER), figsize=(3.6 * len(SITE_ORDER), 3.4), sharey=True)
for ax, site in zip(np.atleast_1d(axes), SITE_ORDER):
    site_metrics = df_trend[df_trend["location"] == site]
    seasons = df_experiments[df_experiments["location"] == site]["season"].drop_duplicates()
    season_grey = dict(zip(seasons, grey_ramp(len(seasons))))
    for (season, field), group in site_metrics.groupby(["year", "field"], sort=False):
        group = group.sort_values("scenario_order")
        filled = FIELD_FILLED.get(field, True)
        ax.plot(
            group["scenario_order"], group[METRIC],
            color=season_grey.get(season, "black"),
            marker=SITE_MARKER.get(site, "o"),
            markerfacecolor=season_grey.get(season, "black") if filled else "none",
            linestyle="-" if filled else "--",
            label=f"{display_season(season)}" + ("" if filled else " RS"),
        )
    ax.set_xticks(range(len(SCENARIO_ORDER)))
    ax.set_xticklabels(SCENARIO_ORDER, rotation=30, ha="right")
    ax.set_title(site)
    ax.legend(fontsize=7)
np.atleast_1d(axes)[0].set_ylabel(METRIC)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / f"dvs_{METRIC.lower()}_by_step_{RUN_ID}.svg", bbox_inches="tight")
plt.show()
print("Figures written to output/phenology/.")
