# %% AMAX: NLD2025 DVS fit; NLD2024 leaf-temperature comparison in the appendix
# A/Ci estimates are excluded. NLD2024 is retained only for the temperature appendix.
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next(p for p in (_here, *_here.parents)
            if (p / "scripts").is_dir() and (p / "data").is_dir())
sys.path.insert(0, str(ROOT / "scripts"))
from manuscript_common import (
    SOURCE_DIR, MAIN_FIG_DIR, SUPP_FIG_DIR, NUMBER_DIR, MANUSCRIPT_RC,
    MODEL_COLOR,
)

for directory in (SOURCE_DIR, MAIN_FIG_DIR, SUPP_FIG_DIR, NUMBER_DIR):
    directory.mkdir(parents=True, exist_ok=True)
TSUM1, TSUM2 = 187.0, 2100.0  # Original exploratory DVS clock, not a physiological fit.
AMAX_CONVERSION = 44.01e-9 * 1e4 * 3600  # 1 umol CO2 m-2 leaf s-1 = 1.58436 kg CO2 ha-1 leaf h-1.


# %% NLD2025 light-response fits only; preserve individual source IDs
# The legacy export can contain other campaigns; filter before any fitting.
df_amax = pd.read_csv(ROOT / "output" / "licor_amax_wofost_fixed_rd.csv",
                      parse_dates=["date", "emergence_date"])
df_amax = df_amax[df_amax["season"].eq(2025)].drop_duplicates("source_id").copy()
df_amax["site"], df_amax["cultivar"] = "NLD", "Innovator"
df_amax["gdd_final"] = df_amax["gdd_emergence"]
df_amax["DVS_final"] = np.where(df_amax["gdd_final"] <= TSUM1,
                                df_amax["gdd_final"] / TSUM1,
                                1 + (df_amax["gdd_final"] - TSUM1) / TSUM2)
df_fit = df_amax.dropna(subset=["DVS_final", "AMAX_wofost"]).copy()
print(f"NLD2025: {len(df_fit)} light-response fits; no A/Ci estimates in the AMAX fit.")


# %% One pooled plateau-and-decline fit; no cultivar or year sensitivity fits
x, y = df_fit["DVS_final"].to_numpy(), df_fit["AMAX_wofost"].to_numpy()
fit = least_squares(lambda b: b[0] - b[2] * np.maximum(0, x - b[1]) - y,
                    x0=[y.mean(), 1.0, 20.0],
                    bounds=([0.0, 0.0, 0.0], [100.0, 2.0, 200.0]))
plateau, breakpoint, decline_rate = fit.x
r_squared = 1 - np.sum(fit.fun ** 2) / np.sum((y - y.mean()) ** 2)
# Retained observations are post-TI; plateau and breakpoint are not independently identified.
# Preserve the exploratory curve shape, without misleading parameter standard errors.
coefficients = pd.DataFrame({"parameter": ["plateau", "breakpoint_DVS", "decline_per_DVS"],
                              "estimate": fit.x,
                              "r_squared": r_squared})
coefficients["unit"] = ["kg CO2 ha(leaf)-1 h-1", "DVS", "kg CO2 ha(leaf)-1 h-1 per DVS"]
coefficients["estimate_umol_CO2_m2_leaf_s"] = np.where(
    coefficients["parameter"].eq("breakpoint_DVS"), np.nan, coefficients["estimate"] / AMAX_CONVERSION)
dvs = np.sort(np.unique(np.r_[np.linspace(0, 2, 401), breakpoint]))
prediction = plateau - decline_rate * np.maximum(0, dvs - breakpoint)
curve = pd.DataFrame({"DVS": dvs, "AMAX_piecewise": prediction,
                       "AMAX_display": np.maximum(0, prediction),
                       "within_observed_DVS": (dvs >= x.min()) & (dvs <= x.max())})
df_fit[["source_id", "site", "season", "date", "Treatment", "cultivar",
        "DVS_final", "AMAX_wofost"]].to_csv(
    SOURCE_DIR / "fig04_amax_piecewise_observations.csv", index=False)
curve.to_csv(SOURCE_DIR / "fig04_amax_piecewise_curve.csv", index=False)
coefficients.to_csv(SOURCE_DIR / "fig04_amax_piecewise_coefficients.csv", index=False)
print(coefficients.round(3).to_string(index=False))
print(f"Pooled fit: n={len(df_fit)}, RMSE={np.sqrt(np.mean(fit.fun ** 2)):.3f}, R²={r_squared:.3f}")


# %% Figure 4 source: NLD2025 observations and crop-default AMAX
points = df_fit.rename(columns={"DVS_final": "x", "AMAX_wofost": "value",
                                "Treatment": "treatment"}).copy()
points["panel"], points["series_kind"], points["n"], points["se"] = "DVS", "observed", 1, np.nan
reference = pd.DataFrame({"panel": "DVS", "x": [0, 2], "value": [33, 33],
                           "series_kind": "default", "source_id": "crop_default", "site": "crop_default"})
data = pd.concat([points, reference], ignore_index=True)
data["value_umol_CO2_m2_leaf_s"] = data["value"] / AMAX_CONVERSION
data.to_csv(SOURCE_DIR / "fig04_amax.csv", index=False)
summary = points.groupby(["site", "season", "cultivar", "treatment"], dropna=False).agg(
    n=("value", "size"), mean_AMAX=("value", "mean"),
    sd_AMAX=("value", "std"), minimum_AMAX=("value", "min"), maximum_AMAX=("value", "max")).reset_index()
for statistic in ["mean_AMAX", "sd_AMAX", "minimum_AMAX", "maximum_AMAX"]:
    summary[statistic + "_umol_CO2_m2_leaf_s"] = summary[statistic] / AMAX_CONVERSION
summary.to_csv(NUMBER_DIR / "figure4_amax_summary.csv", index=False)


# %% Manuscript writing numbers: gross assimilation per unit LEAF area, in both units
# Re-execute from saved fit/observation tables; no refitting or model run.
observations = pd.read_csv(SOURCE_DIR / "fig04_amax_piecewise_observations.csv")
fit_parameters = pd.read_csv(SOURCE_DIR / "fig04_amax_piecewise_coefficients.csv").set_index("parameter")
rates, observed_dvs = observations["AMAX_wofost"], observations["DVS_final"]
level, onset, slope = fit_parameters["estimate"].loc[["plateau", "breakpoint_DVS", "decline_per_DVS"]]
fitted_rates = level - slope * np.maximum(0, observed_dvs - onset)
fitted_start = level - slope * max(0, observed_dvs.min() - onset)
fitted_end = level - slope * max(0, observed_dvs.max() - onset)
writing_numbers = pd.DataFrame({
    "metric": ["observed_mean", "observed_SD", "observed_minimum", "observed_maximum",
               "crop_default", "fitted_at_observed_DVS_min", "fitted_at_observed_DVS_max",
               "decline_per_0.1_DVS", "fit_RMSE"],
    "kg_CO2_ha_leaf_h": [rates.mean(), rates.std(), rates.min(), rates.max(), 33,
                         fitted_start, fitted_end, slope * .1,
                         np.sqrt(np.mean((rates - fitted_rates) ** 2))],
})
writing_numbers["umol_CO2_m2_leaf_s"] = writing_numbers["kg_CO2_ha_leaf_h"] / AMAX_CONVERSION
writing_numbers.to_csv(NUMBER_DIR / "figure4_writing_numbers.csv", index=False)
print("\nGross AMAX per unit leaf area (ha and m2 refer to LEAF area):")
print(writing_numbers.round(3).to_string(index=False))
print(f"n={len(rates)} curves on {observations['date'].nunique()} dates; "
      f"DVS={observed_dvs.min():.3f}-{observed_dvs.max():.3f}; "
      f"R²={fit_parameters['r_squared'].iloc[0]:.3f}; "
      f"fitted decline over observed DVS={100 * (1 - fitted_end / fitted_start):.1f}%")


# %% Figure 4: NLD2025 AMAX versus DVS; rerun from saved source tables
plt.rcParams.update(MANUSCRIPT_RC)
data = pd.read_csv(SOURCE_DIR / "fig04_amax.csv")
curve = pd.read_csv(SOURCE_DIR / "fig04_amax_piecewise_curve.csv")
coefficients = pd.read_csv(SOURCE_DIR / "fig04_amax_piecewise_coefficients.csv").set_index("parameter")
plateau, breakpoint, decline_rate = coefficients["estimate"].loc[
    ["plateau", "breakpoint_DVS", "decline_per_DVS"]]
fig, ax = plt.subplots(figsize=(4, 3))
observed = data[data["series_kind"].eq("observed")]
ax.scatter(observed["x"], observed["value"], marker="o", facecolor="0.65",
            edgecolor="black", alpha=.8, label="Innovator")
reference = data[data["series_kind"].eq("default")].sort_values("x")
ax.plot(reference["x"], reference["value"], color=MODEL_COLOR["default"], linewidth=.8,
         label="Crop default")
ax.plot(curve["DVS"], curve["AMAX_display"], color="black", linestyle="--", lw=1.2,
         label="extrapolation")
supported = curve[curve["within_observed_DVS"]]
ax.plot(supported["DVS"], supported["AMAX_display"], color="black", lw=1.5,
         label="Piecewise fit")
ax.axvline(breakpoint, color="0.65", linestyle=":")
equation = (rf"$gross\ AMAX = {plateau:.1f} - {decline_rate:.1f}$" "\n"
             rf"$\quad\times\max(0, DVS - {breakpoint:.1f})$" "\n"
             rf"$R^2 = {coefficients['r_squared'].iloc[0]:.2f}$")
ax.text(.03, .05, equation, transform=ax.transAxes, fontsize=8, va="bottom")
ax.set(xlabel="DVS", ylabel=r"Gross AMAX (kg CO$_2$ ha$^{-1}$ h$^{-1}$)",
        xlim=(0, 2), ylim=(0, 65))
ax_umol = ax.secondary_yaxis("right", functions=(lambda rate: rate / AMAX_CONVERSION,
                                                lambda rate: rate * AMAX_CONVERSION))
ax_umol.set_ylabel(r"Gross AMAX ($\mu$mol CO$_2$ m$^{-2}$ s$^{-1}$)")
ax.legend(loc="upper right", frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(MAIN_FIG_DIR / "Figure_04.svg", format="svg", bbox_inches="tight")
plt.show()


# %% Supplementary NLD2024 leaf-temperature comparison: explicit source exception
# Fits and high-light observations are distinguished; retain the original 25-C normalization.
temperature = pd.read_csv(ROOT / "output" / "licor_amax_leaf_temperature_NLD2024.csv")
temperature = temperature[temperature["Treatment"].eq("AC") & temperature["Tleaf"].ge(20)
                          & temperature["gdd_emergence"].lt(700)].copy()
reference_amax = temperature.loc[temperature["Tleaf"].eq(25), "AMAX_wofost"].mean()
temperature["relative_AMAX"] = temperature["AMAX_wofost"] / reference_amax
temperature_summary = temperature.groupby(["measurement_kind", "Tleaf"]).agg(
    value=("relative_AMAX", "mean"), se=("relative_AMAX", "sem"), n=("relative_AMAX", "size"),
    gross_AMAX_kg_CO2_ha_leaf_h=("AMAX_wofost", "mean")).reset_index()
temperature_summary["gross_AMAX_umol_CO2_m2_leaf_s"] = (
    temperature_summary["gross_AMAX_kg_CO2_ha_leaf_h"] / AMAX_CONVERSION)
temperature_summary.to_csv(SOURCE_DIR / "amax_leaf_temperature_NLD2024.csv", index=False)
print("NLD2024 supplementary temperature observations:")
print(temperature_summary.round(3).to_string(index=False))


# %% Supplementary temperature plot; rerun from the saved summary
# These NLD2024 observations do not enter the main AMAX fit or a TMPFTB correction.
temperature_summary = pd.read_csv(SOURCE_DIR / "amax_leaf_temperature_NLD2024.csv")
fig_temperature, ax_temperature = plt.subplots(figsize=(4, 3))
ax_temperature.plot([0, 3, 10, 11, 12, 13, 14, 15, 24, 29, 36],
                     [0, 0, .75, .8, .85, .9, .95, 1, 1, .75, .01],
                     color=MODEL_COLOR["default"], label="Crop default")
for kind, marker, label in [("AQ_fit", "o", "Fitted AMAX"),
                             ("high_light", "s", "Observed high-light assimilation")]:
    group = temperature_summary[temperature_summary["measurement_kind"].eq(kind)]
    ax_temperature.errorbar(group["Tleaf"], group["value"], yerr=group["se"],
                              marker=marker, linestyle="none", markerfacecolor="white",
                              color="black", capsize=2, markersize=4, label=label)
ax_temperature.set(xlabel="Leaf temperature (?C)",
                    ylabel="Gross assimilation relative to mean at 25 ?C",
                    xlim=(0, 40), ylim=(0, 1.4), xticks=np.arange(0, 41, 5))
ax_temperature.legend(loc="lower left", frameon=False, fontsize=7)
fig_temperature.tight_layout()
fig_temperature.savefig(SUPP_FIG_DIR / "AMAX_leaf_temperature_NLD2024.svg",
                          format="svg", bbox_inches="tight")
plt.show()


# %% IND2024 only: crop-default TMPFTB over the committed PP crop period
# No revised high-temperature table: excluded NLD2024 observations supplied that fit.
from pcse.models import Wofost73_PP
from utils.loaders import (load_experiments, find_experiment, load_crop_data, load_soil,
    load_site_data, build_parameters, load_weather, load_agro, run_days)
from utils.phenology_helpers import load_phenology_params, obs_dates
from utils import BiomassPartitioning as biop

growth_dir = ROOT / "output" / "pp_growth"
exp = find_experiment(load_experiments("IND"), "IND", "2024-2025", "Kennebec")
pheno = load_phenology_params(field="OpenField", index=2)
pheno = pheno[pheno["site"].eq("IND") & pheno["year"].astype(str).eq("2024-2025")
                & pheno["cultivar"].eq("Kennebec")].iloc[0]
overrides = {name: round(pheno[name]) for name in ["TBASEM", "TEFFMX", "TSUMEM", "TSUM1", "TSUM2"]}
establishment = pd.read_csv(ROOT / "data" / "obs" / "IND_establishment_rate.csv")
overrides["TDWI"] = exp["TDWI"] * establishment.loc[
    establishment["season"].astype(str).eq("2024-2025"), "Establishment_obs"].mean()
slatb = pd.read_csv(growth_dir / "pp_growth_params_v3.csv")
slatb = slatb[slatb["site"].eq("IND") & slatb["cultivar"].eq("Kennebec")
                & slatb["param"].eq("SLATB")].sort_values("dvs")
overrides["SLATB"] = slatb[["dvs", "value"]].to_numpy().ravel().tolist()
partition = pd.read_csv(growth_dir / "pp_partition_params_IND_v1.csv")
partition = partition[partition["site"].eq("IND") & partition["cultivar"].eq("Kennebec")]
named = dict(zip(partition["param"], partition["value"]))
fotb, fltb, fstb = biop.compute_partition_functions(
    FOTBpar={key: named[key] for key in ["FOTBa", "FOTBb"]},
    FLTBpar={key: named[key] for key in ["FLTBa", "FLTBb", "FLTBc", "FLTBd"]}, warn=False)
overrides.update(FOTB=fotb.tolist(), FLTB=fltb.tolist(), FSTB=fstb.tolist(), SPAN=named["SPAN"])
params = build_parameters(load_crop_data("data/crops_calibration"), load_soil(exp["soil_file"]),
                           load_site_data(exp["WAV"], exp["CO2"]), "Kennebec", overrides)
tmpftb = list(params["TMPFTB"])
weather = load_weather(exp["weather_file"])
model = Wofost73_PP(params, weather, load_agro("IND", "2024-2025", "Kennebec"),
                     config=str(ROOT / "data" / "conf" / "Wofost73_PP_SLA.conf"))
days = run_days("IND", "2024-2025")
model.run(days) if days else model.run_till_terminate()
daily = pd.DataFrame(model.get_output())
daily["day"] = pd.to_datetime(daily["day"])
temperature_daily = pd.DataFrame(weather.export())
temperature_daily["DAY"] = pd.to_datetime(temperature_daily["DAY"])
temperature_daily = temperature_daily[temperature_daily["DAY"].between(
    daily["day"].min(), daily["day"].max())].copy()
temperature_daily["TMEAN"] = (temperature_daily["TMIN"] + temperature_daily["TMAX"]) / 2
temperature_daily["DTEMP"] = (temperature_daily["TMEAN"] + temperature_daily["TMAX"]) / 2
temperature_daily["TMPFTB"] = np.interp(temperature_daily["DTEMP"], tmpftb[::2], tmpftb[1::2])
temperature_daily["DAE"] = (temperature_daily["DAY"] - obs_dates(
    "IND", "2024-2025", "Kennebec")["Emergence"]).dt.days
temperature_daily = temperature_daily[temperature_daily["DAE"] >= 0].copy()
temperature_daily.to_csv(SOURCE_DIR / "ind2024_tmpftb_daily.csv", index=False)
print(f"IND2024: {(temperature_daily['TMPFTB'] < 1).sum()} of {len(temperature_daily)} "
      f"crop days have TMPFTB < 1; mean={temperature_daily['TMPFTB'].mean():.3f}")


# %% Supplementary IND2024 TMPFTB diagnostic; rerun from the saved table
temperature_daily = pd.read_csv(SOURCE_DIR / "ind2024_tmpftb_daily.csv")
fig_tmpftb, ax_tmpftb = plt.subplots(figsize=(4.5, 3))
ax_tmpftb.plot(temperature_daily["DAE"], temperature_daily["TMPFTB"], color="black")
ax_tmpftb.fill_between(temperature_daily["DAE"], temperature_daily["TMPFTB"], 1,
                       where=temperature_daily["TMPFTB"] < 1, color="0.8")
ax_tmpftb.set(xlabel="Days after emergence", ylabel="TMPFTB (-)", ylim=(0, 1.05))
fig_tmpftb.tight_layout()
fig_tmpftb.savefig(SUPP_FIG_DIR / "IND2024_TMPFTB.svg", format="svg", bbox_inches="tight")
plt.show()
