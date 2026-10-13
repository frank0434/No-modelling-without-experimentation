# %% IND2024 growth corrections: Figure 3, downstream SLATB and appendix SLA
# Three sequential comparisons: default, TDWI correction, TDWI + SLATB correction.
# Fixed per-season phenology from 02.6 isolates the growth-input corrections.
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pcse.models import Wofost73_PP

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next(p for p in (_here, *_here.parents)
            if (p / "scripts").is_dir() and (p / "data").is_dir())
sys.path.insert(0, str(ROOT / "scripts"))
from utils.loaders import (
    load_experiments, find_experiment, load_agro, load_weather, load_soil,
    load_crop_data, load_site_data, build_parameters, run_days,
    load_obs_lai, load_obs_yield, load_obs_sla,
)
from utils.phenology_helpers import load_phenology_params, obs_dates
from utils.metrics import nrmse, mae
from manuscript_common import (
    SOURCE_DIR, MAIN_FIG_DIR, SUPP_FIG_DIR, NUMBER_DIR, MANUSCRIPT_RC, MODEL_COLOR, LINE_STYLE,
)

SITE, SEASON, CULTIVAR = "IND", "2024-2025", "Kennebec"
FIELD, TREATMENT = "OpenField", "Full_0 Days"
GROWTH_DIR = ROOT / "output" / "pp_growth"
VARIANTS = ["default_tdwi", "corrected_tdwi", "tdwi_slatb"]
for directory in (GROWTH_DIR, SOURCE_DIR, MAIN_FIG_DIR, SUPP_FIG_DIR, NUMBER_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# %% One crop, its existing phenology, and the original correction formulas
exp = find_experiment(load_experiments(SITE), SITE, SEASON, CULTIVAR)
cropd = load_crop_data("data/crops_calibration")
pheno = load_phenology_params(field=FIELD, index=2)
pheno = pheno[pheno["site"].eq(SITE) & pheno["year"].astype(str).eq(SEASON)
              & pheno["cultivar"].eq(CULTIVAR)]
phenology = {name: round(pheno.iloc[0][name])
             for name in ["TBASEM", "TEFFMX", "TSUMEM", "TSUM1", "TSUM2"]}
emergence = obs_dates(SITE, SEASON, CULTIVAR, FIELD)["Emergence"]

# Establishment records are replicates; retain the original seasonal mean.
establishment = pd.read_csv(ROOT / "data" / "obs" / "IND_establishment_rate.csv")
establishment = establishment[establishment["season"].astype(str).eq(SEASON)]
establishment_rate = establishment["Establishment_obs"].mean()
tdwi_corrected = exp["TDWI"] * establishment_rate

sla = load_obs_sla(SITE)
sla = sla[sla["season"].astype(str).eq(SEASON) & sla["cultivar"].eq(CULTIVAR)
          & sla["Field"].eq(FIELD) & sla["treatment"].eq(TREATMENT)]
sla_plateau_m2kg = sla["SLA_obs"].max()
SLATB_CAL = [0.0, 0.0017, 1.0, sla_plateau_m2kg / 1e4,
             2.0, sla_plateau_m2kg / 1e4]
print(f"IND2024: establishment={establishment_rate:.3f}; "
      f"TDWI={exp['TDWI']:.1f} -> {tdwi_corrected:.1f} kg/ha; "
      f"SLA plateau={sla_plateau_m2kg:.1f} m2/kg")

# Preserve the parameter filename/schema read by partitioning and water scripts.
# Only IND receives this correction; RSA growth is excluded.
slatb_parameters = pd.DataFrame({
    "run_id": ["v3"] * 3, "site": [SITE] * 3, "cultivar": [CULTIVAR] * 3,
    "param": ["SLATB"] * 3, "dvs": SLATB_CAL[::2], "value": SLATB_CAL[1::2],
    "note": ["ten Den DVS0 kept; no early-season data to change shape",
             f"plateau anchored to IND2024 peak observed SLA ({sla_plateau_m2kg:.1f} m2/kg)",
             "plateau held flat DVS1-2"],
})
slatb_parameters.to_csv(GROWTH_DIR / "pp_growth_params_v3.csv", index=False)


# %% Three potential-production runs; only TDWI and SLATB differ
simulation_parts = []
for variant in VARIANTS:
    overrides = dict(phenology)
    overrides["TDWI"] = exp["TDWI"] if variant == "default_tdwi" else tdwi_corrected
    if variant == "tdwi_slatb":
        overrides["SLATB"] = SLATB_CAL
    params = build_parameters(cropd, load_soil(exp["soil_file"]),
                              load_site_data(exp["WAV"], exp["CO2"]), CULTIVAR, overrides)
    # This configuration changes recorded variables, not WOFOST crop equations.
    model = Wofost73_PP(params, load_weather(exp["weather_file"]),
                       load_agro(SITE, SEASON, CULTIVAR),
                       config=str(ROOT / "data" / "conf" / "Wofost73_PP_SLA.conf"))
    days = run_days(SITE, SEASON)
    model.run(days) if days else model.run_till_terminate()
    frame = pd.DataFrame(model.get_output())
    frame["day"] = pd.to_datetime(frame["day"])
    frame = frame[frame["DVS"] >= 0].copy()
    if variant == "corrected_tdwi":
        ind_sla_simulation = frame[frame["WLV"] > 0][
            ["day", "DVS", "LAIEXP", "SLAT", "LAI", "WLV"]
        ].assign(site=SITE, season=SEASON)
    if variant == "tdwi_slatb":
        frame = frame[frame["WLV"] > 0].copy()  # Original corrected-SLATB export window.
    frame = frame.melt(id_vars="day", value_vars=["LAI", "TWSO"],
                       var_name="variable", value_name="value")
    frame["variant"] = variant
    frame["series_kind"] = "simulation"
    frame["uncertainty"] = np.nan
    simulation_parts.append(frame)


# %% Existing control observations; keep their means and SD without aggregation
observation_parts = []
for variable, observed, value_column, sd_column in [
    ("LAI", load_obs_lai(SITE), "LAI", "LAI_sd"),
    ("TWSO", load_obs_yield(SITE), "TWSO_obs", "TWSO_obs_sd"),
]:
    observed = observed[
        observed["season"].astype(str).eq(SEASON) & observed["cultivar"].eq(CULTIVAR)
        & observed["Field"].eq(FIELD) & observed["treatment"].eq(TREATMENT)
    ][["day", value_column, sd_column]].dropna(subset=[value_column]).copy()
    observed = observed.rename(columns={value_column: "value", sd_column: "uncertainty"})
    observed["day"] = pd.to_datetime(observed["day"])
    observed["variable"] = variable
    observed["variant"] = "observed"
    observed["series_kind"] = "observation"
    observation_parts.append(observed)

corrections = pd.concat([*simulation_parts, *observation_parts], ignore_index=True)
corrections["site"], corrections["season"] = SITE, SEASON
corrections["cultivar"], corrections["field"] = CULTIVAR, FIELD
corrections["treatment"] = TREATMENT
corrections["DAE"] = (corrections["day"] - emergence).dt.days
corrections = corrections[corrections["DAE"] >= 0].copy()
corrections.to_csv(SOURCE_DIR / "fig03_growth_corrections.csv", index=False)


# %% Figure 3 error summaries from the same observations and three trajectories
corrections = pd.read_csv(SOURCE_DIR / "fig03_growth_corrections.csv")
pairs = corrections.pivot(index=["variable", "day"], columns="variant", values="value")
pairs = pairs.dropna(subset=["observed", *VARIANTS])
metric_rows = []
for variable, group in pairs.groupby("variable"):
    for variant in VARIANTS:
        metric_rows.append({"site": SITE, "season": SEASON, "variable": variable,
                            "variant": variant, "n": len(group),
                            "NRMSE": nrmse(group["observed"], group[variant]),
                            "MAE": mae(group["observed"], group[variant])})
metrics = pd.DataFrame(metric_rows)
baseline_error = metrics[metrics["variant"].eq("default_tdwi")].set_index("variable")["NRMSE"]
metrics["NRMSE_reduction_percent"] = 100 * (
    1 - metrics["NRMSE"] / metrics["variable"].map(baseline_error).replace(0, np.nan)
)
metrics.to_csv(NUMBER_DIR / "figure3_writing_numbers.csv", index=False)
print(metrics.round(3).to_string(index=False))

# TWSO MAE after each correction, compared with the preceding step.
twso_mae = metrics[metrics["variable"].eq("TWSO")].set_index("variant").loc[VARIANTS, ["MAE"]]
twso_mae["MAE_reduction_kg_ha"] = -twso_mae["MAE"].diff()
twso_mae["MAE_reduction_percent"] = 100 * twso_mae["MAE_reduction_kg_ha"] / twso_mae["MAE"].shift()
print("\nTWSO MAE (kg/ha): TDWI correction versus default, then SLATB correction versus TDWI-only")
print(twso_mae.round(2).to_string())


# %% Figure 3: rerun this cell alone from the saved source table
corrections = pd.read_csv(SOURCE_DIR / "fig03_growth_corrections.csv", dtype={"season": str})
plt.rcParams.update(MANUSCRIPT_RC)
fig, axes = plt.subplots(1, 2, figsize=(6, 2.5), sharex=True)
styles = [
    ("default_tdwi", "Default TDWI and SLATB", MODEL_COLOR["default"], "-"),
    ("corrected_tdwi", "Corrected TDWI", MODEL_COLOR["intermediate"], "--"),
    ("tdwi_slatb", "Corrected TDWI + SLATB", MODEL_COLOR["final"], "-"),
]
for ax, variable, panel, ylabel in [
    (axes[0], "LAI", "(a)", "LAI (m$^2$ m$^{-2}$)"),
    (axes[1], "TWSO", "(b)", "TWSO (kg ha$^{-1}$)"),
]:
    data = corrections[corrections["variable"].eq(variable)]
    for variant, label, color, linestyle in styles:
        line = data[data["variant"].eq(variant)].sort_values("DAE")
        ax.plot(line["DAE"], line["value"], label=label,
                color=color, linestyle=linestyle, linewidth=0.8)
    observed = data[data["series_kind"].eq("observation")].sort_values("DAE")
    ax.errorbar(observed["DAE"], observed["value"], yerr=observed["uncertainty"],
                **LINE_STYLE["observed"], capsize=2)
    ax.set(xlabel="Days after emergence", ylabel=ylabel)
    ax.text(0.02, 0.98, panel, transform=ax.transAxes, ha="left", va="top")
    ax.set_ylim(bottom=0)
axes[0].set_ylim(0, 6)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center",
           bbox_to_anchor=(0.5, 1.02),
           ncol=3, frameon=False)
fig.subplots_adjust(top=0.9, wspace=0.30)
fig.savefig(MAIN_FIG_DIR / "Figure_03.svg", format="svg", bbox_inches="tight")
plt.show()
print(f"wrote {MAIN_FIG_DIR / 'Figure_03.svg'}")


# %% Supplementary SLA comparison: reuse IND's default-SLATB run; add NLD2025
# Both use default SLATB; IND retains its establishment-corrected TDWI.
nld_exp = find_experiment(load_experiments("NLD"), "NLD", "2025", "Innovator")
nld_pheno = load_phenology_params(field="OpenField", index=2)
nld_pheno = nld_pheno[nld_pheno["site"].eq("NLD") & nld_pheno["year"].astype(str).eq("2025")
                      & nld_pheno["cultivar"].eq("Innovator")].iloc[0]
nld_overrides = {name: round(nld_pheno[name])
                 for name in ["TBASEM", "TEFFMX", "TSUMEM", "TSUM1", "TSUM2"]}
nld_overrides["TDWI"] = nld_exp["TDWI"]
nld_params = build_parameters(cropd, load_soil(nld_exp["soil_file"]),
                              load_site_data(nld_exp["WAV"], nld_exp["CO2"]), "Innovator", nld_overrides)
nld_model = Wofost73_PP(nld_params, load_weather(nld_exp["weather_file"]),
                       load_agro("NLD", "2025", "Innovator"),
                       config=str(ROOT / "data" / "conf" / "Wofost73_PP_SLA.conf"))
nld_days = run_days("NLD", "2025")
nld_model.run(nld_days) if nld_days else nld_model.run_till_terminate()
nld_frame = pd.DataFrame(nld_model.get_output())
nld_frame["day"] = pd.to_datetime(nld_frame["day"])
sla_simulation = pd.concat([ind_sla_simulation, nld_frame[
    (nld_frame["DVS"] >= 0) & (nld_frame["WLV"] > 0)
][
    ["day", "DVS", "LAIEXP", "SLAT", "LAI", "WLV"]
].assign(site="NLD", season="2025")], ignore_index=True)
sla_simulation["SLA_inst"] = sla_simulation["SLAT"] * 1e4
sla_simulation["SLA_canopy"] = sla_simulation["LAI"] / sla_simulation["WLV"] * 1e4

observed_sla_parts = []
for site, season, cultivar, treatment in [
    ("NLD", "2025", "Innovator", "AC"), ("IND", "2024-2025", "Kennebec", "Full_0 Days"),
]:
    observed_sla = load_obs_sla(site)
    observed_sla = observed_sla[
        observed_sla["season"].astype(str).eq(season) & observed_sla["cultivar"].eq(cultivar)
        & observed_sla["Field"].eq("OpenField") & observed_sla["treatment"].eq(treatment)
    ][["day", "SLA_obs", "SLA_obs_sd"]].copy()
    observed_sla["day"] = pd.to_datetime(observed_sla["day"])
    observed_sla_parts.append(observed_sla.assign(site=site, season=season))
sla_dvs = sla_simulation.merge(pd.concat(observed_sla_parts, ignore_index=True),
                               on=["site", "season", "day"], how="left")
sla_dvs.to_csv(SOURCE_DIR / "s08_sla_dvs.csv", index=False)


# %% Figure S8: instantaneous and integrated canopy SLA, retained sites only
# Re-execute this cell alone from the saved source table.
sla_dvs = pd.read_csv(SOURCE_DIR / "s08_sla_dvs.csv", dtype={"season": str})
plt.rcParams.update(MANUSCRIPT_RC)
fig_sla, axes_sla = plt.subplots(2, 2, figsize=(6, 3), sharey=True, sharex="row")
for column, site, season in [(0, "NLD", "2025"), (1, "IND", "2024-2025")]:
    data = sla_dvs[sla_dvs["site"].eq(site) & sla_dvs["season"].eq(season)].sort_values("day")
    ax_i, ax_c = axes_sla[0, column], axes_sla[1, column]
    ax_i.plot([0, 1, 2], [17, 24, 24], color="0.8", linewidth=5, zorder=1)
    ax_i.plot(data["DVS"], data["SLA_inst"], color="black", linewidth=0.8)
    released = data[data["LAIEXP"] >= 6].head(1)
    ax_i.plot(released["DVS"], released["SLA_inst"], marker="o", linestyle="none",
              markerfacecolor="white", markeredgecolor="black", markersize=7)
    ax_c.plot(data["DVS"], data["SLA_canopy"], color="black", linewidth=0.8)
    observed = data.dropna(subset=["SLA_obs"])
    ax_c.errorbar(observed["DVS"], observed["SLA_obs"], yerr=observed["SLA_obs_sd"],
                  fmt="o", color="black", markerfacecolor="white", markersize=4, capsize=3)
    ax_i.text(0.6, 1.2, f"{site}{season[:4]}", transform=ax_i.transAxes, ha="right", va="top")
    ax_i.tick_params(axis = "x", labelbottom = False)
    for row, ax in enumerate((ax_i, ax_c)):
        ax.text(0.02, 0.98, f"({chr(97 + row * 2 + column)})",
                 transform=ax.transAxes,
                ha="left", va="top")
        ax.set(xlim=(0, 2), ylim=(-1, 32))
    ax_c.set_xlabel("DVS")
axes_sla[0, 0].set_ylabel("Instantaneous\nSLA (m$^2$ kg$^{-1}$)")
axes_sla[1, 0].set_ylabel("Integrated canopy\nSLA (m$^2$ kg$^{-1}$)")
sla_handles = [
    Line2D([], [], color="0.8", linewidth=5, label="Default SLATB table"),
    Line2D([], [], color="black", label="SLA applied (line)"),
    Line2D([], [], marker="o", linestyle="none", color="black", markerfacecolor="white",
           label="LAIEXP reaches 6"),
    Line2D([], [], marker="o", linestyle="none", color="black", markerfacecolor="white",
           label="Observed canopy SLA"),
]
fig_sla.legend(handles=sla_handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                ncol=4, frameon=False)
fig_sla.subplots_adjust(top=0.8, wspace=0.08, hspace=0.1)
fig_sla.savefig(SUPP_FIG_DIR / "Figure_S08.svg", format="svg", bbox_inches="tight")
plt.show()
print(f"wrote {SUPP_FIG_DIR / 'Figure_S08.svg'}")
