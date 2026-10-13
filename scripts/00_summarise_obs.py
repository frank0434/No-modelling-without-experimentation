# %%
# Stage 0b: Observation inventory + benchmark summary  (single raw-data workflow)
#
# All observed data in data/obs/ is treated as raw, replicate-level. This script:
#   Function 1 (inventory)  -> output/obs_inventory.csv
#       counts + uncertainty (n, SD, SE, CV) per location x season x treatment x cultivar x variable.
#   Function 2 (benchmark)  -> output/obs_benchmark_summary.csv
#       tidy mean +/- SE per location x season x field x treatment x cultivar x day x DAP x variable,
#       keyed for direct merge against simulation output later.


import sys
import pathlib
import numpy as np
import pandas as pd


_here = pathlib.Path().resolve()
sys.path.insert(0, str(next(p for p in (_here, *_here.parents)
                            if (p/"scripts").exists() and (p/"data").exists()) / "scripts"))
from utils.paths import ROOT
from utils.loaders import canonicalize_season, load_obs_experiments
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from utils.visualise import (
    panels, outer_labels, row_legend, _series_colors, _padded_limits,
    _grid_spacing, _save, FIGURES_DIR, VIZ_LW, VIZ_MS,
)
from utils.plotstyle import use_style
OBS_DIR = ROOT / "data" / "obs"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

def plot_obs_benchmark(df, variable, x="DAP", hue="treatment", ylabel=None,
                       xlim=(1, 130), output=False, OUTPUT_DIR=FIGURES_DIR, RUN_ID=""):
    """Observed mean +/- SD per treatment, one panel per location x season.

    - x is 'DAP' (days after planting) or 'day'
    - one grey level per treatment, the same grey in every panel
    - returns None when this variable has no plottable rows (e.g. final-harvest yield)
    """
    use_style()
    dv = df[df["variable"] == variable].dropna(subset=[x, "mean"]).copy()
    if dv.empty:
        return None
    dv[hue] = dv[hue].astype(str)
    dv["season"] = dv["season"].astype(str)
    ylabel = ylabel or variable

    # one grey per treatment across the whole figure
    colors = _series_colors(sorted(dv[hue].dropna().unique()))
    locs = sorted(dv["location"].dropna().unique())
    y_all = pd.concat([dv["mean"] - dv["sd"].fillna(0), dv["mean"] + dv["sd"].fillna(0)])
    y_lo, y_hi = _padded_limits(y_all.min(), y_all.max())

    ncols = max(len(dv.loc[dv["location"] == l, "season"].unique()) for l in locs)
    fig, axes, cells = panels(dv, loc_order=locs, ncols=ncols,
                              figsize=(4.2 * ncols, 3.2 * len(locs)), sharey=False)
    # a column's lowest panel may sit above an empty row, so find it per column
    bottom = {c: max(r for r, _c, _ax, _l, _s, _d in cells if _c == c) for _r, c, *_ in cells}

    for r, c, ax, loc, season, sub in cells:
        for hv in sorted(sub[hue].unique()):
            grp = sub[sub[hue] == hv].sort_values(x)
            ax.errorbar(grp[x], grp["mean"], yerr=grp["sd"].fillna(0), marker="o",
                        capsize=3, color=colors[hv], linewidth=VIZ_LW["sim"],
                        markersize=VIZ_MS["errorbar"])
        ax.set_xlim(*xlim)
        ax.set_ylim(y_lo, y_hi)
        outer_labels(ax, c, r, bottom[c], ylabel, x)
        if r != bottom[c]:
            ax.tick_params(labelbottom=False)

    for r, loc in enumerate(locs):
        vals = sorted(dv.loc[dv["location"] == loc, hue].unique())
        row_legend(axes, r, len(dv.loc[dv["location"] == loc, "season"].unique()),
                   [mlines.Line2D([], [], marker="o", color=colors[v], label=v,
                                  linewidth=VIZ_LW["legend"], markersize=VIZ_MS["legend"])
                    for v in vals], title=loc)

    _grid_spacing(fig, left=0.1, hspace=0.35)
    _save(fig, output, OUTPUT_DIR, f"obs_benchmark_{variable}_{RUN_ID}")
    return fig


# Canonical key columns of the harmonised long frame.
KEY_COLS = ["location", "season", "treatment", "cultivar", "field", "day", "DAP"]

# Units per variable — used only to annotate the inventory (CV is unitless and cross-comparable).
VARIABLE_UNITS = {
    "TAGP": "kg/ha", "TWSO": "kg/ha", "LAI": "m2/m2", "SLA": "m2/kg",
    "RD": "cm", "SM": "cm3/cm3", "establishment": "fraction"
}

# Season labels that canonicalize_season() cannot handle (underscore/harvest formats).
SEASON_REMAP = {
    "2024_Autumn": "2024",      # RSA_RS_dm.csv
    "2022_23":     "2022-2023", # guard in case any file still uses underscore form
    "2024_25":     "2024-2025", # RSA_SM_obs_24_25.csv
    "2025-Autumn": "2025-A",    # RSA_SM_obs_25_2nd.csv if it uses season label format
    "2025-Spring": "2025-S",    # RSA_SM_obs_25_2nd.csv if it uses season label format
}

def _norm_season(s):
    s = SEASON_REMAP.get(str(s), str(s))
    return canonicalize_season(s)

# ── Raw-source registry ───────────────────────────────────────────────────────
# Each entry describes ONE CSV file. load_raw() uses three keys:
#
#   "const": columns whose value is FIXED for every row and NOT in the CSV
#            (e.g. location is implied by the filename).
#            {canonical_colname: value}
#
#   "rename": columns IN the CSV whose name differs from the canonical schema.
#             {exact_csv_colname: canonical_colname}
#             Only RSA_dm needs this — remove it once the R file is fixed.
#
#   "vars": the MEASURED columns to extract. Keys are the EXACT column names
#           from the CSV header (copy-paste from there). Values are the canonical
#           variable labels that land in the output "variable" column.
#           {exact_csv_colname: canonical_label}
#           Example: {"TAGP_obs": "TAGP"} -> rows where variable="TAGP"
#
# load_raw() also applies two global fixes before matching any of the above:
#   dap -> DAP   (all files use lowercase dap; DAP is canonical for sim merges)
#   date -> day  (R exports date columns as "date" by default)
#
RAW_SOURCES = [
    # -- Dry matter -----------------------------------------------------------
    # IND_dm.csv:  plotid, season, day, dap, TAGP_obs, TWSO_obs, treatment
    {"path": "IND_dm.csv",
     "const": {"location": "IND", "cultivar": "Kennebec", "field": "OpenField"},
     "vars":  {"TAGP_obs": "TAGP", "TWSO_obs": "TWSO",
               "WLV_obs": "WLV", "WST_obs": "WST", "WDEAD_obs": "WDEAD"}},

    # NLD_dm.csv:  season, treatment, day, dap, TAGP_obs, TWSO_obs
    {"path": "NLD_dm.csv",
     "const": {"location": "NLD", "cultivar": "Innovator", "field": "OpenField"},
     "vars":  {"TAGP_obs": "TAGP", "TWSO_obs": "TWSO",
               "WLV_obs": "WLV", "WST_obs": "WST", "WDEAD_obs": "WDEAD"}},

    # RSA_dm.csv:  day, dap, water, cult, TAGP_obs, TWSO_obs, season, treatment
    # FIX IN R: rename water->treatment, cult->cultivar, then remove "rename" below.
    {"path": "RSA_dm.csv",
     "const":  {"location": "RSA", "field": "OpenField"},
     "rename": {"water": "treatment", "cult": "cultivar"},
     "vars":   {"TAGP_obs": "TAGP", "TWSO_obs": "TWSO",
                "WLV_obs": "WLV", "WST_obs": "WST", "WDEAD_obs": "WDEAD"}},

    # -- LAI ------------------------------------------------------------------
    # IND_LAI.csv: season, treatment, day, dap, LAI_obs, SLA_obs_m2_kg
    # SLA column carries its unit in the name (m2/kg); NLD/RSA SLA_obs are cm2/g.
    {"path": "IND_LAI.csv",
     "const": {"location": "IND", "cultivar": "Kennebec", "field": "OpenField"},
     "vars":  {"LAI_obs": "LAI", "SLA_obs_m2_kg": "SLA"}},

    # NLD_LAI.csv: season, day, dap, treatment, LAI, SLA
    {"path": "NLD_LAI.csv",
     "const": {"location": "NLD", "cultivar": "Innovator", "field": "OpenField"},
     "vars":  {"LAI_obs": "LAI", "SLA_obs": "SLA"}},

    # RSA_LAI.csv: day, dap, LAI, SLA, season, treatment
    {"path": "RSA_LAI_24.csv",
     "const": {"location": "RSA", "field": "OpenField"},
     "vars":  {"LAI_obs": "LAI", "SLA_obs": "SLA"}},

    # -- Establishment rate ---------------------------------------------------
    # IND_establishment_rate.csv: season, plotid, treatment, date, dap, Establishment_obs
    {"path": "IND_establishment_rate.csv",
     "const": {"location": "IND", "cultivar": "Kennebec", "field": "OpenField"},
     "vars":  {"Establishment_obs": "establishment"}},

    # RSA_establishment_rate.csv: field, rep, plot, Establishment_obs, season, treatment
    # No day/dap — single occasion; in inventory only, not the time-series figure.
    {"path": "RSA_establishment_rate.csv",
     "const": {"location": "RSA"},
     "vars":  {"Establishment_obs": "establishment"}},

    # -- RSA final-harvest dry weight -----------------------------------------
    # RSA_RS_dm.csv: field, season, treatment, cultivar, date, dap, TWSO_obs
    {"path": "RSA_RS_dm.csv",
     "const": {"location": "RSA", "field": "RainShelter"},
     "vars":  {"TWSO_obs": "TWSO"}},

    # -- Rooting depth --------------------------------------------------------
    # IND_rooting_depth.csv: season, plotid, treatment, RD_obs  (no day/dap)
    {"path": "IND_rooting_depth.csv",
     "const": {"location": "IND", "cultivar": "Kennebec", "field": "OpenField"},
     "vars":  {"RD_obs": "RD"}},

    # -- Soil moisture --------------------------------------------------------
    # All SM files already contain location, season, day, treatment, cultivar, SM_obs.
    # No const needed. No dap column — SM is measured by calendar date only.
    {"path": "IND_SM_obs.csv",        "const": { "field": "OpenField"}, "vars": {"SM_obs": "SM"}},
    {"path": "NLD_SM_obs.csv",        "const": { "field": "OpenField"}, "vars": {"SM_obs": "SM"}},
    {"path": "RSA_SM_obs_24_25.csv",  "const": {}, "vars": {"SM_obs": "SM"}},
    {"path": "RSA_SM_obs_25_2nd.csv", "const": {}, "vars": {"SM_obs": "SM"}},
]


def load_raw(spec) -> pd.DataFrame:
    """Read one file and return canonical long rows (one row per replicate value)."""
    df = pd.read_csv(OBS_DIR / spec["path"])

    # Two global fixes that apply to every file:
    #   dap -> DAP  (all files use lowercase; DAP is canonical for downstream sim merges)
    #   date -> day (R writes date columns as "date" by default)
    df = df.rename(columns={"dap": "DAP", "date": "day"})

    # File-specific renames for columns with wrong names (not just wrong case).
    # Currently only RSA_dm needs this; remove once fixed in R.
    df = df.rename(columns=spec.get("rename", {}))

    # Inject constant values for columns not present in the file.
    for k, v in spec.get("const", {}).items():
        df[k] = v

    # Fill missing key columns so every source has the same schema.
    if "field" not in df.columns:
        df["field"] = "OpenField"
    if "day" not in df.columns:
        df["day"] = pd.NaT    # single-occasion files (RD, establishment) — no time axis
    if "DAP" not in df.columns:
        df["DAP"] = np.nan    # SM files: measured by date, no DAP

    df["season"] = df["season"].map(_norm_season)
    df["day"] = pd.to_datetime(df["day"], errors="coerce")

    # Melt: pivot measured columns from wide to long.
    # spec["vars"] keys = exact CSV column names (after the renames above).
    # spec["vars"] values = canonical label in the output "variable" column.
    id_cols = [c for c in KEY_COLS if c in df.columns]
    long = df.melt(
        id_vars=id_cols,
        value_vars=list(spec["vars"]),
        var_name="variable",
        value_name="value",
    )
    long["variable"] = long["variable"].map(spec["vars"])
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    return long


# %%
# -- 1. Harmonise every source into one long frame ----------------------------
frames = []
for spec in RAW_SOURCES:
    try:
        frames.append(load_raw(spec))
        print(f"  ok {spec['path']}")
    except FileNotFoundError:
        print(f"  skip {spec['path']} -- not present yet")

df_raw = pd.concat(frames, ignore_index=True)
# drop IND 2023-2024 completely
df_raw = df_raw[~((df_raw['location'] == 'IND') & (df_raw['season'] == '2023-2024'))]
print(f"\nHarmonised {len(df_raw)} replicate rows across variables: {df_raw['variable'].unique()}" )

# SLA is compiled in mixed units across the source LAI files: IND_LAI.csv is in
# m2/kg (column SLA_obs_m2_kg), NLD_LAI.csv and RSA_LAI_24.csv in cm2/g
# (1 m2/kg = 10 cm2/g). Harmonise the benchmark to m2/kg -- divide the cm2/g
# sites by 10; IND is already m2/kg.
_sla_cm2g = (df_raw["variable"] == "SLA") & df_raw["location"].isin(["NLD", "RSA"])
df_raw.loc[_sla_cm2g, "value"] /= 10.0

# RSA OpenField 2025-A and 2025-S: TAGP unreliable — drop; TWSO final harvest only
_rsa_of_2025 = (
    (df_raw['location'] == 'RSA') &
    (df_raw['field'] == 'OpenField') &
    (df_raw['season'].isin(['2025-A', '2025-S']))
)
df_raw = df_raw[~(_rsa_of_2025 & df_raw['variable'].isin(['TAGP', 'WLV', 'WST', 'WDEAD']))].copy()

_rsa_of_2025 = (
    (df_raw['location'] == 'RSA') &
    (df_raw['field'] == 'OpenField') &
    (df_raw['season'].isin(['2025-A', '2025-S']))
)
_twso_rsa = _rsa_of_2025 & (df_raw['variable'] == 'TWSO')
_final_day = (df_raw[_twso_rsa]
              .groupby(['season', 'treatment', 'cultivar'])['day']
              .transform('max'))
df_raw = df_raw[~_twso_rsa | df_raw['day'].eq(_final_day).fillna(False)]

# %%
# -- Correct IND 2024-2025 stem weight (WST) ----------------------------------
# The measured IND 2024-2025 stems are implausibly low. Back them out from the
# 2024-2025 leaves using the 2022-2023 leaf-to-stem ratio as the reference shape:
#   WST_corrected = WLV_2024-2025 / (WLV/WST)_2022-2023
# The 2022-2023 per-occasion mean ratio is paired to the 2024-2025 occasions in
# sampling order (2022-2023 dap 38/66/95 -> 2024-2025 dap 33/60/89). Applied per
# plot from IND_dm.csv; the corrected WST replicate rows then replace the raw ones
# in df_raw so every downstream summary and figure uses the corrected values.
# The diagnostic sections below re-read the raw CSV and still show the before/after.
_dm_ind = pd.read_csv(OBS_DIR / "IND_dm.csv")
_dm_ind = _dm_ind.loc[_dm_ind["WLV_obs"].notna() & _dm_ind["WST_obs"].notna(),
                      ["season", "treatment", "day", "dap", "WLV_obs", "WST_obs"]]
_ref_ls = (_dm_ind[_dm_ind["season"] == "2022-2023"]
           .assign(ls=lambda d: d["WLV_obs"] / d["WST_obs"])
           .groupby("dap")["ls"].mean().sort_index().to_numpy())
_dm_2425 = _dm_ind[_dm_ind["season"] == "2024-2025"].copy()
_dap_ratio = dict(zip(sorted(_dm_2425["dap"].unique()), _ref_ls))
_wst_corr = pd.DataFrame({
    "location":  "IND",
    "season":    _dm_2425["season"].map(_norm_season),
    "treatment": _dm_2425["treatment"],
    "cultivar":  "Kennebec",
    "field":     "OpenField",
    "day":       pd.to_datetime(_dm_2425["day"], errors="coerce"),
    "DAP":       _dm_2425["dap"],
    "variable":  "WST",
    "value":     _dm_2425["WLV_obs"] / _dm_2425["dap"].map(_dap_ratio),
})
_drop = (df_raw["location"] == "IND") & (df_raw["season"] == "2024-2025") & (df_raw["variable"] == "WST")
df_raw = pd.concat([df_raw[~_drop], _wst_corr[KEY_COLS + ["variable", "value"]]], ignore_index=True)
print(f"Corrected IND 2024-2025 WST: replaced {int(_drop.sum())} rows with {len(_wst_corr)} ratio-corrected rows")

# %%
# -- Phenology and TDWI from MainTable.xlsx Experiments sheet ------------------
df_exp = load_obs_experiments().drop(columns=['season']).rename(columns={"season_key": "season"})
df_exp = df_exp[~((df_exp['location'] == 'IND') & (df_exp['season'] == '2023-2024'))]

_PHENO_COLS = {
    "Emergence":       "emergence",
    "TuberInitiation": "tuber_initiation",
    "Maturity":         "Maturity",
}
_pheno_parts = []
for col, var_label in _PHENO_COLS.items():
    sub = (df_exp[["location", "season", "field", "cultivar", col]]
           .rename(columns={col: "day"})
           .dropna(subset=["day"]))
    sub["variable"] = var_label
    sub["value"]    = 1.0
    _pheno_parts.append(sub)

df_tdwi_exp = df_exp[["location", "season", "field", "cultivar", "TDWI"]].rename(columns={"TDWI": "value"})
df_tdwi_exp["variable"] = "TDWI"
df_tdwi_exp["day"]      = pd.NaT

df_pheno_all = pd.concat([*_pheno_parts, df_tdwi_exp], ignore_index=True)
df_pheno_all["treatment"] = np.nan
df_pheno_all["DAP"]       = np.nan
df_pheno_all["day"]       = pd.to_datetime(df_pheno_all["day"], errors="coerce")

df_raw = pd.concat([df_raw, df_pheno_all[KEY_COLS + ["variable", "value"]]], ignore_index=True)
print(f"Added phenology + TDWI: {len(df_pheno_all)} rows")

# %%
# -- Summary table: n_observations + n_observation_dates per group -------------
df_summary = (
    df_raw.groupby(["location", "season", "treatment", "field", "cultivar", "variable"], dropna=False)
    .agg(
        n_observations=("value", "count"),
        n_observation_dates=("day", "nunique"),
    )
    .reset_index()
)
df_summary.to_csv(OUTPUT_DIR / "obs_summary.csv", index=False)
print(f"\nWrote obs_summary.csv ({len(df_summary)} rows)")
print(df_summary.to_string(index=False))

# %%
# -- Collapsed summary: one row per (location, season, field, variable) --------
# Treatment and cultivar are collapsed into descriptor cells; n_observations is
# summed across all treatment x cultivar combinations within each group.

_TREATMENT_DESC = {
    ("IND",  "OpenField",   None):     "2 irrigation quantity × 3 irrigation frequency",
    ("NLD",  "OpenField",   None):     "soil warming in 2 phenology stages",
    ("RSA",  "OpenField",   "2024"):   "2 irrigation quantity × 2 cultivar",
    ("RSA",  "RainShelter", "2024"):   "2 irrigation quantity × Mondial",
    ("RSA",  "OpenField",   "2025-A"): "2 irrigation method × 2 cultivar",
    ("RSA",  "OpenField",   "2025-S"): "2 irrigation method × 2 cultivar",
    ("RSA",  "RainShelter", "2025-A"): "2 irrigation quantity × Mondial",
    ("RSA",  "RainShelter", "2025-S"): "2 irrigation quantity × Mondial",
}
_N_REPS = {("NLD", "2025"): "3–4"}  # all others are "4"

def _treatment_desc(location, field, season):
    return (_TREATMENT_DESC.get((location, field, season))
            or _TREATMENT_DESC.get((location, field, None), ""))

def build_collapsed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (location, season, field, variable), sub in df.groupby(
        ["location", "season", "field", "variable"], dropna=False
    ):
        valid = sub[sub["value"].notna()]
        n_obs = len(valid)
        if n_obs == 0:
            continue
        n_dates = valid["day"].nunique()
        if n_dates == 0:
            n_dates = 1  # single-occasion files where day is NaT
        rows.append({
            "location":            location,
            "season":              season,
            "field":               field,
            "treatment":           _treatment_desc(location, field, season),
            "cultivar":            ", ".join(sorted(sub["cultivar"].dropna().unique())),
            "variable":            variable,
            "n_observations":      n_obs,
            "n_replicates":        _N_REPS.get((location, season), "4"),
            "n_observation_dates": n_dates,
        })
    return pd.DataFrame(rows)

df_collapsed = build_collapsed_summary(df_raw)
df_collapsed.to_csv(OUTPUT_DIR / "obs_collapsed_summary.csv", index=False)
print(f"\nWrote obs_collapsed_summary.csv ({len(df_collapsed)} rows)")
print(df_collapsed.to_string(index=False))

# %%
# -- Final summary: one row per (location, field, variable), seasons collapsed -
_VARIABLE_CATEGORY = {
    "TDWI":            "Phenology",
    "LAI":             "Growth",
    "TAGP":            "Growth",
    "TWSO":            "Growth",
    "WLV":             "Growth",
    "WST":             "Growth",
    "WDEAD":           "Growth",
    "SM":              "Water balance",
    "RD":              "Water balance",
    "emergence":       "Phenology",
    "tuber_initiation":"Phenology",
    "Maturity":         "Phenology",
}

def build_final_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (location, field, variable), sub in df.groupby(["location", "field", "variable"]):
        all_cultivars = sorted({
            c.strip() for cell in sub["cultivar"] for c in cell.split(",") if c.strip()
        })
        treatments = [t for t in sub["treatment"].unique() if pd.notna(t)]
        rows.append({
            "category":            _VARIABLE_CATEGORY.get(variable, "Other"),
            "location":            location,
            "field":               field,
            "variable":            variable,
            "n_crop_cycles":       sub["season"].nunique(),
            "treatment":           "; ".join(treatments),
            "cultivar":            ", ".join(all_cultivars),
            "n_observations":      sub["n_observations"].sum(),
            "n_replicates":        "3–4" if sub["n_replicates"].str.contains("3").any() else "4",
            "n_observation_dates": sub["n_observation_dates"].sum(),
        })
    return pd.DataFrame(rows)

df_final = build_final_summary(df_collapsed)
df_final.to_csv(OUTPUT_DIR / "obs_final_summary.csv", index=False)
print(f"\nWrote obs_final_summary.csv ({len(df_final)} rows)")
print(df_final.to_string(index=False))

# %%
# -- 2. Function 1: observation inventory -------------------------------------

# Mark specific (location, season, field, treatment, cultivar, variable) groups
# as quality-failed. Key tuple must match the raw values before normalisation.
QUALITY_OVERRIDES: dict[tuple, str] = {
    # ("RSA", "2025", "OpenField", ..., "LAI"): "unusable",  # example
}

def build_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Counts per group; output columns feed coverage_figure.build_coverage_figure."""
    grp = ["location", "season", "field", "treatment", "cultivar", "variable"]
    rows = []
    for keys, sub in df.groupby(grp, dropna=False):
        v = sub["value"].dropna()
        n = len(v)
        if n == 0:
            continue
        days = sub.loc[sub["value"].notna(), "day"]
        per_day = v.groupby(days).size()
        rows.append({
            "site":             keys[grp.index("location")],
            "season":           keys[grp.index("season")],
            "setup":            keys[grp.index("field")],
            "treatment":        keys[grp.index("treatment")],
            "cultivar":         keys[grp.index("cultivar")],
            "variable":         keys[grp.index("variable")],
            "n":                n,
            "n_harvest_dates":  int(days.nunique()),
            "n_reps":           int(per_day.median()) if len(per_day) else n,
            "quality_flag":     QUALITY_OVERRIDES.get(keys, "ok"),
            "in_design":        True,
            "unit":             VARIABLE_UNITS.get(keys[grp.index("variable")], ""),
        })
    out_cols = ["site", "season", "setup", "treatment", "cultivar", "variable",
                "n", "n_harvest_dates", "n_reps", "quality_flag", "in_design", "unit"]
    return pd.DataFrame(rows)[out_cols].sort_values(
        ["site", "season", "setup", "treatment", "cultivar", "variable"]
    ).reset_index(drop=True)


df_inventory = build_inventory(df_raw)
df_inventory.to_csv(OUTPUT_DIR / "obs_inventory.csv", index=False)
print(f"Wrote obs_inventory.csv ({len(df_inventory)} rows)")
print(df_inventory.to_string(index=False))


# %%
# -- 3. Function 2: benchmark summary (mean +/- SD per sampling occasion) ------
def build_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    grp = ["location", "season", "field", "treatment", "cultivar", "day", "DAP", "variable"]
    out = (
        df.dropna(subset=["value"])
        .groupby(grp, dropna=False)["value"]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )
    return out


df_benchmark = build_benchmark(df_raw)
df_benchmark.to_csv(OUTPUT_DIR / "obs_benchmark_summary.csv", index=False)
print(f"Wrote obs_benchmark_summary.csv ({len(df_benchmark)} rows)")
df_benchmark_OF = df_benchmark[df_benchmark['field'] == 'OpenField']
#%%
## plot the df_benchmark to check the data

for _variable in sorted(df_benchmark_OF["variable"].unique()):
    _unit = VARIABLE_UNITS.get(_variable, "")
    plot_obs_benchmark(
        df_benchmark_OF,
        variable=_variable,
        ylabel=f"{_variable} ({_unit})" if _unit else _variable,
        output=True,
        OUTPUT_DIR=ROOT / "figures" / "obs_benchmark",
        RUN_ID="obs",
    )
print("  benchmark figures written to figures/")
# %% stem in IND 2024 is very low - check if it can be corrected
# 2022-2023 is left as-is and shown only for reference.
import matplotlib.pyplot as plt

_dm = pd.read_csv(OBS_DIR / "IND_dm.csv")
_est = pd.read_csv(OBS_DIR / "IND_establishment_rate.csv")

_wst = _dm.loc[_dm["WST_obs"].notna(), ["season", "plotid", "treatment", "dap", "WST_obs"]]
_wst = _wst.merge(
    _est[["season", "plotid", "dap", "Establishment_obs"]],
    on=["season", "plotid", "dap"], how="left",
)

_is_2425 = _wst["season"] == "2024-2025"
_wst["WST_corr"] = _wst["WST_obs"]
_wst.loc[_is_2425, "WST_corr"] = _wst.loc[_is_2425, "WST_obs"] / _wst.loc[_is_2425, "Establishment_obs"]

# Mean +/- SD across the 24 plots per (season, dap).
_before = _wst.groupby(["season", "dap"])["WST_obs"].agg(mean="mean", sd="std").reset_index()
_after = _wst[_is_2425].groupby(["season", "dap"])["WST_corr"].agg(mean="mean", sd="std").reset_index()

_ref = _before[_before["season"] == "2022-2023"]
_raw2425 = _before[_before["season"] == "2024-2025"]

fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(_ref["dap"], _ref["mean"], yerr=_ref["sd"], marker="o", capsize=3,
            color="grey", label="2022 (reference, uncorrected)")
ax.errorbar(_raw2425["dap"], _raw2425["mean"], yerr=_raw2425["sd"], marker="s", capsize=3,
            color="tab:red", label="2024 before")
ax.errorbar(_after["dap"], _after["mean"], yerr=_after["sd"], marker="^", capsize=3,
            color="tab:green", label="2024 after (/ establishment)")
ax.set_xlabel("DAP")
ax.set_ylabel("WST (kg/ha)")
ax.set_title("IND stem weight: establishment correction")
ax.legend()
fig.tight_layout()
_figdir = ROOT / "figures" / "obs_benchmark"
_figdir.mkdir(parents=True, exist_ok=True)
fig.savefig(_figdir / "IND_WST_establishment_correction.png", dpi=150)
plt.close(fig)
print("Wrote IND_WST_establishment_correction.png")

# %% leaf-to-stem ratio (WLV/WST) over time
# Compare IND 2022-2023, IND 2024-2025 and NLD 2025. For IND 2024-2025 show the
# ratio both before and after the establishment correction. The correction divides
# only WST by establishment (WLV untouched), so the "after" ratio = "before" ratio
# x establishment fraction; it is a check on whether correcting the low IND 2024-2025
# stems brings the leaf-to-stem ratio into line with the other season/site.

_ind = pd.read_csv(OBS_DIR / "IND_dm.csv")
_est = pd.read_csv(OBS_DIR / "IND_establishment_rate.csv")
_ind = _ind.loc[_ind["WLV_obs"].notna() & _ind["WST_obs"].notna(),
                ["season", "plotid", "dap", "WLV_obs", "WST_obs"]]
_ind = _ind.merge(_est[["season", "plotid", "dap", "Establishment_obs"]],
                  on=["season", "plotid", "dap"], how="left")
_m = _ind["season"] == "2024-2025"
_ind["WST_corr"] = _ind["WST_obs"]
_ind.loc[_m, "WST_corr"] = _ind.loc[_m, "WST_obs"] / _ind.loc[_m, "Establishment_obs"]
_ind["ls_before"] = _ind["WLV_obs"] / _ind["WST_obs"]
_ind["ls_after"] = _ind["WLV_obs"] / _ind["WST_corr"]

_nld = pd.read_csv(OBS_DIR / "NLD_dm.csv")
_nld = _nld[(_nld["season"].astype(str) == "2025") & _nld["WLV_obs"].notna() & _nld["WST_obs"].notna()]
_nld = _nld.assign(ls=_nld["WLV_obs"] / _nld["WST_obs"])

def _ratio_by_dap(frame, col):
    return frame.groupby("dap")[col].agg(mean="mean", sd="std").reset_index()

_ind2223 = _ratio_by_dap(_ind[_ind["season"] == "2022-2023"], "ls_before")
_ind2425_b = _ratio_by_dap(_ind[_m], "ls_before")
_ind2425_a = _ratio_by_dap(_ind[_m], "ls_after")
_nld2025 = _ratio_by_dap(_nld, "ls")

fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(_ind2223["dap"], _ind2223["mean"], yerr=_ind2223["sd"], marker="o", capsize=3,
            color="grey", label="IND 2022")
ax.errorbar(_ind2425_b["dap"], _ind2425_b["mean"], yerr=_ind2425_b["sd"], marker="s", capsize=3,
            color="tab:red", label="IND 2024 before")
ax.errorbar(_ind2425_a["dap"], _ind2425_a["mean"], yerr=_ind2425_a["sd"], marker="^", capsize=3,
            color="tab:green", label="IND 2024 after")
ax.errorbar(_nld2025["dap"], _nld2025["mean"], yerr=_nld2025["sd"], marker="D", capsize=3,
            color="tab:blue", label="NLD 2025")
ax.set_xlabel("DAP")
ax.set_ylabel("leaf-to-stem ratio (WLV/WST)")
ax.set_title("Leaf-to-stem ratio over time")
ax.legend()
fig.tight_layout()
fig.savefig(_figdir / "leaf_stem_ratio.png", dpi=150)
plt.close(fig)
print("Wrote leaf_stem_ratio.png")

# %% correct IND 2024-2025 WST from the 2022-2023 leaf:stem ratio and 2024-2025 WLV

_ind2 = pd.read_csv(OBS_DIR / "IND_dm.csv")
_est2 = pd.read_csv(OBS_DIR / "IND_establishment_rate.csv")
_ind2 = _ind2.loc[_ind2["WLV_obs"].notna() & _ind2["WST_obs"].notna(),
                  ["season", "plotid", "dap", "WLV_obs", "WST_obs"]]
_ind2 = _ind2.merge(_est2[["season", "plotid", "dap", "Establishment_obs"]],
                    on=["season", "plotid", "dap"], how="left")

# 2022-2023 reference ratio per occasion, early -> late.
_ref_ratio = (_ind2[_ind2["season"] == "2022-2023"]
              .assign(ls=lambda d: d["WLV_obs"] / d["WST_obs"])
              .groupby("dap")["ls"].mean().sort_index().to_numpy())

_m2 = _ind2["season"] == "2024-2025"
_dap_2425 = sorted(_ind2.loc[_m2, "dap"].unique())
_dap_to_ratio = dict(zip(_dap_2425, _ref_ratio))

_ind2["WST_estab"] = _ind2["WST_obs"]
_ind2.loc[_m2, "WST_estab"] = _ind2.loc[_m2, "WST_obs"] / _ind2.loc[_m2, "Establishment_obs"]
_ind2["WST_ratio"] = np.nan
_ind2.loc[_m2, "WST_ratio"] = _ind2.loc[_m2, "WLV_obs"] / _ind2.loc[_m2, "dap"].map(_dap_to_ratio)

def _by_dap(frame, col, season):
    return frame[frame["season"] == season].groupby("dap")[col].agg(mean="mean", sd="std").reset_index()

_r2223 = _by_dap(_ind2, "WST_obs", "2022-2023")
_b2425 = _by_dap(_ind2, "WST_obs", "2024-2025")
_e2425 = _by_dap(_ind2, "WST_estab", "2024-2025")
_x2425 = _by_dap(_ind2, "WST_ratio", "2024-2025")

fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(_r2223["dap"], _r2223["mean"], yerr=_r2223["sd"], marker="o", capsize=3,
            color="grey", label="2022 (reference)")
ax.errorbar(_b2425["dap"], _b2425["mean"], yerr=_b2425["sd"], marker="s", capsize=3,
            color="tab:red", label="2024 before")
ax.errorbar(_e2425["dap"], _e2425["mean"], yerr=_e2425["sd"], marker="^", capsize=3,
            color="tab:green", label="2024 after (/ establishment)")
ax.errorbar(_x2425["dap"], _x2425["mean"], yerr=_x2425["sd"], marker="D", capsize=3,
            color="tab:purple", label="2024 after (WLV / ratio)")
ax.set_xlabel("DAP")
ax.set_ylabel("WST (kg/ha)")
ax.set_title("IND stem weight: establishment vs leaf:stem-ratio correction")
ax.legend()
fig.tight_layout()
fig.savefig(_figdir / "IND_WST_ratio_correction.png", dpi=150)
plt.close(fig)
print("Wrote IND_WST_ratio_correction.png")

# %% get location information from weather data
from pcse.input import ExcelWeatherDataProvider
WEATHER_SOURCES = [
    {"path": "India2022_23_wofost.xlsx",      "location": "IND", "season": "2022-2023"},
    {"path": "India2024_25_wofost.xlsx",      "location": "IND", "season": "2024-2025"},
    {"path": "WUR2024_wofost.xlsx",           "location": "NLD", "season": "2024"},
    {"path": "WUR2025_wofost.xlsx",           "location": "NLD", "season": "2025"},    
    {"path": "SouthAfrica2024_wofost.xlsx",   "location": "RSA", "season": "2024"},
    {"path": "SouthAfrica2025_wofost.xlsx",   "location": "RSA", "season": "2025-A"},
    {"path": "SouthAfrica2025_2_wofost.xlsx", "location": "RSA", "season": "2025-S"},
]

WEATHER_DIR = ROOT / "data" / "Weather"

rows = []
for spec in WEATHER_SOURCES:
    wp = ExcelWeatherDataProvider(WEATHER_DIR / spec["path"])
    rows.append({
        "location":   spec["location"],
        "season":     spec["season"],
        "latitude":   round(wp.latitude, 2),
        "longitude":  round(wp.longitude, 2),
        "elevation":  round(wp.elevation),
        "Start_date": wp.first_date,
        "End_date":  wp.last_date,
    })

df_location = pd.DataFrame(rows)

# -- Join collapsed summary with location metadata (season still present) ------
_COLLAPSED_GEO_COLS = [
    "location", "season", "latitude", "longitude", "elevation",
    "field", "treatment", "cultivar",
    "variable",
    "n_observations", "n_replicates", "n_observation_dates",
    "Start_date", "End_date",
]
df_collapsed_geo = df_collapsed.merge(
    df_location[["location", "season", "latitude", "longitude", "elevation", "Start_date", "End_date"]],
    on=["location", "season"],
    how="left",
)[_COLLAPSED_GEO_COLS]
df_collapsed_geo.to_csv(OUTPUT_DIR / "obs_collapsed_summary_geo.csv", index=False, encoding="utf-8-sig")
print(f"\nWrote obs_collapsed_summary_geo.csv ({len(df_collapsed_geo)} rows)")

# -- Join final summary with location metadata (seasons collapsed — no dates) --
_COUNTRY = {"IND": "India", "NLD": "Netherlands", "RSA": "South Africa"}
_CITY    = {"IND": "Gujarat", "NLD": "Wageningen", "RSA": "Pretoria"}

df_location_agg = (
    df_location.groupby("location")[["latitude", "longitude", "elevation"]]
    .first()
    .reset_index()
)
df_final_geo = df_final.merge(df_location_agg, on="location", how="left")
df_final_geo.insert(df_final_geo.columns.get_loc("location"), "country",
                    df_final_geo["location"].map(_COUNTRY))
df_final_geo["location"] = df_final_geo["location"].map(_CITY)

_FINAL_GEO_COLS = [
    "country", "location", "latitude", "longitude", "elevation",
    "field", "n_crop_cycles", "treatment", "cultivar",
    "variable",
    "n_observations", "n_replicates", "n_observation_dates",
]
df_final_geo = df_final_geo[_FINAL_GEO_COLS]
df_final_geo.columns = df_final_geo.columns.str.capitalize()
df_final_geo.columns = df_final_geo.columns.str.replace("_", " ", regex=False).str.title()
df_final_geo.to_csv(OUTPUT_DIR / "obs_final_summary_geo.csv", index=False, encoding="utf-8-sig")
print(f"Wrote obs_final_summary_geo.csv ({len(df_final_geo)} rows)")

# df_raw[['location', 'season', 'cultivar','field']].drop_duplicates()

# %%
# -- 4. Coverage table --------------------------------------------------------
_VAR_ORDER = ["emergence", "tuber_initiation", "Maturity", "TDWI",
              "LAI", "TAGP", "TWSO", "WLV", "WST", "WDEAD",
              "RD", "SM"]
_COL_LABELS = {
    "emergence": "Emergence", "tuber_initiation": "Tuber init.",
    "Maturity": "Haum kill", "TDWI": "TDWI",
    "LAI": "LAI", "TAGP": "TAGP", "TWSO": "TWSO",
    "WLV": "WLV", "WST": "WST", "WDEAD": "WDEAD",
    "RD": "RD", "SM": "SM",
}

_LOC_ORDER   = {"IND": 0, "NLD": 1, "RSA": 2}
_FIELD_ORDER = {"OpenField": 0, "RainShelter": 1}
_RSA_SEASON_DISPLAY = {"2024": "2024-Autumn", "2025-A": "2025-Autumn", "2025-S": "2025-Spring"}

# One row per (location, season, field) — drop cultivar to avoid duplicates
_exp_rows = (df_collapsed[["location", "season", "field"]]
             .drop_duplicates()
             .assign(
                 _loc_ord=lambda d: d["location"].map(_LOC_ORDER),
                 _fld_ord=lambda d: d["field"].map(_FIELD_ORDER),
             )
             .sort_values(["_loc_ord", "_fld_ord", "season"])
             .drop(columns=["_loc_ord", "_fld_ord"])
             .reset_index(drop=True))
_exp_rows["_seq"] = (_exp_rows.groupby("location").cumcount() + 1).astype(str)
_exp_rows["_season_disp"] = _exp_rows.apply(
    lambda r: _RSA_SEASON_DISPLAY.get(r["season"], r["season"]) if r["location"] == "RSA" else r["season"],
    axis=1,
)
_exp_rows["experiment"] = (_exp_rows["location"] + "-" + _exp_rows["_seq"]
                           + "  " + _exp_rows["field"]
                           + " (" + _exp_rows["_season_disp"] + ")")
_exp_order = _exp_rows["experiment"].tolist()
_exp_map   = _exp_rows.set_index(["location", "season", "field"])["experiment"].to_dict()

df_cov = df_collapsed.copy()
df_cov["experiment"] = df_cov.apply(
    lambda r: _exp_map.get((r["location"], r["season"], r["field"]), r["location"]), axis=1
)
pivot_cov = (df_cov
    .pivot_table(index="experiment", columns="variable", values="n_observations", aggfunc="sum")
    .reindex(index=_exp_order,
             columns=[v for v in _VAR_ORDER if v in df_cov["variable"].values])
    .rename(columns=_COL_LABELS)
)

# Column tally row (all columns except SM)
_tally_cols = [c for c in pivot_cov.columns if c != "SM"]
_total = {c: int(pivot_cov[c].sum()) if c in _tally_cols else "–"
          for c in pivot_cov.columns}
display_df = pd.concat([pivot_cov, pd.DataFrame(_total, index=["Total"])])

display_df.to_csv(OUTPUT_DIR / "obs_coverage_table.csv", encoding="utf-8-sig")
print(f"\nWrote obs_coverage_table.csv ({len(display_df)} rows)")
print(display_df.to_string())

# %%
# -- Plot-year table: n treatment×cultivar combinations × replicates per experiment
_plot_years = (
    df_raw[df_raw["value"].notna() & df_raw["treatment"].notna()]
    .groupby(["location", "season", "field"])
    .apply(lambda x: x.drop_duplicates(["treatment", "cultivar"]).shape[0])
    .reset_index(name="n_treatment_cultivar")
)
_plot_years["n_replicates"] = _plot_years.apply(
    lambda r: _N_REPS.get((r["location"], r["season"]), "4"), axis=1
)
_plot_years["n_plots"] = _plot_years.apply(
    lambda r: r["n_treatment_cultivar"] * int(str(r["n_replicates"]).split("–")[0]), axis=1
)
_plot_years.to_csv(OUTPUT_DIR / "obs_plot_years.csv", index=False, encoding="utf-8-sig")
print(f"\nPlot-year table:")
print(_plot_years.to_string(index=False))

# %%
# -- Replicate count diagnostic: find missing observations --------------------
def check_reps(location, season, variable, expected_reps=4):
    """Show per-(treatment, day) rep counts and flag cells below expected."""
    sub = df_raw[
        (df_raw["location"] == location) &
        (df_raw["season"]   == season) &
        (df_raw["variable"] == variable) &
        df_raw["value"].notna()
    ]
    counts = (sub.groupby(["treatment", "day"])
                 .size()
                 .rename("n_reps")
                 .reset_index())
    short = counts[counts["n_reps"] < expected_reps]
    total = counts["n_reps"].sum()
    print(f"\n{location} {season} — {variable}  (total={total})")
    print(counts.pivot(index="treatment", columns="day", values="n_reps").to_string())
    if not short.empty:
        print("  SHORT CELLS:")
        print(short.to_string(index=False))
    else:
        print("  all cells at expected reps")

check_reps("IND", "2022-2023", "LAI")    # IND-1 LAI: expect 72
check_reps("IND", "2024-2025", "TAGP")   # IND-2 TAGP: expect 72
check_reps("NLD", "2024", "LAI")    # NLD-1 LAI
check_reps("NLD", "2025", "TAGP")   # NLD-2 TAGP
# %%
