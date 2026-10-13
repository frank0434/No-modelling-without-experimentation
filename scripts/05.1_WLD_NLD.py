# %% NLD2025 water-limited growth: essential source-data producer
# Shared 05.0 prepares the soil profiles. This script reads the actual AFGEN tables,
# runs raw/corrected three-layer soils once per treatment and writes manuscript CSVs.
# Supplementary S11/S13/S15 rendering is owned by 06.2. No NLD2024 evaluation.
# Main Figure 6 keeps AC/HTI/HB; appendix VWC comparisons use AC/HTI.

import sys
import pathlib
import yaml
import numpy as np
import pandas as pd
from pcse.engine import Engine

_here = pathlib.Path().resolve()
sys.path.insert(0, str(next(p for p in (_here, *_here.parents)
                            if (p/"scripts").exists() and (p/"data").exists()) / "scripts"))
from utils.paths import ROOT
from utils import BiomassPartitioning as biop
from utils import soil_curves as sc
from utils.loaders import (
    load_crop_data, load_experiments, find_experiment,
    load_weather, load_site_data, load_soil_mlwb, build_parameters,
    build_agro_with_irrigation, load_irrigation_csv, canonicalize_season,
    load_obs_yield, load_obs_lai,
)
from utils.phenology_helpers import load_phenology_params, emergence_date
from utils.metrics import MAEDecomposition


MLWB_CONF = str(ROOT / "data" / "conf" / "Wofost73_WLP_MLWB_SM.conf")


# %%
# ── 1. Fixed settings ────────────────────────────────────────────────────────
SITE = "NLD"
SEASON = "2025"
CULTIVAR = "Innovator"
TREATMENTS = ["AC", "HTI", "HB"]

# Three-layer raw/FC-corrected profiles prepared by the shared 05.0 script.
SOIL_FILES = {
    "PTF raw 3-layer": "data/soil/netherlands_soil_MLWB_wosten.yaml",
    "FC corrected 3-layer": "data/soil/netherlands_soil_MLWB.yaml",
}
SOIL_ORDER = list(SOIL_FILES)
APPENDIX_TREATMENTS = ["AC", "HTI"]
SCORE_DEPTH = 40.0   # cm, the block the two probes cover


def layer_layout(soildata):
    """Thicknesses, the layer indices down to SCORE_DEPTH, and the first below it.

    Read the geometry from the profile used in the model.
    """
    thick = [float(l["Thickness"]) for l in soildata["SoilProfileDescription"]["SoilLayers"]]
    bottoms = np.cumsum(thick)
    scored = [i for i, b in enumerate(bottoms) if b <= SCORE_DEPTH]
    return thick, scored, len(scored)

PARTITION_RUN_ID = "v1"
GROWTH_DIR = ROOT / "output" / "pp_growth"
PHENO_PARAMS = ["TBASEM", "TEFFMX", "TSUMEM", "TSUM1", "TSUM2"]

cropd = load_crop_data("data/crops_calibration")
exp = find_experiment(load_experiments(location=SITE), SITE, SEASON, CULTIVAR)
EMERGENCE = emergence_date(SITE, SEASON, CULTIVAR)   # DAE = 0 in every figure below
print(f"[{SITE} {SEASON}]  WAV={exp['WAV']} cm  CO2={exp['CO2']} ppm  "
      f"TDWI(base)={exp['TDWI']} kg/ha  treatments={TREATMENTS}")


# %%
# ── 2. Crop overrides ────────────────────────────────────────────────────────
# Soil-independent, so both soils run the same crop.
overrides = {}

# phenology: NLD keeps its own calibrated TEFFMX, no pin
_pheno = load_phenology_params(field="OpenField", index=2)
_prow = _pheno[(_pheno["site"] == SITE) & (_pheno["cultivar"] == CULTIVAR)
               & (_pheno["year"].astype(str) == SEASON)].iloc[0]
for p in PHENO_PARAMS:
    overrides[p] = round(_prow[p])

# TDWI from the experiment seed weight; SLATB stays at the crop-yaml default
overrides["TDWI"] = exp["TDWI"]
overrides["RDI"] = 20.0  # NLD initial rooting depth, matching the 20 cm planting depth
# partitioning (FOTB/FLTB/FSTB) + RGRLAI + SPAN
_part = pd.read_csv(GROWTH_DIR / f"growth_best_params_{PARTITION_RUN_ID}.csv")
_prows = _part[(_part["site"] == SITE) & (_part["cultivar"] == CULTIVAR)
               & (_part["layer"] == "partitioning")]
_named = {p: float(v) for p, v in zip(_prows["param"], _prows["value"])}
FOTB, FLTB, FSTB = biop.compute_partition_functions(
    FOTBpar={k: _named[k] for k in ("FOTBa", "FOTBb")},
    FLTBpar={k: _named[k] for k in ("FLTBa", "FLTBb", "FLTBc", "FLTBd")},
    warn=False,
)
overrides["FOTB"] = FOTB.tolist()
overrides["FLTB"] = FLTB.tolist()
overrides["FSTB"] = FSTB.tolist()
overrides["RGRLAI"] = _named["RGRLAI"]
overrides["SPAN"] = _named["SPAN"]

# water and roots: CFET and DEPNR stay at the crop default; RDI is fixed at the
# NLD planting depth of 20 cm above. The soil is the only remaining difference.
overrides_mlwb = {**overrides, "RDMCR": 60.0}
print(f"layered: RDMCR={overrides_mlwb['RDMCR']:.1f}  CFET=default (1.0)")


# %%
# ── 3. Drivers: weather, site, irrigation ────────────────────────────────────
wdp = load_weather(exp["weather_file"])
sited = load_site_data(exp["WAV"], exp["CO2"])
with open(ROOT / "data" / "agro" / "netherlands_seasons.yaml") as f:
    run_days = yaml.safe_load(f)["seasons"][SEASON].get("run_days")

irr_all = load_irrigation_csv(ROOT / "data" / "agro" / "NLD_irrigation.csv")
irr_all["season"] = irr_all["season"].astype(str).map(canonicalize_season)
irr_all = irr_all[irr_all["season"] == canonicalize_season(SEASON)]
for t in TREATMENTS:
    sub = irr_all[irr_all["treatment"] == t]
    print(f"{t:5s}  {len(sub):2d} irrigation events, {sub['mm'].sum():5.1f} mm total")

SOILS = {label: load_soil_mlwb(path) for label, path in SOIL_FILES.items()}
for label, sd in SOILS.items():
    thick, scored, deep = layer_layout(sd)
    print(f"{label:22s} layers {thick}, RDMSOL {sd['RDMSOL']:.0f} cm, "
          f"scored layers {scored} (0-{SCORE_DEPTH:.0f} cm), layer below {deep}")


# %% Daily probe means, preserving the existing day/treatment/depth grain
raw_probe = pd.read_csv(ROOT / "data/obs/NLD_SM_obs.csv", dtype={"season": str})
raw_probe["day"] = pd.to_datetime(raw_probe.day)
raw_probe = raw_probe[raw_probe.season.eq(SEASON) & raw_probe.treatment.isin(TREATMENTS)
                      & raw_probe.location.eq(SITE) & raw_probe.cultivar.eq(CULTIVAR)].copy()
raw_probe["DAE"] = (raw_probe.day - EMERGENCE).dt.days
raw_probe = raw_probe[raw_probe.DAE >= 0]
# Same daily averaging as the original probe_obs helper; raw reading rows remain intact.
probe = raw_probe.groupby(["depth", "treatment", "day", "DAE"], as_index=False).agg(
    SM_obs=("SM_obs", "mean"))
assert not probe.duplicated(["depth", "treatment", "day"]).any()
assert set(probe.treatment) == set(TREATMENTS)
source_dir = ROOT / "manuscript_outputs/source_data"
source_dir.mkdir(parents=True, exist_ok=True)

# %% Actual retention curves and field-capacity targets for supplementary S11
# Targets are the approved 05.0 values; probe range follows its pooled-depth convention.
FC_TARGETS = {(0.0, 20.0): 0.15, (20.0, 40.0): 0.25}
pooled_probe = raw_probe.groupby(["treatment", "day"], as_index=False).agg(SM_obs=("SM_obs", "mean"))
obs_lo, obs_hi = pooled_probe.SM_obs.min(), pooled_probe.SM_obs.max()
retention_rows = []
for soil, profile in SOILS.items():
    layers = profile["SoilProfileDescription"]["SoilLayers"]
    thickness = np.array([float(layer["Thickness"]) for layer in layers])
    bottoms = np.cumsum(thickness)
    bounds = list(zip(bottoms - thickness, bottoms))
    assert bounds == [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0)]
    for (top, bottom), layer in zip(bounds, layers):
        table = np.asarray(layer["SMfromPF"], dtype=float)
        target = FC_TARGETS.get((top, bottom), np.nan)
        if soil == "FC corrected 3-layer" and np.isfinite(target):
            assert np.isclose(np.interp(sc.PF_FIELD_CAPACITY, table[0::2], table[1::2]), target)
        retention_rows.extend({
            "site": SITE, "geometry": "3layer", "soil": soil.split(" 3-layer")[0],
            "layer_top": top, "layer_bottom": bottom, "pF": pf, "vwc": value,
            "obs_lo": obs_lo, "obs_hi": obs_hi, "fc_target": target, "wp_target": np.nan,
        } for pf, value in zip(table[0::2], table[1::2]))
# No measured target/correction for the 40-60 cm layer.
assert np.allclose(
    SOILS[SOIL_ORDER[0]]["SoilProfileDescription"]["SoilLayers"][2]["SMfromPF"],
    SOILS[SOIL_ORDER[1]]["SoilProfileDescription"]["SoilLayers"][2]["SMfromPF"])
retention = pd.DataFrame(retention_rows)
assert not retention.duplicated(["site", "geometry", "soil", "layer_top", "layer_bottom", "pF"]).any()
retention.to_csv(source_dir / "s11_nld_retention_curves.csv", index=False)

# %% Six model runs: reuse daily results for appendix and main-figure handoffs
def run_mlwb(treat, label):
    """Layered balance (Engine + MLWB conf) on the soil named by `label`."""
    agro = build_agro_with_irrigation(SITE, SEASON, CULTIVAR,
                                      irr_all[irr_all["treatment"] == treat])
    model = Engine(build_parameters(cropd, SOILS[label], sited, CULTIVAR, overrides_mlwb),
                   wdp, agro, config=MLWB_CONF)
    model.run(run_days) if run_days else model.run_till_terminate()
    df = pd.DataFrame(model.get_output())
    df["day"] = pd.to_datetime(df["day"])
    df["ET0"] = [wdp(day.date()).ET0 for day in df["day"]]
    df["RAIN"] = [wdp(day.date()).RAIN for day in df["day"]]
    df["IRRIGATION"] = df["TOTIRR"].diff().fillna(df["TOTIRR"])
    df["DAE"] = (df["day"] - EMERGENCE).dt.days
    df = df[df["DAE"] >= 0].reset_index(drop=True)
    # SM, WC and Flow come out as arrays; Flow has one entry
    # more than there are layers (Flow_L0 is the top boundary).
    for name in ("SM", "WC", "Flow"):
        arr = np.array([np.asarray(v, dtype=float) for v in df[name]])
        for i in range(arr.shape[1]):
            df[f"{name}_L{i}"] = arr[:, i]
    df = df.drop(columns=["SM", "WC", "Flow"])
    # PCSE returns TRALY=None outside the active crop; that is zero root uptake.
    traly = df.pop("TRALY").apply(pd.Series).fillna(0)
    traly.columns = [f"TRALY_L{i}" for i in traly.columns]
    df[traly.columns] = traly
    # Profile moisture over 0-40 cm: the water the scored layers hold, over their own
    # thickness. SM_L{i} is WC_L{i}/Thickness_i (multilayer_waterbalance.py:787-795),
    # so this is sum(WC)/SCORE_DEPTH on a FIXED depth block -- the quantity the probes
    # can be compared against, and the same quantity in both geometries. It is not
    # SM_MEAN (= W/RD), the mean over the ROOTED zone, which moves as the roots grow.
    thick, scored, _ = layer_layout(SOILS[label])
    df["SM_profile"] = sum(thick[i] * df[f"SM_L{i}"] for i in scored) / sum(thick[i] for i in scored)
    _wc = sum(df[f"WC_L{i}"] for i in scored) / sum(thick[i] for i in scored)
    assert (df["SM_profile"] - _wc).abs().max() < 1e-12, "SM_profile is not sum(WC)/sum(Thickness)"
    return df


runs = {soil: {t: run_mlwb(t, soil) for t in TREATMENTS} for soil in SOIL_ORDER}
for soil, by_treatment in runs.items():
    for treatment, daily in by_treatment.items():
        assert not daily.day.duplicated().any() and (daily.DAE >= 0).all()
        print(f"{soil}: {treatment}, {len(daily)} days; final TWSO={daily.TWSO.iloc[-1]:.0f} kg/ha")

# %% Observed crop series and plot-ready daily source (main Figure 6 and S13)
obs_lai = load_obs_lai(location=SITE).rename(columns={"LAI": "LAI_obs"})
obs_twso = load_obs_yield(location=SITE)
for df in [obs_lai, obs_twso]:
    df["season"] = df.season.astype(str).map(canonicalize_season)
    df["day"] = pd.to_datetime(df.day)
    df["DAE"] = (df.day - EMERGENCE).dt.days
obs_lai = obs_lai[obs_lai.season.eq(SEASON) & obs_lai.cultivar.eq(CULTIVAR)
                  & obs_lai.Field.eq("OpenField") & obs_lai.treatment.isin(TREATMENTS)
                  & (obs_lai.DAE >= 0)]
obs_twso = obs_twso[obs_twso.season.eq(SEASON) & obs_twso.cultivar.eq(CULTIVAR)
                    & obs_twso.Field.eq("OpenField") & obs_twso.treatment.isin(TREATMENTS)
                    & (obs_twso.DAE >= 0)]
series_parts = []
for soil in SOIL_ORDER:
    for treatment in TREATMENTS:
        daily = runs[soil][treatment]
        variables = ["LAI", "TWSO", "ET0", "RAIN", "IRRIGATION", "TRAMX", "TRA"] + [
            c for c in daily if c.startswith(("SM_L", "TRALY_L"))]
        long = daily[["day", "DAE", *variables]].melt(
            id_vars=["day", "DAE"], var_name="variable", value_name="value")
        series_parts.append(long.assign(site=SITE, season=SEASON, treatment=treatment,
                                       soil=soil, series_kind="simulation", SD=np.nan, depth=np.nan))
for variable, df, value_col, sd_col in [
        ("LAI", obs_lai, "LAI_obs", "LAI_sd"),
        ("TWSO", obs_twso, "TWSO_obs", "TWSO_obs_sd")]:
    obs = df.dropna(subset=[value_col])[["day", "DAE", "treatment", value_col, sd_col]].rename(
        columns={value_col: "value", sd_col: "SD"})
    series_parts.append(obs.assign(site=SITE, season=SEASON, soil="observed", variable=variable,
                                   series_kind="observation", depth=np.nan))
series_parts.append(probe.rename(columns={"SM_obs": "value"}).assign(
    site=SITE, season=SEASON, soil="observed", variable="VWC", series_kind="observation", SD=np.nan))
series = pd.concat(series_parts, ignore_index=True)
series = series[["day", "DAE", "variable", "value", "site", "season", "treatment", "soil", "series_kind", "SD", "depth"]]
assert set(series.season) == {SEASON}
assert not series.duplicated(["site", "season", "treatment", "soil", "series_kind", "variable", "day", "depth"]).any()
series.to_csv(source_dir / "s13_s14_nld_timeseries.csv", index=False)

# %% VWC MAE: match treatment/day/probe layer before pooling (same as S13)
PROBE_OF_LAYER = {0: 20, 1: 40}
pair_parts, mae_rows = [], []
for soil in SOIL_ORDER:
    matched = []
    for treatment in APPENDIX_TREATMENTS:
        daily = runs[soil][treatment]
        for layer, depth in PROBE_OF_LAYER.items():
            obs = probe[probe.treatment.eq(treatment) & probe.depth.eq(depth)]
            pairs = obs[["day", "SM_obs"]].merge(
                daily[["day", f"SM_L{layer}"]], on="day", how="inner", validate="one_to_one").dropna()
            matched.append(pairs.rename(columns={f"SM_L{layer}": "SM_sim"}).assign(
                treatment=treatment, layer=layer, depth=depth))
    pairs = pd.concat(matched, ignore_index=True)
    assert not pairs.empty
    pair_parts.append(pairs.assign(season=SEASON, soil=soil, site=SITE))
    mae_rows.append({"season": SEASON, "soil": soil, "n": len(pairs),
                     **MAEDecomposition(pairs.SM_obs, pairs.SM_sim).as_dict()})
sm_pairs = pd.concat(pair_parts, ignore_index=True)
sm_mae = pd.DataFrame(mae_rows)
assert not sm_pairs.duplicated(["site", "season", "soil", "treatment", "layer", "day"]).any()
a, b = [sm_pairs[sm_pairs.soil.eq(soil)].sort_values(["treatment", "layer", "day"]) for soil in SOIL_ORDER]
assert np.array_equal(a[["treatment", "layer", "day"]].to_numpy(), b[["treatment", "layer", "day"]].to_numpy())
assert np.allclose(sm_mae.MAE, sm_mae[["MAE_b", "MAE_p", "MAE_u"]].sum(axis=1))
sm_pairs.to_csv(source_dir / "s15_nld_sm_pairs.csv", index=False)
sm_mae.to_csv(source_dir / "s15_nld_sm_mae.csv", index=False)
print("Appendix VWC MAE: daily layer means, treatments", APPENDIX_TREATMENTS)
print(sm_mae.round(4).to_string(index=False))

# %% Preserve main Figure 6 yield pairs and MAE for AC/HTI/HB, NLD2025 only
pair_parts, yield_rows = [], []
for soil in SOIL_ORDER:
    matched = []
    for treatment in TREATMENTS:
        obs = obs_twso[obs_twso.treatment.eq(treatment)]
        pairs = obs[["day", "DAE", "TWSO_obs"]].merge(
            runs[soil][treatment][["day", "TWSO"]], on="day", how="inner", validate="one_to_one").dropna()
        matched.append(pairs.assign(soil=soil, season=SEASON, treatment=treatment))
    pairs = pd.concat(matched, ignore_index=True)
    pair_parts.append(pairs)
    yield_rows.append({"soil": soil, "season": SEASON, "n": len(pairs),
                       **MAEDecomposition(pairs.TWSO_obs, pairs.TWSO).as_dict()})
yield_pairs = pd.concat(pair_parts, ignore_index=True)
yield_mae = pd.DataFrame(yield_rows)
assert not yield_pairs.duplicated(["season", "soil", "treatment", "day"]).any()
yield_pairs.to_csv(source_dir / "fig06_nld_yield_pairs.csv", index=False)
yield_mae.to_csv(source_dir / "fig06_nld_yield_mae.csv", index=False)
print(yield_mae.round(3).to_string(index=False))
print("NLD2025 source tables written. Render S11/S13/S15 with 06.2.")
