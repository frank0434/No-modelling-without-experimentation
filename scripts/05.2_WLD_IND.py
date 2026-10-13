# %% IND2024 eight-case soil x CFET x PERDL factorial
# Two soils, each with CFET fitted on F0 then PERDL on R12 with CFET frozen.
# Every test is followed by its metrics and plots in visible cells.
# Full calibration/rendering are user-run; no model is published until all cases finish.
import sys
import json
import copy
import pickle
import hashlib
import pathlib
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pcse.input import WOFOST73SiteDataProvider
from pcse.engine import Engine

_here = pathlib.Path().resolve()
sys.path.insert(0, str(next(p for p in (_here, *_here.parents)
                            if (p/"scripts").exists() and (p/"data").exists()) / "scripts"))
from utils.paths import ROOT
from utils import BiomassPartitioning as biop
from utils import plotstyle, soil_curves as sc
from utils.loaders import (
    load_crop_data, load_experiments, find_experiment,
    load_weather, load_soil_mlwb, build_parameters,
    build_agro_with_irrigation, load_irrigation_csv,
    load_obs_yield, load_obs_lai, canonicalize_season,
)
from utils.phenology_helpers import derive_obs_fixed, load_phenology_params, emergence_date
from utils.metrics import summary_metrics, nrmse, MAEDecomposition
from optimization.config import OptimizationConfig, ParameterSpec
from optimization.resolver import ParameterSetResolver
from optimization.optimizer import TwoStageNLOptimizer

plotstyle.use_style()

# -- 1. Fixed settings for this run -------------------------------------------
SITE      = "IND"
SEASON    = "2024-2025"
CULTIVAR  = "Kennebec"
CONTROL   = "Full_0 Days"
TREATMENTS = ["Full_0 Days", "Full_6 Days", "Full_12 Days",
              "Reduced_0 Days", "Reduced_6 Days", "Reduced_12 Days"]

PAN_LAYER   = 1           # 30-35 cm plough pan; the layer above it is rooted
ROOT_BASE   = 30.0        # RDMCR: roots stop at the top of the pan
RAW_SOIL_FILE = "data/soil/india_soil_MLWB_2layer_wosten.yaml"
SOIL_FILE = "output/wld_growth/ind/india_soil_MLWB_2layer_fc_wp.yaml"
MLWB_CONF = str(ROOT / "data" / "conf" / "Wofost73_WLP_MLWB_SM.conf")
RUN_ID    = f"{SITE.lower()}_{SEASON}"

OUTPUT_DIR = ROOT / "output" / "wld_growth" / "ind"
CACHE_DIR  = OUTPUT_DIR / "cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Where the earlier calibration steps wrote their outputs.
SLATB_RUN_ID     = "v3"
PARTITION_RUN_ID = "v1"
GROWTH_DIR   = ROOT / "output" / "pp_growth"
TEFFMX_FIXED = 25.0
PHENO_ROW_PARAMS = ["TBASEM", "TSUM1", "TSUM2"]

cropd     = load_crop_data("data/crops_calibration")
exp       = find_experiment(load_experiments(location=SITE), SITE, SEASON, CULTIVAR)
EMERGENCE = emergence_date(SITE, SEASON, CULTIVAR)   # x=0 for every DAE axis below

soildata    = load_soil_mlwb(RAW_SOIL_FILE)
LAYER_THICK = [float(l["Thickness"])
               for l in soildata["SoilProfileDescription"]["SoilLayers"]]

print(f"[{SITE} {SEASON}] CO2={exp['CO2']} ppm; selected soil={SOIL_FILE}")
print(f"layer boundaries at {list(np.cumsum(LAYER_THICK))} cm; pan = layer {PAN_LAYER}; "
      f"roots stop at {ROOT_BASE:.0f} cm")
# %%
# -- 2. Crop parameters from the retained phenology and growth fits ---------------------
# CFET and PERDL are supplied separately for each soil; DEPNR stays at 3.
overrides = {}

# Use each season's post-emergence thermal sums, but keep emergence internally
# consistent: TEFFMX is fixed at 25 and TSUMEM is integrated under that same cap.
_pheno = load_phenology_params(field="OpenField", index=2)
_prow  = _pheno[(_pheno["site"] == SITE) & (_pheno["cultivar"] == CULTIVAR)
                 & (_pheno["year"].astype(str) == SEASON)].iloc[0]
for p in PHENO_ROW_PARAMS:
    overrides[p] = round(_prow[p])
overrides["TEFFMX"] = TEFFMX_FIXED
overrides["TSUMEM"] = round(derive_obs_fixed(
    SITE, CULTIVAR, [SEASON], teffmx=TEFFMX_FIXED, field="OpenField")["TSUMEM"], 1)

# TDWI scaled down by the observed establishment (emergence) rate for this season.
_est  = pd.read_csv(ROOT / "data" / "obs" / "IND_establishment_rate.csv")
_rate = _est.loc[_est["season"].astype(str) == SEASON, "Establishment_obs"].mean()
overrides["TDWI"] = exp["TDWI"] * _rate

# Calibrated SLATB (specific leaf area against development stage).
_slatb = pd.read_csv(GROWTH_DIR / f"pp_growth_params_{SLATB_RUN_ID}.csv")
_srows = _slatb[(_slatb["site"] == SITE) & (_slatb["cultivar"] == CULTIVAR)
                & (_slatb["param"] == "SLATB")].sort_values("dvs")
overrides["SLATB"] = [v for dvs, val in zip(_srows["dvs"], _srows["value"]) for v in (dvs, val)]

# Calibrated partitioning (FOTB/FLTB/FSTB) + leaf growth (RGRLAI) + leaf span (SPAN).
_part  = pd.read_csv(GROWTH_DIR / f"growth_best_params_{PARTITION_RUN_ID}.csv")
_prows = _part[(_part["site"] == SITE) & (_part["cultivar"] == CULTIVAR)
               & (_part["layer"] == "partitioning")]
_named = {p: float(v) for p, v in zip(_prows["param"], _prows["value"])}
FOTB, FLTB, FSTB = biop.compute_partition_functions(
    FOTBpar={k: _named[k] for k in ("FOTBa", "FOTBb")},
    FLTBpar={k: _named[k] for k in ("FLTBa", "FLTBb", "FLTBc", "FLTBd")},
    warn=False,
)
overrides["FOTB"]   = FOTB.tolist()
overrides["FLTB"]   = FLTB.tolist()
overrides["FSTB"]   = FSTB.tolist()
overrides["RGRLAI"] = _named["RGRLAI"]
overrides["SPAN"]   = _named["SPAN"]

# Max rooting depth: fixed at the top of the pan.
overrides["RDMCR"] = ROOT_BASE
print("crop parameters carried over:")
for k, v in overrides.items():
    print(f"  {k:8s} = {np.round(v, 4) if np.isscalar(v) else '[' + str(len(v)) + ' values]'}")


# %%
# -- 3. Weather, irrigation, agromanagement, site data ------------------------
wdp   = load_weather(exp["weather_file"])
sited = WOFOST73SiteDataProvider(WAV=exp["WAV"], CO2=exp["CO2"])

irr = load_irrigation_csv(ROOT / "data" / "agro" / "IND_irrigation.csv")
irr["season"] = irr["season"].astype(str).map(canonicalize_season)
irr = irr[irr["season"] == canonicalize_season(SEASON)]
agro_by_trt = {t: build_agro_with_irrigation(SITE, SEASON, CULTIVAR,
                                             irr[irr["treatment"] == t], efficiency=1)
               for t in TREATMENTS}

with open(ROOT / "data" / "agro" / "india_seasons.yaml") as f:
    run_days = yaml.safe_load(f)["seasons"][SEASON].get("run_days")

for t in TREATMENTS:
    sub = irr[irr["treatment"] == t]
    print(f"{t:16s} {len(sub):2d} irrigation events, {sub['mm'].sum():5.0f} mm; "
          f"run_days={run_days}")


# %%
# %% Raw PTF and FC+WP soil, with common absolute initial water
SOIL_ORDER = ["PTF raw", "FC + WP corrected"]
CASE_ORDER = ["Default", "CFET only", "PERDL only", "Both fitted"]
LAYER_THICK = [30.0, 5.0]
assert [float(layer["Thickness"]) for layer in soildata["SoilProfileDescription"]["SoilLayers"]] == LAYER_THICK
layers = pd.DataFrame([
    dict(layer_top=0, layer_bottom=30, bulk_density=1.533, topsoil=True),
    dict(layer_top=30, layer_bottom=35, bulk_density=1.75, topsoil=False),
]).assign(clay_pct=1.0, silt_pct=2.0, om_pct=0.69 * 1.724,
          CNRatioSOMI=9.0, Soil_pH=7.0, theta_r=0.01)
vg_raw = sc.vg_from_layers(layers, keep=["layer_top", "layer_bottom", "topsoil",
    "bulk_density", "om_pct", "CNRatioSOMI", "Soil_pH"])
vg_corrected = sc.correct_layers(vg_raw, [(0, 30)], 0.30, 0.10)
SOILS, RETENTION, CONDUCTIVITY = {}, {}, {}
for label, vg in (("PTF raw", vg_raw), ("FC + WP corrected", vg_corrected)):
    ret, cond = sc.curves_from(vg)
    sc.check_tables(ret, cond, label)
    SOILS[label] = sc.soil_dict(vg, ret, cond)
    RETENTION[label], CONDUCTIVITY[label] = ret, cond
sc.check_targets(RETENTION["FC + WP corrected"], [(0, 30)], 0.30, 0.10)
assert SOILS[SOIL_ORDER[0]]["SoilProfileDescription"]["SoilLayers"][1] == SOILS[SOIL_ORDER[1]]["SoilProfileDescription"]["SoilLayers"][1]
for built, original in zip(SOILS["PTF raw"]["SoilProfileDescription"]["SoilLayers"],
                           soildata["SoilProfileDescription"]["SoilLayers"]):
    np.testing.assert_allclose(built["SMfromPF"], original["SMfromPF"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(built["CONDfromPF"], original["CONDfromPF"], rtol=0, atol=1e-12)
common_sm = np.array([0.30, sc.read_pf(RETENTION["PTF raw"][(30, 35)], 4.2)])
SITE_BY_SOIL = {}
for label in SOIL_ORDER:
    wp = np.array([sc.read_pf(values, 4.2) for values in RETENTION[label].values()])
    assert np.all(common_sm >= wp)
    SITE_BY_SOIL[label] = WOFOST73SiteDataProvider(
        WAV=float(np.sum((common_sm - wp) * np.array(LAYER_THICK))),
        SMLIM=float(common_sm[0]), CO2=exp["CO2"])
print("Common initial VWC:", common_sm)

# %% Inspect soil correction immediately: retention above conductivity
fig, axes = plt.subplots(2, 2, figsize=(7, 5), sharex=True, sharey="row")
for col, layer in enumerate(((0, 30), (30, 35))):
    for label, color, line in (("PTF raw", "0.6", "--"), ("FC + WP corrected", "black", "-")):
        axes[0, col].plot(sc.PF_GRID, RETENTION[label][layer], color=color, linestyle=line, label=label)
        axes[1, col].plot(sc.PF_GRID, CONDUCTIVITY[label][layer], color=color, linestyle=line)
        for target_pf in (2.0, 4.2):
            axes[0, col].axvline(target_pf, color = "grey", linestyle=":")
            axes[1, col].axvline(target_pf, color="gray", linestyle=":")
    if col == 0:
        axes[0, col].plot([2.0, 4.2], [0.30, 0.10], "*", color="black", label="FC/WP targets")
    axes[0, col].set_title(f"{layer[0]}-{layer[1]} cm")
    axes[1, col].set_xlabel("pF")
axes[0, 0].set_ylabel("VWC (cm$^3$ cm$^{-3}$)")
axes[1, 0].set_ylabel("log$_{10}$ K (cm d$^{-1}$)")
axes[0, 0].legend(frameon=False)
fig.tight_layout()
plt.show()

# %% Minimal existing simulation and scoring plumbing
def run_mlwb(soil_name, treatment, extra):
    """One layered-balance run, returned from observed emergence onward."""
    effective = {**overrides, **extra}
    assert effective["TEFFMX"] == TEFFMX_FIXED, "IND TEFFMX must remain fixed at 25"
    assert np.isfinite(effective["TSUMEM"]) and effective["TSUMEM"] > 0, \
        "TSUMEM must be a positive value derived under TEFFMX=25"
    model = Engine(build_parameters(cropd, copy.deepcopy(SOILS[soil_name]), SITE_BY_SOIL[soil_name], CULTIVAR, effective),
                   wdp, copy.deepcopy(agro_by_trt[treatment]), config=MLWB_CONF)
    np.testing.assert_allclose(model.soil.states.SM, common_sm, rtol=0, atol=1e-12)
    initial_wc = model.soil.states.WC.copy()
    initial_ss = float(model.soil.states.SS)
    model.run(run_days) if run_days else model.run_till_terminate()
    df = pd.DataFrame(model.get_output())
    df["day"] = pd.to_datetime(df["day"])
    df["DAE"] = (df["day"] - EMERGENCE).dt.days
    # SM, WC, TRALY and Flow come out as one value per layer per day; spread each
    # into SM_L0..L2 etc. TRALY is crop-side and is None on days the crop is not
    # in the model yet, so those days get NaN, not a zero the model never produced.
    for name in ("SM", "WC", "TRALY", "Flow"):
        width = max(len(v) for v in df[name] if v is not None)
        arr = np.full((len(df), width), np.nan)
        for r, v in enumerate(df[name]):
            if v is not None:
                arr[r] = np.asarray(v, dtype=float)
        for i in range(width):
            df[f"{name}_L{i}"] = arr[:, i]
    df = df.drop(columns=["SM", "WC", "TRALY", "Flow"])
    above = LAYER_THICK[:PAN_LAYER]
    df["SM_profile"] = sum(th * df[f"SM_L{i}"] for i, th in enumerate(above)) / sum(above)
    for c in ("DVS", "LAI", "TWSO", "TRA", "RFTRA"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    np.testing.assert_allclose(df[["WC_L0", "WC_L1"]],
        df[["SM_L0", "SM_L1"]].to_numpy() * np.array(LAYER_THICK), rtol=0, atol=1e-10)
    residual = (initial_ss - df["SS"] + initial_wc.sum() - df[["WC_L0", "WC_L1"]].sum(axis=1)
        + df["RAINT"] + df["TOTIRR"] - df["WTRAT"] - df["EVST"] - df["EVWT"]
        - df["TSR"] - df["BOTTOMFLOWT"])
    assert residual.abs().max() < 1e-4
    return df[df["DAE"] >= 0].reset_index(drop=True)


# Crop-yaml defaults, read off a parameter set with nothing overridden.
_params_default = build_parameters(cropd, copy.deepcopy(SOILS["PTF raw"]), SITE_BY_SOIL["PTF raw"], CULTIVAR, overrides)
PARAMS_TRACKED = ["CFET", "DEPNR", "PERDL"]
BASE_DEFAULT   = {p: float(_params_default[p]) for p in PARAMS_TRACKED}
assert BASE_DEFAULT["DEPNR"] == 3.0
print("crop defaults: " + ", ".join(f"{k}={v:g}" for k, v in BASE_DEFAULT.items()))


# %%
# -- 5. Observations per treatment, all from emergence ------------------------
# Row counts are printed before anything is scored.
def _obs_by_trt(df, value_col):
    """One observation table, this season only, from emergence on, per treatment."""
    df = df.copy()
    df["season"] = df["season"].astype(str).map(canonicalize_season)
    df = df[df["season"] == canonicalize_season(SEASON)]
    if "cultivar" in df.columns:
        df = df[(df["cultivar"] == CULTIVAR) & (df["Field"] == "OpenField")]
    df["day"] = pd.to_datetime(df["day"])
    df["DAE"] = (df["day"] - EMERGENCE).dt.days
    df = df[df["DAE"] >= 0]
    assert not df.duplicated(["treatment", "day"]).any()
    return {t: (df[df["treatment"] == t][["day", "DAE", value_col]]
                .dropna().sort_values("day").reset_index(drop=True))
            for t in TREATMENTS}


_sm = pd.read_csv(ROOT / "data" / "obs" / f"{SITE}_SM_obs.csv")
_sm["day"] = pd.to_datetime(_sm["day"])
_sm = (_sm.groupby(["season", "treatment", "day"], as_index=False)
       .agg(SM_obs=("SM_obs", "mean"), SM_obs_sd=("SM_obs", "std")))

obs_sm   = _obs_by_trt(_sm, "SM_obs")
obs_lai  = _obs_by_trt(load_obs_lai(location=SITE).rename(columns={"LAI": "LAI_obs"}), "LAI_obs")
obs_twso = _obs_by_trt(load_obs_yield(location=SITE), "TWSO_obs")

for t in TREATMENTS:
    print(f"{t:16s} {len(obs_sm[t]):3d} SM dates, {len(obs_lai[t]):2d} LAI dates, "
          f"{len(obs_twso[t]):2d} TWSO dates")


# %%
# -- 6. One run, one score ----------------------------------------------------
# Both calibration steps use this same metric bundle and weighted objective.
# Treatment-level evaluations compare parameter sets on the same observations.
JOINT_WEIGHTS = {"sm": 0.5, "lai": 0.5, "twso": 1.0}   # the committed weighting


def _matched(df, obs_t, obs_col, sim_col):
    e = pd.merge(obs_t[["day", obs_col]], df[["day", sim_col]], on="day", how="inner", validate="one_to_one").dropna()
    assert len(e) == len(obs_t)
    return e[obs_col], e[sim_col]


def score(df, treat):
    """The metric bundle for one run: the five metrics on each observable, the
    weighted NRMSE, the matched-day counts and the crop-side terms."""
    o_sm, s_sm = _matched(df, obs_sm[treat],   "SM_obs",   "SM_profile")
    o_l,  s_l  = _matched(df, obs_lai[treat],  "LAI_obs",  "LAI")
    o_y,  s_y  = _matched(df, obs_twso[treat], "TWSO_obs", "TWSO")
    out = {f"sm_{k}": v for k, v in summary_metrics(o_sm, s_sm).items()}
    out.update({f"lai_{k}": v for k, v in summary_metrics(o_l, s_l).items()})
    out.update({f"twso_{k}": v for k, v in summary_metrics(o_y, s_y).items()})
    out["sm_bias"] = float((s_sm.to_numpy() - o_sm.to_numpy()).mean())
    out["sm_objective_component"] = nrmse(o_sm, s_sm) * JOINT_WEIGHTS["sm"]
    out["lai_objective_component"] = nrmse(o_l, s_l) * JOINT_WEIGHTS["lai"]
    out["twso_objective_component"] = nrmse(o_y, s_y) * JOINT_WEIGHTS["twso"]
    out["joint_nrmse"] = (out["sm_objective_component"] + out["lai_objective_component"]
                          + out["twso_objective_component"])
    out.update({"n_sm": len(o_sm), "n_lai": len(o_l), "n_twso": len(o_y),
                "twso_final": float(df["TWSO"].iloc[-1]),
                "laimax": float(df["LAI"].max()),
                "rftra_min": float(df["RFTRA"].min()),
                "days_stressed": int((df["RFTRA"] < 1 - 1e-6).sum()),
                "tra_season": float(df["WTRAT"].iloc[-1] - df["WTRAT"].iloc[0]),
                "evs_season": float(df["EVST"].iloc[-1] - df["EVST"].iloc[0]),
                "drain_season": float(df["BOTTOMFLOWT"].iloc[-1] - df["BOTTOMFLOWT"].iloc[0])})
    return out


# %%
# Final yield ranking uses the observed harvest date and replicate SD.
_yield = load_obs_yield(location=SITE)
_yield["season"] = _yield["season"].astype(str).map(canonicalize_season)
_yield["day"] = pd.to_datetime(_yield["day"])
_yield = _yield[(_yield["season"] == SEASON) & (_yield["cultivar"] == CULTIVAR)
                & (_yield["Field"] == "OpenField") & (_yield["day"] >= EMERGENCE)].dropna(subset=["TWSO_obs"])
assert not _yield.duplicated(["treatment", "day"]).any()
final_harvest_obs = _yield.sort_values("day").groupby("treatment", sort=False).tail(1).set_index("treatment")
assert set(final_harvest_obs.index) == set(TREATMENTS)
CASE_RUNS, CASE_PARAMS = {}, {}
TREATMENT_MARKER = {t: {"0 Days": "o", "6 Days": "s", "12 Days": "^"}[t.split("_")[1]] for t in TREATMENTS}
TREATMENT_COLOR = {t: "#0047E0" if t.startswith("Full_") else "#8B1C1C" for t in TREATMENTS}

# %% Scalar fits: exact bounds, common objective and separate soil/stage caches
BOUNDS = {"CFET": (0.83, 1.29), "PERDL": (0.015, 1.0)}
INITS = {"CFET": 1.06, "PERDL": 0.030}
MAX_EVALS_STAGE1, MAX_EVALS_STAGE2 = 150, 50
SEED, FTOL_REL = 42, 1e-4
CACHE_SCHEMA_VERSION = 4
CALIBRATION_TREATMENTS = {"CFET_fit": CONTROL, "PERDL_fit": "Reduced_12 Days"}
FIT_PARAMS = {"CFET_fit": ("CFET",), "PERDL_fit": ("PERDL",)}
FIT_SPECS = {fit: (ParameterSpec(name=params[0], lower=BOUNDS[params[0]][0],
    upper=BOUNDS[params[0]][1], init=INITS[params[0]],
    step=(BOUNDS[params[0]][1] - BOUNDS[params[0]][0]) * 0.1),)
    for fit, params in FIT_PARAMS.items()}

class _WLDObjective:
    """Minimal adapter satisfying TwoStageNLOptimizer's duck-typed contract:
    callable f(x, grad), .trace (dicts keyed 'x'/'overrides'/'rmse'), .best().
    The trace also keeps the whole score dict, not only the minimised value."""
    def __init__(self, resolver, runner, score_fn, key):
        self.resolver, self.runner, self.score_fn, self.key = resolver, runner, score_fn, key
        self.trace = []
        self._cache = {}

    def __call__(self, x, grad=None):
        k = tuple(np.asarray(x, dtype=float).tolist())
        if k in self._cache:
            return self._cache[k]
        named = self.resolver.vector_to_overrides(x)
        error = None
        try:
            s = self.score_fn(self.runner(named))
            val = float(s[self.key])
            if not np.isfinite(val):
                error = f"non-finite {self.key}: {val}"
                val = 1e9
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            s, val = {}, 1e9
        self.trace.append({"x": np.asarray(x, dtype=float).copy(), "overrides": named,
                           "rmse": val, "score": s, "error": error})
        self._cache[k] = val
        return val

    def best(self):
        return min(self.trace, key=lambda r: r["rmse"]) if self.trace else None


def _input_fingerprint(path):
    """Path, size and modification time for one fit input."""
    path = pathlib.Path(path)
    if not path.is_absolute():
        path = ROOT / path
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def _cache_key(soil_name, fit, fixed_params):
    treatment = CALIBRATION_TREATMENTS[fit]
    payload = dict(cache_schema=CACHE_SCHEMA_VERSION, soil=SOILS[soil_name],
        initial_SM=common_sm.tolist(), WAV=SITE_BY_SOIL[soil_name]["WAV"],
        SMLIM=SITE_BY_SOIL[soil_name]["SMLIM"], CO2=exp["CO2"],
        soil_name=soil_name, treatment=treatment, fit=fit, weights=JOINT_WEIGHTS,
        specs=[(s.name, s.lower, s.upper, s.init, s.step) for s in FIT_SPECS[fit]],
        fixed={p: v for p, v in fixed_params.items() if p not in FIT_PARAMS[fit]},
        run_days=run_days, evals=[MAX_EVALS_STAGE1, MAX_EVALS_STAGE2], seed=SEED,
        ftol=FTOL_REL, overrides=overrides,
        irrigation=irr[irr["treatment"] == treatment].to_json(orient="records"),
        inputs=[_input_fingerprint(path) for path in (exp["weather_file"], MLWB_CONF,
            "data/crops_calibration/potato.yaml", "data/agro/india_seasons.yaml",
            GROWTH_DIR / f"pp_growth_params_{SLATB_RUN_ID}.csv",
            GROWTH_DIR / f"growth_best_params_{PARTITION_RUN_ID}.csv",
            ROOT / "scripts" / "utils" / "soil_curves.py",
            ROOT / "scripts" / "utils" / "loaders.py",
            ROOT / "scripts" / "utils" / "metrics.py")],
        obs={key: data[treatment].to_json(orient="records")
             for key, data in (("sm", obs_sm), ("lai", obs_lai), ("twso", obs_twso))})
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def fit_parameter(soil_name, fit, fixed_params, before_run):
    treatment = CALIBRATION_TREATMENTS[fit]
    specs = FIT_SPECS[fit]
    cfg = OptimizationConfig(target=f"{SITE}-{soil_name}-{fit}", location=SITE,
        cultivar=CULTIVAR, seasons=(SEASON,), active_params=specs,
        max_evals_stage1=MAX_EVALS_STAGE1, max_evals_stage2=MAX_EVALS_STAGE2,
        ftol_rel=FTOL_REL, seed=SEED)
    resolver = ParameterSetResolver(cfg)
    objective = _WLDObjective(resolver,
        lambda ov: run_mlwb(soil_name, treatment, {**fixed_params, **ov}),
        lambda daily: score(daily, treatment), "joint_nrmse")
    soil_id = "raw" if soil_name == "PTF raw" else "fc_wp"
    cache_file = CACHE_DIR / f"opt_factorial_{RUN_ID}_{soil_id}_{fit}_{_cache_key(soil_name, fit, fixed_params)}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            cached = pickle.load(f)
        r1, r2 = cached["r1"], cached["r2"]
        objective.trace = cached["trace"]
    else:
        r1, r2 = TwoStageNLOptimizer(cfg, resolver, objective).optimize()
        with open(cache_file, "wb") as f:
            pickle.dump(dict(r1=r1, r2=r2, trace=objective.trace), f)
    fitted = {**fixed_params, **r2.overrides}
    assert fitted["DEPNR"] == 3.0
    for name in ("CFET", "PERDL", "DEPNR"):
        if name not in FIT_PARAMS[fit]:
            assert fitted[name] == fixed_params[name]
    for spec in specs:
        assert spec.lower <= fitted[spec.name] <= spec.upper
    checked_run = run_mlwb(soil_name, treatment, fitted)
    checked = score(checked_run, treatment)
    assert min(checked["n_sm"], checked["n_lai"], checked["n_twso"]) > 0
    assert np.isfinite(r2.f_best) and r2.f_best < 1e9
    np.testing.assert_allclose(checked["joint_nrmse"], r2.f_best, rtol=1e-8, atol=1e-10)
    before = score(before_run, treatment)
    record = dict(location=SITE, season=SEASON, cultivar=CULTIVAR, soil=soil_name,
        soil_file=RAW_SOIL_FILE if soil_name == "PTF raw" else SOIL_FILE,
        WAV=SITE_BY_SOIL[soil_name]["WAV"], SMLIM=SITE_BY_SOIL[soil_name]["SMLIM"],
        initial_SM_L0=common_sm[0], initial_SM_L1=common_sm[1],
        calibration_treatment=treatment, fit=fit, objective="joint_nrmse", **fitted,
        f_best=r2.f_best, f_recomputed=checked["joint_nrmse"], f_before=before["joint_nrmse"],
        f_stage1=r1.f_best, DEPNR_fixed_default=fitted["DEPNR"],
        n_evals=r1.n_evals + r2.n_evals, **checked,
        **{f"lower_{s.name}": s.lower for s in specs}, **{f"upper_{s.name}": s.upper for s in specs},
        **{f"at_bound_{s.name}": min(fitted[s.name] - s.lower, s.upper - fitted[s.name])
             <= 0.01 * (s.upper - s.lower) for s in specs})
    trace = pd.DataFrame([{"soil": soil_name, "fit": fit, "calibration_treatment": treatment,
        **fixed_params, **rec["overrides"], "score": rec["rmse"], "error": rec.get("error"),
        "stage": "stage1" if index < r1.n_evals else "stage2", **rec["score"]}
        for index, rec in enumerate(objective.trace)])
    trace.to_csv(OUTPUT_DIR / f"factorial_trace_{soil_id}_{fit}_{RUN_ID}.csv", index=False)
    print(f"{soil_name}, {fit}, {treatment}: weighted NRMSE {before['joint_nrmse']:.4f} -> {checked['joint_nrmse']:.4f}")
    print(pd.DataFrame([record]).to_string(index=False))
    return fitted, record, trace, checked_run

raw_default_params = dict(BASE_DEFAULT)
corrected_default_params = dict(BASE_DEFAULT)
# %% Test: PTF raw, Default; six treatments, one shared parameter set
raw_default_runs = {t: run_mlwb("PTF raw", t, raw_default_params) for t in TREATMENTS}
CASE_RUNS[("PTF raw", "Default")] = raw_default_runs
CASE_PARAMS[("PTF raw", "Default")] = dict(raw_default_params)
case_metrics = pd.DataFrame([dict(treatment=t, **raw_default_params, **score(raw_default_runs[t], t)) for t in TREATMENTS])
print("\nPTF raw / Default")
print(case_metrics[["treatment", "CFET", "PERDL", "sm_NRMSE", "lai_NRMSE", "twso_NRMSE", "joint_nrmse"]].round(4).to_string(index=False))

# %% Inspect immediately: PTF raw, Default, yield ranking and six SM trajectories
case_points = pd.DataFrame([dict(treatment=t,
    observed=final_harvest_obs.loc[t, "TWSO_obs"], sd=final_harvest_obs.loc[t, "TWSO_obs_sd"],
    simulated=float(raw_default_runs[t].loc[raw_default_runs[t]["day"] == final_harvest_obs.loc[t, "day"], "TWSO"].iloc[0]))
    for t in TREATMENTS])
rho = case_points.observed.corr(case_points.simulated, method="spearman")
limits = np.r_[case_points.observed - case_points.sd, case_points.observed + case_points.sd,
               case_points.simulated] / 1000
limits = [float(limits.min() * 0.9), float(limits.max() * 1.1)]
fig = plt.figure(figsize=(8, 7))
grid = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 1], hspace=0.4, wspace=0.12)
ax = fig.add_subplot(grid[0, 0])
ax.plot(limits, limits, color="0.6", linestyle="--", linewidth=0.8)
for row in case_points.itertuples():
    ax.errorbar(row.observed / 1000, row.simulated / 1000, xerr=row.sd / 1000,
        fmt=TREATMENT_MARKER[row.treatment], color=TREATMENT_COLOR[row.treatment],
        markerfacecolor="white", capsize=2, markersize=5)
ax.set_xlim(*limits); ax.set_ylim(*limits); ax.set_aspect("equal")
ax.set_xlabel("Observed yield (Mg ha$^{-1}$)")
ax.set_ylabel("Simulated yield (Mg ha$^{-1}$)")
ax.set_title(f"Yield ranking: rho={rho:.2f}")
ax = fig.add_subplot(grid[0, 1:])
ax.set_axis_off()
ax.table(cellText=[[t, *[f"{case_metrics.loc[case_metrics.treatment.eq(t), name].iloc[0]:.3f}"
    for name in ("sm_NRMSE", "lai_NRMSE", "twso_NRMSE", "joint_nrmse")]] for t in TREATMENTS],
    colLabels=["Treatment", "SM nRMSE", "LAI nRMSE", "TWSO nRMSE", "Weighted"], loc="center")
for index, t in enumerate(TREATMENTS):
    row, col = divmod(index, 3)
    ax = fig.add_subplot(grid[row + 1, col])
    ax.plot(raw_default_runs[t]["DAE"], raw_default_runs[t]["SM_profile"], color="black", label="Simulated")
    ax.plot(obs_sm[t]["DAE"], obs_sm[t]["SM_obs"], "o", markerfacecolor="white",
            color="black", markersize=3, label="Observed")
    ax.set_title(t.replace("_", " ").replace(" Days", " d"))
    ax.set_ylim(0, 0.4)
    if col == 0:
        ax.set_ylabel("0-30 cm VWC")
    else:
        ax.tick_params(labelleft=False)
    if row == 1:
        ax.set_xlabel("Days after emergence")
    else:
        ax.tick_params(labelbottom=False)
fig.suptitle(f"PTF raw / Default: CFET={raw_default_params['CFET']:.4f}, PERDL={raw_default_params['PERDL']:.4f}")
plt.show()

# %% Test: FC + WP corrected, Default; six treatments, one shared parameter set
corrected_default_runs = {t: run_mlwb("FC + WP corrected", t, corrected_default_params) for t in TREATMENTS}
CASE_RUNS[("FC + WP corrected", "Default")] = corrected_default_runs
CASE_PARAMS[("FC + WP corrected", "Default")] = dict(corrected_default_params)
case_metrics = pd.DataFrame([dict(treatment=t, **corrected_default_params, **score(corrected_default_runs[t], t)) for t in TREATMENTS])
print("\nFC + WP corrected / Default")
print(case_metrics[["treatment", "CFET", "PERDL", "sm_NRMSE", "lai_NRMSE", "twso_NRMSE", "joint_nrmse"]].round(4).to_string(index=False))

# %% Inspect immediately: FC + WP corrected, Default, yield ranking and six SM trajectories
case_points = pd.DataFrame([dict(treatment=t,
    observed=final_harvest_obs.loc[t, "TWSO_obs"], sd=final_harvest_obs.loc[t, "TWSO_obs_sd"],
    simulated=float(corrected_default_runs[t].loc[corrected_default_runs[t]["day"] == final_harvest_obs.loc[t, "day"], "TWSO"].iloc[0]))
    for t in TREATMENTS])
rho = case_points.observed.corr(case_points.simulated, method="spearman")
limits = np.r_[case_points.observed - case_points.sd, case_points.observed + case_points.sd,
               case_points.simulated] / 1000
limits = [float(limits.min() * 0.9), float(limits.max() * 1.1)]
fig = plt.figure(figsize=(8, 7))
grid = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 1], hspace=0.4, wspace=0.12)
ax = fig.add_subplot(grid[0, 0])
ax.plot(limits, limits, color="0.6", linestyle="--", linewidth=0.8)
for row in case_points.itertuples():
    ax.errorbar(row.observed / 1000, row.simulated / 1000, xerr=row.sd / 1000,
        fmt=TREATMENT_MARKER[row.treatment], color=TREATMENT_COLOR[row.treatment],
        markerfacecolor="white", capsize=2, markersize=5)
ax.set_xlim(*limits); ax.set_ylim(*limits); ax.set_aspect("equal")
ax.set_xlabel("Observed yield (Mg ha$^{-1}$)")
ax.set_ylabel("Simulated yield (Mg ha$^{-1}$)")
ax.set_title(f"Yield ranking: rho={rho:.2f}")
ax = fig.add_subplot(grid[0, 1:])
ax.set_axis_off()
ax.table(cellText=[[t, *[f"{case_metrics.loc[case_metrics.treatment.eq(t), name].iloc[0]:.3f}"
    for name in ("sm_NRMSE", "lai_NRMSE", "twso_NRMSE", "joint_nrmse")]] for t in TREATMENTS],
    colLabels=["Treatment", "SM nRMSE", "LAI nRMSE", "TWSO nRMSE", "Weighted"], loc="center")
for index, t in enumerate(TREATMENTS):
    row, col = divmod(index, 3)
    ax = fig.add_subplot(grid[row + 1, col])
    ax.plot(corrected_default_runs[t]["DAE"], corrected_default_runs[t]["SM_profile"], color="black", label="Simulated")
    ax.plot(obs_sm[t]["DAE"], obs_sm[t]["SM_obs"], "o", markerfacecolor="white",
            color="black", markersize=3, label="Observed")
    ax.set_title(t.replace("_", " ").replace(" Days", " d"))
    ax.set_ylim(0, 0.4)
    if col == 0:
        ax.set_ylabel("0-30 cm VWC")
    else:
        ax.tick_params(labelleft=False)
    if row == 1:
        ax.set_xlabel("Days after emergence")
    else:
        ax.tick_params(labelbottom=False)
fig.suptitle(f"FC + WP corrected / Default: CFET={corrected_default_params['CFET']:.4f}, PERDL={corrected_default_params['PERDL']:.4f}")
plt.show()

# %% Fit CFET_fit: PTF raw, Full_0 Days; one active scalar
raw_cfet_params, raw_cfet_record, raw_cfet_trace, raw_cfet_checked = fit_parameter(
    "PTF raw", "CFET_fit", raw_default_params, raw_default_runs[CONTROL])

# %% Inspect fitting immediately: trace and same-treatment before/after
fig, axes = plt.subplots(2, 2, figsize=(7, 5))
axes[0, 0].plot(np.arange(len(raw_cfet_trace)), raw_cfet_trace["score"], color="black")
axes[0, 0].set_xlabel("Evaluation")
axes[0, 0].set_ylabel("Weighted nRMSE")
for ax, sim, obs, measured, ylabel in (
        (axes[0, 1], "SM_profile", obs_sm["Full_0 Days"], "SM_obs", "VWC"),
        (axes[1, 0], "LAI", obs_lai["Full_0 Days"], "LAI_obs", "LAI"),
        (axes[1, 1], "TWSO", obs_twso["Full_0 Days"], "TWSO_obs", "TWSO (kg/ha)")):
    ax.plot(raw_default_runs[CONTROL]["DAE"], raw_default_runs[CONTROL][sim], color="0.6", linestyle="--", label="Before")
    ax.plot(raw_cfet_checked["DAE"], raw_cfet_checked[sim], color="black", label="After")
    ax.plot(obs["DAE"], obs[measured], "o", color="black", markerfacecolor="white", markersize=3, label="Observed")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Days after emergence")
axes[0, 1].legend(frameon=False)
fig.suptitle("PTF raw: CFET_fit on Full_0 Days")
fig.tight_layout()
plt.show()

# %% Test: PTF raw, CFET only; six treatments, one shared parameter set
raw_cfet_runs = {t: raw_cfet_checked if t == "Full_0 Days" else run_mlwb("PTF raw", t, raw_cfet_params) for t in TREATMENTS}
CASE_RUNS[("PTF raw", "CFET only")] = raw_cfet_runs
CASE_PARAMS[("PTF raw", "CFET only")] = dict(raw_cfet_params)
case_metrics = pd.DataFrame([dict(treatment=t, **raw_cfet_params, **score(raw_cfet_runs[t], t)) for t in TREATMENTS])
print("\nPTF raw / CFET only")
print(case_metrics[["treatment", "CFET", "PERDL", "sm_NRMSE", "lai_NRMSE", "twso_NRMSE", "joint_nrmse"]].round(4).to_string(index=False))

# %% Inspect immediately: PTF raw, CFET only, yield ranking and six SM trajectories
case_points = pd.DataFrame([dict(treatment=t,
    observed=final_harvest_obs.loc[t, "TWSO_obs"], sd=final_harvest_obs.loc[t, "TWSO_obs_sd"],
    simulated=float(raw_cfet_runs[t].loc[raw_cfet_runs[t]["day"] == final_harvest_obs.loc[t, "day"], "TWSO"].iloc[0]))
    for t in TREATMENTS])
rho = case_points.observed.corr(case_points.simulated, method="spearman")
limits = np.r_[case_points.observed - case_points.sd, case_points.observed + case_points.sd,
               case_points.simulated] / 1000
limits = [float(limits.min() * 0.9), float(limits.max() * 1.1)]
fig = plt.figure(figsize=(8, 7))
grid = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 1], hspace=0.4, wspace=0.12)
ax = fig.add_subplot(grid[0, 0])
ax.plot(limits, limits, color="0.6", linestyle="--", linewidth=0.8)
for row in case_points.itertuples():
    ax.errorbar(row.observed / 1000, row.simulated / 1000, xerr=row.sd / 1000,
        fmt=TREATMENT_MARKER[row.treatment], color=TREATMENT_COLOR[row.treatment],
        markerfacecolor="white", capsize=2, markersize=5)
ax.set_xlim(*limits); ax.set_ylim(*limits); ax.set_aspect("equal")
ax.set_xlabel("Observed yield (Mg ha$^{-1}$)")
ax.set_ylabel("Simulated yield (Mg ha$^{-1}$)")
ax.set_title(f"Yield ranking: rho={rho:.2f}")
ax = fig.add_subplot(grid[0, 1:])
ax.set_axis_off()
ax.table(cellText=[[t, *[f"{case_metrics.loc[case_metrics.treatment.eq(t), name].iloc[0]:.3f}"
    for name in ("sm_NRMSE", "lai_NRMSE", "twso_NRMSE", "joint_nrmse")]] for t in TREATMENTS],
    colLabels=["Treatment", "SM nRMSE", "LAI nRMSE", "TWSO nRMSE", "Weighted"], loc="center")
for index, t in enumerate(TREATMENTS):
    row, col = divmod(index, 3)
    ax = fig.add_subplot(grid[row + 1, col])
    ax.plot(raw_cfet_runs[t]["DAE"], raw_cfet_runs[t]["SM_profile"], color="black", label="Simulated")
    ax.plot(obs_sm[t]["DAE"], obs_sm[t]["SM_obs"], "o", markerfacecolor="white",
            color="black", markersize=3, label="Observed")
    ax.set_title(t.replace("_", " ").replace(" Days", " d"))
    ax.set_ylim(0, 0.4)
    if col == 0:
        ax.set_ylabel("0-30 cm VWC")
    else:
        ax.tick_params(labelleft=False)
    if row == 1:
        ax.set_xlabel("Days after emergence")
    else:
        ax.tick_params(labelbottom=False)
fig.suptitle(f"PTF raw / CFET only: CFET={raw_cfet_params['CFET']:.4f}, PERDL={raw_cfet_params['PERDL']:.4f}")
plt.show()

# %% Fit PERDL_fit: PTF raw, Reduced_12 Days; one active scalar
raw_both_params, raw_both_record, raw_both_trace, raw_both_checked = fit_parameter(
    "PTF raw", "PERDL_fit", raw_cfet_params, raw_cfet_runs["Reduced_12 Days"])

# %% Inspect fitting immediately: trace and same-treatment before/after
fig, axes = plt.subplots(2, 2, figsize=(7, 5))
axes[0, 0].plot(np.arange(len(raw_both_trace)), raw_both_trace["score"], color="black")
axes[0, 0].set_xlabel("Evaluation")
axes[0, 0].set_ylabel("Weighted nRMSE")
for ax, sim, obs, measured, ylabel in (
        (axes[0, 1], "SM_profile", obs_sm["Reduced_12 Days"], "SM_obs", "VWC"),
        (axes[1, 0], "LAI", obs_lai["Reduced_12 Days"], "LAI_obs", "LAI"),
        (axes[1, 1], "TWSO", obs_twso["Reduced_12 Days"], "TWSO_obs", "TWSO (kg/ha)")):
    ax.plot(raw_cfet_runs["Reduced_12 Days"]["DAE"], raw_cfet_runs["Reduced_12 Days"][sim], color="0.6", linestyle="--", label="Before")
    ax.plot(raw_both_checked["DAE"], raw_both_checked[sim], color="black", label="After")
    ax.plot(obs["DAE"], obs[measured], "o", color="black", markerfacecolor="white", markersize=3, label="Observed")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Days after emergence")
axes[0, 1].legend(frameon=False)
fig.suptitle("PTF raw: PERDL_fit on Reduced_12 Days")
fig.tight_layout()
plt.show()

# %% Test: PTF raw, Both fitted; six treatments, one shared parameter set
raw_both_runs = {t: raw_both_checked if t == "Reduced_12 Days" else run_mlwb("PTF raw", t, raw_both_params) for t in TREATMENTS}
CASE_RUNS[("PTF raw", "Both fitted")] = raw_both_runs
CASE_PARAMS[("PTF raw", "Both fitted")] = dict(raw_both_params)
case_metrics = pd.DataFrame([dict(treatment=t, **raw_both_params, **score(raw_both_runs[t], t)) for t in TREATMENTS])
print("\nPTF raw / Both fitted")
print(case_metrics[["treatment", "CFET", "PERDL", "sm_NRMSE", "lai_NRMSE", "twso_NRMSE", "joint_nrmse"]].round(4).to_string(index=False))

# %% Inspect immediately: PTF raw, Both fitted, yield ranking and six SM trajectories
case_points = pd.DataFrame([dict(treatment=t,
    observed=final_harvest_obs.loc[t, "TWSO_obs"], sd=final_harvest_obs.loc[t, "TWSO_obs_sd"],
    simulated=float(raw_both_runs[t].loc[raw_both_runs[t]["day"] == final_harvest_obs.loc[t, "day"], "TWSO"].iloc[0]))
    for t in TREATMENTS])
rho = case_points.observed.corr(case_points.simulated, method="spearman")
limits = np.r_[case_points.observed - case_points.sd, case_points.observed + case_points.sd,
               case_points.simulated] / 1000
limits = [float(limits.min() * 0.9), float(limits.max() * 1.1)]
fig = plt.figure(figsize=(8, 7))
grid = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 1], hspace=0.4, wspace=0.12)
ax = fig.add_subplot(grid[0, 0])
ax.plot(limits, limits, color="0.6", linestyle="--", linewidth=0.8)
for row in case_points.itertuples():
    ax.errorbar(row.observed / 1000, row.simulated / 1000, xerr=row.sd / 1000,
        fmt=TREATMENT_MARKER[row.treatment], color=TREATMENT_COLOR[row.treatment],
        markerfacecolor="white", capsize=2, markersize=5)
ax.set_xlim(*limits); ax.set_ylim(*limits); ax.set_aspect("equal")
ax.set_xlabel("Observed yield (Mg ha$^{-1}$)")
ax.set_ylabel("Simulated yield (Mg ha$^{-1}$)")
ax.set_title(f"Yield ranking: rho={rho:.2f}")
ax = fig.add_subplot(grid[0, 1:])
ax.set_axis_off()
ax.table(cellText=[[t, *[f"{case_metrics.loc[case_metrics.treatment.eq(t), name].iloc[0]:.3f}"
    for name in ("sm_NRMSE", "lai_NRMSE", "twso_NRMSE", "joint_nrmse")]] for t in TREATMENTS],
    colLabels=["Treatment", "SM nRMSE", "LAI nRMSE", "TWSO nRMSE", "Weighted"], loc="center")
for index, t in enumerate(TREATMENTS):
    row, col = divmod(index, 3)
    ax = fig.add_subplot(grid[row + 1, col])
    ax.plot(raw_both_runs[t]["DAE"], raw_both_runs[t]["SM_profile"], color="black", label="Simulated")
    ax.plot(obs_sm[t]["DAE"], obs_sm[t]["SM_obs"], "o", markerfacecolor="white",
            color="black", markersize=3, label="Observed")
    ax.set_title(t.replace("_", " ").replace(" Days", " d"))
    ax.set_ylim(0, 0.4)
    if col == 0:
        ax.set_ylabel("0-30 cm VWC")
    else:
        ax.tick_params(labelleft=False)
    if row == 1:
        ax.set_xlabel("Days after emergence")
    else:
        ax.tick_params(labelbottom=False)
fig.suptitle(f"PTF raw / Both fitted: CFET={raw_both_params['CFET']:.4f}, PERDL={raw_both_params['PERDL']:.4f}")
plt.show()

# %% Cross-test parameters: PTF raw, default CFET with its sequentially fitted PERDL
raw_perdl_params = {**BASE_DEFAULT, "PERDL": raw_both_params["PERDL"]}
# %% Test: PTF raw, PERDL only; six treatments, one shared parameter set
raw_perdl_runs = {t: run_mlwb("PTF raw", t, raw_perdl_params) for t in TREATMENTS}
CASE_RUNS[("PTF raw", "PERDL only")] = raw_perdl_runs
CASE_PARAMS[("PTF raw", "PERDL only")] = dict(raw_perdl_params)
case_metrics = pd.DataFrame([dict(treatment=t, **raw_perdl_params, **score(raw_perdl_runs[t], t)) for t in TREATMENTS])
print("\nPTF raw / PERDL only")
print(case_metrics[["treatment", "CFET", "PERDL", "sm_NRMSE", "lai_NRMSE", "twso_NRMSE", "joint_nrmse"]].round(4).to_string(index=False))

# %% Inspect immediately: PTF raw, PERDL only, yield ranking and six SM trajectories
case_points = pd.DataFrame([dict(treatment=t,
    observed=final_harvest_obs.loc[t, "TWSO_obs"], sd=final_harvest_obs.loc[t, "TWSO_obs_sd"],
    simulated=float(raw_perdl_runs[t].loc[raw_perdl_runs[t]["day"] == final_harvest_obs.loc[t, "day"], "TWSO"].iloc[0]))
    for t in TREATMENTS])
rho = case_points.observed.corr(case_points.simulated, method="spearman")
limits = np.r_[case_points.observed - case_points.sd, case_points.observed + case_points.sd,
               case_points.simulated] / 1000
limits = [float(limits.min() * 0.9), float(limits.max() * 1.1)]
fig = plt.figure(figsize=(8, 7))
grid = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 1], hspace=0.4, wspace=0.12)
ax = fig.add_subplot(grid[0, 0])
ax.plot(limits, limits, color="0.6", linestyle="--", linewidth=0.8)
for row in case_points.itertuples():
    ax.errorbar(row.observed / 1000, row.simulated / 1000, xerr=row.sd / 1000,
        fmt=TREATMENT_MARKER[row.treatment], color=TREATMENT_COLOR[row.treatment],
        markerfacecolor="white", capsize=2, markersize=5)
ax.set_xlim(*limits); ax.set_ylim(*limits); ax.set_aspect("equal")
ax.set_xlabel("Observed yield (Mg ha$^{-1}$)")
ax.set_ylabel("Simulated yield (Mg ha$^{-1}$)")
ax.set_title(f"Yield ranking: rho={rho:.2f}")
ax = fig.add_subplot(grid[0, 1:])
ax.set_axis_off()
ax.table(cellText=[[t, *[f"{case_metrics.loc[case_metrics.treatment.eq(t), name].iloc[0]:.3f}"
    for name in ("sm_NRMSE", "lai_NRMSE", "twso_NRMSE", "joint_nrmse")]] for t in TREATMENTS],
    colLabels=["Treatment", "SM nRMSE", "LAI nRMSE", "TWSO nRMSE", "Weighted"], loc="center")
for index, t in enumerate(TREATMENTS):
    row, col = divmod(index, 3)
    ax = fig.add_subplot(grid[row + 1, col])
    ax.plot(raw_perdl_runs[t]["DAE"], raw_perdl_runs[t]["SM_profile"], color="black", label="Simulated")
    ax.plot(obs_sm[t]["DAE"], obs_sm[t]["SM_obs"], "o", markerfacecolor="white",
            color="black", markersize=3, label="Observed")
    ax.set_title(t.replace("_", " ").replace(" Days", " d"))
    ax.set_ylim(0, 0.4)
    if col == 0:
        ax.set_ylabel("0-30 cm VWC")
    else:
        ax.tick_params(labelleft=False)
    if row == 1:
        ax.set_xlabel("Days after emergence")
    else:
        ax.tick_params(labelbottom=False)
fig.suptitle(f"PTF raw / PERDL only: CFET={raw_perdl_params['CFET']:.4f}, PERDL={raw_perdl_params['PERDL']:.4f}")
plt.show()

# %% Fit CFET_fit: FC + WP corrected, Full_0 Days; one active scalar
corrected_cfet_params, corrected_cfet_record, corrected_cfet_trace, corrected_cfet_checked = fit_parameter(
    "FC + WP corrected", "CFET_fit", corrected_default_params, corrected_default_runs[CONTROL])

# %% Inspect fitting immediately: trace and same-treatment before/after
fig, axes = plt.subplots(2, 2, figsize=(7, 5))
axes[0, 0].plot(np.arange(len(corrected_cfet_trace)), corrected_cfet_trace["score"], color="black")
axes[0, 0].set_xlabel("Evaluation")
axes[0, 0].set_ylabel("Weighted nRMSE")
for ax, sim, obs, measured, ylabel in (
        (axes[0, 1], "SM_profile", obs_sm["Full_0 Days"], "SM_obs", "VWC"),
        (axes[1, 0], "LAI", obs_lai["Full_0 Days"], "LAI_obs", "LAI"),
        (axes[1, 1], "TWSO", obs_twso["Full_0 Days"], "TWSO_obs", "TWSO (kg/ha)")):
    ax.plot(corrected_default_runs[CONTROL]["DAE"], corrected_default_runs[CONTROL][sim], color="0.6", linestyle="--", label="Before")
    ax.plot(corrected_cfet_checked["DAE"], corrected_cfet_checked[sim], color="black", label="After")
    ax.plot(obs["DAE"], obs[measured], "o", color="black", markerfacecolor="white", markersize=3, label="Observed")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Days after emergence")
axes[0, 1].legend(frameon=False)
fig.suptitle("FC + WP corrected: CFET_fit on Full_0 Days")
fig.tight_layout()
plt.show()

# %% Test: FC + WP corrected, CFET only; six treatments, one shared parameter set
corrected_cfet_runs = {t: corrected_cfet_checked if t == "Full_0 Days" else run_mlwb("FC + WP corrected", t, corrected_cfet_params) for t in TREATMENTS}
CASE_RUNS[("FC + WP corrected", "CFET only")] = corrected_cfet_runs
CASE_PARAMS[("FC + WP corrected", "CFET only")] = dict(corrected_cfet_params)
case_metrics = pd.DataFrame([dict(treatment=t, **corrected_cfet_params, **score(corrected_cfet_runs[t], t)) for t in TREATMENTS])
print("\nFC + WP corrected / CFET only")
print(case_metrics[["treatment", "CFET", "PERDL", "sm_NRMSE", "lai_NRMSE", "twso_NRMSE", "joint_nrmse"]].round(4).to_string(index=False))

# %% Inspect immediately: FC + WP corrected, CFET only, yield ranking and six SM trajectories
case_points = pd.DataFrame([dict(treatment=t,
    observed=final_harvest_obs.loc[t, "TWSO_obs"], sd=final_harvest_obs.loc[t, "TWSO_obs_sd"],
    simulated=float(corrected_cfet_runs[t].loc[corrected_cfet_runs[t]["day"] == final_harvest_obs.loc[t, "day"], "TWSO"].iloc[0]))
    for t in TREATMENTS])
rho = case_points.observed.corr(case_points.simulated, method="spearman")
limits = np.r_[case_points.observed - case_points.sd, case_points.observed + case_points.sd,
               case_points.simulated] / 1000
limits = [float(limits.min() * 0.9), float(limits.max() * 1.1)]
fig = plt.figure(figsize=(8, 7))
grid = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 1], hspace=0.4, wspace=0.12)
ax = fig.add_subplot(grid[0, 0])
ax.plot(limits, limits, color="0.6", linestyle="--", linewidth=0.8)
for row in case_points.itertuples():
    ax.errorbar(row.observed / 1000, row.simulated / 1000, xerr=row.sd / 1000,
        fmt=TREATMENT_MARKER[row.treatment], color=TREATMENT_COLOR[row.treatment],
        markerfacecolor="white", capsize=2, markersize=5)
ax.set_xlim(*limits); ax.set_ylim(*limits); ax.set_aspect("equal")
ax.set_xlabel("Observed yield (Mg ha$^{-1}$)")
ax.set_ylabel("Simulated yield (Mg ha$^{-1}$)")
ax.set_title(f"Yield ranking: rho={rho:.2f}")
ax = fig.add_subplot(grid[0, 1:])
ax.set_axis_off()
ax.table(cellText=[[t, *[f"{case_metrics.loc[case_metrics.treatment.eq(t), name].iloc[0]:.3f}"
    for name in ("sm_NRMSE", "lai_NRMSE", "twso_NRMSE", "joint_nrmse")]] for t in TREATMENTS],
    colLabels=["Treatment", "SM nRMSE", "LAI nRMSE", "TWSO nRMSE", "Weighted"], loc="center")
for index, t in enumerate(TREATMENTS):
    row, col = divmod(index, 3)
    ax = fig.add_subplot(grid[row + 1, col])
    ax.plot(corrected_cfet_runs[t]["DAE"], corrected_cfet_runs[t]["SM_profile"], color="black", label="Simulated")
    ax.plot(obs_sm[t]["DAE"], obs_sm[t]["SM_obs"], "o", markerfacecolor="white",
            color="black", markersize=3, label="Observed")
    ax.set_title(t.replace("_", " ").replace(" Days", " d"))
    ax.set_ylim(0, 0.4)
    if col == 0:
        ax.set_ylabel("0-30 cm VWC")
    else:
        ax.tick_params(labelleft=False)
    if row == 1:
        ax.set_xlabel("Days after emergence")
    else:
        ax.tick_params(labelbottom=False)
fig.suptitle(f"FC + WP corrected / CFET only: CFET={corrected_cfet_params['CFET']:.4f}, PERDL={corrected_cfet_params['PERDL']:.4f}")
plt.show()

# %% Fit PERDL_fit: FC + WP corrected, Reduced_12 Days; one active scalar
corrected_both_params, corrected_both_record, corrected_both_trace, corrected_both_checked = fit_parameter(
    "FC + WP corrected", "PERDL_fit", corrected_cfet_params, corrected_cfet_runs["Reduced_12 Days"])

# %% Inspect fitting immediately: trace and same-treatment before/after
fig, axes = plt.subplots(2, 2, figsize=(7, 5))
axes[0, 0].plot(np.arange(len(corrected_both_trace)), corrected_both_trace["score"], color="black")
axes[0, 0].set_xlabel("Evaluation")
axes[0, 0].set_ylabel("Weighted nRMSE")
for ax, sim, obs, measured, ylabel in (
        (axes[0, 1], "SM_profile", obs_sm["Reduced_12 Days"], "SM_obs", "VWC"),
        (axes[1, 0], "LAI", obs_lai["Reduced_12 Days"], "LAI_obs", "LAI"),
        (axes[1, 1], "TWSO", obs_twso["Reduced_12 Days"], "TWSO_obs", "TWSO (kg/ha)")):
    ax.plot(corrected_cfet_runs["Reduced_12 Days"]["DAE"], corrected_cfet_runs["Reduced_12 Days"][sim], color="0.6", linestyle="--", label="Before")
    ax.plot(corrected_both_checked["DAE"], corrected_both_checked[sim], color="black", label="After")
    ax.plot(obs["DAE"], obs[measured], "o", color="black", markerfacecolor="white", markersize=3, label="Observed")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Days after emergence")
axes[0, 1].legend(frameon=False)
fig.suptitle("FC + WP corrected: PERDL_fit on Reduced_12 Days")
fig.tight_layout()
plt.show()

# %% Test: FC + WP corrected, Both fitted; six treatments, one shared parameter set
corrected_both_runs = {t: corrected_both_checked if t == "Reduced_12 Days" else run_mlwb("FC + WP corrected", t, corrected_both_params) for t in TREATMENTS}
CASE_RUNS[("FC + WP corrected", "Both fitted")] = corrected_both_runs
CASE_PARAMS[("FC + WP corrected", "Both fitted")] = dict(corrected_both_params)
case_metrics = pd.DataFrame([dict(treatment=t, **corrected_both_params, **score(corrected_both_runs[t], t)) for t in TREATMENTS])
print("\nFC + WP corrected / Both fitted")
print(case_metrics[["treatment", "CFET", "PERDL", "sm_NRMSE", "lai_NRMSE", "twso_NRMSE", "joint_nrmse"]].round(4).to_string(index=False))

# %% Inspect immediately: FC + WP corrected, Both fitted, yield ranking and six SM trajectories
case_points = pd.DataFrame([dict(treatment=t,
    observed=final_harvest_obs.loc[t, "TWSO_obs"], sd=final_harvest_obs.loc[t, "TWSO_obs_sd"],
    simulated=float(corrected_both_runs[t].loc[corrected_both_runs[t]["day"] == final_harvest_obs.loc[t, "day"], "TWSO"].iloc[0]))
    for t in TREATMENTS])
rho = case_points.observed.corr(case_points.simulated, method="spearman")
limits = np.r_[case_points.observed - case_points.sd, case_points.observed + case_points.sd,
               case_points.simulated] / 1000
limits = [float(limits.min() * 0.9), float(limits.max() * 1.1)]
fig = plt.figure(figsize=(8, 7))
grid = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 1], hspace=0.4, wspace=0.12)
ax = fig.add_subplot(grid[0, 0])
ax.plot(limits, limits, color="0.6", linestyle="--", linewidth=0.8)
for row in case_points.itertuples():
    ax.errorbar(row.observed / 1000, row.simulated / 1000, xerr=row.sd / 1000,
        fmt=TREATMENT_MARKER[row.treatment], color=TREATMENT_COLOR[row.treatment],
        markerfacecolor="white", capsize=2, markersize=5)
ax.set_xlim(*limits); ax.set_ylim(*limits); ax.set_aspect("equal")
ax.set_xlabel("Observed yield (Mg ha$^{-1}$)")
ax.set_ylabel("Simulated yield (Mg ha$^{-1}$)")
ax.set_title(f"Yield ranking: rho={rho:.2f}")
ax = fig.add_subplot(grid[0, 1:])
ax.set_axis_off()
ax.table(cellText=[[t, *[f"{case_metrics.loc[case_metrics.treatment.eq(t), name].iloc[0]:.3f}"
    for name in ("sm_NRMSE", "lai_NRMSE", "twso_NRMSE", "joint_nrmse")]] for t in TREATMENTS],
    colLabels=["Treatment", "SM nRMSE", "LAI nRMSE", "TWSO nRMSE", "Weighted"], loc="center")
for index, t in enumerate(TREATMENTS):
    row, col = divmod(index, 3)
    ax = fig.add_subplot(grid[row + 1, col])
    ax.plot(corrected_both_runs[t]["DAE"], corrected_both_runs[t]["SM_profile"], color="black", label="Simulated")
    ax.plot(obs_sm[t]["DAE"], obs_sm[t]["SM_obs"], "o", markerfacecolor="white",
            color="black", markersize=3, label="Observed")
    ax.set_title(t.replace("_", " ").replace(" Days", " d"))
    ax.set_ylim(0, 0.4)
    if col == 0:
        ax.set_ylabel("0-30 cm VWC")
    else:
        ax.tick_params(labelleft=False)
    if row == 1:
        ax.set_xlabel("Days after emergence")
    else:
        ax.tick_params(labelbottom=False)
fig.suptitle(f"FC + WP corrected / Both fitted: CFET={corrected_both_params['CFET']:.4f}, PERDL={corrected_both_params['PERDL']:.4f}")
plt.show()

# %% Cross-test parameters: FC + WP corrected, default CFET with its sequentially fitted PERDL
corrected_perdl_params = {**BASE_DEFAULT, "PERDL": corrected_both_params["PERDL"]}
# %% Test: FC + WP corrected, PERDL only; six treatments, one shared parameter set
corrected_perdl_runs = {t: run_mlwb("FC + WP corrected", t, corrected_perdl_params) for t in TREATMENTS}
CASE_RUNS[("FC + WP corrected", "PERDL only")] = corrected_perdl_runs
CASE_PARAMS[("FC + WP corrected", "PERDL only")] = dict(corrected_perdl_params)
case_metrics = pd.DataFrame([dict(treatment=t, **corrected_perdl_params, **score(corrected_perdl_runs[t], t)) for t in TREATMENTS])
print("\nFC + WP corrected / PERDL only")
print(case_metrics[["treatment", "CFET", "PERDL", "sm_NRMSE", "lai_NRMSE", "twso_NRMSE", "joint_nrmse"]].round(4).to_string(index=False))

# %% Inspect immediately: FC + WP corrected, PERDL only, yield ranking and six SM trajectories
case_points = pd.DataFrame([dict(treatment=t,
    observed=final_harvest_obs.loc[t, "TWSO_obs"], sd=final_harvest_obs.loc[t, "TWSO_obs_sd"],
    simulated=float(corrected_perdl_runs[t].loc[corrected_perdl_runs[t]["day"] == final_harvest_obs.loc[t, "day"], "TWSO"].iloc[0]))
    for t in TREATMENTS])
rho = case_points.observed.corr(case_points.simulated, method="spearman")
limits = np.r_[case_points.observed - case_points.sd, case_points.observed + case_points.sd,
               case_points.simulated] / 1000
limits = [float(limits.min() * 0.9), float(limits.max() * 1.1)]
fig = plt.figure(figsize=(8, 7))
grid = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 1], hspace=0.4, wspace=0.12)
ax = fig.add_subplot(grid[0, 0])
ax.plot(limits, limits, color="0.6", linestyle="--", linewidth=0.8)
for row in case_points.itertuples():
    ax.errorbar(row.observed / 1000, row.simulated / 1000, xerr=row.sd / 1000,
        fmt=TREATMENT_MARKER[row.treatment], color=TREATMENT_COLOR[row.treatment],
        markerfacecolor="white", capsize=2, markersize=5)
ax.set_xlim(*limits); ax.set_ylim(*limits); ax.set_aspect("equal")
ax.set_xlabel("Observed yield (Mg ha$^{-1}$)")
ax.set_ylabel("Simulated yield (Mg ha$^{-1}$)")
ax.set_title(f"Yield ranking: rho={rho:.2f}")
ax = fig.add_subplot(grid[0, 1:])
ax.set_axis_off()
ax.table(cellText=[[t, *[f"{case_metrics.loc[case_metrics.treatment.eq(t), name].iloc[0]:.3f}"
    for name in ("sm_NRMSE", "lai_NRMSE", "twso_NRMSE", "joint_nrmse")]] for t in TREATMENTS],
    colLabels=["Treatment", "SM nRMSE", "LAI nRMSE", "TWSO nRMSE", "Weighted"], loc="center")
for index, t in enumerate(TREATMENTS):
    row, col = divmod(index, 3)
    ax = fig.add_subplot(grid[row + 1, col])
    ax.plot(corrected_perdl_runs[t]["DAE"], corrected_perdl_runs[t]["SM_profile"], color="black", label="Simulated")
    ax.plot(obs_sm[t]["DAE"], obs_sm[t]["SM_obs"], "o", markerfacecolor="white",
            color="black", markersize=3, label="Observed")
    ax.set_title(t.replace("_", " ").replace(" Days", " d"))
    ax.set_ylim(0, 0.4)
    if col == 0:
        ax.set_ylabel("0-30 cm VWC")
    else:
        ax.tick_params(labelleft=False)
    if row == 1:
        ax.set_xlabel("Days after emergence")
    else:
        ax.tick_params(labelbottom=False)
fig.suptitle(f"FC + WP corrected / PERDL only: CFET={corrected_perdl_params['CFET']:.4f}, PERDL={corrected_perdl_params['PERDL']:.4f}")
plt.show()

# %% Factorial summary: eight cases / 48 treatment records, within-soil effects
factorial_rows, daily_parts, effect_rows = [], [], []
for soil_name in SOIL_ORDER:
    for case in CASE_ORDER:
        params = CASE_PARAMS[(soil_name, case)]
        assert params["DEPNR"] == 3.0
        for treatment in TREATMENTS:
            daily = CASE_RUNS[(soil_name, case)][treatment]
            day = final_harvest_obs.loc[treatment, "day"]
            matched_yield = daily.loc[daily["day"] == day, "TWSO"]
            assert len(matched_yield) == 1 and np.isfinite(matched_yield.iloc[0])
            factorial_rows.append(dict(soil=soil_name, case=case, treatment=treatment, day=day,
                CFET_level="fitted" if case in ("CFET only", "Both fitted") else "default",
                PERDL_level="fitted" if case in ("PERDL only", "Both fitted") else "default",
                **params, TWSO_obs=final_harvest_obs.loc[treatment, "TWSO_obs"],
                TWSO_obs_sd=final_harvest_obs.loc[treatment, "TWSO_obs_sd"],
                TWSO_sim=float(matched_yield.iloc[0]), **score(daily, treatment)))
            daily_parts.append(daily.assign(soil=soil_name, case=case, treatment=treatment, **params))
    for treatment in TREATMENTS:
        day = final_harvest_obs.loc[treatment, "day"]
        values = {case: float(CASE_RUNS[(soil_name, case)][treatment].loc[
            CASE_RUNS[(soil_name, case)][treatment]["day"] == day, "TWSO"].iloc[0]) for case in CASE_ORDER}
        cfet_effect = values["CFET only"] - values["Default"]
        perdl_effect = values["PERDL only"] - values["Default"]
        interaction = values["Both fitted"] - values["CFET only"] - values["PERDL only"] + values["Default"]
        total = values["Both fitted"] - values["Default"]
        assert np.isclose(cfet_effect + perdl_effect + interaction, total, rtol=0, atol=1e-8)
        effect_rows.append(dict(soil=soil_name, treatment=treatment, CFET_effect=cfet_effect,
                               PERDL_effect=perdl_effect, interaction=interaction, total_change=total))
df_factorial = pd.DataFrame(factorial_rows)
assert len(CASE_RUNS) == 8 and len(df_factorial) == 48
assert not df_factorial.duplicated(["soil", "case", "treatment"]).any()
assert df_factorial.groupby(["soil", "case"])[["CFET", "PERDL", "DEPNR"]].nunique().eq(1).all().all()
for soil_name in SOIL_ORDER:
    assert CASE_PARAMS[(soil_name, "CFET only")]["PERDL"] == BASE_DEFAULT["PERDL"]
    assert CASE_PARAMS[(soil_name, "PERDL only")]["CFET"] == BASE_DEFAULT["CFET"]
    assert CASE_PARAMS[(soil_name, "Both fitted")]["CFET"] == CASE_PARAMS[(soil_name, "CFET only")]["CFET"]
    assert CASE_PARAMS[(soil_name, "Both fitted")]["PERDL"] == CASE_PARAMS[(soil_name, "PERDL only")]["PERDL"]
df_effects = pd.DataFrame(effect_rows)
df_all_fits = pd.DataFrame([raw_cfet_record, raw_both_record, corrected_cfet_record, corrected_both_record])
assert len(df_all_fits) == 4
assert df_all_fits.groupby(["soil", "fit"]).size().eq(1).all()
df_factorial.to_csv(OUTPUT_DIR / f"factorial_metrics_{RUN_ID}.csv", index=False)
df_effects.to_csv(OUTPUT_DIR / f"factorial_yield_effects_{RUN_ID}.csv", index=False)
df_all_fits.to_csv(OUTPUT_DIR / f"opt_params_by_soil_{RUN_ID}.csv", index=False)
pd.concat(daily_parts, ignore_index=True).to_csv(OUTPUT_DIR / f"factorial_daily_{RUN_ID}.csv", index=False)
pd.concat([raw_cfet_trace, raw_both_trace, corrected_cfet_trace, corrected_both_trace], ignore_index=True).to_csv(
    OUTPUT_DIR / f"factorial_trace_{RUN_ID}.csv", index=False)
print("\nEight-case summary:")
print(df_factorial[["soil", "case", "treatment", "CFET", "PERDL", "joint_nrmse"]].round(4).to_string(index=False))
print("\nWithin-soil yield effects (Mg/ha):")
print(df_effects.set_index(["soil", "treatment"]).div(1000).round(3).to_string())
print("Cross-soil fitted comparisons include recalibration; fitted values are shared across treatments within each soil.")

# %% Publish FC+WP with fitted CFET and default PERDL, after all eight cases finish
with open(ROOT / SOIL_FILE, "w") as f:
    yaml.safe_dump(SOILS["FC + WP corrected"], f, sort_keys=False)
df_opt = pd.DataFrame([corrected_cfet_record])
assert len(df_opt) == 1 and df_opt["fit"].eq("CFET_fit").all()
assert df_opt["PERDL"].eq(BASE_DEFAULT["PERDL"]).all()
assert df_opt["soil"].eq("FC + WP corrected").all()
df_opt.to_csv(OUTPUT_DIR / f"opt_params_shared_{RUN_ID}.csv", index=False)
TABLE_DIR = ROOT / "manuscript_outputs" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)
df_opt.to_csv(TABLE_DIR / "opt_params_shared_IND.csv", index=False)
corrected_cfet_trace.assign(
    location=SITE, season=SEASON).to_csv(OUTPUT_DIR / f"opt_trace_{RUN_ID}.csv", index=False)
df_selected_metrics = df_factorial[(df_factorial["soil"] == "FC + WP corrected")
    & df_factorial["case"].isin(["Default", "CFET only"])].rename(columns={"case": "version"}).copy()
df_selected_metrics["version"] = df_selected_metrics["version"].map({
    "Default": "crop default", "CFET only": "CFET_fit"})
df_selected_metrics["calibration_role"] = df_selected_metrics["treatment"].map({
    CONTROL: "CFET calibration"}).fillna("unused in CFET fitting")
df_selected_metrics["n_shared_fitted"] = df_selected_metrics["version"].map({
    "crop default": 0, "CFET_fit": 1})
df_selected_metrics.assign(location=SITE, season=SEASON).to_csv(
    OUTPUT_DIR / f"opt_metrics_by_treatment_{RUN_ID}.csv", index=False)
runs_default = corrected_default_runs
runs_fitted = {("CFET_fit", t): corrected_cfet_runs[t] for t in TREATMENTS}
shared_params = {"CFET_fit": corrected_cfet_params}
selected_handoff = dict(soil="FC + WP corrected", soil_file=SOIL_FILE,
    WAV=SITE_BY_SOIL["FC + WP corrected"]["WAV"], SMLIM=SITE_BY_SOIL["FC + WP corrected"]["SMLIM"],
    initial_SM_L0=common_sm[0], initial_SM_L1=common_sm[1])

# %% Figure 7 source: final harvest under defaults and one shared cultivar pair
SOURCE_DIR = ROOT / "manuscript_outputs" / "source_data"
SOURCE_DIR.mkdir(parents=True, exist_ok=True)
final_params = shared_params["CFET_fit"]
assert set(final_params) == {"CFET", "DEPNR", "PERDL"}
assert final_params["DEPNR"] == 3.0
assert final_params["PERDL"] == BASE_DEFAULT["PERDL"]
pd.DataFrame([dict(location=SITE, season=SEASON, cultivar=CULTIVAR,
                   **final_params, **selected_handoff)]).to_csv(TABLE_DIR / "ind_cultivar_water_parameters.csv", index=False)

_yield = load_obs_yield(location=SITE)
_yield["season"] = _yield["season"].astype(str).map(canonicalize_season)
_yield["day"] = pd.to_datetime(_yield["day"])
_yield = _yield[(_yield["season"] == SEASON) & (_yield["day"] >= EMERGENCE)
                & (_yield["cultivar"] == CULTIVAR) & (_yield["Field"] == "OpenField")].dropna(subset=["TWSO_obs"])
assert not _yield.duplicated(["treatment", "day"]).any()
final_obs = _yield.sort_values("day").groupby("treatment", sort=False).tail(1)
assert set(final_obs["treatment"]) == set(TREATMENTS)
rank_rows = []
for version, runs, params in (
        ("layered, crop default", runs_default, BASE_DEFAULT),
        ("layered, CFET_fit", {t: runs_fitted[("CFET_fit", t)] for t in TREATMENTS}, final_params)):
    for t in TREATMENTS:
        obs = final_obs[final_obs["treatment"] == t].iloc[0]
        sim = runs[t].loc[runs[t]["day"] == obs["day"], "TWSO"]
        assert len(sim) == 1, f"Missing final-harvest simulation: {version}/{t}"
        rank_rows.append(dict(location=SITE, season=SEASON, cultivar=CULTIVAR,
                              version=version, treatment=t, day=obs["day"],
                              TWSO_obs=obs["TWSO_obs"], TWSO_obs_sd=obs["TWSO_obs_sd"],
                              TWSO_sim=float(sim.iloc[0]), **params, **selected_handoff))
df_rank = pd.DataFrame(rank_rows)
assert not df_rank.duplicated(["version", "treatment", "day"]).any()
df_rank.to_csv(OUTPUT_DIR / f"opt_yield_order_{RUN_ID}.csv", index=False)
print("Shared Kennebec water parameters:")
print(pd.DataFrame([final_params]).to_string(index=False))

# %% S12: raw versus FC+WP retention, both manual targets
assert LAYER_THICK == [30.0, 5.0], "Regenerate the two-layer IND soil with 05.0"
raw_layers = SOILS["PTF raw"]["SoilProfileDescription"]["SoilLayers"]
corrected_layers = SOILS["FC + WP corrected"]["SoilProfileDescription"]["SoilLayers"]
assert raw_layers[1] == corrected_layers[1], "The pan must retain its raw tables"
fc_table = corrected_layers[0]["SMfromPF"]
assert np.isclose(np.interp(2.0, fc_table[0::2], fc_table[1::2]), 0.30, atol=1e-6)
retention_rows = []
obs_range = pd.concat(list(obs_sm.values()), ignore_index=True)["SM_obs"]
for soil, sd in SOILS.items():
    top = 0.0
    for layer, lay in enumerate(sd["SoilProfileDescription"]["SoilLayers"]):
        bottom = top + float(lay["Thickness"])
        for pf, vwc in zip(lay["SMfromPF"][0::2], lay["SMfromPF"][1::2]):
            retention_rows.append(dict(site=SITE, season=SEASON, geometry="2layer", soil=soil,
                                       layer_top=top, layer_bottom=bottom, pF=pf, vwc=vwc,
                                       fc_target=0.30 if layer == 0 else np.nan,
                                       wp_target=0.10 if layer == 0 else np.nan, obs_lo=obs_range.min(), obs_hi=obs_range.max()))
        top = bottom
retention = pd.DataFrame(retention_rows)
retention.to_csv(SOURCE_DIR / "s12_ind_retention_curves.csv", index=False)

# %% S16/S17: isolate the soil correction with identical crop water defaults
# Corrected runs are reused from panel a. CFET/PERDL are unchanged between soils.
soil_runs = {"PTF raw": raw_default_runs, "FC + WP corrected": runs_default}
pair_parts, mae_rows, series_parts = [], [], []
for soil, runs in soil_runs.items():
    for t in TREATMENTS:
        daily = runs[t]
        pairs = obs_sm[t].merge(daily[["day", "SM_profile"]], on="day", how="inner",
                                validate="one_to_one").dropna()
        assert not pairs.empty, f"No matched VWC observations: {soil}/{t}"
        pair_parts.append(pairs.assign(site=SITE, season=SEASON, treatment=t, soil=soil))
        mae_rows.append(dict(site=SITE, season=SEASON, treatment=t, soil=soil, n=len(pairs),
                             **MAEDecomposition(pairs.SM_obs, pairs.SM_profile).as_dict()))
        series_parts.append(daily[["day", "DAE", "SM_profile", "SM_L1"]].melt(
            id_vars=["day", "DAE"], var_name="variable", value_name="value").assign(
                site=SITE, season=SEASON, treatment=t, soil=soil, series_kind="simulation"))
for t in TREATMENTS:
    series_parts.append(obs_sm[t].rename(columns={"SM_obs": "value"}).assign(
        site=SITE, season=SEASON, treatment=t, soil="observed", variable="SM_profile",
        series_kind="observation"))
sm_pairs = pd.concat(pair_parts, ignore_index=True)
sm_mae = pd.DataFrame(mae_rows)
series = pd.concat(series_parts, ignore_index=True)
a, b = [sm_pairs[sm_pairs.soil.eq(soil)].sort_values(["treatment", "day"])
        for soil in SOIL_ORDER]
assert np.array_equal(a[["treatment", "day"]].to_numpy(), b[["treatment", "day"]].to_numpy())
assert np.allclose(a.SM_obs.to_numpy(), b.SM_obs.to_numpy())
assert not sm_pairs.duplicated(["site", "season", "treatment", "soil", "day"]).any()
assert not series.duplicated(["site", "season", "treatment", "soil", "series_kind", "variable", "day"]).any()
assert np.allclose(sm_mae.MAE, sm_mae[["MAE_b", "MAE_p", "MAE_u"]].sum(axis=1))
sm_pairs.to_csv(SOURCE_DIR / "s16_s18_ind_sm_pairs.csv", index=False)
sm_mae.to_csv(SOURCE_DIR / "s18_ind_mae.csv", index=False)
series.to_csv(SOURCE_DIR / "s17_ind_timeseries.csv", index=False)
print("IND2024 VWC MAE, crop water defaults held fixed between soils:")
print(sm_mae.round(4).to_string(index=False))

# %% Retained S18: IND2024 crop trajectories with crop water defaults
# Preserve the original unfitted corrected-soil comparison; no IND2022 source.
crop_parts = []
_lai = load_obs_lai(location=SITE).rename(columns={"LAI": "LAI_obs"})
_lai["season"] = _lai["season"].astype(str).map(canonicalize_season)
_lai["day"] = pd.to_datetime(_lai["day"])
_lai = _lai[(_lai["season"] == SEASON) & (_lai["day"] >= EMERGENCE)
            & (_lai["cultivar"] == CULTIVAR) & (_lai["Field"] == "OpenField")]
for t in TREATMENTS:
    crop_parts.append(runs_default[t][["day", "DAE", "LAI", "TWSO"]].melt(
        id_vars=["day", "DAE"], var_name="variable", value_name="value").assign(
            site=SITE, season=SEASON, treatment=t, variant="committed",
            series_kind="simulation", uncertainty=np.nan))
    for variable, frame, obs_col, sd_col in (
            ("LAI", _lai, "LAI_obs", "LAI_sd"),
            ("TWSO", _yield, "TWSO_obs", "TWSO_obs_sd")):
        obs = frame[frame["treatment"] == t][["day", obs_col, sd_col]].copy()
        obs["DAE"] = (obs["day"] - EMERGENCE).dt.days
        crop_parts.append(obs.rename(columns={obs_col: "value", sd_col: "uncertainty"}).assign(
            site=SITE, season=SEASON, treatment=t, variable=variable,
            variant="observed", series_kind="observation"))
pd.concat(crop_parts, ignore_index=True).to_csv(
    SOURCE_DIR / "s19_ind_committed_timeseries.csv", index=False)
