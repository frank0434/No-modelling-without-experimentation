# %% Potential growth: partitioning + SPAN calibration and manuscript handoffs
# NLD2025 Innovator and IND2024 Kennebec only; exploratory calibration.
# Fixed phenology -> establishment/SLATB corrections -> joint partitioning/SPAN fit.
# GSA and SPAN mechanism diagnostics remain separate. Figure rendering is in 06.1/06.2.
# Method adapted from H. Berghuijs' WOFOST calibration tutorial.

import sys
import pathlib
import json
import hashlib
import pickle
from functools import partial
from importlib.metadata import version
import numpy as np
import pandas as pd

# -- path bootstrap: put scripts/ on sys.path (same as 03.3)
_here = pathlib.Path().resolve()
_root = next(p for p in (_here, *_here.parents)
             if (p / "scripts").exists() and (p / "data").exists())
sys.path.insert(0, str(_root / "scripts"))

from pcse.models import Wofost73_PP
from utils import BiomassPartitioning as biop
from utils.paths import ROOT
from utils.loaders import (
    load_crop_data, load_experiments, canonicalize_season,
    load_agro, load_weather, load_soil, load_site_data, build_parameters,
    load_obs_yield, load_obs_lai, _benchmark_wide, run_days,
)
from utils.phenology_helpers import load_phenology_params, PHENO_RUN_ID, obs_dates

from utils.metrics import rmse, mae
from optimization.config import OptimizationConfig, ParameterSpec
from optimization.resolver import ParameterSetResolver
from optimization.optimizer import TwoStageNLOptimizer

GROWTH_DIR = ROOT / "output" / "pp_growth"
GROWTH_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = "v1"
SLATB_RUN_ID = "v3"
PARTITION_RUN_ID = RUN_ID

# Same PP model + repo-local conf as 03.3, so WLV/WST are in the daily output.
class Wofost73_PP_SLA(Wofost73_PP):
    config = str(ROOT / "data" / "conf" / "Wofost73_PP_SLA.conf")
MODEL = Wofost73_PP_SLA

# obs->sim mapping and objective weights -- identical to 03.3's GrowthObjective.
VAR_SPECS = [
    ("TWSO", "TWSO_obs", "TWSO_obs_sd", 1.00),
    ("WST",  "WST_obs",  "WST_obs_sd",  0.25),
    ("WLV",  "WLV_obs",  "WLV_obs_sd",  0.50),
    ("LAI",  "LAI_obs",  "LAI_sd",      0.50),
]

# Calibration only: no independent-season growth evaluation in this workflow.
SITES = {
    "NLD": dict(location="NLD", field="OpenField", cultivar="Innovator",
                control="AC", good="2025", teffmx25=False, slatb=False,
                est_csv=None, est_cultivar=False),
    "IND": dict(location="IND", field="OpenField", cultivar="Kennebec",
                control="Full_0 Days", good="2024-2025", teffmx25=True, slatb=True,
                est_csv="IND_establishment_rate.csv", est_cultivar=False),
}
RETAINED = {("NLD", "2025"), ("IND", "2024-2025")}

# %%
# -- Persisted parameter layers (read once) -----------------------------------
_CROPD = load_crop_data("data/crops_calibration")

# Preserve the original index-2, rounded phenology and IND TEFFMX=25 policy.
PHENO_PARAMS = ["TBASEM", "TEFFMX", "TSUMEM", "TSUM1", "TSUM2"]
_pheno = load_phenology_params(field="OpenField", index=2)
_pheno = _pheno.loc[[_key in RETAINED for _key in zip(_pheno.site, _pheno.year.astype(str))]]
phenology_lookup = {
    (r["site"], r["cultivar"], str(r["year"]), r["field"]): {p: round(r[p]) for p in PHENO_PARAMS}
    for _, r in _pheno.iterrows()
}

# SLATB: IND uses its anchored curve; NLD keeps the crop default.
_slatb_table = pd.read_csv(GROWTH_DIR / f"pp_growth_params_{SLATB_RUN_ID}.csv")


def slatb_for(site, cultivar):
    rows = _slatb_table[(_slatb_table["site"] == site)
                        & (_slatb_table["cultivar"] == cultivar)
                        & (_slatb_table["param"] == "SLATB")].sort_values("dvs")
    return [v for dvs, val in zip(rows["dvs"], rows["value"]) for v in (dvs, val)]


def partition_for(site, cultivar):
    """(named scalars dict, override dict with FLTB/FOTB/FSTB + SPAN).

    RGRLAI is not calibrated (dropped from 03.3's ACTIVE_PARAMS) -- left out of
    the override so it stays at the crop-file default for every site.
    """
    rows = _part[(_part["site"] == site) & (_part["cultivar"] == cultivar)]
    named = dict(zip(rows["param"], rows["value"]))
    FLTBpar = {k: named[k] for k in ("FLTBa", "FLTBb", "FLTBc", "FLTBd")}
    FOTBpar = {k: named[k] for k in ("FOTBa", "FOTBb")}
    FOTB, FLTB, FSTB = biop.compute_partition_functions(
        FOTBpar=FOTBpar, FLTBpar=FLTBpar, warn=False)
    over = {"FOTB": FOTB.tolist(), "FLTB": FLTB.tolist(), "FSTB": FSTB.tolist(),
            "SPAN": named["SPAN"]}
    return named, over


def est_rate(site):
    """Season -> mean observed establishment rate (for corrected TDWI). {} if none."""
    cfg = SITES[site]
    if not cfg["est_csv"]:
        return {}
    df = pd.read_csv(ROOT / "data" / "obs" / cfg["est_csv"])
    if cfg["est_cultivar"]:
        df = df[df["cultivar"] == cfg["cultivar"]]
    if "field" in df.columns:
        df = df[df["field"] == "OpenField"]
    return {str(k): v for k, v
            in df.groupby("season")["Establishment_obs"].mean().to_dict().items()}


# %%
# -- Experiments, run helper, observations (generic over site) ----------------
_EXP = {s: {canonicalize_season(e["season"]): e
            for e in load_experiments(location=SITES[s]["location"])
            if e["cultivar"] == SITES[s]["cultivar"]
            and canonicalize_season(e["season"]) == SITES[s]["good"]}
        for s in SITES}


def run_model(site, season, overrides):
    """Run the retained calibration setup with the given overrides; daily df."""
    cfg = SITES[site]
    exp = _EXP[site][season]
    params = build_parameters(_CROPD, load_soil(exp["soil_file"]),
                              load_site_data(exp["WAV"], exp["CO2"]), cfg["cultivar"], overrides)
    model = MODEL(params, load_weather(exp["weather_file"]),
                  load_agro(cfg["location"], exp["season"], cfg["cultivar"]))
    rd = run_days(cfg["location"], season)
    model.run(rd) if rd else model.run_till_terminate()
    df = pd.DataFrame(model.get_output())
    df["day"] = pd.to_datetime(df["day"])
    return df


def load_obs(site, season, field="OpenField", control=None):
    """Observed organ series for one setup/season, control treatment (as 03.3)."""
    cfg = SITES[site]
    twso = load_obs_yield(location=cfg["location"])
    lai = load_obs_lai(location=cfg["location"]).rename(columns={"LAI": "LAI_obs"})
    organs = _benchmark_wide(["WLV", "WST"]).rename(columns={
        "mean__WLV": "WLV_obs", "sd__WLV": "WLV_obs_sd",
        "mean__WST": "WST_obs", "sd__WST": "WST_obs_sd"})

    if control is None:
        control = cfg.get("control_by_season", {}).get(str(season), cfg["control"])

    def _filt(df):
        return df[(df["season"].astype(str) == season)
                  & (df["location"] == site)
                  & (df["cultivar"] == cfg["cultivar"])
                  & (df["Field"] == field)
                  & (df["treatment"] == control)].copy()

    src = {"TWSO": _filt(twso), "LAI": _filt(lai),
           "WLV": _filt(organs), "WST": _filt(organs)}
    out = {}
    for sim_col, obs_col, sd_col, _ in VAR_SPECS:
        d = src[sim_col][["day", obs_col, sd_col]].dropna(subset=[obs_col])
        d[sd_col] = d[sd_col].fillna(0.0)
        d["day"] = pd.to_datetime(d["day"])
        out[sim_col] = d
    return out


def phenology_overrides(site, season):
    """Best-fit phenology (index==2), with TEFFMX pinned to 25 for IND."""
    cfg = SITES[site]
    over = dict(phenology_lookup[(cfg["location"], cfg["cultivar"], season, cfg["field"])])
    if cfg["teffmx25"]:
        over["TEFFMX"] = 25
    return over



# %% Seven fitted parameters; preserve bounds, initial values and steps
ACTIVE_PARAMS = (
    ParameterSpec("FLTBa", lower=5.0,  upper=25.0, init=20.0,  step=2.0),
    ParameterSpec("FLTBb", lower=0.8,  upper=1.3,  init=1.05,  step=0.105),
    ParameterSpec("FLTBc", lower=0.7,  upper=0.95, init=0.83,  step=0.083),
    ParameterSpec("FLTBd", lower=0.0,  upper=0.1,  init=0.0,   step=0.01),
    ParameterSpec("FOTBa", lower=1.0,  upper=25.0, init=7.0,   step=1.0),
    ParameterSpec("FOTBb", lower=0.8,  upper=1.3,  init=1.05,  step=0.105),
    ParameterSpec("SPAN",  lower=20.0, upper=50.0, init=37.0,  step=5.0),
)
# NOTE (identifiability): SPAN also drives end-season leaf-age death and the SLA
# jump handled in 03.1, so it is partly confounded with the fixed SLATB.


# %% Existing weighted NRMSE objective and two-stage optimiser interface
class GrowthObjective:
    def __init__(self, season, fixed_overrides, resolver, obs_by_var,
                 var_specs, runner, cache=True):
        self.season = season
        self.fixed_overrides = fixed_overrides
        self.resolver = resolver
        self.obs_by_var = obs_by_var
        self.var_specs = var_specs
        self.cache = cache
        self.runner = runner   # (season, overrides) -> daily df
        self._cache = {}
        self.trace = []

    def overrides_from_named(self, named: dict) -> dict:
        """Partitioning tables with fitted SPAN, or fixed SPAN for a trial."""
        named = {**self.fixed_overrides, **named}
        FLTBpar = dict(FLTBa=named["FLTBa"], FLTBb=named["FLTBb"],
                       FLTBc=named["FLTBc"], FLTBd=named["FLTBd"])
        FOTBpar = dict(FOTBa=named["FOTBa"], FOTBb=named["FOTBb"])
        FOTB, FLTB, FSTB = biop.compute_partition_functions(
            FOTBpar=FOTBpar, FLTBpar=FLTBpar, warn=False)
        return {**self.fixed_overrides,
                "FOTB": FOTB.tolist(), "FLTB": FLTB.tolist(), "FSTB": FSTB.tolist(),
                "SPAN": named["SPAN"] }

    def _score(self, df_sim: pd.DataFrame):
        weighted, sse_tot, n_tot = 0.0, 0.0, 0
        for sim_col, obs_col, _sd_col, weight in self.var_specs:
            obs = self.obs_by_var[sim_col]
            if obs.empty:
                continue
            m = pd.merge(obs, df_sim[["day", sim_col]], on="day", how="inner").dropna()
            if m.empty:
                continue
            mean_obs = m[obs_col].mean()
            resid = (m[sim_col] - m[obs_col]) / mean_obs
            weighted += float(np.sqrt(np.mean(resid ** 2))) * weight
            sse_tot += float(np.sum((m[sim_col] - m[obs_col]) ** 2))
            n_tot += len(m)
        return weighted, sse_tot, n_tot

    def __call__(self, x, grad=None):
        x = np.asarray(x, dtype=float)
        key = tuple(x.tolist())
        if self.cache and key in self._cache:
            return self._cache[key]
        named = self.resolver.vector_to_overrides(x)
        try:
            df = self.runner(self.season, self.overrides_from_named(named))
            wnrmse, sse, n = self._score(df)
            if n == 0:
                wnrmse = 1e9
        except Exception:
            wnrmse, sse, n = 1e9, 0.0, 0
        self.trace.append({"x": x.copy(), "overrides": named,
                           "rmse": wnrmse, "sse": sse, "n": n})
        if self.cache:
            self._cache[key] = wnrmse
        return wnrmse

    def best(self):
        return min(self.trace, key=lambda r: r["rmse"]) if self.trace else None


# %% Cache identity: current model, configuration and simulation inputs
def _opt_cache_key(cfg, active_params, fixed, var_specs, obs, input_fingerprints) -> str:
    payload = {
        "inputs": input_fingerprints,
        "pcse_version": version("pcse"),
        "nlopt_version": version("nlopt"),
        "target": cfg.target, "seasons": list(cfg.seasons),
        "params": [(s.name, s.lower, s.upper, s.init, s.step) for s in active_params],
        "max1": cfg.max_evals_stage1, "max2": cfg.max_evals_stage2,
        "ftol": cfg.ftol_rel, "seed": cfg.seed,
        "fixed": fixed,
        "weights": [(v[0], v[3]) for v in var_specs],
        "obs": {k: df.to_json(orient="records") for k, df in obs.items()},
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


CACHE_DIR = GROWTH_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
# File content fingerprints also invalidate fits after upstream input changes.
common_inputs = [
    ROOT / "data/conf/Wofost73_PP_SLA.conf",
    ROOT / "data/params/experiments.yaml",
    ROOT / "data/obs/MainTable.xlsx",
    ROOT / "data/obs/emergence_plot_data.csv",
    ROOT / "output/obs_benchmark_summary.csv",
    GROWTH_DIR / "pp_growth_params_v3.csv",
    ROOT / "output/phenology" / f"phenology_params_{PHENO_RUN_ID}.csv",
    ROOT / "scripts/utils/loaders.py",
    ROOT / "scripts/utils/phenology_helpers.py",
    ROOT / "scripts/utils/BiomassPartitioning.py",
    ROOT / "scripts/optimization/optimizer.py",
    ROOT / "scripts/03.3_p_growth_partition.py",
    *sorted((ROOT / "data/crops_calibration").glob("*.yaml")),
]

# %% Optimise the two retained calibration datasets and export parameters/traces
parameter_parts, trace_by_site = [], {}
overrides_by_site = {}
fixed_by_site, observations, defaults_by_site = {}, {}, {}
for site, cfg in SITES.items():
    season, cultivar = cfg["good"], cfg["cultivar"]
    exp = _EXP[site][season]
    fixed = phenology_overrides(site, season)
    rates = est_rate(site)
    fixed["TDWI"] = exp["TDWI"] * rates.get(season, 1.0)
    if cfg["slatb"]:
        fixed["SLATB"] = slatb_for(site, cultivar)
    obs = load_obs(site, season)
    assert all(not obs[v].empty for v, *_ in VAR_SPECS)
    fixed_by_site[site], observations[site] = fixed, obs
    default_parameters = build_parameters(
        _CROPD, load_soil(exp["soil_file"]),
        load_site_data(exp["WAV"], exp["CO2"]), cultivar, {})
    # The crop provider changes its active cultivar on every build; snapshot defaults now.
    defaults_by_site[site] = {
        name: default_parameters[name]
        for name in [*PHENO_PARAMS, "SLATB", "SPAN", "RGRLAI", "FLTB", "FOTB", "FSTB"]
    }
    config = OptimizationConfig(
        target=f"{site}-{cultivar}", location=site, cultivar=cultivar,
        seasons=(season,), active_params=ACTIVE_PARAMS,
        crop_dir="data/crops_calibration", run_id=RUN_ID,
        max_evals_stage1=800, max_evals_stage2=300, ftol_rel=1e-4, seed=19730703,
    )
    resolver = ParameterSetResolver(config)
    objective = GrowthObjective(season, fixed, resolver, obs, VAR_SPECS,
                                runner=partial(run_model, site))
    pre_value = objective(resolver.init_vector())
    print(f"{site}: weighted NRMSE at start = {pre_value:.4f}")
    input_files = [*common_inputs, ROOT / exp["weather_file"], ROOT / exp["soil_file"],
                   ROOT / "data/agro" / {"IND": "india_seasons.yaml", "NLD": "netherlands_seasons.yaml"}[site]]
    if cfg["est_csv"]:
        input_files.append(ROOT / "data/obs" / cfg["est_csv"])
    fingerprints = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in input_files}
    cache_file = CACHE_DIR / f"opt_{RUN_ID}_{_opt_cache_key(config, ACTIVE_PARAMS, fixed, VAR_SPECS, obs, fingerprints)}.pkl"
    if cache_file.exists():
        with cache_file.open("rb") as f:
            cached = pickle.load(f)
        r1, r2 = cached["r1"], cached["r2"]
        objective.trace = cached["trace"]
        print(f"loaded {cache_file.name}")
    else:
        r1, r2 = TwoStageNLOptimizer(config, resolver, objective).optimize()
        with cache_file.open("wb") as f:
            pickle.dump({"r1": r1, "r2": r2, "trace": objective.trace}, f)
    print(f"{site}: global best={r1.f_best:.4f} ({r1.n_evals} evals); "
          f"local best={r2.f_best:.4f} ({r2.n_evals} evals)")
    named = resolver.vector_to_overrides(np.asarray(r2.x_best, dtype=float))
    overrides_by_site[site] = objective.overrides_from_named(named)
    df_params = pd.DataFrame([
        {"run_id": RUN_ID, "site": site, "cultivar": cultivar, "param": name,
         "value": named[name], "lower": spec.lower, "upper": spec.upper, "init": spec.init}
        for name, spec in zip(resolver.names, ACTIVE_PARAMS)
    ])
    assert set(df_params.param) == set(resolver.names)
    df_params.to_csv(GROWTH_DIR / f"pp_partition_params_{site}_{RUN_ID}.csv", index=False)
    parameter_parts.append(df_params)
    trace_rows = []
    for i, rec in enumerate(objective.trace):
        trace_rows.append({"eval": i, "rmse": rec["rmse"], "n": rec["n"], **rec["overrides"]})
    df_trace = pd.DataFrame(trace_rows)
    df_trace.to_csv(GROWTH_DIR / f"pp_partition_trace_{site}_{RUN_ID}.csv", index=False)
    trace_by_site[site] = df_trace
    print(df_params.to_string(index=False))
_part = pd.concat(parameter_parts, ignore_index=True)

# %% Baseline and optimised calibration simulations; reuse in every handoff
simulations, metric_parts = {}, []
for site, cfg in SITES.items():
    season = cfg["good"]
    simulations[site] = {
        "default": run_model(site, season, fixed_by_site[site]),
        "calibrated": run_model(site, season, overrides_by_site[site]),
    }
    metric_rows = []
    for variant, daily in simulations[site].items():
        for sim_col, obs_col, _sd_col, _weight in VAR_SPECS:
            paired = observations[site][sim_col][["day", obs_col]].merge(
                daily[["day", sim_col]], on="day", how="inner").dropna()
            assert not paired.empty
            metric_rows.append({
                "season": season, "stage": "cal", "variant": variant,
                "variable": sim_col, "n": len(paired),
                "rmse": rmse(paired[obs_col], paired[sim_col]),
                "mae": mae(paired[obs_col], paired[sim_col]),
                "bias": float((paired[sim_col] - paired[obs_col]).mean()),
            })
    df_metrics = pd.DataFrame(metric_rows)
    df_metrics.to_csv(GROWTH_DIR / f"pp_partition_metrics_{site}_{RUN_ID}.csv", index=False)
    metric_parts.append(df_metrics.assign(site=site))
    print(df_metrics.to_string(index=False))

# %% Downstream growth parameter handoff and supplementary Table S2
SUMMARY_SEASONS = {site: cfg["good"] for site, cfg in SITES.items()}
def summary_rows(site, season):
    cfg = SITES[site]
    exp = _EXP[site][season]
    rows = []

    def add(layer, param, value, dvs="", source=""):
        rows.append({"site": site, "cultivar": cfg["cultivar"], "season": season,
                     "layer": layer, "param": param, "dvs": dvs,
                     "value": value, "source": source})

    # 1. phenology
    for p, v in phenology_overrides(site, season).items():
        src = ("TEFFMX pinned to 25 (committed)"
               if (p == "TEFFMX" and cfg["teffmx25"]) else f"{PHENO_RUN_ID} index==2")
        add("phenology", p, v, source=src)

    # 2. corrected TDWI
    rates = est_rate(site)
    if str(season) in rates:
        add("TDWI", "TDWI", exp["TDWI"] * rates[str(season)],
            source="seed weight x mean establishment rate")
    else:
        add("TDWI", "TDWI", exp["TDWI"], source="seed weight (no correction)")

    # 3. SLATB
    if cfg["slatb"]:
        note = "anchored to peak obs SLA (03.1)"
        slat = slatb_for(site, cfg["cultivar"])
        for dvs, val in zip(slat[0::2], slat[1::2]):
            add("SLATB", "SLATB", val, dvs=dvs, source=note)
    else:
        add("SLATB", "SLATB", "crop default", source="no correction (crop default)")

    # 4. partitioning + SPAN (RGRLAI not calibrated)
    named, _ = partition_for(site, cfg["cultivar"])
    ptag = f"pp_partition_params_{site}_{PARTITION_RUN_ID}.csv"
    for p in ("FLTBa", "FLTBb", "FLTBc", "FLTBd", "FOTBa", "FOTBb", "SPAN"):
        add("partitioning", p, named[p], source=ptag)
    # RGRLAI is not calibrated anywhere; write the numeric crop-file default (not
    # a string) so 04.x/05.x/06_overview.py, which all read this CSV and do
    # float(_named["RGRLAI"]), keep working -- a no-op override either way.
    crop_default = defaults_by_site[site]
    add("partitioning", "RGRLAI", float(crop_default["RGRLAI"]), source="crop default, not calibrated")

    return rows


summary = pd.concat([pd.DataFrame(summary_rows(s, SUMMARY_SEASONS[s])) for s in SUMMARY_SEASONS],
                    ignore_index=True)
summary = summary[["site", "cultivar", "season", "layer", "param", "dvs", "value", "source"]]
summary.to_csv(GROWTH_DIR / f"growth_best_params_{RUN_ID}.csv", index=False)
print(f"best-parameter handoff -> {GROWTH_DIR / f'growth_best_params_{RUN_ID}.csv'}")
print(summary.to_string(index=False))

# Table S2 source: crop-file defaults against the complete IND/NLD handoff.
# RGRLAI is intentionally absent because it was dropped from calibration.
_manuscript_source = ROOT / "manuscript_outputs" / "source_data"
_manuscript_source.mkdir(parents=True, exist_ok=True)
_table_s2_rows = []
_units = {
    "TBASEM": "degC", "TEFFMX": "degC", "TSUMEM": "degC d",
    "TSUM1": "degC d", "TSUM2": "degC d", "TDWI": "kg ha-1",
    "SLATB": "ha kg-1", "SPAN": "d",
}
for _site in ("IND", "NLD"):
    _cfg, _season = SITES[_site], SUMMARY_SEASONS[_site]
    _exp = _EXP[_site][_season]
    _defaults = defaults_by_site[_site]
    _cal = summary[summary["site"] == _site]

    for _param in PHENO_PARAMS:
        _row = _cal[(_cal["layer"] == "phenology") & (_cal["param"] == _param)].iloc[0]
        _table_s2_rows.append({
            "site": _site, "cultivar": _cfg["cultivar"], "season": _season,
            "layer": "phenology", "param": _param, "dvs": np.nan,
            "default": float(_defaults[_param]), "calibrated": float(_row["value"]),
            "unit": _units[_param], "source": _row["source"],
        })

    _tdwi = _cal[(_cal["layer"] == "TDWI") & (_cal["param"] == "TDWI")].iloc[0]
    _table_s2_rows.append({
        "site": _site, "cultivar": _cfg["cultivar"], "season": _season,
        "layer": "TDWI", "param": "TDWI", "dvs": np.nan,
        "default": float(_exp["TDWI"]), "calibrated": float(_tdwi["value"]),
        "unit": _units["TDWI"], "source": _tdwi["source"],
    })

    _default_slatb = list(_defaults["SLATB"])
    _calibrated_slatb = (slatb_for(_site, _cfg["cultivar"])
                         if _cfg["slatb"] else _default_slatb)
    for _dvs, _default_value in zip(_default_slatb[0::2], _default_slatb[1::2]):
        _cal_map = dict(zip(_calibrated_slatb[0::2], _calibrated_slatb[1::2]))
        _table_s2_rows.append({
            "site": _site, "cultivar": _cfg["cultivar"], "season": _season,
            "layer": "SLATB", "param": "SLATB", "dvs": float(_dvs),
            "default": float(_default_value), "calibrated": float(_cal_map[_dvs]),
            "unit": _units["SLATB"],
            "source": "anchored to peak observed SLA" if _cfg["slatb"] else "crop default",
        })

    _part_rows = _part[(_part["site"] == _site) & (_part["cultivar"] == _cfg["cultivar"])]
    _partition_shape_params = {"FLTBa", "FLTBb", "FLTBc", "FLTBd", "FOTBa", "FOTBb"}
    for _row in _part_rows[_part_rows["param"] != "RGRLAI"].itertuples():
        _table_s2_rows.append({
            "site": _site, "cultivar": _cfg["cultivar"], "season": _season,
            "layer": "partitioning", "param": _row.param, "dvs": np.nan,
            "default": (np.nan if _row.param in _partition_shape_params
                        else float(_defaults[_row.param])),
            "calibrated": float(_row.value),
            "unit": _units.get(_row.param, "model parameter"),
            "source": ("crop baseline is an AFGEN table; calibrated logistic scalar"
                       if _row.param in _partition_shape_params
                       else f"crop default; pp_partition_params_{PARTITION_RUN_ID}"),
        })

pd.DataFrame(_table_s2_rows).to_csv(
    _manuscript_source / "table_s02_parameters.csv", index=False)


# %% Fitted scalar comparison: effective SPAN, not measured leaf lifespan
cal_season = SUMMARY_SEASONS
_df_span = _part[["site", "param", "value"]].copy()
_df_span["site_season"] = _df_span.site.map(lambda site: f"{site} ({cal_season[site]})")
table_span = _df_span.pivot(index="param", columns="site_season", values="value")
table_span.to_csv(GROWTH_DIR / f"pp_partition_calibrated_params_{RUN_ID}.csv")
print(table_span.to_string())

# %% Figure 5 trajectories: DAE from the current shared observed emergence
PANEL_ORDER = ["LAI", "WLV", "WST", "TWSO"]
SD_COL = {v: sd for v, _obs, sd, _weight in VAR_SPECS}
growth_parts = []
for site, cfg in SITES.items():
    season = cfg["good"]
    emergence = obs_dates(site, season, cfg["cultivar"])["Emergence"]
    for variant, daily in simulations[site].items():
        daily = daily[daily.day >= emergence].copy()
        daily["DAE"] = (daily.day - emergence).dt.days
        long = daily.melt(id_vars=["DAE", "DVS"], value_vars=PANEL_ORDER,
                          var_name="variable", value_name="value")
        long = long.assign(site=site, season=season, cultivar=cfg["cultivar"],
                           variant=variant, series_kind="simulation", uncertainty=np.nan)
        growth_parts.append(long)
    for variable, obs_col, sd_col, _weight in VAR_SPECS:
        obs = observations[site][variable]
        obs = obs[obs.day >= emergence]
        # Assign modelled DVS at each observation date from the calibrated run.
        obs = obs.merge(simulations[site]["calibrated"][["day", "DVS"]],
                        on="day", how="left", validate="one_to_one")
        assert obs.DVS.notna().all()
        growth_parts.append(pd.DataFrame({
            "DAE": (obs.day - emergence).dt.days, "DVS": obs.DVS,
            "variable": variable,
            "value": obs[obs_col], "site": site, "variant": "observed",
            "series_kind": "observation", "uncertainty": obs[sd_col],
            "season": season, "cultivar": cfg["cultivar"],
        }))
growth = pd.concat(growth_parts, ignore_index=True)
growth = growth[["DAE", "DVS", "variable", "value", "site", "variant", "series_kind",
                 "uncertainty", "season", "cultivar"]]
assert set(zip(growth.site, growth.season)) == RETAINED
assert not growth.duplicated(["site", "season", "variant", "variable", "series_kind", "DAE"]).any()
growth.to_csv(_manuscript_source / "fig05_growth_timeseries.csv", index=False)
# Preserve the former CSV interface for existing consumers, with retained rows only.
growth_all = growth.assign(field="OpenField", treatment="")
for site, cfg in SITES.items():
    growth_all.loc[growth_all.site.eq(site) & growth_all.variant.eq("observed"), "treatment"] = cfg["control"]
growth_all = growth_all[["DAE", "variable", "value", "variant", "site", "season", "field",
                         "cultivar", "treatment", "series_kind", "uncertainty"]]
growth_all.to_csv(_manuscript_source / "fig05_growth_all_seasons.csv", index=False)

# %% Actual AFGEN allocation tables: main Figure 5 and supplementary S10
partition_rows = []
for site, cfg in SITES.items():
    for variant, tables in [("default", defaults_by_site[site]),
                            ("calibrated", overrides_by_site[site])]:
        for table in ["FLTB", "FOTB", "FSTB"]:
            values = np.asarray(tables[table], dtype=float)
            partition_rows.extend({
                "site": site, "season": cfg["good"], "table": table, "variant": variant,
                "dvs": float(dvs), "value": float(value),
            } for dvs, value in zip(values[0::2], values[1::2]))
    fitted = overrides_by_site[site]
    fractions = np.array([fitted[table][1::2] for table in ["FLTB", "FOTB", "FSTB"]])
    assert (fractions >= -1e-12).all() and (fractions <= 1 + 1e-12).all()
    assert np.allclose(fractions.sum(axis=0), 1)
partition_curves = pd.DataFrame(partition_rows)
assert not partition_curves.duplicated(["site", "season", "table", "variant", "dvs"]).any()
partition_curves.to_csv(_manuscript_source / "s10_partition_curves.csv", index=False)
print("Figure 5/S10 sources and growth handoff written; render separately in 06.1/06.2.")
