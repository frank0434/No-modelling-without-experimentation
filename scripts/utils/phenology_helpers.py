"""
utils/phenology_helpers.py
--------------------------
Phenology helpers that were copied verbatim across the 02.x_pp_phenology* scripts,
collected here so each function is defined once. Two families were duplicated:

Cluster A -- emergence/thermal helpers keyed (location, season, cultivar):
    tavg_window, thermal_afgen, thermal_emergence, obs_dates, site_dtsmtb, crop_default
    (02.6_pp_phenology_thermal.py, 02.4_pp_teffmx_identifiability_check.py,
     02.5_pp_emergence_response_shapes.py)

Cluster B -- broken-stick / Seijo / DVS-mapping helpers keyed (season, ..., location, cultivar):
    obs_row, tavg_window_cb, inc_broken_stick, seijo_gdd, integrate_window, tt_to_dvs,
    dtsmtb_for_base, seijo_maturity_date, rederive_tsum
    (02.4_pp_phenology_gsa.py; earlier photoperiod/structure checks are archived)

Cluster A's tavg_window(location, season, cultivar, start, end) and Cluster B's
tavg_window(season, start, end, location, cultivar) return the same daily-mean series but take
their arguments in a different order. Both orders are kept -- Cluster B's as tavg_window_cb --
so no call site has to change argument order. Cluster B scripts import it as
`tavg_window_cb as tavg_window`.
"""

import numpy as np
import pandas as pd

from utils.loaders import (
    ROOT, load_experiments, find_experiment, load_weather, load_soil,
    load_site_data, load_crop_data, build_parameters, load_obs_experiments,
    load_obs_yield,
)

CROP_DIR = "data/crops_calibration"
cropd = load_crop_data(CROP_DIR)

# Seijo-Rodriguez et al. 2018 base-2/cap-26 1700-GDD anchor.
SEIJO_TOTAL = 1700.0
SEIJO_TBASE, SEIJO_TUPPER = 2.0, 26.0
IND_CULT = "Kennebec"

# RSA calibration season: 2024 is the only RSA season with intermediate harvests, so it is the only
# one that can date tuber initiation. Every 02.x script calibrates RSA on it.
RSA_TI_SEASON = "2024"


def rsa_tuber_init():
    """RSA tuber-init date, from the 2024 open-field Mondial TWSO series.

    RSA recorded no tuber-init date. The earliest 2024 sampling carrying tubers (36 DAP, 3-9 kg/ha
    TWSO) puts initiation at or before it, so this date stands in for the observation and is an
    upper bound. 2025-A and 2025-S have a final harvest only. Open field: the rain shelter was
    sampled at final harvest alone, so both RSA setups take this date.
    """
    y = load_obs_yield("RSA")
    return y[(y["season"].astype(str) == RSA_TI_SEASON) & (y["Field"] == "OpenField")
             & (y["cultivar"] == "Mondial") & y["TWSO_obs"].notna()]["day"].min()


# -- Committed phenology parameters (02.6's output, read by the growth and water steps) ----------

# Names output/phenology/phenology_params_*.csv and phenology_metrics_*.csv, plus the optimisation
# cache under output/phenology/optimization/. Written by 02.6_pp_phenology_thermal.py and read by
# every 03.x/04.x/05.x/06 script: one constant so the producer and its consumers cannot drift.
PHENO_RUN_ID = "pheno_thermal_v1"


def load_phenology_params(field, run_id=PHENO_RUN_ID, index=2):
    """Rows of output/phenology/phenology_params_{run_id}.csv for one setup.

    RSA has two setups per season, so (site, cultivar, year) alone is no longer a unique key --
    without the field filter a lookup would return whichever of the two rows comes first. field is
    required for that reason. Files written before the rain shelter was added carry no 'field'
    column; those are all open field.
    """
    df = pd.read_csv(ROOT / "output" / "phenology" / f"phenology_params_{run_id}.csv")
    if "field" not in df.columns:
        df["field"] = "OpenField"
    return df[(df["index"] == index) & (df["field"] == field)].copy()


# -- Cluster A: emergence/thermal helpers, keyed (location, season, cultivar) --------------------

def tavg_window(location, season, cultivar, start, end):
    """Daily mean temperature (TMIN+TMAX)/2 over [start, end) as a list."""
    exp = find_experiment(load_experiments(location), location, season, cultivar)
    wdp = load_weather(exp["weather_file"])
    days = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
    return [(wdp(d.date()).TMIN + wdp(d.date()).TMAX) / 2.0 for d in days]


def thermal_afgen(temps, dtsmtb):
    """Sum daily effective temperature from the DTSMTB lookup (linear interpolation)."""
    xt, yt = dtsmtb[0::2], dtsmtb[1::2]
    return float(np.sum(np.interp(temps, xt, yt)))


def thermal_emergence(temps, tbasem, teffmx):
    """Sum daily effective temperature for the pre-emergence phase: clip(T-TBASEM, 0,
    TEFFMX-TBASEM), per pcse's WOFOST phenology 'emerging' stage (STAGE == "emerging" in
    pcse/crop/phenology.py). DTSMTB does not apply here -- it is only used post-emergence
    ('vegetative'/'reproductive' stages, i.e. TSUM1/TSUM2)."""
    return float(np.sum(np.clip(np.asarray(temps) - tbasem, 0.0, teffmx - tbasem)))


def obs_dates(location, season, cultivar, field="OpenField"):
    """Raw observed phenology row (Start, Emergence, TuberInitiation, Maturity).

    RSA has two setups per season, 'OpenField' and 'RainShelter'; IND and NLD have
    OpenField only. The setup is part of the key because the shelter has its own
    emergence date (2025-A emerges 5 days earlier than the open field).
    """
    df = load_obs_experiments()
    r = df[(df["location"] == location) & (df["field"] == field)
           & (df["season_key"] == season) & (df["cultivar"] == cultivar)]
    if r.empty:
        raise ValueError(f"No {field} experiment row for {location}/{season}/{cultivar}")
    return r.iloc[0]


def emergence_date(location, season, cultivar, field="OpenField"):
    """Observed emergence date for one experiment -- the day-zero anchor for a
    days-after-emergence x-axis, the same date obs_dates()["Emergence"] feeds into
    the TSUM1/TSUM2 windows above."""
    return obs_dates(location, season, cultivar, field)["Emergence"]


def site_dtsmtb(location, season, cultivar):
    """DTSMTB lookup (base-2 effective-temperature table) for integrating observed thermal windows."""
    exp = find_experiment(load_experiments(location), location, season, cultivar)
    soil = load_soil(exp["soil_file"])
    return list(build_parameters(cropd, soil, load_site_data(10.0, exp["CO2"]), cultivar, {})["DTSMTB"])


def crop_default(location, season, cultivar, param_name):
    """Crop-file default for `param_name` for one experiment (build only, no model run).

    A completed WOFOST run resets the ParameterProvider's overrides back to the crop-file defaults
    once the crop reaches maturity (a PCSE quirk, harmless to the simulation itself but unreliable
    to read back afterwards) -- reading straight after build_parameters avoids it.
    """
    exp = find_experiment(load_experiments(location), location, season, cultivar)
    soild = load_soil(exp["soil_file"])
    sited = load_site_data(exp["WAV"], exp["CO2"])
    return float(build_parameters(cropd, soild, sited, cultivar, {})[param_name])


def derive_obs_fixed(site, cultivar, seasons, teffmx=None, field="OpenField"):
    """TSUMEM/TSUM1/TSUM2 integrated from observed inter-stage windows, averaged over `seasons`.

    One rule at every site, each quantity a DTSMTB sum over a [start, end) window:
        TSUMEM = Start -> Emergence          (pre-emergence clip formula, thermal_emergence)
        TSUM1  = Emergence -> TuberInit
        TSUM2  = total - TSUM1
    TSUM2 by subtraction equals the TuberInit -> Maturity integral exactly, the windows being
    contiguous, and stays defined where the tuber-init date is missing.

    Only the window boundaries differ by site:
        total       -- observed Emergence -> Maturity at NLD and RSA, whose Maturity is observed
                       senescence (DVS=2). IND's Maturity is a haulm kill, so its total is the
                       Seijo-Rodriguez et al. (2018) 1700 GDD literature value.
        TuberInit   -- observed at IND and NLD. RSA records none and takes rsa_tuber_init(), the
                       2024 first tuber-bearing harvest (an upper bound); TSUM1 is integrated on
                       2024 and carried to the other RSA seasons, which cannot date the stage.

    teffmx is the upper limit on daily warming used for TSUMEM only: a number applies that cap,
    np.inf assumes no cap, None takes the crop-file default.

    Used by 02.6_pp_phenology_thermal.py (calibration) and 02.1_pp_emergence.py (identifiability)
    (which passes np.inf explicitly)."""
    tbasem = crop_default(site, seasons[0], cultivar, "TBASEM")
    if teffmx is None:
        teffmx = crop_default(site, seasons[0], cultivar, "TEFFMX")
    # season closing the TSUM1 window: its own, except RSA, where only 2024 dates tuber initiation
    s1 = RSA_TI_SEASON if site == "RSA" else None
    tsumem_vals, tsum1_vals, tsum2_vals = [], [], []
    for s in seasons:
        try:
            dt = site_dtsmtb(site, s, cultivar)
            d = obs_dates(site, s, cultivar, field)
            tsumem_vals.append(thermal_emergence(
                tavg_window(site, s, cultivar, d["Start"], d["Emergence"]), tbasem, teffmx))
            d1 = obs_dates(site, s1 or s, cultivar, field)
            ti = rsa_tuber_init() if site == "RSA" else d1["TuberInitiation"]
            tsum1 = thermal_afgen(tavg_window(site, s1 or s, cultivar, d1["Emergence"], ti),
                                  site_dtsmtb(site, s1 or s, cultivar))
            total = SEIJO_TOTAL if site == "IND" else thermal_afgen(
                tavg_window(site, s, cultivar, d["Emergence"], d["Maturity"]), dt)
            tsum1_vals.append(tsum1)
            tsum2_vals.append(total - tsum1)
        except Exception as e:
            print(f"{site} {s}: cannot integrate observed window ({e}); excluded from average.")

    return {"TSUMEM": float(np.mean(tsumem_vals)),
            "TSUM1": float(np.mean(tsum1_vals)),
            "TSUM2": float(np.mean(tsum2_vals))}


def ind_maturity_dvs(season, cultivar, tsum1, tsum2, dtsmtb):
    """DVS at IND's observed Maturity date, read off the DVS 1->2 line.

    That date is a pre-maturity haulm kill, not DVS=2. Its emergence->Maturity thermal time is
    mapped linearly between DVS=1 (tsum1) and DVS=2 (tsum1+tsum2). Derived, not observed."""
    d = obs_dates("IND", season, cultivar)
    tt = thermal_afgen(tavg_window("IND", season, cultivar, d["Emergence"], d["Maturity"]), dtsmtb)
    return 1.0 + (tt - tsum1) / tsum2


# -- Cluster B: broken-stick / Seijo / DVS-mapping helpers, keyed (season, ..., location, cultivar) --

def obs_row(season, location="IND", cultivar=IND_CULT, field="OpenField"):
    """Cluster-B argument order of obs_dates (season first). Same observed row."""
    return obs_dates(location, season, cultivar, field)


def tavg_window_cb(season, start, end, location="IND", cultivar=IND_CULT):
    """Cluster-B argument order of tavg_window (season first). Daily mean (TMIN+TMAX)/2 over
    [start, end) (end-exclusive, matching 02.6_pp_phenology_thermal.py's TSUM window convention so
    base-2 reproduces the frozen numbers)."""
    return tavg_window(location, season, cultivar, start, end)


def inc_broken_stick(T, table):
    """AFGEN piecewise-linear interpolation of a broken-stick DTSMTB table [x0,y0, x1,y1, ...]."""
    xt, yt = table[0::2], table[1::2]
    return float(np.interp(T, xt, yt))


def seijo_gdd(t):
    return max(min(t, SEIJO_TUPPER), SEIJO_TBASE) - SEIJO_TBASE


def integrate_window(location, season, cultivar, start, end, inc_fn):
    """Cumulative thermal increment of inc_fn over the end-exclusive window [start, end)."""
    return float(sum(inc_fn(t) for t in tavg_window_cb(season, start, end, location, cultivar)))


def tt_to_dvs(tt, tsum1, tsum2):
    """Map cumulative post-emergence thermal time to DVS on the 0->1 (/TSUM1) then 1->2 (/TSUM2)
    two-segment line."""
    return tt / tsum1 if tt <= tsum1 else 1.0 + (tt - tsum1) / tsum2


def dtsmtb_for_base(N):
    """DTSMTB table for base temperature N as a single STRAIGHT line pivoting about the upper knot
    (30, 28): response is 0 below N, then rises linearly from (N, 0) to (30, 28). Raising N steepens
    the slope (28 / (30 - N)) instead of adding a mid break -- the x=13 knot stays on the line, so
    there is no kink. N=2 reproduces the original base-2 line y = x - 2 -> [0,0, 2,0, 13,11, 30,28];
    N=5 -> slope 1.12, y = 1.12*x - 5.6 -> [0,0, 5,0, 13,8.96, 30,28]. Knots at x = [0, N, 13, 30]."""
    slope = 28.0 / (30.0 - N)
    return [0.0, 0.0, float(N), 0.0, 13.0, slope * (13.0 - N), 30.0, 28.0]


def seijo_maturity_date(season, emergence, location="IND", cultivar=IND_CULT):
    """Date the Seijo base-2/cap-26 GDD line reaches 1700 from emergence -- independent of the
    candidate DTSMTB. None if the weather record ends first (expected for IND: pre-maturity
    haulm-kill, field weather stops well short of 1700)."""
    exp = find_experiment(load_experiments(location), location, season, cultivar)
    wdp = load_weather(exp["weather_file"])
    end = min((emergence + pd.Timedelta(days=250)).date(), wdp.last_date)
    days = pd.date_range(emergence, pd.Timestamp(end), freq="D")
    cum = 0.0
    for d in days:
        wd = wdp(d.date())
        cum += seijo_gdd((wd.TMIN + wd.TMAX) / 2.0)
        if cum >= SEIJO_TOTAL:
            return d
    return None


def rederive_tsum(location, cultivar, cal_season, inc_fn, field="OpenField"):
    """TSUM1/TSUM2 for one increment function, re-integrated from the CAL season's observed windows.

    Same rule as derive_obs_fixed, with inc_fn in place of DTSMTB:
        TSUM1 = integrate(Emergence -> TuberInit);  TSUM2 = integrate(Emergence -> Maturity) - TSUM1
    Only the two dates differ by site:
        Maturity  -- observed senescence (DVS=2) at NLD and RSA. IND's record is a haulm kill, so
                     the date comes from Seijo's fixed base-2/cap-26 1700 GDD line (end-inclusive,
                     hence the +1 day); where the weather record ends before 1700 GDD is reached,
                     TSUM2 falls back to the direct plug 1700 - TSUM1.
        TuberInit -- observed at IND and NLD; rsa_tuber_init() at RSA, whose only dated season is
                     RSA_TI_SEASON, so RSA must calibrate on it."""
    r = obs_row(cal_season, location, cultivar, field)
    if location == "RSA":
        assert cal_season == RSA_TI_SEASON, f"RSA TSUM1 needs the {RSA_TI_SEASON} tuber-init date"
    ti = rsa_tuber_init() if location == "RSA" else r["TuberInitiation"]
    tsum1 = integrate_window(location, cal_season, cultivar, r["Emergence"], ti, inc_fn)

    if location == "IND":
        maturity = seijo_maturity_date(cal_season, r["Emergence"], location, cultivar)
        if maturity is None:
            return tsum1, SEIJO_TOTAL - tsum1
        maturity = maturity + pd.Timedelta(days=1)
    else:
        maturity = r["Maturity"]
    total = integrate_window(location, cal_season, cultivar, r["Emergence"], maturity, inc_fn)
    return tsum1, total - tsum1
