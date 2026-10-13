"""
utils/soil_curves.py
--------------------
Texture -> van Genuchten parameters -> retention and conductivity tables, plus
the FC-only correction used at both sites. The former FC + WP correction is
retained for explicitly historical diagnostics.

Equations, symbols, units and what each correction moves besides its target points:
docs/soil_retention_correction.md.

Lifted verbatim from the two soil-build scripts now in scripts/archive/, which held
the same functions in six copies across the 00 and 05.x scripts. Nothing here knows
about a site: callers supply texture, bulk density and targets.
"""

import numpy as np
import pandas as pd

# PCSE's own pF grid (SoilProfile docstring, pcse/soil/soil_profile.py), with exact
# nodes at saturation, field capacity and wilting point. Followed Berghuijs et al. (2026).
PF_GRID = np.array([-1.0, 1.0, 1.3, 1.7, 2.0, 2.3, 2.4, 2.7,
                    3.0, 3.3, 3.7, 4.0, 4.17, 4.2, 6.0])

PF_SATURATION = -1.0
PF_FIELD_CAPACITY = 2.0
PF_WILTING_POINT = 4.2
# aeration-stress threshold where CRAIRC is read; Wessling (1991) via Berghuijs et al. (2026)
PF_OXYGEN_STRESS = np.log10(25.0)

PSI_FC = 10.0 ** PF_FIELD_CAPACITY   # cm suction at field capacity
PSI_WP = 10.0 ** PF_WILTING_POINT    # cm suction at wilting point

THETA_R_PTF = 0.01   # residual water content, fixed rather than predicted; Berghuijs et al. (2024)
OM_FLOOR = 0.01      # % organic matter; Wosten's log(OM) and 1/OM are undefined at OM = 0


# ── layer geometry ───────────────────────────────────────────────────────────

def rescale_to_layers(df_raw, layer_bounds, theta_r=THETA_R_PTF):
    """Raw horizons -> target layers by centroid-based linear interpolation.

    Followed Berghuijs et al. (2024) supplementary text S7. df_raw needs columns
    profile_code, layer_top, layer_bottom, centroid, bulk_density, clay_pct,
    silt_pct, om_pct, c_n, ph. Layer i spans [layer_bounds[i], layer_bounds[i+1]].
    """
    rows = []
    for profile_code, profile in df_raw.groupby("profile_code"):
        centroids = profile["centroid"].to_numpy()
        for top, bottom in zip(layer_bounds[:-1], layer_bounds[1:]):
            centroid = (top + bottom) / 2
            in_layer = profile[(profile["layer_top"] <= centroid) & (centroid < profile["layer_bottom"])]
            topsoil = bool((in_layer["layer_top"] == 0).iloc[0])
            rows.append({
                "profile_code": profile_code,
                "layer_top": top,
                "layer_bottom": bottom,
                "centroid": centroid,
                "bulk_density": np.interp(centroid, centroids, profile["bulk_density"]),
                "clay_pct": np.interp(centroid, centroids, profile["clay_pct"]),
                "silt_pct": np.interp(centroid, centroids, profile["silt_pct"]),
                "om_pct": np.interp(centroid, centroids, profile["om_pct"]),
                "CNRatioSOMI": np.interp(centroid, centroids, profile["c_n"]),
                "Soil_pH": np.interp(centroid, centroids, profile["ph"]),
                "theta_r": theta_r,
                "topsoil": topsoil,
            })
    return pd.DataFrame(rows)


# ── Wosten and Nemes (2004) pedotransfer function ────────────────────────────
# The continuous HYPRES set. clay, silt, om in %, bulk_density in g/cm3,
# topsoil a 0/1 flag.

def wosten_theta_s(clay, bulk_density, silt, om, topsoil):
    return (0.7919 + 0.001691 * clay - 0.29619 * bulk_density - 0.000001491 * silt**2
            + 0.0000821 * om**2 + 0.02427 / clay + 0.01113 / silt + 0.01472 * np.log(silt)
            - 0.0000733 * om * clay - 0.000619 * bulk_density * clay
            - 0.001183 * bulk_density * om - 0.0001664 * topsoil * silt)


def wosten_alpha(clay, bulk_density, silt, om, topsoil):
    t_alpha = (-14.96 + 0.03135 * clay + 0.0351 * silt + 0.646 * om + 15.29 * bulk_density
               - 0.192 * topsoil - 4.671 * bulk_density**2 - 0.000781 * clay**2
               - 0.00687 * om**2 + 0.0449 / om + 0.0663 * np.log(silt) + 0.1482 * np.log(om)
               - 0.04546 * bulk_density * silt - 0.4852 * bulk_density * om
               + 0.00673 * topsoil * clay)
    return np.exp(t_alpha)


def wosten_n(clay, bulk_density, silt, om, topsoil):
    t_n = (-25.23 - 0.02195 * clay + 0.0074 * silt - 0.1940 * om + 45.5 * bulk_density
           - 7.24 * bulk_density**2 + 0.0003658 * clay**2 + 0.002885 * om**2
           - 12.81 / bulk_density - 0.1524 / silt - 0.01958 / om - 0.2876 * np.log(silt)
           - 0.0709 * np.log(om) - 44.6 * np.log(bulk_density) - 0.02264 * bulk_density * clay
           + 0.0896 * bulk_density * om + 0.00718 * topsoil * clay)
    return np.exp(t_n) + 1


def wosten_lambda(clay, bulk_density, silt, om, topsoil):
    t_lambda = (0.0202 + 0.0006193 * clay**2 - 0.001136 * om**2 - 0.2316 * np.log(om)
                - 0.03544 * bulk_density * clay + 0.00283 * bulk_density * silt
                + 0.0488 * bulk_density * om)
    return (10 * np.exp(t_lambda) - 10) / (1 + np.exp(t_lambda))


def wosten_k_sat(clay, bulk_density, silt, om, topsoil):
    t_k_sat = (7.755 + 0.0352 * silt + 0.93 * topsoil - 0.967 * bulk_density**2
               - 0.000484 * clay**2 - 0.000322 * silt**2 + 0.001 / silt - 0.0748 / om
               - 0.643 * np.log(silt) - 0.01398 * bulk_density * clay
               - 0.1673 * bulk_density * om + 0.02986 * topsoil * clay
               - 0.03305 * topsoil * silt)
    return np.exp(t_k_sat)


def wosten_van_genuchten(clay, bulk_density, silt, om, theta_r, topsoil):
    """Van Genuchten-Mualem parameters from texture via the Wosten PTF."""
    return {
        "theta_r": theta_r,
        "theta_s": wosten_theta_s(clay, bulk_density, silt, om, topsoil),
        "alpha": wosten_alpha(clay, bulk_density, silt, om, topsoil),
        "n": wosten_n(clay, bulk_density, silt, om, topsoil),
        "lambda": wosten_lambda(clay, bulk_density, silt, om, topsoil),
        "k_sat": wosten_k_sat(clay, bulk_density, silt, om, topsoil),
    }


def vg_from_layers(df_layers, keep=()):
    """PTF parameters for every row of df_layers, alongside the columns named in `keep`.

    df_layers needs clay_pct, bulk_density, silt_pct, om_pct, theta_r, topsoil.
    """
    params = [wosten_van_genuchten(row.clay_pct, row.bulk_density, row.silt_pct,
                                   row.om_pct, row.theta_r, row.topsoil)
              for row in df_layers.itertuples()]
    return pd.concat([df_layers[list(keep)], pd.DataFrame(params)], axis=1)


# ── van Genuchten (1980) retention and Mualem (1976) conductivity ────────────

def van_genuchten_theta(pF, alpha, n, theta_r, theta_s):
    """Volumetric water content (cm3 cm-3) at suction 10**pF cm."""
    psi = np.power(10, pF)
    return theta_r + (theta_s - theta_r) / np.power(1 + (alpha * psi)**n, 1 - 1 / n)


def van_genuchten_cond(pF, alpha, lam, k_sat, n):
    """log10 unsaturated hydraulic conductivity (cm day-1), Mualem-van Genuchten."""
    psi = np.power(10, pF)
    m = 1 - 1 / n
    ah = alpha * psi
    h1 = np.power(1 + ah**n, m)
    h2 = ah**(n - 1)
    denom = np.power(1 + ah**n, m * (lam + 2))
    return np.log10(k_sat * (h1 - h2)**2 / denom)


def curves_from(df_vg, pf_grid=PF_GRID):
    """Per-layer retention and log10-conductivity tables, keyed on (layer_top, layer_bottom)."""
    ret, cond = {}, {}
    for _, row in df_vg.iterrows():
        key = (row["layer_top"], row["layer_bottom"])
        ret[key] = van_genuchten_theta(pf_grid, row["alpha"], row["n"],
                                       row["theta_r"], row["theta_s"])
        cond[key] = van_genuchten_cond(pf_grid, row["alpha"], row["lambda"],
                                       row["k_sat"], row["n"])
    return ret, cond


# ── the two corrections ──────────────────────────────────────────────────────

def solve_alpha(fc_target, theta_r, theta_s, n, pf=PF_FIELD_CAPACITY):
    """Correction A: alpha (cm-1) putting theta(pF `pf`) on `fc_target`.

    theta_r, theta_s and lambda, n, k_sat stay put. Valid for theta_r < fc_target
    < theta_s; outside that the bracket is negative and there is no real root. `pf`
    is an argument because 05.4.1 scans which suction counts as field capacity.
    NaN where theta_s <= fc_target, which the bulk-density scan in 05.4.2 reaches
    on purpose: there the target cannot be met at all.
    docs/soil_retention_correction.md section 3.
    """
    if theta_s <= fc_target:
        return np.nan
    m = 1 - 1 / n
    return (((theta_s - theta_r) / (fc_target - theta_r)) ** (1 / m) - 1) ** (1 / n) / 10.0 ** pf


def solve_fc_wp(fc_target, wp_target, theta_s, n, n_iter=50):
    """Historical correction: alpha and theta_r putting the curve through both targets.

    theta_s, n, lambda and k_sat stay put. alpha has a closed form once theta_r is
    known and theta_r follows from the wilting-point equation once alpha is known,
    so the two are iterated from theta_r = wp_target. Valid for wp_target <
    fc_target < theta_s. Convergence is not proven; check the result with
    check_targets. (NaN, NaN) where theta_s <= fc_target, as in solve_alpha.
    docs/soil_retention_correction.md section 4.
    """
    if theta_s <= fc_target:
        return np.nan, np.nan
    m = 1 - 1 / n
    theta_r = wp_target
    for _ in range(n_iter):
        alpha = solve_alpha(fc_target, theta_r, theta_s, n)
        s_wp = 1 / (1 + (alpha * PSI_WP) ** n) ** m
        theta_r = (wp_target - theta_s * s_wp) / (1 - s_wp)
    return solve_alpha(fc_target, theta_r, theta_s, n), theta_r


def correct_layers(df_vg, layers, fc_target, wp_target=None):
    """A copy of df_vg with the named (top, bottom) layers re-solved onto the targets.

    wp_target None -> current FC-only correction (alpha only). A wp_target is
    used only for historical diagnostics (alpha and theta_r). Other layers stay raw.
    """
    out = df_vg.copy()
    for top, bottom in layers:
        i = out.index[(out["layer_top"] == top) & (out["layer_bottom"] == bottom)][0]
        if wp_target is None:
            out.loc[i, "alpha"] = solve_alpha(fc_target, out.loc[i, "theta_r"],
                                              out.loc[i, "theta_s"], out.loc[i, "n"])
        else:
            out.loc[i, "alpha"], out.loc[i, "theta_r"] = solve_fc_wp(
                fc_target, wp_target, out.loc[i, "theta_s"], out.loc[i, "n"])
    return out


# ── reading a table, and the checks before a soil is run ─────────────────────

def read_pf(table, pf, pf_grid=PF_GRID):
    """Value of a tabulated curve at one pF."""
    return float(np.interp(pf, pf_grid, np.asarray(table)))


def pf_table(values, pf_grid=PF_GRID):
    """[pF, value, pF, value, ...] flat list of native floats, the form PCSE reads."""
    return [v for pair in zip(pf_grid.tolist(), np.asarray(values).tolist()) for v in pair]


def check_targets(ret, layers, fc_target, wp_target=None, tol=1e-6):
    """Read theta back off the tabulated curve and compare against the targets.

    Stands in for a convergence proof of solve_fc_wp, and catches a target set
    outside the valid range. docs/soil_retention_correction.md section 7, check 1.
    """
    for key in layers:
        got_fc = read_pf(ret[key], PF_FIELD_CAPACITY)
        if abs(got_fc - fc_target) > tol:
            raise ValueError(f"{key}: theta(pF 2.0) = {got_fc:.6f}, target {fc_target}")
        if wp_target is not None:
            got_wp = read_pf(ret[key], PF_WILTING_POINT)
            if abs(got_wp - wp_target) > tol:
                raise ValueError(f"{key}: theta(pF 4.2) = {got_wp:.6f}, target {wp_target}")


def check_tables(ret, cond, label=""):
    """Both curves must fall with suction and stay finite, or PCSE's dry-flow
    iteration fails. docs/soil_retention_correction.md section 7, check 2."""
    for key in ret:
        sm, lk = np.asarray(ret[key]), np.asarray(cond[key])
        if not np.all(np.diff(sm) < 0):
            raise ValueError(f"{label} {key}: retention does not fall with pF")
        if not (np.all(np.diff(lk) < 0) and np.all(np.isfinite(lk))):
            raise ValueError(f"{label} {key}: conductivity not falling or not finite")


def table_summary(ret, cond):
    """SM0, SMFCF, SMW, CRAIRC and K0 read off each layer's tables, as a frame."""
    rows = []
    for key in ret:
        sm, lk = ret[key], cond[key]
        rows.append({
            "layer": f"{key[0]:.0f}-{key[1]:.0f} cm",
            "SM0": read_pf(sm, PF_SATURATION),
            "SMFCF": read_pf(sm, PF_FIELD_CAPACITY),
            "SMW": read_pf(sm, PF_WILTING_POINT),
            "CRAIRC": read_pf(sm, PF_SATURATION) - read_pf(sm, PF_OXYGEN_STRESS),
            "K0": 10 ** read_pf(lk, PF_SATURATION),
        })
    return pd.DataFrame(rows)


# ── assembling the file PCSE reads ───────────────────────────────────────────

OMEGAC = 0.7               # schema compatibility only; the layered balance does not read it
SUBSOIL_THICKNESS = 200.0  # cm below the deepest modelled layer; duplicates its properties


def soil_dict(df_vg, ret, cond, pf_grid=PF_GRID):
    """The WOFOST multi-layer soil mapping, ready for yaml.safe_dump.

    Schema follows Berghuijs et al. (2026) SoilFileWriter, matched against
    pcse/soil/soil_profile.py. CRAIRC is a read-off, not a parameter:
    theta(pF -1) - theta(pF log10(25)) from the same tables.
    """
    soil_layers = [
        {
            "Thickness": float(row.layer_bottom - row.layer_top),
            "CNRatioSOMI": float(row.CNRatioSOMI),
            "CRAIRC": float(van_genuchten_theta(PF_SATURATION, row.alpha, row.n, row.theta_r, row.theta_s)
                            - van_genuchten_theta(PF_OXYGEN_STRESS, row.alpha, row.n, row.theta_r, row.theta_s)),
            "FSOMI": float(row.om_pct / 100),
            "RHOD": float(row.bulk_density),
            "Soil_pH": float(row.Soil_pH),
            "SMfromPF": pf_table(ret[(row.layer_top, row.layer_bottom)], pf_grid),
            "CONDfromPF": pf_table(cond[(row.layer_top, row.layer_bottom)], pf_grid),
        }
        for row in df_vg.itertuples()
    ]
    subsoil_layer = dict(soil_layers[-1])
    subsoil_layer["Thickness"] = SUBSOIL_THICKNESS
    return {
        "RDMSOL": float(sum(layer["Thickness"] for layer in soil_layers)),
        "omegac": OMEGAC,
        "SoilProfileDescription": {
            "PFWiltingPoint": PF_WILTING_POINT,
            "PFFieldCapacity": PF_FIELD_CAPACITY,
            # k_sat of the surface layer, cm/day; not the top of CONDfromPF, which
            # sits at pF -1 rather than at zero suction
            "SurfaceConductivity": float(df_vg.iloc[0]["k_sat"]),
            "GroundWater": False,
            "SoilLayers": soil_layers,
            "SubSoilType": subsoil_layer,
        },
    }
