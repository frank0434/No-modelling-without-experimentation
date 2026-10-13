# Soil profiles for the multi-layer water balance, both sites, one script.
#
# Texture and bulk density -> Wosten pedotransfer function -> van Genuchten
# parameters -> retention and conductivity tables -> a WOFOST soil file.
# two layers at both sites:
#   IND  0-30 cm (the layer the 20 cm probe sits in) and 30-35 cm plough pan, roots
#        stopped at 30 cm by RDMCR on the layer boundary
#   NLD  0-40 cm (spanning both probe depths) and 40-60 cm
#
# The equations, every symbol with its unit, and what each correction moves besides
# its target points are in docs/soil_retention_correction.md. They are implemented
# once, in utils/soil_curves.py; this script supplies texture, density and targets.
#
#   1    settings and geometry, both sites
#   2-4  NLD: texture, the probes and the field-capacity correction, the files
#   5-7  IND: texture, the probes and the field-capacity correction, the files
#   8    what was written, read back off the files

# %%
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

_here = pathlib.Path().resolve()
sys.path.insert(0, str(next(p for p in (_here, *_here.parents)
                            if (p/"scripts").exists() and (p/"data").exists()) / "scripts"))
from utils.plotstyle import display_season
from utils.paths import DATA, OUTPUT, ROOT
from utils import plotstyle
from utils import soil_curves as sc

plotstyle.use_style()
OUTPUT_DIR = OUTPUT / "soil_ptf"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %%
# ── 1. Geometry, both sites ──────────────────────────────────────────────────
# NLD builds BOTH geometries every run -- they write different filenames, nothing is
# overwritten, and 05.1 runs all four resulting soils against each other.
#   3layer  0-20 / 20-40 / 40-60 cm, a boundary at each probe depth
#   2layer  0-40 / 40-60 cm, the boundary between the two probes removed. Merging
#           lands the raw curve at 0.2008 against a 0.20 target, which leaves the
#           correction with nothing to do where the probes can see.
# IND builds one, selected here. Merging 0-10 and 10-30 matters for more than
# tidiness there: the layered balance takes soil evaporation out of layer 0 first and
# cascades deeper only once that layer reaches wilting point
# (multilayer_waterbalance.py:709-726), which a 10 cm layer holding about 2.8 cm of
# extractable water never does, so one 0-30 cm layer puts evaporation on the layer
# the probe sits in. "3layer" regenerates the profile 05.4.1 and 05.4.2 were written
# against.
I_GEOMETRY = "2layer"

GEOMETRIES = {
    "IND": {
        # 30 is the root base (RDMCR must land on a layer boundary); 30-35 is the
        # plough pan the penetrometer located. rhod is unmeasured (see section 5);
        # rhod is Dry bulk density of the soil, g cm⁻³ 
        # the merged layer takes the thickness-weighted mean, (1.50*10 + 1.55*20)/30.
        "2layer": {"bounds": [0, 30, 35], "rhod": [1.533, 1.75], "suffix": "_2layer"},
        "3layer": {"bounds": [0, 10, 30, 35], "rhod": [1.50, 1.55, 1.75], "suffix": ""},
    },
    "NLD": {
        # 20 and 40 are the NLD_SM_obs.csv sensor depths; 60 is the rooting depth
        # (RDMSOL = 60.0, data/soil/netherlands_soil.soil). Bulk density comes from
        # the BOFEK texture, so no placeholder is needed.
        "2layer": {"bounds": [0, 40, 60], "suffix": "_2layer"},
        "3layer": {"bounds": [0, 20, 40, 60], "suffix": ""},
    },
}

# a wetting event is a daily rise above EVENT_RISE; its plateau is the reading 2-4
# days later once the daily change has fallen below PLATEAU_FLAT
EVENT_RISE = 0.02    # cm3 cm-3 per day
PLATEAU_FLAT = 0.005  # cm3 cm-3 per day


def plateau_scan(series_by_group):
    """Post-wetting plateaus per group. Evidence behind the manual target, not input.

    Returns every wetting event found, with the plateau day and reading where one
    was reached and NaT/NaN where the scan found none, so the figures below can show
    the events that produced no plateau alongside the ones that did.
    """
    rows = []
    for name, series in series_by_group.items():
        change = series.diff()
        for event in change[change > EVENT_RISE].index:
            hit = None
            for lag in (2, 3, 4):
                day = event + pd.Timedelta(days=lag)
                nxt = day + pd.Timedelta(days=1)
                if (day in series.index and abs(change.get(day, np.nan)) < PLATEAU_FLAT
                        and change.get(nxt, 0) <= PLATEAU_FLAT):
                    hit = day
                    break
            rows.append({"group": name, "event": event,
                         "plateau_day": hit if hit is not None else pd.NaT,
                         "plateau_SM": float(series[hit]) if hit is not None else np.nan})
    df = pd.DataFrame(rows, columns=["group", "event", "plateau_day", "plateau_SM"])
    for r in df.dropna(subset=["plateau_SM"]).itertuples():
        print(f"  {r.group}: event {r.event.date()} -> plateau "
              f"{r.plateau_day.date()}  {r.plateau_SM:.3f}")
    hits = df["plateau_SM"].dropna()
    print(f"  plateau scan: {len(df)} wetting events, {len(hits)} reached a plateau"
          + (f", median {hits.median():.3f}, range {hits.min():.3f}-{hits.max():.3f}"
             if len(hits) else ""))
    return df


def write_soil(df_vg, fname):
    """Curves from a parameter set, both usability checks, then the file."""
    ret, cond = sc.curves_from(df_vg)
    sc.check_tables(ret, cond, fname)
    fp = DATA / "soil" / fname
    with open(fp, "w") as f:
        yaml.safe_dump(sc.soil_dict(df_vg, ret, cond), f, sort_keys=False)
    print(f"wrote {fp.name}")
    return ret, cond


# %%
# ── 2. NLD: raw BOFEK texture rescaled to the target layers, then Wosten ─────
# Raw BOFEK horizons (0-25, 25-50, 50-120 cm) are interpolated to each target
# layer's centroid, following Berghuijs et al. (2024) supplementary S7. The curves
# of neighbouring layers are never averaged: a mean of two retention curves is not
# a retention curve of anything.
n_raw = pd.read_csv(DATA / "soil" / "soil_profiles_raw.csv")
n_raw["bulk_density"] = n_raw["bulkdensity"] / 1000                    # kg/m3 -> g/cm3
n_raw["clay_pct"] = n_raw["pclay"] * 100                               # fraction -> %
n_raw["silt_pct"] = n_raw["psilt"] * 100
n_raw["om_pct"] = (n_raw["orgmatter"] * 100).clip(lower=sc.OM_FLOOR)
n_raw["centroid"] = (n_raw["layer_top"] + n_raw["layer_bottom"]) / 2

N_GEOMS = list(GEOMETRIES["NLD"])
n_vg = {}
for g in N_GEOMS:
    lay = sc.rescale_to_layers(n_raw, GEOMETRIES["NLD"][g]["bounds"])
    n_vg[g] = sc.vg_from_layers(lay, keep=["profile_code", "layer_top", "layer_bottom",
                                           "topsoil", "bulk_density", "om_pct",
                                           "CNRatioSOMI", "Soil_pH"])
    n_vg[g].to_csv(OUTPUT_DIR / f"nld_van_genuchten_params{GEOMETRIES['NLD'][g]['suffix']}.csv",
                   index=False)
    print(f"\nNLD {g}, bounds {GEOMETRIES['NLD'][g]['bounds']}:")
    print(n_vg[g].round(4).to_string(index=False))


# %%
# ── 3. NLD: the probes, and the field-capacity correction ────────────────────
N_SEASON = "2025"
n_sm = pd.read_csv(DATA / "obs" / "NLD_SM_obs.csv", parse_dates=["day"])
n_sm = n_sm[n_sm["season"].astype(str) == N_SEASON]
print("per depth, before averaging:")
print(n_sm.groupby("depth")["SM_obs"].agg(["count", "min", "mean", "max"]).round(3))

n_pooled = n_sm.groupby(["treatment", "day"])["SM_obs"].mean().reset_index()
print("\naveraged over both depths, one series per treatment:")
print(n_pooled.groupby("treatment")["SM_obs"].agg(["count", "min", "mean", "max"]).round(3))
n_scan = plateau_scan({t: g.set_index("day")["SM_obs"].sort_index()
                       for t, g in n_pooled.groupby("treatment")})

# The 3-layer targets are one per probe depth, so each gets its own evidence rather
# than sharing the pooled number. Same wetting-event/plateau rule, run per depth.
n_by_depth = (n_sm.groupby(["depth", "treatment", "day"])["SM_obs"].mean().reset_index())
for depth in sorted(n_by_depth["depth"].unique()):
    d = n_by_depth[n_by_depth["depth"] == depth]
    print(f"\n{depth:.0f} cm probe, plateaus per treatment:")
    hits = plateau_scan({t: g.set_index("day")["SM_obs"].sort_index()
                         for t, g in d.groupby("treatment")})["plateau_SM"].dropna()
    print(f"  {depth:.0f} cm season mean {d['SM_obs'].mean():.3f}, "
          f"range {d['SM_obs'].min():.3f}-{d['SM_obs'].max():.3f}")

# Manual field-capacity targets, judgement calls on the series above. Only layers a
# probe observes are corrected; the rest keep their raw pedotransfer curve, the same
# rule the pan follows at IND.
#   2layer  0.20 on 0-40 cm, the thickness-weighted mean of the two targets the
#           3-layer profile carried, so the merged layer holds the same water over
#           0-40 cm as the two it replaces. The plateau scan above lands lower, near
#           0.17; the scan is biased low under an active canopy, which is why it is
#           printed and not used. 
#   3layer  0.15 on 0-20 cm and 0.25 on 20-40 cm, read off the two probe depths
#           separately -- the probes have the top layer drier than the one below and
#           the raw curves have it the other way round, which is what these two
#           reverse. 40-60 cm is NOT corrected here: it was in the earlier build,
#           inheriting 0.25 from the layer above, but no probe reaches it. Leaving it
#           raw means this file no longer matches that earlier one in its deepest
#           layer or in the 200 cm subsoil that duplicates it.
# ---- THE KNOB: edit a value, re-run this file, the soils are rewritten ----------
# Volumetric water content (cm3 cm-3) the corrected curve must pass through at pF 2.0.
# One entry per layer that a probe observes; a layer left out keeps its raw curve.
# alpha is re-solved to meet it (sc.solve_alpha); theta_s, n, lambda and k_sat do not
# move. The readout below and the section 4 figure show what a change does before any
# crop is run.
N_FC_TARGET = {
    "2layer": {(0, 40): 0.20},
    "3layer": {(0, 20): 0.15, (20, 40): 0.25},
}
# --------------------------------------------------------------------------------

n_vg_corr = {}
print("\nfield-capacity targets (pF 2.0), and what alpha does to meet them:")
for g in N_GEOMS:
    corr = n_vg[g].copy()
    for layer, target in N_FC_TARGET[g].items():
        corr = sc.correct_layers(corr, [layer], target)
    n_vg_corr[g] = corr
    print(f"  {g}:")
    for i in n_vg[g].index:
        layer = (n_vg[g].loc[i, "layer_top"], n_vg[g].loc[i, "layer_bottom"])
        target = N_FC_TARGET[g].get(layer)
        label = f"target {target:.2f}" if target is not None else "no probe, left raw"
        print(f"    {layer[0]:.0f}-{layer[1]:.0f} cm  {label}: "
              f"alpha {n_vg[g].loc[i,'alpha']:.4f} -> {corr.loc[i,'alpha']:.4g}")

# What the chosen targets leave the crop, read straight off the corrected curves --
# no model run. SMFCF and SMW are the read-offs at pF 2.0 and 4.2, available is
# (SMFCF - SMW) * thickness. The last line is the 0-40 cm thickness-weighted field
# capacity, the block the pooled probe series is scored against, so a pair of
# per-layer targets can be judged against what the probes actually read.
SCORE_DEPTH = 40.0   # cm, the block the two probes cover
print(f"\nwhat the targets leave the crop (probe mean over 0-40 cm "
      f"{n_pooled['SM_obs'].mean():.3f}, range {n_pooled['SM_obs'].min():.3f}-"
      f"{n_pooled['SM_obs'].max():.3f}):")
for g in N_GEOMS:
    ret_g, _ = sc.curves_from(n_vg_corr[g])
    fc_block, mm_block, thick_block = 0.0, 0.0, 0.0
    print(f"  {g}:")
    for (top, bottom), tab in ret_g.items():
        th = bottom - top
        fc = sc.read_pf(tab, sc.PF_FIELD_CAPACITY)
        wp = sc.read_pf(tab, sc.PF_WILTING_POINT)
        print(f"    {top:.0f}-{bottom:.0f} cm  SMFCF {fc:.4f}  SMW {wp:.4f}  "
              f"available {(fc - wp) * th * 10:5.1f} mm")
        if bottom <= SCORE_DEPTH:
            fc_block += fc * th
            mm_block += (fc - wp) * th * 10
            thick_block += th
    print(f"    0-{SCORE_DEPTH:.0f} cm  field capacity {fc_block / thick_block:.4f}, "
          f"available {mm_block:.1f} mm")


# FIGURE: the scan the target is judged against. One panel per treatment: the
# averaged probe series, every wetting event the scan found, the plateau it settled
# on where it reached one, and the target that was set by hand. The scan is biased
# low under an active canopy, so this shows how far the target sits above it.
SCAN_FIGSIZE = (6, 3.2)
SCAN_YLIM = (0.10, 0.28)
SCAN_SERIES = dict(color="black", lw=0.9)
SCAN_EVENT = dict(color="0.7", lw=0.8, linestyle="dashed")
SCAN_PLATEAU = dict(marker="o", markersize=5, color="black", linestyle="none")
SCAN_TARGET = dict(color="black", lw=1.0, linestyle="dotted")

print("\nscan rows per treatment (events, of which reached a plateau):")
print(n_scan.assign(hit=n_scan["plateau_SM"].notna())
      .groupby("group").agg(events=("event", "size"), plateaus=("hit", "sum")).to_string())
_missing = sorted(set(n_pooled["treatment"]) - set(n_scan["group"]))
print(f"treatments with no wetting event at all: {_missing or 'none'}")

# The obs here is one series over 0-40 cm, so the line to compare it against is
# the thickness-weighted mean of the targets over that block -- 0.20 either way.
# both geometries put the same water over 0-40 cm, so this is one line either way
_n_target = (sum((b - a) * v for (a, b), v in N_FC_TARGET["3layer"].items())
             / sum(b - a for (a, b) in N_FC_TARGET["3layer"]))
_n_groups = sorted(n_pooled["treatment"].unique())
fig, axes = plt.subplots(1, len(_n_groups), figsize=SCAN_FIGSIZE, sharey=True)
for ax, t in zip(np.atleast_1d(axes), _n_groups):
    g = n_pooled[n_pooled["treatment"] == t].sort_values("day")
    ax.plot(g["day"], g["SM_obs"], label="probes, both depths averaged", **SCAN_SERIES)
    sc_t = n_scan[n_scan["group"] == t]
    for d in sc_t["event"]:
        ax.axvline(d, **SCAN_EVENT)
    hit = sc_t.dropna(subset=["plateau_SM"])
    ax.plot(hit["plateau_day"], hit["plateau_SM"], label="plateau", **SCAN_PLATEAU)
    ax.axhline(_n_target, label=f"target {_n_target:.2f}", **SCAN_TARGET)
    ax.set_ylim(*SCAN_YLIM)
    ax.set_title(t, fontsize=9)
    ax.tick_params(axis="x", labelrotation=45, labelsize=6)
np.atleast_1d(axes)[0].set_ylabel("SM 0-40 cm (cm$^3$ cm$^{-3}$)")
np.atleast_1d(axes)[0].legend(fontsize=6)
fig.suptitle(f"NLD {N_SEASON} — the plateau scan behind the field-capacity target")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "nld_plateau_scan.svg", dpi=300, bbox_inches="tight")
plt.show()


# %%
# ── 4. NLD: retention curves before and after, and the two files ─────────────
## Supplementary soil retention curve corrections
# The observed range the curves have to describe, drawn on the same axes as the
# curves: the averaged probe series, min to max, at the pF the target sits on.
n_obs_lo, n_obs_hi = n_pooled["SM_obs"].min(), n_pooled["SM_obs"].max()

n_ret_raw, n_cond_raw, n_ret_cor, n_cond_cor = {}, {}, {}, {}
for g in N_GEOMS:
    sfx = GEOMETRIES["NLD"][g]["suffix"]
    n_ret_raw[g], n_cond_raw[g] = write_soil(n_vg[g], f"netherlands_soil_MLWB{sfx}_wosten.yaml")
    n_ret_cor[g], n_cond_cor[g] = write_soil(n_vg_corr[g], f"netherlands_soil_MLWB{sfx}.yaml")
    for layer, target in N_FC_TARGET[g].items():
        sc.check_targets(n_ret_cor[g], [layer], target)

    # How far apart the two curves are, so a panel where they coincide is not read as
    # a missing series. On the 2-layer profile the merged 0-40 cm layer comes out of
    # the pedotransfer function already at the target, so its two curves overlap.
    for key in n_ret_raw[g]:
        gap = np.abs(np.asarray(n_ret_raw[g][key]) - np.asarray(n_ret_cor[g][key]))
        print(f"  {g} {key[0]:.0f}-{key[1]:.0f} cm: max |raw - corrected| over the pF "
              f"grid {gap.max():.5f} at pF {sc.PF_GRID[gap.argmax()]}")

    layers = list(n_ret_raw[g])
    fig, axes = plt.subplots(1, len(layers), figsize=(7, 2), sharey=True)
    for ax, key in zip(np.atleast_1d(axes), layers):
        ax.plot(sc.PF_GRID, n_ret_raw[g][key], marker="o",
                **plotstyle.before_after("before"), label="PTF raw")
        ax.plot(sc.PF_GRID, n_ret_cor[g][key], marker="o", markersize=3,
                **plotstyle.before_after("after"), label="FC corrected")
        # the probe range and the target only go on panels a probe observes
        if key in N_FC_TARGET[g]:
            ax.fill_between([sc.PF_FIELD_CAPACITY - 0.15, sc.PF_FIELD_CAPACITY + 0.15],
                            n_obs_lo, n_obs_hi, color="black", alpha=0.15, label="probe range")
            ax.plot(sc.PF_FIELD_CAPACITY, N_FC_TARGET[g][key], marker="*", markersize=6,
                    color="black", linestyle="none", label="FC target")
        ax.vlines([sc.PF_SATURATION, sc.PF_FIELD_CAPACITY, sc.PF_WILTING_POINT],
                  ymin=0, ymax=0.5, colors="k", linestyles="dashed", alpha=0.5)
        ax.set_xlim(-1, 6)
        ax.set_ylim(0, 0.5)
        ax.set_xlabel("pF (suction)")
        ax.set_title(f"{key[0]:.0f}-{key[1]:.0f} cm"
                     + ("" if key in N_FC_TARGET[g] else "  (no measurements)"))
    np.atleast_1d(axes)[0].set_ylabel("VWC (cm$^3$ cm$^{-3}$)")
    np.atleast_1d(axes)[0].legend(fontsize=8, bbox_to_anchor=(2.9,1.4), ncol=4, frameon=False)
    fig.subplots_adjust(wspace = 0.1)
    fig.savefig(OUTPUT_DIR / f"nld_retention_curves{sfx}.svg", dpi=300, bbox_inches="tight")
    plt.show()


# %%
# ── 5. IND: one field texture measurement, assigned to every layer ───────────
# 97% sand, 2% silt, 1% clay, 0.69% soil organic carbon, pH 7. There is no horizon
# data, so the layers differ only in bulk density and the topsoil flag.
#
# Assumptions with no measurement behind them:
#   - bulk density: 1.533 g/cm3 over 0-30 cm, 1.75 in the pan (consistent with a
#     penetrometer resistance above 1.5 MPa). Section 6 prints how far theta_s and
#     k_sat move over +/-0.15 g/cm3.
#   - organic matter from organic carbon via the Van Bemmelen factor 1.724 (Van
#     Bemmelen 1890; reviewed in Pribyl 2010, Geoderma 156(3-4):75-83), so 0.69%
#     SOC -> 1.19% OM.
#   - C:N of soil organic matter 9.0, carried over from the earlier inline profile.
#
# The pan is represented by its bulk density alone, through Wosten. Nothing is
# added on top: it restricts drainage without stopping it, and water reaching 30 cm
# still leaves the profile. An earlier version dropped the pan conductivity by 4-6
# log10, which pinned the probe layer at saturation for the whole season. A scan of
# that shift against the probe series put the best fit at no shift at all. The
# penetrometer constrains root penetration, already carried by RDMCR = 30 cm; it
# says nothing about hydraulic conductivity, and a compacted sand can impede roots
# while still passing water.
I_BOUNDS = GEOMETRIES["IND"][I_GEOMETRY]["bounds"]
I_RHOD = GEOMETRIES["IND"][I_GEOMETRY]["rhod"]
I_SUFFIX = GEOMETRIES["IND"][I_GEOMETRY]["suffix"]

i_layers = pd.DataFrame([
    {"layer_top": top, "layer_bottom": bottom, "bulk_density": I_RHOD[i],
     "clay_pct": 1.0, "silt_pct": 2.0, "om_pct": 0.69 * 1.724,
     "CNRatioSOMI": 9.0, "Soil_pH": 7.0, "theta_r": sc.THETA_R_PTF, "topsoil": top == 0}
    for i, (top, bottom) in enumerate(zip(I_BOUNDS[:-1], I_BOUNDS[1:]))
])
i_vg = sc.vg_from_layers(i_layers, keep=["layer_top", "layer_bottom", "topsoil",
                                         "bulk_density", "om_pct", "CNRatioSOMI", "Soil_pH"])
i_vg.to_csv(OUTPUT_DIR / f"ind_van_genuchten_params{I_SUFFIX}.csv", index=False)
print(i_vg.round(4).to_string(index=False))

print(f"\npan k_sat from bulk density alone: {i_vg.iloc[-1]['k_sat']:.1f} cm/day against "
      f"{i_vg.iloc[0]['k_sat']:.1f} above it "
      f"(factor {i_vg.iloc[0]['k_sat'] / i_vg.iloc[-1]['k_sat']:.1f} slower)")

# bulk density is the unmeasured input; theta_s and k_sat are what it moves most
BD_STEP = 0.15
print(f"\n{'layer':>12s} {'BD':>6s} {'theta_s':>9s} {'k_sat cm/d':>11s}")
for row in i_layers.itertuples():
    for bd in (row.bulk_density - BD_STEP, row.bulk_density, row.bulk_density + BD_STEP):
        mark = " <- assumed" if bd == row.bulk_density else ""
        print(f"{row.layer_top:>4.0f}-{row.layer_bottom:<3.0f} cm  {bd:6.2f} "
              f"{sc.wosten_theta_s(row.clay_pct, bd, row.silt_pct, row.om_pct, row.topsoil):9.3f} "
              f"{sc.wosten_k_sat(row.clay_pct, bd, row.silt_pct, row.om_pct, row.topsoil):11.1f}{mark}")


# %%
# ── 6. IND: the probe, and the field-capacity correction ─────────────────────────
# One probe depth (20 cm), the same soil under every treatment, so all six
# treatments are scanned. Readings are averaged per treatment per day into one
# observed series, the series 05.2 scores the 0-30 cm profile mean against.
I_SEASON = "2024-2025"
i_sm = pd.read_csv(DATA / "obs" / "IND_SM_obs.csv", parse_dates=["day"])
i_sm = i_sm[i_sm["season"].astype(str) == I_SEASON]
i_pooled = i_sm.groupby(["treatment", "day"])["SM_obs"].mean().reset_index()
print(i_pooled.groupby("treatment")["SM_obs"].agg(["count", "min", "mean", "max"]).round(3))
i_scan = plateau_scan({t: g.set_index("day")["SM_obs"].sort_index()
                       for t, g in i_pooled.groupby("treatment")})

# Manual field-capacity target from the probe series, a judgement call.
#   I_FC_TARGET  the plateau the scan above lands on. Cross-check, not input: the
#                earlier single-layer fit put SMFCF at 0.294 on the same series.
# Only alpha is adjusted; theta_r stays at 0.01 and WP is a curve read-off.
I_FC_TARGET = 0.30
# Applied above the pan only. The probe sits at 20 cm and observes nothing below
# the root base, so the pan keeps its raw Wosten curve.
i_corrected_layers = list(zip(I_BOUNDS[:-2], I_BOUNDS[1:-1]))
i_vg_corr = sc.correct_layers(i_vg, i_corrected_layers, I_FC_TARGET)

print(f"\ntarget FC {I_FC_TARGET} (pF 2.0) applied to "
      + ", ".join(f"{t:.0f}-{b:.0f} cm" for t, b in i_corrected_layers))
for i in range(len(i_corrected_layers)):
    print(f"  {i_vg.loc[i,'layer_top']:.0f}-{i_vg.loc[i,'layer_bottom']:.0f} cm: "
          f"alpha {i_vg.loc[i,'alpha']:.4f} -> {i_vg_corr.loc[i,'alpha']:.4g}, "
          f"theta_r {i_vg.loc[i,'theta_r']:.4f} -> {i_vg_corr.loc[i,'theta_r']:.4g}")
# theta_r, n, theta_s, lambda and k_sat retain their raw PTF values.


# FIGURE: the same scan at IND, drawn against the assumed FC target.
SCAN_FIGSIZE = (12, 3.2)
SCAN_YLIM = (0.05, 0.40)
SCAN_SERIES = dict(color="black", lw=0.9)
SCAN_EVENT = dict(color="0.7", lw=0.8, linestyle="dashed")
SCAN_PLATEAU = dict(marker="o", markersize=5, color="black", linestyle="none")
SCAN_TARGET = dict(color="black", lw=1.0, linestyle="dotted")

print("\nscan rows per treatment (events, of which reached a plateau):")
print(i_scan.assign(hit=i_scan["plateau_SM"].notna())
      .groupby("group").agg(events=("event", "size"), plateaus=("hit", "sum")).to_string())
_missing = sorted(set(i_pooled["treatment"]) - set(i_scan["group"]))
print(f"treatments with no wetting event at all: {_missing or 'none'}")

_i_groups = sorted(i_pooled["treatment"].unique())
fig, axes = plt.subplots(1, len(_i_groups), figsize=SCAN_FIGSIZE, sharey=True)
for ax, t in zip(np.atleast_1d(axes), _i_groups):
    g = i_pooled[i_pooled["treatment"] == t].sort_values("day")
    ax.plot(g["day"], g["SM_obs"], label=f"{20} cm probe", **SCAN_SERIES)
    sc_t = i_scan[i_scan["group"] == t]
    for d in sc_t["event"]:
        ax.axvline(d, **SCAN_EVENT)
    hit = sc_t.dropna(subset=["plateau_SM"])
    ax.plot(hit["plateau_day"], hit["plateau_SM"], label="plateau", **SCAN_PLATEAU)
    ax.axhline(I_FC_TARGET, label=f"FC target {I_FC_TARGET:.2f}", **SCAN_TARGET)
    ax.set_ylim(*SCAN_YLIM)
    ax.set_title(t, fontsize=8)
    ax.tick_params(axis="x", labelrotation=45, labelsize=6)
np.atleast_1d(axes)[0].set_ylabel("SM at 20 cm (cm$^3$ cm$^{-3}$)")
np.atleast_1d(axes)[0].legend(fontsize=6)
fig.suptitle(f"IND {display_season(I_SEASON)} — the plateau scan behind the field-capacity target")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / f"ind_plateau_scan{I_SUFFIX}.svg", dpi=300, bbox_inches="tight")
plt.show()


# %%
# ── 7. IND: retention curves before and after, and the two files ─────────────
i_ret_raw, i_cond_raw = write_soil(i_vg, f"india_soil_MLWB{I_SUFFIX}_wosten.yaml")
i_ret_cor, i_cond_cor = write_soil(i_vg_corr, f"india_soil_MLWB{I_SUFFIX}.yaml")
sc.check_targets(i_ret_cor, i_corrected_layers, I_FC_TARGET)

# Plot-ready retention curves for Figures S11-S12. S11 uses the NLD 3-layer
# geometry only; S12 uses the currently selected IND geometry.
_manuscript_source = ROOT / "manuscript_outputs" / "source_data"
_manuscript_source.mkdir(parents=True, exist_ok=True)
_retention_rows = []
_i_fc = {key: I_FC_TARGET for key in i_corrected_layers}
_i_wp = {}  # retained CSV column; no WP target in the current method
for _site, _geometry, _soil, _curves, _obs_lo, _obs_hi, _fc, _wp in (
        ("NLD", "3layer", "PTF raw", n_ret_raw["3layer"],
         n_obs_lo, n_obs_hi, N_FC_TARGET["3layer"], {}),
        ("NLD", "3layer", "FC corrected", n_ret_cor["3layer"],
         n_obs_lo, n_obs_hi, N_FC_TARGET["3layer"], {}),
        ("IND", I_GEOMETRY, "PTF raw", i_ret_raw,
         i_pooled["SM_obs"].min(), i_pooled["SM_obs"].max(), _i_fc, _i_wp),
        ("IND", I_GEOMETRY, "FC corrected", i_ret_cor,
         i_pooled["SM_obs"].min(), i_pooled["SM_obs"].max(), _i_fc, _i_wp)):
    for (_top, _bottom), _values in _curves.items():
        for _pf, _value in zip(sc.PF_GRID, _values):
            _retention_rows.append({
                "site": _site, "geometry": _geometry, "soil": _soil,
                "layer_top": _top, "layer_bottom": _bottom,
                "pF": float(_pf), "vwc": float(_value),
                "obs_lo": _obs_lo, "obs_hi": _obs_hi,
                "fc_target": _fc.get((_top, _bottom), np.nan),
                "wp_target": _wp.get((_top, _bottom), np.nan),
            })
pd.DataFrame(_retention_rows).to_csv(
    _manuscript_source / "s11_s12_retention_curves.csv", index=False)

i_obs_lo, i_obs_hi = i_pooled["SM_obs"].min(), i_pooled["SM_obs"].max()

# As in section 4: the pan is not corrected, so its two curves coincide.
for key in i_ret_raw:
    gap = np.abs(np.asarray(i_ret_raw[key]) - np.asarray(i_ret_cor[key]))
    print(f"  {key[0]:.0f}-{key[1]:.0f} cm: max |raw - corrected| over the pF grid "
          f"{gap.max():.5f} at pF {sc.PF_GRID[gap.argmax()]}")

layers = list(i_ret_raw)
fig, axes = plt.subplots(1, len(layers), figsize=(5, 2), sharey=True)
for ax, key in zip(np.atleast_1d(axes), layers):
    ax.plot(sc.PF_GRID, i_ret_raw[key], marker="o", **plotstyle.before_after("before"),
             label="PTF raw", markersize=3)
    ax.plot(sc.PF_GRID, i_ret_cor[key], marker="o", **plotstyle.before_after("after"),
             label="FC corrected", markersize=3)
    ax.fill_between([sc.PF_FIELD_CAPACITY - 0.15, sc.PF_FIELD_CAPACITY + 0.15],
                    i_obs_lo, i_obs_hi, color="black", alpha=0.15, label="probe range")
    if key in i_corrected_layers:
        ax.plot(sc.PF_FIELD_CAPACITY, I_FC_TARGET,
                marker="*", markersize=6, color="black", linestyle="none", label="assumed FC target")
    ax.vlines([sc.PF_SATURATION, sc.PF_FIELD_CAPACITY, sc.PF_WILTING_POINT],
              ymin=0, ymax=0.5, colors="k", linestyles="dashed", alpha=0.5)
    ax.set_xlim(-1, 6)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("pF (suction)")
    ax.set_title(f"{key[0]:.0f}-{key[1]:.0f} cm" + ("  (compacted layer)" if key == layers[-1] else ""))
np.atleast_1d(axes)[0].set_ylabel("VWC (cm$^3$ cm$^{-3}$)")
np.atleast_1d(axes)[0].legend(fontsize=8)
np.atleast_1d(axes)[0].legend(fontsize=8, bbox_to_anchor=(1.8,1.4), ncol=2, frameon=False)

fig.subplots_adjust(wspace = 0.05)
fig.savefig(OUTPUT_DIR / f"ind_retention_curves{I_SUFFIX}.svg", dpi=300, bbox_inches="tight")
plt.show()


# %%
# ── 8. What was written, read back off the tables ────────────────────────────
# SM0, SMFCF, SMW and CRAIRC are read-offs from the tables, not parameters. CRAIRC
# moves a long way under a correction and nothing measured constrains it. K0 is the
# wettest grid node (pF -1, suction 0.1 cm), so it sits below SurfaceConductivity,
# which is k_sat itself.
rows = []
for g in N_GEOMS:
    rows.append(sc.table_summary(n_ret_raw[g], n_cond_raw[g]).assign(
        site="NLD", geometry=g, soil="PTF raw"))
    rows.append(sc.table_summary(n_ret_cor[g], n_cond_cor[g]).assign(
        site="NLD", geometry=g, soil="corrected"))
rows.append(sc.table_summary(i_ret_raw, i_cond_raw).assign(
    site="IND", geometry=I_GEOMETRY, soil="PTF raw"))
rows.append(sc.table_summary(i_ret_cor, i_cond_cor).assign(
    site="IND", geometry=I_GEOMETRY, soil="corrected"))
summary = pd.concat(rows, ignore_index=True)[
    ["site", "geometry", "soil", "layer", "SM0", "SMFCF", "SMW", "CRAIRC", "K0"]]
print(summary.round(4).to_string(index=False))
summary.to_csv(OUTPUT_DIR / "soil_table_summary.csv", index=False)

# %%
