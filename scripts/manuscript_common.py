"""Constants shared by the flat manuscript assembly scripts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_ROOT = ROOT / "manuscript_outputs"
SOURCE_DIR = MANUSCRIPT_ROOT / "source_data"
MAIN_FIG_DIR = MANUSCRIPT_ROOT / "figures" / "main"
SUPP_FIG_DIR = MANUSCRIPT_ROOT / "figures" / "supplementary"
TABLE_DIR = MANUSCRIPT_ROOT / "tables"
NUMBER_DIR = MANUSCRIPT_ROOT / "numbers"

OUTPUT_DIRECTORIES = [
    SOURCE_DIR, MAIN_FIG_DIR, SUPP_FIG_DIR, TABLE_DIR, NUMBER_DIR,
]

TREATMENT_ORDER = [
    "Full_0 Days", "Full_6 Days", "Full_12 Days",
    "Reduced_0 Days", "Reduced_6 Days", "Reduced_12 Days",
]
TREATMENT_SHORT = {
    "Full_0 Days": "F0", "Full_6 Days": "F6", "Full_12 Days": "F12",
    "Reduced_0 Days": "R0", "Reduced_6 Days": "R6",
    "Reduced_12 Days": "R12",
}
# Start with black and grey; add blue for a third series.
# Country colours are only needed when countries share an axis.
PALETTE = {
    "black": "black", "grey": "0.6", "blue": "#000080",
    "orange": "#FF4F00", "green": "#007A4D", "red":"#FF0000"
}
SITE_COLOR = {
    "NLD": PALETTE["orange"], "IND": PALETTE["blue"],
    "RSA": PALETTE["green"], "CAN": PALETTE["red"],
}
SITE_MARKER = {"NLD": "o", "IND": "s", "RSA": "^", "CAN": "D"}
SEASON_MARKER = {
    "1992": "o", "1993": "s",
    "2022": "o", "2022-2023": "o", "2024": "s", "2024-2025": "^",
    "2025": "D", "2025-A": "D", "2025-S": "v",
}
DAYLENGTH_YEAR_LINE = {
    "1992": "-", "1993": "--", "2022": ":", "2024": "-", "2025": "--",
    "2025-A": "--", "2025-S": ":",
}
CULTIVAR_MARKER = {
    "Innovator": "o", "Kennebec": "s", "Mondial": "^",
    "Sifra": "D", "Désirée": "v",
}
DRY_SPELL_MARKER = {"0 Days": "o", "6 Days": "s", "12 Days": "^"}
DRY_SPELL_LINE = {"0 Days": "-", "6 Days": "--", "12 Days": ":"}
TREATMENT_COLOR = {
    treatment: PALETTE["black"] if treatment.startswith("Full_") else PALETTE["grey"]
    for treatment in TREATMENT_ORDER
}
TREATMENT_MARKER = {
    treatment: DRY_SPELL_MARKER[treatment.split("_")[1]]
    for treatment in TREATMENT_ORDER
}
TREATMENT_LINE = {
    treatment: DRY_SPELL_LINE[treatment.split("_")[1]]
    for treatment in TREATMENT_ORDER
}
HEATING_COLOR = {
    "AC": PALETTE["black"], "HE": PALETTE["grey"],
    "HTI": "#8B1C1C", "HB": PALETTE["orange"],
}
HEATING_MARKER = {"AC": "o", "HE": "o", "HTI": "^", "HB": "^"}
HEATING_FILLED = {"AC": True, "HE": False, "HTI": True, "HB": False}
HEATING_LINE = {"AC": "-", "HE": "-", "HTI": "--", "HB": "-."}
MODEL_COLOR = {
    "default": PALETTE["grey"], "intermediate": PALETTE["blue"],
    "final": PALETTE["black"],
}
REF_COLOR = "0.55"

LINE_STYLE = {
    "default": dict(color=MODEL_COLOR["default"], linestyle="--"),
    "before": dict(color=MODEL_COLOR["default"], linestyle="--"),
    "after": dict(color=MODEL_COLOR["final"], linestyle="-"),
    "observed": dict(
        color="black", marker="o", linestyle="none", markersize=3.5,
        markerfacecolor="white", markeredgewidth=0.7,
    ),
}

MANUSCRIPT_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "dejavuserif",
    "svg.fonttype": "none",
    "figure.dpi": 180,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 0.8,
}

SOURCE_KEYS = {
    "s08_sla_dvs.csv": ["site", "season", "day"],
    "fig02_photoperiod_comparison.csv": ["panel", "site", "season", "scenario"],
    "s01_compaction_profile.csv": ["depth"],
    "s02_emergence_observed.csv": ["site", "season", "group", "dap"],
    "s02_emergence_curves.csv": ["site", "season", "group", "model", "dap"],
    "s03_weather_temperature.csv": ["site", "season", "day", "variable"],
    "s05_teffmx_profiles.csv": ["site", "season", "field", "TEFFMX"],
    "s06_emergence_response_curves.csv": ["temperature", "rule"],
    "fig03_growth_corrections.csv": [
        "site", "season", "cultivar", "field", "treatment", "variable", "variant", "day",
    ],
    "fig04_amax.csv": ["panel", "series_kind", "source_id", "x"],
    "fig05_growth_timeseries.csv": [
        "site", "season", "variant", "variable", "series_kind", "DAE",
    ],
    "s10_partition_curves.csv": ["site", "table", "variant", "dvs"],
    "s11_nld_retention_curves.csv": [
        "site", "geometry", "soil", "layer_top", "layer_bottom", "pF",
    ],
    "s12_ind_retention_curves.csv": ["site", "season", "geometry", "soil", "layer_top", "layer_bottom", "pF"],
    "s11_s12_retention_curves.csv": [
        "site", "geometry", "soil", "layer_top", "layer_bottom", "pF",
    ],
    "fig06_nld_yield_pairs.csv": ["season", "soil", "treatment", "day"],
    "fig06_nld_yield_mae.csv": ["season", "soil"],
    "s13_s14_nld_timeseries.csv": [
        "site", "season", "treatment", "soil", "series_kind", "variable",
        "day", "depth",
    ],
    "s15_nld_sm_pairs.csv": [
        "site", "season", "soil", "treatment", "layer", "day",
    ],
    "s15_nld_sm_mae.csv": ["season", "soil"],
    "s16_s18_ind_sm_pairs.csv": ["site", "season", "treatment", "soil", "day"],
    "s18_ind_mae.csv": ["treatment", "soil"],
    "s17_ind_timeseries.csv": [
        "site", "season", "treatment", "soil", "series_kind", "variable", "day",
    ],
    "s19_ind_committed_timeseries.csv": [
        "site", "season", "treatment", "variant", "series_kind", "variable", "day",
    ],
    "table_s02_parameters.csv": ["site", "layer", "param", "dvs"],
}

S2_EXTERNAL_DEPENDENCY = (
    r"C:\Users\liu283\GitRepos\Ch2_WUR\scripts\0x_recheck_emergence.R "
    "and its NLD/RSA/IND emergence workbooks and CSV files"
)
