# No modelling without experimentation

Analysis code for the manuscript's main figures, copied from the current working files in `ch3_model` on 13 October 2026.

This is a code-only copy: experimental inputs, model configuration files, prepared CSVs, cached fits, and generated figures are not included. The scripts retain their original `# %%` cells and calculations. Historical alternatives, sensitivity-analysis scripts, supplementary assembly, and unrelated diagnostics are excluded from the file selection. Some retained analysis scripts also contain their own supplementary or diagnostic cells.

## Setup and inputs

Run commands from this repository's root, with Python and R available. Install the Python dependencies with:

```sh
python -m pip install -r requirements.txt
```

The R scripts use `data.table`, `readxl`, and `ggplot2`:

```r
install.packages(c("data.table", "readxl", "ggplot2"))
```

Supply the original inputs in the same relative locations before running the analyses:

- `data/obs/`: `MainTable.xlsx`, emergence counts, establishment records, dry matter, LAI, SLA, and soil moisture used by the retained scripts.
- `data/Weather/`: the NLD, IND, RSA weather workbooks and the two CAN weather CSVs.
- `data/params/experiments.yaml`, `data/crops_calibration/`, `data/agro/`, `data/soil/`, and the WOFOST configurations in `data/conf/`.
- `data/LICOR_field_controlled_2021_2024_2025.csv` for the LICOR fits.

`00_LICOR_data.R` compiles the LICOR CSV from external measurement exports and trial-layout workbooks. Update its `OD` input directory for your machine; skip this script if the compiled CSV is already available. `05.0_WLD_soil_profiles.py` starts from the supplied `data/soil/soil_profiles_raw.csv` and soil-moisture observations.

The models require PCSE with the WOFOST 7.2/7.3 model classes used in the scripts. NLopt performs calibration; SALib is imported by the shared plotting module. The manuscript style uses Times New Roman, which should be installed if that font is required.

## Analysis order

Each command can also be run through the script's visible cells.

```sh
python scripts/00_summarise_obs.py
python scripts/01_weather.py
python scripts/02.1_pp_emergence.py
python scripts/02.6_pp_phenology_thermal.py
python scripts/02.7_pp_photoperiod.py
python scripts/03.1_p_growth_corrections.py
python scripts/03.3_p_growth_partition.py
Rscript scripts/00_LICOR_wofost_respiration.R
python scripts/03.2_p_growth_AMAX.py
python scripts/05.0_WLD_soil_profiles.py
python scripts/05.1_WLD_NLD.py
python scripts/05.2_WLD_IND_opt.py
python scripts/06.1_manuscript_main_figures.py
```

Growth partitioning precedes the complete AMAX script because the latter's IND temperature diagnostic reads the fitted partition parameters. The main AMAX fit itself uses the LICOR exports.

| Main output | Analysis script |
| --- | --- |
| Daylength plot | `01_weather.py` prepares the CSV; `06.1` plots it |
| Figure 1 | `02.1_pp_emergence.py` prepares the comparison; `06.1` plots it |
| Figure 2 | `02.7_pp_photoperiod.py` calculates the comparison; `06.1` plots it |
| Figure 3 | `03.1_p_growth_corrections.py` calculates and plots it |
| Figure 4 | `00_LICOR_wofost_respiration.R` and `03.2_p_growth_AMAX.py` calculate it; `03.2` plots it |
| Figure 5 | `03.3_p_growth_partition.py` prepares the allocation and growth CSVs; `06.1` plots it |
| Figure 6 | `05.0_WLD_soil_profiles.py` and `05.1_WLD_NLD.py` prepare the soil and growth results; `06.1` plots it |
| Figure 7 | `05.2_WLD_IND_opt.py` prepares the yield comparison; `06.1` plots it |

`06.1` reads the persisted analysis CSVs under `output/`, `manuscript_outputs/source_data/`, and `manuscript_outputs/numbers/`. Figures 3 and 4 and their number summaries are produced by `03.1` and `03.2`; `06.1` reads those summaries. Figures are written under `manuscript_outputs/figures/main/`, with the daylength SVG under `manuscript_outputs/figures/`.

## Verification of this copy

Python syntax and local-module dependencies were checked, and script contents were compared byte for byte with the source working files. Analyses and figure rendering were not executed. Package versions and execution on a fresh machine have not been validated.
