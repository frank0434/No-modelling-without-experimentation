"""
optimization/runner.py
----------------------
PhenologySimulationRunner: run WOFOST73_PP for one parameter set across
one experiment dict and return a tidy DataFrame of daily outputs.
"""

from __future__ import annotations
import sys
import pathlib
import yaml
import pandas as pd
from pcse.models import Wofost73_PP

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from utils.loaders import (
    load_weather,
    load_soil,
    load_site_data,
    load_agro,
    load_crop_data,
    build_parameters,
)

_LOC_AGRO_FILE = {
    "IND": "india_seasons.yaml",
    "NLD": "netherlands_seasons.yaml",
    "RSA": "southafrica_seasons.yaml",
}

OUTPUT_VARS = ["DVS", "LAI", "TAGP", "TWSO"]


class PhenologySimulationRunner:
    """Run WOFOST for one experiment + override set; return daily output."""

    def __init__(self, crop_dir: str = "data/crops_calibration"):
        self._cropd = load_crop_data(crop_dir)
        self._agro_run_days: dict[tuple, int | None] = {}

    def _get_run_days(self, location: str, season: str) -> int | None:
        key = (location, season)
        if key not in self._agro_run_days:
            fname = _LOC_AGRO_FILE[location]
            path = ROOT / "data" / "agro" / fname
            with open(path) as f:
                data = yaml.safe_load(f)
            self._agro_run_days[key] = data["seasons"].get(season, {}).get("run_days")
        return self._agro_run_days[key]

    def run(self, experiment: dict, overrides: dict) -> pd.DataFrame:
        """Run one simulation and return a DataFrame with daily outputs.

        Parameters
        ----------
        experiment :
            One entry from experiments.yaml (dict with location, season,
            cultivar, weather_file, soil_file, WAV keys).
        overrides :
            Combined dict of active-parameter values and fixed overrides.

        Returns
        -------
        DataFrame with columns: day, DVS, LAI, TAGP, TWSO,
        plus experiment, location, season, cultivar.
        """
        loc = experiment["location"]
        season = experiment["season"]
        cultivar = experiment["cultivar"]

        wdp = load_weather(experiment["weather_file"])
        soild = load_soil(experiment["soil_file"])
        sited = load_site_data(experiment["WAV"], experiment["CO2"])
        agro = load_agro(loc, season, cultivar)
        params = build_parameters(self._cropd, soild, sited, cultivar, overrides)

        model = Wofost73_PP(params, wdp, agro)
        run_days = self._get_run_days(loc, season)
        if run_days:
            model.run(run_days)
        else:
            model.run_till_terminate()

        df = pd.DataFrame(model.get_output())
        df["day"] = pd.to_datetime(df["day"])
        df["experiment"] = experiment["name"]
        df["location"] = loc
        df["season"] = season
        df["cultivar"] = cultivar

        keep = ["day", "experiment", "location", "season", "cultivar"] + [
            v for v in OUTPUT_VARS if v in df.columns
        ]
        return df[keep]
