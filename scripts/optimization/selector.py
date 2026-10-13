"""
optimization/selector.py
------------------------
ExperimentSelector: load and validate target experiments + observed phenology
for one (location, cultivar, seasons) combination.
"""

from __future__ import annotations
import pandas as pd
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from utils.loaders import (
    load_experiments,
    load_obs_phenology,
    canonicalize_season,
)
from .config import OptimizationConfig


class ExperimentSelector:
    """Loads experiments and either loads or validates supplied observations for one target."""

    def __init__(self, config: OptimizationConfig, observations: pd.DataFrame | None = None):
        self._config = config
        self._experiments: list[dict] | None = None
        self._obs: pd.DataFrame | None = None
        self._provided_obs = None if observations is None else observations.copy()

    def experiments(self) -> list[dict]:
        if self._experiments is None:
            self._experiments = self._load_experiments()
        return self._experiments

    def obs(self) -> pd.DataFrame:
        if self._obs is None:
            self._obs = self._load_obs()
        return self._obs

    def _load_experiments(self) -> list[dict]:
        cfg = self._config
        canonical_seasons = {canonicalize_season(s) for s in cfg.seasons}
        all_exps = load_experiments(location=cfg.location)
        matched = [
            e for e in all_exps
            if e["cultivar"] == cfg.cultivar
            and canonicalize_season(e["season"]) in canonical_seasons
        ]
        if not matched:
            raise ValueError(
                f"No experiments found for {cfg.location}/{cfg.cultivar} "
                f"in seasons {list(cfg.seasons)}"
            )
        return matched

    def _load_obs(self) -> pd.DataFrame:
        cfg = self._config
        df = (load_obs_phenology(location=cfg.location) if self._provided_obs is None
              else self._provided_obs.copy())
        required = {"location", "season", "field", "cultivar", "day", "obs_dvs"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Provided observations missing columns: {sorted(missing)}")
        df = df[df["location"] == cfg.location].copy()
        df = df[df["cultivar"] == cfg.cultivar].copy()
        canonical_seasons = {canonicalize_season(s) for s in cfg.seasons}
        df = df[df["season"].map(canonicalize_season).isin(canonical_seasons)]
        df = df[df["field"] == cfg.obs_field].copy()
        df["day"] = pd.to_datetime(df["day"])
        df = df.dropna(subset=["day", "obs_dvs"])
        if df.empty:
            raise ValueError(
                f"No observed phenology for {cfg.location}/{cfg.cultivar} "
                f"in seasons {list(cfg.seasons)}"
            )
        return df.reset_index(drop=True)
