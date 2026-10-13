"""
optimization/objective.py
-------------------------
PhenologyObjective: NLOPT-callable loss function.
Computes RMSE as the scalar objective and captures diagnostics on each call.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from .config import OptimizationConfig
from .resolver import ParameterSetResolver
from .runner import PhenologySimulationRunner
from .selector import ExperimentSelector


class PhenologyObjective:
    """NLOPT objective function for phenology calibration.

    Call signature matches NLOPT's expected ``f(x, grad) -> float``.
    After optimization, ``trace`` holds every evaluated (x, loss, diagnostics).
    """

    def __init__(
        self,
        config: OptimizationConfig,
        resolver: ParameterSetResolver,
        runner: PhenologySimulationRunner,
        selector: ExperimentSelector,
    ):
        self._config = config
        self._resolver = resolver
        self._runner = runner
        self._selector = selector
        self.trace: list[dict] = []
        self._cache: dict[tuple, float] = {}

    def __call__(self, x: np.ndarray, grad: np.ndarray) -> float:
        key = tuple(x.tolist())
        if self._config.cache and key in self._cache:
            return self._cache[key]

        active_overrides = self._resolver.vector_to_overrides(x)
        overrides = {**self._config.fixed_overrides, **active_overrides}

        sim_frames = []
        for exp in self._selector.experiments():
            try:
                df = self._runner.run(exp, overrides)
                sim_frames.append(df)
            except Exception:
                if self._config.cache:
                    self._cache[key] = 1e9
                return 1e9

        if not sim_frames:
            return 1e9

        sim_all = pd.concat(sim_frames, ignore_index=True)
        obs_all = self._selector.obs()

        merged = pd.merge(
            obs_all[["location", "season", "cultivar", "day", "obs_dvs"]],
            sim_all[["location", "season", "cultivar", "day", "DVS"]],
            on=["location", "season", "cultivar", "day"],
            how="inner",
        )
        merged = merged.dropna(subset=["obs_dvs", "DVS"])

        if len(merged) < 2:
            return 1e9

        obs = merged["obs_dvs"].values
        sim = merged["DVS"].values
        residuals = sim - obs
        sse = float(np.sum(residuals ** 2))
        rmse_val = float(np.sqrt(np.mean(residuals ** 2)))
        mae_val = float(np.mean(np.abs(residuals)))

        record = {
            "x": x.copy(),
            "overrides": active_overrides,
            "rmse": rmse_val,
            "mae": mae_val,
            "sse": sse,
            "n": len(obs),
        }
        self.trace.append(record)

        if self._config.cache:
            self._cache[key] = rmse_val
        return rmse_val

    def evaluate(self, x: np.ndarray) -> dict:
        """Run the objective and return the full record (rmse, mae, sse, n) for one x.

        For diagnostic callers that want both metrics; ``__call__`` still returns the
        scalar rmse NLOPT minimises. A failed run yields rmse == mae == 1e9.
        """
        rmse = self(x, None)
        if self.trace and np.array_equal(self.trace[-1]["x"], x):
            return self.trace[-1]
        return {"x": x.copy(), "rmse": rmse, "mae": rmse, "sse": float("nan"), "n": 0}

    def best(self) -> dict | None:
        if not self.trace:
            return None
        return min(self.trace, key=lambda r: r["rmse"])
