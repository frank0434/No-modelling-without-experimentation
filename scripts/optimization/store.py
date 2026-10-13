"""
optimization/store.py
---------------------
OptimizationArtifactStore: persist and reload stage outputs, traces,
diagnostics, and config snapshots.

Artifact layout::

    <output_root>/<run_id>/<target>/
        config.json
        stage1_global/
            result.json
            trace.parquet
        stage2_local/
            result.json
            trace.parquet
        failures.jsonl
"""

from __future__ import annotations
import json
import pathlib
import numpy as np
import pandas as pd
from .config import OptimizationConfig
from .optimizer import OptimizationResult


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


class OptimizationArtifactStore:
    """Write and read optimization artifacts for one target."""

    def __init__(self, config: OptimizationConfig):
        self._config = config
        self._root = (
            pathlib.Path(config.output_root)
            / config.run_id
            / config.target
        )
        self._root.mkdir(parents=True, exist_ok=True)

    # ── Config snapshot ────────────────────────────────────────────────────

    def save_config(self) -> None:
        cfg = self._config
        snap = {
            "target": cfg.target,
            "location": cfg.location,
            "cultivar": cfg.cultivar,
            "seasons": list(cfg.seasons),
            "active_params": [
                {"name": s.name, "lower": s.lower, "upper": s.upper,
                 "init": s.init, "step": s.step}
                for s in cfg.active_params
            ],
            "fixed_overrides": cfg.fixed_overrides,
            "crop_dir": cfg.crop_dir,
            "run_id": cfg.run_id,
            "max_evals_stage1": cfg.max_evals_stage1,
            "max_evals_stage2": cfg.max_evals_stage2,
            "ftol_rel": cfg.ftol_rel,
            "cache": cfg.cache,
            "seed": cfg.seed,
        }
        (self._root / "config.json").write_text(json.dumps(snap, indent=2))

    # ── Stage results ──────────────────────────────────────────────────────

    def save_stage(self, result: OptimizationResult, trace: list[dict]) -> None:
        stage_dir = self._root / f"{result.stage}_global" if result.stage == "stage1" \
            else self._root / f"{result.stage}_local"
        stage_dir.mkdir(exist_ok=True)

        result_doc = {
            "stage": result.stage,
            "x_best": result.x_best.tolist(),
            "f_best": result.f_best,
            "n_evals": result.n_evals,
            "overrides": result.overrides,
            "converged": result.converged,
        }
        (stage_dir / "result.json").write_text(
            json.dumps(result_doc, indent=2, cls=_NumpyEncoder)
        )

        if trace:
            rows = [
                {
                    "eval_idx": i,
                    "rmse": r["rmse"],
                    "sse": r["sse"],
                    "n": r["n"],
                    **{f"x_{k}": v for k, v in r["overrides"].items()},
                }
                for i, r in enumerate(trace)
            ]
            pd.DataFrame(rows).to_parquet(stage_dir / "trace.parquet", index=False)

    def save_failure(self, experiment_name: str, error: str) -> None:
        with open(self._root / "failures.jsonl", "a") as f:
            json.dump({"experiment": experiment_name, "error": error}, f)
            f.write("\n")

    # ── Resume support ─────────────────────────────────────────────────────

    def load_stage(self, stage: str) -> OptimizationResult | None:
        stage_dir = self._root / (
            "stage1_global" if stage == "stage1" else "stage2_local"
        )
        result_path = stage_dir / "result.json"
        if not result_path.exists():
            return None
        doc = json.loads(result_path.read_text())
        return OptimizationResult(
            stage=doc["stage"],
            x_best=np.array(doc["x_best"]),
            f_best=doc["f_best"],
            n_evals=doc["n_evals"],
            overrides=doc["overrides"],
            converged=doc["converged"],
        )

    def stage_done(self, stage: str) -> bool:
        return self.load_stage(stage) is not None
