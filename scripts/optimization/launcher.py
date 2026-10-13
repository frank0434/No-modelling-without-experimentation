"""
optimization/launcher.py
------------------------
OptimizationLauncher: orchestrate a single local run or a multi-job HPC fanout.

Local run:
    launcher = OptimizationLauncher(config)
    r1, r2 = launcher.run()

HPC run:
    OptimizationLauncher.run_from_manifest("manifest.json", target="IND-Kennebec")

After all HPC jobs finish, call:
    OptimizationLauncher.aggregate(output_root, run_id) -> pd.DataFrame
"""

from __future__ import annotations
import json
import pathlib
import sys
import numpy as np
import pandas as pd

_HERE = pathlib.Path(__file__).resolve().parent
ROOT = _HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from .config import OptimizationConfig, ParameterSpec
from .resolver import ParameterSetResolver
from .selector import ExperimentSelector
from .runner import PhenologySimulationRunner
from .objective import PhenologyObjective
from .optimizer import TwoStageNLOptimizer, OptimizationResult
from .store import OptimizationArtifactStore


class OptimizationLauncher:
    """Wires all components together and runs one optimization job."""

    def __init__(self, config: OptimizationConfig,
                 observations: pd.DataFrame | None = None):
        """Build one job; ``observations`` optionally supplies its explicit DVS dataframe."""
        self._config = config
        self._store = OptimizationArtifactStore(config)
        self._resolver = ParameterSetResolver(config)
        self._selector = ExperimentSelector(config, observations=observations)
        self._runner = PhenologySimulationRunner(config.crop_dir)
        self._objective = PhenologyObjective(
            config, self._resolver, self._runner, self._selector
        )
        self._opt = TwoStageNLOptimizer(config, self._resolver, self._objective)

    def run(self, resume: bool = True) -> tuple[OptimizationResult, OptimizationResult]:
        """Run stage 1 then stage 2, with optional resume from saved artifacts.

        Parameters
        ----------
        resume :
            If True and a stage is already complete on disk, skip it and load
            the saved result instead of re-running.

        Returns
        -------
        (stage1_result, stage2_result)
        """
        self._store.save_config()

        if resume and self._store.stage_done("stage1"):
            r1 = self._store.load_stage("stage1")
            print(f"[{self._config.target}] stage1 loaded from disk (f={r1.f_best:.6f})")
        else:
            print(f"[{self._config.target}] stage1 starting ({self._config.max_evals_stage1} max evals)")
            r1 = self._opt.run_stage1_global()
            stage1_trace = self._objective.trace[:]
            self._store.save_stage(r1, stage1_trace)
            print(f"[{self._config.target}] stage1 done: f={r1.f_best:.6f}, evals={r1.n_evals}, converged={r1.converged}")

        if resume and self._store.stage_done("stage2"):
            r2 = self._store.load_stage("stage2")
            print(f"[{self._config.target}] stage2 loaded from disk (f={r2.f_best:.6f})")
        else:
            print(f"[{self._config.target}] stage2 starting ({self._config.max_evals_stage2} max evals)")
            trace_before = len(self._objective.trace)
            r2 = self._opt.run_stage2_local(r1.x_best)
            stage2_trace = self._objective.trace[trace_before:]
            self._store.save_stage(r2, stage2_trace)
            print(f"[{self._config.target}] stage2 done: f={r2.f_best:.6f}, evals={r2.n_evals}, converged={r2.converged}")

        return r1, r2

    # ── HPC support ────────────────────────────────────────────────────────

    @staticmethod
    def run_from_manifest(manifest_path: str | pathlib.Path, target: str) -> None:
        """Load a manifest JSON and run the job for *target*.

        Intended for HPC job scripts:
            python -c "from optimization.launcher import OptimizationLauncher; \\
                       OptimizationLauncher.run_from_manifest('manifest.json', 'IND-Kennebec')"
        """
        configs = _load_manifest(manifest_path)
        matched = [c for c in configs if c.target == target]
        if not matched:
            raise ValueError(f"Target '{target}' not found in {manifest_path}")
        launcher = OptimizationLauncher(matched[0])
        launcher.run(resume=True)

    @staticmethod
    def aggregate(
        output_root: str | pathlib.Path,
        run_id: str,
    ) -> pd.DataFrame:
        """Collect stage-2 best results from all targets into one summary table.

        Parameters
        ----------
        output_root :
            Same value used in OptimizationConfig.output_root.
        run_id :
            Version tag, e.g. ``"v1"``.

        Returns
        -------
        DataFrame with one row per target and columns:
        target, f_best, n_evals, converged, <override columns>.
        """
        root = pathlib.Path(output_root) / run_id
        rows = []
        for target_dir in sorted(root.iterdir()):
            if not target_dir.is_dir():
                continue
            result_path = target_dir / "stage2_local" / "result.json"
            if not result_path.exists():
                result_path = target_dir / "stage1_global" / "result.json"
            if not result_path.exists():
                continue
            doc = json.loads(result_path.read_text())
            row = {
                "target": target_dir.name,
                "stage": doc["stage"],
                "f_best": doc["f_best"],
                "n_evals": doc["n_evals"],
                "converged": doc["converged"],
            }
            row.update(doc["overrides"])
            rows.append(row)
        return pd.DataFrame(rows)


# ── Manifest helpers ──────────────────────────────────────────────────────────

def _load_manifest(path: str | pathlib.Path) -> list[OptimizationConfig]:
    """Load a list of OptimizationConfig objects from a JSON manifest file."""
    doc = json.loads(pathlib.Path(path).read_text())
    configs = []
    for entry in doc["jobs"]:
        specs = [
            ParameterSpec(
                name=s["name"],
                lower=s["lower"],
                upper=s["upper"],
                init=s["init"],
                step=s.get("step", 0.0),
            )
            for s in entry["active_params"]
        ]
        configs.append(OptimizationConfig(
            target=entry["target"],
            location=entry["location"],
            cultivar=entry["cultivar"],
            seasons=tuple(entry["seasons"]),
            active_params=tuple(specs),
            fixed_overrides=entry.get("fixed_overrides", {}),
            crop_dir=entry.get("crop_dir", "data/crops_calibration"),
            run_id=entry.get("run_id", "v1"),
            output_root=pathlib.Path(entry.get("output_root", "output/optimization")),
            max_evals_stage1=entry.get("max_evals_stage1", 1000),
            max_evals_stage2=entry.get("max_evals_stage2", 500),
            ftol_rel=entry.get("ftol_rel", 1e-6),
            cache=entry.get("cache", True),
            seed=entry.get("seed", 0),
        ))
    return configs


def save_manifest(
    configs: list[OptimizationConfig],
    path: str | pathlib.Path,
) -> None:
    """Serialise a list of OptimizationConfig objects to a JSON manifest."""
    jobs = []
    for cfg in configs:
        jobs.append({
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
            "output_root": str(cfg.output_root),
            "max_evals_stage1": cfg.max_evals_stage1,
            "max_evals_stage2": cfg.max_evals_stage2,
            "ftol_rel": cfg.ftol_rel,
            "cache": cfg.cache,
            "seed": cfg.seed,
        })
    pathlib.Path(path).write_text(json.dumps({"jobs": jobs}, indent=2))
