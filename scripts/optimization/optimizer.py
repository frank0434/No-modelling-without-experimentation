"""
optimization/optimizer.py
-------------------------
TwoStageNLOptimizer and OptimizationResult.
Stage 1: GN_DIRECT_L (global, derivative-free).
Stage 2: LN_SUBPLEX (local, derivative-free).
"""

from __future__ import annotations
import numpy as np
import nlopt
from dataclasses import dataclass
from .config import OptimizationConfig
from .resolver import ParameterSetResolver
from .objective import PhenologyObjective


@dataclass
class OptimizationResult:
    """Normalized result for one stage or the whole run."""
    stage: str          # "stage1" or "stage2"
    x_best: np.ndarray
    f_best: float
    n_evals: int
    overrides: dict     # named parameter values at x_best
    converged: bool


class TwoStageNLOptimizer:
    """Encapsulates all NLOPT setup and two-stage orchestration.

    Stage 1 (global): ``GN_DIRECT_L`` explores the full bounds space.
    Stage 2 (local):  ``LN_SUBPLEX`` polishes from the stage-1 best point.
    """

    def __init__(
        self,
        config: OptimizationConfig,
        resolver: ParameterSetResolver,
        objective: PhenologyObjective,
    ):
        self._config = config
        self._resolver = resolver
        self._objective = objective

    def _build_optimizer(self, algorithm, x0: np.ndarray) -> nlopt.opt:
        n = self._resolver.n
        opt = nlopt.opt(algorithm, n)
        opt.set_lower_bounds(self._resolver.lower_bounds())
        opt.set_upper_bounds(self._resolver.upper_bounds())
        opt.set_min_objective(self._objective)
        opt.set_ftol_rel(self._config.ftol_rel)
        return opt

    def run_stage1_global(self) -> OptimizationResult:
        n_before = len(self._objective.trace)

        opt = self._build_optimizer(nlopt.GN_DIRECT_L, self._resolver.init_vector())
        opt.set_maxeval(self._config.max_evals_stage1)
        if self._config.seed:
            nlopt.srand(self._config.seed)

        x0 = self._resolver.init_vector()
        try:
            x_opt = opt.optimize(x0)
            f_opt = opt.last_optimum_value()
            converged = True
        except nlopt.RoundoffLimited:
            best = self._objective.best()
            x_opt = best["x"] if best else x0
            f_opt = best["rmse"] if best else 1e9
            converged = False

        n_evals = len(self._objective.trace) - n_before
        return OptimizationResult(
            stage="stage1",
            x_best=x_opt,
            f_best=f_opt,
            n_evals=n_evals,
            overrides=self._resolver.vector_to_overrides(x_opt),
            converged=converged,
        )

    def run_stage2_local(self, x0: np.ndarray) -> OptimizationResult:
        n_before = len(self._objective.trace)

        opt = self._build_optimizer(nlopt.LN_SBPLX, x0)
        opt.set_maxeval(self._config.max_evals_stage2)

        steps = self._resolver.step_vector()
        if np.any(steps > 0):
            opt.set_initial_step(steps)

        try:
            x_opt = opt.optimize(x0.copy())
            f_opt = opt.last_optimum_value()
            converged = True
        except nlopt.RoundoffLimited:
            best = self._objective.best()
            x_opt = best["x"] if best else x0
            f_opt = best["rmse"] if best else 1e9
            converged = False

        n_evals = len(self._objective.trace) - n_before
        return OptimizationResult(
            stage="stage2",
            x_best=x_opt,
            f_best=f_opt,
            n_evals=n_evals,
            overrides=self._resolver.vector_to_overrides(x_opt),
            converged=converged,
        )

    def optimize(self) -> tuple[OptimizationResult, OptimizationResult]:
        """Run stage 1 then stage 2; return (stage1_result, stage2_result)."""
        r1 = self.run_stage1_global()
        r2 = self.run_stage2_local(r1.x_best)
        return r1, r2
