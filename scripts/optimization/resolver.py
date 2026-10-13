"""
optimization/resolver.py
------------------------
ParameterSetResolver: consistent vector ordering, bounds, step sizes,
and vector-to-override conversion across all targets.
"""

from __future__ import annotations
import numpy as np
from .config import OptimizationConfig


class ParameterSetResolver:
    """Translates between NLOPT's flat parameter vector and named overrides.

    Enforces a deterministic ordering (the order of ``config.active_params``).
    All bounds and step sizes come from ParameterSpec, so callers never
    hard-code indices.
    """

    def __init__(self, config: OptimizationConfig):
        self._specs = config.active_params  # ordered tuple of ParameterSpec

    @property
    def n(self) -> int:
        return len(self._specs)

    @property
    def names(self) -> list[str]:
        return [s.name for s in self._specs]

    def lower_bounds(self) -> list[float]:
        return [s.lower for s in self._specs]

    def upper_bounds(self) -> list[float]:
        return [s.upper for s in self._specs]

    def init_vector(self) -> np.ndarray:
        return np.array([s.init for s in self._specs], dtype=float)

    def step_vector(self) -> np.ndarray:
        return np.array([s.step for s in self._specs], dtype=float)

    def vector_to_overrides(self, x: np.ndarray) -> dict:
        """Convert an NLOPT parameter vector to a named-override dict."""
        return {spec.name: float(x[i]) for i, spec in enumerate(self._specs)}
