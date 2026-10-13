"""
optimization/config.py
----------------------
OptimizationConfig and ParameterSpec: the immutable run description contract.
One OptimizationConfig fully specifies a single optimization job with no
notebook-global constants.
"""

from __future__ import annotations
from codecs import lookup
import pathlib
from dataclasses import dataclass, field
from tkinter.font import names
from typing import Optional


@dataclass(frozen=True)
class ParameterSpec:
    """Metadata for one optimizable parameter."""
    name: str
    lower: float
    upper: float
    init: float       # starting point passed to stage-2 local optimizer
    step: float = 0.0  # initial step size hint (0 = let NLOPT choose)


@dataclass(frozen=True)
class OptimizationConfig:
    """Immutable description of one optimization job.

    Attributes
    ----------
    target :
        Human-readable label, e.g. ``"IND-Kennebec"``.
    location :
        Location code: ``"IND"``, ``"NLD"``, or ``"RSA"``.
    cultivar :
        Cultivar name matching experiments.yaml and the crop YAML.
    seasons :
        List of season keys to include, e.g. ``["2022-2023", "2024-2025"]``.
    obs_field :
        Experimental setup whose observations are fitted: ``"OpenField"`` or,
        for RSA, ``"RainShelter"``. Named obs_field rather than field because
        ``field`` is already taken by dataclasses.field in this module.
    active_params :
        Parameters that NLOPT will optimize (ordered list of ParameterSpec).
    fixed_overrides :
        Parameters held fixed for this target (applied before every simulation).
    crop_dir :
        Path relative to project root for the crop YAML directory,
        e.g. ``"data/crops_calibration"``.
    run_id :
        Version tag for output file naming, e.g. ``"v1"``.
    output_root :
        Directory where artifacts are written.
    max_evals_stage1 :
        Maximum function evaluations for the global stage.
    max_evals_stage2 :
        Maximum function evaluations for the local stage.
    ftol_rel :
        Relative function tolerance for convergence (both stages).
    cache :
        Whether to cache repeated objective evaluations.
    seed :
        Random seed passed to NLOPT global algorithms (0 = no seed).
    """
    target: str
    location: str
    cultivar: str
    seasons: tuple[str, ...]
    active_params: tuple[ParameterSpec, ...]
    fixed_overrides: dict = field(default_factory=dict)
    obs_field: str = "OpenField"
    crop_dir: str = "data/crops_calibration"
    run_id: str = "v1"
    output_root: pathlib.Path = pathlib.Path("output/optimization")
    max_evals_stage1: int = 1000
    max_evals_stage2: int = 500
    ftol_rel: float = 1e-6
    cache: bool = True
    seed: int = 0

    # dataclass frozen=True requires hashable field defaults
    def __post_init__(self):

        # coerce mutable defaults to immutable types
        if isinstance(self.seasons, list):
            object.__setattr__(self, "seasons", tuple(self.seasons))
        if isinstance(self.active_params, list):
            object.__setattr__(self, "active_params", tuple(self.active_params))
        if isinstance(self.fixed_overrides, dict):
            object.__setattr__(self, "fixed_overrides", dict(self.fixed_overrides))
        if isinstance(self.output_root, str):
            object.__setattr__(self, "output_root", pathlib.Path(self.output_root))


def load_param_specs(
    names: list[str],
    path: str | pathlib.Path = "data/obs/MainTable.xlsx",
    sheet: str = "Phenology",
    step_frac: float = 0.1,
) -> tuple[ParameterSpec, ...]:
    """Load ParameterSpec objects from the MainTable Parameters sheet.

    Expects columns: ``Parameter``, ``Default values (bounds)``.
    The ``Default values (bounds)`` cell format is ``"<init> [<lower>, <upper>]"``,
    e.g. ``"3 [2, 4]"`` or ``"170 [170,255]"``.

    Parameters
    ----------
    names :
        Ordered list of parameter names to load. Raises KeyError for any
        name not found in the sheet.
    path :
        Path to MainTable.xlsx, relative to the project root or absolute.
    sheet :
        Sheet name containing the parameter table.
    step_frac :
        Initial step size as a fraction of (upper - lower). Default 0.1.

    Returns
    -------
    Tuple of ParameterSpec in the same order as *names*.

    Example
    -------
    >>> specs = load_param_specs(["TBASEM", "TSUMEM", "TSUM1"])
    >>> config = OptimizationConfig(..., active_params=specs)
    """
    import pandas as pd

    p = pathlib.Path(path)
    if not p.is_absolute():
        # resolve relative to project root (two levels above this file)
        p = pathlib.Path(__file__).resolve().parents[2] / p

    df = pd.read_excel(p, sheet_name=sheet)
    if df["Default"].isna().any():
        raise ValueError("Found NaN in 'Default' column of Phenology sheet")  
    df["Parameters"] = df["Parameters"].str.strip()
    
    lookup = dict(zip(df["Parameters"], zip(df["Default"], df["Bounds"])))
    specs = []
    for name in names:
        default_val, bounds = lookup[name]
        bounds = str(bounds).strip()
        print(f"{name}: default={default_val}, bounds={bounds}")

        lower_str, upper_str = bounds.strip("[]").split(",", 1)
        init = float(default_val)
        lower = float(lower_str.strip())
        upper = float(upper_str.strip())
        step = (upper - lower) * step_frac
        specs.append(ParameterSpec(name=name, lower=lower, upper=upper, init=init, step=step))

    return tuple(specs)
