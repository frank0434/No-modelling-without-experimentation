"""
utils/metrics.py
----------------
Model evaluation metrics for comparing simulated vs observed values.
All functions accept array-like obs and sim of equal length.
"""

import numpy as np
import pandas as pd


def rmse(obs, sim) -> float:
    """Root Mean Square Error."""
    obs, sim = np.array(obs, dtype=float), np.array(sim, dtype=float)
    mask = ~(np.isnan(obs) | np.isnan(sim))
    return float(np.sqrt(np.mean((sim[mask] - obs[mask]) ** 2)))


def nrmse(obs, sim) -> float:
    """RMSE normalised by the mean of observations (dimensionless; ×100 for %)."""
    obs, sim = np.array(obs, dtype=float), np.array(sim, dtype=float)
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs_mean = np.mean(obs[mask])
    if obs_mean == 0:
        return float("nan")
    return float((rmse(obs, sim) / obs_mean))


def mae(obs, sim) -> float:
    """Mean Absolute Error."""
    obs, sim = np.array(obs, dtype=float), np.array(sim, dtype=float)
    mask = ~(np.isnan(obs) | np.isnan(sim))
    return float(np.mean(np.abs(sim[mask] - obs[mask])))


def ef(obs, sim) -> float:
    """Nash-Sutcliffe Model Efficiency (EF).
    EF = 1 → perfect; EF = 0 → model is as good as the mean; EF < 0 → worse than mean.
    """
    obs, sim = np.array(obs, dtype=float), np.array(sim, dtype=float)
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs_m, sim_m = obs[mask], sim[mask]
    ss_res = np.sum((obs_m - sim_m) ** 2)
    ss_tot = np.sum((obs_m - np.mean(obs_m)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def d_index(obs, sim) -> float:
    """Willmott's Index of Agreement (d).
    d = 1 → perfect agreement; d = 0 → no agreement.
    """
    obs, sim = np.array(obs, dtype=float), np.array(sim, dtype=float)
    mask = ~(np.isnan(obs) | np.isnan(sim))
    obs_m, sim_m = obs[mask], sim[mask]
    obs_mean = np.mean(obs_m)
    numerator = np.sum((obs_m - sim_m) ** 2)
    denominator = np.sum((np.abs(sim_m - obs_mean) + np.abs(obs_m - obs_mean)) ** 2)
    if denominator == 0:
        return float("nan")
    return float(1.0 - numerator / denominator)


class SummaryMetrics:
    """Bundle standard goodness-of-fit metrics for a sim vs obs comparison.

    Stores RMSE, NRMSE, MAE, EF (Nash-Sutcliffe) and Willmott's d as
    attributes. Signature matches MAEDecomposition(obs, sim) and
    MSEDecomposition(obs, sim).
    """

    def __init__(self, obs, sim):
        obs = np.asarray(obs, dtype=float)
        sim = np.asarray(sim, dtype=float)
        mask = np.isfinite(obs) & np.isfinite(sim)
        self.obs = obs[mask]
        self.sim = sim[mask]
        self.n = len(self.obs)

        self.RMSE = rmse(self.obs, self.sim)
        self.NRMSE = nrmse(self.obs, self.sim)
        self.MAE = mae(self.obs, self.sim)
        self.EF = ef(self.obs, self.sim)
        self.d = d_index(self.obs, self.sim)

    def as_dict(self) -> dict:
        return {
            "RMSE": self.RMSE,
            "NRMSE": self.NRMSE,
            "MAE": self.MAE,
            "EF": self.EF,
            "d": self.d,
        }

    def __repr__(self):
        return (f"SummaryMetrics(RMSE={self.RMSE:.4f}, NRMSE={self.NRMSE:.4f}, "
                f"MAE={self.MAE:.4f}, EF={self.EF:.4f}, d={self.d:.4f})")


def summary_metrics(obs, sim) -> dict:
    """Return a dict with RMSE, NRMSE, MAE, EF, and d-index."""
    return SummaryMetrics(obs, sim).as_dict()


def metrics_table(df: pd.DataFrame, obs_col: str, sim_col: str,
                  group_cols: list[str] = None) -> pd.DataFrame:
    """Compute metrics, optionally grouped by columns (e.g. ['Cultivar', 'Season']).

    Returns a DataFrame with one row per group (or one row overall).
    """
    def _row(sub):
        return pd.Series(summary_metrics(sub[obs_col], sub[sim_col]))

    if group_cols:
        return df.groupby(group_cols).apply(_row).reset_index()
    else:
        return pd.DataFrame([summary_metrics(df[obs_col], df[sim_col])])


def calc_aicc(observed, simulated, p):
    """Calculate AICc for a group of variables.
    Calculates the AICc for a group of variables.

    Parameters:
    observed (array-like): The measured data points.
    simulated (array-like): The values predicted by the model.
    p (int): The number of parameters being estimated.

    Returns:
    float: The calculated AICc value.
    """
    observed = np.asarray(observed, dtype=float)
    simulated = np.asarray(simulated, dtype=float)
    mask = np.isfinite(observed) & np.isfinite(simulated)
    observed = observed[mask]
    simulated = simulated[mask]
    n = len(observed)
    if n == 0:
        return np.nan, 0, np.nan
    ss = np.sum((observed - simulated) ** 2)
    if ss <= 0:
        ss = 1e-12
    if n - p - 1 > 0:
        aicc = n * np.log(ss / n) + 2 * p + (2 * p * (p + 1)) / (n - p - 1)
    else:
        aicc = np.inf
    return aicc, n, ss


class MAEDecomposition:
    """Decompose MAE into bias, proportional, and unsystematic components.

    Robeson, S.M. and Willmott, C.J. (2023). Decomposition of the mean
    absolute error (MAE) into systematic and unsystematic components.
    PLOS ONE 18(2): e0279774.

    All intermediate quantities are stored as attributes (abserr, mbe,
    sim_prime, slope/intercept of the sim_prime~obs regression, phat_prime,
    the per-point weights, and the three MAE components).
    """

    def __init__(self, obs, sim):
        obs = np.asarray(obs, dtype=float)
        sim = np.asarray(sim, dtype=float)
        mask = np.isfinite(obs) & np.isfinite(sim)
        self.obs = obs[mask]
        self.sim = sim[mask]
        self.n = len(self.obs)

        if np.std(self.obs) == 0:
            raise ValueError(
                "Observations are constant; the sim~obs regression is undefined."
            )

        self.abserr = np.abs(self.sim - self.obs)
        self.MAE = float(np.mean(self.abserr))
        self.mbe = np.mean(self.sim) - np.mean(self.obs)
        self.sim_prime = self.sim - self.mbe

        # OLS regression: sim_prime ~ obs
        self.slope, self.intercept = np.polyfit(self.obs, self.sim_prime, 1)
        self.phat_prime = self.intercept + self.slope * self.obs

        # per-point weights
        self.b = np.abs(self.mbe)
        self.p = np.abs(self.phat_prime - self.obs)
        self.u = np.abs(self.sim_prime - self.phat_prime)
        denom = self.b + self.p + self.u
        # perfect-fit points have b = p = u = 0 (and abserr = 0); avoid 0/0.
        denom_safe = np.where(denom == 0, 1.0, denom)
        self.bias_weight = self.b / denom_safe
        self.prop_weight = self.p / denom_safe
        self.unsys_weight = self.u / denom_safe

        # components of MAE
        self.MAE_b = float(np.sum(self.bias_weight * self.abserr) / self.n)
        self.MAE_p = float(np.sum(self.prop_weight * self.abserr) / self.n)
        self.MAE_u = float(np.sum(self.unsys_weight * self.abserr) / self.n)

        if not np.isclose(self.MAE, self.MAE_b + self.MAE_p + self.MAE_u, atol=1e-12):
            raise RuntimeError(
                "MAE decomposition failed: components do not sum to total MAE "
                f"({self.MAE} != {self.MAE_b + self.MAE_p + self.MAE_u})."
            )

    def as_dict(self) -> dict:
        return {
            "MAE": self.MAE,
            "MAE_b": self.MAE_b,
            "MAE_p": self.MAE_p,
            "MAE_u": self.MAE_u,
            "MBE": float(self.mbe),
            "slope": float(self.slope),
            "intercept": float(self.intercept),
        }

    def __repr__(self):
        return (f"MAEDecomposition(MAE_b={self.MAE_b:.4f}, "
                f"MAE_p={self.MAE_p:.4f}, MAE_u={self.MAE_u:.4f})")


class MSEDecomposition:
    """Decompose MSE into Bias, Sequence, and Distribution components.

    Hodson, T.O., Over, T.M. and Foks, S.S. (2021). Mean squared error,
    deconstructed. Journal of Advances in Modeling Earth Systems 13,
    e2021MS002681.

    Two identities are computed and checked:
        MSE = Bias_component + Variance
        MSE = Bias_component + Distribution + Sequence
    where Variance = Distribution + Sequence.
    """

    def __init__(self, obs, sim):
        obs = np.asarray(obs, dtype=float)
        sim = np.asarray(sim, dtype=float)
        mask = np.isfinite(obs) & np.isfinite(sim)
        self.obs = obs[mask]
        self.sim = sim[mask]
        self.n = len(self.obs)

        if self.n < 3:
            raise ValueError(
                f"At least 3 valid obs/sim pairs are required, got {self.n}."
            )

        error = self.sim - self.obs
        self.MSE = float(np.mean(error ** 2))
        self.Bias = float(np.mean(error))
        self.Bias_component = float(self.Bias ** 2)
        self.Variance = float(np.var(error, ddof=0))

        if not np.isclose(self.MSE, self.Bias_component + self.Variance, atol=1e-12):
            raise RuntimeError(
                "MSE decomposition failed: MSE != Bias_component + Variance "
                f"({self.MSE} != {self.Bias_component + self.Variance})."
            )

        # independently sorted series isolate distribution vs sequence error
        omega = np.sort(self.sim) - np.sort(self.obs)
        self.Distribution = float(np.var(omega, ddof=0))
        self.Sequence = float(self.Variance - self.Distribution)

        if not np.isclose(
            self.MSE,
            self.Bias_component + self.Distribution + self.Sequence,
            atol=1e-12,
        ):
            raise RuntimeError(
                "MSE decomposition failed: components do not sum to MSE "
                f"({self.MSE} != "
                f"{self.Bias_component + self.Distribution + self.Sequence})."
            )

    def as_dict(self) -> dict:
        return {
            "MSE": self.MSE,
            "Bias": self.Bias,
            "Bias_component": self.Bias_component,
            "Variance": self.Variance,
            "Distribution": self.Distribution,
            "Sequence": self.Sequence,
        }

    def __repr__(self):
        return (f"MSEDecomposition(Bias_component={self.Bias_component:.4f}, "
                f"Distribution={self.Distribution:.4f}, "
                f"Sequence={self.Sequence:.4f})")


def calc_aicc_by_group(df, obs_col, sim_col, p, group_cols=None):
    """Compute AICc for each group in *df*.

    Parameters
    ----------
    df : pd.DataFrame
    obs_col : str   column with observed values
    sim_col  : str   column with simulated values
    p        : int   number of estimated parameters
    group_cols : list[str], optional defaults to ["location", "cultivar", "Field", "treatment"]

    Returns
    -------
    pd.DataFrame with one row per group and columns: <group_cols>, n, SSE, AICc
    """
    if group_cols is None:
        group_cols = ["location", "cultivar", "Field", "treatment"]
    rows = []
    for keys, sub in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        aicc, n, ss = calc_aicc(sub[obs_col], sub[sim_col], p=p)
        row = dict(zip(group_cols, keys))
        row.update({"n": n, "SSE": ss, "AICc": aicc})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["location", "AICc"], na_position="last")


class WeightedRMSECalculator:
    """Computes a weighted normalized RMSE across DVS, LAI, and TWSO.

    Each variable is normalized by its observed mean before RMSE is computed,
    so all three are on the same scale before weighting.  The final metric is
    a weighted sum of per-variable normalized RMSEs, grouped by
    (location, cultivar).  Used to rank which default cultivar parameter set
    best fits an experiment, as a calibration template.

    The (obs_col, sim_col, weight) weights below are this class's own choice;
    they are not derived from any other module.  Note that a (location,
    cultivar) group missing observations for a variable simply contributes
    fewer terms to its weighted sum — harmless when ranking within a location
    (all candidate cultivars share the same obs coverage), but the raw
    weighted_NRMSE magnitude is not comparable across locations.

    Assumptions
    -----------
    All obs DataFrames must contain columns:
        location, season, cultivar, day
    plus the variable column listed below.
    df_obs_phen : obs_dvs
    df_obs_lai  : LAI
    df_obs_dm   : TWSO_obs
    """

    # (obs_col, sim_col, weight)
    variable_mapping = [
        ("obs_dvs",  "DVS",  1.0),
        ("LAI_obs",  "LAI_sim", 0.25),
        ("TWSO_obs", "TWSO", 0.5),
    ]

    def __init__(self, df_sim, df_obs_phen, df_obs_dm, df_obs_lai):
        # Merge only on location/season/day — obs has real cultivar names that
        # don't match the default parameter sets in df_sim.  The sim's cultivar
        # column is kept so we can later rank which default cultivar fits best.
        join_keys = ["location", "season", "day"]
        sim_keys  = join_keys + ["cultivar"]

        self._dvs = pd.merge(
            df_obs_phen[join_keys + ["obs_dvs"]],
            df_sim[sim_keys + ["DVS"]],
            on=join_keys, how="right",
        )
        self._lai = pd.merge(
            df_obs_lai[join_keys + ["LAI"]],
            df_sim[sim_keys + ["LAI"]],
            on=join_keys, how="right",
            suffixes=("_obs", "_sim"),
        )
        self._twso = pd.merge(
            df_obs_dm[join_keys + ["TWSO_obs"]],
            df_sim[sim_keys + ["TWSO"]],
            on=join_keys, how="right",
        )

        self._merged = {"obs_dvs": self._dvs, "LAI_obs": self._lai, "TWSO_obs": self._twso}

    def compute(self):
        """Return DataFrame with weighted NRMSE per (location, cultivar), sorted best first."""
        group_keys = ["location", "cultivar"]
        result_rows = {}

        for obs_col, sim_col, weight in self.variable_mapping:
            df = self._merged[obs_col].dropna(subset=[obs_col, sim_col])
            for group, sub in df.groupby(group_keys):
                if sub[obs_col].mean() == 0:
                    continue
                nrmse_val = nrmse(sub[obs_col], sub[sim_col])
                key = group  # (location, cultivar)
                if key not in result_rows:
                    result_rows[key] = {"location": key[0], "cultivar": key[1], "weighted_NRMSE": 0.0}
                result_rows[key][f"NRMSE_{sim_col}"] = nrmse_val
                result_rows[key]["weighted_NRMSE"] += nrmse_val * weight

        return (
            pd.DataFrame(result_rows.values())
            .sort_values(["location", "weighted_NRMSE"])
            .reset_index(drop=True)
        )
