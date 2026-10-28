"""Scores for race predictions, with race-level bootstrap confidence intervals.

Rows within a race are not independent (exactly one driver wins), so every
interval here resamples whole races, never single rows.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-6


def brier(y: pd.Series, p: pd.Series) -> float:
    """Mean squared error of a probability. 0 is perfect; lower is better."""
    return float(np.mean((p.to_numpy() - y.to_numpy()) ** 2))


def log_loss(y: pd.Series, p: pd.Series) -> float:
    q = np.clip(p.to_numpy(), EPS, 1 - EPS)
    t = y.to_numpy()
    return float(-np.mean(t * np.log(q) + (1 - t) * np.log(1 - q)))


def position_mae(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean absolute error, in places, of the predicted finishing order."""
    return float(np.mean(np.abs(actual.to_numpy() - predicted.to_numpy())))


def winner_hit_rate(scored: pd.DataFrame, groups: Sequence[str] = ("race_id",)) -> float:
    """Share of races where the favourite (highest p_win) actually won."""
    favourite = scored.loc[scored.groupby(list(groups))["p_win"].idxmax()]
    return float(favourite["label_win"].mean())


def expected_calibration_error(y: pd.Series, p: pd.Series, bins: int = 10) -> float:
    table = calibration_table(y, p, bins)
    weights = table["n"] / table["n"].sum()
    return float((weights * (table["mean_predicted"] - table["observed_rate"]).abs()).sum())


def calibration_table(y: pd.Series, p: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed rate in equal-width probability bins (empty bins dropped)."""
    edges = np.linspace(0, 1, bins + 1)
    bucket = np.clip(np.digitize(p.to_numpy(), edges[1:-1]), 0, bins - 1)
    frame = pd.DataFrame({"bucket": bucket, "p": p.to_numpy(), "y": y.to_numpy()})
    table = frame.groupby("bucket").agg(
        n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean")
    )
    table["bin_low"] = edges[table.index]
    table["bin_high"] = edges[table.index + 1]
    return table.reset_index(drop=True)


METRICS: dict[str, Callable[[pd.DataFrame], float]] = {
    "brier_win": lambda d: brier(d["label_win"], d["p_win"]),
    "brier_podium": lambda d: brier(d["label_podium"], d["p_podium"]),
    "log_loss_win": lambda d: log_loss(d["label_win"], d["p_win"]),
    "log_loss_podium": lambda d: log_loss(d["label_podium"], d["p_podium"]),
    "position_mae": lambda d: position_mae(d["label_finishing_order"], d["pred_rank"]),
}

# Metrics where a smaller number is better (all of them, today).
LOWER_IS_BETTER = frozenset(METRICS)


def score(scored: pd.DataFrame, groups: Sequence[str] = ("race_id",)) -> dict[str, float]:
    result = {name: fn(scored) for name, fn in METRICS.items()}
    result["winner_hit_rate"] = winner_hit_rate(scored, groups)
    result["ece_podium"] = expected_calibration_error(scored["label_podium"], scored["p_podium"])
    return result


@dataclass(frozen=True)
class PairedDifference:
    """Model minus baseline for one metric, with a 95% bootstrap interval over races."""

    metric: str
    model: float
    baseline: float
    diff: float
    low: float
    high: float

    @property
    def relative(self) -> float:
        """Improvement as a share of the baseline. Positive means the model is better."""
        return -self.diff / self.baseline if self.baseline else 0.0

    @property
    def significant(self) -> bool:
        """True when the whole 95% interval is on one side of zero."""
        return self.high < 0 or self.low > 0


def paired_bootstrap(
    model: pd.DataFrame,
    baseline: pd.DataFrame,
    metrics: Sequence[str] = ("brier_win", "brier_podium", "position_mae"),
    n_boot: int = 2000,
    seed: int = 0,
) -> list[PairedDifference]:
    """Resample races with replacement and recompute (model - baseline) each time.

    Both frames must hold the same rows in the same order. Each metric here is a
    mean of per-row errors, so we precompute per-race error sums and counts and
    resample those, which keeps 2,000 resamples fast.
    """
    if len(model) != len(baseline):
        raise ValueError("Model and baseline predictions must cover the same rows.")
    rng = np.random.default_rng(seed)
    race = model["race_id"].to_numpy()
    races, race_index = np.unique(race, return_inverse=True)
    counts = np.bincount(race_index).astype(float)
    picks = rng.integers(0, len(races), size=(n_boot, len(races)))
    weights = np.stack([np.bincount(row, minlength=len(races)) for row in picks])

    results = []
    for metric in metrics:
        err_m = _row_errors(model, metric)
        err_b = _row_errors(baseline, metric)
        sum_m = np.bincount(race_index, weights=err_m)
        sum_b = np.bincount(race_index, weights=err_b)
        n = weights @ counts
        boot = (weights @ sum_m - weights @ sum_b) / n
        point_m, point_b = err_m.mean(), err_b.mean()
        low, high = np.percentile(boot, [2.5, 97.5])
        results.append(
            PairedDifference(
                metric, float(point_m), float(point_b), float(point_m - point_b), low, high
            )
        )
    return results


def _row_errors(df: pd.DataFrame, metric: str) -> np.ndarray[Any, np.dtype[np.float64]]:
    if metric == "brier_win":
        error = df["p_win"] - df["label_win"]
        return error.to_numpy(dtype=float) ** 2
    if metric == "brier_podium":
        error = df["p_podium"] - df["label_podium"]
        return error.to_numpy(dtype=float) ** 2
    if metric == "position_mae":
        error = df["pred_rank"] - df["label_finishing_order"]
        return np.abs(error.to_numpy(dtype=float))
    raise KeyError(f"No per-row error defined for {metric}")
