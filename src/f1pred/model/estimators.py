"""Baselines and learned models behind one interface.

Every model returns, for each row (a driver in a race, or a driver at the end
of a lap), three predictions that are consistent within each race:

- `p_win`: win probabilities summing to 1 across the race;
- `p_podium`: podium probabilities summing to 3 (each at most 1);
- `pred_rank`: a predicted finishing order 1..n.

Coherence is enforced in `finalize`, so models can be compared like for like.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

# Rows that share one race state: a race (pre-race) or a race at one lap.
Groups = Sequence[str]
PRE_RACE_GROUPS: tuple[str, ...] = ("race_id",)
LAP_GROUPS: tuple[str, ...] = ("race_id", "lap")


class RaceModel(Protocol):
    name: str

    def fit(self, df: pd.DataFrame) -> RaceModel: ...

    def predict(self, df: pd.DataFrame) -> pd.DataFrame: ...


def finalize(
    df: pd.DataFrame,
    groups: Groups,
    win_score: np.ndarray[Any, Any],
    podium_score: np.ndarray[Any, Any],
    expected_position: np.ndarray[Any, Any],
    tie_break: str,
) -> pd.DataFrame:
    """Turn raw per-row scores into race-consistent probabilities and an order."""
    out = df[[*groups]].copy()
    out["p_win"] = np.clip(win_score, 1e-6, None)
    out["p_win"] /= out.groupby(list(groups))["p_win"].transform("sum")
    out["p_podium"] = _scale_with_cap(out, groups, np.clip(podium_score, 1e-6, 1.0), total=3)
    out["expected_position"] = expected_position
    out["_tie"] = df[tie_break].to_numpy()
    out["pred_rank"] = (
        out.sort_values(["expected_position", "_tie"])
        .groupby(list(groups))
        .cumcount()
        .add(1)
        .reindex(out.index)
    )
    return out.drop(columns=["_tie", *groups])


def _scale_with_cap(
    df: pd.DataFrame, groups: Groups, p: np.ndarray[Any, Any], total: int
) -> np.ndarray[Any, Any]:
    """Scale each group's probabilities to sum to min(total, n), none above 1."""
    keys = df.groupby(list(groups)).ngroup().to_numpy()
    counts = np.bincount(keys)
    target = np.minimum(total, counts)[keys]
    p = p.astype(float).copy()
    capped = np.zeros(len(p), dtype=bool)
    # Each pass fixes any newly capped probabilities at 1 and rescales the rest
    # to fill what is left. It ends in at most one pass per car.
    for _ in range(int(np.max(counts)) + 1):
        fixed = np.bincount(keys, weights=capped.astype(float), minlength=len(counts))[keys]
        free_sum = np.bincount(keys, weights=np.where(capped, 0.0, p), minlength=len(counts))
        scale = (target - fixed) / np.maximum(free_sum[keys], 1e-12)
        p = np.where(capped, 1.0, p * scale)
        newly = (p >= 1.0) & ~capped
        if not newly.any():
            break
        capped |= newly
    p = np.where(capped, 1.0, p)
    return p


# --- Baselines -------------------------------------------------------------------


@dataclass
class GridBaseline:
    """Pre-race baseline: "finish where you start".

    The predicted order is the grid. Win and podium probabilities are the
    historical rates for each grid slot in the training seasons, smoothed toward
    the overall rate so rare slots do not get exactly 0.
    """

    name: str = "Grid baseline"
    smoothing: float = 2.0
    _rates: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def fit(self, df: pd.DataFrame) -> GridBaseline:
        self._rates = _smoothed_rates(df, ["grid_position"], self.smoothing)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        rates = _lookup(df, self._rates, ["grid_position"])
        return finalize(
            df,
            PRE_RACE_GROUPS,
            rates["label_win"],
            rates["label_podium"],
            df["grid_position"].to_numpy(dtype=float),
            tie_break="grid_position",
        )


@dataclass
class PositionBaseline:
    """Lap-by-lap baseline: "the order now is the final order".

    Probabilities are the historical win and podium rates for a car in this
    position at this stage of the race (in tenths of race distance).
    """

    name: str = "Current position baseline"
    smoothing: float = 2.0
    _rates: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def fit(self, df: pd.DataFrame) -> PositionBaseline:
        self._rates = _smoothed_rates(_with_stage(df), ["position", "stage"], self.smoothing)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        rates = _lookup(_with_stage(df), self._rates, ["position", "stage"])
        return finalize(
            df,
            LAP_GROUPS,
            rates["label_win"],
            rates["label_podium"],
            df["position"].to_numpy(dtype=float),
            tie_break="position",
        )


def _with_stage(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(stage=np.minimum((df["race_progress"] * 10).astype(int), 9))


def _smoothed_rates(df: pd.DataFrame, keys: list[str], smoothing: float) -> pd.DataFrame:
    grouped = df.groupby(keys)[["label_win", "label_podium"]].agg(["sum", "count"])
    rates = pd.DataFrame(index=grouped.index)
    for label in ("label_win", "label_podium"):
        prior = df[label].mean()
        hits, n = grouped[(label, "sum")], grouped[(label, "count")]
        rates[label] = (hits + smoothing * prior) / (n + smoothing)
    return rates.reset_index()


def _lookup(df: pd.DataFrame, rates: pd.DataFrame, keys: list[str]) -> dict[str, Any]:
    """Rates for each row. Unseen keys (e.g. grid slot 22) take the nearest seen key."""
    probe = df[keys].copy()
    for key in keys:
        seen = np.sort(rates[key].unique())
        idx = np.clip(np.searchsorted(seen, probe[key].fillna(seen[-1])), 0, len(seen) - 1)
        probe[key] = seen[idx]
    merged = probe.merge(rates, on=keys, how="left")
    return {label: merged[label].to_numpy() for label in ("label_win", "label_podium")}


# --- Learned models --------------------------------------------------------------


@dataclass
class LinearModel:
    """Logistic regression for win and podium, ridge regression for position.

    Missing values (a debut has no form) are imputed with the training median,
    with an indicator column so the model can tell "imputed" from "typical".
    """

    features: Sequence[str]
    groups: Groups
    tie_break: str
    C: float = 1.0
    name: str = "Logistic regression"
    _models: dict[str, Pipeline] = field(default_factory=dict, repr=False)

    def fit(self, df: pd.DataFrame) -> LinearModel:
        x = df[list(self.features)]
        for label in ("label_win", "label_podium"):
            clf = LogisticRegression(C=self.C, max_iter=2000)
            self._models[label] = self._pipeline(clf).fit(x, df[label])
        reg = Ridge(alpha=1.0 / self.C)
        self._models["label_finishing_order"] = self._pipeline(reg).fit(
            x, df["label_finishing_order"]
        )
        return self

    @staticmethod
    def _pipeline(estimator: Any) -> Pipeline:
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            StandardScaler(),
            estimator,
        )

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[list(self.features)]
        return finalize(
            df,
            self.groups,
            self._models["label_win"].predict_proba(x)[:, 1],
            self._models["label_podium"].predict_proba(x)[:, 1],
            self._models["label_finishing_order"].predict(x),
            tie_break=self.tie_break,
        )


@dataclass
class BoostedModel:
    """XGBoost classifiers for win and podium, an XGBoost regressor for position.

    Shallow trees and strong regularisation: there are only about 24 races a season.
    XGBoost handles missing values itself, so no imputation.
    """

    features: Sequence[str]
    groups: Groups
    tie_break: str
    max_depth: int = 3
    n_estimators: int = 300
    learning_rate: float = 0.05
    seed: int = 0
    name: str = "XGBoost"
    _models: dict[str, Any] = field(default_factory=dict, repr=False)

    def _params(self) -> dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 5.0,
            "tree_method": "hist",
            "random_state": self.seed,
            "n_jobs": 4,
        }

    def fit(self, df: pd.DataFrame) -> BoostedModel:
        x = df[list(self.features)]
        for label in ("label_win", "label_podium"):
            clf = XGBClassifier(objective="binary:logistic", **self._params())
            self._models[label] = clf.fit(x, df[label])
        reg = XGBRegressor(objective="reg:squarederror", **self._params())
        self._models["label_finishing_order"] = reg.fit(x, df["label_finishing_order"])
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df[list(self.features)]
        return finalize(
            df,
            self.groups,
            self._models["label_win"].predict_proba(x)[:, 1],
            self._models["label_podium"].predict_proba(x)[:, 1],
            self._models["label_finishing_order"].predict(x),
            tie_break=self.tie_break,
        )

    def feature_importance(self) -> pd.Series:
        """Mean gain of each feature across the three boosted models."""
        gains = [
            pd.Series(m.get_booster().get_score(importance_type="gain"))
            for m in self._models.values()
        ]
        table = pd.concat(gains, axis=1).reindex(list(self.features)).fillna(0.0)
        share = table / table.sum()
        return share.mean(axis=1).sort_values(ascending=False)
