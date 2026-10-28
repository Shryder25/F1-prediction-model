"""Tune, train and score every model on seasons it never saw.

The protocol, the same for the pre-race and lap-by-lap tasks:

1. Fit each candidate setting on the training seasons; score it on the
   validation season; keep the best (lowest win + podium Brier score).
2. Refit the chosen settings on training + validation seasons.
3. Score once on the test season and on any holdout seasons. The test season
   plays no part in any choice, so its numbers are an honest estimate.
4. Compare each model with the baseline on the test season, with a 95%
   bootstrap interval over races.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from f1pred.model.dataset import (
    KEYS,
    LABELS,
    LAP_FEATURES,
    PRE_RACE_FEATURES,
    SeasonSplit,
    seasons_in,
)
from f1pred.model.estimators import (
    LAP_GROUPS,
    PRE_RACE_GROUPS,
    BoostedModel,
    GridBaseline,
    LinearModel,
    PositionBaseline,
    RaceModel,
)
from f1pred.model.metrics import PairedDifference, paired_bootstrap, score

log = logging.getLogger(__name__)

Params = dict[str, Any]

# Stages of a race used to break down the lap-by-lap scores.
STAGES: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.25, "0-25%"),
    (0.25, 0.5, "25-50%"),
    (0.5, 0.75, "50-75%"),
    (0.75, 1.01, "75-100%"),
)


@dataclass(frozen=True)
class Candidate:
    """A model family and the settings to try for it. The first is the baseline."""

    name: str
    build: Callable[[Params], RaceModel]
    grid: tuple[Params, ...] = ({},)
    is_baseline: bool = False


def pre_race_candidates(seed: int = 0, quick: bool = False) -> list[Candidate]:
    def linear(p: Params) -> RaceModel:
        return LinearModel(PRE_RACE_FEATURES, PRE_RACE_GROUPS, "grid_position", **p)

    def boosted(p: Params) -> RaceModel:
        return BoostedModel(PRE_RACE_FEATURES, PRE_RACE_GROUPS, "grid_position", seed=seed, **p)

    c_grid = ({"C": 0.1}, {"C": 1.0}) if quick else tuple({"C": c} for c in (0.01, 0.1, 1.0, 10.0))
    tree_grid = (
        ({"max_depth": 2, "n_estimators": 150},)
        if quick
        else tuple({"max_depth": d, "n_estimators": n} for d in (2, 3, 4) for n in (150, 400))
    )
    return [
        Candidate("Grid baseline", lambda _: GridBaseline(), is_baseline=True),
        Candidate("Logistic regression", linear, c_grid),
        Candidate("XGBoost", boosted, tree_grid),
    ]


def lap_candidates(seed: int = 0, quick: bool = False) -> list[Candidate]:
    def linear(p: Params) -> RaceModel:
        return LinearModel(LAP_FEATURES, LAP_GROUPS, "position", **p)

    def boosted(p: Params) -> RaceModel:
        return BoostedModel(LAP_FEATURES, LAP_GROUPS, "position", seed=seed, **p)

    c_grid = ({"C": 1.0},) if quick else ({"C": 0.1}, {"C": 1.0})
    tree_grid = (
        ({"max_depth": 3, "n_estimators": 100},)
        if quick
        else ({"max_depth": 3, "n_estimators": 300}, {"max_depth": 5, "n_estimators": 300})
    )
    return [
        Candidate("Current position baseline", lambda _: PositionBaseline(), is_baseline=True),
        Candidate("Logistic regression", linear, c_grid),
        Candidate("XGBoost", boosted, tree_grid),
    ]


@dataclass
class TaskResult:
    """Everything one task produced: scores, predictions and the fitted models."""

    task: str
    split: SeasonSplit
    chosen: dict[str, Params]
    validation_scores: dict[str, dict[str, float]]
    # (model name, season) -> predictions joined to keys and labels.
    predictions: dict[tuple[str, int], pd.DataFrame]
    models: dict[str, RaceModel]
    baseline: str
    stage_scores: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def model_names(self) -> list[str]:
        return list(self.models)

    def scores(self) -> pd.DataFrame:
        """One row per (model, season) with every metric."""
        groups = PRE_RACE_GROUPS if self.task == "pre_race" else LAP_GROUPS
        rows = [
            {"model": name, "season": season, **score(pred, groups)}
            for (name, season), pred in self.predictions.items()
        ]
        return pd.DataFrame(rows)

    def comparisons(self, n_boot: int = 2000, seed: int = 0) -> dict[str, list[PairedDifference]]:
        """Each learned model vs the baseline on the test season."""
        base = self.predictions[(self.baseline, self.split.test)]
        return {
            name: paired_bootstrap(
                self.predictions[(name, self.split.test)], base, n_boot=n_boot, seed=seed
            )
            for name in self.models
            if name != self.baseline
        }

    def best_learned(self) -> str:
        """The learned model with the best validation score: the one to deploy."""
        learned = {k: v for k, v in self.validation_scores.items() if k != self.baseline}
        return min(learned, key=lambda k: _selection_score(learned[k]))


def _selection_score(scores: dict[str, float]) -> float:
    return scores["brier_win"] + scores["brier_podium"]


def _scored(df: pd.DataFrame, model: RaceModel) -> pd.DataFrame:
    extra = [c for c in ("lap", "race_progress", "position", "grid_position") if c in df.columns]
    base = df[[*KEYS, *extra, *LABELS]].reset_index(drop=True)
    return pd.concat([base, model.predict(df).reset_index(drop=True)], axis=1)


def run_task(
    task: str,
    df: pd.DataFrame,
    split: SeasonSplit,
    candidates: Sequence[Candidate],
) -> TaskResult:
    groups = PRE_RACE_GROUPS if task == "pre_race" else LAP_GROUPS
    train = seasons_in(df, split.train)
    validation = seasons_in(df, [split.validation])
    for name, part in (("training", train), ("validation", validation)):
        if part.empty:
            raise ValueError(f"No {task} rows in the {name} seasons of {split.describe()}.")

    chosen: dict[str, Params] = {}
    validation_scores: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        best: tuple[float, Params, dict[str, float]] | None = None
        for params in candidate.grid:
            model = candidate.build(params).fit(train)
            scores = score(_scored(validation, model), groups)
            log.info("%s %s %s: %s", task, candidate.name, params, _fmt(scores))
            if best is None or _selection_score(scores) < best[0]:
                best = (_selection_score(scores), params, scores)
        assert best is not None
        chosen[candidate.name] = best[1]
        validation_scores[candidate.name] = best[2]

    fit_on = seasons_in(df, [*split.train, split.validation])
    models: dict[str, RaceModel] = {}
    predictions: dict[tuple[str, int], pd.DataFrame] = {}
    for candidate in candidates:
        model = candidate.build(chosen[candidate.name]).fit(fit_on)
        models[candidate.name] = model
        for season in (split.test, *split.holdout):
            part = seasons_in(df, [season])
            if not part.empty:
                predictions[(candidate.name, season)] = _scored(part, model)

    baseline = next(c.name for c in candidates if c.is_baseline)
    result = TaskResult(task, split, chosen, validation_scores, predictions, models, baseline)
    if task == "lap":
        result.stage_scores = _stage_scores(result)
    return result


def _stage_scores(result: TaskResult) -> pd.DataFrame:
    """Test-season Brier scores by how far into the race the prediction was made."""
    rows = []
    for name in result.models:
        pred = result.predictions[(name, result.split.test)]
        for low, high, label in STAGES:
            part = pred[(pred["race_progress"] >= low) & (pred["race_progress"] < high)]
            if part.empty:
                continue
            scores = score(part, LAP_GROUPS)
            rows.append(
                {
                    "model": name,
                    "stage": label,
                    "brier_win": scores["brier_win"],
                    "brier_podium": scores["brier_podium"],
                    "position_mae": scores["position_mae"],
                }
            )
    return pd.DataFrame(rows)


def deploy_model(
    task: str, df: pd.DataFrame, split: SeasonSplit, candidate: Candidate, params: Params
) -> RaceModel:
    """Refit the chosen model on every season up to and including the test season.

    The evaluation numbers come from the frozen model in `run_task`; this one,
    trained on more data, is what the live predictor (stage 4) loads.
    """
    return candidate.build(params).fit(seasons_in(df, [*split.train, split.validation, split.test]))


def _fmt(scores: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.4f}" for k, v in scores.items())
