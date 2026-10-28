"""Run the full evaluation and write it up: report.md, metrics.json, figures, models.

    f1pred evaluate            # real warehouse
    f1pred --demo evaluate     # synthetic demo warehouse

Output goes to <data_dir>/reports/model/ unless --out says otherwise.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import duckdb
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from f1pred.demo import DEMO_MARKER_TABLE
from f1pred.model.dataset import (
    LAP_FEATURES,
    PRE_RACE_FEATURES,
    SeasonSplit,
    load_laps,
    load_pre_race,
)
from f1pred.model.estimators import BoostedModel
from f1pred.model.evaluate import (
    TaskResult,
    deploy_model,
    lap_candidates,
    pre_race_candidates,
    run_task,
)
from f1pred.model.metrics import PairedDifference, calibration_table

log = logging.getLogger(__name__)

# Neutral grey for the baseline, then the first two categorical slots of the
# validated default palette (see the dataviz guidance) for the learned models.
COLOURS = ("#8c8b86", "#2a78d6", "#eb6834")
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"

# Calibration bins with fewer predictions than this are too noisy to plot.
MIN_BIN_ROWS = 5

METRIC_LABELS = {
    "brier_win": "Brier (win)",
    "brier_podium": "Brier (podium)",
    "log_loss_win": "Log loss (win)",
    "log_loss_podium": "Log loss (podium)",
    "position_mae": "Position MAE",
    "winner_hit_rate": "Favourite won",
    "ece_podium": "Calibration error (podium)",
}


def build_report(
    warehouse_path: Path,
    out_dir: Path,
    split: SeasonSplit | None = None,
    quick: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate both tasks and write the report. Returns the metrics written to JSON."""
    with duckdb.connect(str(warehouse_path), read_only=True) as con:
        pre_race = load_pre_race(con)
        laps = load_laps(con)
        is_demo = bool(
            con.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
                [DEMO_MARKER_TABLE],
            ).fetchone()[0]  # type: ignore[index]
        )
    split = split or SeasonSplit.latest(sorted(pre_race["season"].unique().tolist()))
    log.info("Season split: %s", split.describe())

    pre = run_task("pre_race", pre_race, split, pre_race_candidates(seed, quick))
    lap = run_task("lap", laps, split, lap_candidates(seed, quick))

    out_dir.mkdir(parents=True, exist_ok=True)
    figures = out_dir / "figures"
    figures.mkdir(exist_ok=True)
    _calibration_figure(pre, figures / "calibration_pre_race.png")
    _stage_figure(lap, figures / "lap_brier_by_stage.png")
    n_boot = 500 if quick else 2000
    comparisons = {"pre_race": pre.comparisons(n_boot, seed), "lap": lap.comparisons(n_boot, seed)}

    saved = _save_models(out_dir / "models", pre_race, laps, split, seed, quick, pre, lap)
    metrics = {
        "demo_data": is_demo,
        "split": asdict(split),
        "rows": {"pre_race": len(pre_race), "lap": len(laps)},
        "chosen_settings": {"pre_race": pre.chosen, "lap": lap.chosen},
        "scores": {
            "pre_race": pre.scores().to_dict(orient="records"),
            "lap": lap.scores().to_dict(orient="records"),
            "lap_by_stage": lap.stage_scores.to_dict(orient="records"),
        },
        "vs_baseline": {
            task: {name: [asdict(d) for d in diffs] for name, diffs in by_model.items()}
            for task, by_model in comparisons.items()
        },
        "deployed_models": saved,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (out_dir / "report.md").write_text(_markdown(pre, lap, comparisons, pre_race, is_demo, saved))
    return metrics


def _save_models(
    model_dir: Path,
    pre_race: pd.DataFrame,
    laps: pd.DataFrame,
    split: SeasonSplit,
    seed: int,
    quick: bool,
    pre: TaskResult,
    lap: TaskResult,
) -> dict[str, str]:
    model_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for task, df, result, candidates in (
        ("pre_race", pre_race, pre, pre_race_candidates(seed, quick)),
        ("lap", laps, lap, lap_candidates(seed, quick)),
    ):
        name = result.best_learned()
        candidate = next(c for c in candidates if c.name == name)
        model = deploy_model(task, df, split, candidate, result.chosen[name])
        joblib.dump(model, model_dir / f"{task}.joblib")
        saved[task] = name
    return saved


# --- Figures ---------------------------------------------------------------------


def _style(ax: Any) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _calibration_figure(result: TaskResult, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), dpi=150)
    season = result.split.test
    for ax, (label, prob, title) in zip(
        axes,
        (("label_win", "p_win", "Win"), ("label_podium", "p_podium", "Podium")),
        strict=True,
    ):
        _style(ax)
        ax.plot(
            [0, 1], [0, 1], linestyle="--", color=MUTED, linewidth=1, label="Perfectly calibrated"
        )
        for colour, name in zip(COLOURS, result.model_names, strict=False):
            pred = result.predictions[(name, season)]
            table = calibration_table(pred[label], pred[prob], bins=10)
            table = table[table["n"] >= MIN_BIN_ROWS]
            ax.plot(
                table["mean_predicted"],
                table["observed_rate"],
                color=colour,
                linewidth=2,
                marker="o",
                markersize=5,
                label=name,
            )
        ax.set_title(f"{title} probability, {season} (unseen)", color=TEXT, fontsize=11, loc="left")
        ax.set_xlabel("Predicted probability", color=MUTED, fontsize=9)
        ax.set_ylabel("Observed rate", color=MUTED, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _stage_figure(result: TaskResult, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), dpi=150)
    stages = result.stage_scores
    for ax, (metric, title) in zip(
        axes, (("brier_win", "Win"), ("brier_podium", "Podium")), strict=True
    ):
        _style(ax)
        for colour, name in zip(COLOURS, result.model_names, strict=False):
            part = stages[stages["model"] == name]
            ax.plot(
                part["stage"],
                part[metric],
                color=colour,
                linewidth=2,
                marker="o",
                markersize=5,
                label=name,
            )
        ax.set_title(
            f"{title} Brier score by race stage, {result.split.test}",
            color=TEXT,
            fontsize=11,
            loc="left",
        )
        ax.set_xlabel("Share of race distance completed", color=MUTED, fontsize=9)
        ax.set_ylabel("Brier score (lower is better)", color=MUTED, fontsize=9)
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --- Markdown --------------------------------------------------------------------


def _table(df: pd.DataFrame) -> str:
    """A GitHub Markdown table, without needing the optional `tabulate` package."""
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(_cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) < 10 else f"{value:.2f}"
    return str(value)


def _score_table(result: TaskResult, season: int) -> str:
    scores = result.scores()
    scores = scores[scores["season"] == season].drop(columns="season")
    keep = [
        "model",
        "brier_win",
        "brier_podium",
        "log_loss_win",
        "position_mae",
        "winner_hit_rate",
        "ece_podium",
    ]
    return _table(scores[keep].rename(columns=METRIC_LABELS).rename(columns={"model": "Model"}))


def _comparison_table(diffs: dict[str, list[PairedDifference]]) -> str:
    rows = []
    for name, items in diffs.items():
        for d in items:
            verdict = (
                ("better" if d.diff < 0 else "worse") if d.significant else "no clear difference"
            )
            rows.append(
                {
                    "Model": name,
                    "Metric": METRIC_LABELS[d.metric],
                    "Model score": d.model,
                    "Baseline": d.baseline,
                    "Improvement": f"{d.relative:+.1%}",
                    "95% interval of difference": f"[{d.low:+.4f}, {d.high:+.4f}]",
                    "Verdict": verdict,
                }
            )
    return _table(pd.DataFrame(rows))


def _headline(task: str, result: TaskResult, diffs: dict[str, list[PairedDifference]]) -> str:
    """One sentence per learned model on podium Brier score, the headline metric."""
    when = "Before the race" if task == "pre_race" else "Lap by lap"
    base = result.baseline.lower()
    parts = []
    for name, items in diffs.items():
        d = next(x for x in items if x.metric == "brier_podium")
        interval = f"95% interval of the difference {d.low:+.4f} to {d.high:+.4f}"
        if d.significant and d.diff < 0:
            parts.append(f"{name} beat the {base} by {d.relative:.1%} ({interval})")
        elif d.significant:
            parts.append(f"{name} was worse than the {base} by {-d.relative:.1%} ({interval})")
        else:
            parts.append(f"{name} was not clearly different from the {base} ({interval})")
    return f"- **{when}**, podium Brier score on {result.split.test}: " + "; ".join(parts) + "."


def _markdown(
    pre: TaskResult,
    lap: TaskResult,
    comparisons: dict[str, dict[str, list[PairedDifference]]],
    pre_race: pd.DataFrame,
    is_demo: bool,
    saved: dict[str, str],
) -> str:
    split = pre.split
    races = pre_race.groupby("season")["race_id"].nunique()
    lines = ["# Model evaluation", ""]
    if is_demo:
        lines += [
            "> **Demo data.** These numbers come from the synthetic demo warehouse, not real",
            "> races. They show the pipeline works; they say nothing about real F1.",
            "",
        ]
    lines += [
        f"Season split: {split.describe()}. Settings were chosen on {split.validation}; "
        f"{split.test} was used once, for the numbers below.",
        f"Races per season: {', '.join(f'{s}: {n}' for s, n in races.items())}.",
        "",
        "## Headline",
        "",
        _headline("pre_race", pre, comparisons["pre_race"]),
        _headline("lap", lap, comparisons["lap"]),
        "",
        f"The test season has {races.get(split.test, 0)} races, so intervals are wide: a",
        "difference only counts when its whole interval sits on one side of zero.",
        "",
        'Lower is better for every score except "Favourite won" (the share of races the',
        "highest win probability got right). Brier score is the mean squared error of a",
        "probability. Position MAE is the mean number of places the predicted order is off.",
        "",
        "## Pre-race model",
        "",
        f"### Test season {split.test}",
        "",
        _score_table(pre, split.test),
        "",
        "### Compared with the baseline",
        "",
        "Differences are model minus baseline (negative is better). Intervals come from",
        "resampling whole races 2,000 times, so they reflect how few races a season has.",
        "",
        _comparison_table(comparisons["pre_race"]),
        "",
        "### Calibration",
        "",
        "If the model says 30%, it should happen about 30% of the time: points on the",
        f"dashed line are well calibrated. Bins with fewer than {MIN_BIN_ROWS} predictions",
        "are hidden.",
        "",
        "![Calibration](figures/calibration_pre_race.png)",
        "",
    ]
    boosted = pre.models.get("XGBoost")
    if isinstance(boosted, BoostedModel):
        importance = boosted.feature_importance().head(8)
        lines += [
            "### What the XGBoost model leans on",
            "",
            "Share of total gain, averaged over the win, podium and position models.",
            "",
            _table(
                pd.DataFrame(
                    {"Feature": importance.index, "Share of gain": [f"{v:.1%}" for v in importance]}
                )
            ),
            "",
        ]
    for season in split.holdout:
        if (pre.baseline, season) in pre.predictions:
            lines += [
                f"### Holdout season {season}",
                "",
                "Scored separately: a new rules era, where past form should matter less.",
                "",
                _score_table(pre, season),
                "",
            ]
    lines += [
        "## Lap-by-lap model",
        "",
        "Scored on every lap of every race in the test season.",
        "",
        f"### Test season {split.test}",
        "",
        _score_table(lap, split.test),
        "",
        "### Compared with the baseline",
        "",
        _comparison_table(comparisons["lap"]),
        "",
        "### By race stage",
        "",
        "![Brier by stage](figures/lap_brier_by_stage.png)",
        "",
        _table(
            lap.stage_scores.rename(columns={"model": "Model", "stage": "Stage", **METRIC_LABELS})
        ),
        "",
    ]
    for season in split.holdout:
        if (lap.baseline, season) in lap.predictions:
            lines += [f"### Holdout season {season}", "", _score_table(lap, season), ""]
    lines += [
        "## Settings chosen on the validation season",
        "",
        _table(
            pd.DataFrame(
                [
                    {
                        "Task": task,
                        "Model": name,
                        "Settings": json.dumps(params) if params else "none",
                    }
                    for task, result in (("pre-race", pre), ("lap", lap))
                    for name, params in result.chosen.items()
                ]
            )
        ),
        "",
        "## Saved models",
        "",
        f"`models/pre_race.joblib` ({saved['pre_race']}) and `models/lap.joblib` "
        f"({saved['lap']}): the learned model with the best validation score, refit on "
        f"every season up to {split.test}. The live predictor in stage 4 loads these.",
        "",
        f"Features: {len(PRE_RACE_FEATURES)} pre-race, {len(LAP_FEATURES)} per lap. See "
        "`docs/model/README.md` for definitions and known limitations.",
        "",
    ]
    return "\n".join(lines)
