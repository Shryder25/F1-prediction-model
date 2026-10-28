"""Stage 3 tests: model plumbing on hand-made frames, then end to end on demo data."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import pytest

from f1pred import cli, demo, transform
from f1pred.model.dataset import (
    LABELS,
    LAP_FEATURES,
    PRE_RACE_FEATURES,
    SeasonSplit,
    load_laps,
    load_pre_race,
)
from f1pred.model.estimators import PRE_RACE_GROUPS, GridBaseline, finalize
from f1pred.model.metrics import brier, calibration_table, paired_bootstrap, position_mae
from f1pred.model.report import build_report

DEMO_SEASONS = range(2020, 2026)


# --- Unit tests --------------------------------------------------------------------


def _race_frame(n_races: int = 3, drivers: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for race in range(n_races):
        order = rng.permutation(drivers) + 1
        for d in range(drivers):
            rows.append(
                {
                    "race_id": 202401 + race,
                    "season": 2024,
                    "driver_id": f"d{d}",
                    "grid_position": d + 1,
                    "label_finishing_order": order[d],
                    "label_win": int(order[d] == 1),
                    "label_podium": int(order[d] <= 3),
                }
            )
    return pd.DataFrame(rows)


def test_finalize_makes_race_consistent_predictions() -> None:
    df = _race_frame()
    rng = np.random.default_rng(1)
    out = finalize(
        df,
        PRE_RACE_GROUPS,
        rng.random(len(df)),
        rng.random(len(df)) * 3,  # raw scores above 1 must still be capped
        rng.random(len(df)),
        tie_break="grid_position",
    )
    joined = pd.concat([df, out], axis=1)
    by_race = joined.groupby("race_id")
    assert by_race["p_win"].sum().to_numpy() == pytest.approx(1.0)
    assert by_race["p_podium"].sum().to_numpy() == pytest.approx(3.0)
    assert joined["p_podium"].max() <= 1.0
    for _, race in by_race:
        assert sorted(race["pred_rank"]) == list(range(1, len(race) + 1))


def test_podium_probabilities_with_fewer_than_three_cars() -> None:
    df = _race_frame(n_races=1, drivers=2)
    out = finalize(
        df, PRE_RACE_GROUPS, np.ones(2), np.array([0.2, 0.9]), np.zeros(2), "grid_position"
    )
    assert out["p_podium"].tolist() == pytest.approx([1.0, 1.0])


def test_grid_baseline_predicts_the_grid_order() -> None:
    df = _race_frame(n_races=20, drivers=6)
    pred = GridBaseline().fit(df).predict(df)
    assert (pred["pred_rank"] == df["grid_position"]).all()
    assert pred["p_win"].between(0, 1).all()


def test_season_split_follows_the_plan_and_rejects_leaks() -> None:
    split = SeasonSplit.latest(list(range(2018, 2027)))
    assert split == SeasonSplit(tuple(range(2018, 2024)), 2024, 2025, (2026,))
    assert split.describe() == "train 2018-2023, validate 2024, test 2025, holdout 2026"
    with pytest.raises(ValueError):
        SeasonSplit((2018, 2024), 2024, 2025)  # validation season also in training
    with pytest.raises(ValueError):
        SeasonSplit((2018,), 2025, 2024)  # test before validation
    with pytest.raises(ValueError, match="at least 3 seasons"):
        SeasonSplit.latest([2023, 2024])


def test_labels_are_never_features() -> None:
    assert not set(LABELS) & set(PRE_RACE_FEATURES)
    assert not set(LABELS) & set(LAP_FEATURES)


def test_metrics_on_known_values() -> None:
    y = pd.Series([1, 0, 0, 1])
    p = pd.Series([1.0, 0.0, 0.5, 0.5])
    assert brier(y, p) == pytest.approx((0 + 0 + 0.25 + 0.25) / 4)
    assert position_mae(pd.Series([1, 2, 3]), pd.Series([2, 2, 1])) == pytest.approx(1.0)
    table = calibration_table(y, p, bins=2)
    assert table["n"].sum() == 4


def test_paired_bootstrap_of_identical_predictions_is_zero() -> None:
    df = _race_frame(n_races=10)
    pred = pd.concat([df, GridBaseline().fit(df).predict(df)], axis=1)
    for diff in paired_bootstrap(pred, pred, n_boot=200):
        assert diff.diff == 0
        assert diff.low == diff.high == 0
        assert not diff.significant


# --- End to end on the demo warehouse --------------------------------------------


@pytest.fixture(scope="module")
def warehouse_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data_dir = tmp_path_factory.mktemp("model_demo")
    demo.build(data_dir, seasons=DEMO_SEASONS)
    path = data_dir / "warehouse.duckdb"
    assert transform.run_dbt(path, ["build"]), "dbt build failed: see the log above"
    return path


def test_feature_tables_cover_every_start_and_lap(warehouse_path: Path) -> None:
    with duckdb.connect(str(warehouse_path), read_only=True) as con:
        pre = load_pre_race(con)
        laps = load_laps(con)
        starts = con.execute("SELECT count(*) FROM marts.fct_race_results").fetchone()
        lap_rows = con.execute("SELECT count(*) FROM marts.fct_laps").fetchone()
    assert starts is not None and lap_rows is not None
    assert len(pre) == starts[0]
    assert len(laps) == lap_rows[0]
    assert pre.groupby("race_id")["label_win"].sum().eq(1).all()


def test_pre_race_features_only_look_backwards(warehouse_path: Path) -> None:
    with duckdb.connect(str(warehouse_path), read_only=True) as con:
        pre = load_pre_race(con)
    first_race = pre["race_id"].min()
    debut = pre[pre["race_id"] == first_race]
    assert debut["driver_finish_last5"].isna().all()
    assert (debut["driver_career_starts"] == 0).all()
    # Standings before round 1 do not exist yet.
    assert pre.loc[pre["round"] == 1, "championship_position_before"].isna().all()
    # Form at a driver's second start is exactly their first result.
    second = pre[pre["race_id"] == sorted(pre["race_id"].unique())[1]].set_index("driver_id")
    first = pre[pre["race_id"] == first_race].set_index("driver_id")
    pd.testing.assert_series_equal(
        second["driver_finish_last5"],
        first["label_finishing_order"].astype(float),
        check_names=False,
    )


def test_evaluation_report_end_to_end(warehouse_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "report"
    metrics = build_report(warehouse_path, out, quick=True)

    assert metrics["demo_data"] is True
    assert metrics["split"]["test"] == max(DEMO_SEASONS)
    report = (out / "report.md").read_text()
    assert "Demo data" in report
    assert "Grid baseline" in report and "XGBoost" in report
    for figure in ("calibration_pre_race.png", "lap_brier_by_stage.png"):
        assert (out / "figures" / figure).stat().st_size > 10_000
    assert json.loads((out / "metrics.json").read_text())["rows"]["pre_race"] > 0

    # Every model's test-season scores are present and sane.
    scores = pd.DataFrame(metrics["scores"]["pre_race"])
    assert set(scores["model"]) == {"Grid baseline", "Logistic regression", "XGBoost"}
    assert scores["brier_win"].between(0, 0.25).all()

    # Late in a race the current order is far better than the starting grid.
    pre = scores.set_index("model").loc["Grid baseline", "position_mae"]
    stages = pd.DataFrame(metrics["scores"]["lap_by_stage"])
    late = stages[(stages["model"] == "Current position baseline") & (stages["stage"] == "75-100%")]
    assert late["position_mae"].iloc[0] < pre

    # The saved models load and give coherent predictions for a race.
    with duckdb.connect(str(warehouse_path), read_only=True) as con:
        race = load_pre_race(con).query(f"race_id == {max(DEMO_SEASONS) * 100 + 1}")
    model = joblib.load(out / "models" / "pre_race.joblib")
    pred = model.predict(race)
    assert pred["p_win"].sum() == pytest.approx(1.0)


def test_evaluate_cli_with_an_explicit_split(
    warehouse_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("F1PRED_DATA_DIR", str(warehouse_path.parent))
    out = tmp_path / "cli_report"
    args = ["evaluate", "--train", "2020-2022", "--validation", "2023", "--test", "2024"]
    assert cli.main([*args, "--holdout", "2025", "--quick", "--out", str(out)]) == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["split"]["holdout"] == [2025]
    assert cli.main(["evaluate", "--train", "2020-2022"]) == 2
