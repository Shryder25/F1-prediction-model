from __future__ import annotations

from pathlib import Path

import pandas as pd

from f1pred import demo


def test_generate_is_deterministic(tmp_path: Path) -> None:
    demo.generate(tmp_path / "a", seed=3)
    demo.generate(tmp_path / "b", seed=3)
    for table in ("results", "laps", "pit_stops", "fastf1_laps"):
        part = Path(table) / "season=2024" / "round=01" / "data.parquet"
        a = pd.read_parquet(tmp_path / "a" / part)
        b = pd.read_parquet(tmp_path / "b" / part)
        pd.testing.assert_frame_equal(a, b)


def test_results_are_a_valid_classification(tmp_path: Path) -> None:
    demo.generate(tmp_path)
    results = pd.read_parquet(tmp_path / "results" / "season=2023" / "round=02" / "data.parquet")
    assert sorted(results["position"]) == list(range(1, len(demo.DRIVERS) + 1))
    assert (results["grid"] == 0).sum() == 1  # one pit lane start in round 2
    classified = results[results["position_text"] != "R"]
    assert classified["points"].sum() <= sum(demo.POINTS)
