"""Strategy metrics on hand-built races where the right answer is known on paper,
then end to end on the demo warehouse."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from f1pred import demo, strategy, transform
from f1pred.cli import main
from f1pred.strategy import analyse_race
from f1pred.strategy.battles import BATTLE_COLUMNS
from f1pred.strategy.safety_car import SC_STOP_COLUMNS
from f1pred.strategy.stints import STINT_COLUMNS
from f1pred.strategy.windows import WINDOW_COLUMNS
from f1pred.transform import DBT_PROJECT_DIR
from tests.strategy_fixtures import Car, build_race

DEG = {"SOFT": 0.10, "MEDIUM": 0.06, "HARD": 0.03}


# --- The contract the post-race debrief codes against -------------------------------


def test_columns_match_the_debrief_contract() -> None:
    assert STINT_COLUMNS[:11] == [
        "race_id", "driver_id", "stint", "compound", "start_lap", "end_lap", "laps",
        "clean_laps", "fuel_corrected_pace_s", "degradation_s_per_lap", "degradation_r2",
    ]  # fmt: skip
    assert "is_usable" in STINT_COLUMNS
    assert BATTLE_COLUMNS == [
        "race_id", "attacker_id", "defender_id", "kind", "attacker_pit_lap",
        "defender_pit_lap", "gap_before_s", "gap_after_s", "gain_s", "succeeded",
    ]  # fmt: skip
    assert SC_STOP_COLUMNS == [
        "race_id", "driver_id", "lap", "track_status", "pit_loss_s", "green_pit_loss_s",
        "time_saved_s", "positions_gained",
    ]  # fmt: skip
    assert WINDOW_COLUMNS == [
        "race_id", "driver_id", "stint", "compound", "actual_pit_lap", "optimal_pit_lap",
        "window_open", "window_close", "verdict", "time_lost_s",
    ]  # fmt: skip


def test_a_race_without_tyre_data_gives_empty_frames_with_the_columns() -> None:
    laps, stops = build_race(
        [Car("a", [("SOFT", 20), ("HARD", 50)])], total_laps=50, degradation=DEG
    )
    laps["has_tyre_data"] = False
    for col in ("compound", "stint", "tyre_age_laps"):
        laps[col] = None
    result = analyse_race(laps, stops)
    assert result.stints.empty and list(result.stints.columns) == STINT_COLUMNS
    assert result.pit_windows.empty and list(result.pit_windows.columns) == WINDOW_COLUMNS
    empty = analyse_race(laps.iloc[0:0])
    assert all(frame.empty for frame in empty.frames().values())


def test_fuel_correction_matches_the_dbt_var() -> None:
    text = (DBT_PROJECT_DIR / "dbt_project.yml").read_text()
    match = re.search(r"fuel_correction_s_per_lap:\s*([0-9.]+)", text)
    assert match is not None
    assert float(match.group(1)) == strategy.DEFAULT_CONFIG.fuel_correction_s_per_lap


# --- Stint pace and degradation ----------------------------------------------------


def test_stint_degradation_is_recovered_after_fuel_correction() -> None:
    traffic = {12: 9.0, 13: 7.0}  # stuck behind a backmarker: must be ignored
    laps, stops = build_race(
        [Car("a", [("SOFT", 20), ("HARD", 50)], extra_s=traffic)],
        total_laps=50,
        degradation=DEG,
        sc_laps=[30, 31],
    )
    stints = analyse_race(laps, stops).stints.set_index("stint")
    soft, hard = stints.loc[1], stints.loc[2]
    assert soft["degradation_s_per_lap"] == pytest.approx(0.10, abs=1e-3)
    assert hard["degradation_s_per_lap"] == pytest.approx(0.03, abs=1e-3)
    assert soft["degradation_r2"] > 0.99
    # Lap 1, the in-lap, two traffic laps: 20 - 4 = 16 clean laps.
    assert soft["clean_laps"] == 16
    # Out-lap and two SC laps dropped from the hard stint.
    assert hard["clean_laps"] == 30 - 3
    assert (soft["start_lap"], soft["end_lap"], soft["laps"]) == (1, 20, 20)
    # Fuel-corrected pace is the median clean lap on an empty tank: base + deg x the
    # median clean tyre age (ages 2-11 and 14-19, so 9.5).
    assert soft["fuel_corrected_pace_s"] == pytest.approx(90.0 + 0.10 * 9.5, abs=1e-3)
    assert bool(soft["is_usable"]) and bool(hard["is_usable"])


# --- Undercut and overcut ----------------------------------------------------------


def test_undercut_that_works() -> None:
    # Same car pace. "attacker" starts 1 s behind, pits 2 laps earlier on to fresh tyres.
    laps, stops = build_race(
        [
            Car("defender", [("SOFT", 12), ("HARD", 50)]),
            Car("attacker", [("SOFT", 10), ("HARD", 50)], start_gap_s=1.0),
        ],
        total_laps=50,
        degradation=DEG,
    )
    battles = analyse_race(laps, stops).pit_battles
    assert len(battles) == 1
    b = battles.iloc[0]
    assert (b["attacker_id"], b["defender_id"], b["kind"]) == ("attacker", "defender", "undercut")
    assert (b["attacker_pit_lap"], b["defender_pit_lap"]) == (10, 12)
    assert b["gap_before_s"] == pytest.approx(1.0, abs=1e-6)
    # Laps 11-12: attacker on hard age 1-2 vs defender on soft age 11-12.
    # Lap 13: the defender's out-lap on hard age 1 vs the attacker's hard age 3.
    expected_gain = (0.10 * 11 - 0.03) + (0.10 * 12 - 0.06) + (0.03 * 1 - 0.03 * 3)
    assert b["gain_s"] == pytest.approx(expected_gain, abs=1e-3)
    assert b["gap_after_s"] == pytest.approx(expected_gain - 1.0, abs=1e-3)
    assert bool(b["succeeded"])


def test_overcut_when_the_new_tyre_is_slow_to_pay_off() -> None:
    # The defender pits first on to a hard that is 0.8 s a lap slower when new. The
    # attacker stays out on a soft that barely degrades and comes out ahead.
    laps, stops = build_race(
        [
            Car("defender", [("SOFT", 10), ("HARD", 50)]),
            Car("attacker", [("SOFT", 13), ("HARD", 50)], start_gap_s=1.0),
        ],
        total_laps=50,
        degradation={"SOFT": 0.02, "HARD": 0.03},
        compound_offset={"HARD": 0.8},
    )
    battles = analyse_race(laps, stops).pit_battles
    assert len(battles) == 1
    b = battles.iloc[0]
    assert (b["attacker_id"], b["kind"]) == ("attacker", "overcut")
    assert (b["attacker_pit_lap"], b["defender_pit_lap"]) == (13, 10)
    assert bool(b["succeeded"]) and b["gap_after_s"] > 0


def test_cars_too_far_apart_or_pitting_together_are_not_battles() -> None:
    laps, stops = build_race(
        [
            Car("a", [("SOFT", 12), ("HARD", 50)]),
            Car("b", [("SOFT", 10), ("HARD", 50)], start_gap_s=8.0),  # 8 s back
            Car("c", [("SOFT", 12), ("HARD", 50)], start_gap_s=0.5),  # same lap as a
        ],
        total_laps=50,
        degradation=DEG,
    )
    battles = analyse_race(laps, stops).pit_battles
    assert battles.empty


def test_a_pit_cycle_under_the_safety_car_is_not_an_undercut() -> None:
    laps, stops = build_race(
        [
            Car("defender", [("SOFT", 12), ("HARD", 50)]),
            Car("attacker", [("SOFT", 10), ("HARD", 50)], start_gap_s=1.0),
        ],
        total_laps=50,
        degradation=DEG,
        sc_laps=[11],
    )
    assert analyse_race(laps, stops).pit_battles.empty


# --- Safety car stops --------------------------------------------------------------


def test_safety_car_stop_saves_about_half_the_pit_loss() -> None:
    cars = [
        Car("lucky", [("MEDIUM", 21), ("HARD", 50)]),  # pits under the SC
        Car("b", [("MEDIUM", 15), ("HARD", 50)], start_gap_s=5.0),
        Car("c", [("MEDIUM", 28), ("HARD", 50)], start_gap_s=10.0),
        Car("d", [("MEDIUM", 32), ("HARD", 50)], start_gap_s=15.0),
    ]
    laps, stops = build_race(
        cars, total_laps=50, degradation=DEG, sc_laps=[20, 21, 22], pit_loss_s=22.0
    )
    sc = analyse_race(laps, stops).safety_car_stops
    assert len(sc) == 1
    row = sc.iloc[0]
    assert (row["driver_id"], row["lap"], row["track_status"]) == ("lucky", 21, "SC")
    # It still pays 11 s in the pit lane, minus the fresh tyre's out-lap advantage.
    assert row["pit_loss_s"] == pytest.approx(11.0, abs=1.5)
    assert row["green_pit_loss_s"] == pytest.approx(22.0, abs=1.5)
    # Pit loss is measured against the cars that stayed out, whose tyres are older or
    # newer than the stopping car's, so green losses spread by about +-1 s.
    assert 9.5 <= row["time_saved_s"] <= 14.0
    assert row["time_saved_s"] == pytest.approx(row["green_pit_loss_s"] - row["pit_loss_s"])
    # At the end of its out-lap it leads c by 0.3 s, d by 5.3 s and b by 10.2 s. Paying
    # the full green loss would have dropped it behind all three.
    assert row["positions_gained"] == 3
    # The tyre-only window model does not judge a stop the safety car decided.
    windows = analyse_race(laps, stops).pit_windows.set_index("driver_id")
    assert windows.loc["lucky", "verdict"] == "safety car"
    assert windows.loc["b", "verdict"] != "safety car"


# --- Pit windows -------------------------------------------------------------------


def test_pit_window_finds_the_best_lap_and_judges_each_stop() -> None:
    # Soft then hard, no pace offset: T(s) is smallest near s = N x d2 / (d1 + d2) = 15.
    laps, stops = build_race(
        [
            Car("on_time", [("SOFT", 15), ("HARD", 60)]),
            Car("early", [("SOFT", 7), ("HARD", 60)], start_gap_s=20),
            Car("late", [("SOFT", 30), ("HARD", 60)], start_gap_s=40),
        ],
        total_laps=60,
        degradation={"SOFT": 0.09, "HARD": 0.03},
    )
    windows = analyse_race(laps, stops).pit_windows.set_index("driver_id")
    on_time = windows.loc["on_time"]
    assert abs(int(on_time["optimal_pit_lap"]) - 15) <= 1
    assert on_time["verdict"] == "in window"
    assert on_time["window_open"] <= 15 <= on_time["window_close"]
    assert windows.loc["early", "verdict"] == "early"
    assert windows.loc["late", "verdict"] == "late"
    assert windows.loc["late", "time_lost_s"] > windows.loc["on_time", "time_lost_s"]
    # The window is the same race and tyres for everyone, so it agrees across drivers.
    assert windows["optimal_pit_lap"].sub(15).abs().max() <= 1


def test_a_retirement_does_not_drag_the_pit_window_earlier() -> None:
    laps, stops = build_race(
        [Car("a", [("SOFT", 15), ("HARD", 30)])],  # retired on lap 30 of 60
        total_laps=60,
        degradation={"SOFT": 0.09, "HARD": 0.03},
    )
    window = analyse_race(laps, stops).pit_windows.iloc[0]
    assert abs(int(window["optimal_pit_lap"]) - 15) <= 1


# --- Season summaries --------------------------------------------------------------


def test_success_rates_and_summaries() -> None:
    battles = pd.DataFrame(
        {
            "kind": ["undercut", "undercut", "undercut", "overcut"],
            "succeeded": [True, True, False, False],
            "gain_s": [2.0, 1.0, -1.0, -0.5],
        }
    )
    rates = strategy.undercut_success_rates(battles).set_index("kind")
    assert rates.loc["undercut", "attempts"] == 3
    assert rates.loc["undercut", "success_rate"] == pytest.approx(2 / 3)
    assert rates.loc["overcut", "success_rate"] == 0
    assert strategy.undercut_success_rates(battles.iloc[0:0]).empty


# --- End to end on the demo warehouse ----------------------------------------------


@pytest.fixture(scope="module")
def demo_warehouse(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data_dir = tmp_path_factory.mktemp("strategy-demo")
    demo.build(data_dir)
    path = data_dir / "warehouse.duckdb"
    assert transform.run_dbt(path, ["build"]), "dbt build failed: see the log above"
    with duckdb.connect(str(path)) as con:
        counts = strategy.build_tables(con)
    assert all(counts[name] > 0 for name in ("stints", "pit_battles", "pit_windows"))
    return path


@pytest.fixture
def con(demo_warehouse: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    with duckdb.connect(str(demo_warehouse), read_only=True) as connection:
        yield connection


def test_demo_degradation_recovers_the_true_rates(con: duckdb.DuckDBPyConnection) -> None:
    stints = strategy.load_tables(con).stints
    usable = stints.loc[stints["is_usable"]]
    multiplier = {c.circuit_id: c.deg_multiplier for c in demo.CIRCUITS}
    usable = usable.assign(
        normalised=usable["degradation_s_per_lap"] / usable["circuit_id"].map(multiplier)
    )
    for compound, truth in demo.COMPOUND_DEG.items():
        estimate = usable.loc[usable["compound"] == compound, "normalised"].median()
        assert estimate == pytest.approx(truth, rel=0.2), compound


def test_demo_safety_car_stops_save_time(con: duckdb.DuckDBPyConnection) -> None:
    sc = strategy.load_tables(con).safety_car_stops
    assert not sc.empty and set(sc["track_status"]) == {"SC"}
    # Demo SC stops cost half the circuit's pit loss, so each saves roughly the other half.
    lane = {c.circuit_id: c.pit_lane_s - 2 for c in demo.CIRCUITS}
    expected = sc["circuit_id"].map(lane) * (1 - demo.SC_PIT_COST_SHARE)
    assert (sc["time_saved_s"] - expected).abs().median() < 1.5


def test_race_strategy_matches_the_season_tables(con: duckdb.DuckDBPyConnection) -> None:
    season = strategy.load_tables(con)
    race_id = int(season.pit_battles["race_id"].iloc[0])
    one = strategy.race_strategy(con, race_id)
    expected = season.pit_battles.loc[season.pit_battles["race_id"] == race_id, BATTLE_COLUMNS]
    pd.testing.assert_frame_equal(
        one.pit_battles.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )
    assert set(one.stints["race_id"]) == {race_id}


def test_headline_findings(con: duckdb.DuckDBPyConnection) -> None:
    findings = strategy.headline_findings(strategy.load_tables(con, seasons=[2024]))
    text = "\n".join(findings)
    assert "Undercuts worked" in text
    assert "pit window" in text
    assert "degradation" in text


def test_strategy_cli(
    demo_warehouse: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("F1PRED_DATA_DIR", str(demo_warehouse.parent))
    assert main(["strategy"]) == 0
    out = capsys.readouterr().out
    assert "strategy.pit_battles" in out and "Undercuts worked" in out
    assert main(["strategy", "--race", "202403"]) == 0
    assert "== safety_car_stops" in capsys.readouterr().out
