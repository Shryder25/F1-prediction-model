"""Post-race debrief: predicted vs actual on the demo warehouse, strategy section from
fixture frames, Markdown/HTML rendering, the Pages site, the CLI and the Dagster asset."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from dagster import materialize

from f1pred import cli
from f1pred.debrief import Debrief, build_debrief, latest_race_id
from f1pred.debrief.build import locked_in_lap, strategy_section
from f1pred.debrief.render import to_html, to_markdown
from f1pred.debrief.site import build_site, load_debriefs, write_debrief
from f1pred.orchestration.definitions import (
    DataLake,
    debrief_covers_every_starter,
    debrief_has_strategy_when_tyre_data,
    race_debrief,
)
from tests.conftest import MODEL_DEMO_SEASONS
from tests.test_live import model_dir, models  # noqa: F401  (shared fixtures)

LAST_SEASON = MODEL_DEMO_SEASONS[-1]
RACE = LAST_SEASON * 100 + 3  # the demo's safety car race


@pytest.fixture(scope="module")
def debrief_models(model_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:  # noqa: F811
    """Saved models plus the metrics.json `f1pred evaluate` writes next to them."""
    root = tmp_path_factory.mktemp("evaluate")
    shutil.copytree(model_dir, root / "models")
    split = {"train": list(MODEL_DEMO_SEASONS[:-2]), "validation": LAST_SEASON - 1}
    split |= {"test": LAST_SEASON - 1, "holdout": [LAST_SEASON]}
    (root / "metrics.json").write_text(json.dumps({"split": split}))
    return root / "models"


@pytest.fixture(scope="module")
def debrief(demo_model_warehouse: Path, debrief_models: Path) -> Debrief:
    return build_debrief(demo_model_warehouse, RACE, debrief_models)


def _results(warehouse: Path, race_id: int) -> pd.DataFrame:
    with duckdb.connect(str(warehouse), read_only=True) as con:
        return con.execute(
            "SELECT driver_id, grid_position, classified_position "
            "FROM marts.fct_race_results WHERE race_id = ?",
            [race_id],
        ).df()


def test_predicted_vs_actual(debrief: Debrief, demo_model_warehouse: Path) -> None:
    truth = _results(demo_model_warehouse, RACE).set_index("driver_id")
    assert len(debrief.results) == len(truth)
    assert sorted(r.predicted_rank for r in debrief.results) == list(range(1, len(truth) + 1))
    for r in debrief.results:
        expected = truth.at[r.driver_id, "classified_position"]
        assert r.finish == (None if pd.isna(expected) else int(expected))
        assert r.grid == truth.at[r.driver_id, "grid_position"]
        if r.finish is not None:
            assert r.beat_prediction_by == r.predicted_rank - r.finish
    assert abs(sum(r.p_win or 0 for r in debrief.results) - 1) < 1e-3
    s = debrief.score
    assert s.source == "saved models" and not s.in_sample  # last season is the holdout
    assert 0 <= s.podium_hits <= 3 and s.brier_win is not None
    assert debrief.race.is_demo and debrief.headlines


def test_win_probability_trace(debrief: Debrief) -> None:
    assert 1 <= len(debrief.win_probability) <= 5
    for series in debrief.win_probability:
        # Lap 0 (pre-race) then one value per lap.
        assert len(series.p_win) == debrief.race.total_laps + 1
        assert all(0 <= p <= 1 for p in series.p_win)


def test_in_sample_flag(demo_model_warehouse: Path, debrief_models: Path) -> None:
    trained_race = (LAST_SEASON - 2) * 100 + 1
    d = build_debrief(demo_model_warehouse, trained_race, debrief_models)
    assert d.score.in_sample
    assert "in-sample" in to_markdown(d)


def test_without_models_falls_back_to_grid(demo_model_warehouse: Path, tmp_path: Path) -> None:
    d = build_debrief(demo_model_warehouse, RACE, tmp_path / "no-models")
    assert d.score.source.startswith("grid order")
    assert all(r.predicted_rank == r.grid for r in d.results)
    assert d.score.position_mae == d.score.grid_position_mae
    assert d.score.brier_win is None and not d.win_probability


def test_latest_race(demo_model_warehouse: Path) -> None:
    assert latest_race_id(demo_model_warehouse) // 100 == LAST_SEASON


def test_unknown_race(demo_model_warehouse: Path) -> None:
    with pytest.raises(KeyError):
        build_debrief(demo_model_warehouse, 199901, None)


def test_locked_in_lap() -> None:
    trace = [{"a": 0.6, "b": 0.4}, {"a": 0.3, "b": 0.7}, {"a": 0.6, "b": 0.4}, {"a": 0.9}]
    assert locked_in_lap(trace, "a") == 2
    assert locked_in_lap(trace, "b") is None
    assert locked_in_lap([{"a": 0.5, "b": 0.5}], "a") == 0  # a tie still counts
    assert locked_in_lap([], "a") is None


# --- strategy section, from frames shaped like f1pred.strategy.race_strategy -----------

ENTRIES = pd.DataFrame(
    {"driver_id": ["ver", "nor", "lec"], "code": ["VER", "NOR", "LEC"], "finish": [1, 2, None]}
)


def _stints() -> pd.DataFrame:
    rows = [
        ("ver", 1, "MEDIUM", 1, 20, 0.05, True),
        ("ver", 2, "HARD", 21, 57, 0.03, True),
        ("nor", 1, "MEDIUM", 1, 18, 0.07, True),
        ("nor", 2, "HARD", 19, 57, 0.02, True),
        ("lec", 1, "SOFT", 1, 12, 0.11, True),
        ("lec", 2, "HARD", 13, 30, None, False),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "driver_id",
            "stint",
            "compound",
            "start_lap",
            "end_lap",
            "degradation_s_per_lap",
            "is_usable",
        ],
    )
    return df.assign(
        race_id=1,
        laps=df["end_lap"] - df["start_lap"] + 1,
        clean_laps=10,
        fuel_corrected_pace_s=92.0,
        degradation_r2=0.5,
    )


PIT_BATTLES = pd.DataFrame(
    [
        {
            "race_id": 1,
            "attacker_id": "nor",
            "defender_id": "ver",
            "kind": "undercut",
            "attacker_pit_lap": 18,
            "defender_pit_lap": 20,
            "gap_before_s": 1.4,
            "gap_after_s": -0.6,
            "gain_s": 0.8,
            "succeeded": False,
        }
    ]
)
SC_STOPS = pd.DataFrame(
    [
        {
            "race_id": 1,
            "driver_id": "lec",
            "lap": 12,
            "track_status": "SC",
            "pit_loss_s": 11.0,
            "green_pit_loss_s": 22.0,
            "time_saved_s": 11.0,
            "positions_gained": 0,
        }
    ]
)
PIT_WINDOWS = pd.DataFrame(
    [
        {
            "race_id": 1,
            "driver_id": "nor",
            "stint": 1,
            "compound": "MEDIUM",
            "actual_pit_lap": 18,
            "optimal_pit_lap": 21,
            "window_open": 20,
            "window_close": 24,
            "verdict": "early",
            "time_lost_s": 0.9,
        },
        {
            "race_id": 1,
            "driver_id": "lec",
            "stint": 1,
            "compound": "SOFT",
            "actual_pit_lap": 12,
            "optimal_pit_lap": 16,
            "window_open": 14,
            "window_close": 18,
            "verdict": "safety car",
            "time_lost_s": 0.0,
        },
    ]
)


def test_strategy_section() -> None:
    st = strategy_section(_stints(), PIT_BATTLES, SC_STOPS, PIT_WINDOWS, ENTRIES)
    assert st.available
    assert [d.code for d in st.drivers] == ["VER", "NOR", "LEC"]  # by result, DNF last
    nor = st.drivers[1]
    assert nor.stops == 1 and [s.compound for s in nor.stints] == ["MEDIUM", "HARD"]
    assert nor.pit_verdicts == ["lap 18: early (window 20-24)"]
    assert st.drivers[2].pit_verdicts == ["lap 12: under safety car"]
    # Unusable stints are left out; the highest degradation comes first.
    assert [c.compound for c in st.degradation] == ["SOFT", "MEDIUM", "HARD"]
    assert st.degradation[1].median_degradation_s_per_lap == pytest.approx(0.06)
    assert st.degradation[2].stints == 2
    battle = st.pit_battles[0]
    assert (battle.attacker, battle.defender, battle.succeeded) == ("NOR", "VER", False)
    assert st.safety_car_stops[0].code == "LEC"


def test_strategy_section_without_tyre_data() -> None:
    empty = _stints().iloc[0:0]
    st = strategy_section(empty, PIT_BATTLES.iloc[0:0], SC_STOPS.iloc[0:0], None, ENTRIES)
    assert not st.available and st.note and "FastF1" in st.note


def test_strategy_rendering(debrief: Debrief) -> None:
    st = strategy_section(_stints(), PIT_BATTLES, SC_STOPS, PIT_WINDOWS, ENTRIES)
    d = debrief.model_copy(update={"strategy": st})
    md = to_markdown(d)
    assert "| NOR | VER | undercut | 18 vs 20 | 1.4 s behind | 0.6 s behind | no |" in md
    assert "M 1-18 > H 19-57" in md
    html = to_html(d)
    assert 'class="tyres"' in html and "#da291c" in html  # soft stint bar
    assert "Stops under safety car" in html


# --- rendering and the site --------------------------------------------------------------


def test_markdown(debrief: Debrief) -> None:
    md = to_markdown(debrief)
    for heading in ("## Summary", "## Predicted vs actual", "## Strategy"):
        assert heading in md
    assert "Synthetic demo data" in md
    assert all(r.code in md for r in debrief.results)


def test_html_escapes(debrief: Debrief) -> None:
    race = debrief.race.model_copy(update={"race_name": "<script>x</script> GP"})
    html = to_html(debrief.model_copy(update={"race": race}))
    assert "<script>x" not in html and "&lt;script&gt;" in html
    assert html.startswith("<!doctype html>") and "<svg" in html


def test_site(debrief: Debrief, tmp_path: Path) -> None:
    src = tmp_path / "debriefs"
    write_debrief(debrief, src)
    assert (src / f"{debrief.slug}.json").exists()
    assert debrief.slug in (src / "README.md").read_text()
    assert load_debriefs(src) == [debrief]  # JSON round trip loses nothing

    pages = build_site(src, tmp_path / "site")
    index = (tmp_path / "site" / "index.html").read_text()
    assert f'href="{debrief.slug}.html"' in index
    assert (tmp_path / "site" / ".nojekyll").exists()
    assert len(pages) == 2


def test_cli(demo_model_warehouse: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("F1PRED_DATA_DIR", str(demo_model_warehouse.parent))
    out = tmp_path / "debriefs"
    assert cli.main(["debrief", str(RACE), "--out", str(out)]) == 0
    assert cli.main(["debrief", "--season", str(LAST_SEASON), "--out", str(out)]) == 0
    assert len(list(out.glob("*.json"))) == len(_season_races(demo_model_warehouse))
    assert cli.main(["debrief-site", "--src", str(out), "--out", str(tmp_path / "site")]) == 0
    assert cli.main(["debrief", "199901", "--out", str(out)]) == 2


def _season_races(warehouse: Path) -> list[int]:
    with duckdb.connect(str(warehouse), read_only=True) as con:
        rows = con.execute(
            "SELECT DISTINCT race_id FROM marts.dim_race WHERE season = ?", [LAST_SEASON]
        ).fetchall()
    return [int(r[0]) for r in rows]


def test_dagster_asset(demo_model_warehouse: Path, tmp_path: Path) -> None:
    lake = DataLake(data_dir=str(demo_model_warehouse.parent), debrief_dir=str(tmp_path))
    result = materialize(
        [race_debrief, debrief_covers_every_starter, debrief_has_strategy_when_tyre_data],
        resources={"data_lake": lake},
    )
    assert result.success
    checks = {e.check_name: e.passed for e in result.get_asset_check_evaluations()}
    assert checks["debrief_covers_every_starter"]
    assert len(list(tmp_path.glob("*.json"))) == 1
