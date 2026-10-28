"""Stage 4 tests: streaming features match dbt, predictions stay coherent, SSE works."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from f1pred import cli
from f1pred.live.events import CarLap, Entry, LapSnapshot, RaceInfo, RaceUpdate
from f1pred.live.features import LapFeatureBuilder
from f1pred.live.pipeline import Broadcaster, run_race
from f1pred.live.predictor import Models, RacePredictor, pit_windows
from f1pred.live.replay import WarehouseReplay, list_races, load_race_info, load_snapshots
from f1pred.live.server import _with_heartbeat, create_app
from f1pred.model.dataset import LAP_FEATURES, PRE_RACE_FEATURES, load_laps, load_pre_race
from f1pred.model.estimators import (
    LAP_GROUPS,
    PRE_RACE_GROUPS,
    BoostedModel,
    LinearModel,
    PositionBaseline,
)
from tests.conftest import MODEL_DEMO_SEASONS

FIRST_RACE = MODEL_DEMO_SEASONS[0] * 100 + 1
LAST_SEASON = MODEL_DEMO_SEASONS[-1]
# The demo puts a safety car in round 3 and a pit lane start in round 2.
SAFETY_CAR_RACE = LAST_SEASON * 100 + 3


@pytest.fixture(scope="module")
def models(demo_model_warehouse: Path) -> Models:
    """Small, fast models of the same kinds `f1pred evaluate` deploys."""
    with duckdb.connect(str(demo_model_warehouse), read_only=True) as con:
        pre = load_pre_race(con)
        laps = load_laps(con)
    train = MODEL_DEMO_SEASONS[:-1]
    pre_model = LinearModel(PRE_RACE_FEATURES, PRE_RACE_GROUPS, "grid_position")
    lap_model = BoostedModel(LAP_FEATURES, LAP_GROUPS, "position", n_estimators=40)
    return Models(
        pre_race=pre_model.fit(pre[pre["season"].isin(train)]),
        lap=lap_model.fit(laps[laps["season"].isin(train)]),
    )


@pytest.fixture(scope="module")
def model_dir(models: Models, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("models")
    joblib.dump(models.pre_race, path / "pre_race.joblib")
    joblib.dump(models.lap, path / "lap.joblib")
    return path


async def _collect(updates: AsyncIterator[RaceUpdate]) -> list[RaceUpdate]:
    return [u async for u in updates]


# --- Training / serving parity -------------------------------------------------


def test_streaming_features_match_the_dbt_table(demo_model_warehouse: Path) -> None:
    """Replaying every race of a season rebuilds ml_lap_features exactly."""
    with duckdb.connect(str(demo_model_warehouse), read_only=True) as con:
        expected = load_laps(con)
        expected = expected[expected["season"] == LAST_SEASON]
        built = []
        for race_id in sorted(expected["race_id"].unique()):
            info, _ = load_race_info(con, int(race_id))
            builder = LapFeatureBuilder(info)
            built += [builder.update(s) for s in load_snapshots(con, int(race_id))]
    got = pd.concat(built, ignore_index=True)
    keys = ["race_id", "lap", "driver_id"]
    columns = [*keys, *LAP_FEATURES]
    want = expected[columns].sort_values(keys).reset_index(drop=True)
    got = got[columns].sort_values(keys).reset_index(drop=True)
    assert len(got) == len(want)
    pd.testing.assert_frame_equal(got, want, check_dtype=False, atol=1e-6)
    # The safety car race exercises the flags; pit stops are counted.
    assert got.loc[got["race_id"] == SAFETY_CAR_RACE, "is_safety_car"].sum() > 0
    assert got["pit_stops_made"].max() >= 1


def _info(n: int = 3, total_laps: int = 10) -> RaceInfo:
    entries = tuple(Entry(driver_id=f"d{i}", code=f"D{i}", grid_position=i + 1) for i in range(n))
    return RaceInfo(
        race_id=202401, season=2024, round=1, race_name="Test GP",
        total_laps=total_laps, pit_loss_s=20.0, entries=entries,
    )  # fmt: skip


def _lap(lap: int, times: list[float], **flags: bool) -> LapSnapshot:
    cars = tuple(
        CarLap(driver_id=f"d{i}", position=i + 1, lap_time_s=t, compound="SOFT", **flags)
        for i, t in enumerate(times)
    )
    return LapSnapshot(lap=lap, cars=cars)


def test_feature_builder_gaps_and_order_checks() -> None:
    builder = LapFeatureBuilder(_info())
    first = builder.update(_lap(1, [90.0, 91.0, 93.5]))
    assert first["gap_to_leader_s"].tolist() == pytest.approx([0.0, 1.0, 3.5])
    assert first["gap_to_car_ahead_s"].tolist()[1:] == pytest.approx([1.0, 2.5])
    assert np.isnan(first["gap_to_car_ahead_s"].iloc[0])
    assert np.isnan(first["recent_pace_vs_field_s"]).all()  # lap 1 is never clean
    second = builder.update(_lap(2, [90.0, 89.0, 90.0]))
    assert second["recent_pace_vs_field_s"].tolist() == pytest.approx([0.0, -1.0, 0.0])
    assert second["compound_soft"].eq(1.0).all()
    with pytest.raises(ValueError, match="arrived after"):
        builder.update(_lap(2, [90.0, 90.0, 90.0]))


def test_neutralised_laps_are_not_clean() -> None:
    builder = LapFeatureBuilder(_info())
    builder.update(_lap(1, [90.0, 91.0, 92.0]))
    sc = builder.update(_lap(2, [120.0, 121.0, 119.0], is_safety_car=True))
    assert sc["recent_pace_vs_field_s"].isna().all()
    assert sc["is_safety_car"].eq(1.0).all()


# --- Predictions -----------------------------------------------------------------


async def test_replay_gives_coherent_predictions_every_lap(
    demo_model_warehouse: Path, models: Models
) -> None:
    source = WarehouseReplay(demo_model_warehouse, SAFETY_CAR_RACE, lap_interval_s=0)
    updates = await _collect(run_race(source, models))
    info = await source.race_info()
    assert [u.lap for u in updates] == list(range(info.total_laps + 1))
    assert updates[-1].is_final and not any(u.is_final for u in updates[:-1])
    assert "SC" in {u.track_status for u in updates}
    for update in updates:
        n = len(update.drivers)
        assert sum(d.p_win for d in update.drivers) == pytest.approx(1.0, abs=1e-3)
        assert sum(d.p_podium for d in update.drivers) == pytest.approx(min(3, n), abs=1e-3)
        assert sorted(d.predicted_position for d in update.drivers) == list(range(1, n + 1))
        assert sorted(d.position for d in update.drivers) == list(range(1, n + 1))
    leader = min(updates[-1].drivers, key=lambda d: d.position)
    assert leader.gap_to_leader_s == 0 and leader.pit is not None


async def test_reconnecting_replay_resumes_with_the_same_numbers(
    demo_model_warehouse: Path, models: Models
) -> None:
    full = await _collect(run_race(WarehouseReplay(demo_model_warehouse, FIRST_RACE, 0), models))
    resumed = await _collect(
        run_race(WarehouseReplay(demo_model_warehouse, FIRST_RACE, 0, 20), models, after_lap=20)
    )
    assert resumed[0].lap == 21
    assert resumed == full[21:]


def test_pre_race_view_without_features_falls_back_to_the_grid(models: Models) -> None:
    update = RacePredictor(models, _info(n=4)).before_start(None)
    assert [d.predicted_position for d in update.drivers] == [1, 2, 3, 4]
    assert sum(d.p_win for d in update.drivers) == pytest.approx(1.0)


def test_pit_windows() -> None:
    frame = pd.DataFrame(
        {
            "driver_id": ["a", "b", "c", "d"],
            "position": [1, 2, 3, 4],
            "gap_to_leader_s": [0.0, 5.0, 30.0, 31.0],
            "gap_to_car_behind_s": [5.0, 25.0, 1.0, None],
        }
    )
    windows = pit_windows(frame, pit_loss_s=20.0)
    assert windows["a"].places_lost == 1 and windows["a"].rejoin_position == 2
    assert windows["b"].places_lost == 0 and windows["b"].margin_s == pytest.approx(5.0)
    assert windows["c"].places_lost == 1 and windows["c"].margin_s == pytest.approx(-19.0)
    assert windows["d"].places_lost == 0 and windows["d"].margin_s is None


def test_models_load_from_the_evaluate_folder(model_dir: Path, tmp_path: Path) -> None:
    loaded = Models.load(model_dir)
    assert isinstance(loaded.lap, BoostedModel)
    with pytest.raises(FileNotFoundError, match="f1pred evaluate"):
        Models.load(tmp_path)


def test_list_races(demo_model_warehouse: Path) -> None:
    races = list_races(demo_model_warehouse)
    assert len(races) == 4 * len(MODEL_DEMO_SEASONS)
    assert races[0].season == LAST_SEASON  # newest first
    assert all(r.total_laps > 0 for r in races)


# --- Fan-out and SSE ---------------------------------------------------------------


def _update(lap: int) -> RaceUpdate:
    return RaceUpdate(
        race_id=1, race_name="x", season=2024, lap=lap, total_laps=3,
        track_status="GREEN", drivers=(),
    )  # fmt: skip


async def test_broadcaster_gives_late_joiners_the_history() -> None:
    gate = asyncio.Event()

    async def feed() -> AsyncIterator[RaceUpdate]:
        yield _update(0)
        yield _update(1)
        await gate.wait()
        yield _update(2)

    hub = Broadcaster(feed())
    hub.start()
    await asyncio.sleep(0.01)
    early = hub.subscribe()
    late = hub.subscribe(after_lap=0)
    first = await anext(early)
    assert first.lap == 0
    assert (await anext(late)).lap == 1
    gate.set()
    assert [u.lap async for u in early] == [1, 2]
    assert [u.lap async for u in late] == [2]
    assert hub.done and hub.error is None
    await hub.stop()


async def test_heartbeat_while_the_feed_is_quiet() -> None:
    async def slow() -> AsyncIterator[RaceUpdate]:
        await asyncio.sleep(0.05)
        yield _update(0)

    chunks = [c async for c in _with_heartbeat(slow(), every_s=0.01)]
    assert chunks[0] == ": keep-alive\n\n"
    assert any(c.startswith("event: update\nid: 0\n") for c in chunks)
    assert chunks[-1].startswith("event: end")


def _events(body: str) -> list[dict[str, str]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in block.splitlines() if not line.startswith(":")
        )
        if fields:
            events.append(fields)
    return events


def test_server_streams_a_replay(demo_model_warehouse: Path, model_dir: Path) -> None:
    app = create_app(demo_model_warehouse, model_dir)
    with TestClient(app) as client:
        health = client.get("/api/health").json()
        assert health["models_loaded"] is True and health["demo_data"] is True
        races = client.get("/api/races").json()
        assert {"race_id", "race_name", "total_laps"} <= set(races[0])
        assert "EventSource" in client.get("/").text

        response = client.get(f"/api/races/{FIRST_RACE}/stream", params={"interval": 0})
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _events(response.text)
        updates = [RaceUpdate.model_validate_json(e["data"]) for e in events[:-1]]
        assert events[-1]["event"] == "end"
        assert [int(e["id"]) for e in events[:-1]] == [u.lap for u in updates]
        assert updates[0].lap == 0 and updates[-1].is_final

        resumed = client.get(
            f"/api/races/{FIRST_RACE}/stream",
            params={"interval": 0},
            headers={"Last-Event-ID": "30"},
        )
        assert int(_events(resumed.text)[0]["id"]) == 31

        assert client.get("/api/races/190001/stream").status_code == 404
        assert client.get("/api/races/1/stream", params={"interval": -1}).status_code == 422
        assert client.get("/api/live/stream").status_code == 404


def test_server_without_models_reports_why(demo_model_warehouse: Path, tmp_path: Path) -> None:
    with TestClient(create_app(demo_model_warehouse, tmp_path)) as client:
        health = client.get("/api/health").json()
        assert health["status"] == "degraded"
        assert "f1pred evaluate" in health["model_error"]
        response = client.get(f"/api/races/{FIRST_RACE}/stream")
        assert response.status_code == 503


def test_server_live_stream_is_shared(demo_model_warehouse: Path, models: Models) -> None:
    def live() -> WarehouseReplay:
        return WarehouseReplay(demo_model_warehouse, FIRST_RACE, lap_interval_s=0)

    app = create_app(demo_model_warehouse, Path("unused"), live=live, models=models)
    with TestClient(app) as client:
        first = _events(client.get("/api/live/stream").text)
        second = _events(client.get("/api/live/stream").text)
        assert first == second
        assert json.loads(first[0]["data"])["lap"] == 0
        assert client.get("/api/health").json()["live"]["done"] is True


def test_replay_cli(
    demo_model_warehouse: Path,
    model_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("F1PRED_DATA_DIR", str(demo_model_warehouse.parent))
    assert cli.main(["replay", str(FIRST_RACE), "--models", str(model_dir)]) == 0
    out = capsys.readouterr().out
    assert "start" in out and "win:" in out
    assert cli.main(["replay", "190001", "--models", str(model_dir)]) == 2


def test_position_baseline_also_serves(demo_model_warehouse: Path, models: Models) -> None:
    """Any RaceModel works behind the predictor, including the baseline."""
    with duckdb.connect(str(demo_model_warehouse), read_only=True) as con:
        laps = load_laps(con)
    baseline = Models(pre_race=models.pre_race, lap=PositionBaseline().fit(laps))
    source = WarehouseReplay(demo_model_warehouse, FIRST_RACE, 0)
    updates = asyncio.run(_collect(run_race(source, baseline)))
    final = updates[-1]
    assert all(d.predicted_position == d.position for d in final.drivers)
