"""OpenF1 bridge: rows become the same lap snapshots a replay produces."""

from __future__ import annotations

import datetime as dt
from typing import Any

import httpx
import pytest
import respx

from f1pred.jolpica.ratelimit import SlidingWindowLimiter
from f1pred.live.openf1 import (
    OpenF1Client,
    OpenF1Error,
    OpenF1Source,
    assemble_snapshots,
    lap_end_times,
    starting_grid,
    track_status_by_lap,
)

BASE = "https://openf1.test/v1"
START = dt.datetime(2024, 3, 2, 15, 3, tzinfo=dt.UTC)
CARS = {1: "max_verstappen", 4: "norris", 16: "leclerc"}


def _iso(seconds: float) -> str:
    return (START + dt.timedelta(seconds=seconds)).isoformat()


LAP_TIMES = {1: [95.0, 92.0, 92.5], 4: [96.0, 92.2, 92.4], 16: [97.0, 92.1, 115.0]}
STINTS = [
    {"driver_number": c, "stint_number": 1, "compound": "MEDIUM", "lap_start": 1,
     "lap_end": 2 if c == 16 else None, "tyre_age_at_start": 0}
    for c in CARS
] + [
    {"driver_number": 16, "stint_number": 2, "compound": "HARD", "lap_start": 3,
     "lap_end": None, "tyre_age_at_start": 0}
]  # fmt: skip
PITS = [{"driver_number": 16, "lap_number": 2, "pit_duration": 22.1}]
CONTROL = [
    {"date": _iso(150), "lap_number": 2, "category": "SafetyCar", "flag": None,
     "message": "VIRTUAL SAFETY CAR DEPLOYED"},
    {"date": _iso(200), "lap_number": 3, "category": "SafetyCar", "flag": None,
     "message": "VIRTUAL SAFETY CAR ENDING"},
]  # fmt: skip


def _complete_laps() -> list[dict[str, Any]]:
    """OpenF1 /laps rows. Lap 1 has no duration, as in the real feed."""
    rows = []
    for car, times in LAP_TIMES.items():
        for n, seconds in enumerate(times, start=1):
            rows.append(
                {
                    "driver_number": car,
                    "lap_number": n,
                    "date_start": _iso(sum(times[: n - 1])),
                    "lap_duration": None if n == 1 else seconds,
                    "is_pit_out_lap": car == 16 and n == 3,
                }
            )
    return rows


def test_lap_end_times_use_the_next_lap_when_duration_is_missing() -> None:
    ends = lap_end_times(_complete_laps())
    assert ends[(4, 1)] == START + dt.timedelta(seconds=96.0)
    assert ends[(1, 3)] == START + dt.timedelta(seconds=95.0 + 92.0 + 92.5)


def test_snapshots_from_a_finished_session() -> None:
    snaps = assemble_snapshots(
        _complete_laps(), STINTS, PITS, CONTROL, START, CARS, session_complete=True
    )
    assert [s.lap for s in snaps] == [1, 2, 3]
    lap1 = snaps[0]
    assert [c.driver_id for c in lap1.cars] == ["max_verstappen", "norris", "leclerc"]
    assert [c.lap_time_s for c in lap1.cars] == [95.0, 96.0, 97.0]
    lap3 = {c.driver_id: c for c in snaps[2].cars}
    assert snaps[2].cars[-1].driver_id == "leclerc"  # lost time in the pits
    assert lap3["leclerc"].compound == "HARD" and lap3["leclerc"].tyre_age_laps == 1
    assert lap3["leclerc"].is_pit_out_lap and lap3["norris"].tyre_age_laps == 3
    pit_lap = {c.driver_id: c for c in snaps[1].cars}["leclerc"]
    assert pit_lap.pit_stops == 1 and pit_lap.is_pit_in_lap
    assert [s.track_status for s in snaps] == ["GREEN", "VSC", "VSC"]


def test_live_waits_until_every_car_has_finished_the_lap() -> None:
    rows = [
        r for r in _complete_laps()
        if not (r["driver_number"] == 16 and r["lap_number"] == 3)
    ]  # fmt: skip
    # Leclerc has not finished lap 3, and the leader has not finished lap 5.
    snaps = assemble_snapshots(rows, STINTS, PITS, CONTROL, START, CARS)
    assert [s.lap for s in snaps] == [1, 2]
    assert assemble_snapshots(rows, STINTS, PITS, CONTROL, START, CARS, after_lap=2) == []


def test_live_drops_a_car_once_the_leader_is_two_laps_on() -> None:
    rows = [r for r in _complete_laps() if r["driver_number"] != 16 or r["lap_number"] < 3]
    leader_lap = {"driver_number": 1, "lap_number": 4, "date_start": _iso(279.5),
                  "lap_duration": 92.0, "is_pit_out_lap": False}  # fmt: skip
    leader_lap5 = {**leader_lap, "lap_number": 5, "date_start": _iso(371.5)}
    snaps = assemble_snapshots([*rows, leader_lap, leader_lap5], STINTS, PITS, [], START, CARS)
    lap3 = next(s for s in snaps if s.lap == 3)
    assert {c.driver_id for c in lap3.cars} == {"max_verstappen", "norris"}


def test_track_status_open_neutralisation_runs_to_the_last_lap() -> None:
    control = [
        {"date": _iso(10), "lap_number": 5, "message": "SAFETY CAR DEPLOYED"},
        {"date": _iso(20), "lap_number": 8, "flag": "RED", "message": "RED FLAG"},
        {"date": _iso(30), "lap_number": 9, "flag": "GREEN", "message": "TRACK CLEAR"},
    ]
    status = track_status_by_lap(control, last_lap=10)
    assert status.safety_car == frozenset(range(5, 11))
    assert status.red_flag == frozenset({8, 9})


def test_starting_grid_is_the_first_position_reported() -> None:
    rows = [
        {"driver_number": 1, "position": 2, "date": _iso(-600)},
        {"driver_number": 1, "position": 1, "date": _iso(30)},
        {"driver_number": 4, "position": 1, "date": _iso(-600)},
    ]
    assert starting_grid(rows) == {1: 2, 4: 1}


def _client(http: httpx.AsyncClient | None = None) -> OpenF1Client:
    return OpenF1Client(BASE, token="", http=http, limiter=SlidingWindowLimiter([(1000, 1.0)]))


@respx.mock
async def test_client_retries_and_explains_auth_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("f1pred.live.openf1.asyncio.sleep", no_sleep)
    route = respx.get(f"{BASE}/laps")
    route.side_effect = [httpx.Response(429), httpx.Response(200, json=[{"lap_number": 1}])]
    respx.get(f"{BASE}/pit").respond(404, json={"detail": "No results found."})
    respx.get(f"{BASE}/stints").respond(401)
    async with _client() as client:
        assert await client.get("laps", session_key=1) == [{"lap_number": 1}]
        assert await client.get("pit", session_key=1) == []
        with pytest.raises(OpenF1Error, match="F1PRED_OPENF1_TOKEN"):
            await client.get("stints", session_key=1)
    assert route.call_count == 2


@respx.mock
async def test_client_sends_the_token() -> None:
    route = respx.get(f"{BASE}/sessions").respond(json=[])
    client = OpenF1Client(BASE, token="secret", limiter=SlidingWindowLimiter([(1000, 1.0)]))
    async with client:
        await client.get("sessions")
    assert route.calls[0].request.headers["Authorization"] == "Bearer secret"


@respx.mock
async def test_source_streams_a_past_session() -> None:
    session = {"session_key": 9472, "session_type": "Race", "meeting_key": 1229, "year": 2024,
               "date_start": _iso(-120), "date_end": _iso(3600), "circuit_short_name": "Sakhir",
               "country_name": "Bahrain"}  # fmt: skip
    drivers = [
        {"driver_number": c, "name_acronym": code, "team_name": "Team", "team_colour": "3671C6"}
        for c, code in ((1, "VER"), (4, "NOR"), (16, "LEC"))
    ]
    positions = [{"driver_number": c, "position": i, "date": _iso(-60)} for i, c in
                 enumerate((1, 4, 16), start=1)]  # fmt: skip
    respx.get(f"{BASE}/sessions").respond(json=[session])
    respx.get(f"{BASE}/drivers").respond(json=drivers)
    respx.get(f"{BASE}/position").respond(json=positions)
    respx.get(f"{BASE}/meetings").respond(json=[{"meeting_name": "Bahrain Grand Prix"}])
    respx.get(f"{BASE}/laps").respond(json=_complete_laps())
    respx.get(f"{BASE}/stints").respond(json=STINTS)
    respx.get(f"{BASE}/pit").respond(json=PITS)
    respx.get(f"{BASE}/race_control").respond(json=CONTROL)

    async with _client() as client:
        source = OpenF1Source(client, session_key=9472, replay_interval_s=0)
        info = await source.race_info()
        snaps = [s async for s in source.laps()]
    assert info.race_name == "Bahrain Grand Prix" and info.total_laps == 3
    assert [e.code for e in info.entries] == ["VER", "NOR", "LEC"]
    assert info.entries[0].team_colour == "#3671C6"
    assert await source.pre_race_features() is None
    assert [s.lap for s in snaps] == [1, 2, 3]
    assert snaps[0].cars[0].driver_id == "ver"


@respx.mock
async def test_live_session_needs_the_race_distance() -> None:
    now = dt.datetime.now(dt.UTC)
    session = {"session_key": 1, "date_start": now.isoformat(),
               "date_end": (now + dt.timedelta(hours=2)).isoformat()}  # fmt: skip
    respx.get(f"{BASE}/sessions").respond(json=[session])
    for endpoint in ("drivers", "position", "meetings"):
        respx.get(f"{BASE}/{endpoint}").respond(json=[])
    async with _client() as client:
        with pytest.raises(OpenF1Error, match="--total-laps"):
            await OpenF1Source(client, session_key="latest").race_info()
