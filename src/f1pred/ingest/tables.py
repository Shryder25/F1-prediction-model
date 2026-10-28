"""Flatten nested API models into one row per fact.

Every row carries `season` and `round` so tables join cleanly in the warehouse.
"""

from __future__ import annotations

from typing import Any

from f1pred.jolpica.models import (
    Circuit,
    ConstructorStanding,
    DriverStanding,
    Race,
)

Row = dict[str, Any]


def lap_time_to_seconds(value: str | None) -> float | None:
    """Convert '1:23.456' or '23.456' to seconds. Returns None for blanks."""
    if not value:
        return None
    minutes, _, seconds = value.rpartition(":")
    return (int(minutes) * 60 if minutes else 0) + float(seconds)


def races_rows(race: Race) -> list[Row]:
    return [
        {
            "season": race.season,
            "round": race.round,
            "race_name": race.name,
            "circuit_id": race.circuit.circuit_id,
            "race_date": race.date,
        }
    ]


def circuits_rows(circuits: list[Circuit]) -> list[Row]:
    return [
        {
            "circuit_id": c.circuit_id,
            "circuit_name": c.name,
            "locality": c.location.locality,
            "country": c.location.country,
            "lat": c.location.lat,
            "lon": c.location.long,
        }
        for c in circuits
    ]


def results_rows(race: Race) -> list[Row]:
    return [
        {
            "season": race.season,
            "round": race.round,
            "driver_id": r.driver.driver_id,
            "driver_code": r.driver.code,
            "driver_number": r.number,
            "permanent_number": r.driver.permanent_number,
            "given_name": r.driver.given_name,
            "family_name": r.driver.family_name,
            "date_of_birth": r.driver.date_of_birth,
            "constructor_id": r.constructor.constructor_id,
            "grid": r.grid,
            "position": r.position,
            "position_text": r.position_text,
            "points": r.points,
            "laps": r.laps,
            "status": r.status,
            "race_time_ms": r.time.millis if r.time else None,
        }
        for r in race.results
    ]


def qualifying_rows(race: Race) -> list[Row]:
    return [
        {
            "season": race.season,
            "round": race.round,
            "driver_id": q.driver.driver_id,
            "constructor_id": q.constructor.constructor_id,
            "position": q.position,
            "q1_s": lap_time_to_seconds(q.q1),
            "q2_s": lap_time_to_seconds(q.q2),
            "q3_s": lap_time_to_seconds(q.q3),
        }
        for q in race.qualifying
    ]


def pit_stops_rows(race: Race) -> list[Row]:
    return [
        {
            "season": race.season,
            "round": race.round,
            "driver_id": p.driver_id,
            "stop": p.stop,
            "lap": p.lap,
            # Long stops (red flags) come as '1:02.345', so parse like a lap time.
            "duration_s": lap_time_to_seconds(p.duration),
        }
        for p in race.pit_stops
    ]


def laps_rows(race: Race) -> list[Row]:
    return [
        {
            "season": race.season,
            "round": race.round,
            "lap": lap.number,
            "driver_id": t.driver_id,
            "position": t.position,
            "lap_time_s": lap_time_to_seconds(t.time),
        }
        for lap in race.laps
        for t in lap.timings
    ]


def driver_standings_rows(season: int, round_: int, rows: list[DriverStanding]) -> list[Row]:
    return [
        {
            "season": season,
            "round": round_,
            "driver_id": s.driver.driver_id,
            "position": s.position,
            "points": s.points,
            "wins": s.wins,
        }
        for s in rows
    ]


def constructor_standings_rows(
    season: int, round_: int, rows: list[ConstructorStanding]
) -> list[Row]:
    return [
        {
            "season": season,
            "round": round_,
            "constructor_id": s.constructor.constructor_id,
            "position": s.position,
            "points": s.points,
            "wins": s.wins,
        }
        for s in rows
    ]
