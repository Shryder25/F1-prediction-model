"""Replay a past race from the warehouse, lap by lap, as if it were live.

The replay reads only what a timing screen would have shown at each lap
(positions, lap times, tyres, flags, pit stops) from `marts.fct_laps` and
`marts.fct_pit_stops`. It does not read the lap features table: those are
rebuilt on the fly by `LapFeatureBuilder`, exactly as in a live race.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from f1pred.live.events import CarLap, Entry, LapSnapshot, RaceInfo
from f1pred.model.dataset import load_pre_race

# Used when a circuit has no pit stop history: a typical pit lane time.
DEFAULT_PIT_LOSS_S = 22.0


@dataclass(frozen=True)
class RaceSummary:
    race_id: int
    season: int
    round: int
    race_name: str
    total_laps: int


def list_races(warehouse_path: Path) -> list[RaceSummary]:
    """Races that can be replayed: they have laps and pre-race features."""
    with duckdb.connect(str(warehouse_path), read_only=True) as con:
        rows = con.execute(
            """
            SELECT r.race_id, r.season, r.round, r.race_name, r.total_laps
            FROM marts.dim_race AS r
            WHERE r.total_laps > 0
              AND r.race_id IN (SELECT race_id FROM marts.ml_pre_race_features)
            ORDER BY r.season DESC, r.round DESC
            """
        ).fetchall()
    return [RaceSummary(*row) for row in rows]


class WarehouseReplay:
    """A `RaceSource` that replays one race from the warehouse.

    `lap_interval_s` is the wall-clock pause between laps (0 for tests). Laps up
    to `fast_forward_to` come without a pause, so a reconnecting client catches
    up at once. They are still sent: the lap features need every lap.
    """

    def __init__(
        self,
        warehouse_path: Path,
        race_id: int,
        lap_interval_s: float = 1.0,
        fast_forward_to: int = 0,
    ) -> None:
        self.warehouse_path = warehouse_path
        self.race_id = race_id
        self.lap_interval_s = lap_interval_s
        self.fast_forward_to = fast_forward_to
        self._info: RaceInfo | None = None
        self._pre_race: pd.DataFrame | None = None
        self._laps: list[LapSnapshot] = []

    async def race_info(self) -> RaceInfo:
        if self._info is None:
            # DuckDB calls block; keep them off the event loop.
            await asyncio.to_thread(self._load)
        assert self._info is not None
        return self._info

    async def pre_race_features(self) -> pd.DataFrame | None:
        await self.race_info()
        return self._pre_race

    async def laps(self) -> AsyncIterator[LapSnapshot]:
        await self.race_info()
        for snapshot in self._laps:
            if snapshot.lap > self.fast_forward_to and self.lap_interval_s > 0:
                await asyncio.sleep(self.lap_interval_s)
            yield snapshot

    def _load(self) -> None:
        with duckdb.connect(str(self.warehouse_path), read_only=True) as con:
            self._info, self._pre_race = load_race_info(con, self.race_id)
            self._laps = load_snapshots(con, self.race_id)


def load_race_info(con: duckdb.DuckDBPyConnection, race_id: int) -> tuple[RaceInfo, pd.DataFrame]:
    race = con.execute(
        "SELECT season, round, race_name, circuit_id, total_laps FROM marts.dim_race "
        "WHERE race_id = ?",
        [race_id],
    ).fetchone()
    if race is None:
        raise KeyError(f"Race {race_id} is not in the warehouse.")
    season, round_, race_name, circuit_id, total_laps = race

    pre_race = load_pre_race(con)
    pre_race = pre_race[pre_race["race_id"] == race_id].reset_index(drop=True)
    if pre_race.empty:
        raise KeyError(f"Race {race_id} has no pre-race features. Run `f1pred transform`.")

    teams = con.execute(
        """
        SELECT e.driver_id, d.driver_code, e.car_number, e.constructor_id,
               c.constructor_name, c.team_colour
        FROM marts.fct_race_results AS e
        LEFT JOIN marts.dim_driver AS d USING (driver_id)
        LEFT JOIN marts.dim_constructor AS c USING (constructor_id)
        WHERE e.race_id = ?
        """,
        [race_id],
    ).df()
    team_of: dict[Any, dict[Any, Any]] = teams.set_index("driver_id").to_dict(orient="index")

    entries = []
    for row in pre_race.sort_values("grid_position").to_dict(orient="records"):
        team: dict[str, Any] = team_of.get(row["driver_id"], {})
        entries.append(
            Entry(
                driver_id=row["driver_id"],
                code=_text(team.get("driver_code")) or row["driver_id"][:3].upper(),
                car_number=_int(team.get("car_number")),
                constructor_id=_text(team.get("constructor_id")),
                team_name=_text(team.get("constructor_name")),
                team_colour=_text(team.get("team_colour")) or "#888888",
                grid_position=int(row["grid_position"]),
                gap_to_pole_pct=_float(row["gap_to_pole_pct"]),
                driver_finish_last5=_float(row["driver_finish_last5"]),
                constructor_points_last5=_float(row["constructor_points_last5"]),
            )
        )
    info = RaceInfo(
        race_id=race_id,
        season=int(season),
        round=int(round_),
        race_name=str(race_name),
        circuit_id=circuit_id,
        total_laps=int(total_laps),
        pit_loss_s=pit_loss(con, race_id, circuit_id),
        entries=tuple(entries),
    )
    return info, pre_race


def pit_loss(con: duckdb.DuckDBPyConnection, race_id: int, circuit_id: str | None) -> float:
    """Median pit lane time at this circuit in earlier races (no look-ahead).

    Falls back to all earlier races, then to a typical value.
    """
    for where, params in (
        ("p.race_id < ? AND p.circuit_id = ?", [race_id, circuit_id]),
        ("p.race_id < ?", [race_id]),
    ):
        row = con.execute(
            f"SELECT median(p.pit_lane_time_s) FROM marts.fct_pit_stops AS p "
            f"WHERE NOT p.is_outlier AND {where}",
            params,
        ).fetchone()
        if row is not None and row[0] is not None:
            return round(float(row[0]), 3)
    return DEFAULT_PIT_LOSS_S


def load_snapshots(con: duckdb.DuckDBPyConnection, race_id: int) -> list[LapSnapshot]:
    laps = con.execute(
        """
        SELECT l.lap, l.driver_id, l.position, l.lap_time_s, l.compound, l.stint,
               l.tyre_age_laps, l.is_pit_in_lap, l.is_pit_out_lap, l.is_safety_car,
               l.is_virtual_safety_car, l.is_red_flag,
               coalesce(s.stops, 0) AS pit_stops
        FROM marts.fct_laps AS l
        LEFT JOIN (
            SELECT driver_id, lap, count(*) AS stops
            FROM marts.fct_pit_stops WHERE race_id = ? GROUP BY ALL
        ) AS s USING (driver_id, lap)
        WHERE l.race_id = ?
        ORDER BY l.lap, l.position
        """,
        [race_id, race_id],
    ).df()
    snapshots = []
    for lap_number, group in laps.groupby("lap", sort=True):
        cars = tuple(
            CarLap(
                driver_id=r["driver_id"],
                position=int(r["position"]),
                lap_time_s=_float(r["lap_time_s"]),
                compound=_text(r["compound"]),
                stint=_int(r["stint"]),
                tyre_age_laps=_int(r["tyre_age_laps"]),
                is_pit_in_lap=_bool(r["is_pit_in_lap"]),
                is_pit_out_lap=_bool(r["is_pit_out_lap"]),
                is_safety_car=_bool(r["is_safety_car"]),
                is_virtual_safety_car=_bool(r["is_virtual_safety_car"]),
                is_red_flag=_bool(r["is_red_flag"]),
                pit_stops=int(r["pit_stops"]),
            )
            for r in group.to_dict(orient="records")
        )
        snapshots.append(LapSnapshot(lap=_int(lap_number) or 0, cars=cars))
    return snapshots


def _missing(value: Any) -> bool:
    return value is None or (not isinstance(value, str) and bool(pd.isna(value)))


def _float(value: Any) -> float | None:
    return None if _missing(value) else float(value)


def _int(value: Any) -> int | None:
    return None if _missing(value) else int(value)


def _bool(value: Any) -> bool | None:
    return None if _missing(value) else bool(value)


def _text(value: Any) -> str | None:
    return None if _missing(value) else str(value)
