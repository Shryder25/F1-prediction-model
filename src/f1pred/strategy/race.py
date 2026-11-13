"""Strategy metrics for one race, or many: the entry points the debrief and CLI use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import duckdb
import pandas as pd

from f1pred.strategy.battles import BATTLE_COLUMNS, pit_battles
from f1pred.strategy.config import DEFAULT_CONFIG, StrategyConfig
from f1pred.strategy.safety_car import SC_STOP_COLUMNS, pit_losses, safety_car_stops
from f1pred.strategy.stints import STINT_COLUMNS, stint_summary
from f1pred.strategy.windows import WINDOW_COLUMNS, pit_windows

LAP_COLUMNS = (
    "race_id",
    "driver_id",
    "lap",
    "total_laps",
    "position",
    "lap_time_s",
    "elapsed_s",
    "compound",
    "stint",
    "tyre_age_laps",
    "is_pit_in_lap",
    "is_pit_out_lap",
    "is_safety_car",
    "is_virtual_safety_car",
    "is_red_flag",
    "is_accurate",
    "has_tyre_data",
)


@dataclass(frozen=True)
class RaceStrategy:
    """Strategy tables for one or more races. Every frame is keyed by race_id.

    A race without FastF1 tyre data (before 2018) has empty stints, safety car stops
    and pit windows, with the same columns. Pit battles only need lap times and pit
    stops, so they are filled for every season.
    """

    stints: pd.DataFrame
    pit_battles: pd.DataFrame
    safety_car_stops: pd.DataFrame
    pit_windows: pd.DataFrame

    TABLES: ClassVar[tuple[str, ...]] = ("stints", "pit_battles", "safety_car_stops", "pit_windows")

    def frames(self) -> dict[str, pd.DataFrame]:
        return {name: getattr(self, name) for name in self.TABLES}


def analyse_race(
    laps: pd.DataFrame,
    pit_stops: pd.DataFrame | None = None,
    config: StrategyConfig = DEFAULT_CONFIG,
) -> RaceStrategy:
    """Compute every strategy table from lap and pit stop frames.

    `laps` has the columns of marts.fct_laps (see LAP_COLUMNS). `pit_stops` needs
    race_id, driver_id and lap (marts.fct_pit_stops); without it, FastF1 pit-in flags
    are used. Works on one race or many.
    """
    if laps.empty:
        return RaceStrategy(
            pd.DataFrame(columns=STINT_COLUMNS),
            pd.DataFrame(columns=BATTLE_COLUMNS),
            pd.DataFrame(columns=SC_STOP_COLUMNS),
            pd.DataFrame(columns=WINDOW_COLUMNS),
        )
    if "is_accurate" not in laps:
        laps = laps.assign(is_accurate=True)
    stints = stint_summary(laps, config)
    losses = pit_losses(laps, pit_stops)
    return RaceStrategy(
        stints=stints,
        pit_battles=pit_battles(laps, pit_stops, config),
        safety_car_stops=safety_car_stops(laps, pit_stops, losses),
        pit_windows=pit_windows(laps, stints, config),
    )


def load_laps(
    con: duckdb.DuckDBPyConnection, race_id: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(laps, pit_stops) from the dbt marts, for one race or every race."""
    available = {
        row[0]
        for row in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'marts' AND table_name = 'fct_laps'"
        ).fetchall()
    }
    columns = ", ".join(c for c in LAP_COLUMNS if c in available)
    where = "WHERE race_id = ?" if race_id is not None else ""
    params = [race_id] if race_id is not None else []
    laps = con.execute(
        f"SELECT {columns} FROM marts.fct_laps {where} ORDER BY race_id, lap, position", params
    ).df()
    stops = con.execute(
        f"SELECT race_id, driver_id, lap FROM marts.fct_pit_stops {where}", params
    ).df()
    return laps, stops


def race_strategy(
    con: duckdb.DuckDBPyConnection, race_id: int, config: StrategyConfig = DEFAULT_CONFIG
) -> RaceStrategy:
    """Every strategy table for one race (race_id like 202405), read from the marts."""
    laps, stops = load_laps(con, race_id)
    return analyse_race(laps, stops, config)
