"""Ingest lap-level tyre, stint, track status and weather data from FastF1.

Jolpica has no tyre or track status data, so this fills that gap from 2018 onward
(the first season FastF1 has full timing data for).

FastF1 is wrapped behind `SessionLoader` so tests can run without the network.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from f1pred.ingest.pipeline import write_partition

log = logging.getLogger(__name__)

FIRST_SEASON = 2018


@dataclass(frozen=True)
class RaceFrames:
    laps: pd.DataFrame
    weather: pd.DataFrame


class SessionLoader(Protocol):
    def raced_rounds(self, season: int) -> list[int]: ...

    def load_race(self, season: int, round_: int) -> RaceFrames: ...


class FastF1Loader:
    """The real loader. FastF1 keeps its own on-disk cache in `cache_dir`."""

    def __init__(self, cache_dir: Path) -> None:
        import fastf1  # type: ignore[import-untyped]

        cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
        self._fastf1 = fastf1

    def raced_rounds(self, season: int) -> list[int]:
        schedule = self._fastf1.get_event_schedule(season, include_testing=False)
        today = pd.Timestamp(dt.date.today())
        raced = schedule[schedule["EventDate"] < today]
        return [int(r) for r in raced["RoundNumber"]]

    def load_race(self, season: int, round_: int) -> RaceFrames:
        session = self._fastf1.get_session(season, round_, "R")
        session.load(laps=True, telemetry=False, weather=True, messages=False)
        return RaceFrames(laps=pd.DataFrame(session.laps), weather=session.weather_data)


def _seconds(series: pd.Series) -> pd.Series:
    return pd.to_timedelta(series).dt.total_seconds()


def normalize_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per driver per lap, with plain column names and seconds as floats."""
    out = pd.DataFrame(
        {
            "driver_code": laps["Driver"],
            "driver_number": pd.to_numeric(laps["DriverNumber"]).astype("Int64"),
            "team": laps["Team"],
            "lap": laps["LapNumber"].astype("Int64"),
            "position": laps["Position"].astype("Int64"),
            "lap_time_s": _seconds(laps["LapTime"]),
            "sector1_s": _seconds(laps["Sector1Time"]),
            "sector2_s": _seconds(laps["Sector2Time"]),
            "sector3_s": _seconds(laps["Sector3Time"]),
            "stint": laps["Stint"].astype("Int64"),
            "compound": laps["Compound"].str.upper(),
            "tyre_life": laps["TyreLife"].astype("Int64"),
            "fresh_tyre": laps["FreshTyre"].astype("boolean"),
            "pit_in": laps["PitInTime"].notna(),
            "pit_out": laps["PitOutTime"].notna(),
            # FastF1 track status is a string of codes seen during the lap, e.g. "14".
            # 1 green, 2 yellow, 4 safety car, 5 red flag, 6 VSC, 7 VSC ending.
            "track_status": laps["TrackStatus"].astype("string"),
            "is_accurate": laps["IsAccurate"].astype("boolean"),
        }
    )
    status = out["track_status"].fillna("")
    out["safety_car"] = status.str.contains("4")
    out["virtual_safety_car"] = status.str.contains("6|7", regex=True)
    out["red_flag"] = status.str.contains("5")
    return out.dropna(subset=["lap"])


def normalize_weather(weather: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_time_s": _seconds(weather["Time"]),
            "air_temp_c": weather["AirTemp"],
            "track_temp_c": weather["TrackTemp"],
            "humidity_pct": weather["Humidity"],
            "pressure_mbar": weather["Pressure"],
            "rainfall": weather["Rainfall"].astype(bool),
            "wind_speed_ms": weather["WindSpeed"],
            "wind_direction_deg": weather["WindDirection"],
        }
    )


def ingest_fastf1_season(loader: SessionLoader, raw_dir: Path, season: int) -> int:
    """Write fastf1_laps and fastf1_weather partitions for every raced round."""
    if season < FIRST_SEASON:
        raise ValueError(f"FastF1 lap data starts in {FIRST_SEASON}")
    written = 0
    for round_ in loader.raced_rounds(season):
        frames = loader.load_race(season, round_)
        laps = normalize_laps(frames.laps)
        weather = normalize_weather(frames.weather)
        for table, frame in (("fastf1_laps", laps), ("fastf1_weather", weather)):
            rows = [{str(k): v for k, v in row.items()} for row in frame.to_dict("records")]
            if write_partition(raw_dir, table, rows, season, round_):
                written += len(frame)
        log.info("FastF1 %s round %02d: %d laps", season, round_, len(laps))
    return written
