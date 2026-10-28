"""FastF1 normalisation and ingestion, using a fake loader shaped like FastF1's output."""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from f1pred import warehouse
from f1pred.config import Settings
from f1pred.ingest.fastf1_source import (
    RaceFrames,
    ingest_fastf1_season,
    normalize_laps,
)
from f1pred.ingest.pipeline import write_partition


def fastf1_laps() -> pd.DataFrame:
    """Two drivers, three laps; NOR pits at the end of lap 2 under a safety car."""

    def td(seconds: float) -> pd.Timedelta:
        return pd.Timedelta(seconds=seconds)

    rows = []
    for code, number, team in (("VER", "1", "Red Bull Racing"), ("NOR", "4", "McLaren")):
        for lap in (1, 2, 3):
            pitted = code == "NOR" and lap == 2
            rows.append(
                {
                    "Driver": code,
                    "DriverNumber": number,
                    "Team": team,
                    "LapNumber": float(lap),
                    "Position": 1.0 if code == "VER" else 2.0,
                    "LapTime": td(90 + lap + 0.5),
                    "Sector1Time": td(30.1),
                    "Sector2Time": td(31.2),
                    "Sector3Time": pd.NaT if lap == 3 else td(29.9),
                    "Stint": 2.0 if code == "NOR" and lap == 3 else 1.0,
                    "Compound": "hard" if code == "NOR" and lap == 3 else "MEDIUM",
                    "TyreLife": 1.0 if code == "NOR" and lap == 3 else float(lap + 2),
                    "FreshTyre": code == "NOR" and lap == 3,
                    "PitInTime": td(3900) if pitted else pd.NaT,
                    "PitOutTime": td(3925) if code == "NOR" and lap == 3 else pd.NaT,
                    "TrackStatus": "14" if lap == 2 else "1",
                    "IsAccurate": True,
                }
            )
    return pd.DataFrame(rows)


def fastf1_weather() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Time": [pd.Timedelta(seconds=30), pd.Timedelta(seconds=90)],
            "AirTemp": [24.1, 24.3],
            "TrackTemp": [38.0, 38.4],
            "Humidity": [45.0, 46.0],
            "Pressure": [1012.1, 1012.0],
            "Rainfall": [False, False],
            "WindSpeed": [1.2, 1.5],
            "WindDirection": [180, 190],
        }
    )


class FakeLoader:
    def raced_rounds(self, season: int) -> list[int]:
        return [1]

    def load_race(self, season: int, round_: int) -> RaceFrames:
        return RaceFrames(laps=fastf1_laps(), weather=fastf1_weather())


def test_normalize_laps() -> None:
    laps = normalize_laps(fastf1_laps())
    nor = laps[laps["driver_code"] == "NOR"].set_index("lap")

    assert len(laps) == 6
    assert nor.loc[1, "lap_time_s"] == pytest.approx(91.5)
    assert nor.loc[2, "pit_in"]
    assert nor.loc[3, "pit_out"]
    assert nor.loc[3, "compound"] == "HARD"
    assert nor.loc[3, "stint"] == 2
    assert nor.loc[2, "safety_car"]
    assert not nor.loc[1, "safety_car"]
    assert np.isnan(nor.loc[3, "sector3_s"])
    assert laps["driver_number"].tolist()[:1] == [1]


def test_ingest_fastf1_to_warehouse(settings: Settings) -> None:
    written = ingest_fastf1_season(FakeLoader(), settings.raw_dir, 2024)
    assert written == 6 + 2

    # dim_driver comes from Jolpica results, so add a minimal results partition.
    results = [
        {
            "season": 2024,
            "round": 1,
            "driver_id": d,
            "driver_code": c,
            "driver_number": n,
            "permanent_number": n,
            "given_name": "x",
            "family_name": "y",
            "date_of_birth": None,
            "position": p,
        }
        for d, c, n, p in (("max_verstappen", "VER", 1, 1), ("norris", "NOR", 4, 2))
    ]
    write_partition(settings.raw_dir, "results", results, 2024, 1)

    counts = warehouse.load(settings.raw_dir, settings.warehouse_path)
    assert counts["fastf1_laps"] == 6
    assert counts["fastf1_weather"] == 2

    checks = dict(warehouse.run_checks(settings.warehouse_path))
    fastf1_checks = {name: bad for name, bad in checks.items() if name.startswith("fastf1")}
    assert len(fastf1_checks) == 4
    assert all(bad == 0 for bad in fastf1_checks.values()), fastf1_checks

    with duckdb.connect(str(settings.warehouse_path)) as con:
        types = dict(
            con.execute(
                "SELECT column_name, column_type FROM (DESCRIBE raw_fastf1_laps)"
            ).fetchall()
        )
    assert types["season"] == "INTEGER"
    assert types["lap_time_s"] == "DOUBLE"
    assert types["compound"] == "VARCHAR"


def test_rejects_seasons_before_fastf1_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ingest_fastf1_season(FakeLoader(), tmp_path, 2017)
