"""Builders for API payloads shaped like real Jolpica responses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from f1pred.config import Settings

# Enough seasons for a train / validation / test split, and a holdout.
MODEL_DEMO_SEASONS = range(2020, 2026)

DRIVERS = [
    ("max_verstappen", "VER", 1, "Max", "Verstappen", "red_bull"),
    ("norris", "NOR", 4, "Lando", "Norris", "mclaren"),
    ("leclerc", "LEC", 16, "Charles", "Leclerc", "ferrari"),
]

CIRCUIT = {
    "circuitId": "bahrain",
    "circuitName": "Bahrain International Circuit",
    "Location": {"lat": "26.0325", "long": "50.5106", "locality": "Sakhir", "country": "Bahrain"},
}


def driver(i: int) -> dict[str, Any]:
    driver_id, code, number, given, family, _ = DRIVERS[i]
    return {
        "driverId": driver_id,
        "permanentNumber": str(number),
        "code": code,
        "givenName": given,
        "familyName": family,
        "dateOfBirth": "1997-09-30",
        "nationality": "Dutch",
    }


def constructor(i: int) -> dict[str, Any]:
    return {"constructorId": DRIVERS[i][5], "name": DRIVERS[i][5].title(), "nationality": "x"}


def race(season: int = 2024, round_: int = 1, **inner: Any) -> dict[str, Any]:
    return {
        "season": str(season),
        "round": str(round_),
        "raceName": "Bahrain Grand Prix",
        "Circuit": CIRCUIT,
        "date": "2024-03-02",
        **inner,
    }


def mrdata(
    table: str, items_key: str, items: list[Any], total: int, offset: int = 0
) -> dict[str, Any]:
    return {
        "MRData": {
            "limit": "100",
            "offset": str(offset),
            "total": str(total),
            table: {items_key: items},
        }
    }


def results(season: int = 2024, round_: int = 1) -> list[dict[str, Any]]:
    return [
        {
            "number": str(DRIVERS[i][2]),
            "position": str(i + 1),
            "positionText": str(i + 1),
            "points": str([25, 18, 15][i]),
            "Driver": driver(i),
            "Constructor": constructor(i),
            "grid": str(i + 1),
            "laps": "57",
            "status": "Finished",
            "Time": {"millis": str(5_504_742 + i * 1000), "time": "1:31:44.742"},
        }
        for i in range(3)
    ]


def qualifying() -> list[dict[str, Any]]:
    return [
        {
            "number": str(DRIVERS[i][2]),
            "position": str(i + 1),
            "Driver": driver(i),
            "Constructor": constructor(i),
            "Q1": "1:30.031",
            "Q2": "1:29.374",
            "Q3": f"1:29.{179 + i * 100}",
        }
        for i in range(3)
    ]


def pit_stops() -> list[dict[str, Any]]:
    return [
        {
            "driverId": DRIVERS[i][0],
            "lap": "17",
            "stop": "1",
            "time": "18:43:12",
            "duration": "23.1",
        }
        for i in range(3)
    ]


def laps(n: int = 2) -> list[dict[str, Any]]:
    return [
        {
            "number": str(lap),
            "Timings": [
                {"driverId": DRIVERS[i][0], "position": str(i + 1), "time": f"1:3{i}.000"}
                for i in range(3)
            ],
        }
        for lap in range(1, n + 1)
    ]


def standings(kind: str) -> dict[str, Any]:
    if kind == "driver":
        rows = [
            {
                "position": str(i + 1),
                "points": "25",
                "wins": "1",
                "Driver": driver(i),
                "Constructors": [constructor(i)],
            }
            for i in range(3)
        ]
        key = "DriverStandings"
    else:
        rows = [
            {"position": str(i + 1), "points": "25", "wins": "1", "Constructor": constructor(i)}
            for i in range(3)
        ]
        key = "ConstructorStandings"
    return mrdata(
        "StandingsTable", "StandingsLists", [{"season": "2024", "round": "1", key: rows}], 3
    )


@pytest.fixture
def settings(tmp_path: Any) -> Settings:
    return Settings(jolpica_base_url="https://api.test/ergast/f1", data_dir=tmp_path)


@pytest.fixture(scope="session")
def demo_model_warehouse(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A demo warehouse with dbt built, shared by the model and live tests. Read only."""
    from f1pred import demo, transform

    data_dir = tmp_path_factory.mktemp("model_demo")
    demo.build(data_dir, seasons=MODEL_DEMO_SEASONS)
    path = data_dir / "warehouse.duckdb"
    assert transform.run_dbt(path, ["build"]), "dbt build failed: see the log above"
    return path
