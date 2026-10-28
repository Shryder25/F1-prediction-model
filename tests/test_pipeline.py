"""End to end: mocked API -> Parquet -> DuckDB -> quality checks."""

from pathlib import Path

import duckdb
import respx

from f1pred import warehouse
from f1pred.config import Settings
from f1pred.ingest.pipeline import ingest_circuits, ingest_season
from f1pred.jolpica.cache import ResponseCache
from f1pred.jolpica.client import JolpicaClient
from f1pred.jolpica.ratelimit import SlidingWindowLimiter
from tests.conftest import (
    CIRCUIT,
    laps,
    mrdata,
    pit_stops,
    qualifying,
    race,
    results,
    standings,
)

BASE = "https://api.test/ergast/f1"


def mock_api() -> None:
    def races(**inner: object) -> dict[str, object]:
        return mrdata("RaceTable", "Races", [race(2023, **inner)], total=3)

    respx.get(f"{BASE}/circuits.json").respond(
        json=mrdata("CircuitTable", "Circuits", [CIRCUIT], 1)
    )
    respx.get(f"{BASE}/2023.json").respond(json=mrdata("RaceTable", "Races", [race(2023)], 1))
    respx.get(f"{BASE}/2023/1/results.json").respond(json=races(Results=results()))
    respx.get(f"{BASE}/2023/1/qualifying.json").respond(json=races(QualifyingResults=qualifying()))
    respx.get(f"{BASE}/2023/1/pitstops.json").respond(json=races(PitStops=pit_stops()))
    respx.get(f"{BASE}/2023/1/laps.json").respond(json=races(Laps=laps(2)))
    respx.get(f"{BASE}/2023/1/driverStandings.json").respond(json=standings("driver"))
    respx.get(f"{BASE}/2023/1/constructorStandings.json").respond(json=standings("constructor"))


async def run_ingest(settings: Settings) -> None:
    client = JolpicaClient(
        settings, cache=ResponseCache(":memory:"), limiter=SlidingWindowLimiter([(1000, 1.0)])
    )
    async with client:
        await ingest_circuits(client, settings.raw_dir)
        await ingest_season(client, settings.raw_dir, 2023)


@respx.mock
async def test_ingest_load_and_check(settings: Settings) -> None:
    mock_api()
    await run_ingest(settings)

    assert (settings.raw_dir / "laps" / "season=2023" / "round=01" / "data.parquet").exists()

    counts = warehouse.load(settings.raw_dir, settings.warehouse_path)
    assert counts["results"] == 3
    assert counts["laps"] == 6
    assert counts["circuits"] == 1

    checks = warehouse.run_checks(settings.warehouse_path)
    assert len(checks) == 6
    assert all(bad == 0 for _, bad in checks), checks

    with duckdb.connect(str(settings.warehouse_path)) as con:
        row = con.execute(
            "SELECT driver_code, latest_car_number FROM dim_driver WHERE driver_id = 'norris'"
        ).fetchone()
        season_type = con.execute("SELECT typeof(season) FROM raw_results LIMIT 1").fetchone()
    assert row == ("NOR", 4)
    assert season_type is not None and season_type[0] in {"BIGINT", "INTEGER"}


@respx.mock
async def test_rerun_is_idempotent(settings: Settings) -> None:
    mock_api()
    await run_ingest(settings)
    await run_ingest(settings)
    counts = warehouse.load(settings.raw_dir, settings.warehouse_path)
    assert counts["results"] == 3
    assert counts["laps"] == 6


def test_checks_catch_duplicates(tmp_path: Path) -> None:
    db = tmp_path / "w.duckdb"
    with duckdb.connect(str(db)) as con:
        con.execute(
            "CREATE TABLE raw_laps AS SELECT * FROM (VALUES "
            "(2023, 1, 1, 'norris', 1, 90.0), (2023, 1, 1, 'norris', 1, 90.0)) "
            "t(season, round, lap, driver_id, position, lap_time_s)"
        )
    results = dict(warehouse.run_checks(db))
    assert results["laps: one row per driver per lap"] == 1
