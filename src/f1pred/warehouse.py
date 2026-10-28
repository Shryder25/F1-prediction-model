"""Load raw Parquet into a DuckDB warehouse and run data quality checks.

Tables are rebuilt from the Parquet files on every load, so the warehouse always
matches the raw layer exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

PARTITIONED_TABLES = (
    "races",
    "results",
    "qualifying",
    "pit_stops",
    "laps",
    "driver_standings",
    "constructor_standings",
    "fastf1_laps",
    "fastf1_weather",
)


def load(raw_dir: Path, warehouse_path: Path) -> dict[str, int]:
    """(Re)build raw_* tables from Parquet. Returns row counts per table."""
    warehouse_path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with duckdb.connect(str(warehouse_path)) as con:
        for table in PARTITIONED_TABLES:
            pattern = raw_dir / table / "*" / "*" / "*.parquet"
            if not any(raw_dir.glob(f"{table}/*/*/*.parquet")):
                continue
            con.execute(
                f"CREATE OR REPLACE TABLE raw_{table} AS "
                "SELECT * FROM read_parquet(?, hive_partitioning = true, union_by_name = true, "
                "hive_types = {'season': INTEGER, 'round': INTEGER})",
                [str(pattern)],
            )
            counts[table] = _count(con, f"raw_{table}")
        circuits = raw_dir / "circuits" / "circuits.parquet"
        if circuits.exists():
            con.execute(
                "CREATE OR REPLACE TABLE raw_circuits AS SELECT * FROM read_parquet(?)",
                [str(circuits)],
            )
            counts["circuits"] = _count(con, "raw_circuits")
        if "results" in counts:
            _build_dim_driver(con)
    return counts


def _count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    row = con.execute(f"SELECT count(*) FROM {table}").fetchone()
    return int(row[0]) if row else 0


def _build_dim_driver(con: duckdb.DuckDBPyConnection) -> None:
    """One row per driver, mapping Ergast IDs to the codes and numbers live timing uses."""
    con.execute(
        """
        CREATE OR REPLACE TABLE dim_driver AS
        SELECT
            driver_id,
            arg_max(driver_code, season * 100 + round) AS driver_code,
            arg_max(driver_number, season * 100 + round) AS latest_car_number,
            any_value(permanent_number) AS permanent_number,
            any_value(given_name) || ' ' || any_value(family_name) AS full_name,
            any_value(date_of_birth) AS date_of_birth,
            min(season) AS first_season,
            max(season) AS last_season
        FROM raw_results
        GROUP BY driver_id
        """
    )


@dataclass(frozen=True)
class Check:
    name: str
    tables: tuple[str, ...]
    sql: str  # returns the offending rows; zero rows means the check passed


CHECKS = (
    Check(
        "results: one row per driver per race",
        ("raw_results",),
        "SELECT season, round, driver_id FROM raw_results GROUP BY ALL HAVING count(*) > 1",
    ),
    Check(
        "results: positions are 1..N within a race",
        ("raw_results",),
        "SELECT season, round FROM raw_results GROUP BY ALL "
        "HAVING max(position) != count(*) OR min(position) != 1",
    ),
    Check(
        "laps: one row per driver per lap",
        ("raw_laps",),
        "SELECT season, round, lap, driver_id FROM raw_laps GROUP BY ALL HAVING count(*) > 1",
    ),
    Check(
        "laps: lap times are positive",
        ("raw_laps",),
        "SELECT * FROM raw_laps WHERE lap_time_s IS NOT NULL AND lap_time_s <= 0",
    ),
    Check(
        "pit stops: every driver appears in that race's results",
        ("raw_pit_stops", "raw_results"),
        "SELECT p.* FROM raw_pit_stops p ANTI JOIN raw_results r USING (season, round, driver_id)",
    ),
    Check(
        "fastf1 laps: one row per driver per lap",
        ("raw_fastf1_laps",),
        "SELECT season, round, driver_code, lap FROM raw_fastf1_laps "
        "GROUP BY ALL HAVING count(*) > 1",
    ),
    Check(
        "fastf1 laps: compounds are known Pirelli compounds",
        ("raw_fastf1_laps",),
        "SELECT * FROM raw_fastf1_laps WHERE compound NOT IN ("
        "'HYPERSOFT', 'ULTRASOFT', 'SUPERSOFT', 'SOFT', 'MEDIUM', 'HARD', 'SUPERHARD', "
        "'INTERMEDIATE', 'WET', 'UNKNOWN', 'TEST_UNKNOWN')",
    ),
    Check(
        "fastf1 laps: tyre life is at least 1 lap",
        ("raw_fastf1_laps",),
        "SELECT * FROM raw_fastf1_laps WHERE tyre_life < 1",
    ),
    Check(
        "fastf1 laps: every driver code maps to dim_driver",
        ("raw_fastf1_laps", "dim_driver"),
        "SELECT DISTINCT driver_code FROM raw_fastf1_laps ANTI JOIN dim_driver USING (driver_code)",
    ),
    Check(
        "dim_driver: every driver has a code",
        ("dim_driver",),
        "SELECT * FROM dim_driver WHERE driver_code IS NULL AND last_season >= 2014",
    ),
)


def run_checks(warehouse_path: Path) -> list[tuple[str, int]]:
    """Run every check whose tables exist. Returns (check name, failing row count)."""
    results: list[tuple[str, int]] = []
    with duckdb.connect(str(warehouse_path), read_only=True) as con:
        existing = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        for check in CHECKS:
            if not set(check.tables) <= existing:
                continue
            row = con.execute(f"SELECT count(*) FROM ({check.sql})").fetchone()
            results.append((check.name, int(row[0]) if row else 0))
    return results
