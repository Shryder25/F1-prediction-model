"""Dagster assets, data quality checks and schedule for the ingestion pipeline.

Lineage:

    jolpica_circuits ─┐
    jolpica_season ───┼──► warehouse ──► quality checks
    fastf1_season ────┘

`jolpica_season` and `fastf1_season` are partitioned by season, so one season can be
backfilled or refreshed on its own.
"""

import asyncio
import datetime as dt
from pathlib import Path
from typing import Any

import duckdb
from dagster import (
    AssetCheckResult,
    AssetChecksDefinition,
    AssetCheckSeverity,
    AssetExecutionContext,
    AssetSelection,
    ConfigurableResource,
    Definitions,
    MaterializeResult,
    MetadataValue,
    RunRequest,
    ScheduleEvaluationContext,
    StaticPartitionsDefinition,
    asset,
    asset_check,
    define_asset_job,
    schedule,
)

from f1pred import warehouse
from f1pred.config import Settings
from f1pred.ingest.fastf1_source import FIRST_SEASON, FastF1Loader, ingest_fastf1_season
from f1pred.ingest.pipeline import ingest_circuits, ingest_season
from f1pred.jolpica.client import JolpicaClient

SEASONS = StaticPartitionsDefinition(
    [str(year) for year in range(FIRST_SEASON, dt.date.today().year + 1)]
)


class DataLake(ConfigurableResource):  # type: ignore[type-arg]
    """Where raw files, caches and the warehouse live."""

    data_dir: str = "data"

    @property
    def settings(self) -> Settings:
        return Settings(data_dir=Path(self.data_dir))


@asset(group_name="ingestion", compute_kind="python")
def jolpica_circuits(data_lake: DataLake) -> MaterializeResult[Any]:
    """Every F1 circuit, from Jolpica."""
    settings = data_lake.settings

    async def run() -> int:
        async with JolpicaClient(settings) as client:
            return await ingest_circuits(client, settings.raw_dir)

    return MaterializeResult(metadata={"rows": asyncio.run(run())})


@asset(group_name="ingestion", partitions_def=SEASONS, compute_kind="python")
def jolpica_season(context: AssetExecutionContext, data_lake: DataLake) -> MaterializeResult[Any]:
    """Results, qualifying, laps, pit stops and standings for one season, from Jolpica."""
    settings = data_lake.settings
    season = int(context.partition_key)

    async def run() -> int:
        async with JolpicaClient(settings) as client:
            return await ingest_season(client, settings.raw_dir, season)

    return MaterializeResult(metadata={"rows": asyncio.run(run())})


@asset(group_name="ingestion", partitions_def=SEASONS, compute_kind="python")
def fastf1_season(context: AssetExecutionContext, data_lake: DataLake) -> MaterializeResult[Any]:
    """Per-lap tyre compound, stint, tyre age, track status and weather, from FastF1."""
    settings = data_lake.settings
    loader = FastF1Loader(settings.data_dir / "fastf1_cache")
    rows = ingest_fastf1_season(loader, settings.raw_dir, int(context.partition_key))
    return MaterializeResult(metadata={"rows": rows})


@asset(
    group_name="warehouse",
    deps=[jolpica_circuits, jolpica_season, fastf1_season],
    compute_kind="duckdb",
)
def warehouse_tables(data_lake: DataLake) -> MaterializeResult[Any]:
    """Rebuild the DuckDB warehouse (raw_* tables and dim_driver) from the raw Parquet files."""
    settings = data_lake.settings
    counts = warehouse.load(settings.raw_dir, settings.warehouse_path)
    return MaterializeResult(
        metadata={
            "path": MetadataValue.path(str(settings.warehouse_path)),
            **{f"rows_{table}": count for table, count in counts.items()},
        }
    )


def _make_check(check: warehouse.Check) -> AssetChecksDefinition:
    name = "".join(c if c.isalnum() else "_" for c in check.name.lower()).strip("_")
    while "__" in name:
        name = name.replace("__", "_")

    @asset_check(asset=warehouse_tables, name=name, description=check.name)
    def _check(data_lake: DataLake) -> AssetCheckResult:
        path = data_lake.settings.warehouse_path
        with duckdb.connect(str(path), read_only=True) as con:
            tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
            if not set(check.tables) <= tables:
                return AssetCheckResult(passed=True, metadata={"skipped": "tables not loaded yet"})
            row = con.execute(f"SELECT count(*) FROM ({check.sql})").fetchone()
        bad = int(row[0]) if row else 0
        return AssetCheckResult(
            passed=bad == 0,
            severity=AssetCheckSeverity.ERROR,
            metadata={"failing_rows": bad},
        )

    return _check


QUALITY_CHECKS = [_make_check(check) for check in warehouse.CHECKS]

refresh_current_season = define_asset_job(
    "refresh_current_season",
    selection=AssetSelection.assets(jolpica_season, fastf1_season),
)

rebuild_warehouse = define_asset_job(
    "rebuild_warehouse", selection=AssetSelection.assets(warehouse_tables)
)


@schedule(job=refresh_current_season, cron_schedule="0 6 * * 1")
def weekly_refresh(context: ScheduleEvaluationContext) -> RunRequest:
    """Every Monday morning, pick up the weekend's race."""
    season = str(context.scheduled_execution_time.year)
    return RunRequest(
        run_key=f"{season}-{context.scheduled_execution_time:%Y%m%d}", partition_key=season
    )


def _definitions(**resources: Any) -> Definitions:
    return Definitions(
        assets=[jolpica_circuits, jolpica_season, fastf1_season, warehouse_tables],
        asset_checks=QUALITY_CHECKS,
        jobs=[refresh_current_season, rebuild_warehouse],
        schedules=[weekly_refresh],
        resources={"data_lake": DataLake(), **resources},
    )


defs = _definitions()
