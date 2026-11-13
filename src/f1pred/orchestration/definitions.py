"""Dagster assets, data quality checks and schedule for the ingestion pipeline.

Lineage:

    jolpica_circuits ─┐
    jolpica_season ───┼──► warehouse ──► dbt_marts ──► race_debrief
    fastf1_season ────┘       │                            │
                         quality checks              debrief checks

`jolpica_season` and `fastf1_season` are partitioned by season, so one season can be
backfilled or refreshed on its own.

Monday after a race: `weekly_refresh` pulls the weekend's data at 06:00, and when that
run succeeds `debrief_after_refresh` rebuilds the warehouse and dbt marts and writes
the debrief for the latest race.
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
    DagsterRunStatus,
    Definitions,
    MaterializeResult,
    MetadataValue,
    RunRequest,
    RunStatusSensorContext,
    ScheduleEvaluationContext,
    StaticPartitionsDefinition,
    asset,
    asset_check,
    define_asset_job,
    run_status_sensor,
    schedule,
)

from f1pred import transform, warehouse
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
    # Where debriefs are written. Real ones are committed and published to GitHub Pages.
    debrief_dir: str = "docs/debriefs"

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


@asset(group_name="warehouse", deps=[warehouse_tables], compute_kind="dbt")
def dbt_marts(data_lake: DataLake) -> MaterializeResult[Any]:
    """Build and test every dbt model (`dbt build`): staging, marts and ML features."""
    if not transform.run_dbt(data_lake.settings.warehouse_path, ["build"]):
        raise RuntimeError("dbt build failed: see the run logs.")
    return MaterializeResult()


@asset(group_name="debrief", deps=[dbt_marts], compute_kind="python")
def race_debrief(data_lake: DataLake) -> MaterializeResult[Any]:
    """Post-race debrief for the latest race: predicted vs actual, strategy, tyres, undercuts."""
    from f1pred.debrief import build_debrief, latest_race_id
    from f1pred.debrief.site import write_debrief

    settings = data_lake.settings
    race_id = latest_race_id(settings.warehouse_path)
    debrief = build_debrief(settings.warehouse_path, race_id, settings.model_dir)
    path = write_debrief(debrief, Path(data_lake.debrief_dir))
    return MaterializeResult(
        metadata={
            "race_id": race_id,
            "path": MetadataValue.path(str(path)),
            "summary": MetadataValue.md("\n".join(f"- {h}" for h in debrief.headlines)),
            "model_source": debrief.score.source,
            "strategy_available": debrief.strategy.available,
        }
    )


@asset_check(asset=race_debrief, description="Every starter has a prediction and a result")
def debrief_covers_every_starter(data_lake: DataLake) -> AssetCheckResult:
    from f1pred.debrief.site import load_debriefs

    debriefs = load_debriefs(Path(data_lake.debrief_dir))
    if not debriefs:
        return AssetCheckResult(passed=False, metadata={"error": "no debrief written"})
    latest = debriefs[0]
    path = data_lake.settings.warehouse_path
    with duckdb.connect(str(path), read_only=True) as con:
        row = con.execute(
            "SELECT count(*) FROM marts.fct_race_results WHERE race_id = ?",
            [latest.race.race_id],
        ).fetchone()
    starters = int(row[0]) if row else 0
    ranks = sorted(r.predicted_rank for r in latest.results)
    return AssetCheckResult(
        passed=len(latest.results) == starters and ranks == list(range(1, starters + 1)),
        severity=AssetCheckSeverity.ERROR,
        metadata={"starters": starters, "in_debrief": len(latest.results)},
    )


@asset_check(
    asset=race_debrief, description="Strategy section is present when the race has tyre data"
)
def debrief_has_strategy_when_tyre_data(data_lake: DataLake) -> AssetCheckResult:
    from f1pred.debrief.site import load_debriefs

    debriefs = load_debriefs(Path(data_lake.debrief_dir))
    if not debriefs:
        return AssetCheckResult(passed=False, metadata={"error": "no debrief written"})
    latest = debriefs[0]
    with duckdb.connect(str(data_lake.settings.warehouse_path), read_only=True) as con:
        row = con.execute(
            "SELECT has_tyre_data FROM marts.dim_race WHERE race_id = ?", [latest.race.race_id]
        ).fetchone()
    has_tyres = bool(row and row[0])
    return AssetCheckResult(
        # A warning, not an error: FastF1 data can lag the Jolpica results by a day.
        passed=latest.strategy.available or not has_tyres,
        severity=AssetCheckSeverity.WARN,
        metadata={"has_tyre_data": has_tyres, "strategy_available": latest.strategy.available},
    )


refresh_current_season = define_asset_job(
    "refresh_current_season",
    selection=AssetSelection.assets(jolpica_season, fastf1_season),
)

rebuild_warehouse = define_asset_job(
    "rebuild_warehouse", selection=AssetSelection.assets(warehouse_tables)
)


post_race_debrief = define_asset_job(
    "post_race_debrief",
    selection=AssetSelection.assets(warehouse_tables, dbt_marts, race_debrief),
)


@schedule(job=refresh_current_season, cron_schedule="0 6 * * 1")
def weekly_refresh(context: ScheduleEvaluationContext) -> RunRequest:
    """Every Monday morning, pick up the weekend's race."""
    season = str(context.scheduled_execution_time.year)
    return RunRequest(
        run_key=f"{season}-{context.scheduled_execution_time:%Y%m%d}", partition_key=season
    )


@run_status_sensor(
    run_status=DagsterRunStatus.SUCCESS,
    monitored_jobs=[refresh_current_season],
    request_job=post_race_debrief,
)
def debrief_after_refresh(context: RunStatusSensorContext) -> RunRequest:
    """Once the weekend's data is in, rebuild the marts and write the debrief."""
    return RunRequest(run_key=context.dagster_run.run_id)


def _definitions(**resources: Any) -> Definitions:
    return Definitions(
        assets=[
            jolpica_circuits,
            jolpica_season,
            fastf1_season,
            warehouse_tables,
            dbt_marts,
            race_debrief,
        ],
        asset_checks=[
            *QUALITY_CHECKS,
            debrief_covers_every_starter,
            debrief_has_strategy_when_tyre_data,
        ],
        jobs=[refresh_current_season, rebuild_warehouse, post_race_debrief],
        schedules=[weekly_refresh],
        sensors=[debrief_after_refresh],
        resources={"data_lake": DataLake(), **resources},
    )


defs = _definitions()
