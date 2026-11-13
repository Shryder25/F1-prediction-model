"""Dagster assets, checks, jobs and sensors: one asset graph from the F1 APIs to the debrief.

Lineage (every dbt model is its own asset, abridged here):

    jolpica_circuits ─┐
    jolpica_season ───┼──► warehouse_tables ──► stg_* ──► int_* ──► fct_* / dim_* / ml_*
    fastf1_season ────┘          │                                      │
                           quality checks                ┌──────────────┴──────┐
                                                   strategy_tables        race_debrief
                                                                               │
                                                                        debrief checks

dbt tests show up as asset checks on the model they test (see `dbt_models.py`).

`jolpica_season` and `fastf1_season` are partitioned by season, so one season can be
backfilled or refreshed on its own.

After a race weekend: `race_weekend_sensor` watches the season calendar and, once a race
has had time to settle (Monday 06:00 UTC by default), re-ingests that season. When the
ingestion run succeeds, `debrief_after_refresh` runs `post_race_debrief`: warehouse, every
dbt model and test, the strategy tables and the debrief for the latest race.
"""

import asyncio
import datetime as dt
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
from dagster import (
    AssetCheckResult,
    AssetChecksDefinition,
    AssetCheckSeverity,
    AssetExecutionContext,
    AssetKey,
    AssetSelection,
    AssetSpec,
    ConfigurableResource,
    DagsterRunStatus,
    Definitions,
    MaterializeResult,
    MetadataValue,
    RunRequest,
    RunStatusSensorContext,
    SensorEvaluationContext,
    SkipReason,
    StaticPartitionsDefinition,
    asset,
    asset_check,
    define_asset_job,
    multi_asset,
    run_status_sensor,
    sensor,
)

from f1pred import strategy, warehouse
from f1pred.config import Settings
from f1pred.ingest.fastf1_source import FIRST_SEASON, FastF1Loader, ingest_fastf1_season
from f1pred.ingest.pipeline import ingest_circuits, ingest_season
from f1pred.jolpica.client import JolpicaClient
from f1pred.orchestration.dbt_models import dbt_resource, f1pred_dbt_models
from f1pred.orchestration.resources import DataLake

SEASONS = StaticPartitionsDefinition(
    [str(year) for year in range(FIRST_SEASON, dt.date.today().year + 1)]
)


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


def _raw_table_spec(table: str) -> AssetSpec:
    if table == "circuits":
        upstream, source = [jolpica_circuits], "Jolpica"
    elif table.startswith("fastf1_"):
        upstream, source = [fastf1_season], "FastF1"
    else:
        upstream, source = [jolpica_season], "Jolpica"
    return AssetSpec(
        f"raw_{table}",
        deps=upstream,
        description=f"main.raw_{table}: {source} Parquet loaded into DuckDB as is.",
        group_name="warehouse",
        kinds={"duckdb"},
        # FastF1 starts in 2018 and may be missing entirely, so these tables are optional.
        skippable=True,
    )


RAW_TABLES = (*warehouse.PARTITIONED_TABLES, "circuits")


@multi_asset(
    specs=[_raw_table_spec(table) for table in RAW_TABLES],
    name="warehouse_tables",
    can_subset=False,
)
def warehouse_tables(data_lake: DataLake) -> Iterator[MaterializeResult[Any]]:
    """Rebuild the DuckDB warehouse (raw_* tables and dim_driver) from the raw Parquet files.

    One asset per raw_* table, so each dbt source (and the staging model reading it)
    hangs off the table it actually reads in the asset graph.
    """
    settings = data_lake.settings
    counts = warehouse.load(settings.raw_dir, settings.warehouse_path)
    for table, count in counts.items():
        yield MaterializeResult(
            asset_key=f"raw_{table}",
            metadata={
                "rows": count,
                "path": MetadataValue.path(str(settings.warehouse_path)),
            },
        )


def _make_check(check: warehouse.Check) -> AssetChecksDefinition:
    name = "".join(c if c.isalnum() else "_" for c in check.name.lower()).strip("_")
    while "__" in name:
        name = name.replace("__", "_")

    # Each check sits on the first table it reads. main.dim_driver is built from raw_results
    # in the same step, so its checks sit on raw_results.
    table = "raw_results" if check.tables[0] == "dim_driver" else check.tables[0]

    @asset_check(asset=AssetKey(table), name=name, description=check.name)
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


def _mart(name: str) -> AssetKey:
    """Asset key of the dbt mart marts/<name>. Checked, so a renamed model fails at load."""
    key = AssetKey(["marts", name])
    if key not in f1pred_dbt_models.keys:
        raise KeyError(f"No dbt model named {name}")
    return key


@asset(
    group_name="strategy",
    deps=[_mart("fct_laps"), _mart("fct_pit_stops"), _mart("dim_race")],
    compute_kind="python",
)
def strategy_tables(data_lake: DataLake) -> MaterializeResult[Any]:
    """strategy.* tables: stint pace, undercuts and overcuts, safety car stops, pit windows."""
    with duckdb.connect(str(data_lake.settings.warehouse_path)) as con:
        counts = strategy.build_tables(con)
        findings = strategy.headline_findings(strategy.load_tables(con))
    return MaterializeResult(
        metadata={
            **{f"rows_{table}": count for table, count in counts.items()},
            "findings": MetadataValue.md("\n".join(f"- {f}" for f in findings)),
        }
    )


@asset(
    group_name="debrief",
    deps=[
        _mart(name)
        for name in (
            "dim_race",
            "dim_driver",
            "dim_constructor",
            "fct_race_results",
            "fct_laps",
            "fct_pit_stops",
            "ml_pre_race_features",
            "ml_lap_features",
        )
    ],
    compute_kind="python",
)
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
    selection=(
        AssetSelection.assets(warehouse_tables, strategy_tables, race_debrief)
        | AssetSelection.assets(f1pred_dbt_models)
    ),
)


class RaceCalendar(ConfigurableResource):  # type: ignore[type-arg]
    """The season's race dates, from Jolpica's schedule (one request per sensor tick)."""

    # Hours after midnight UTC on race day before the data is worth pulling. 30 hours is
    # Monday 06:00 UTC: Jolpica results and FastF1 timing are both in by then.
    settle_hours: float = 30.0

    def race_dates(self, season: int) -> dict[int, dt.date]:
        """race_id (season * 100 + round, as in dbt) -> race date."""
        settings = Settings()

        async def run() -> dict[int, dt.date]:
            async with JolpicaClient(settings) as client:
                schedule = await client.schedule(season)
            return {race.season * 100 + race.round: race.date for race in schedule}

        return asyncio.run(run())

    def ready_at(self, race_date: dt.date) -> dt.datetime:
        midnight = dt.datetime.combine(race_date, dt.time(), tzinfo=dt.UTC)
        return midnight + dt.timedelta(hours=self.settle_hours)


@sensor(job=refresh_current_season, minimum_interval_seconds=60 * 60)
def race_weekend_sensor(
    context: SensorEvaluationContext, race_calendar: RaceCalendar
) -> RunRequest | SkipReason:
    """After each race weekend, re-ingest the season once the results have settled.

    The cursor is the last race_id handled, so each race triggers exactly one run. On the
    first tick (no cursor) only the most recent race counts: switching the sensor on
    mid-season does not set off one run per past race.
    """
    now = dt.datetime.now(dt.UTC)
    season = now.year
    dates = race_calendar.race_dates(season)
    finished = sorted(rid for rid, day in dates.items() if race_calendar.ready_at(day) <= now)
    if not finished:
        return SkipReason(f"No {season} race has finished and settled yet.")
    latest = finished[-1]
    if context.cursor is not None and latest <= int(context.cursor):
        upcoming = [day for rid, day in sorted(dates.items()) if rid > latest]
        nxt = f"; next race {upcoming[0]:%Y-%m-%d}" if upcoming else ""
        return SkipReason(f"Race {latest} already handled{nxt}.")
    context.update_cursor(str(latest))
    return RunRequest(
        run_key=f"race-{latest}",
        partition_key=str(season),
        tags={"f1pred/race_id": str(latest)},
    )


@run_status_sensor(
    run_status=DagsterRunStatus.SUCCESS,
    monitored_jobs=[refresh_current_season],
    request_job=post_race_debrief,
)
def debrief_after_refresh(context: RunStatusSensorContext) -> RunRequest:
    """Once the weekend's data is in, rebuild the warehouse, dbt models, strategy and debrief."""
    return RunRequest(run_key=context.dagster_run.run_id)


def _definitions(**resources: Any) -> Definitions:
    return Definitions(
        assets=[
            jolpica_circuits,
            jolpica_season,
            fastf1_season,
            warehouse_tables,
            f1pred_dbt_models,
            strategy_tables,
            race_debrief,
        ],
        asset_checks=[
            *QUALITY_CHECKS,
            debrief_covers_every_starter,
            debrief_has_strategy_when_tyre_data,
        ],
        jobs=[refresh_current_season, rebuild_warehouse, post_race_debrief],
        sensors=[race_weekend_sensor, debrief_after_refresh],
        resources={
            "data_lake": DataLake(),
            "dbt": dbt_resource(),
            "race_calendar": RaceCalendar(),
            **resources,
        },
    )


defs = _definitions()
