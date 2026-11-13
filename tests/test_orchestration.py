"""The Dagster code location loads as one graph from the APIs to the debrief, the assets
run end to end on fixture data, and the race weekend sensor fires once per race."""

import datetime as dt
from pathlib import Path

import duckdb
import pytest
from dagster import (
    AssetCheckKey,
    AssetKey,
    DagsterInstance,
    Definitions,
    RunRequest,
    SkipReason,
    build_sensor_context,
    materialize,
)

from f1pred import demo, warehouse
from f1pred.config import Settings
from f1pred.ingest.fastf1_source import ingest_fastf1_season
from f1pred.orchestration.dbt_models import F1DbtTranslator, dbt_resource, f1pred_dbt_models
from f1pred.orchestration.definitions import (
    QUALITY_CHECKS,
    DataLake,
    RaceCalendar,
    defs,
    post_race_debrief,
    race_debrief,
    race_weekend_sensor,
    strategy_tables,
    warehouse_tables,
)
from tests.test_fastf1 import FakeLoader


def test_definitions_load() -> None:
    Definitions.validate_loadable(defs)
    assert len(QUALITY_CHECKS) == len(warehouse.CHECKS)
    # Every quality check sits on a real raw table asset.
    raw_keys = set(warehouse_tables.keys)
    assert all(k.asset_key in raw_keys for c in QUALITY_CHECKS for k in c.check_keys)


def test_one_graph_from_api_to_debrief() -> None:
    """Walk upstream from the debrief: it must reach the ingestion assets through dbt."""
    graph = defs.resolve_asset_graph()
    seen: set[AssetKey] = set()
    frontier = {AssetKey("race_debrief")}
    while frontier:
        key = frontier.pop()
        seen.add(key)
        frontier |= set(graph.get(key).parent_keys) - seen
    names = {key.to_user_string() for key in seen}
    assert {"jolpica_season", "fastf1_season", "jolpica_circuits"} <= names
    assert {"raw_laps", "staging/stg_jolpica__laps", "marts/fct_laps"} <= names
    assert {"marts/ml_pre_race_features", "reference/constructor_branding"} <= names
    # Strategy reads the same marts.
    strategy_parents = {
        k.to_user_string() for k in graph.get(AssetKey("strategy_tables")).parent_keys
    }
    assert {"marts/fct_laps", "marts/fct_pit_stops"} <= strategy_parents


def test_every_dbt_source_is_a_warehouse_table() -> None:
    raw_keys = set(warehouse_tables.keys)
    source_keys = {dep.asset_key for dep in f1pred_dbt_models.specs for dep in dep.deps}
    external = {k for k in source_keys if k not in f1pred_dbt_models.keys}
    assert external, "dbt models should depend on the raw tables"
    assert external <= raw_keys


def test_dbt_tests_are_asset_checks() -> None:
    checks = list(f1pred_dbt_models.check_keys)
    assert len(checks) > 20
    assert any(c.asset_key == AssetKey(["marts", "fct_race_results"]) for c in checks)


def test_translator_groups_by_folder() -> None:
    translator = F1DbtTranslator()
    model = {"resource_type": "model", "fqn": ["f1pred", "marts", "core", "fct_laps"]}
    assert translator.get_group_name(model) == "dbt_marts"
    assert translator.get_group_name({"resource_type": "seed", "fqn": []}) == "dbt_seeds"
    assert translator.get_asset_key({"resource_type": "source", "name": "raw_laps"}) == AssetKey(
        "raw_laps"
    )


def test_post_race_job_covers_the_whole_chain() -> None:
    graph = defs.resolve_asset_graph()
    selected = post_race_debrief.selection.resolve(graph)  # type: ignore[union-attr]
    assert set(warehouse_tables.keys) <= selected
    assert set(f1pred_dbt_models.keys) <= selected
    assert {AssetKey("strategy_tables"), AssetKey("race_debrief")} <= selected


def test_warehouse_asset_and_checks(settings: Settings) -> None:
    ingest_fastf1_season(FakeLoader(), settings.raw_dir, 2024)
    result = materialize(
        [warehouse_tables, *QUALITY_CHECKS],
        resources={"data_lake": DataLake(data_dir=str(settings.data_dir))},
    )
    assert result.success
    # Only FastF1 laps were ingested, so only those tables were loaded and checked.
    materialized = {e.asset_key for e in result.get_asset_materialization_events()}
    assert materialized == {AssetKey("raw_fastf1_laps"), AssetKey("raw_fastf1_weather")}
    evaluations = result.get_asset_check_evaluations()
    assert evaluations
    assert all(e.asset_key.path[0].startswith("raw_fastf1") for e in evaluations)
    assert all(e.passed for e in evaluations)


def test_end_to_end_on_demo_raw_layer(tmp_path: Path) -> None:
    """Raw Parquet -> warehouse -> every dbt model and test -> strategy -> debrief, in one run."""
    data_dir = tmp_path / "data"
    demo.generate(data_dir / "raw", seasons=(2024,))
    lake = DataLake(data_dir=str(data_dir), debrief_dir=str(tmp_path / "debriefs"))
    result = materialize(
        [warehouse_tables, *QUALITY_CHECKS, f1pred_dbt_models, strategy_tables, race_debrief],
        resources={"data_lake": lake, "dbt": dbt_resource()},
    )
    assert result.success
    checks = result.get_asset_check_evaluations()
    # Every warehouse check ran (all raw tables exist in the demo) and every dbt test passed.
    assert {e.check_name for e in checks} >= {c.name for c in _quality_check_keys()}
    assert all(e.passed for e in checks if e.severity.value == "ERROR")
    dbt_checks = [e for e in checks if e.asset_key in f1pred_dbt_models.keys]
    assert len(dbt_checks) == len(list(f1pred_dbt_models.check_keys))
    with duckdb.connect(str(data_dir / "warehouse.duckdb"), read_only=True) as con:
        row = con.execute("SELECT count(*) FROM strategy.stints").fetchone()
    assert row and row[0] > 0
    assert len(list((tmp_path / "debriefs").glob("*.json"))) == 1


def _quality_check_keys() -> list[AssetCheckKey]:
    return [key for check in QUALITY_CHECKS for key in check.check_keys]


class FakeCalendar(RaceCalendar):
    """Three races this season, dated relative to now, so the test never goes stale."""

    def race_dates(self, season: int) -> dict[int, dt.date]:
        today = dt.datetime.now(dt.UTC).date()
        return {
            season * 100 + 1: today - dt.timedelta(days=14),
            season * 100 + 2: today - dt.timedelta(days=7),
            season * 100 + 3: today + dt.timedelta(days=7),
        }


def _tick(cursor: str | None, calendar: RaceCalendar) -> tuple[RunRequest | SkipReason, str | None]:
    with DagsterInstance.ephemeral() as instance:
        context = build_sensor_context(
            instance=instance, cursor=cursor, resources={"race_calendar": calendar}
        )
        result = race_weekend_sensor(context)
        return result, context.cursor


def test_race_weekend_sensor_fires_once_per_race() -> None:
    season = dt.datetime.now(dt.UTC).year
    # First tick: only the latest finished race, not one run per past race.
    result, cursor = _tick(None, FakeCalendar())
    assert isinstance(result, RunRequest)
    assert result.partition_key == str(season)
    assert result.run_key == f"race-{season * 100 + 2}"
    assert cursor == str(season * 100 + 2)
    # Same race again: skip, and say when the next one is.
    result, _ = _tick(cursor, FakeCalendar())
    assert isinstance(result, SkipReason)
    assert "next race" in (result.skip_message or "")
    # A cursor from before round 2 fires for round 2.
    result, cursor = _tick(str(season * 100 + 1), FakeCalendar())
    assert isinstance(result, RunRequest)
    assert cursor == str(season * 100 + 2)


def test_race_weekend_sensor_waits_for_results_to_settle() -> None:
    class SundayCalendar(RaceCalendar):
        def race_dates(self, season: int) -> dict[int, dt.date]:
            return {season * 100 + 1: dt.datetime.now(dt.UTC).date()}

    result, cursor = _tick(None, SundayCalendar())
    assert isinstance(result, SkipReason)
    assert cursor is None
    # With no settle time, a race today counts as finished.
    result, _ = _tick(None, SundayCalendar(settle_hours=0))
    assert isinstance(result, RunRequest)


@pytest.mark.parametrize("hours", [30.0, 54.0])
def test_ready_at(hours: float) -> None:
    sunday = dt.date(2026, 11, 22)
    ready = RaceCalendar(settle_hours=hours).ready_at(sunday)
    assert ready == dt.datetime(2026, 11, 22, tzinfo=dt.UTC) + dt.timedelta(hours=hours)


def test_sensor_triggers_ingestion_then_debrief() -> None:
    # The race weekend sensor starts ingestion; the run-status sensor takes it from there.
    assert race_weekend_sensor.job_name == "refresh_current_season"
    sensors = {s.name for s in defs.sensors or []}
    assert sensors == {"race_weekend_sensor", "debrief_after_refresh"}
