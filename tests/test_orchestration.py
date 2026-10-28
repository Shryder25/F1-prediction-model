"""The Dagster code location loads, and the warehouse asset and its checks run."""

from dagster import Definitions, materialize

from f1pred import warehouse
from f1pred.config import Settings
from f1pred.ingest.fastf1_source import ingest_fastf1_season
from f1pred.orchestration.definitions import (
    QUALITY_CHECKS,
    DataLake,
    defs,
    warehouse_tables,
)
from tests.test_fastf1 import FakeLoader


def test_definitions_load() -> None:
    Definitions.validate_loadable(defs)
    assert len(QUALITY_CHECKS) == len(warehouse.CHECKS)


def test_warehouse_asset_and_checks(settings: Settings) -> None:
    ingest_fastf1_season(FakeLoader(), settings.raw_dir, 2024)
    result = materialize(
        [warehouse_tables, *QUALITY_CHECKS],
        resources={"data_lake": DataLake(data_dir=str(settings.data_dir))},
    )
    assert result.success
    evaluations = result.get_asset_check_evaluations()
    assert len(evaluations) == len(warehouse.CHECKS)
    assert all(e.passed for e in evaluations)
