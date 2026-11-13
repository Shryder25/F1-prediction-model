"""The dbt project in `transform/` as Dagster assets, one asset per dbt model and seed.

dagster-dbt reads dbt's manifest, so every model, seed and test shows up in the same
asset graph as ingestion: `jolpica_season` -> `warehouse_tables` -> `stg_jolpica__laps`
-> `fct_laps` -> `race_debrief`. dbt tests become Dagster asset checks on the model
they test, so a failing `not_null` shows on that model in the UI.

Each dbt source maps to the asset of the same name (`raw_laps`, `raw_fastf1_laps`, ...),
which `warehouse_tables` loads. That is what joins ingestion and dbt into one graph.
"""

import os
import subprocess
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from dagster import AssetExecutionContext, AssetKey
from dagster_dbt import DagsterDbtTranslator, DbtCliResource, DbtProject, dbt_assets

from f1pred.orchestration.resources import DataLake
from f1pred.transform import DBT_PROJECT_DIR, _dbt_executable

MANIFEST_PATH = DBT_PROJECT_DIR / "target" / "manifest.json"

# Files whose change means the manifest has to be parsed again.
_DBT_SOURCES = ("models", "macros", "seeds", "tests", "dbt_project.yml")


def _newest_source_mtime(project_dir: Path) -> float:
    newest = 0.0
    for name in _DBT_SOURCES:
        path = project_dir / name
        files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
        newest = max([newest, *(f.stat().st_mtime for f in files)])
    return newest


def ensure_manifest(project_dir: Path = DBT_PROJECT_DIR) -> Path:
    """Run `dbt parse` when the manifest is missing or older than the dbt project.

    Parsing needs neither the warehouse nor the network, so the code location loads
    on a fresh checkout. `f1pred transform` writes the same manifest as a side effect.
    """
    manifest = project_dir / "target" / "manifest.json"
    if manifest.exists() and manifest.stat().st_mtime >= _newest_source_mtime(project_dir):
        return manifest
    project = str(project_dir)
    subprocess.run(
        [_dbt_executable(), "parse", "--project-dir", project, "--profiles-dir", project],
        env={**os.environ, "DO_NOT_TRACK": "1"},
        check=True,
        capture_output=True,
    )
    return manifest


class F1DbtTranslator(DagsterDbtTranslator):
    """Asset keys and groups for the dbt nodes."""

    def get_asset_key(self, dbt_resource_props: Mapping[str, Any]) -> AssetKey:
        # A source is a raw_* table that `warehouse_tables` loads: key it by table name.
        if dbt_resource_props["resource_type"] == "source":
            return AssetKey(dbt_resource_props["name"])
        return super().get_asset_key(dbt_resource_props)

    def get_group_name(self, dbt_resource_props: Mapping[str, Any]) -> str | None:
        # The folder under models/: dbt_staging, dbt_intermediate, dbt_marts; seeds -> dbt_seeds.
        if dbt_resource_props["resource_type"] == "seed":
            return "dbt_seeds"
        fqn = dbt_resource_props.get("fqn", [])
        return f"dbt_{fqn[1]}" if len(fqn) > 2 else "dbt"


DBT_PROJECT = DbtProject(project_dir=DBT_PROJECT_DIR)
TRANSLATOR = F1DbtTranslator()


def dbt_resource() -> DbtCliResource:
    return DbtCliResource(
        project_dir=DBT_PROJECT,
        profiles_dir=str(DBT_PROJECT_DIR),
        dbt_executable=_dbt_executable(),
    )


@dbt_assets(
    manifest=ensure_manifest(),
    project=DBT_PROJECT,
    dagster_dbt_translator=TRANSLATOR,
)
def f1pred_dbt_models(
    context: AssetExecutionContext, dbt: DbtCliResource, data_lake: DataLake
) -> Iterator[Any]:
    """`dbt build` for the selected models: run each model, then its tests."""
    warehouse_path = data_lake.settings.warehouse_path
    if not warehouse_path.exists():
        raise FileNotFoundError(f"No warehouse at {warehouse_path}: materialize warehouse_tables.")
    # profiles.yml reads the warehouse path from this variable; dbt runs as a subprocess
    # of this run's process, so the setting stays inside this run.
    os.environ["F1PRED_WAREHOUSE"] = str(warehouse_path.resolve())
    os.environ.setdefault("DO_NOT_TRACK", "1")
    yield from dbt.cli(["build"], context=context).stream()
