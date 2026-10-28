"""Run the dbt project in `transform/` against a DuckDB warehouse."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

DBT_PROJECT_DIR = Path(__file__).resolve().parents[2] / "transform"


def _dbt_executable() -> str:
    """The dbt installed next to this Python, so `uv run` and venvs pick the right one."""
    local = Path(sys.executable).parent / "dbt"
    if local.exists():
        return str(local)
    found = shutil.which("dbt")
    if found is None:
        raise FileNotFoundError("dbt is not installed. Run `uv sync --all-groups`.")
    return found


def run_dbt(warehouse_path: Path, args: Sequence[str]) -> bool:
    """Run a dbt command, e.g. ["build"]. Returns True when it succeeded.

    dbt runs in its own process: dbt-duckdb holds its connection open for the life
    of the process, which would block anything else reading the warehouse.
    The warehouse path reaches dbt through F1PRED_WAREHOUSE (see transform/profiles.yml).
    """
    if not warehouse_path.exists():
        raise FileNotFoundError(f"No warehouse at {warehouse_path}. Run `f1pred load` first.")
    project = str(DBT_PROJECT_DIR)
    env = {**os.environ, "F1PRED_WAREHOUSE": str(warehouse_path.resolve())}
    command = [_dbt_executable(), *args, "--project-dir", project, "--profiles-dir", project]
    return subprocess.run(command, env=env, check=False).returncode == 0
