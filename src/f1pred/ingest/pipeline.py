"""Ingest seasons from Jolpica into partitioned Parquet files.

Layout: <raw_dir>/<table>/season=YYYY/round=RR/data.parquet (Hive-style partitions)

Each (table, season, round) file is written whole and replaced on re-run, so the
pipeline is idempotent: running it twice never duplicates rows.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd

from f1pred.ingest import tables
from f1pred.jolpica.client import JolpicaClient

log = logging.getLogger(__name__)


def write_partition(
    raw_dir: Path, table: str, rows: list[tables.Row], season: int, round_: int
) -> Path | None:
    if not rows:
        return None
    path = raw_dir / table / f"season={season}" / f"round={round_:02d}" / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).drop(columns=["season", "round"], errors="ignore")
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)  # atomic swap: readers never see a half-written file
    return path


async def ingest_round(client: JolpicaClient, raw_dir: Path, season: int, round_: int) -> int:
    """Fetch every per-race table for one round. Returns the number of rows written."""
    written = 0
    fetchers = {
        "results": (client.results, tables.results_rows),
        "qualifying": (client.qualifying, tables.qualifying_rows),
        "pit_stops": (client.pit_stops, tables.pit_stops_rows),
        "laps": (client.laps, tables.laps_rows),
    }
    for table, (fetch, flatten) in fetchers.items():
        race = await fetch(season, round_)
        rows = flatten(race) if race else []
        if write_partition(raw_dir, table, rows, season, round_):
            written += len(rows)

    driver_rows = tables.driver_standings_rows(
        season, round_, await client.driver_standings(season, round_)
    )
    constructor_rows = tables.constructor_standings_rows(
        season, round_, await client.constructor_standings(season, round_)
    )
    for table, rows in (
        ("driver_standings", driver_rows),
        ("constructor_standings", constructor_rows),
    ):
        if write_partition(raw_dir, table, rows, season, round_):
            written += len(rows)
    return written


async def ingest_season(client: JolpicaClient, raw_dir: Path, season: int) -> int:
    """Ingest every round of a season that has already been raced."""
    schedule = await client.schedule(season)
    today = dt.date.today()
    written = 0
    for race in schedule:
        if race.date >= today:
            log.info("Skipping %s round %s: not raced yet", season, race.round)
            continue
        write_partition(raw_dir, "races", tables.races_rows(race), season, race.round)
        count = await ingest_round(client, raw_dir, season, race.round)
        log.info("%s round %02d (%s): %d rows", season, race.round, race.name, count)
        written += count
    return written


async def ingest_circuits(client: JolpicaClient, raw_dir: Path) -> int:
    rows = tables.circuits_rows(await client.circuits())
    path = raw_dir / "circuits" / "circuits.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return len(rows)
