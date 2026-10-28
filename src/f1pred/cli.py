"""Command line entry point: `f1pred ingest`, `ingest-fastf1`, `load`, `check`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from f1pred import warehouse
from f1pred.config import Settings
from f1pred.ingest.fastf1_source import FastF1Loader, ingest_fastf1_season
from f1pred.ingest.pipeline import ingest_circuits, ingest_season
from f1pred.jolpica.client import JolpicaClient


def parse_seasons(text: str) -> list[int]:
    """'2018-2020' -> [2018, 2019, 2020]; '2021,2023' -> [2021, 2023]."""
    seasons: list[int] = []
    for part in text.split(","):
        start, _, end = part.partition("-")
        seasons.extend(range(int(start), int(end or start) + 1))
    return seasons


async def _ingest(settings: Settings, seasons: list[int]) -> None:
    async with JolpicaClient(settings) as client:
        await ingest_circuits(client, settings.raw_dir)
        for season in seasons:
            rows = await ingest_season(client, settings.raw_dir, season)
            logging.info("Season %s done: %d rows", season, rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="f1pred")
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Fetch seasons from Jolpica into Parquet")
    ingest.add_argument("--seasons", default="2018-2025", type=parse_seasons)
    fastf1 = sub.add_parser("ingest-fastf1", help="Fetch FastF1 tyre and weather data")
    fastf1.add_argument("--seasons", default="2018-2025", type=parse_seasons)
    sub.add_parser("load", help="Rebuild the DuckDB warehouse from Parquet")
    sub.add_parser("check", help="Run data quality checks on the warehouse")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings()

    if args.command == "ingest":
        asyncio.run(_ingest(settings, args.seasons))
    elif args.command == "ingest-fastf1":
        loader = FastF1Loader(settings.data_dir / "fastf1_cache")
        for season in args.seasons:
            rows = ingest_fastf1_season(loader, settings.raw_dir, season)
            logging.info("FastF1 season %s done: %d rows", season, rows)
    elif args.command == "load":
        for table, count in warehouse.load(settings.raw_dir, settings.warehouse_path).items():
            print(f"{table:24} {count:>10,} rows")
    elif args.command == "check":
        failed = 0
        for name, bad_rows in warehouse.run_checks(settings.warehouse_path):
            status = "PASS" if bad_rows == 0 else f"FAIL ({bad_rows} rows)"
            print(f"{status:16} {name}")
            failed += bad_rows > 0
        return 1 if failed else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
