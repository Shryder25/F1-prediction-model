"""Command line entry point.

Stage 1: `ingest`, `ingest-fastf1`, `load`, `check`.
Stage 2: `transform` (dbt build), `docs`, `findings`, `dashboard`, and `demo`.
Stage 3: `evaluate` (train, score against baselines, write the report).
Stage 4: `replay` (a past race in the terminal) and `serve` (the SSE service and HUD).
`pipeline` runs every step from ingestion to the model report in order.
Add `--demo` before any command to use the synthetic demo warehouse in data/demo.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import logging
import subprocess
import sys
from pathlib import Path

from f1pred import demo, insights, transform, warehouse
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
    parser.add_argument(
        "--demo", action="store_true", help="Use the synthetic demo warehouse in <data>/demo"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Fetch seasons from Jolpica into Parquet")
    ingest.add_argument("--seasons", default="2018-2025", type=parse_seasons)
    fastf1 = sub.add_parser("ingest-fastf1", help="Fetch FastF1 tyre and weather data")
    fastf1.add_argument("--seasons", default="2018-2025", type=parse_seasons)
    sub.add_parser("load", help="Rebuild the DuckDB warehouse from Parquet")
    sub.add_parser("check", help="Run data quality checks on the warehouse")
    demo_cmd = sub.add_parser("demo", help="Build a synthetic demo warehouse (no network)")
    demo_cmd.add_argument("--seed", type=int, default=7)
    demo_cmd.add_argument(
        "--seasons",
        type=parse_seasons,
        default=list(demo.SEASONS),
        help="Seasons to simulate. The model needs 3+, e.g. 2019-2026",
    )
    dbt = sub.add_parser("transform", help="Build and test the dbt models (dbt build)")
    dbt.add_argument("--select", "-s", help="dbt node selection, e.g. +mart_grid_conversion")
    sub.add_parser("docs", help="Generate dbt docs (open with `dbt docs serve`)")
    sub.add_parser("findings", help="Print the headline finding of each analysis")
    sub.add_parser("dashboard", help="Open the Streamlit insights dashboard")
    evaluate = sub.add_parser(
        "evaluate", help="Train the models, score them on unseen seasons, write the report"
    )
    evaluate.add_argument("--train", type=parse_seasons, help="Training seasons, e.g. 2018-2023")
    evaluate.add_argument("--validation", type=int, help="Season used to choose settings")
    evaluate.add_argument("--test", type=int, help="Season scored once, for the report")
    evaluate.add_argument(
        "--holdout", type=parse_seasons, default=[], help="Seasons scored separately"
    )
    evaluate.add_argument("--out", type=Path, help="Report folder (default <data>/reports/model)")
    evaluate.add_argument("--quick", action="store_true", help="Smaller search, for smoke tests")
    replay = sub.add_parser("replay", help="Replay a past race lap by lap in the terminal")
    replay.add_argument("race_id", type=int, help="e.g. 202405 for 2024 round 5")
    replay.add_argument("--interval", type=float, default=0.0, help="Seconds between laps")
    replay.add_argument("--models", type=Path, help="Model folder (default from evaluate)")
    pipeline = sub.add_parser(
        "pipeline", help="Run every step, ingestion to model report, and write results.md"
    )
    pipeline.add_argument("--seasons", default="2018-2026", type=parse_seasons)
    pipeline.add_argument(
        "--from",
        dest="start",
        choices=PIPELINE_STEPS,
        default=PIPELINE_STEPS[0],
        help="Start at this step, e.g. --from transform once the data is loaded",
    )
    pipeline.add_argument("--quick", action="store_true", help="Smaller model search")
    serve = sub.add_parser("serve", help="Serve replays (and a live race) to the HUD")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--models", type=Path, help="Model folder (default from evaluate)")
    serve.add_argument(
        "--openf1-session",
        help="Also stream this OpenF1 race session: a session key, or 'latest'",
    )
    serve.add_argument("--total-laps", type=int, help="Scheduled laps of the OpenF1 race")
    serve.add_argument(
        "--race-id", type=int, help="Warehouse race id of the OpenF1 race, for pre-race features"
    )
    serve.add_argument(
        "--poll", type=float, default=15.0, help="Seconds between OpenF1 polls while live"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = Settings()
    if args.demo or args.command == "demo":
        settings = dataclasses.replace(settings, data_dir=settings.data_dir / "demo")

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
    elif args.command == "demo":
        counts = demo.build(settings.data_dir, seed=args.seed, seasons=args.seasons)
        for table, count in counts.items():
            print(f"{table:24} {count:>10,} rows")
        print(f"Demo warehouse: {settings.warehouse_path}. Next: f1pred --demo transform")
    elif args.command == "transform":
        # "cautious" skips tests that need models outside the selection.
        select = (
            ["--select", args.select, "--indirect-selection", "cautious"] if args.select else []
        )
        return 0 if transform.run_dbt(settings.warehouse_path, ["build", *select]) else 1
    elif args.command == "docs":
        return 0 if transform.run_dbt(settings.warehouse_path, ["docs", "generate"]) else 1
    elif args.command == "findings":
        with insights.connect(settings.warehouse_path) as con:
            if insights.is_demo(con):
                print("(demo data: synthetic, not real races)")
            for finding in insights.headline_findings(con, insights.seasons(con)):
                print(f"- {finding}")
    elif args.command == "dashboard":
        app = Path(__file__).parent / "dashboard" / "app.py"
        command = [sys.executable, "-m", "streamlit", "run", str(app)]
        return subprocess.call([*command, "--", "--warehouse", str(settings.warehouse_path)])
    elif args.command == "evaluate":
        return _evaluate(settings, args)
    elif args.command == "replay":
        return _replay(settings, args)
    elif args.command == "serve":
        return _serve(settings, args)
    elif args.command == "pipeline":
        return _pipeline(settings, args)
    return 0


PIPELINE_STEPS = ("ingest", "ingest-fastf1", "load", "check", "transform", "evaluate")


def _pipeline(settings: Settings, args: argparse.Namespace) -> int:
    """Run the steps in order, stop at the first failure, then write results.md."""
    seasons = ",".join(str(s) for s in args.seasons)
    commands = {
        "ingest": ["ingest", "--seasons", seasons],
        "ingest-fastf1": ["ingest-fastf1", "--seasons", seasons],
        "load": ["load"],
        "check": ["check"],
        "transform": ["transform"],
        "evaluate": ["evaluate", *(["--quick"] if args.quick else [])],
    }
    prefix = ["--demo"] if args.demo else []
    for step in PIPELINE_STEPS[PIPELINE_STEPS.index(args.start) :]:
        print(f"\n=== f1pred {' '.join(commands[step])} ===", flush=True)
        code = main([*prefix, *commands[step]])
        if code:
            print(f"Step '{step}' failed. Fix it, then resume with: f1pred pipeline --from {step}")
            return code
    path = write_results(settings)
    print(f"\nDone. Summary of findings and model scores: {path}")
    return 0


def write_results(settings: Settings) -> Path:
    """Put the insight findings and the model report in one file to share."""
    with insights.connect(settings.warehouse_path) as con:
        demo_note = "Synthetic demo data, not real races.\n\n" if insights.is_demo(con) else ""
        loaded = insights.seasons(con)
        findings = insights.headline_findings(con, loaded)
    report = settings.data_dir / "reports" / "model" / "report.md"
    lines = [
        "# f1pred results",
        "",
        f"{demo_note}Seasons loaded: {loaded[0]}-{loaded[-1]}." if loaded else demo_note,
        "",
        "## Insight findings",
        "",
        *(f"- {finding}" for finding in findings),
        "",
        "## Model report",
        "",
        report.read_text() if report.exists() else "_No model report yet: run f1pred evaluate._",
    ]
    path = settings.data_dir / "reports" / "results.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def _evaluate(settings: Settings, args: argparse.Namespace) -> int:
    # Imported here: the model libraries are an optional dependency group.
    from f1pred.model.dataset import SeasonSplit
    from f1pred.model.report import build_report

    split = None
    explicit = (args.train, args.validation, args.test)
    if any(explicit):
        if not all(explicit):
            print("Give all of --train, --validation and --test, or none of them.")
            return 2
        split = SeasonSplit(tuple(args.train), args.validation, args.test, tuple(args.holdout))
    out = args.out or settings.data_dir / "reports" / "model"
    try:
        metrics = build_report(settings.warehouse_path, out, split=split, quick=args.quick)
    except ValueError as error:  # e.g. too few seasons loaded for a split
        print(f"Cannot evaluate: {error}")
        return 2
    print(f"Report: {out / 'report.md'}")
    print(f"Saved models: {metrics['deployed_models']}")
    return 0


def _replay(settings: Settings, args: argparse.Namespace) -> int:
    from f1pred.live.pipeline import run_race
    from f1pred.live.predictor import Models
    from f1pred.live.replay import WarehouseReplay

    try:
        models = Models.load(args.models or settings.model_dir)
    except FileNotFoundError as error:
        print(error)
        return 2

    async def run() -> None:
        source = WarehouseReplay(settings.warehouse_path, args.race_id, args.interval)
        async for update in run_race(source, models):
            top = sorted(update.drivers, key=lambda d: -d.p_win)[:3]
            leader = min(update.drivers, key=lambda d: d.position, default=None)
            favourites = "  ".join(f"{d.code} {d.p_win:5.1%}" for d in top)
            lap = "start" if update.lap == 0 else f"lap {update.lap:>2}/{update.total_laps}"
            led = f"leader {leader.code}" if leader and update.lap else "grid order"
            print(f"{lap:>9}  {update.track_status:5}  {led:11}  win: {favourites}")

    try:
        asyncio.run(run())
    except KeyError as error:
        print(error.args[0])
        return 2
    return 0


def _serve(settings: Settings, args: argparse.Namespace) -> int:
    import uvicorn

    from f1pred.live.openf1 import OpenF1Client, OpenF1Source
    from f1pred.live.pipeline import RaceSource
    from f1pred.live.server import create_app

    live = None
    if args.openf1_session:

        def live() -> RaceSource:
            return OpenF1Source(
                OpenF1Client(),
                session_key=args.openf1_session,
                total_laps=args.total_laps,
                warehouse_path=settings.warehouse_path if args.race_id else None,
                race_id=args.race_id,
                poll_interval_s=args.poll,
            )

    app = create_app(settings.warehouse_path, args.models or settings.model_dir, live=live)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
