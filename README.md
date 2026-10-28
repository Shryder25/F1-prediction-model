# F1 Prediction Model

An F1 data platform and lap-by-lap race outcome predictor.

The project ingests historical race data, models it into a DuckDB warehouse, and (in later
stages) trains a model that updates win, podium and finishing-position probabilities every
lap, during a replay or a live race.

## Status

| Stage | What | Status |
| --- | --- | --- |
| 1 | Ingestion pipeline: Jolpica client, Parquet raw layer, DuckDB warehouse, quality checks | In progress |
| 2 | Warehouse modelling (dbt) and insight dashboards | Planned |
| 3 | Prediction model and evaluation | Planned |
| 4 | Streaming replay/live pipeline and HUD | Planned |

## Architecture (stage 1)

```
Jolpica API ──► async client ──► SQLite response cache
   (rate limited: 4 req/s, 500 req/h; retries with backoff)
                     │
                     ▼
        pydantic models (validates every response)
                     │
                     ▼
  data/raw/<table>/season=YYYY/round=RR/data.parquet   (idempotent, atomic writes)
                     │
                     ▼
       data/warehouse.duckdb: raw_* tables + dim_driver ──► data quality checks
```

Data source: [Jolpica F1](https://github.com/jolpica/jolpica-f1), the maintained
replacement for the Ergast API (which shut down in 2024). Tyre and live timing data will come
from [FastF1](https://docs.fastf1.dev/) and [OpenF1](https://openf1.org/) in later stages.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
uv run f1pred ingest --seasons 2024      # fetch one season (cached, safe to re-run)
uv run f1pred load                       # build data/warehouse.duckdb
uv run f1pred check                      # run data quality checks
```

Jolpica allows about 500 requests per hour, and one season takes roughly that many
(lap times are paginated). Every response is cached, so an interrupted run resumes where it
stopped.

## Development

```bash
uv run pytest            # tests (HTTP is mocked; no network needed)
uv run ruff check .      # lint
uv run mypy src          # strict type checking
uv run pre-commit install
```

CI runs lint, type checks and tests on every push and pull request.

## Tables

| Table | Grain |
| --- | --- |
| `raw_races` | one row per race |
| `raw_results` | one row per driver per race |
| `raw_qualifying` | one row per driver per qualifying session |
| `raw_laps` | one row per driver per lap |
| `raw_pit_stops` | one row per pit stop |
| `raw_driver_standings` | one row per driver after each round |
| `raw_constructor_standings` | one row per constructor after each round |
| `raw_circuits` | one row per circuit |
| `dim_driver` | one row per driver, mapping Ergast IDs to codes and car numbers |
