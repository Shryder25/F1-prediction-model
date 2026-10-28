# F1 Prediction Model

An F1 data platform and lap-by-lap race outcome predictor.

The project ingests historical race data, models it into a DuckDB warehouse, trains a model
that updates win, podium and finishing-position probabilities every lap, and streams those
predictions to a browser HUD during a replay or a live race.

## Status

| Stage | What | Status |
| --- | --- | --- |
| 1 | Ingestion pipeline: Jolpica + FastF1 ingestion, Parquet raw layer, DuckDB warehouse, Dagster orchestration, quality checks | In progress |
| 2 | Warehouse modelling (dbt), three insight analyses and a Streamlit dashboard | In progress |
| 3 | Pre-race and lap-by-lap prediction models, evaluated against baselines on unseen seasons | In progress |
| 4 | Streaming replay/live pipeline, FastAPI SSE service, HUD and Docker | In progress |

## Architecture (stage 1)

```
Jolpica API ──► async client ──► SQLite response cache
   (rate limited: 4 req/s, 500 req/h; retries with backoff; pydantic validation)
                                                         │
FastF1 ──► per-lap tyres, stints, track status, weather  │
                                                         ▼
          data/raw/<table>/season=YYYY/round=RR/data.parquet   (idempotent, atomic writes)
                                                         │
                                                         ▼
          data/warehouse.duckdb: raw_* tables + dim_driver ──► 10 data quality checks

Orchestrated by Dagster: season-partitioned assets, asset checks, weekly refresh schedule.
```

## Architecture (stage 2)

```
raw_* tables ──► dbt staging (views: typed, renamed, one model per source table)
                     │
                     ▼
             dbt intermediate (race entries, laps + tyres, clean stint laps, stints, pit stops)
                     │
                     ▼
   marts.dim_*  / marts.fct_*   ──►  marts.mart_* insight tables  ──►  Streamlit dashboard
   (race, circuit, driver,           (grid conversion, overtaking,     `f1pred findings`
    constructor; results, quali,      race craft, pit stops, tyre
    laps, pit stops)                  degradation)
```

56 dbt data tests (keys, relationships, ranges, accepted values) and 3 dbt unit tests run
on every build. FastF1 data is optional: without it the tyre models build empty instead
of failing.

## Architecture (stage 3)

```
marts.fct_* ──► marts.ml_pre_race_features   (one row per driver per race; only pre-race info)
            └─► marts.ml_lap_features        (one row per driver per lap; only laps so far)
                     │
                     ▼
   season split: train 2018-2023 · choose settings on 2024 · score once on 2025 · 2026 apart
                     │
                     ▼
   baselines (grid order, current order) vs logistic regression vs XGBoost
                     │
                     ▼
   report.md: Brier score, log loss, position MAE, calibration, 95% race-bootstrap intervals
   models/pre_race.joblib, models/lap.joblib  (for the stage 4 live predictor)
```

A dbt test recomputes driver form independently and fails if any feature sees the race
it predicts. Method and limitations: [`docs/model`](docs/model).

## Architecture (stage 4)

```
WarehouseReplay (fct_laps)  ─┐
OpenF1Source (live / past)  ─┴─► LapSnapshot per lap ──► LapFeatureBuilder ──► saved models
                                                                                    │
            HUD in the browser ◄── Server-Sent Events (FastAPI) ◄── RaceUpdate ◄────┘
            win probability chart · running order vs predicted finish · pit window
```

Replays and live races share one code path: sources only report what a timing screen
shows, and the features are rebuilt lap by lap. A test replays a whole season and checks
every streamed feature against the dbt table the model was trained on. Design and limits:
[`docs/live`](docs/live).

Sources: [Jolpica F1](https://github.com/jolpica/jolpica-f1), the maintained replacement for
the Ergast API (which shut down in 2024), and [FastF1](https://docs.fastf1.dev/) for tyre,
stint, track status and weather data from 2018 onward.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
uv run f1pred ingest --seasons 2024      # fetch one season (cached, safe to re-run)
uv run f1pred load                       # build data/warehouse.duckdb
uv run f1pred check                      # run data quality checks
uv run f1pred ingest-fastf1 --seasons 2024   # add tyre, stint and weather data
```

### Modelling and insights (stage 2)

```bash
uv run f1pred transform      # dbt build: seeds, models and tests
uv run f1pred findings       # one headline finding per analysis
uv run f1pred dashboard      # Streamlit dashboard at http://localhost:8501
uv run f1pred docs           # dbt docs; view with `cd transform && uv run dbt docs serve`
```

No API access? Build a synthetic demo warehouse and run everything against it. The
dashboard shows a banner when it is reading demo data.

```bash
uv run f1pred demo
uv run f1pred --demo transform
uv run f1pred --demo dashboard
```

The write-ups for each analysis are in [`docs/insights`](docs/insights).

### Prediction model (stage 3)

```bash
uv run f1pred transform      # builds the ml_* feature tables with the other marts
uv run f1pred evaluate       # tune, train, score; writes data/reports/model/report.md
```

The default split follows the plan (train 2018-2023, validate 2024, test 2025, 2026 scored
separately) and adapts to the seasons you have loaded. Set it yourself with
`--train 2018-2022 --validation 2023 --test 2024 --holdout 2025`.

On the demo data, simulate enough seasons for a split first:

```bash
uv run f1pred demo --seasons 2019-2026
uv run f1pred --demo transform
uv run f1pred --demo evaluate
```

### Replay and HUD (stage 4)

```bash
uv run f1pred replay 202405              # a past race, lap by lap, in the terminal
uv run f1pred serve                      # HUD at http://localhost:8000
uv run f1pred serve --openf1-session 9472 --race-id 202401   # also stream an OpenF1 session
```

`serve` loads the models saved by `f1pred evaluate`. Add `--demo` to use the demo
warehouse. Following a race that is running now needs an OpenF1 account token in
`F1PRED_OPENF1_TOKEN` and the race distance (`--total-laps 57`); past sessions are free.

The whole demo, with nothing installed but Docker (the first start builds the synthetic
warehouse and models, which takes a few minutes):

```bash
docker compose up                        # http://localhost:8000
```

### Running with Dagster

```bash
uv run dagster dev       # opens the UI at http://localhost:3000
```

In the UI you can see the asset lineage graph, backfill any season partition, and view
data quality check results. The `weekly_refresh` schedule re-ingests the current season
every Monday at 06:00 UTC.

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

CI runs lint, type checks and tests on every push and pull request, and builds the
Docker image and streams a replay from it. The tests include an
end-to-end run: synthetic data with known tyre degradation and pit crew speeds goes through
DuckDB and `dbt build`, and the tests check that the marts recover those true values. CI
also publishes the dbt docs site (lineage graph included) as a build artifact.

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
| `raw_fastf1_laps` | one row per driver per lap: compound, tyre life, stint, pit in/out, track status |
| `raw_fastf1_weather` | one row per weather sample (about every minute) |
| `dim_driver` | one row per driver, mapping Ergast IDs to codes and car numbers |

### dbt models (stage 2)

Built into the `staging`, `intermediate` and `marts` schemas of the same warehouse.

| Model | Grain |
| --- | --- |
| `dim_race` | one row per race, with circuit and race length |
| `dim_circuit` | one row per circuit |
| `dim_driver` | one row per driver, Ergast id to code and car number |
| `dim_constructor` | one row per team, with display name and colour |
| `fct_race_results` | one row per driver per race start, with grid slot and places gained |
| `fct_qualifying` | one row per driver per qualifying, with % gap to pole and team mate |
| `fct_laps` | one row per driver per lap, with gap to leader and tyre data |
| `fct_pit_stops` | one row per pit stop, with delta to the race's median stop |
| `mart_grid_conversion` | season x grid slot: wins, podiums, points |
| `mart_circuit_overtaking` | season x circuit: how closely the finish follows the grid |
| `mart_driver_race_craft` | season x driver: places gained versus grid expectation |
| `mart_pit_stop_team_performance` | season x team: pit stop speed and consistency |
| `mart_tyre_degradation` | season x circuit x compound: fuel-corrected degradation |
| `ml_pre_race_features` | one row per driver per race start: grid, qualifying gaps, form, track history, labels |
| `ml_lap_features` | one row per driver per lap: position, gaps, recent pace, tyres, stops, labels |
