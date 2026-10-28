# Live predictions: architecture and limits

Stage 4 replays any past race lap by lap, or follows a race that is running now, and
streams predictions to a browser HUD. This page explains how it is built and what it
does not do.

## Data flow

```
             ┌──────────────────────┐
             │ WarehouseReplay      │  marts.fct_laps + fct_pit_stops (DuckDB)
             │ OpenF1Source         │  OpenF1 /laps /stints /pit /race_control
             └──────────┬───────────┘
                        │ RaceInfo once, then one LapSnapshot per lap
                        ▼            (positions, lap times, tyres, flags, stops)
             ┌──────────────────────┐
             │ LapFeatureBuilder    │  rebuilds ml_lap_features one lap at a time
             └──────────┬───────────┘
                        ▼
             ┌──────────────────────┐
             │ RacePredictor        │  pre_race.joblib before the start,
             │                      │  lap.joblib after every lap, plus pit windows
             └──────────┬───────────┘
                        │ RaceUpdate (pydantic, JSON)
                        ▼
             ┌──────────────────────┐
             │ FastAPI + SSE        │  replays: one stream per viewer
             │                      │  live: one Broadcaster shared by every viewer
             └──────────┬───────────┘
                        ▼
                    HUD (browser)
```

Code: [`src/f1pred/live`](../../src/f1pred/live).

## One code path for replays and live races

A source only says what a timing screen shows. It never hands the model ready-made
features. The replay deliberately reads `fct_laps`, not the `ml_lap_features` table the
model was trained on, so every replay exercises the same feature code a live race uses.

That raises the classic risk of serving: the features computed live drifting away from
the ones used in training. `LapFeatureBuilder` is a line-by-line port of the dbt SQL, and
`tests/test_live.py::test_streaming_features_match_the_dbt_table` replays every race of a
demo season through it and compares all 23 features, row by row, with the dbt table. Any
change to one side without the other fails CI.

## Server-Sent Events

The browser only listens, so SSE is simpler than WebSockets: plain HTTP, automatic
reconnects, and it passes through proxies.

| Endpoint | What it does |
| --- | --- |
| `GET /` | The HUD: one HTML file, no build step, no external scripts |
| `GET /api/health` | Whether models loaded, demo or real data, live feed state |
| `GET /api/races` | Races that can be replayed, newest first |
| `GET /api/races/{race_id}/stream?interval=1` | A replay, `interval` seconds per lap |
| `GET /api/live/stream` | The live race, when the server was started with `--openf1-session` |

Every message is `event: update`, `id: <lap>` and a `RaceUpdate` as JSON. Details that
matter in production:

- **Resume.** A browser that loses the connection reconnects with `Last-Event-ID`. The
  server rebuilds the race state up to that lap without sending it, then carries on.
- **Heartbeats.** Live laps are 80 to 100 seconds apart, longer than many proxies keep an
  idle connection. A comment line goes out every 15 seconds while the feed is quiet.
- **Fan-out.** A live race is polled once and shared. Late joiners get the laps so far,
  so their chart is complete.
- **End of race.** The last update has `is_final`, then an `end` event tells the browser
  not to reconnect.

## The pit window panel

For each car: "if it pitted at the end of this lap, where would it rejoin?" A car rejoins
behind every car whose gap to the leader is smaller than its own gap plus the pit loss.

- **Pit loss** is the median pit lane time at this circuit in earlier races (no
  look-ahead), falling back to all earlier races, then 22 seconds.
- Under a safety car it is scaled to 50% (VSC 60%), because the field is slow.

This is a first answer to "does pitting now cost us a place?", not a strategy model. It
ignores traffic on the out lap, the pace of fresh tyres, and the fact that pit lane time
is not exactly the time lost against a car that stays out (that is usually a few seconds
less). A natural next step is to simulate the undercut with the tyre degradation mart.

## OpenF1 (live and past sessions)

`f1pred serve --openf1-session <key|latest>` adds a live stream. OpenF1 gives laps,
stints, pit stops and race control messages; `openf1.py` turns them into `LapSnapshot`s:

- A car's lap ends when it crosses the line (`date_start` + `lap_duration`, or the start
  of its next lap). The order of crossing is the race order, and the time since the start
  gives the gaps.
- Live, lap N is published once every car that finished lap N-1 has finished lap N, or
  once the leader is two laps further on. A car still missing then has stopped or been
  lapped twice, and is left out of that lap.
- Safety car, VSC and red flag laps come from race control messages.

**Access.** Past sessions are free and need no account, so you can point the live code
path at a finished race to test it. Data for a session that is running now needs a paid
OpenF1 account: put its token in `F1PRED_OPENF1_TOKEN`. The server keeps working without
it; `/api/health` reports the error.

**Linking to the warehouse.** Pass `--race-id 202605` (season × 100 + round) when the
race is in the warehouse, and drivers are matched by car number. That enables the
pre-race model and the lap model's three pre-race strength features. For a race that is
not loaded yet those are missing: the lap model handles missing values, and the lap 0
view falls back to the grid order. The warehouse builds pre-race features from race
results, so today a race only gets them once it is over; building them from qualifying
alone, so a live race gets them too, is the main gap to close before a race weekend.

## Known limitations

- **Lapped cars.** Jolpica (training) lists a lapped car's lap N when it completes it;
  live, a car two laps down is dropped from that lap, as above.
- **Race distance.** OpenF1 does not give the scheduled number of laps, so live races
  need `--total-laps` (or `--race-id`). A red-flag-shortened race keeps the scheduled
  distance, as in training.
- **Polling.** OpenF1's REST API is polled every 15 seconds (`--poll`), well inside its
  rate limit. Its MQTT stream would cut latency and is the next step for a real
  race-weekend deployment.
- **Model files are pickles.** `joblib` runs code on load, so only load models you
  trained.

## Running it

```bash
uv run f1pred --demo replay 202603                # a race in the terminal
uv run f1pred --demo serve                        # HUD at http://localhost:8000
uv run f1pred serve --openf1-session 9472         # also stream a past OpenF1 session
docker compose up                                 # the whole demo, nothing else needed
```

Deploying the public demo: the image listens on `$PORT`, builds the synthetic demo on
first start and reports health at `/api/health`, so it runs as is on Render, Fly.io,
Railway or a Hugging Face Docker Space. For real data, build the warehouse and models
locally and ship the `data` folder in the volume with `F1PRED_DEMO=0`.
