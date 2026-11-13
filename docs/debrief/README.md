# Automated post-race debrief

After every race, `f1pred debrief` writes the report a race insight analyst would send on
Monday morning, and GitHub Pages publishes it:

- **Summary**: five or six sentences: who won and whether we saw it coming, how the
  prediction scored, when the lap model became sure of the winner, the biggest surprise,
  the strategy story.
- **Predicted vs actual**: the pre-race prediction for every driver next to the result,
  with podium hits, position error against the "finish where you started" baseline, and
  the Brier score of the win probabilities.
- **Win probability by lap**: the race replayed through the live lap model, which is what
  the HUD would have shown.
- **Strategy calls**: each driver's stints and compounds, with each stop judged early, in
  window or late against the modelled pit window.
- **Tyre degradation**: fuel-corrected seconds lost per lap of tyre age, per compound.
- **Undercuts and overcuts**: every attempt between cars running close, the gap before and
  after, and whether it gained the place.
- **Stops under safety car**: who pitted under SC or VSC and how much time it saved.

Example layout (demo data): see the site published from this repository's Pages.

## How it is built

```
weekly_refresh (Mon 06:00)            Dagster schedule: pull the weekend's season partition
        │ on success
        ▼
debrief_after_refresh (sensor) ──► post_race_debrief job:
        warehouse_tables ──► dbt_marts (dbt build, all tests) ──► race_debrief
                                                                    │  + 2 asset checks
                                                                    ▼
                                         docs/debriefs/<season>-<round>.json / .md
        git push ──► .github/workflows/debrief-pages.yml ──► GitHub Pages
```

| Part | Where |
| --- | --- |
| Gathering the numbers | `src/f1pred/debrief/build.py` |
| The report as data (JSON schema) | `src/f1pred/debrief/model.py` |
| Markdown and HTML | `src/f1pred/debrief/render.py` |
| Site and index | `src/f1pred/debrief/site.py` |
| Strategy metrics | `f1pred.strategy.race_strategy` (the strategy module; the debrief only reports them) |
| Orchestration | `race_debrief` asset, checks and sensor in `src/f1pred/orchestration/definitions.py` |

Design choices:

- **The debrief is data first.** Each race is saved as JSON and rendered to Markdown (read
  on GitHub, reviewable in a PR) and HTML (Pages). The site rebuilds from the JSON in
  seconds, without the warehouse or the models.
- **No new metric logic.** Predictions come from the models `f1pred evaluate` saved,
  replayed with the same code as the live HUD. Strategy numbers come from the strategy
  module. The debrief only joins, scores and writes.
- **Honest scoring.** The deployed models are trained on every season up to the test
  season. If a debriefed race falls inside those seasons, the report says it is
  in-sample. Without saved models, it falls back to grid order and says so.
- **Checked like any other asset.** `debrief_covers_every_starter` (error) fails if a
  starter is missing or the predicted order has gaps. `debrief_has_strategy_when_tyre_data`
  (warning) flags a race with FastF1 data but no strategy section, usually because FastF1
  published later than Jolpica.

## Run it

On demo data (no network):

```bash
uv run f1pred demo --seasons 2019-2026
uv run f1pred --demo transform
uv run f1pred --demo evaluate --quick
uv run f1pred --demo debrief --season 2026        # data/demo/debriefs/*.md
uv run f1pred debrief-site --src data/demo/debriefs --out site
open site/index.html
```

On real data, after `f1pred pipeline` has run once:

```bash
uv run f1pred ingest --seasons 2026 && uv run f1pred ingest-fastf1 --seasons 2026
uv run f1pred load && uv run f1pred transform
uv run f1pred debrief                 # latest race -> docs/debriefs/
uv run f1pred debrief 202605 202606   # or chosen races; --season 2026 for all of them
git add docs/debriefs && git commit -m "Debrief: <race>" && git push
```

Or let Dagster do it every Monday: `uv run dagster dev`, then switch on the
`weekly_refresh` schedule and the `debrief_after_refresh` sensor. The run writes the
debrief to `docs/debriefs/`; commit and push it to publish.

## Publish on GitHub Pages (one time)

1. Repository **Settings > Pages > Build and deployment > Source: GitHub Actions**.
2. Run the **Publish race debriefs** workflow once (Actions tab > Run workflow), or push a
   debrief. The site is at `https://<user>.github.io/F1-prediction-model/`.

Until a real debrief is committed, the workflow publishes the demo season's debriefs with a
"synthetic demo data" banner, so the link works from day one.

## Limits

- The debrief needs the race in the warehouse: Jolpica usually has results within hours,
  FastF1 tyre data within a day. Re-run the debrief if the strategy section was missing.
- Win probabilities come from the lap model replayed after the fact over the same timing
  data the live feed would have had. They are not a recording of a live run.
- Commit-and-push is manual on purpose: a person reads the report before it goes public.
