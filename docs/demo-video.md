# Demo video: script and shot list

**Length:** 2 to 2.5 minutes. Recruiters skim, so the first 15 seconds say what it is and
why it matters.
**Audience:** hiring managers for F1 data engineer and data insight analyst roles.
**Story:** raw timing data in, trusted tables in the middle, decisions out. Show the
engineering (pipeline, tests, orchestration) and the insight (what the data says about
strategy), then the live prediction as the payoff.
**Record:** 1080p, browser zoom 125%, dark theme, notifications off. OBS or the macOS
screen recorder is enough. Record each shot separately and cut them together.

Fill the `[brackets]` from `data/reports/results.md` after the real pipeline run.

## Shot list

| # | Time | On screen | Voice-over |
| --- | --- | --- | --- |
| 1 | 0:00-0:12 | HUD replaying a real race, win probability chart moving | "This is a race predictor I built that updates every driver's chance of winning, lap by lap, from live timing data. Here it is replaying the [2025 race name]." |
| 2 | 0:12-0:30 | Architecture diagram from the README, zoomed and panned left to right | "Under it is a full data platform: Jolpica and FastF1 for history, OpenF1 for live timing, a DuckDB warehouse modelled with dbt, Dagster for orchestration, and a FastAPI service streaming to the browser." |
| 3 | 0:30-0:50 | Terminal: `uv run f1pred pipeline --from load`, sped up 4x, then the checks printing PASS | "One command runs the pipeline. Ingestion is rate limited, cached and idempotent, so a failed run resumes where it stopped. Ten data quality checks run on every load." |
| 4 | 0:50-1:05 | Dagster UI: asset lineage graph, then a season partition and its asset checks | "Dagster runs it as season-partitioned assets on a weekly schedule, with the checks attached to the assets they test." |
| 5 | 1:05-1:20 | dbt docs lineage graph, then a test list in the terminal (`f1pred transform` output ending in PASS=...) | "The warehouse has staging, intermediate and mart layers, with [56] data tests, including one that fails the build if a model feature could see the race it predicts." |
| 6 | 1:20-1:45 | Streamlit dashboard: Grid to finish tab, then Tyres, then Pit stops | "The marts answer strategy questions. From pole, [X]% of races since 2018 were won. At [circuit], the finish follows the grid least, so strategy matters most there. [One tyre or pit stop finding.]" |
| 7 | 1:45-2:05 | `data/reports/model/report.md`: the score table, then the calibration plot | "The model is tested on a season it never saw. Against the grid-order baseline it cuts the win Brier score by [X]%, and when it says 30%, it happens about 30% of the time." |
| 8 | 2:05-2:20 | Back to the HUD: a pit stop or safety car moves the probabilities; hover the pit window | "Live, the same code reads OpenF1. When a safety car comes out, you can see the chances shift and the pit window open." |
| 9 | 2:20-2:30 | README top with the live demo link and GitHub URL | "The demo is live at [link] and the code is on GitHub. Thanks for watching." |

## Tips

- Pick a replay race with a safety car or a late lead change, so shot 1 and shot 8 move.
  Scrub through a few in the HUD first.
- Say numbers once and show them on screen at the same time.
- Keep the terminal font at 18pt or more; viewers watch on phones.
- Add captions (YouTube auto-captions, then fix the F1 names). Many people watch muted.
- Post it on LinkedIn as a native video with a three-line summary and the repo link, and
  put the YouTube link in the README.
