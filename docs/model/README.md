# Prediction model: method and limitations

Stage 3 answers two questions:

1. **Before the race:** what is each driver's chance of winning, of a podium, and where
   will they finish?
2. **Lap by lap:** given the race so far, how do those chances change?

Run `f1pred evaluate` to train everything and write `data/reports/model/report.md`. The
numbers in that report are the result; this page explains how they are produced and
how far to trust them.

## Features

All features are built in dbt (`transform/models/marts/ml/`), next to the marts they come
from, so they are versioned, documented and tested like any other table.

**Pre-race** (`ml_pre_race_features`, one row per driver per race start)

| Feature | Meaning |
| --- | --- |
| `grid_position`, `started_from_pit_lane` | Where the car starts, after penalties |
| `qualifying_position`, `gap_to_pole_pct`, `gap_to_team_mate_pct` | One-lap pace |
| `driver_finish_last5`, `driver_points_last5`, `driver_gained_last5` | Form over the previous 5 starts |
| `driver_dnf_rate_last10` | Reliability and incident rate over the previous 10 starts |
| `driver_track_gained`, `driver_track_starts` | Places gained at this circuit in earlier seasons |
| `constructor_points_last5`, `constructor_finish_last5` | Car form over the previous 5 races |
| `championship_position_before`, `championship_points_before` | Standings after the previous round |

**Per lap** (`ml_lap_features`, one row per driver per completed lap): race progress,
position, gap to the leader and to the cars ahead and behind, recent pace against the
field (last 3 clean laps), tyre compound and age, stint, pit stops made, safety car flags,
and three pre-race strength features.

### No look-ahead

A feature that knows the result makes a model look brilliant and is worthless live. Three
guards stop that:

- every rolling window ends at the previous race (`rows between N preceding and 1 preceding`),
  and standings are joined from the previous round;
- the dbt test `ml_form_uses_only_past_races` recomputes driver form a second way and fails
  the build if the two disagree;
- labels are prefixed `label_` and a unit test checks that no label is in a feature list.

## Evaluation protocol

Seasons, not random rows, are split, because the model will always be used on a season it
has not seen:

| Seasons | Role |
| --- | --- |
| 2018 to 2023 | Train |
| 2024 | Choose settings (regularisation strength, tree depth and count) |
| 2025 | Test: scored once, after every choice is made |
| 2026 | Holdout, reported on its own: new car regulations, so past form should transfer less well |

Each model family is compared with a baseline that needs no model at all:

- **Pre-race:** the grid baseline. Predicted order is the grid; win and podium chances are
  the historical rates for each grid slot.
- **Lap by lap:** the current position baseline. Predicted order is the order now; chances
  are the historical rates for that position at that stage of the race.

Beating these is the bar. In F1 the grid already explains most of the result, so a model
that only matches it has learned nothing useful.

### Scores

- **Brier score** for win and podium: the mean squared error of the probability. Lower is
  better, and it rewards both being right and being honest about uncertainty.
- **Log loss**: like Brier, but punishes confident misses harder.
- **Position MAE**: how many places the predicted finishing order is off, on average.
- **Calibration**: when the model says 30%, does it happen 30% of the time? Shown as a
  reliability plot and summarised as expected calibration error.
- **Favourite won**: the share of races the top win probability got right. Easy to read,
  but noisy, so it is never used to pick a model.

### Confidence intervals

A season has about 24 races, so a small difference in score can be luck. Every model is
compared with its baseline on the same races, and the races are resampled with replacement
2,000 times to give a 95% interval for the difference. A model is only called better when
the whole interval is below zero. Races, not rows, are resampled because the drivers in one
race are not independent: exactly one of them wins.

### Making predictions consistent

Win, podium and position are separate models, so their raw outputs need not agree. Before
scoring, win probabilities in each race are scaled to sum to 1, podium probabilities to sum
to 3 (none above 1), and expected positions are turned into a ranking. Every model, the
baselines included, goes through the same step.

## Models

- **Logistic regression** (with ridge regression for position): missing values imputed
  with the training median plus a missing-value flag, features standardised.
- **XGBoost**: shallow trees with strong regularisation, because the data is small.

The model with the better validation score is refit on 2018 to 2025 and saved to
`models/pre_race.joblib` and `models/lap.joblib` for the stage 4 live predictor.

## Known limitations

- **Small data.** About 24 races and 20 drivers per season. Expect wide intervals, and
  treat any single-race story as an anecdote.
- **Truncated history.** Form features start in the first loaded season, so a driver's
  first races in 2018 look like a debut. Loading earlier Jolpica seasons would fix this for
  the non-tyre features.
- **Race length.** `total_laps` is the distance actually raced. It matches the scheduled
  distance except in races shortened by a red flag, where it slightly leaks the future.
- **No weather or strategy model yet.** Rain and safety cars are only seen through their
  effect on lap times and flags. The lap model does not know about pit windows; that is a
  stage 4 question.
- **Independent rows in the lap model.** Laps from one race are strongly correlated. The
  intervals account for this by resampling races, but the model itself treats every lap as
  a separate example, which overweights long races.
- **The demo data is not evidence.** The synthetic races are simple and give only a few
  races per season. Use them to check the pipeline runs, not to judge the model.
