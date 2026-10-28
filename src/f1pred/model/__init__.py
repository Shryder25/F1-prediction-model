"""Stage 3: race outcome models and their evaluation.

Two predictors share one interface (`RaceModel`):

- pre-race: before the start, from the grid, qualifying and recent form;
- lap-by-lap: at the end of every lap, from the race state so far.

Both are scored on seasons they never saw, against simple baselines.
"""
