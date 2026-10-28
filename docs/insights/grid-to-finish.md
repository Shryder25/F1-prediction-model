# Qualifying to race conversion

**Question.** How much does the starting position decide the result, where is that less
true, and which drivers beat their grid slot?

**Why it matters.** It quantifies how much a qualifying gain is worth on Sunday, which
circuits reward an aggressive race strategy, and it gives the baseline that the step 3
model must beat ("finish where you started").

## Method

1. One row per race start (`fct_race_results`). Pit lane starters are placed behind the
   last car on the grid. Did-not-start and withdrawn entries are excluded.
2. **Conversion by grid slot** (`mart_grid_conversion`): win, podium and points rates. The
   mart stores counts, so rates stay correct when pooling seasons.
3. **Overtaking by circuit** (`mart_circuit_overtaking`): correlation between grid and
   finishing position among classified finishers. Near 1 means the race finished in grid
   order.
4. **Race craft by driver** (`mart_driver_race_craft`): average places finished ahead of
   the average finish for their grid slot, across all seasons.

## Caveats

- Retirements are excluded from places gained, so a driver who often retires from the
  front can look better than they were. The mart reports the non-classified rate beside it.
- "Beating the grid slot" mixes driver skill with a car that is better in race trim than
  over one lap.
- About 20 races a season means small samples per circuit; look at several seasons.

## Finding

_To fill in from real data with `f1pred findings` and the Grid to finish tab._
