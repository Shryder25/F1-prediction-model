# Pit stop performance by team

**Question.** Which teams turn their cars around fastest and most reliably?

**Why it matters.** A second lost in the pit lane can lose the undercut or a position, so
crew speed and consistency are part of a team's race strategy options.

## Method

1. Jolpica pit stop durations are pit lane time, from entry to exit, not the stationary time.
2. Pit lane length varies by circuit, so raw times are not comparable across races. Each
   stop is compared with the **median stop in the same race**.
3. Stops over 60 s (red flags, penalties, repairs) are flagged as outliers, counted, and
   left out of the medians.
4. Per team and season: median delta to the race median, and the share of slow stops
   (more than 2 s over the race median).

## Caveats

- Pit lane time includes the driver's entry and exit, not only the crew.
- Drive-through and stop-go penalties under 60 s are not separated from normal stops.
- The race median depends on which teams stopped in that race.

## Finding

_To fill in from real data with `f1pred findings` and the Pit stops tab._
