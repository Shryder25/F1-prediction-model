# Strategy analysis

Four questions a race strategist asks after every race, answered from lap timing and
tyre data:

1. How fast was each stint once fuel is taken out, and how quickly did the tyres fade?
2. When two cars fought through the pit stops, did the undercut or the overcut work?
3. How much did a stop under a safety car or VSC save?
4. Was each stop on the right lap, given the tyres the team chose?

Code: [`src/f1pred/strategy`](../../src/f1pred/strategy). Tests with hand-built races
where the answer is known on paper: [`tests/test_strategy.py`](../../tests/test_strategy.py).

```bash
uv run f1pred strategy                   # every race -> strategy.* tables + headline rates
uv run f1pred strategy --race 202405     # one race, printed
```

```python
from f1pred.strategy import race_strategy

rs = race_strategy(con, 202405)  # RaceStrategy: four DataFrames
rs.stints, rs.pit_battles, rs.safety_car_stops, rs.pit_windows
```

`analyse_race(laps, pit_stops)` does the same on plain DataFrames, so it also works on a
live session or a notebook extract. The post-race debrief builds on these tables.

All tunable assumptions live in one place, `StrategyConfig`
([`config.py`](../../src/f1pred/strategy/config.py)).

## 1. Fuel-corrected stint pace and degradation

**Method.** Keep representative laps only: slicks, not lap 1, not in or out laps, no SC,
VSC or red flag, FastF1 `IsAccurate`, and no slower than 7% over the driver's median for
that stint (traffic, mistakes). Each lap is corrected by 0.05 s for every lap of fuel
still on board, so it reads as if on an empty tank. Then per stint:

- `fuel_corrected_pace_s`: median corrected lap.
- `degradation_s_per_lap`: slope of a straight line of corrected lap time on tyre age.
- `degradation_r2`: how well the line fits. Low values mean the driver was managing
  pace, or the stint was broken up by traffic.
- `is_usable`: at least 5 clean laps.

The rules match the dbt model `int_stint_laps`, and a test checks the fuel correction
equals the dbt var, so these numbers agree with `mart_tyre_degradation`.

**Limits.** One fuel effect for every car and circuit. Track evolution makes later laps
faster and hides some wear. Drivers lift and coast to save tyres, so a low slope can be
"managed", not "durable".

## 2. Undercut and overcut success

**Method.** For every pit stop, look at the cars directly ahead and behind at the end of
the lap before. A pair is a pit battle when the car behind is within 3.5 s and the other
car pits within 6 laps:

- **Undercut**: the car behind (the attacker) pits first.
- **Overcut**: the car ahead pits first and the attacker stays out.

Gaps are total race time at the line. The result is judged at the end of the second
car's out-lap, when both have stopped:

| Column | Meaning |
| --- | --- |
| `gap_before_s` | attacker's deficit before the first stop (positive = behind) |
| `gap_after_s` | after both stops (positive = attacker now ahead) |
| `gain_s` | time the attacker made up over the cycle |
| `succeeded` | the attacker came out ahead |

Cycles that touch a safety car, VSC or red flag are left out, because they are a
different decision (section 3). Cars pitting on the same lap are not a battle.

**Limits.** A team may pit "first" for reasons other than attacking (tyre damage,
covering a car further back). The rule cannot see intent, only the pair and the timing.
Success is judged one lap after the second stop; an attacker can pass and then lose
the place on cold tyres a lap later.

## 3. Safety car pit gains

**Method.** The pit loss of a stop is the in-lap plus out-lap, minus the median in-lap
and out-lap of the cars that did not pit on those laps. Comparing with the field on the
same laps removes everything that hits everybody: the safety car itself, fuel, track
evolution. A stop is a safety car stop when the SC or VSC was out on its in-lap.

- `green_pit_loss_s`: that race's median green-flag pit loss.
- `time_saved_s`: green loss minus this stop's loss.
- `positions_gained`: places the stop kept compared with the same stop under green. At
  the end of the out-lap, count the cars that would have been ahead had the stop cost
  the green loss, minus the cars that actually were. The other cars are held as raced.

**Limits.** The cars staying out are on older or newer tyres than the car that stops,
so single green losses spread by about a second; the median across the race smooths
that. The places counterfactual moves only the stopping car; when half the field
pits under the same safety car it overstates the gain for each of them.

## 4. Pit window model

**Method.** Each stint's lap time is modelled as `fresh pace + degradation x tyre age`,
from the stint fits in section 1 (falling back to the race's median for that compound
when a stint has too few clean laps). For each stop, keep everything else as raced
(compounds, number of stops, the other stops) and slide this stop's lap. The modelled
time over the two stints either side is a parabola in the stop lap:

- `optimal_pit_lap`: the lap with the lowest modelled time.
- `window_open` / `window_close`: every lap within 1.0 s of that best time.
- `verdict`: `early`, `in window` or `late`, or `safety car` for a stop made under a
  SC, VSC or red flag (the tyre model cannot judge those; see section 3).
- `time_lost_s`: modelled cost of the actual lap against the best.

The pit loss and fuel are the same whichever lap the stop is on, so they drop out. A
driver who retired is modelled as if they had raced to the flag.

**Limits.** The model knows tyres and nothing else. Real stops leave the window for good
reasons: covering an undercut, avoiding traffic on the out-lap, a safety car, or a tyre
about to fall off a cliff that a straight line does not predict. "Early" is a question
to ask, not a verdict on the strategist.

## How to read the results

Run `uv run f1pred strategy` on the real 2018 onward warehouse, and the
`## Strategy findings` section of `data/reports/results.md` fills in. On the demo
warehouse the undercut almost always works, because the synthetic tyres have no warm-up
phase; real data will not be that one-sided.

## Finding

_To fill in from real data with `f1pred strategy`._

## Talking points

Short versions of the decisions behind this, for interviews:

- **Why fuel-correct?** A car is roughly 0.05 s a lap faster per lap of fuel burned,
  which is about the same size as hard-tyre wear. Without it, hards look like they get
  faster with age.
- **Why measure pit loss against the field, not the pit lane time?** Pit lane time is
  the same under a safety car; what changes is how slow everyone else is. The delta to
  cars staying out is what decides track position.
- **Why judge undercuts at the end of the second out-lap?** That is the first lap where
  both cars have paid the same pit loss, so the gap compares like with like.
- **Why a window, not a single best lap?** Near the optimum the cost curve is flat:
  a few laps either side often cost under a second, which is why teams react to
  rivals rather than chase the exact lap.
