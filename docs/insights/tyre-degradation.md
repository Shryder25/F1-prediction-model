# Tyre degradation by compound and circuit

**Question.** How much lap time does each compound lose per lap of use, at each circuit?

**Why it matters.** Degradation sets the length of each stint, so it decides the number of
stops and when the undercut works. It is also a core feature for the step 3 model.

## Method

1. Start from FastF1 laps (2018 onward), joined to Jolpica laps by race, car number and lap.
2. Keep representative laps only: slick tyres, not lap 1, not in or out laps, no safety car,
   VSC or red flag, FastF1 `IsAccurate`, and within 7% of the driver's median for that stint.
3. **Fuel correction.** Cars get faster as fuel burns off, which hides tyre wear. Each lap
   is corrected by 0.05 s per lap of fuel still on board (about 0.03 s per kg at about
   1.7 kg per lap). This is the `fuel_correction_s_per_lap` dbt var.
4. Fit a straight line of fuel-corrected lap time against tyre age for every stint with at
   least 5 clean laps. The slope is the degradation in seconds per lap.
5. Report the median slope (and the 25th to 75th percentile range) per season, circuit and
   compound.

The end-to-end tests run this on synthetic data with known degradation and check the
models recover it to within 15%.

## Caveats

- Drivers manage tyres, so a low slope can mean "managed pace", not "durable tyre".
- One fuel correction for every car and circuit is a simplification.
- Track evolution (the surface gets faster through a race) partly offsets wear, so these
  slopes understate raw tyre wear a little.
- Compound names are relative: a 2019 SOFT at one circuit is not the same rubber as at
  another (Pirelli picks C1 to C5 per event).

## Finding

_To fill in from real data with `f1pred findings` and the Tyres tab._
