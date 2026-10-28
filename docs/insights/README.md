# Insight analyses

Three analyses built on the dbt marts. Each write-up states the question, the method and
its limits. The **Finding** sections are filled in from the real 2018 onward warehouse:
run `uv run f1pred findings` after ingesting, and paste the result with a sentence on
what it means for race strategy.

| Analysis | Marts | Dashboard tab |
| --- | --- | --- |
| [Tyre degradation by compound and circuit](tyre-degradation.md) | `mart_tyre_degradation` | Tyres |
| [Pit stop performance by team](pit-stops.md) | `mart_pit_stop_team_performance` | Pit stops |
| [Qualifying to race conversion](grid-to-finish.md) | `mart_grid_conversion`, `mart_circuit_overtaking`, `mart_driver_race_craft` | Grid to finish, Overtaking, Race craft |
