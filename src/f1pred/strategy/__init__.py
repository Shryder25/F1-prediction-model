"""Race strategy analysis: stint pace, tyre degradation, undercuts, safety car stops and
pit windows.

    from f1pred.strategy import race_strategy
    rs = race_strategy(con, 202405)
    rs.pit_battles

See docs/strategy/README.md for the method behind each metric.
"""

from f1pred.strategy.config import DEFAULT_CONFIG, StrategyConfig
from f1pred.strategy.race import RaceStrategy, analyse_race, load_laps, race_strategy
from f1pred.strategy.season import (
    build_tables,
    headline_findings,
    safety_car_gains,
    undercut_success_rates,
    window_adherence,
)

__all__ = [
    "DEFAULT_CONFIG",
    "RaceStrategy",
    "StrategyConfig",
    "analyse_race",
    "build_tables",
    "headline_findings",
    "load_laps",
    "race_strategy",
    "safety_car_gains",
    "undercut_success_rates",
    "window_adherence",
]
