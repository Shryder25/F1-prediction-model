"""Assumptions behind the strategy metrics, in one place so they are easy to tune."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyConfig:
    # Lap time lost per lap of fuel still on board. Must match the dbt var
    # `fuel_correction_s_per_lap` so stint pace here agrees with the marts (a test
    # checks this).
    fuel_correction_s_per_lap: float = 0.05
    # A stint needs this many clean laps before its pace and degradation are trusted.
    min_stint_laps: int = 5
    # Clean laps slower than the stint median by more than this ratio are traffic,
    # mistakes or damage, not tyre performance.
    outlier_ratio: float = 1.07
    # Two cars are "in a pit battle" when the one behind is within this many
    # seconds before the first of them pits. About the most an undercut can gain.
    battle_max_gap_s: float = 3.5
    # ...and the second car pits within this many laps of the first.
    battle_max_lap_gap: int = 6
    # A stop lap is "in the window" when the modelled race time is within this many
    # seconds of the best possible lap for that stop.
    window_tolerance_s: float = 1.0


DEFAULT_CONFIG = StrategyConfig()
