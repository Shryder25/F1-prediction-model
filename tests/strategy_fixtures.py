"""Hand-built races with known physics, for the strategy tests.

Every lap time is exact (no noise), so a test can work out what the metric should
be on paper. Frames have the columns of marts.fct_laps and marts.fct_pit_stops.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import pandas as pd

FUEL_S_PER_LAP = 0.05


@dataclass
class Car:
    driver_id: str
    stints: Sequence[tuple[str, int]]  # (compound, last lap of the stint)
    pace_s: float = 0.0  # lap time offset from the base
    start_gap_s: float = 0.0  # extra time on lap 1, i.e. grid position
    extra_s: dict[int, float] = field(default_factory=dict)  # lap -> added time (traffic)


def build_race(
    cars: Iterable[Car],
    *,
    total_laps: int,
    degradation: dict[str, float],
    compound_offset: dict[str, float] | None = None,
    sc_laps: Sequence[int] = (),
    base_s: float = 90.0,
    pit_loss_s: float = 22.0,
    sc_slowdown_s: float = 30.0,
    sc_pit_share: float = 0.5,
    race_id: int = 202401,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a race. Returns (laps, pit_stops).

    Lap time = base + car pace + compound offset + degradation x tyre age + fuel
    + SC slowdown + pit loss on the out-lap (a share of it if the in-lap was under SC).
    """
    offsets = compound_offset or {}
    rows = []
    stops = []
    for car in cars:
        elapsed = 0.0
        start = 1
        for stint_no, (compound, end) in enumerate(car.stints, start=1):
            for lap in range(start, end + 1):
                age = lap - start + 1
                lap_s = (
                    base_s
                    + car.pace_s
                    + offsets.get(compound, 0.0)
                    + degradation[compound] * age
                    + FUEL_S_PER_LAP * (total_laps - lap)
                    + car.extra_s.get(lap, 0.0)
                )
                if lap == 1:
                    lap_s += car.start_gap_s
                if lap in sc_laps:
                    lap_s += sc_slowdown_s
                pit_out = stint_no > 1 and lap == start
                pit_in = lap == end and stint_no < len(car.stints)
                if pit_out:
                    lap_s += pit_loss_s * (sc_pit_share if lap - 1 in sc_laps else 1.0)
                if pit_in:
                    stops.append({"race_id": race_id, "driver_id": car.driver_id, "lap": lap})
                elapsed += lap_s
                rows.append(
                    {
                        "race_id": race_id,
                        "driver_id": car.driver_id,
                        "lap": lap,
                        "total_laps": total_laps,
                        "lap_time_s": round(lap_s, 3),
                        "elapsed_s": round(elapsed, 3),
                        "compound": compound,
                        "stint": stint_no,
                        "tyre_age_laps": age,
                        "is_pit_in_lap": pit_in,
                        "is_pit_out_lap": pit_out,
                        "is_safety_car": lap in sc_laps,
                        "is_virtual_safety_car": False,
                        "is_red_flag": False,
                        "is_accurate": not (pit_in or pit_out or lap == 1 or lap in sc_laps),
                        "has_tyre_data": True,
                    }
                )
            start = end + 1
    laps = pd.DataFrame(rows)
    laps["position"] = laps.groupby("lap")["elapsed_s"].rank(method="first").astype(int)
    return laps.sort_values(["lap", "position"], ignore_index=True), pd.DataFrame(stops)
