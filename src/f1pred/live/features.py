"""Lap-by-lap model features, built incrementally from a stream of laps.

This is the streaming twin of the dbt model `ml_lap_features`: same columns,
same definitions, but computed one lap at a time from what a timing feed shows,
because the warehouse does not exist mid-race. A parity test
(`tests/test_live.py`) replays demo races through this builder and checks every
feature against the dbt table, so training and serving cannot drift apart
without a failing test.

The SQL is the reference. Where it has a quirk, this file copies it on purpose
(for example, recent pace averages the clean laps among the driver's last 3
laps, not the last 3 clean laps).
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from f1pred.live.events import CarLap, LapSnapshot, RaceInfo
from f1pred.model.dataset import COMPOUNDS, LAP_FEATURES

# Must match the dbt var `recent_pace_laps`.
RECENT_PACE_LAPS = 3
# A lap slower than this multiple of the field median is not a clean lap.
CLEAN_LAP_LIMIT = 1.07


@dataclass
class _DriverState:
    elapsed_s: float | None = None
    pit_stops: int = 0
    # (lap time - field median) for each of the driver's recent laps; None if not clean.
    recent: deque[float | None] = field(default_factory=lambda: deque(maxlen=RECENT_PACE_LAPS))


class LapFeatureBuilder:
    """Feed it each `LapSnapshot` in order; get the lap model's input rows back."""

    def __init__(self, info: RaceInfo) -> None:
        self.info = info
        self._entries = {e.driver_id: e for e in info.entries}
        self._state: dict[str, _DriverState] = {}
        self._last_lap = 0

    def update(self, snapshot: LapSnapshot) -> pd.DataFrame:
        if snapshot.lap <= self._last_lap:
            raise ValueError(f"Lap {snapshot.lap} arrived after lap {self._last_lap}.")
        self._last_lap = snapshot.lap
        # Cars the race never listed (e.g. not in the starting grid) cannot be scored.
        cars = sorted(
            (c for c in snapshot.cars if c.driver_id in self._entries), key=lambda c: c.position
        )
        times = [c.lap_time_s for c in cars if c.lap_time_s is not None]
        median = statistics.median(times) if times else None

        for car in cars:
            state = self._state.setdefault(car.driver_id, _DriverState())
            if car.lap_time_s is not None:
                state.elapsed_s = (state.elapsed_s or 0.0) + car.lap_time_s
            state.pit_stops += car.pit_stops
            clean = _is_clean(car, snapshot.lap, median)
            state.recent.append(
                car.lap_time_s - median
                if clean and car.lap_time_s is not None and median is not None
                else None
            )

        leader = next((c for c in cars if c.position == 1), None)
        leader_elapsed = self._state[leader.driver_id].elapsed_s if leader else None
        gaps = [_minus(self._state[c.driver_id].elapsed_s, leader_elapsed) for c in cars]

        rows = []
        for i, car in enumerate(cars):
            entry = self._entries[car.driver_id]
            state = self._state[car.driver_id]
            gap = gaps[i]
            ahead = gaps[i - 1] if i > 0 else None
            behind = gaps[i + 1] if i + 1 < len(cars) else None
            recent = [v for v in state.recent if v is not None]
            row: dict[str, object] = {
                "race_id": self.info.race_id,
                "season": self.info.season,
                "round": self.info.round,
                "driver_id": car.driver_id,
                "lap": snapshot.lap,
                "total_laps": self.info.total_laps,
                "race_progress": snapshot.lap / self.info.total_laps,
                "laps_remaining": self.info.total_laps - snapshot.lap,
                "position": car.position,
                "cars_running": len(cars),
                "grid_position": entry.grid_position,
                "positions_gained_so_far": entry.grid_position - car.position,
                "gap_to_leader_s": gap,
                # Same clamping, and the same null rules, as the SQL.
                "gap_to_car_ahead_s": _clamped_gap(gap, ahead) if ahead is not None else None,
                "gap_to_car_behind_s": _clamped_gap(behind, gap) if behind is not None else None,
                "recent_pace_vs_field_s": sum(recent) / len(recent) if recent else None,
                "compound": car.compound,
                "tyre_age_laps": car.tyre_age_laps,
                "stint": car.stint,
                "pit_stops_made": state.pit_stops,
                "is_safety_car": bool(car.is_safety_car),
                "is_virtual_safety_car": bool(car.is_virtual_safety_car),
                "gap_to_pole_pct": entry.gap_to_pole_pct,
                "driver_finish_last5": entry.driver_finish_last5,
                "constructor_points_last5": entry.constructor_points_last5,
            }
            for compound in COMPOUNDS:
                row[f"compound_{compound.lower()}"] = (
                    float("nan") if car.compound is None else float(car.compound == compound)
                )
            rows.append(row)

        frame = pd.DataFrame(rows)
        if frame.empty:
            return frame
        for column in LAP_FEATURES:
            frame[column] = pd.to_numeric(frame[column].astype("float64"), errors="coerce")
        return frame


def _is_clean(car: CarLap, lap: int, median: float | None) -> bool:
    if lap <= 1 or car.lap_time_s is None or median is None:
        return False
    if car.is_pit_in_lap or car.is_pit_out_lap:
        return False
    if car.is_safety_car or car.is_virtual_safety_car or car.is_red_flag:
        return False
    return car.lap_time_s < CLEAN_LAP_LIMIT * median


def _minus(a: float | None, b: float | None) -> float | None:
    return None if a is None or b is None else a - b


def _clamped_gap(behind: float | None, ahead: float | None) -> float:
    """greatest(behind - ahead, 0). DuckDB's greatest() skips nulls, so null gives 0."""
    return 0.0 if behind is None or ahead is None else max(behind - ahead, 0.0)
