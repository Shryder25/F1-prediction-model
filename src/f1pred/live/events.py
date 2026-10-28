"""The messages that flow through the live pipeline.

A source (a replay of a past race, or a live timing feed) produces a `RaceInfo`
once, then one `LapSnapshot` per completed race lap. The predictor turns each
snapshot into a `RaceUpdate`, which is what the HUD receives over SSE.

Sources only describe what a timing screen shows (positions, lap times, tyres,
flags). Every model feature is derived from that in `features.py`, the same way
for replays and live races, so the live path is exercised by every replay.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Entry(_Frozen):
    """One car in the race, as known before the start."""

    driver_id: str
    code: str
    car_number: int | None = None
    constructor_id: str | None = None
    team_name: str | None = None
    team_colour: str = "#888888"
    grid_position: int
    # Pre-race strength features for the lap model. None when unknown (live,
    # for a race not in the warehouse yet); the models handle missing values.
    gap_to_pole_pct: float | None = None
    driver_finish_last5: float | None = None
    constructor_points_last5: float | None = None


class RaceInfo(_Frozen):
    race_id: int
    season: int
    round: int
    race_name: str
    circuit_id: str | None = None
    # Scheduled race distance. Known before the start.
    total_laps: int
    # Typical time lost to a pit stop at this circuit, for the pit window panel.
    pit_loss_s: float
    entries: tuple[Entry, ...]


class CarLap(_Frozen):
    """One car crossing the line at the end of a lap."""

    driver_id: str
    position: int
    lap_time_s: float | None
    compound: str | None = None
    stint: int | None = None
    tyre_age_laps: int | None = None
    is_pit_in_lap: bool | None = None
    is_pit_out_lap: bool | None = None
    # Track status the car saw on this lap. None when the feed has no tyre data.
    is_safety_car: bool | None = None
    is_virtual_safety_car: bool | None = None
    is_red_flag: bool | None = None
    # Pit stops recorded on this lap (Jolpica / OpenF1 pit data), usually 0 or 1.
    pit_stops: int = 0


class LapSnapshot(_Frozen):
    """Every running car at the end of race lap `lap`."""

    lap: int
    cars: tuple[CarLap, ...]

    @property
    def track_status(self) -> str:
        """The most serious neutralisation any car saw on this lap."""
        if any(c.is_red_flag for c in self.cars):
            return "RED"
        if any(c.is_safety_car for c in self.cars):
            return "SC"
        if any(c.is_virtual_safety_car for c in self.cars):
            return "VSC"
        return "GREEN"


class PitWindow(_Frozen):
    """What a pit stop now would cost, from the gaps behind this car.

    `rejoin_position` assumes the cars behind keep their gaps and the stop
    costs `pit_loss_s`. It ignores traffic and tyre offsets: a first answer to
    "does pitting now lose us a place?", not a strategy simulation.
    """

    rejoin_position: int
    places_lost: int
    # Seconds to spare (positive) or short (negative) of a stop with no places lost.
    margin_s: float | None


class DriverState(_Frozen):
    driver_id: str
    code: str
    team_name: str | None
    team_colour: str
    position: int
    grid_position: int
    gap_to_leader_s: float | None
    gap_to_car_ahead_s: float | None
    compound: str | None
    tyre_age_laps: int | None
    pit_stops: int
    p_win: float = Field(ge=0, le=1)
    p_podium: float = Field(ge=0, le=1)
    predicted_position: int
    pit: PitWindow | None = None


class RaceUpdate(_Frozen):
    """The race state and predictions after one lap (lap 0: before the start)."""

    race_id: int
    race_name: str
    season: int
    lap: int
    total_laps: int
    track_status: str  # GREEN, SC, VSC or RED
    drivers: tuple[DriverState, ...]
    is_final: bool = False
