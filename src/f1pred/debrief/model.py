"""The debrief as data. Markdown, HTML and the site index are all rendered from this.

Saving the debrief as JSON next to the Markdown means the site can be rebuilt
(for a new page design, say) without the warehouse or the models.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RaceMeta(_Model):
    race_id: int
    season: int
    round: int
    race_name: str
    circuit_name: str | None = None
    race_date: str | None = None
    total_laps: int
    is_demo: bool = False


class DriverResult(_Model):
    """One starter: what the pre-race model expected and what happened."""

    driver_id: str
    code: str
    name: str
    team: str | None = None
    team_colour: str = "#888888"
    grid: int
    predicted_rank: int
    p_win: float | None = None
    p_podium: float | None = None
    finish: int | None = None  # classified position; None for a DNF
    status: str
    # predicted_rank - finish: positive means the driver beat the prediction.
    beat_prediction_by: int | None = None


class PredictionScore(_Model):
    source: str  # "saved models" or "grid order (no saved models)"
    in_sample: bool  # the models were trained on this race's season
    winner_predicted: bool
    winner_pre_race_p_win: float | None = None
    podium_hits: int  # of the predicted top 3, how many finished on the podium
    position_mae: float  # classified drivers only
    grid_position_mae: float  # the same, for "finish where you started"
    brier_win: float | None = None
    # First lap from which the eventual winner stayed the model's favourite.
    winner_locked_in_lap: int | None = None


class WinProbabilitySeries(_Model):
    code: str
    team_colour: str
    p_win: list[float]  # one value per lap, lap 0 = before the start


class Stint(_Model):
    compound: str | None
    start_lap: int
    end_lap: int


class DriverStrategy(_Model):
    code: str
    finish: int | None
    stops: int
    stints: list[Stint]
    pit_verdicts: list[str] = []  # e.g. "lap 18: early (window 20-24)"


class CompoundDegradation(_Model):
    compound: str
    stints: int
    median_degradation_s_per_lap: float
    median_pace_s: float | None = None


class PitBattle(_Model):
    attacker: str
    defender: str
    kind: str  # undercut or overcut
    attacker_pit_lap: int
    defender_pit_lap: int
    gap_before_s: float
    gap_after_s: float
    succeeded: bool


class SafetyCarStop(_Model):
    code: str
    lap: int
    track_status: str
    time_saved_s: float
    positions_gained: int | None = None


class StrategySection(_Model):
    available: bool
    note: str | None = None
    drivers: list[DriverStrategy] = []
    degradation: list[CompoundDegradation] = []
    pit_battles: list[PitBattle] = []
    safety_car_stops: list[SafetyCarStop] = []


class Debrief(_Model):
    race: RaceMeta
    generated_at: str
    headlines: list[str]
    score: PredictionScore
    results: list[DriverResult]
    win_probability: list[WinProbabilitySeries] = []
    strategy: StrategySection

    @property
    def slug(self) -> str:
        return f"{self.race.season}-{self.race.round:02d}"
