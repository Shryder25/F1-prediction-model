"""Turn race snapshots into predictions, with the models saved by `f1pred evaluate`."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from f1pred.live.events import DriverState, Entry, LapSnapshot, PitWindow, RaceInfo, RaceUpdate
from f1pred.live.features import LapFeatureBuilder
from f1pred.model.estimators import RaceModel

# Under a safety car the field slows down, so a stop costs roughly half as much.
NEUTRALISED_PIT_FACTOR = {"SC": 0.5, "VSC": 0.6, "RED": 0.0}


@dataclass(frozen=True)
class Models:
    pre_race: RaceModel
    lap: RaceModel

    @classmethod
    def load(cls, model_dir: Path) -> Models:
        """Load `pre_race.joblib` and `lap.joblib`, written by `f1pred evaluate`.

        joblib files are pickles: only load models you trained yourself.
        """
        paths = {task: model_dir / f"{task}.joblib" for task in ("pre_race", "lap")}
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"No saved models at {', '.join(missing)}. Run `f1pred evaluate` first."
            )
        return cls(pre_race=joblib.load(paths["pre_race"]), lap=joblib.load(paths["lap"]))


class RacePredictor:
    """Prediction state for one race. Create one per race (or per replay)."""

    def __init__(self, models: Models, info: RaceInfo) -> None:
        self.models = models
        self.info = info
        self._entries = {e.driver_id: e for e in info.entries}
        self._features = LapFeatureBuilder(info)

    def before_start(self, pre_race_features: pd.DataFrame | None) -> RaceUpdate:
        """Lap 0: the pre-race model's view, or the grid order if it has no inputs."""
        entries = sorted(self.info.entries, key=lambda e: e.grid_position)
        if pre_race_features is None or pre_race_features.empty:
            n = len(entries)
            drivers = tuple(
                self._state(e, e.grid_position, None, None, None, None, 0, 1 / n, 3 / n, i + 1)
                for i, e in enumerate(entries)
            )
        else:
            frame = pre_race_features.reset_index(drop=True)
            pred = self.models.pre_race.predict(frame).reset_index(drop=True)
            joined = pd.concat([frame[["driver_id"]], pred], axis=1).set_index("driver_id")
            drivers = tuple(
                self._state(
                    e,
                    e.grid_position,
                    None,
                    None,
                    None,
                    None,
                    0,
                    _get(joined, e.driver_id, "p_win", 0.0),
                    _get(joined, e.driver_id, "p_podium", 0.0),
                    int(_get(joined, e.driver_id, "pred_rank", e.grid_position)),
                )
                for e in entries
            )
        return self._update(0, "GREEN", drivers)

    def after_lap(self, snapshot: LapSnapshot) -> RaceUpdate:
        features = self._features.update(snapshot)
        status = snapshot.track_status
        if features.empty:
            return self._update(snapshot.lap, status, ())
        pred = self.models.lap.predict(features).reset_index(drop=True)
        frame = pd.concat([features.reset_index(drop=True), pred], axis=1)
        loss = self.info.pit_loss_s * NEUTRALISED_PIT_FACTOR.get(status, 1.0)
        windows = pit_windows(frame, loss)
        drivers = tuple(
            self._state(
                self._entries[r["driver_id"]],
                int(r["position"]),
                _opt(r["gap_to_leader_s"]),
                _opt(r["gap_to_car_ahead_s"]),
                r["compound"] if isinstance(r["compound"], str) else None,
                None if _opt(r["tyre_age_laps"]) is None else int(r["tyre_age_laps"]),
                int(r["pit_stops_made"]),
                float(r["p_win"]),
                float(r["p_podium"]),
                int(r["pred_rank"]),
                windows.get(r["driver_id"]),
            )
            for r in frame.sort_values("position").to_dict(orient="records")
        )
        return self._update(snapshot.lap, status, drivers)

    def _state(
        self,
        entry: Entry,
        position: int,
        gap: float | None,
        gap_ahead: float | None,
        compound: str | None,
        tyre_age: int | None,
        stops: int,
        p_win: float,
        p_podium: float,
        predicted: int,
        pit: PitWindow | None = None,
    ) -> DriverState:
        return DriverState(
            driver_id=entry.driver_id,
            code=entry.code,
            team_name=entry.team_name,
            team_colour=entry.team_colour,
            position=position,
            grid_position=entry.grid_position,
            gap_to_leader_s=_round(gap),
            gap_to_car_ahead_s=_round(gap_ahead),
            compound=compound,
            tyre_age_laps=tyre_age,
            pit_stops=stops,
            p_win=_prob(p_win),
            p_podium=_prob(p_podium),
            predicted_position=predicted,
            pit=pit,
        )

    def _update(self, lap: int, status: str, drivers: tuple[DriverState, ...]) -> RaceUpdate:
        return RaceUpdate(
            race_id=self.info.race_id,
            race_name=self.info.race_name,
            season=self.info.season,
            lap=lap,
            total_laps=self.info.total_laps,
            track_status=status,
            drivers=drivers,
            is_final=lap >= self.info.total_laps,
        )


def pit_windows(frame: pd.DataFrame, pit_loss_s: float) -> dict[str, PitWindow]:
    """For each car: where it would rejoin if it pitted at the end of this lap.

    A car rejoins behind every car whose gap to the leader is less than its own
    gap plus the pit loss. The cars it drops behind are the places lost.
    """
    rows = frame[["driver_id", "position", "gap_to_leader_s", "gap_to_car_behind_s"]]
    known = rows.dropna(subset=["gap_to_leader_s"]).sort_values("position")
    gaps = known["gap_to_leader_s"].to_numpy(dtype=float)
    out: dict[str, PitWindow] = {}
    for i, r in enumerate(known.to_dict(orient="records")):
        behind = gaps[i + 1 :]
        lost = int((behind < r["gap_to_leader_s"] + pit_loss_s).sum())
        margin = None
        if not _missing(r["gap_to_car_behind_s"]):
            margin = round(float(r["gap_to_car_behind_s"]) - pit_loss_s, 1)
        out[str(r["driver_id"])] = PitWindow(
            rejoin_position=int(r["position"]) + lost, places_lost=lost, margin_s=margin
        )
    return out


def _get(frame: pd.DataFrame, driver_id: str, column: str, default: float) -> float:
    if driver_id not in frame.index:
        return default
    value: Any = frame.at[driver_id, column]
    return default if _missing(value) else float(value)


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _opt(value: Any) -> float | None:
    return None if _missing(value) else float(value)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


def _prob(value: float) -> float:
    return round(min(max(value, 0.0), 1.0), 4)
