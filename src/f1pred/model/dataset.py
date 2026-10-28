"""Load model inputs from the dbt marts and split them by season.

The feature SQL lives in transform/models/marts/ml/. This module only reads it,
turns categories into numbers, and names which columns are features and which
are labels, so no label can slip into a feature list by accident.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import duckdb
import pandas as pd

PRE_RACE_FEATURES: tuple[str, ...] = (
    "grid_position",
    "started_from_pit_lane",
    "qualifying_position",
    "gap_to_pole_pct",
    "gap_to_team_mate_pct",
    "driver_finish_last5",
    "driver_points_last5",
    "driver_gained_last5",
    "driver_dnf_rate_last10",
    "driver_career_starts",
    "driver_track_gained",
    "driver_track_starts",
    "constructor_points_last5",
    "constructor_finish_last5",
    "championship_position_before",
    "championship_points_before",
    "starters",
)

COMPOUNDS: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET")

LAP_FEATURES: tuple[str, ...] = (
    "race_progress",
    "laps_remaining",
    "position",
    "cars_running",
    "grid_position",
    "positions_gained_so_far",
    "gap_to_leader_s",
    "gap_to_car_ahead_s",
    "gap_to_car_behind_s",
    "recent_pace_vs_field_s",
    "tyre_age_laps",
    "stint",
    "pit_stops_made",
    "is_safety_car",
    "is_virtual_safety_car",
    *(f"compound_{c.lower()}" for c in COMPOUNDS),
    # Pre-race strength, so the lap model knows a fast car stuck in traffic.
    "gap_to_pole_pct",
    "driver_finish_last5",
    "constructor_points_last5",
)

LABELS: tuple[str, ...] = (
    "label_finishing_order",
    "label_win",
    "label_podium",
    "label_classified",
)

# Rows of both tables carry these, so predictions can be joined back and grouped.
KEYS: tuple[str, ...] = ("race_id", "season", "round", "driver_id")


def load_pre_race(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("SELECT * FROM marts.ml_pre_race_features ORDER BY race_id, driver_id").df()
    return _numeric(df, PRE_RACE_FEATURES)


def load_laps(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT l.*, p.gap_to_pole_pct, p.driver_finish_last5, p.constructor_points_last5
        FROM marts.ml_lap_features AS l
        LEFT JOIN marts.ml_pre_race_features AS p USING (race_id, driver_id)
        ORDER BY l.race_id, l.lap, l.position
        """
    ).df()
    for compound in COMPOUNDS:
        df[f"compound_{compound.lower()}"] = (df["compound"] == compound).astype(float)
    # With no tyre data for a lap we do not know the compound: NaN, not "not soft".
    no_tyre = df["compound"].isna()
    df.loc[no_tyre, [f"compound_{c.lower()}" for c in COMPOUNDS]] = float("nan")
    return _numeric(df, LAP_FEATURES)


def _numeric(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    """Cast features to float (NaN for missing) and labels to plain ints."""
    missing = [c for c in (*KEYS, *features, *LABELS) if c not in df.columns]
    if missing:
        raise KeyError(f"Feature table is missing columns {missing}. Re-run `f1pred transform`.")
    out = df.copy()
    for column in features:
        out[column] = pd.to_numeric(out[column].astype("float64"), errors="coerce")
    for column in LABELS:
        out[column] = out[column].fillna(False).astype(int)
    return out


@dataclass(frozen=True)
class SeasonSplit:
    """Train on older seasons, pick settings on `validation`, report on `test`.

    `holdout` seasons are scored separately: in the real data that is 2026, the
    first season under new car regulations, where history should matter less.
    """

    train: tuple[int, ...]
    validation: int
    test: int
    holdout: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.train:
            raise ValueError("The split needs at least one training season.")
        if max(self.train) >= self.validation or self.validation >= self.test:
            raise ValueError(f"Seasons must run train < validation < test, got {self}.")
        if any(season <= self.test for season in self.holdout):
            raise ValueError(f"Holdout seasons must come after the test season, got {self}.")

    @classmethod
    def latest(cls, seasons: Sequence[int], holdout: Sequence[int] = (2026,)) -> SeasonSplit:
        """The plan's split, from whatever seasons exist.

        With 2018 to 2026 loaded: train 2018-2023, validate 2024, test 2025, holdout 2026.
        """
        held = tuple(s for s in sorted(set(holdout)) if s in seasons)
        rest = sorted(s for s in set(seasons) if s not in held)
        if len(rest) < 3:
            raise ValueError(
                f"Need at least 3 seasons besides the holdout to train, validate and test; "
                f"found {rest}. Load more seasons (or `f1pred demo --seasons 2019-2026`)."
            )
        return cls(train=tuple(rest[:-2]), validation=rest[-2], test=rest[-1], holdout=held)

    def describe(self) -> str:
        train = f"{self.train[0]}-{self.train[-1]}" if len(self.train) > 1 else str(self.train[0])
        text = f"train {train}, validate {self.validation}, test {self.test}"
        if self.holdout:
            text += f", holdout {', '.join(map(str, self.holdout))}"
        return text


def seasons_in(df: pd.DataFrame, seasons: Sequence[int]) -> pd.DataFrame:
    return df[df["season"].isin(list(seasons))].reset_index(drop=True)
