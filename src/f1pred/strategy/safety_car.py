"""How much a pit stop costs, and how much a safety car or VSC makes it cheaper.

The pit loss of a stop is the time the driver lost over the in-lap and out-lap,
compared with the cars that stayed out on those same laps:

    pit_loss = (in-lap + out-lap) - (median in-lap + median out-lap of cars not pitting)

Comparing against the field on the same laps removes what affects everybody: the
safety car itself, fuel load, track evolution. Under a safety car the field is slow,
so the time spent in the pit lane costs less relative to the cars staying out. The
saving is the race's median green-flag pit loss minus the neutralised stop's loss.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from f1pred.strategy.battles import stop_laps

SC_STOP_COLUMNS = [
    "race_id",
    "driver_id",
    "lap",  # lap of pit entry
    "track_status",  # 'SC' or 'VSC'
    "pit_loss_s",
    "green_pit_loss_s",  # median pit loss of the race's green-flag stops
    "time_saved_s",  # green_pit_loss_s - pit_loss_s
    # Position at the end of the lap before the stop minus position at the end of
    # the out-lap. Negative means places were lost in the stop.
    "positions_gained",
]

PIT_LOSS_COLUMNS = ["race_id", "driver_id", "lap", "track_status", "pit_loss_s"]


def _flag(laps: pd.DataFrame, column: str) -> pd.Series:
    if column not in laps:
        return pd.Series(False, index=laps.index)
    return laps[column].fillna(False).astype(bool)


def pit_losses(laps: pd.DataFrame, pit_stops: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pit loss and track status of every stop (see PIT_LOSS_COLUMNS).

    track_status is 'green', 'SC', 'VSC' or 'red' (a red flag stop is a free tyre
    change and has no meaningful loss). It describes the in-lap: a car entering the
    pits while the safety car is out.
    """
    stops = stop_laps(laps, pit_stops)
    if stops.empty:
        return pd.DataFrame(columns=PIT_LOSS_COLUMNS)
    laps = laps.assign(
        _sc=_flag(laps, "is_safety_car"),
        _vsc=_flag(laps, "is_virtual_safety_car"),
        _red=_flag(laps, "is_red_flag"),
    )
    status_by_lap = laps.groupby(["race_id", "lap"])[["_sc", "_vsc", "_red"]].any()
    stopping = set(zip(stops["race_id"], stops["driver_id"], stops["lap"], strict=True))
    out_laps = {(r, d, lap + 1) for r, d, lap in stopping}
    in_or_out = [
        (r, d, lap) in stopping or (r, d, lap) in out_laps
        for r, d, lap in zip(laps["race_id"], laps["driver_id"], laps["lap"], strict=True)
    ]
    reference = (
        laps.loc[~pd.Series(in_or_out, index=laps.index) & (laps["lap"] > 1)]
        .groupby(["race_id", "lap"])["lap_time_s"]
        .median()
    )
    lap_time = laps.set_index(["race_id", "driver_id", "lap"])["lap_time_s"]
    lap_time = lap_time[~lap_time.index.duplicated()]

    rows: list[dict[str, Any]] = []
    for race_id, driver_id, lap in stopping:
        keys = [(race_id, driver_id, lap), (race_id, driver_id, lap + 1)]
        refs = [(race_id, lap), (race_id, lap + 1)]
        if not all(k in lap_time.index for k in keys) or not all(
            r in reference.index for r in refs
        ):
            continue  # retired in the pits, or the last lap
        loss = sum(float(lap_time[k]) for k in keys) - sum(float(reference[r]) for r in refs)
        if pd.isna(loss):
            continue
        sc, vsc, red = status_by_lap.loc[(race_id, lap)]
        status = "red" if red else "SC" if sc else "VSC" if vsc else "green"
        rows.append(
            {
                "race_id": race_id,
                "driver_id": driver_id,
                "lap": lap,
                "track_status": status,
                "pit_loss_s": round(loss, 3),
            }
        )
    if not rows:
        return pd.DataFrame(columns=PIT_LOSS_COLUMNS)
    return pd.DataFrame(rows, columns=PIT_LOSS_COLUMNS).sort_values(
        ["race_id", "lap", "driver_id"], ignore_index=True
    )


def green_pit_loss(losses: pd.DataFrame) -> pd.Series:
    """Median green-flag pit loss per race_id."""
    green = losses.loc[losses["track_status"] == "green"]
    return green.groupby("race_id")["pit_loss_s"].median()


def safety_car_stops(
    laps: pd.DataFrame,
    pit_stops: pd.DataFrame | None = None,
    losses: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Stops made under a safety car or VSC, with the time they saved (see SC_STOP_COLUMNS)."""
    if losses is None:
        losses = pit_losses(laps, pit_stops)
    neutral = losses.loc[losses["track_status"].isin(["SC", "VSC"])].copy()
    if neutral.empty:
        return pd.DataFrame(columns=SC_STOP_COLUMNS)
    neutral["green_pit_loss_s"] = neutral["race_id"].map(green_pit_loss(losses)).astype(float)
    neutral["time_saved_s"] = (neutral["green_pit_loss_s"] - neutral["pit_loss_s"]).round(3)

    position = laps.set_index(["race_id", "driver_id", "lap"])["position"]
    position = position[~position.index.duplicated()]

    def gained(row: pd.Series) -> float:
        before = (row["race_id"], row["driver_id"], row["lap"] - 1)
        after = (row["race_id"], row["driver_id"], row["lap"] + 1)
        if before not in position.index or after not in position.index:
            return float("nan")
        return float(position[before] - position[after])

    neutral["positions_gained"] = neutral.apply(gained, axis=1).astype("Int64")
    return neutral[SC_STOP_COLUMNS].reset_index(drop=True)
