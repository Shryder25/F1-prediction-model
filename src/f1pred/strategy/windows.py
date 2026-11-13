"""Pit window model: given the tyres a driver chose, when should each stop have been?

Each stint's lap time is modelled as a straight line in tyre age, fitted on its
clean fuel-corrected laps (see `stints.py`):

    lap_time(age) = fresh_tyre_pace + degradation * age

For one stop, between stint k and stint k+1, we keep everything else as raced (the
compounds, the other stops, the number of stops) and slide the stop lap. The modelled
time over the two stints is

    T(s) = sum of stint k laps from its first lap to s  +  sum of stint k+1 laps from s+1

The best lap minimises T. The window is every lap within `window_tolerance_s` of
that minimum. With linear wear T is a parabola in s, so the window is one range of
laps and is wider when the tyres degrade slowly.

The pit loss and fuel effect are the same whichever lap the stop is on, so they drop
out. What the model does not know: traffic on the out-lap, undercut threats, safety
cars, and tyre life cliffs. Those are why real stops leave the window, and the
debrief should say so rather than call every early stop a mistake.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from f1pred.strategy.config import DEFAULT_CONFIG, StrategyConfig

WINDOW_COLUMNS = [
    "race_id",
    "driver_id",
    "stint",  # the stint that ended with this stop
    "compound",  # tyre of that stint
    "actual_pit_lap",
    "optimal_pit_lap",
    "window_open",
    "window_close",
    "verdict",  # 'early', 'in window' or 'late'
    "time_lost_s",  # modelled seconds lost by stopping on the actual lap, not the optimal
]


def _tyre_model(
    stints: pd.DataFrame,
) -> tuple[dict[tuple[Any, str, int], tuple[float, float]], dict[tuple[Any, str], float]]:
    """(fresh pace, degradation) per usable stint, and race median degradation per compound."""
    usable = stints.loc[
        stints["is_usable"].astype(bool)
        & stints["degradation_s_per_lap"].notna()
        & stints["fresh_tyre_pace_s"].notna()
    ]
    own = {
        (r, str(d), int(s)): (float(b), max(float(k), 0.0))
        for r, d, s, b, k in zip(
            usable["race_id"],
            usable["driver_id"],
            usable["stint"],
            usable["fresh_tyre_pace_s"],
            usable["degradation_s_per_lap"],
            strict=True,
        )
    }
    medians = usable.groupby(["race_id", "compound"])["degradation_s_per_lap"].median()
    race_deg: dict[tuple[Any, str], float] = {}
    for key, value in medians.items():
        race: Any = key[0]  # type: ignore[index]
        compound: Any = key[1]  # type: ignore[index]
        race_deg[(race, str(compound))] = max(float(value), 0.0)
    return own, race_deg


def _race_compound_offset(stints: pd.DataFrame, race_id: Any, a: str, b: str) -> float:
    """Median fresh-tyre pace of compound b minus compound a in this race (0 if unknown)."""
    if a == b:
        return 0.0
    usable = stints.loc[(stints["race_id"] == race_id) & stints["is_usable"].astype(bool)]
    pace = usable.groupby("compound")["fresh_tyre_pace_s"].median()
    if a in pace.index and b in pace.index:
        return float(pace[b] - pace[a])
    return 0.0


def pit_windows(
    laps: pd.DataFrame, stints: pd.DataFrame, config: StrategyConfig = DEFAULT_CONFIG
) -> pd.DataFrame:
    """One row per pit stop that changed tyres between two dry stints (see WINDOW_COLUMNS)."""
    if stints.empty:
        return pd.DataFrame(columns=WINDOW_COLUMNS)
    own, race_deg = _tyre_model(stints)
    first_age = (
        laps.loc[laps["stint"].notna()]
        .groupby(["race_id", "driver_id", "stint"])["tyre_age_laps"]
        .min()
    )
    race_laps = laps.groupby("race_id")["total_laps"].max()
    rows: list[dict[str, Any]] = []
    ordered = stints.sort_values(["race_id", "driver_id", "stint"])
    for (race_id, driver_id), driver_stints in ordered.groupby(["race_id", "driver_id"]):
        records: list[dict[str, Any]] = [
            {str(k): v for k, v in r.items()} for r in driver_stints.to_dict("records")
        ]
        # A driver who retired would have raced to the flag: the last stint runs to
        # the scheduled distance, so a retirement does not drag the window earlier.
        records[-1]["end_lap"] = max(int(records[-1]["end_lap"]), int(race_laps.get(race_id, 0)))
        for cur, nxt in itertools.pairwise(records):
            row = _window(
                race_id, str(driver_id), cur, nxt, own, race_deg, first_age, stints, config
            )
            if row is not None:
                rows.append(row)
    if not rows:
        return pd.DataFrame(columns=WINDOW_COLUMNS)
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)


def _window(
    race_id: Any,
    driver_id: str,
    cur: dict[str, Any],
    nxt: dict[str, Any],
    own: dict[tuple[Any, str, int], tuple[float, float]],
    race_deg: dict[tuple[Any, str], float],
    first_age: pd.Series,
    stints: pd.DataFrame,
    config: StrategyConfig,
) -> dict[str, Any] | None:
    compounds = (str(cur["compound"]), str(nxt["compound"]))
    if not all(c in ("SOFT", "MEDIUM", "HARD") for c in compounds):
        return None  # wet or unknown tyres: the dry wear model does not apply
    if int(nxt["start_lap"]) != int(cur["end_lap"]) + 1:
        return None  # a gap in the data (red flag, missing laps)

    degs: list[float] = []
    for stint, compound in ((cur, compounds[0]), (nxt, compounds[1])):
        key = (race_id, driver_id, int(stint["stint"]))
        if key in own:
            degs.append(own[key][1])
        elif (race_id, compound) in race_deg:
            degs.append(race_deg[(race_id, compound)])
        else:
            return None
    cur_key = (race_id, driver_id, int(cur["stint"]))
    nxt_key = (race_id, driver_id, int(nxt["stint"]))
    if cur_key in own and nxt_key in own:
        offset = own[nxt_key][0] - own[cur_key][0]  # same car, so pace is comparable
    else:
        offset = _race_compound_offset(stints, race_id, *compounds)

    def age0(stint: dict[str, Any]) -> float:
        value = first_age.get((race_id, driver_id, stint["stint"]), 1)
        return 1.0 if pd.isna(value) else float(value)

    start, end = int(cur["start_lap"]), int(nxt["end_lap"])
    actual = int(cur["end_lap"])
    candidates = np.arange(start, end)  # every stint keeps at least one lap
    a1, a2 = age0(cur), age0(nxt)
    n1 = candidates - start + 1  # laps on the first tyre
    n2 = end - candidates  # laps on the second tyre
    # Sum of an arithmetic series: n laps starting at tyre age a0.
    wear1 = degs[0] * (n1 * a1 + n1 * (n1 - 1) / 2)
    wear2 = degs[1] * (n2 * a2 + n2 * (n2 - 1) / 2)
    total = wear1 + wear2 + offset * n2
    best = float(np.min(total))
    inside = candidates[total - best <= config.window_tolerance_s]
    optimal = int(candidates[int(np.argmin(total))])
    window_open, window_close = int(np.min(inside)), int(np.max(inside))
    verdict = "early" if actual < window_open else "late" if actual > window_close else "in window"
    return {
        "race_id": race_id,
        "driver_id": driver_id,
        "stint": int(cur["stint"]),
        "compound": compounds[0],
        "actual_pit_lap": actual,
        "optimal_pit_lap": optimal,
        "window_open": window_open,
        "window_close": window_close,
        "verdict": verdict,
        "time_lost_s": round(float(total[actual - start]) - best, 3),
    }
