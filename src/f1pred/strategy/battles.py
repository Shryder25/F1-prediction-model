"""Undercut and overcut attempts: two cars close on track whose pit stops are a few laps apart.

For every pit stop we look at the cars directly ahead and behind on the lap before.
If one of them is close enough (see `battle_max_gap_s`) and pits a few laps later, the
pair is a pit battle:

- **undercut**: the car behind (the attacker) pits first and tries to use the faster
  new tyre to jump the car ahead while it is still on old tyres.
- **overcut**: the car ahead pits first and the attacker behind stays out, hoping
  clear air on old tyres beats the defender's out-lap and warm-up.

Gaps are race time at the end of a lap (`elapsed_s`), so they are measured where
both cars cross the line, not with GPS. The result is judged at the end of the later
car's out-lap, when both have completed their stops. Cycles that overlap a safety car,
VSC or red flag are left out: a neutralised stop is a different decision (see
`safety_car.py`).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from f1pred.strategy.config import DEFAULT_CONFIG, StrategyConfig

BATTLE_COLUMNS = [
    "race_id",
    "attacker_id",  # the car behind before the stops, trying to get ahead
    "defender_id",
    "kind",  # 'undercut' (attacker pitted first) or 'overcut' (defender pitted first)
    "attacker_pit_lap",
    "defender_pit_lap",
    "gap_before_s",  # attacker's deficit before the first stop; positive = behind
    "gap_after_s",  # after both stops; positive = attacker now ahead
    "gain_s",  # gap_before_s + gap_after_s: time the attacker made up over the cycle
    "succeeded",
]


def stop_laps(laps: pd.DataFrame, pit_stops: pd.DataFrame | None) -> pd.DataFrame:
    """(race_id, driver_id, lap) of every pit entry.

    Jolpica pit stops cover every season. FastF1 pit-in flags are the fallback when no
    pit stop table is given.
    """
    if pit_stops is not None and not pit_stops.empty:
        stops = pit_stops[["race_id", "driver_id", "lap"]]
    else:
        stops = laps.loc[laps["is_pit_in_lap"].fillna(False).astype(bool)]
        stops = stops[["race_id", "driver_id", "lap"]]
    return stops.drop_duplicates().sort_values(["race_id", "driver_id", "lap"], ignore_index=True)


def neutralised_laps(laps: pd.DataFrame) -> set[int]:
    """Laps of one race where any car reported a safety car, VSC or red flag."""
    flags = [c for c in ("is_safety_car", "is_virtual_safety_car", "is_red_flag") if c in laps]
    if not flags:
        return set()
    hit = laps[flags].fillna(False).astype(bool).any(axis=1)
    return {int(lap) for lap in laps.loc[hit, "lap"].unique()}


def pit_battles(
    laps: pd.DataFrame,
    pit_stops: pd.DataFrame | None = None,
    config: StrategyConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Every undercut and overcut attempt (see BATTLE_COLUMNS), across all races in `laps`."""
    rows: list[dict[str, Any]] = []
    stops = stop_laps(laps, pit_stops)
    for race_id in laps["race_id"].unique():
        race_laps = laps.loc[laps["race_id"] == race_id]
        race_stops = stops.loc[stops["race_id"] == race_id]
        rows.extend(_race_battles(int(race_id), race_laps, race_stops, config))
    if not rows:
        return pd.DataFrame(columns=BATTLE_COLUMNS)
    out = pd.DataFrame(rows, columns=BATTLE_COLUMNS)
    out = out.drop_duplicates(
        subset=["race_id", "attacker_id", "defender_id", "attacker_pit_lap", "defender_pit_lap"]
    )
    return out.sort_values(by=["race_id", "attacker_pit_lap", "attacker_id"], ignore_index=True)


def _race_battles(
    race_id: int, laps: pd.DataFrame, stops: pd.DataFrame, config: StrategyConfig
) -> list[dict[str, Any]]:
    elapsed = laps.pivot_table(
        index="lap", columns="driver_id", values="elapsed_s", aggfunc="first"
    )
    neutral = neutralised_laps(laps)
    by_driver: dict[str, list[int]] = {
        str(d): sorted(int(x) for x in g["lap"]) for d, g in stops.groupby("driver_id")
    }

    def next_stop(driver: str, after: int) -> int | None:
        later = [lap for lap in by_driver.get(driver, []) if lap > after]
        return later[0] if later else None

    def at(driver: str, lap: int) -> float | None:
        if lap not in elapsed.index or driver not in elapsed.columns:
            return None
        value: Any = elapsed.at[lap, driver]
        return None if pd.isna(value) else float(value)

    rows: list[dict[str, Any]] = []
    for driver, driver_stops in by_driver.items():
        for first in driver_stops:
            before = first - 1
            if before < 1 or before not in elapsed.index:
                continue
            order: pd.Series = pd.Series(elapsed.loc[before]).dropna().sort_values()
            if driver not in order.index:
                continue
            i = order.index.get_loc(driver)
            assert isinstance(i, int)
            neighbours = []
            if i > 0:  # car ahead: our stop is an undercut attempt on it
                neighbours.append(("undercut", driver, str(order.index[i - 1])))
            if i + 1 < len(order):  # car behind: it may try to overcut us
                neighbours.append(("overcut", str(order.index[i + 1]), driver))
            for kind, attacker, defender in neighbours:
                other = defender if kind == "undercut" else attacker
                if first in by_driver.get(other, []):
                    continue  # both pitted on the same lap: no undercut either way
                second = next_stop(other, first)
                if second is None or second - first > config.battle_max_lap_gap:
                    continue
                own_next = next_stop(driver, first)
                if own_next is not None and own_next <= second:
                    continue  # the first car stopped again before the second car's stop
                gap_before = float(order[attacker] - order[defender])
                if gap_before > config.battle_max_gap_s:
                    continue
                judged = second + 1  # end of the second car's out-lap
                if neutral & set(range(before, judged + 1)):
                    continue
                attacker_after, defender_after = at(attacker, judged), at(defender, judged)
                if attacker_after is None or defender_after is None:
                    continue  # one of them retired during the cycle
                gap_after = defender_after - attacker_after
                attacker_lap, defender_lap = (
                    (first, second) if kind == "undercut" else (second, first)
                )
                rows.append(
                    {
                        "race_id": race_id,
                        "attacker_id": attacker,
                        "defender_id": defender,
                        "kind": kind,
                        "attacker_pit_lap": attacker_lap,
                        "defender_pit_lap": defender_lap,
                        "gap_before_s": round(gap_before, 3),
                        "gap_after_s": round(gap_after, 3),
                        "gain_s": round(gap_before + gap_after, 3),
                        "succeeded": gap_after > 0,
                    }
                )
    return rows
