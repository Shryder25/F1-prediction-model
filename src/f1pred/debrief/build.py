"""Gather everything the debrief says about one race into a `Debrief`.

Inputs, all read only:
- the dbt marts (result, grid, team, race facts),
- the models saved by `f1pred evaluate` (pre-race prediction, and a lap-by-lap
  replay for the win probability chart). Without them the debrief falls back to
  grid order and says so,
- `f1pred.strategy.race_strategy` for stints, degradation, undercuts, safety car
  stops and pit windows. The debrief only reports those numbers; it does not
  compute strategy metrics itself.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from f1pred.debrief.model import (
    CompoundDegradation,
    Debrief,
    DriverResult,
    DriverStrategy,
    PitBattle,
    PredictionScore,
    RaceMeta,
    SafetyCarStop,
    Stint,
    StrategySection,
    WinProbabilitySeries,
)
from f1pred.demo import DEMO_MARKER_TABLE
from f1pred.strategy import race_strategy

log = logging.getLogger(__name__)

# Drivers shown on the win probability chart: everyone who was ever this likely to win.
CHART_MIN_P_WIN = 0.10
CHART_MAX_DRIVERS = 5


def latest_race_id(warehouse_path: Path) -> int:
    """The most recent race with a result in the warehouse."""
    with duckdb.connect(str(warehouse_path), read_only=True) as con:
        row = con.execute(
            "SELECT max(race_id) FROM marts.fct_race_results WHERE classified_position IS NOT NULL"
        ).fetchone()
    if row is None or row[0] is None:
        raise KeyError("No race results in the warehouse yet. Run the pipeline first.")
    return int(row[0])


def build_debrief(
    warehouse_path: Path,
    race_id: int,
    model_dir: Path | None = None,
    now: dt.datetime | None = None,
) -> Debrief:
    with duckdb.connect(str(warehouse_path), read_only=True) as con:
        race = _race_meta(con, race_id)
        entries = _entries(con, race_id)
        pre_race, source, in_sample, trace = _predictions(con, warehouse_path, race, model_dir)
        strategy = _strategy(con, race_id, entries)

    results = _results(entries, pre_race)
    score = _score(results, source, in_sample, trace)
    series = _series(trace, results)
    debrief = Debrief(
        race=race,
        generated_at=(now or dt.datetime.now(dt.UTC)).strftime("%Y-%m-%d %H:%M UTC"),
        headlines=[],
        score=score,
        results=results,
        win_probability=series,
        strategy=strategy,
    )
    return debrief.model_copy(update={"headlines": headlines(debrief)})


# --- race facts -------------------------------------------------------------------------


def _race_meta(con: duckdb.DuckDBPyConnection, race_id: int) -> RaceMeta:
    row = con.execute(
        "SELECT season, round, race_name, circuit_name, race_date, total_laps "
        "FROM marts.dim_race WHERE race_id = ?",
        [race_id],
    ).fetchone()
    if row is None:
        raise KeyError(f"Race {race_id} is not in the warehouse.")
    season, round_, name, circuit, date, laps = row
    demo = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
        [DEMO_MARKER_TABLE],
    ).fetchone()
    return RaceMeta(
        race_id=race_id,
        season=int(season),
        round=int(round_),
        race_name=str(name),
        circuit_name=None if circuit is None else str(circuit),
        race_date=None if date is None else str(date),
        total_laps=int(laps or 0),
        is_demo=bool(demo and demo[0]),
    )


def _entries(con: duckdb.DuckDBPyConnection, race_id: int) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT e.driver_id,
               coalesce(d.driver_code, upper(left(e.driver_id, 3))) AS code,
               coalesce(d.full_name, e.driver_id) AS name,
               c.constructor_name AS team,
               coalesce(c.team_colour, '#888888') AS team_colour,
               e.grid_position AS grid,
               e.classified_position AS finish,
               e.status
        FROM marts.fct_race_results AS e
        LEFT JOIN marts.dim_driver AS d USING (driver_id)
        LEFT JOIN marts.dim_constructor AS c USING (constructor_id)
        WHERE e.race_id = ?
        ORDER BY e.grid_position
        """,
        [race_id],
    ).df()
    if df.empty:
        raise KeyError(f"Race {race_id} has no results yet.")
    return df


# --- predictions ------------------------------------------------------------------------


def _trained_seasons(model_dir: Path) -> set[int]:
    """Seasons the saved models were fitted on, from the metrics.json next to them."""
    path = model_dir.parent / "metrics.json"
    try:
        split = json.loads(path.read_text())["split"]
    except (OSError, KeyError, ValueError):
        return set()
    return {*split["train"], split["validation"], split["test"]}


def _predictions(
    con: duckdb.DuckDBPyConnection,
    warehouse_path: Path,
    race: RaceMeta,
    model_dir: Path | None,
) -> tuple[pd.DataFrame, str, bool, list[dict[str, float]]]:
    """Pre-race prediction per driver, its source, and the lap-by-lap win probabilities.

    Returns (frame with driver_id, predicted_rank, p_win, p_podium; source; in_sample;
    trace: one {driver_id: p_win} dict per lap from lap 0).
    """
    models = None
    if model_dir is not None:
        try:
            from f1pred.live.predictor import Models

            models = Models.load(model_dir)
        except (FileNotFoundError, ImportError) as error:
            log.warning("Debrief without models: %s", error)
    if models is None:
        return pd.DataFrame(), "grid order (no saved models)", False, []

    from f1pred.live.pipeline import run_race
    from f1pred.live.replay import WarehouseReplay

    async def replay() -> list[dict[str, Any]]:
        source = WarehouseReplay(warehouse_path, race.race_id, lap_interval_s=0)
        return [
            {d.driver_id: (d.p_win, d.p_podium, d.predicted_position) for d in update.drivers}
            async for update in run_race(source, models)
        ]

    try:
        updates = asyncio.run(replay())
    except KeyError as error:  # no pre-race features for this race
        log.warning("Debrief without models: %s", error)
        return pd.DataFrame(), "grid order (no pre-race features)", False, []
    pre = pd.DataFrame(
        [
            {"driver_id": k, "p_win": v[0], "p_podium": v[1], "predicted_rank": v[2]}
            for k, v in updates[0].items()
        ]
    )
    trace = [{k: v[0] for k, v in update.items()} for update in updates]
    in_sample = race.season in _trained_seasons(model_dir) if model_dir else False
    return pre, "saved models", in_sample, trace


def _results(entries: pd.DataFrame, pre_race: pd.DataFrame) -> list[DriverResult]:
    df = entries.copy()
    if pre_race.empty:
        df["predicted_rank"] = df["grid"]
        df["p_win"] = None
        df["p_podium"] = None
    else:
        df = df.merge(pre_race, on="driver_id", how="left")
        df["predicted_rank"] = df["predicted_rank"].fillna(df["grid"])
    out = []
    for r in df.sort_values(["predicted_rank", "grid"]).to_dict(orient="records"):
        finish = _int(r["finish"])
        predicted = int(r["predicted_rank"])
        out.append(
            DriverResult(
                driver_id=str(r["driver_id"]),
                code=str(r["code"]),
                name=str(r["name"]),
                team=_text(r["team"]),
                team_colour=str(r["team_colour"]),
                grid=int(r["grid"]),
                predicted_rank=predicted,
                p_win=_float(r["p_win"]),
                p_podium=_float(r["p_podium"]),
                finish=finish,
                status=str(r["status"]),
                beat_prediction_by=None if finish is None else predicted - finish,
            )
        )
    return out


def _score(
    results: Sequence[DriverResult],
    source: str,
    in_sample: bool,
    trace: list[dict[str, float]],
) -> PredictionScore:
    classified = [r for r in results if r.finish is not None]
    winner = next((r for r in results if r.finish == 1), None)
    favourite = min(results, key=lambda r: r.predicted_rank)
    predicted_top3 = {r.driver_id for r in results if r.predicted_rank <= 3}
    podium = {r.driver_id for r in classified if r.finish is not None and r.finish <= 3}

    def mae(predicted: dict[str, int]) -> float:
        if not classified:
            return float("nan")
        errors = [abs(predicted[r.driver_id] - (r.finish or 0)) for r in classified]
        return round(sum(errors) / len(errors), 2)

    brier = None
    if all(r.p_win is not None for r in results):
        brier = round(
            sum(((r.p_win or 0.0) - (r.finish == 1)) ** 2 for r in results) / len(results), 4
        )
    return PredictionScore(
        source=source,
        in_sample=in_sample,
        winner_predicted=winner is not None and favourite.driver_id == winner.driver_id,
        winner_pre_race_p_win=None if winner is None else winner.p_win,
        podium_hits=len(predicted_top3 & podium),
        position_mae=mae({r.driver_id: r.predicted_rank for r in results}),
        grid_position_mae=mae({r.driver_id: r.grid for r in results}),
        brier_win=brier,
        winner_locked_in_lap=None if winner is None else locked_in_lap(trace, winner.driver_id),
    )


def locked_in_lap(trace: list[dict[str, float]], driver_id: str) -> int | None:
    """First lap from which `driver_id` was the favourite on every remaining lap."""
    lap = None
    for i in range(len(trace) - 1, -1, -1):
        probs = trace[i]
        if not probs or driver_id not in probs or probs[driver_id] < max(probs.values()):
            break
        lap = i
    return lap


def _series(
    trace: list[dict[str, float]], results: Sequence[DriverResult]
) -> list[WinProbabilitySeries]:
    if not trace:
        return []
    peak = {r.driver_id: max(lap.get(r.driver_id, 0.0) for lap in trace) for r in results}
    chosen = sorted(
        (r for r in results if peak[r.driver_id] >= CHART_MIN_P_WIN),
        key=lambda r: -peak[r.driver_id],
    )[:CHART_MAX_DRIVERS]
    return [
        WinProbabilitySeries(
            code=r.code,
            team_colour=r.team_colour,
            p_win=[round(lap.get(r.driver_id, 0.0), 4) for lap in trace],
        )
        for r in chosen
    ]


# --- strategy (numbers from f1pred.strategy) --------------------------------------------


def _strategy(
    con: duckdb.DuckDBPyConnection, race_id: int, entries: pd.DataFrame
) -> StrategySection:
    rs = race_strategy(con, race_id)
    return strategy_section(rs.stints, rs.pit_battles, rs.safety_car_stops, rs.pit_windows, entries)


def strategy_section(
    stints: pd.DataFrame,
    pit_battles: pd.DataFrame,
    safety_car_stops: pd.DataFrame,
    pit_windows: pd.DataFrame | None,
    entries: pd.DataFrame,
) -> StrategySection:
    """Turn the strategy module's per-race frames into the debrief's strategy section."""
    if stints.empty:
        return StrategySection(
            available=False,
            note="No tyre data for this race (FastF1 covers 2018 onward), so no strategy section.",
        )
    code = dict(zip(entries["driver_id"], entries["code"], strict=True))
    finish = {d: _int(f) for d, f in zip(entries["driver_id"], entries["finish"], strict=True)}

    verdicts: dict[str, list[str]] = {}
    if pit_windows is not None and not pit_windows.empty:
        for w in pit_windows.sort_values(["driver_id", "stint"]).to_dict(orient="records"):
            if _int(w["actual_pit_lap"]) is None:
                continue
            verdicts.setdefault(w["driver_id"], []).append(
                f"lap {int(w['actual_pit_lap'])}: {w['verdict']} "
                f"(window {int(w['window_open'])}-{int(w['window_close'])})"
            )

    drivers = []
    for driver_id, group in stints.sort_values("stint").groupby("driver_id", sort=False):
        drivers.append(
            DriverStrategy(
                code=code.get(driver_id, str(driver_id)[:3].upper()),
                finish=finish.get(driver_id),
                stops=len(group) - 1,
                stints=[
                    Stint(
                        compound=_text(s["compound"]),
                        start_lap=int(s["start_lap"]),
                        end_lap=int(s["end_lap"]),
                    )
                    for s in group.to_dict(orient="records")
                ],
                pit_verdicts=verdicts.get(str(driver_id), []),
            )
        )
    drivers.sort(key=lambda d: (d.finish is None, d.finish or 0, d.code))

    usable = stints[stints["is_usable"].astype(bool)] if "is_usable" in stints else stints
    usable = usable.dropna(subset=["degradation_s_per_lap", "compound"])
    degradation = [
        CompoundDegradation(
            compound=str(compound),
            stints=len(group),
            median_degradation_s_per_lap=round(float(group["degradation_s_per_lap"].median()), 3),
            median_pace_s=_round(group["fuel_corrected_pace_s"].median(), 3),
        )
        for compound, group in usable.groupby("compound")
    ]
    degradation.sort(key=lambda c: -c.median_degradation_s_per_lap)

    battles = [
        PitBattle(
            attacker=code.get(b["attacker_id"], b["attacker_id"]),
            defender=code.get(b["defender_id"], b["defender_id"]),
            kind=str(b["kind"]),
            attacker_pit_lap=int(b["attacker_pit_lap"]),
            defender_pit_lap=int(b["defender_pit_lap"]),
            gap_before_s=round(float(b["gap_before_s"]), 2),
            gap_after_s=round(float(b["gap_after_s"]), 2),
            succeeded=bool(b["succeeded"]),
        )
        for b in pit_battles.sort_values("attacker_pit_lap").to_dict(orient="records")
    ]
    sc_stops = [
        SafetyCarStop(
            code=code.get(s["driver_id"], s["driver_id"]),
            lap=int(s["lap"]),
            track_status=str(s["track_status"]),
            time_saved_s=round(float(s["time_saved_s"]), 1),
            positions_gained=_int(s.get("positions_gained")),
        )
        for s in safety_car_stops.sort_values("time_saved_s", ascending=False).to_dict(
            orient="records"
        )
    ]
    return StrategySection(
        available=True,
        drivers=drivers,
        degradation=degradation,
        pit_battles=battles,
        safety_car_stops=sc_stops,
    )


# --- headlines --------------------------------------------------------------------------


def headlines(d: Debrief) -> list[str]:
    """The five or so sentences an engineer reads before deciding to scroll."""
    out: list[str] = []
    s = d.score
    winner = next((r for r in d.results if r.finish == 1), None)
    favourite = min(d.results, key=lambda r: r.predicted_rank)
    if winner is not None:
        odds = f" (pre-race win probability {winner.p_win:.0%})" if winner.p_win is not None else ""
        if s.winner_predicted:
            out.append(f"{winner.name} won from P{winner.grid}, as predicted{odds}.")
        else:
            out.append(
                f"{winner.name} won from P{winner.grid}{odds}; the pre-race favourite was "
                f"{favourite.name}, who {_outcome(favourite)}."
            )
    better = s.position_mae < s.grid_position_mae
    out.append(
        f"Prediction: {s.podium_hits} of 3 podium places right; finishing order off by "
        f"{s.position_mae:.1f} places per driver, "
        f"{'better than' if better else 'no better than'} 'finish where you started' "
        f"({s.grid_position_mae:.1f})."
    )
    if s.winner_locked_in_lap is not None and winner is not None and d.win_probability:
        when = (
            "before the start"
            if s.winner_locked_in_lap == 0
            else (f"from lap {s.winner_locked_in_lap} of {d.race.total_laps}")
        )
        out.append(f"The lap model made {winner.code} favourite {when} and never changed its mind.")

    movers = [r for r in d.results if r.beat_prediction_by is not None]
    if movers:
        best = max(movers, key=lambda r: r.beat_prediction_by or 0)
        if (best.beat_prediction_by or 0) >= 3:
            out.append(
                f"Biggest surprise: {best.name} finished P{best.finish}, "
                f"{best.beat_prediction_by} places better than predicted (P{best.predicted_rank})."
            )
    dnfs = [r for r in d.results if r.finish is None and r.predicted_rank <= 5]
    for r in dnfs[:2]:
        out.append(f"{r.name}, predicted P{r.predicted_rank}, did not finish ({r.status}).")

    st = d.strategy
    if st.available:
        if st.pit_battles:
            worked = sum(b.succeeded for b in st.pit_battles)
            kinds = sorted({b.kind for b in st.pit_battles})
            out.append(
                f"Strategy: {worked} of {len(st.pit_battles)} {' and '.join(kinds)} attempts "
                "gained the place."
            )
        if st.safety_car_stops:
            top = st.safety_car_stops[0]
            out.append(
                f"Best-timed stop: {top.code} pitted under the {top.track_status} on lap "
                f"{top.lap} and saved {top.time_saved_s:.1f} s."
            )
        if st.degradation:
            worst = st.degradation[0]
            out.append(
                f"Tyres: the {worst.compound.lower()} degraded most, "
                f"{worst.median_degradation_s_per_lap:.3f} s per lap (median of "
                f"{worst.stints} stints, fuel corrected)."
            )
    return out


def _outcome(r: DriverResult) -> str:
    return f"finished P{r.finish}" if r.finish is not None else f"retired ({r.status})"


# --- small helpers ----------------------------------------------------------------------


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or value is pd.NA


def _int(value: Any) -> int | None:
    return None if _missing(value) else int(value)


def _float(value: Any) -> float | None:
    return None if _missing(value) else float(value)


def _round(value: Any, digits: int) -> float | None:
    return None if _missing(value) else round(float(value), digits)


def _text(value: Any) -> str | None:
    return None if _missing(value) else str(value)
