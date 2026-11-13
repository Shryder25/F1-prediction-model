"""Season tables and summary rates built from the per-race strategy metrics."""

from __future__ import annotations

from collections.abc import Sequence

import duckdb
import pandas as pd

from f1pred.strategy.config import DEFAULT_CONFIG, StrategyConfig
from f1pred.strategy.race import RaceStrategy, analyse_race, load_laps

SCHEMA = "strategy"


def build_tables(
    con: duckdb.DuckDBPyConnection, config: StrategyConfig = DEFAULT_CONFIG
) -> dict[str, int]:
    """Compute every race and (re)write the strategy.* tables. Returns row counts.

    Needs the dbt marts (`f1pred transform`) and a writable connection.
    """
    laps, stops = load_laps(con)
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in RaceStrategy.TABLES}
    for race_id, race_laps in laps.groupby("race_id", sort=True):
        result = analyse_race(race_laps, stops.loc[stops["race_id"] == race_id], config)
        for name, frame in result.frames().items():
            if not frame.empty:
                parts[name].append(frame)
    empty = analyse_race(laps.iloc[0:0]).frames()
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    counts: dict[str, int] = {}
    for name, frames in parts.items():
        table = pd.concat(frames, ignore_index=True) if frames else empty[name]
        con.register("_strategy_frame", table)
        con.execute(f"CREATE OR REPLACE TABLE {SCHEMA}.{name} AS SELECT * FROM _strategy_frame")
        con.unregister("_strategy_frame")
        counts[name] = len(table)
    return counts


def has_tables(con: duckdb.DuckDBPyConnection) -> bool:
    row = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_schema = ?", [SCHEMA]
    ).fetchone()
    return bool(row and row[0] >= len(RaceStrategy.TABLES))


def load_tables(
    con: duckdb.DuckDBPyConnection, seasons: Sequence[int] | None = None
) -> RaceStrategy:
    """Read the strategy.* tables (optionally for some seasons) with season and circuit."""
    frames = {}
    for name in RaceStrategy.TABLES:
        frames[name] = con.execute(
            f"""
            SELECT t.*, r.season, r.circuit_id
            FROM {SCHEMA}.{name} AS t
            LEFT JOIN marts.dim_race AS r USING (race_id)
            WHERE ? OR r.season IN (SELECT unnest(?::INTEGER[]))
            """,
            [seasons is None, list(seasons or [])],
        ).df()
    return RaceStrategy(**frames)


def undercut_success_rates(battles: pd.DataFrame, by: Sequence[str] = ("kind",)) -> pd.DataFrame:
    """Attempts, success rate and median time gained, per group (default: per kind)."""
    keys = list(by)
    if battles.empty:
        return pd.DataFrame(
            columns=[*keys, "attempts", "successes", "success_rate", "median_gain_s"]
        )
    out = battles.groupby(keys, as_index=False).agg(
        attempts=("succeeded", "size"),
        successes=("succeeded", "sum"),
        median_gain_s=("gain_s", "median"),
    )
    out["successes"] = out["successes"].astype(int)
    out["success_rate"] = out["successes"] / out["attempts"]
    return out[[*keys, "attempts", "successes", "success_rate", "median_gain_s"]]


def safety_car_gains(stops: pd.DataFrame, by: Sequence[str] = ("track_status",)) -> pd.DataFrame:
    """How much cheaper a stop is under SC / VSC: median seconds and places saved."""
    keys = list(by)
    columns = [*keys, "stops", "median_pit_loss_s", "median_time_saved_s", "median_positions"]
    if stops.empty:
        return pd.DataFrame(columns=columns)
    out = stops.groupby(keys, as_index=False).agg(
        stops=("pit_loss_s", "size"),
        median_pit_loss_s=("pit_loss_s", "median"),
        median_time_saved_s=("time_saved_s", "median"),
        median_positions=("positions_gained", lambda s: s.astype(float).median()),
    )
    return out[columns]


def window_adherence(windows: pd.DataFrame, by: Sequence[str] = ("compound",)) -> pd.DataFrame:
    """Share of stops early, in the modelled window, or late, and median time lost."""
    keys = list(by)
    columns = [*keys, "stops", "early", "in_window", "late", "median_time_lost_s"]
    if windows.empty:
        return pd.DataFrame(columns=columns)
    out = windows.groupby(keys, as_index=False).agg(
        stops=("verdict", "size"),
        early=("verdict", lambda s: (s == "early").mean()),
        in_window=("verdict", lambda s: (s == "in window").mean()),
        late=("verdict", lambda s: (s == "late").mean()),
        median_time_lost_s=("time_lost_s", "median"),
    )
    return out[columns]


def headline_findings(rs: RaceStrategy) -> list[str]:
    """One sentence per metric, for `f1pred strategy` and results.md."""
    lines: list[str] = []
    rates = undercut_success_rates(rs.pit_battles)
    for row in rates.itertuples():
        lines.append(
            f"{str(row.kind).capitalize()}s worked {row.success_rate:.0%} of the time "
            f"({row.successes}/{row.attempts} attempts), median gain {row.median_gain_s:+.1f} s."
        )
    sc = safety_car_gains(rs.safety_car_stops)
    for row in sc.itertuples():
        lines.append(
            f"A stop under {row.track_status} saved a median {row.median_time_saved_s:.1f} s "
            f"against a green-flag stop ({row.stops} stops)."
        )
    windows = rs.pit_windows
    if not windows.empty:
        in_window = (windows["verdict"] == "in window").mean()
        early = (windows["verdict"] == "early").mean()
        lines.append(
            f"{in_window:.0%} of stops fell inside the modelled pit window and {early:.0%} "
            f"were early (median {windows['time_lost_s'].median():.1f} s lost to the model's "
            "best lap)."
        )
    usable = rs.stints.loc[rs.stints["is_usable"].astype(bool)] if not rs.stints.empty else None
    if usable is not None and not usable.empty:
        deg = usable.groupby("compound")["degradation_s_per_lap"].median()
        parts = [f"{c} {deg[c]:.3f}" for c in ("SOFT", "MEDIUM", "HARD") if c in deg.index]
        lines.append(f"Median fuel-corrected degradation, s per lap: {', '.join(parts)}.")
    return lines
