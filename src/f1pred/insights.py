"""Queries behind the insight analyses and dashboard.

Each function reads the dbt marts and returns a tidy DataFrame. Keeping the logic
here, not in the dashboard, means it is typed, tested and reusable from notebooks.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import duckdb
import pandas as pd

from f1pred.demo import DEMO_MARKER_TABLE

Connection = duckdb.DuckDBPyConnection


def connect(warehouse_path: Path) -> Connection:
    return duckdb.connect(str(warehouse_path), read_only=True)


def is_demo(con: Connection) -> bool:
    row = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
        [DEMO_MARKER_TABLE],
    ).fetchone()
    return bool(row and row[0])


def seasons(con: Connection) -> list[int]:
    rows = con.execute("SELECT DISTINCT season FROM marts.dim_race ORDER BY season").fetchall()
    return [int(r[0]) for r in rows]


def grid_conversion(con: Connection, seasons: Sequence[int]) -> pd.DataFrame:
    """Win, podium and points rates by grid slot, pooled over `seasons`."""
    return con.execute(
        """
        SELECT
            grid_position,
            sum(starts) AS starts,
            sum(wins) / sum(starts) AS win_rate,
            sum(podiums) / sum(starts) AS podium_rate,
            sum(points_finishes) / sum(starts) AS points_rate,
            sum(non_classified) / sum(starts) AS non_classified_rate,
            sum(sum_classified_position) / nullif(sum(classified_finishes), 0)
                AS avg_classified_position
        FROM marts.mart_grid_conversion
        WHERE season IN (SELECT unnest(?::INTEGER[]))
        GROUP BY grid_position
        ORDER BY grid_position
        """,
        [list(seasons)],
    ).df()


def circuit_overtaking(con: Connection, seasons: Sequence[int]) -> pd.DataFrame:
    """Grid-to-finish correlation per circuit, weighted by races held."""
    return con.execute(
        """
        SELECT
            m.circuit_id,
            c.circuit_name,
            sum(m.races) AS races,
            sum(m.grid_finish_correlation * m.races) / sum(m.races)
                AS grid_finish_correlation,
            sum(m.avg_abs_positions_changed * m.races) / sum(m.races)
                AS avg_abs_positions_changed,
            sum(m.pole_wins) / sum(m.races) AS pole_win_rate
        FROM marts.mart_circuit_overtaking AS m
        JOIN marts.dim_circuit AS c USING (circuit_id)
        WHERE m.season IN (SELECT unnest(?::INTEGER[]))
          AND m.grid_finish_correlation IS NOT NULL
        GROUP BY m.circuit_id, c.circuit_name
        ORDER BY grid_finish_correlation DESC
        """,
        [list(seasons)],
    ).df()


def driver_race_craft(con: Connection, season: int) -> pd.DataFrame:
    """Places gained versus grid expectation, per driver, for one season."""
    return con.execute(
        """
        SELECT
            d.full_name AS driver,
            d.driver_code,
            k.constructor_name AS team,
            k.team_colour,
            m.starts,
            m.avg_grid_position,
            m.avg_classified_position,
            m.avg_places_vs_expected,
            m.non_classified_rate,
            m.points
        FROM marts.mart_driver_race_craft AS m
        JOIN marts.dim_driver AS d USING (driver_id)
        JOIN marts.dim_constructor AS k USING (constructor_id)
        WHERE m.season = ?
        ORDER BY m.avg_places_vs_expected DESC NULLS LAST
        """,
        [season],
    ).df()


def pit_stop_performance(con: Connection, season: int) -> pd.DataFrame:
    """Team pit stop ranking for one season (delta to the race median stop)."""
    return con.execute(
        """
        SELECT
            k.constructor_name AS team,
            k.team_colour,
            m.stops,
            m.outlier_stops,
            m.median_delta_to_race_median_s,
            m.median_pit_lane_time_s,
            m.slow_stop_rate,
            m.season_rank
        FROM marts.mart_pit_stop_team_performance AS m
        JOIN marts.dim_constructor AS k USING (constructor_id)
        WHERE m.season = ?
        ORDER BY m.season_rank
        """,
        [season],
    ).df()


def tyre_degradation(con: Connection, season: int) -> pd.DataFrame:
    """Median degradation per circuit and compound for one season."""
    return con.execute(
        """
        SELECT
            c.circuit_name,
            m.circuit_id,
            m.compound,
            m.stints,
            m.median_degradation_s_per_lap,
            m.p25_degradation_s_per_lap,
            m.p75_degradation_s_per_lap,
            m.median_stint_span_laps
        FROM marts.mart_tyre_degradation AS m
        JOIN marts.dim_circuit AS c USING (circuit_id)
        WHERE m.season = ?
        ORDER BY m.circuit_id,
            CASE m.compound WHEN 'SOFT' THEN 1 WHEN 'MEDIUM' THEN 2 ELSE 3 END
        """,
        [season],
    ).df()


def headline_findings(con: Connection, seasons: Sequence[int]) -> list[str]:
    """One plain-English sentence per analysis, computed from the marts."""
    findings: list[str] = []
    if not seasons:
        return findings
    span = f"{min(seasons)}-{max(seasons)}" if len(seasons) > 1 else str(seasons[0])

    grid = grid_conversion(con, seasons)
    if not grid.empty:
        pole = grid.loc[grid["grid_position"] == 1]
        outside = grid.loc[grid["grid_position"] > 3]
        if not pole.empty:
            outside_wins = (outside["win_rate"] * outside["starts"]).sum()
            outside_starts = outside["starts"].sum()
            findings.append(
                f"Grid to finish ({span}): pole converted to a win "
                f"{pole['win_rate'].iloc[0]:.0%} of the time; cars starting outside the top 3 "
                f"won {outside_wins / max(outside_starts, 1):.1%} of their starts."
            )

    circuits = circuit_overtaking(con, seasons)
    if len(circuits) >= 2:
        most, least = circuits.iloc[0], circuits.iloc[-1]
        findings.append(
            f"Overtaking ({span}): {most['circuit_name']} held grid order most closely "
            f"(correlation {most['grid_finish_correlation']:.2f}); {least['circuit_name']} "
            f"shuffled it most ({least['grid_finish_correlation']:.2f})."
        )

    latest = max(seasons)
    pits = pit_stop_performance(con, latest)
    if len(pits) >= 2:
        best, worst = pits.iloc[0], pits.iloc[-1]
        spread = worst["median_delta_to_race_median_s"] - best["median_delta_to_race_median_s"]
        findings.append(
            f"Pit stops ({latest}): {best['team']} had the quickest stops relative to each race's "
            f"median; {worst['team']} were {spread:.2f} s slower per stop, a gap that can decide "
            "an undercut."
        )

    tyres = tyre_degradation(con, latest)
    if not tyres.empty:
        top = tyres.sort_values("median_degradation_s_per_lap", ascending=False).iloc[0]
        findings.append(
            f"Tyres ({latest}): the highest degradation was the {top['compound'].lower()} at "
            f"{top['circuit_name']}, losing {top['median_degradation_s_per_lap']:.3f} s per lap "
            "of tyre age after fuel correction."
        )
    return findings
