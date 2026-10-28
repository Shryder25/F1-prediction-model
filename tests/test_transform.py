"""End-to-end tests: synthetic raw data -> DuckDB -> dbt build -> marts.

The demo generator knows the true tyre degradation and pit crew speeds it used,
so these tests check that the models recover them, not just that SQL runs.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from f1pred import demo, insights, transform, warehouse


@pytest.fixture(scope="module")
def built_warehouse(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data_dir = tmp_path_factory.mktemp("demo")
    demo.build(data_dir)
    path = data_dir / "warehouse.duckdb"
    assert transform.run_dbt(path, ["build"]), "dbt build failed: see the log above"
    return path


@pytest.fixture
def con(built_warehouse: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    with duckdb.connect(str(built_warehouse), read_only=True) as connection:
        yield connection


def _scalar(con: duckdb.DuckDBPyConnection, sql: str) -> float:
    row = con.execute(sql).fetchone()
    assert row is not None
    return float(row[0])


def test_demo_data_passes_stage_one_quality_checks(built_warehouse: Path) -> None:
    results = warehouse.run_checks(built_warehouse)
    assert results
    assert all(bad_rows == 0 for _, bad_rows in results), results


def test_fct_laps_keeps_every_raw_lap(con: duckdb.DuckDBPyConnection) -> None:
    raw = _scalar(con, "SELECT count(*) FROM main.raw_laps")
    assert _scalar(con, "SELECT count(*) FROM marts.fct_laps") == raw
    # The car in P1 is, by definition, zero seconds behind the leader.
    leader_gap = _scalar(
        con, "SELECT max(abs(gap_to_leader_s)) FROM marts.fct_laps WHERE position = 1"
    )
    assert leader_gap == pytest.approx(0, abs=1e-6)
    assert _scalar(con, "SELECT min(gap_to_leader_s) FROM marts.fct_laps") >= -1e-6
    assert _scalar(con, "SELECT count(*) FROM marts.fct_laps WHERE compound IS NULL") == 0


def test_pit_lane_starters_start_from_the_back(con: duckdb.DuckDBPyConnection) -> None:
    rows = con.execute(
        """
        SELECT f.grid_position, (SELECT count(*) FROM marts.fct_race_results x
                                 WHERE x.race_id = f.race_id) AS starters
        FROM marts.fct_race_results AS f
        WHERE f.started_from_pit_lane
        """
    ).fetchall()
    assert rows, "the demo puts one car in the pit lane each season"
    assert all(grid == starters for grid, starters in rows)


@pytest.mark.parametrize("compound", ["SOFT", "MEDIUM", "HARD"])
def test_tyre_degradation_recovers_the_true_rate(
    con: duckdb.DuckDBPyConnection, compound: str
) -> None:
    for circuit in demo.CIRCUITS:
        expected = demo.COMPOUND_DEG[compound] * circuit.deg_multiplier
        measured = _scalar(
            con,
            f"""
            SELECT median(s.degradation_s_per_lap)
            FROM intermediate.int_stints AS s
            JOIN marts.dim_race AS r USING (race_id)
            WHERE s.is_usable AND s.compound = '{compound}'
              AND r.circuit_id = '{circuit.circuit_id}'
            """,
        )
        assert measured == pytest.approx(expected, rel=0.15), circuit.circuit_id


def test_pit_stop_ranking_matches_true_crew_speed(con: duckdb.DuckDBPyConnection) -> None:
    measured = [
        row[0]
        for row in con.execute(
            """
            SELECT constructor_id FROM marts.fct_pit_stops
            GROUP BY constructor_id ORDER BY median(delta_to_race_median_s)
            """
        ).fetchall()
    ]
    truth = sorted(demo.TEAMS, key=lambda team: demo.TEAMS[team].crew_delta_s)
    assert measured == truth


def test_grid_conversion_counts_add_up(con: duckdb.DuckDBPyConnection) -> None:
    starts = _scalar(con, "SELECT sum(starts) FROM marts.mart_grid_conversion")
    assert starts == _scalar(con, "SELECT count(*) FROM marts.fct_race_results")
    assert _scalar(con, "SELECT sum(wins) FROM marts.mart_grid_conversion") == _scalar(
        con, "SELECT count(*) FROM marts.dim_race"
    )


def test_insight_queries_and_findings(con: duckdb.DuckDBPyConnection) -> None:
    assert insights.is_demo(con)
    seasons = insights.seasons(con)
    assert seasons == list(demo.SEASONS)

    grid = insights.grid_conversion(con, seasons)
    assert grid["grid_position"].tolist() == list(range(1, len(demo.DRIVERS) + 1))
    assert grid["win_rate"].sum() == pytest.approx(1.0)

    assert len(insights.circuit_overtaking(con, seasons)) == len(demo.CIRCUITS)
    assert len(insights.driver_race_craft(con, seasons[-1])) == len(demo.DRIVERS)
    assert insights.pit_stop_performance(con, seasons[-1])["team"].iloc[0] == "Red Bull"
    assert set(insights.tyre_degradation(con, seasons[-1])["compound"]) == {
        "SOFT",
        "MEDIUM",
        "HARD",
    }

    findings = insights.headline_findings(con, seasons)
    assert len(findings) == 4
    assert any("Red Bull" in f for f in findings)


def test_models_build_without_fastf1_data(tmp_path: Path) -> None:
    """FastF1 is optional: without it, tyre marts are empty but everything builds."""
    demo.build(tmp_path)
    path = tmp_path / "warehouse.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute("DROP TABLE raw_fastf1_laps")
    assert transform.run_dbt(path, ["build"])
    with duckdb.connect(str(path), read_only=True) as connection:
        assert _scalar(connection, "SELECT count(*) FROM marts.mart_tyre_degradation") == 0
        assert _scalar(connection, "SELECT count(*) FROM marts.fct_laps") > 0
        assert insights.tyre_degradation(connection, 2024).empty
