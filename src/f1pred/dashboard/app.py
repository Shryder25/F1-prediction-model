"""F1 insights dashboard on top of the dbt marts.

Run with `f1pred dashboard` (or `f1pred dashboard --demo` for synthetic data).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from f1pred import insights
from f1pred.config import Settings


def _warehouse_path() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warehouse", type=Path, default=None)
    args, _ = parser.parse_known_args()
    if args.warehouse:
        return Path(args.warehouse)
    return Path(os.environ.get("F1PRED_WAREHOUSE", Settings().warehouse_path))


@st.cache_data(show_spinner=False)
def _query(name: str, warehouse: str, *params: object) -> pd.DataFrame:
    with insights.connect(Path(warehouse)) as con:
        result: pd.DataFrame = getattr(insights, name)(con, *params)
        return result


def _colour_map(frame: pd.DataFrame, key: str) -> dict[str, str]:
    return dict(zip(frame[key], frame["team_colour"], strict=False))


def main() -> None:
    st.set_page_config(page_title="F1 Insights", layout="wide")
    warehouse = _warehouse_path()
    if not warehouse.exists():
        st.error(f"No warehouse at `{warehouse}`. Run `f1pred load` and `f1pred transform`.")
        st.stop()

    with insights.connect(warehouse) as con:
        demo = insights.is_demo(con)
        all_seasons = insights.seasons(con)
    if not all_seasons:
        st.error("The warehouse has no races yet.")
        st.stop()

    st.title("F1 race insights")
    if demo:
        st.warning("Demo data: these numbers come from a synthetic dataset, not real races.")

    picked = st.sidebar.multiselect("Seasons (pooled analyses)", all_seasons, all_seasons)
    season = st.sidebar.selectbox("Season (per-season analyses)", all_seasons[::-1])
    seasons = picked or all_seasons
    path = str(warehouse)

    with insights.connect(warehouse) as con:
        findings = insights.headline_findings(con, seasons)
    if findings:
        st.subheader("Headline findings")
        for finding in findings:
            st.markdown(f"- {finding}")

    grid_tab, circuit_tab, driver_tab, pit_tab, tyre_tab = st.tabs(
        ["Grid to finish", "Overtaking by circuit", "Race craft", "Pit stops", "Tyres"]
    )

    with grid_tab:
        st.caption("How often each starting slot converts. Pooled over the selected seasons.")
        grid = _query("grid_conversion", path, seasons)
        long = grid.melt(
            id_vars="grid_position",
            value_vars=["win_rate", "podium_rate", "points_rate"],
            var_name="outcome",
            value_name="rate",
        )
        long["outcome"] = long["outcome"].str.removesuffix("_rate").str.title()
        fig = px.line(long, x="grid_position", y="rate", color="outcome", markers=True)
        fig.update_layout(yaxis_tickformat=".0%", xaxis_title="Grid position", yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.dataframe(grid, hide_index=True, width="stretch")

    with circuit_tab:
        st.caption(
            "Correlation between grid and finishing order among classified finishers. "
            "Near 1 means the race finished in grid order."
        )
        circuits = _query("circuit_overtaking", path, seasons)
        fig = px.bar(
            circuits.sort_values("grid_finish_correlation"),
            x="grid_finish_correlation",
            y="circuit_name",
            orientation="h",
            hover_data=["races", "avg_abs_positions_changed", "pole_win_rate"],
        )
        fig.update_layout(xaxis_title="Grid/finish correlation", yaxis_title=None)
        st.plotly_chart(fig, width="stretch")

    with driver_tab:
        st.caption(
            "Average places finished ahead of what the driver's grid slot usually gives. "
            "Positive is good race craft (or a car better on Sunday than Saturday)."
        )
        drivers = _query("driver_race_craft", path, season)
        fig = px.bar(
            drivers,
            x="avg_places_vs_expected",
            y="driver",
            color="team",
            color_discrete_map=_colour_map(drivers, "team"),
            orientation="h",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, yaxis_title=None)
        st.plotly_chart(fig, width="stretch")

    with pit_tab:
        st.caption(
            "Median pit lane time relative to the median stop in the same race. "
            "Comparing within a race removes differences in pit lane length."
        )
        pits = _query("pit_stop_performance", path, season)
        fig = px.bar(
            pits,
            x="team",
            y="median_delta_to_race_median_s",
            color="team",
            color_discrete_map=_colour_map(pits, "team"),
            hover_data=["stops", "slow_stop_rate", "outlier_stops"],
        )
        fig.update_layout(showlegend=False, yaxis_title="Seconds vs race median", xaxis_title=None)
        st.plotly_chart(fig, width="stretch")

    with tyre_tab:
        st.caption(
            "Fuel-corrected lap time lost per lap of tyre age, median over clean green-flag "
            "stints. Needs FastF1 data."
        )
        tyres = _query("tyre_degradation", path, season)
        if tyres.empty:
            st.info("No tyre data for this season. Run `f1pred ingest-fastf1` and reload.")
        else:
            fig = px.bar(
                tyres,
                x="circuit_name",
                y="median_degradation_s_per_lap",
                color="compound",
                barmode="group",
                color_discrete_map={"SOFT": "#DA291C", "MEDIUM": "#FFD12E", "HARD": "#B0B0B0"},
                hover_data=["stints", "p25_degradation_s_per_lap", "p75_degradation_s_per_lap"],
            )
            fig.update_layout(yaxis_title="Seconds per lap", xaxis_title=None)
            st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    main()
