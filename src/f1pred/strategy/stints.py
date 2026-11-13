"""Fuel-corrected stint pace and tyre degradation, one row per driver stint.

A car gets faster as it burns fuel, which hides tyre wear. Each clean lap is
corrected to "as if on an empty tank" by adding back the fuel effect, then a
straight line of corrected lap time against tyre age gives the degradation rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1pred.strategy.config import DEFAULT_CONFIG, StrategyConfig

STINT_COLUMNS = [
    "race_id",
    "driver_id",
    "stint",
    "compound",
    "start_lap",
    "end_lap",
    "laps",
    "clean_laps",
    "fuel_corrected_pace_s",
    "degradation_s_per_lap",
    "degradation_r2",
    # Modelled fuel-corrected lap time on a brand new tyre (the fit's intercept at
    # tyre age 0). Used by the pit window model; not part of the debrief contract.
    "fresh_tyre_pace_s",
    "is_usable",
]

SLICKS = ("SOFT", "MEDIUM", "HARD")


def clean_laps(laps: pd.DataFrame, config: StrategyConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Representative green-flag laps on slicks, with `fuel_corrected_lap_time_s` added.

    Dropped: lap 1, in and out laps, SC / VSC / red flag laps, laps FastF1 marks as
    inaccurate, non-slick tyres, and laps more than `outlier_ratio` slower than the
    driver's median for that stint. Same rules as the dbt model int_stint_laps.
    """
    accurate = laps.get("is_accurate", True)
    keep = (
        laps["has_tyre_data"].fillna(False).astype(bool)
        & laps["compound"].isin(SLICKS)
        & (laps["lap"] > 1)
        & ~laps["is_pit_in_lap"].fillna(False).astype(bool)
        & ~laps["is_pit_out_lap"].fillna(False).astype(bool)
        & ~laps["is_safety_car"].fillna(False).astype(bool)
        & ~laps["is_virtual_safety_car"].fillna(False).astype(bool)
        & ~laps["is_red_flag"].fillna(False).astype(bool)
        & pd.Series(accurate, index=laps.index).fillna(False).astype(bool)
        & laps["lap_time_s"].notna()
        & laps["tyre_age_laps"].notna()
    )
    out = laps.loc[keep].copy()
    out["fuel_corrected_lap_time_s"] = out["lap_time_s"] - config.fuel_correction_s_per_lap * (
        out["total_laps"] - out["lap"]
    )
    median = out.groupby(["race_id", "driver_id", "stint"])["lap_time_s"].transform("median")
    return out.loc[out["lap_time_s"] <= median * config.outlier_ratio]


def _fit(group: pd.DataFrame) -> pd.Series:
    x = group["tyre_age_laps"].to_numpy(dtype=float)
    y = group["fuel_corrected_lap_time_s"].to_numpy(dtype=float)
    slope = intercept = r2 = np.nan
    if len(x) >= 2 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        residual = y - (slope * x + intercept)
        total = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - float((residual**2).sum()) / total if total > 0 else np.nan
    return pd.Series(
        {
            "clean_laps": len(x),
            "fuel_corrected_pace_s": float(np.median(y)) if len(y) else np.nan,
            "degradation_s_per_lap": slope,
            "degradation_r2": r2,
            "fresh_tyre_pace_s": intercept,
        }
    )


def stint_summary(laps: pd.DataFrame, config: StrategyConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """One row per driver stint with fuel-corrected pace and degradation (see STINT_COLUMNS)."""
    tyred = laps.loc[laps["has_tyre_data"].fillna(False).astype(bool) & laps["stint"].notna()]
    if tyred.empty:
        return pd.DataFrame(columns=STINT_COLUMNS)
    keys = ["race_id", "driver_id", "stint"]
    spans = tyred.groupby(keys, as_index=False).agg(
        compound=("compound", lambda s: s.mode().iat[0] if s.notna().any() else None),
        start_lap=("lap", "min"),
        end_lap=("lap", "max"),
        laps=("lap", "count"),
    )
    clean = clean_laps(tyred, config)
    if clean.empty:
        fits = pd.DataFrame(columns=[*keys, *STINT_COLUMNS[7:12]])
    else:
        fits = (
            clean.groupby(keys)[["tyre_age_laps", "fuel_corrected_lap_time_s"]]
            .apply(_fit)
            .reset_index()
        )
    out = spans.merge(fits, on=keys, how="left")
    out["clean_laps"] = out["clean_laps"].fillna(0).astype(int)
    out["is_usable"] = out["clean_laps"] >= config.min_stint_laps
    for column in ("degradation_s_per_lap", "degradation_r2", "fresh_tyre_pace_s"):
        out[column] = out[column].astype(float).round(5)
    out["fuel_corrected_pace_s"] = out["fuel_corrected_pace_s"].astype(float).round(3)
    out["stint"] = out["stint"].astype(int)
    return out[STINT_COLUMNS].sort_values(["race_id", "driver_id", "stint"], ignore_index=True)
