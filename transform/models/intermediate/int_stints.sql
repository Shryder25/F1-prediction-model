-- One row per driver stint, with a degradation slope fitted to the clean laps:
-- seconds of fuel-corrected lap time lost per lap of tyre age.

select
    race_id,
    season,
    round,
    driver_id,
    stint,
    compound,
    count(*) as clean_laps,
    min(tyre_age_laps) as first_tyre_age,
    max(tyre_age_laps) as last_tyre_age,
    avg(fuel_corrected_lap_time_s) as avg_fuel_corrected_lap_time_s,
    round(regr_slope(fuel_corrected_lap_time_s, tyre_age_laps), 5) as degradation_s_per_lap,
    round(regr_r2(fuel_corrected_lap_time_s, tyre_age_laps), 4) as degradation_fit_r2,
    count(*) >= {{ var('min_stint_laps') }} as is_usable
from {{ ref('int_stint_laps') }}
group by race_id, season, round, driver_id, stint, compound
