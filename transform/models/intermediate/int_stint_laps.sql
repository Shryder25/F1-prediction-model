-- Representative green-flag laps for tyre degradation, with a fuel correction.
--
-- Dropped: lap 1, in and out laps, safety car / VSC / red flag laps, laps FastF1
-- marks as inaccurate, non-slick tyres, and laps more than 7% slower than the
-- driver's median for that stint (traffic, mistakes, damage).

with laps as (
    select * from {{ ref('int_laps_with_tyres') }}
    where has_tyre_data
      and is_slick
      and lap > 1
      and not is_pit_in_lap
      and not is_pit_out_lap
      and not is_safety_car
      and not is_virtual_safety_car
      and not is_red_flag
      and is_accurate
      and lap_time_s is not null
      and tyre_age_laps is not null
),

corrected as (
    select
        *,
        -- A car gets faster as it burns fuel. Remove that effect so the remaining
        -- trend over a stint is the tyre.
        lap_time_s - {{ var('fuel_correction_s_per_lap') }} * (total_laps - lap)
            as fuel_corrected_lap_time_s,
        median(lap_time_s) over (partition by race_id, driver_id, stint) as stint_median_s
    from laps
)

select
    race_id,
    season,
    round,
    driver_id,
    car_number,
    stint,
    compound,
    lap,
    tyre_age_laps,
    lap_time_s,
    fuel_corrected_lap_time_s
from corrected
where lap_time_s <= stint_median_s * 1.07
