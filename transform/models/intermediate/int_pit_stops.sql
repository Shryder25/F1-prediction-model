-- Pit stops with team, circuit and a delta to the race's typical stop.
-- Pit lane length differs by circuit, so raw times are only comparable within a race.

with stops as (
    select * from {{ ref('stg_jolpica__pit_stops') }}
),

entries as (
    select race_id, driver_id, constructor_id from {{ ref('stg_jolpica__results') }}
),

races as (
    select race_id, circuit_id from {{ ref('stg_jolpica__races') }}
),

flagged as (
    select
        s.*,
        e.constructor_id,
        r.circuit_id,
        s.pit_lane_time_s > {{ var('pit_stop_outlier_s') }} as is_outlier
    from stops as s
    left join entries as e using (race_id, driver_id)
    left join races as r using (race_id)
),

race_medians as (
    select race_id, median(pit_lane_time_s) as race_median_pit_lane_time_s
    from flagged
    where not is_outlier
    group by race_id
)

select
    f.*,
    m.race_median_pit_lane_time_s,
    case when not f.is_outlier then f.pit_lane_time_s - m.race_median_pit_lane_time_s end
        as delta_to_race_median_s
from flagged as f
left join race_medians as m using (race_id)
