-- One row per driver per lap, every season. Tyre columns are filled from 2018
-- onward (when FastF1 data is loaded). This is the base for lap-by-lap features.

with laps as (
    select * from {{ ref('int_laps_with_tyres') }}
),

elapsed as (
    select
        *,
        sum(lap_time_s) over (
            partition by race_id, driver_id order by lap
            rows between unbounded preceding and current row
        ) as elapsed_s
    from laps
),

leader as (
    select race_id, lap, min(elapsed_s) filter (where position = 1) as leader_elapsed_s
    from elapsed
    group by race_id, lap
)

select
    e.race_id,
    e.season,
    e.round,
    e.driver_id,
    e.car_number,
    e.lap,
    e.total_laps,
    e.position,
    e.lap_time_s,
    e.elapsed_s,
    e.elapsed_s - l.leader_elapsed_s as gap_to_leader_s,
    e.compound,
    e.stint,
    e.tyre_age_laps,
    e.is_fresh_tyre,
    e.is_pit_in_lap,
    e.is_pit_out_lap,
    e.is_safety_car,
    e.is_virtual_safety_car,
    e.is_red_flag,
    e.has_tyre_data
from elapsed as e
left join leader as l using (race_id, lap)
