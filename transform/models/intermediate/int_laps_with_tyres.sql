-- Jolpica laps (every season) enriched with FastF1 tyre data where it exists.
-- FastF1 identifies drivers by car number, so we map driver_id -> car number per race.

with laps as (
    select * from {{ ref('stg_jolpica__laps') }}
),

car_numbers as (
    select race_id, driver_id, car_number from {{ ref('stg_jolpica__results') }}
),

tyres as (
    select * from {{ ref('stg_fastf1__laps') }}
),

race_length as (
    select race_id, max(lap) as total_laps from laps group by race_id
)

select
    l.race_id,
    l.season,
    l.round,
    l.driver_id,
    c.car_number,
    l.lap,
    rl.total_laps,
    l.position,
    l.lap_time_s,
    t.compound,
    t.is_slick,
    t.stint,
    t.tyre_age_laps,
    t.is_fresh_tyre,
    t.is_pit_in_lap,
    t.is_pit_out_lap,
    t.is_safety_car,
    t.is_virtual_safety_car,
    t.is_red_flag,
    t.is_accurate,
    t.race_id is not null as has_tyre_data
from laps as l
left join car_numbers as c using (race_id, driver_id)
left join race_length as rl using (race_id)
left join tyres as t
    on t.race_id = l.race_id
    and t.car_number = c.car_number
    and t.lap = l.lap
