with races as (
    select * from {{ ref('stg_jolpica__races') }}
),

circuits as (
    select * from {{ ref('stg_jolpica__circuits') }}
),

laps as (
    select
        race_id,
        max(lap) as total_laps,
        bool_or(has_tyre_data) as has_tyre_data
    from {{ ref('int_laps_with_tyres') }}
    group by race_id
)

select
    r.race_id,
    r.season,
    r.round,
    r.race_name,
    r.race_date,
    r.circuit_id,
    c.circuit_name,
    c.country,
    l.total_laps,
    coalesce(l.has_tyre_data, false) as has_tyre_data
from races as r
left join circuits as c using (circuit_id)
left join laps as l using (race_id)
