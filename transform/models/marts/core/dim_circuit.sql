with circuits as (
    select * from {{ ref('stg_jolpica__circuits') }}
),

usage as (
    select
        circuit_id,
        count(*) as races_held,
        min(season) as first_season,
        max(season) as last_season
    from {{ ref('stg_jolpica__races') }}
    group by circuit_id
)

select
    c.circuit_id,
    c.circuit_name,
    c.locality,
    c.country,
    c.latitude,
    c.longitude,
    coalesce(u.races_held, 0) as races_held,
    u.first_season,
    u.last_season
from circuits as c
left join usage as u using (circuit_id)
