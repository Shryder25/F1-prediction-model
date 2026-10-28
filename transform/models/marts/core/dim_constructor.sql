with constructors as (
    select
        constructor_id,
        min(season) as first_season,
        max(season) as last_season,
        count(distinct race_id) as races_entered
    from {{ ref('stg_jolpica__results') }}
    group by constructor_id
),

branding as (
    select * from {{ ref('constructor_branding') }}
)

select
    c.constructor_id,
    -- Unknown teams fall back to a readable version of their id.
    coalesce(
        b.constructor_name,
        array_to_string(
            list_transform(string_split(c.constructor_id, '_'), lambda w: upper(w[1]) || w[2:]),
            ' '
        )
    )
        as constructor_name,
    coalesce(b.team_colour, '#888888') as team_colour,
    c.first_season,
    c.last_season,
    c.races_entered
from constructors as c
left join branding as b using (constructor_id)
