select
    {{ race_id('season', 'round') }} as race_id,
    season::integer as season,
    round::integer as round,
    driver_id,
    position::integer as championship_position,
    points::double as championship_points,
    wins::integer as wins
from {{ source('jolpica', 'raw_driver_standings') }}
