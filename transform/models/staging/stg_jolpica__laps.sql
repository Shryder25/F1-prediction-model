select
    {{ race_id('season', 'round') }} as race_id,
    season::integer as season,
    round::integer as round,
    driver_id,
    lap::integer as lap,
    position::integer as position,
    lap_time_s::double as lap_time_s
from {{ source('jolpica', 'raw_laps') }}
