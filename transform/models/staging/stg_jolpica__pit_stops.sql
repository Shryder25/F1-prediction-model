select
    {{ race_id('season', 'round') }} as race_id,
    season::integer as season,
    round::integer as round,
    driver_id,
    stop::integer as stop_number,
    lap::integer as lap,
    duration_s::double as pit_lane_time_s
from {{ source('jolpica', 'raw_pit_stops') }}
