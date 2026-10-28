select
    {{ race_id('season', 'round') }} as race_id,
    season::integer as season,
    round::integer as round,
    driver_id,
    constructor_id,
    position::integer as qualifying_position,
    q1_s::double as q1_s,
    q2_s::double as q2_s,
    q3_s::double as q3_s,
    least(q1_s, q2_s, q3_s)::double as best_lap_s
from {{ source('jolpica', 'raw_qualifying') }}
