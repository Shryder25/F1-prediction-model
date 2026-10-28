select
    {{ race_id('season', 'round') }} as race_id,
    season::integer as season,
    round::integer as round,
    race_name,
    circuit_id,
    race_date::date as race_date
from {{ source('jolpica', 'raw_races') }}
