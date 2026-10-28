select
    {{ race_id('season', 'round') }} as race_id,
    season::integer as season,
    round::integer as round,
    driver_id,
    driver_code,
    driver_number::integer as car_number,
    permanent_number::integer as permanent_number,
    given_name,
    family_name,
    date_of_birth::date as date_of_birth,
    constructor_id,
    grid::integer as grid,
    -- Ergast orders every entrant, including retirements; only numeric
    -- position_text values are classified finishes.
    position::integer as finishing_order,
    position_text,
    regexp_full_match(position_text, '\d+') as is_classified,
    case when regexp_full_match(position_text, '\d+') then position::integer end
        as classified_position,
    points::double as points,
    laps::integer as laps_completed,
    status,
    race_time_ms::bigint as race_time_ms
from {{ source('jolpica', 'raw_results') }}
