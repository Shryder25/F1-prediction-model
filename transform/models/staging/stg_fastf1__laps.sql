{%- set columns = {
    "season": "integer",
    "round": "integer",
    "driver_number": "bigint",
    "driver_code": "varchar",
    "team": "varchar",
    "lap": "bigint",
    "lap_time_s": "double",
    "compound": "varchar",
    "tyre_life": "bigint",
    "stint": "bigint",
    "fresh_tyre": "boolean",
    "is_accurate": "boolean",
    "pit_in": "boolean",
    "pit_out": "boolean",
    "safety_car": "boolean",
    "virtual_safety_car": "boolean",
    "red_flag": "boolean",
} -%}

with source as (
    {{ optional_source('fastf1', 'raw_fastf1_laps', columns) }}
)

select
    {{ race_id('season', 'round') }} as race_id,
    season::integer as season,
    round::integer as round,
    driver_number::integer as car_number,
    driver_code,
    team,
    lap::integer as lap,
    lap_time_s,
    -- FastF1 also reports UNKNOWN / TEST_UNKNOWN; treat those as missing.
    case
        when upper(compound) in ('SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET')
            then upper(compound)
    end as compound,
    upper(compound) in ('SOFT', 'MEDIUM', 'HARD') as is_slick,
    tyre_life::integer as tyre_age_laps,
    stint::integer as stint,
    fresh_tyre as is_fresh_tyre,
    coalesce(is_accurate, false) as is_accurate,
    coalesce(pit_in, false) as is_pit_in_lap,
    coalesce(pit_out, false) as is_pit_out_lap,
    coalesce(safety_car, false) as is_safety_car,
    coalesce(virtual_safety_car, false) as is_virtual_safety_car,
    coalesce(red_flag, false) as is_red_flag
from source
