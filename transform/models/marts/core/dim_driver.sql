-- One row per driver. Maps Ergast driver ids to the 3-letter codes and car numbers
-- that live timing uses, taking each driver's most recent values.

select
    driver_id,
    arg_max(driver_code, race_id) as driver_code,
    arg_max(given_name, race_id) || ' ' || arg_max(family_name, race_id) as full_name,
    arg_max(car_number, race_id) as latest_car_number,
    arg_max(permanent_number, race_id) as permanent_number,
    arg_max(constructor_id, race_id) as latest_constructor_id,
    any_value(date_of_birth) as date_of_birth,
    min(season) as first_season,
    max(season) as last_season,
    count(*) as race_entries
from {{ ref('stg_jolpica__results') }}
group by driver_id
