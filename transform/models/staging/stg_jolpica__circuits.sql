select
    circuit_id,
    circuit_name,
    locality,
    country,
    lat::double as latitude,
    lon::double as longitude
from {{ source('jolpica', 'raw_circuits') }}
