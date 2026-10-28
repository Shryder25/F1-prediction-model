-- Tyre degradation by circuit, compound and season, from FastF1 lap data.
-- Each stint's slope is fuel-corrected lap time lost per lap of tyre age. Medians
-- are used because single stints are noisy (traffic, managing pace, DRS trains).
-- Empty until FastF1 data has been loaded.

with stints as (
    select s.*, r.circuit_id
    from {{ ref('int_stints') }} as s
    inner join {{ ref('dim_race') }} as r using (race_id)
    where s.is_usable
)

select
    season,
    circuit_id,
    compound,
    count(*) as stints,
    count(distinct race_id) as races,
    median(degradation_s_per_lap) as median_degradation_s_per_lap,
    quantile_cont(degradation_s_per_lap, 0.25) as p25_degradation_s_per_lap,
    quantile_cont(degradation_s_per_lap, 0.75) as p75_degradation_s_per_lap,
    median(last_tyre_age - first_tyre_age + 1) as median_stint_span_laps,
    median(avg_fuel_corrected_lap_time_s) as median_fuel_corrected_lap_time_s
from stints
group by season, circuit_id, compound
