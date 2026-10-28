-- Who gains places on Sunday, beyond what their grid slot predicts.
-- Grain: season x driver. Expected finish for a grid slot is the average classified
-- finish from that slot across every season in the warehouse.

with results as (
    select * from {{ ref('fct_race_results') }}
),

expected as (
    select
        grid_position,
        avg(classified_position) filter (where is_classified) as expected_position
    from results
    group by grid_position
)

select
    r.season,
    r.driver_id,
    arg_max(r.constructor_id, r.race_id) as constructor_id,
    count(*) as starts,
    count(*) filter (where r.is_classified) as classified_finishes,
    avg(r.grid_position) as avg_grid_position,
    avg(r.classified_position) filter (where r.is_classified) as avg_classified_position,
    avg(r.positions_gained) filter (where r.is_classified) as avg_positions_gained,
    -- Positive means the driver finished ahead of what their grid slot usually gives.
    avg(e.expected_position - r.classified_position) filter (where r.is_classified)
        as avg_places_vs_expected,
    count(*) filter (where not r.is_classified) / count(*) as non_classified_rate,
    sum(r.points) as points
from results as r
left join expected as e using (grid_position)
group by r.season, r.driver_id
