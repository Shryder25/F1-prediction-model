-- How much the grid order holds at each circuit.
-- Grain: season x circuit. A grid/finish correlation near 1 means the race mostly
-- finished in grid order (hard to overtake); lower means more position changes.

with results as (
    select f.*, r.circuit_id
    from {{ ref('fct_race_results') }} as f
    inner join {{ ref('dim_race') }} as r using (race_id)
)

select
    season,
    circuit_id,
    count(distinct race_id) as races,
    count(*) filter (where is_classified) as classified_finishes,
    corr(grid_position, classified_position) filter (where is_classified)
        as grid_finish_correlation,
    avg(abs(positions_gained)) filter (where is_classified) as avg_abs_positions_changed,
    count(*) filter (where is_win and grid_position = 1) as pole_wins,
    count(*) filter (where is_win and grid_position > 3) as wins_from_outside_top_3
from results
group by season, circuit_id
