-- How often each grid slot converts into wins, podiums and points.
-- Grain: season x grid position. Counts are additive across seasons; recompute the
-- rates from the counts when you combine seasons.

select
    season,
    grid_position,
    count(*) as starts,
    count(*) filter (where is_win) as wins,
    count(*) filter (where is_podium) as podiums,
    count(*) filter (where is_points_finish) as points_finishes,
    count(*) filter (where not is_classified) as non_classified,
    sum(classified_position) filter (where is_classified) as sum_classified_position,
    count(*) filter (where is_classified) as classified_finishes,
    count(*) filter (where is_win) / count(*) as win_rate,
    count(*) filter (where is_podium) / count(*) as podium_rate,
    count(*) filter (where is_points_finish) / count(*) as points_rate,
    count(*) filter (where not is_classified) / count(*) as non_classified_rate,
    avg(classified_position) filter (where is_classified) as avg_classified_position
from {{ ref('fct_race_results') }}
group by season, grid_position
