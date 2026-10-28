-- Pit stop performance by team and season.
-- Pit lane length differs by circuit, so teams are compared on their delta to the
-- median stop in the same race, not on raw times. Outliers (red flags, penalties,
-- garage stops) are excluded from the medians but counted.

with stops as (
    select * from {{ ref('fct_pit_stops') }}
)

select
    season,
    constructor_id,
    count(*) as stops,
    count(*) filter (where is_outlier) as outlier_stops,
    median(pit_lane_time_s) filter (where not is_outlier) as median_pit_lane_time_s,
    median(delta_to_race_median_s) as median_delta_to_race_median_s,
    avg(delta_to_race_median_s) as avg_delta_to_race_median_s,
    -- A stop more than 2 s slower than the race median almost always means a problem.
    count(*) filter (where delta_to_race_median_s > 2) / count(*) filter (where not is_outlier)
        as slow_stop_rate,
    rank() over (
        partition by season order by median(delta_to_race_median_s)
    ) as season_rank
from stops
where constructor_id is not null
group by season, constructor_id
