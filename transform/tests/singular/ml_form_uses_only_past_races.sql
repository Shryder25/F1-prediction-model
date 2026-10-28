-- Leakage guard: recompute driver form a second, independent way (a self-join on
-- strictly earlier races) and fail on any row where the mart disagrees.
-- If the window frame ever included the current race, this would catch it.

with features as (
    select race_id, driver_id, driver_finish_last5, driver_career_starts
    from {{ ref('ml_pre_race_features') }}
),

numbered as (
    select
        race_id,
        driver_id,
        finishing_order,
        row_number() over (partition by driver_id order by race_id) as start_number
    from {{ ref('fct_race_results') }}
),

recomputed as (
    select
        cur.race_id,
        cur.driver_id,
        avg(prev.finishing_order) as expected_finish_last5,
        count(prev.race_id) as expected_career_starts_capped
    from numbered as cur
    left join numbered as prev
        on prev.driver_id = cur.driver_id
        and prev.start_number between cur.start_number - {{ var('form_window_races') }}
            and cur.start_number - 1
    group by cur.race_id, cur.driver_id
)

select f.*, r.expected_finish_last5
from features as f
join recomputed as r using (race_id, driver_id)
where not (
    (f.driver_finish_last5 is null and r.expected_finish_last5 is null)
    or abs(f.driver_finish_last5 - r.expected_finish_last5) < 1e-9
)
or least(f.driver_career_starts, {{ var('form_window_races') }})
    != r.expected_career_starts_capped
