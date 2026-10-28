-- One row per driver per race start: everything known before the lights go out,
-- plus the race outcome as labels.
--
-- Point in time: every rolling feature uses only races strictly before this one
-- (window frames end at "1 preceding"), and the championship table is the one
-- after the previous round. A feature that peeks at the race it predicts would
-- make the evaluation worthless, so tests/ml_no_future_form.sql checks this.

with entries as (
    select * from {{ ref('fct_race_results') }}
),

races as (
    select race_id, circuit_id from {{ ref('dim_race') }}
),

qualifying as (
    select race_id, driver_id, gap_to_pole_pct, gap_to_team_mate_pct
    from {{ ref('fct_qualifying') }}
),

-- Constructor form is measured per race, then rolled, so a two-car team counts once.
constructor_races as (
    select
        constructor_id,
        race_id,
        avg(points) as points_per_car,
        avg(finishing_order) as avg_finishing_order
    from entries
    group by constructor_id, race_id
),

constructor_form as (
    select
        constructor_id,
        race_id,
        avg(points_per_car) over w as constructor_points_last5,
        avg(avg_finishing_order) over w as constructor_finish_last5
    from constructor_races
    window w as (
        partition by constructor_id order by race_id
        rows between {{ var('form_window_races') }} preceding and 1 preceding
    )
),

-- Standings after the previous round of the same season. Round 1 has none.
standings_before as (
    select
        season,
        round + 1 as round,
        driver_id,
        championship_position,
        championship_points
    from {{ ref('stg_jolpica__driver_standings') }}
),

driver_history as (
    select
        e.race_id,
        e.driver_id,
        avg(e.finishing_order) over form as driver_finish_last5,
        avg(e.points) over form as driver_points_last5,
        avg(e.positions_gained) over form as driver_gained_last5,
        avg(case when e.is_classified then 0 else 1 end) over dnf as driver_dnf_rate_last10,
        count(*) over career as driver_career_starts,
        -- Track affinity: how this driver has done here before, relative to grid.
        avg(e.positions_gained) over track as driver_track_gained,
        count(*) over track as driver_track_starts
    from entries as e
    join races as r using (race_id)
    window
        form as (
            partition by e.driver_id order by e.race_id
            rows between {{ var('form_window_races') }} preceding and 1 preceding
        ),
        dnf as (
            partition by e.driver_id order by e.race_id
            rows between 10 preceding and 1 preceding
        ),
        career as (
            partition by e.driver_id order by e.race_id
            rows between unbounded preceding and 1 preceding
        ),
        track as (
            partition by e.driver_id, r.circuit_id order by e.race_id
            rows between unbounded preceding and 1 preceding
        )
)

select
    e.race_id,
    e.season,
    e.round,
    r.circuit_id,
    e.driver_id,
    e.constructor_id,
    count(*) over (partition by e.race_id) as starters,

    -- Features: the grid and qualifying
    e.grid_position,
    e.started_from_pit_lane,
    e.qualifying_position,
    q.gap_to_pole_pct,
    q.gap_to_team_mate_pct,

    -- Features: form before this race
    h.driver_finish_last5,
    h.driver_points_last5,
    h.driver_gained_last5,
    h.driver_dnf_rate_last10,
    h.driver_career_starts,
    h.driver_track_gained,
    h.driver_track_starts,
    c.constructor_points_last5,
    c.constructor_finish_last5,
    s.championship_position as championship_position_before,
    coalesce(s.championship_points, 0) as championship_points_before,

    -- Labels: the outcome. Never use these as features.
    e.finishing_order as label_finishing_order,
    e.is_win as label_win,
    e.is_podium as label_podium,
    e.is_classified as label_classified
from entries as e
join races as r using (race_id)
left join qualifying as q using (race_id, driver_id)
left join driver_history as h using (race_id, driver_id)
left join constructor_form as c using (constructor_id, race_id)
left join standings_before as s
    on s.season = e.season and s.round = e.round and s.driver_id = e.driver_id
