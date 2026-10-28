-- One row per driver per completed lap: the race state at the end of that lap,
-- plus the final outcome as labels. The lap-by-lap model learns
-- "given the state now, where does this car finish?".
--
-- Point in time: every feature uses only this lap and the laps before it.
-- total_laps is the scheduled distance, which teams know before the start
-- (it is only wrong for races shortened by a red flag).

with laps as (
    select * from {{ ref('fct_laps') }}
),

entries as (
    select
        race_id,
        driver_id,
        grid_position,
        finishing_order,
        is_win,
        is_podium,
        is_classified
    from {{ ref('fct_race_results') }}
),

stops as (
    select race_id, driver_id, lap as stop_lap from {{ ref('fct_pit_stops') }}
),

lap_medians as (
    select race_id, lap, median(lap_time_s) as field_median_lap_s
    from laps
    group by race_id, lap
),

marked as (
    select
        l.*,
        m.field_median_lap_s,
        -- Clean laps show true pace: no lap 1, pit laps, neutralisations or big losses.
        l.lap > 1
            and not coalesce(l.is_pit_in_lap or l.is_pit_out_lap, false)
            and not coalesce(l.is_safety_car or l.is_virtual_safety_car or l.is_red_flag, false)
            and l.lap_time_s < 1.07 * m.field_median_lap_s
            as is_clean_lap
    from laps as l
    join lap_medians as m using (race_id, lap)
),

paced as (
    select
        *,
        avg(case when is_clean_lap then lap_time_s - field_median_lap_s end) over (
            partition by race_id, driver_id order by lap
            rows between {{ var('recent_pace_laps') - 1 }} preceding and current row
        ) as recent_pace_vs_field_s,
        lag(gap_to_leader_s) over by_position as gap_ahead_leader_s,
        lead(gap_to_leader_s) over by_position as gap_behind_leader_s,
        count(*) over (partition by race_id, lap) as cars_running
    from marked
    window by_position as (partition by race_id, lap order by position)
),

stops_so_far as (
    select p.race_id, p.driver_id, p.lap, count(s.stop_lap) as pit_stops_made
    from paced as p
    left join stops as s
        on s.race_id = p.race_id and s.driver_id = p.driver_id and s.stop_lap <= p.lap
    group by p.race_id, p.driver_id, p.lap
)

select
    p.race_id,
    p.season,
    p.round,
    p.driver_id,
    p.lap,
    p.total_laps,

    -- Features: race state at the end of this lap
    p.lap / p.total_laps as race_progress,
    p.total_laps - p.lap as laps_remaining,
    p.position,
    p.cars_running,
    e.grid_position,
    e.grid_position - p.position as positions_gained_so_far,
    p.gap_to_leader_s,
    -- Summed lap times can disagree slightly with official positions; clamp at 0.
    -- DuckDB's greatest() skips nulls, so keep the leader's and last car's nulls explicitly.
    case when p.gap_ahead_leader_s is not null
        then greatest(p.gap_to_leader_s - p.gap_ahead_leader_s, 0) end as gap_to_car_ahead_s,
    case when p.gap_behind_leader_s is not null
        then greatest(p.gap_behind_leader_s - p.gap_to_leader_s, 0) end as gap_to_car_behind_s,
    p.recent_pace_vs_field_s,
    p.compound,
    p.tyre_age_laps,
    p.stint,
    st.pit_stops_made,
    coalesce(p.is_safety_car, false) as is_safety_car,
    coalesce(p.is_virtual_safety_car, false) as is_virtual_safety_car,

    -- Labels: the outcome. Never use these as features.
    e.finishing_order as label_finishing_order,
    e.is_win as label_win,
    e.is_podium as label_podium,
    e.is_classified as label_classified
from paced as p
join entries as e using (race_id, driver_id)
join stops_so_far as st using (race_id, driver_id, lap)
