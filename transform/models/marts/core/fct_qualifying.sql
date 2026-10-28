-- One row per driver per qualifying session, with gaps that compare across eras:
-- percent off pole, and percent off the team mate.
-- Gaps use each driver's best lap across Q1 to Q3, so they are a rough measure.

with qualifying as (
    select
        *,
        min(best_lap_s) over (partition by race_id) as pole_lap_s,
        min(best_lap_s) over (partition by race_id, constructor_id) as team_fastest_s,
        max(best_lap_s) over (partition by race_id, constructor_id) as team_slowest_s,
        count(best_lap_s) over (partition by race_id, constructor_id) as team_laps_set
    from {{ ref('stg_jolpica__qualifying') }}
)

select
    race_id,
    season,
    round,
    driver_id,
    constructor_id,
    qualifying_position,
    q1_s,
    q2_s,
    q3_s,
    best_lap_s,
    100 * (best_lap_s / pole_lap_s - 1) as gap_to_pole_pct,
    -- Positive means slower than the team mate. Null when both did not set a time.
    case
        when team_laps_set = 2 then 100 * (
            best_lap_s
            / case when best_lap_s = team_fastest_s then team_slowest_s else team_fastest_s end
            - 1
        )
    end as gap_to_team_mate_pct
from qualifying
