-- One row per driver per race start: grid slot, qualifying and result side by side.
-- This is the base for every grid-to-finish analysis.

with results as (
    select * from {{ ref('stg_jolpica__results') }}
),

qualifying as (
    select race_id, driver_id, qualifying_position, best_lap_s
    from {{ ref('stg_jolpica__qualifying') }}
),

starters as (
    select *
    from results
    -- Withdrawn, did-not-start and failed-to-qualify entries never took the start.
    where position_text not in ('W', 'F')
      and status not in ('Did not start', 'Did not qualify', 'Did not prequalify', 'Withdrew')
),

slotted as (
    select
        *,
        -- Pit lane starters (grid 0) go behind the last car on the grid,
        -- in the order they finished the race.
        case
            when grid > 0 then grid
            else
                coalesce(max(case when grid > 0 then grid end) over (partition by race_id), 0)
                + row_number() over (partition by race_id, grid = 0 order by finishing_order)
        end as grid_position
    from starters
)

select
    s.race_id,
    s.season,
    s.round,
    s.driver_id,
    s.constructor_id,
    s.car_number,
    q.qualifying_position,
    q.best_lap_s as qualifying_best_lap_s,
    s.grid = 0 as started_from_pit_lane,
    s.grid_position::integer as grid_position,
    s.finishing_order,
    s.is_classified,
    s.classified_position,
    case when s.is_classified then s.grid_position - s.classified_position end
        as positions_gained,
    coalesce(s.classified_position = 1, false) as is_win,
    coalesce(s.classified_position <= 3, false) as is_podium,
    s.points > 0 as is_points_finish,
    s.points,
    s.laps_completed,
    s.status
from slotted as s
left join qualifying as q using (race_id, driver_id)
