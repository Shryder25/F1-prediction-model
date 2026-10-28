-- One row per pit stop. pit_lane_time_s is entry to exit, not the stationary time.
select * from {{ ref('int_pit_stops') }}
