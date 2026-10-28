-- One row per driver per race start.
select * from {{ ref('int_race_entries') }}
