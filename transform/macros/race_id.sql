{#- Integer race key: 2024 round 5 -> 202405. Sorts in calendar order. -#}
{% macro race_id(season, round) -%}
    ({{ season }} * 100 + {{ round }})::integer
{%- endmacro %}
