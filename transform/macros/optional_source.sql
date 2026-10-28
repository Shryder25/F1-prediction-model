{#-
    Select from a source that may not have been loaded yet.

    FastF1 data is optional: it only exists from 2018 and is loaded by a separate
    pipeline. When the table is missing this returns zero rows with the expected
    column types, so downstream models still build (empty) instead of failing.
-#}
{% macro optional_source(source_name, table_name, columns) %}
    {%- set relation = source(source_name, table_name) -%}
    {%- set exists = execute and adapter.get_relation(
        database=relation.database, schema=relation.schema, identifier=relation.identifier
    ) is not none -%}
    {%- if exists -%}
        select {% for name in columns %}{{ name }}{{ ", " if not loop.last }}{% endfor %}
        from {{ relation }}
    {%- else -%}
        select
        {%- for name, type in columns.items() %}
            cast(null as {{ type }}) as {{ name }}{{ "," if not loop.last }}
        {%- endfor %}
        where false
    {%- endif -%}
{% endmacro %}
