"""Render a `Debrief` as Markdown (for the repo and GitHub) and as an HTML page (for Pages)."""

from __future__ import annotations

from collections.abc import Sequence
from html import escape

from f1pred.debrief.model import Debrief, DriverResult, Stint

COMPOUND_COLOUR = {
    "SOFT": "#da291c",
    "MEDIUM": "#ffd12e",
    "HARD": "#d9d9d6",
    "INTERMEDIATE": "#43b02a",
    "WET": "#0067ad",
}
DASH = ' stroke-dasharray="5 3"'
DEMO_NOTE = "Synthetic demo data: these are not real races or real results."


def title(d: Debrief) -> str:
    return f"{d.race.season} {d.race.race_name}: post-race debrief"


def _pct(p: float | None) -> str:
    return "n/a" if p is None else f"{p:.0%}" if p >= 0.01 else "<1%"


def _finish(r: DriverResult) -> str:
    return f"P{r.finish}" if r.finish is not None else "DNF"


def _delta(r: DriverResult) -> str:
    if r.beat_prediction_by is None:
        return ""
    return f"{r.beat_prediction_by:+d}" if r.beat_prediction_by else "0"


def _source_note(d: Debrief) -> str:
    s = d.score
    note = f"Predictions from {s.source}, made before the start using only pre-race data."
    if s.in_sample:
        note += (
            " The saved models were trained on this season, so this race is in-sample: "
            "treat the scores as optimistic."
        )
    return note


def _results_rows(d: Debrief) -> list[list[str]]:
    return [
        [
            str(r.predicted_rank),
            f"{r.code} ({r.name})",
            r.team or "",
            f"P{r.grid}",
            _pct(r.p_win),
            _pct(r.p_podium),
            _finish(r),
            _delta(r),
        ]
        for r in d.results
    ]


RESULT_HEADERS = ["Pred.", "Driver", "Team", "Grid", "P(win)", "P(podium)", "Result", "+/-"]


def _score_rows(d: Debrief) -> list[list[str]]:
    s = d.score
    rows = [
        ["Winner predicted", "yes" if s.winner_predicted else "no"],
        ["Winner's pre-race win probability", _pct(s.winner_pre_race_p_win)],
        ["Podium places right", f"{s.podium_hits} of 3"],
        ["Position error (predicted)", f"{s.position_mae:.2f} places per classified driver"],
        ["Position error (grid order)", f"{s.grid_position_mae:.2f}"],
    ]
    if s.brier_win is not None:
        rows.append(["Brier score, win", f"{s.brier_win:.4f} (lower is better)"])
    if s.winner_locked_in_lap is not None and d.win_probability:
        rows.append(["Winner became and stayed favourite", f"lap {s.winner_locked_in_lap}"])
    return rows


def _stints_text(stints: Sequence[Stint]) -> str:
    parts = []
    for s in stints:
        compound = s.compound or "?"
        parts.append(f"{compound[0]} {s.start_lap}-{s.end_lap}")
    return " > ".join(parts)


# --- Markdown ---------------------------------------------------------------------------


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"

    return [line(headers), line(["---"] * len(headers)), *(line(r) for r in rows), ""]


def to_markdown(d: Debrief) -> str:
    r = d.race
    lines = [f"# {title(d)}", ""]
    if r.is_demo:
        lines += [f"> **{DEMO_NOTE}**", ""]
    where = ", ".join(x for x in (r.circuit_name, r.race_date) if x)
    lines += [f"Round {r.round}{f' · {where}' if where else ''} · {r.total_laps} laps", ""]
    lines += ["## Summary", "", *(f"- {h}" for h in d.headlines), ""]
    lines += ["## Predicted vs actual", "", _source_note(d), ""]
    lines += _md_table(["Measure", "Value"], _score_rows(d))
    lines += _md_table(RESULT_HEADERS, _results_rows(d))
    lines += ["`+/-` is predicted position minus result: positive beat the prediction.", ""]
    if d.win_probability:
        lines += ["### Win probability by lap", ""]
        marks = [0, *range(10, r.total_laps, 10), r.total_laps]
        headers = ["Driver", *(f"L{m}" for m in marks if m < len(d.win_probability[0].p_win))]
        rows = [
            [s.code, *(_pct(s.p_win[m]) for m in marks if m < len(s.p_win))]
            for s in d.win_probability
        ]
        lines += _md_table(headers, rows)

    st = d.strategy
    lines += ["## Strategy", ""]
    if not st.available:
        lines += [st.note or "No strategy data.", ""]
    else:
        lines += ["### Strategy calls", ""]
        lines += _md_table(
            ["Result", "Driver", "Stops", "Stints (compound laps)", "Pit timing"],
            [
                [
                    f"P{x.finish}" if x.finish else "DNF",
                    x.code,
                    str(x.stops),
                    _stints_text(x.stints),
                    "; ".join(x.pit_verdicts),
                ]
                for x in st.drivers
            ],
        )
        lines += ["### Tyre degradation", ""]
        if st.degradation:
            lines += _md_table(
                ["Compound", "Stints", "Degradation (s/lap)", "Fuel-corrected pace (s)"],
                [
                    [
                        c.compound,
                        str(c.stints),
                        f"{c.median_degradation_s_per_lap:.3f}",
                        "" if c.median_pace_s is None else f"{c.median_pace_s:.3f}",
                    ]
                    for c in st.degradation
                ],
            )
        else:
            lines += ["Too few clean laps to fit degradation.", ""]
        lines += ["### Undercuts and overcuts", ""]
        if st.pit_battles:
            lines += _md_table(
                [
                    "Attacker",
                    "Defender",
                    "Move",
                    "Stops (laps)",
                    "Gap before",
                    "Gap after",
                    "Worked",
                ],
                [
                    [
                        b.attacker,
                        b.defender,
                        b.kind,
                        f"{b.attacker_pit_lap} vs {b.defender_pit_lap}",
                        f"{b.gap_before_s:.1f} s behind",
                        f"{abs(b.gap_after_s):.1f} s {'ahead' if b.gap_after_s > 0 else 'behind'}",
                        "yes" if b.succeeded else "no",
                    ]
                    for b in st.pit_battles
                ],
            )
        else:
            lines += ["No undercut or overcut attempts between cars running close together.", ""]
        if st.safety_car_stops:
            lines += ["### Stops under safety car", ""]
            lines += _md_table(
                ["Driver", "Lap", "Under", "Time saved", "Places gained"],
                [
                    [
                        s.code,
                        str(s.lap),
                        s.track_status,
                        f"{s.time_saved_s:.1f} s",
                        "" if s.positions_gained is None else f"{s.positions_gained:+d}",
                    ]
                    for s in st.safety_car_stops
                ],
            )
    lines += [
        "---",
        "",
        f"Generated {d.generated_at} by `f1pred debrief {r.race_id}`. Method: "
        "[docs/debrief](https://github.com/Shryder25/F1-prediction-model/tree/main/docs/debrief).",
        "",
    ]
    return "\n".join(lines)


# --- HTML -------------------------------------------------------------------------------

CSS = """
:root{--bg:#fbfbfa;--fg:#1b1b1f;--muted:#5f6068;--line:#e3e3e0;--card:#fff;--accent:#e10600;
--good:#1a7f37;--bad:#b42318}
@media (prefers-color-scheme:dark){:root{--bg:#121216;--fg:#ececf1;--muted:#a0a1ab;
--line:#2c2c34;--card:#1b1b21;--good:#4ac26b;--bad:#ff7b72}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
a{color:inherit}
header .kicker{color:var(--accent);font-weight:700;letter-spacing:.08em;text-transform:uppercase;
font-size:12px}
h1{font-size:28px;line-height:1.2;margin:6px 0 4px}
h2{font-size:20px;margin:36px 0 12px;padding-top:12px;border-top:1px solid var(--line)}
h3{font-size:16px;margin:24px 0 8px}
.meta,.note,footer{color:var(--muted);font-size:13px}
.demo{background:#fff4ce;color:#5c4400;border-radius:8px;padding:8px 12px;font-weight:600}
@media (prefers-color-scheme:dark){.demo{background:#3d3214;color:#ffe08a}}
ul.headlines{padding-left:18px}ul.headlines li{margin:4px 0}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px;background:var(--card)}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:12px}
table.kv{width:auto;min-width:min(100%,520px)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.swatch{display:inline-block;width:4px;height:14px;border-radius:2px;margin-right:6px;
vertical-align:-2px}
.pos{color:var(--good)}.neg{color:var(--bad)}
.tyres{display:flex;height:14px;min-width:280px;border-radius:3px;overflow:hidden;
background:var(--line)}
.tyres span{height:100%;border-right:2px solid var(--card)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:6px 0}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}
svg text{fill:var(--muted);font-size:11px}
svg .grid{stroke:var(--line)}
"""


def _table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    numeric: Sequence[int] = (),
    css: str = "",
) -> str:
    head = "".join(
        f'<th class="num">{escape(h)}</th>' if i in numeric else f"<th>{escape(h)}</th>"
        for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="num">{c}</td>' if i in numeric else f"<td>{c}</td>"
            for i, c in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    cls = f' class="{css}"' if css else ""
    return (
        f'<div class="scroll"><table{cls}><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _win_chart(d: Debrief) -> str:
    series = d.win_probability
    if not series:
        return ""
    width, height, left, bottom, top = 920, 260, 36, 24, 8
    laps = len(series[0].p_win) - 1 or 1
    plot_w, plot_h = width - left - 10, height - bottom - top

    def x(i: int) -> float:
        return left + plot_w * i / laps

    def y(p: float) -> float:
        return top + plot_h * (1 - p)

    grid = "".join(
        f'<line class="grid" x1="{left}" x2="{width - 10}" y1="{y(p):.1f}" y2="{y(p):.1f}"/>'
        f'<text x="{left - 6}" y="{y(p) + 4:.1f}" text-anchor="end">{p:.0%}</text>'
        for p in (0, 0.25, 0.5, 0.75, 1)
    )
    ticks = "".join(
        f'<text x="{x(m):.1f}" y="{height - 6}" text-anchor="middle">'
        f"{'start' if m == 0 else m}</text>"
        for m in sorted({0, *range(10, laps, 10), laps})
    )
    # Team-mates share a colour: draw the second one dashed.
    seen: set[str] = set()
    dashed = []
    for s in series:
        dashed.append(s.team_colour in seen)
        seen.add(s.team_colour)
    lines = "".join(
        f'<polyline fill="none" stroke="{escape(s.team_colour)}" stroke-width="2" '
        f'stroke-linejoin="round"{DASH if dash else ""} points="'
        + " ".join(f"{x(i):.1f},{y(p):.1f}" for i, p in enumerate(s.p_win))
        + f'"><title>{escape(s.code)}</title></polyline>'
        for s, dash in zip(series, dashed, strict=True)
    )
    legend = "".join(
        f'<span><svg width="18" height="8" style="margin-right:4px"><line x1="0" x2="18" '
        f'y1="4" y2="4" stroke="{escape(s.team_colour)}" stroke-width="3"'
        f"{DASH if dash else ''}/></svg>{escape(s.code)}</span>"
        for s, dash in zip(series, dashed, strict=True)
    )
    return (
        "<h3>Win probability by lap</h3>"
        f'<div class="legend">{legend}</div>'
        f'<div class="scroll"><svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="Win probability by lap">{grid}{ticks}{lines}</svg></div>'
        '<p class="note">From a lap-by-lap replay through the live model: what the HUD would '
        "have shown during the race.</p>"
    )


def _tyre_bar(stints: Sequence[Stint], total_laps: int) -> str:
    spans = []
    for s in stints:
        start, end = s.start_lap, s.end_lap
        compound = s.compound or ""
        width = 100 * (end - start + 1) / max(total_laps, 1)
        colour = COMPOUND_COLOUR.get(compound, "#888888")
        spans.append(
            f'<span style="width:{width:.2f}%;background:{colour}" '
            f'title="{escape(compound.title())} laps {start}-{end}"></span>'
        )
    return f'<div class="tyres">{"".join(spans)}</div>'


def to_html(d: Debrief, home: str = "index.html") -> str:
    r = d.race
    where = " · ".join(x for x in (f"Round {r.round}", r.circuit_name, r.race_date) if x)
    parts = [
        "<header>",
        f'<div class="kicker"><a href="{escape(home)}">Race debriefs</a></div>',
        f"<h1>{escape(title(d))}</h1>",
        f'<div class="meta">{escape(where)} · {r.total_laps} laps</div>',
        "</header>",
    ]
    if r.is_demo:
        parts.append(f'<p class="demo">{escape(DEMO_NOTE)}</p>')
    parts += [
        "<h2>Summary</h2>",
        '<ul class="headlines">' + "".join(f"<li>{escape(h)}</li>" for h in d.headlines) + "</ul>",
        "<h2>Predicted vs actual</h2>",
        f'<p class="note">{escape(_source_note(d))}</p>',
        _table(["Measure", "Value"], [[escape(a), escape(b)] for a, b in _score_rows(d)], css="kv"),
        "<h3>Every driver</h3>",
    ]
    rows = []
    for x in d.results:
        delta = x.beat_prediction_by
        css = "pos" if delta and delta > 0 else "neg" if delta and delta < 0 else ""
        rows.append(
            [
                str(x.predicted_rank),
                f'<span class="swatch" style="background:{escape(x.team_colour)}"></span>'
                f"<strong>{escape(x.code)}</strong> {escape(x.name)}",
                escape(x.team or ""),
                f"P{x.grid}",
                _pct(x.p_win),
                _pct(x.p_podium),
                escape(_finish(x)) if x.finish else f'<span title="{escape(x.status)}">DNF</span>',
                f'<span class="{css}">{_delta(x)}</span>',
            ]
        )
    parts.append(_table(RESULT_HEADERS, rows, numeric=(0, 3, 4, 5, 6, 7)))
    parts.append(
        '<p class="note">+/- is predicted position minus result: positive beat the prediction.</p>'
    )
    parts.append(_win_chart(d))

    st = d.strategy
    parts.append("<h2>Strategy</h2>")
    if not st.available:
        parts.append(f"<p>{escape(st.note or 'No strategy data.')}</p>")
    else:
        legend = "".join(
            f'<span><i style="background:{c}"></i>{n.title()}</span>'
            for n, c in COMPOUND_COLOUR.items()
            if any(s.compound == n for x in st.drivers for s in x.stints)
        )
        parts += ["<h3>Strategy calls</h3>", f'<div class="legend">{legend}</div>']
        parts.append(
            _table(
                ["Result", "Driver", "Stops", "Tyres", "Pit timing"],
                [
                    [
                        f"P{x.finish}" if x.finish else "DNF",
                        f"<strong>{escape(x.code)}</strong>",
                        str(x.stops),
                        _tyre_bar(x.stints, r.total_laps),
                        escape("; ".join(x.pit_verdicts)),
                    ]
                    for x in st.drivers
                ],
                numeric=(2,),
            )
        )
        parts.append("<h3>Tyre degradation</h3>")
        if st.degradation:
            parts.append(
                _table(
                    ["Compound", "Stints", "Degradation (s/lap)", "Fuel-corrected pace (s)"],
                    [
                        [
                            escape(c.compound.title()),
                            str(c.stints),
                            f"{c.median_degradation_s_per_lap:.3f}",
                            "" if c.median_pace_s is None else f"{c.median_pace_s:.3f}",
                        ]
                        for c in st.degradation
                    ],
                    numeric=(1, 2, 3),
                )
            )
            parts.append(
                '<p class="note">Median slope of fuel-corrected lap time against tyre age, over '
                "stints with enough clean green-flag laps.</p>"
            )
        else:
            parts.append("<p>Too few clean laps to fit degradation.</p>")
        parts.append("<h3>Undercuts and overcuts</h3>")
        if st.pit_battles:
            parts.append(
                _table(
                    [
                        "Attacker",
                        "Defender",
                        "Move",
                        "Stops (laps)",
                        "Gap before",
                        "Gap after",
                        "Worked",
                    ],
                    [
                        [
                            f"<strong>{escape(b.attacker)}</strong>",
                            escape(b.defender),
                            escape(b.kind),
                            f"{b.attacker_pit_lap} vs {b.defender_pit_lap}",
                            f"{b.gap_before_s:.1f} s behind",
                            f"{abs(b.gap_after_s):.1f} s "
                            f"{'ahead' if b.gap_after_s > 0 else 'behind'}",
                            '<span class="pos">yes</span>'
                            if b.succeeded
                            else '<span class="neg">no</span>',
                        ]
                        for b in st.pit_battles
                    ],
                )
            )
        else:
            parts.append("<p>No undercut or overcut attempts between cars running close.</p>")
        if st.safety_car_stops:
            parts.append("<h3>Stops under safety car</h3>")
            parts.append(
                _table(
                    ["Driver", "Lap", "Under", "Time saved", "Places gained"],
                    [
                        [
                            f"<strong>{escape(s.code)}</strong>",
                            str(s.lap),
                            escape(s.track_status),
                            f"{s.time_saved_s:.1f} s",
                            "" if s.positions_gained is None else f"{s.positions_gained:+d}",
                        ]
                        for s in st.safety_car_stops
                    ],
                    numeric=(1, 3, 4),
                )
            )
    parts.append(
        f"<footer><p>Generated {escape(d.generated_at)} by <code>f1pred debrief "
        f'{r.race_id}</code>. <a href="https://github.com/Shryder25/F1-prediction-model/tree/'
        'main/docs/debrief">How this is built</a>.</p></footer>'
    )
    return page(title(d), "".join(parts))


def page(page_title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(page_title)}</title><style>{CSS}</style></head>"
        f"<body><main>{body}</main></body></html>\n"
    )
