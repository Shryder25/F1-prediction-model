"""A `RaceSource` backed by OpenF1 (https://openf1.org): live races and past sessions.

OpenF1 serves timing data as JSON per session: laps, stints, pit stops, race
control messages and drivers. This module turns it into the same `LapSnapshot`
messages the warehouse replay produces, so the predictor cannot tell them apart.

Access: past sessions are free. Data for a session that is running now needs an
OpenF1 account; put its access token in `F1PRED_OPENF1_TOKEN`.

How a lap becomes a snapshot:

- A car's lap N ends when it crosses the line: `date_start` of lap N plus its
  `lap_duration`, or else the `date_start` of lap N+1.
- The order in which cars end lap N is the race order at lap N, and the time
  since the start is their elapsed race time, so gaps come straight from it.
- Live, lap N is published once every car that finished lap N-1 has finished
  lap N, or once the leader is two laps further on (a car missing by then has
  stopped or is far behind, and is left out of that lap).
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import duckdb
import httpx
import pandas as pd

from f1pred.jolpica.ratelimit import SlidingWindowLimiter
from f1pred.live.events import CarLap, Entry, LapSnapshot, RaceInfo
from f1pred.live.replay import DEFAULT_PIT_LOSS_S, load_race_info

log = logging.getLogger(__name__)

BASE_URL = "https://api.openf1.org/v1"
RETRY_STATUSES = {429, 500, 502, 503, 504}
MAX_RETRIES = 4
# The leader this many laps ahead means a car missing from lap N is not coming.
LAPS_TO_WAIT = 2

Rows = list[dict[str, Any]]


class OpenF1Error(RuntimeError):
    pass


class OpenF1Client:
    """Small async client: bearer token, rate limit (3/s, 30/min), retries."""

    def __init__(
        self,
        base_url: str = BASE_URL,
        token: str | None = None,
        http: httpx.AsyncClient | None = None,
        limiter: SlidingWindowLimiter | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        token = token if token is not None else os.environ.get("F1PRED_OPENF1_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._http = http or httpx.AsyncClient(timeout=30.0)
        self._headers = headers
        self._limiter = limiter or SlidingWindowLimiter([(3, 1.0), (30, 60.0)])

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._http.aclose()

    async def get(self, endpoint: str, **params: Any) -> Rows:
        url = f"{self.base_url}/{endpoint}"
        for attempt in range(MAX_RETRIES + 1):
            await self._limiter.acquire()
            response = await self._http.get(url, params=params, headers=self._headers)
            if response.status_code in RETRY_STATUSES and attempt < MAX_RETRIES:
                await asyncio.sleep(min(2**attempt, 30))
                continue
            if response.status_code in (401, 403):
                raise OpenF1Error(
                    f"OpenF1 refused {endpoint} ({response.status_code}). Data for a session "
                    "that is running now needs an OpenF1 account token in F1PRED_OPENF1_TOKEN."
                )
            if response.status_code == 404:
                return []  # OpenF1 answers 404 when a filter matches nothing
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):
                raise OpenF1Error(f"Unexpected OpenF1 response for {endpoint}: {data!r:.200}")
            return data
        raise OpenF1Error(f"OpenF1 {endpoint} kept failing after {MAX_RETRIES} retries.")


# --- Turning OpenF1 rows into snapshots (pure functions, unit tested) -------------


def _time(value: Any) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def lap_end_times(laps: Rows) -> dict[tuple[int, int], dt.datetime]:
    """(car number, lap) -> when the car crossed the line to end that lap."""
    starts = {
        (int(r["driver_number"]), int(r["lap_number"])): _time(r.get("date_start")) for r in laps
    }
    ends: dict[tuple[int, int], dt.datetime] = {}
    for r in laps:
        key = (int(r["driver_number"]), int(r["lap_number"]))
        start, duration = starts[key], r.get("lap_duration")
        if start is not None and duration is not None:
            ends[key] = start + dt.timedelta(seconds=float(duration))
        elif (nxt := starts.get((key[0], key[1] + 1))) is not None:
            ends[key] = nxt
    return ends


@dataclass(frozen=True)
class _Status:
    safety_car: frozenset[int]
    virtual_safety_car: frozenset[int]
    red_flag: frozenset[int]


def track_status_by_lap(race_control: Rows, last_lap: int) -> _Status:
    """Which laps had a safety car, a virtual safety car or a red flag.

    Read from race control messages: a neutralisation runs from the lap it is
    deployed to the lap it ends (or the last lap seen, if it has not ended).
    """
    laps: dict[str, set[int]] = {"SC": set(), "VSC": set(), "RED": set()}
    open_since: dict[str, int] = {}
    rows = sorted(race_control, key=lambda r: str(r.get("date") or ""))
    for r in rows:
        lap = r.get("lap_number")
        if lap is None:
            continue
        message = str(r.get("message") or "").upper()
        flag = str(r.get("flag") or "").upper()
        kind = "VSC" if "VIRTUAL" in message else "SC"
        if "SAFETY CAR DEPLOYED" in message:
            open_since.setdefault(kind, int(lap))
        elif ("IN THIS LAP" in message or "ENDING" in message) and kind in open_since:
            laps[kind].update(range(open_since.pop(kind), int(lap) + 1))
        elif flag == "RED":
            open_since.setdefault("RED", int(lap))
        elif flag == "GREEN" and "RED" in open_since:
            laps["RED"].update(range(open_since.pop("RED"), int(lap) + 1))
    for kind, since in open_since.items():
        laps[kind].update(range(since, last_lap + 1))
    return _Status(frozenset(laps["SC"]), frozenset(laps["VSC"]), frozenset(laps["RED"]))


def _stint_for(stints: Rows, car: int, lap: int) -> dict[str, Any] | None:
    for s in stints:
        if int(s["driver_number"]) != car or s.get("lap_start") is None:
            continue
        end = s.get("lap_end")
        if int(s["lap_start"]) <= lap and (end is None or lap <= int(end)):
            return s
    return None


def assemble_snapshots(
    laps: Rows,
    stints: Rows,
    pits: Rows,
    race_control: Rows,
    race_start: dt.datetime,
    driver_ids: dict[int, str],
    after_lap: int = 0,
    session_complete: bool = False,
) -> list[LapSnapshot]:
    """Every lap after `after_lap` that is ready to publish, in order.

    With `session_complete`, every lap is ready (a past session).
    """
    ends = lap_end_times(laps)
    if not ends:
        return []
    last_lap = max(lap for _, lap in ends)
    status = track_status_by_lap(race_control, last_lap)
    pit_out = {
        (int(r["driver_number"]), int(r["lap_number"])): bool(r.get("is_pit_out_lap")) for r in laps
    }
    pit_counts: dict[tuple[int, int], int] = {}
    for p in pits:
        if p.get("lap_number") is None:
            continue
        key = (int(p["driver_number"]), int(p["lap_number"]))
        pit_counts[key] = pit_counts.get(key, 0) + 1

    snapshots: list[LapSnapshot] = []
    previous: set[int] | None = None
    for lap in range(1, last_lap + 1):
        finished = {car: t for (car, n), t in ends.items() if n == lap and car in driver_ids}
        if not finished:
            break
        expected = previous if previous is not None else set(finished)
        leader_far_ahead = any(n >= lap + LAPS_TO_WAIT for _, n in ends)
        if not session_complete and not expected <= set(finished) and not leader_far_ahead:
            break  # lap N is still being completed
        previous = set(finished)
        if lap <= after_lap:
            continue
        cars = []
        order = sorted(finished, key=lambda car: finished[car])
        for position, car in enumerate(order, start=1):
            before = ends.get((car, lap - 1), race_start if lap == 1 else None)
            lap_time = (finished[car] - before).total_seconds() if before else None
            stint = _stint_for(stints, car, lap)
            age = None
            if stint is not None and stint.get("tyre_age_at_start") is not None:
                age = int(stint["tyre_age_at_start"]) + lap - int(stint["lap_start"]) + 1
            stops = pit_counts.get((car, lap), 0)
            cars.append(
                CarLap(
                    driver_id=driver_ids[car],
                    position=position,
                    lap_time_s=None if lap_time is None else round(lap_time, 3),
                    compound=(str(stint["compound"]).upper() if stint and stint.get("compound")
                              else None),
                    stint=int(stint["stint_number"]) if stint else None,
                    tyre_age_laps=age,
                    is_pit_in_lap=stops > 0,
                    is_pit_out_lap=pit_out.get((car, lap), False),
                    is_safety_car=lap in status.safety_car,
                    is_virtual_safety_car=lap in status.virtual_safety_car,
                    is_red_flag=lap in status.red_flag,
                    pit_stops=stops,
                )
            )  # fmt: skip
        snapshots.append(LapSnapshot(lap=lap, cars=tuple(cars)))
    return snapshots


def starting_grid(positions: Rows) -> dict[int, int]:
    """Car number -> grid slot, from the first position OpenF1 reports for each car."""
    first: dict[int, tuple[str, int]] = {}
    for r in positions:
        car, when = int(r["driver_number"]), str(r.get("date") or "")
        if car not in first or when < first[car][0]:
            first[car] = (when, int(r["position"]))
    return {car: pos for car, (_, pos) in first.items()}


# --- The source --------------------------------------------------------------------


class OpenF1Source:
    """Stream a race session from OpenF1, live or after the fact.

    `warehouse_path` and `race_id` are optional. With them, drivers are matched
    to warehouse ids by car number, so the pre-race model and the lap model's
    pre-race strength features work; the pit loss comes from circuit history.
    """

    def __init__(
        self,
        client: OpenF1Client,
        session_key: int | str = "latest",
        total_laps: int | None = None,
        warehouse_path: Path | None = None,
        race_id: int | None = None,
        poll_interval_s: float = 15.0,
        replay_interval_s: float = 1.0,
    ) -> None:
        self.client = client
        self.session_key = session_key
        self.total_laps = total_laps
        self.warehouse_path = warehouse_path
        self.race_id = race_id
        self.poll_interval_s = poll_interval_s
        self.replay_interval_s = replay_interval_s
        self._info: RaceInfo | None = None
        self._pre_race: pd.DataFrame | None = None
        self._session: dict[str, Any] = {}
        self._car_to_driver: dict[int, str] = {}

    async def race_info(self) -> RaceInfo:
        if self._info is None:
            self._info = await self._load_info()
        return self._info

    async def pre_race_features(self) -> pd.DataFrame | None:
        await self.race_info()
        return self._pre_race

    def _session_complete(self) -> bool:
        end = _time(self._session.get("date_end"))
        return end is not None and end + dt.timedelta(hours=1) < dt.datetime.now(dt.UTC)

    async def _load_info(self) -> RaceInfo:
        sessions = await self.client.get("sessions", session_key=self.session_key)
        if not sessions:
            raise OpenF1Error(f"OpenF1 has no session {self.session_key!r}.")
        self._session = sessions[0]
        key = self._session["session_key"]
        if str(self._session.get("session_type", "Race")) != "Race":
            log.warning("Session %s is a %s, not a race", key, self._session.get("session_type"))
        drivers = await self.client.get("drivers", session_key=key)
        grid = starting_grid(await self.client.get("position", session_key=key))
        meetings = await self.client.get("meetings", meeting_key=self._session.get("meeting_key"))
        name = meetings[0].get("meeting_name") if meetings else None
        year = int(self._session.get("year") or dt.date.today().year)

        warehouse_entries: dict[int, Entry] = {}
        pit_loss = DEFAULT_PIT_LOSS_S
        total_laps = self.total_laps
        if self.warehouse_path is not None and self.race_id is not None:
            with duckdb.connect(str(self.warehouse_path), read_only=True) as con:
                info, self._pre_race = load_race_info(con, self.race_id)
            warehouse_entries = {e.car_number: e for e in info.entries if e.car_number is not None}
            pit_loss = info.pit_loss_s
            total_laps = total_laps or info.total_laps
        if total_laps is None:
            if not self._session_complete():
                raise OpenF1Error(
                    "Give the scheduled race distance (--total-laps) for a live session."
                )
            laps = await self.client.get("laps", session_key=key)
            total_laps = max((int(r["lap_number"]) for r in laps), default=0)

        entries = []
        for d in drivers:
            car = int(d["driver_number"])
            known = warehouse_entries.get(car)
            colour = d.get("team_colour")
            entry = Entry(
                driver_id=known.driver_id if known else str(d.get("name_acronym", car)).lower(),
                code=str(d.get("name_acronym") or car),
                car_number=car,
                constructor_id=known.constructor_id if known else None,
                team_name=d.get("team_name"),
                team_colour=f"#{colour}" if colour else "#888888",
                grid_position=known.grid_position if known else grid.get(car, len(drivers)),
                gap_to_pole_pct=known.gap_to_pole_pct if known else None,
                driver_finish_last5=known.driver_finish_last5 if known else None,
                constructor_points_last5=known.constructor_points_last5 if known else None,
            )
            entries.append(entry)
            self._car_to_driver[car] = entry.driver_id
        return RaceInfo(
            race_id=self.race_id or int(key),
            season=year,
            round=0,
            race_name=name or f"{self._session.get('country_name', '')} {year}".strip(),
            circuit_id=self._session.get("circuit_short_name"),
            total_laps=total_laps,
            pit_loss_s=pit_loss,
            entries=tuple(sorted(entries, key=lambda e: e.grid_position)),
        )

    async def _fetch(self) -> tuple[Rows, Rows, Rows, Rows]:
        key = self._session["session_key"]
        laps = await self.client.get("laps", session_key=key)
        stints = await self.client.get("stints", session_key=key)
        pits = await self.client.get("pit", session_key=key)
        control = await self.client.get("race_control", session_key=key)
        return laps, stints, pits, control

    def _race_start(self, laps: Rows) -> dt.datetime:
        firsts = [t for r in laps if r.get("lap_number") == 1 and (t := _time(r.get("date_start")))]
        start = min(firsts) if firsts else _time(self._session.get("date_start"))
        if start is None:
            raise OpenF1Error("Cannot tell when the race started.")
        return start

    async def laps(self) -> AsyncIterator[LapSnapshot]:
        await self.race_info()
        sent = 0
        complete = self._session_complete()
        while True:
            laps, stints, pits, control = await self._fetch()
            new = (
                assemble_snapshots(
                    laps, stints, pits, control, self._race_start(laps), self._car_to_driver,
                    after_lap=sent, session_complete=complete,
                )
                if laps
                else []
            )  # fmt: skip
            for i, snapshot in enumerate(new):
                if complete and i and self.replay_interval_s > 0:
                    await asyncio.sleep(self.replay_interval_s)
                yield snapshot
                sent = snapshot.lap
            if complete or sent >= (self._info.total_laps if self._info else 0):
                return
            await asyncio.sleep(self.poll_interval_s)
            complete = self._session_complete()
