"""Async Jolpica client with caching, rate limiting, retries and pagination."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from typing import Any, Self

import httpx

from f1pred.config import Settings
from f1pred.jolpica.cache import ResponseCache
from f1pred.jolpica.models import (
    Circuit,
    ConstructorStanding,
    DriverStanding,
    Race,
)
from f1pred.jolpica.ratelimit import SlidingWindowLimiter

log = logging.getLogger(__name__)

PAGE_SIZE = 100  # Jolpica's maximum
MAX_RETRIES = 5
RETRY_STATUSES = {429, 500, 502, 503, 504}


class JolpicaError(RuntimeError):
    pass


def merge_race_pages(races: list[dict[str, Any]], inner_key: str) -> list[dict[str, Any]]:
    """Join races split across pages.

    Pagination counts inner rows (results, laps...), so one race can be cut across two
    pages. Consecutive entries for the same season and round are merged into one.
    """
    merged: list[dict[str, Any]] = []
    for race in races:
        if merged and (merged[-1]["season"], merged[-1]["round"]) == (
            race["season"],
            race["round"],
        ):
            merged[-1] = {
                **merged[-1],
                inner_key: merged[-1].get(inner_key, []) + race.get(inner_key, []),
            }
        else:
            merged.append(dict(race))
    if inner_key == "Laps":
        merged = [_merge_laps(race) for race in merged]
    return merged


def _merge_laps(race: dict[str, Any]) -> dict[str, Any]:
    """A single lap's timings can also be split across pages."""
    laps: dict[str, dict[str, Any]] = {}
    for lap in race.get("Laps", []):
        if lap["number"] in laps:
            laps[lap["number"]]["Timings"] += lap["Timings"]
        else:
            laps[lap["number"]] = {"number": lap["number"], "Timings": list(lap["Timings"])}
    return {**race, "Laps": list(laps.values())}


class JolpicaClient:
    def __init__(
        self,
        settings: Settings | None = None,
        http: httpx.AsyncClient | None = None,
        cache: ResponseCache | None = None,
        limiter: SlidingWindowLimiter | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self._http = http or httpx.AsyncClient(timeout=30.0)
        self._cache = cache or ResponseCache(self.settings.cache_path)
        self._limiter = limiter or SlidingWindowLimiter(
            [(self.settings.requests_per_second, 1.0), (self.settings.requests_per_hour, 3600.0)]
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._http.aclose()
        self._cache.close()

    def _max_age(self, season: int | None) -> float | None:
        """Past seasons never change, so they are cached forever."""
        if season is not None and season < dt.date.today().year:
            return None
        return self.settings.current_season_cache_hours * 3600

    async def get_json(self, path: str, season: int | None = None, **params: int) -> dict[str, Any]:
        url = str(httpx.URL(f"{self.settings.jolpica_base_url}/{path}.json", params=params))
        cached = self._cache.get(url, self._max_age(season))
        if cached is not None:
            return cached

        for attempt in range(MAX_RETRIES):
            await self._limiter.acquire()
            try:
                response = await self._http.get(url)
            except httpx.TransportError as exc:
                log.warning("Network error on %s (%s), retrying", url, exc)
            else:
                if response.status_code == 200:
                    body: dict[str, Any] = response.json()
                    self._cache.put(url, body)
                    return body
                if response.status_code not in RETRY_STATUSES:
                    raise JolpicaError(f"{response.status_code} from {url}")
                log.warning("HTTP %s on %s, retrying", response.status_code, url)
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    await asyncio.sleep(int(retry_after))
                    continue
            await asyncio.sleep(2**attempt)
        raise JolpicaError(f"Gave up on {url} after {MAX_RETRIES} attempts")

    async def _paginate(
        self, path: str, table: str, items: str, season: int | None = None
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            body = await self.get_json(path, season, limit=PAGE_SIZE, offset=offset)
            data = body["MRData"]
            rows.extend(data[table][items])
            offset += PAGE_SIZE
            if offset >= int(data["total"]):
                return rows

    async def _races(self, path: str, season: int, inner_key: str | None) -> list[Race]:
        races = await self._paginate(path, "RaceTable", "Races", season)
        if inner_key:
            races = merge_race_pages(races, inner_key)
        return [Race.model_validate(race) for race in races]

    # Endpoints

    async def circuits(self) -> list[Circuit]:
        rows = await self._paginate("circuits", "CircuitTable", "Circuits")
        return [Circuit.model_validate(row) for row in rows]

    async def schedule(self, season: int) -> list[Race]:
        return await self._races(f"{season}", season, None)

    async def results(self, season: int, round_: int) -> Race | None:
        return _one(await self._races(f"{season}/{round_}/results", season, "Results"))

    async def qualifying(self, season: int, round_: int) -> Race | None:
        return _one(await self._races(f"{season}/{round_}/qualifying", season, "QualifyingResults"))

    async def pit_stops(self, season: int, round_: int) -> Race | None:
        return _one(await self._races(f"{season}/{round_}/pitstops", season, "PitStops"))

    async def laps(self, season: int, round_: int) -> Race | None:
        return _one(await self._races(f"{season}/{round_}/laps", season, "Laps"))

    async def driver_standings(self, season: int, round_: int) -> list[DriverStanding]:
        lists = await self._paginate(
            f"{season}/{round_}/driverStandings", "StandingsTable", "StandingsLists", season
        )
        return [DriverStanding.model_validate(s) for lst in lists for s in lst["DriverStandings"]]

    async def constructor_standings(self, season: int, round_: int) -> list[ConstructorStanding]:
        lists = await self._paginate(
            f"{season}/{round_}/constructorStandings", "StandingsTable", "StandingsLists", season
        )
        return [
            ConstructorStanding.model_validate(s)
            for lst in lists
            for s in lst["ConstructorStandings"]
        ]


def _one(races: list[Race]) -> Race | None:
    return races[0] if races else None
